"""Source projection filtering helpers for Twin query responses."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .._lightrag_compat import (
    GraphAnswerEnvelopeError,
    build_sources_from_raw_data,
    collect_chunk_ids,
)
from .doc_lookup import (
    _chunk_to_meta,
    _resolve_chunk_to_doc_id,
    _resolve_doc_for_chunk,
    _resolve_doc_for_file_path,
    _safe_get_score,
)
from .source_filters import (
    UNKNOWN_SOURCE_NAME,
    _doc_filter_terms,
    _doc_tags_match_filter,
    _source_doc_candidates,
    _source_file_path_candidate,
    _source_matches_doc_filter,
    _tag_filter_terms,
)

logger = logging.getLogger(__name__)

_PUBLIC_SOURCE_KEYS = frozenset(("_lightrag_reference_name_fallback",))


def _public_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal source markers before returning public responses."""
    return [
        {key: value for key, value in source.items() if key not in _PUBLIC_SOURCE_KEYS}
        for source in sources
    ]


def _filter_sources_by_min_score(
    sources: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    if min_score <= 0:
        return sources
    return [
        source
        for source in sources
        if isinstance(source.get("score"), (int, float))
        and float(source["score"]) >= min_score
    ]


async def _enrich_sources_doc_ids_from_file_path(
    rag: Any,
    sources: list[dict[str, Any]],
) -> None:
    """Fill missing ``doc_id`` values in-place from projected source paths."""
    file_paths = [
        candidate
        for source in sources
        if not source.get("doc_id")
        for candidate in [_source_file_path_candidate(source)]
        if candidate
    ]
    if not file_paths:
        return

    unique = list(dict.fromkeys(file_paths))
    resolved = await asyncio.gather(
        *(_resolve_doc_for_file_path(rag, file_path) for file_path in unique),
        return_exceptions=False,
    )
    file_path_to_doc_id = {
        file_path: doc_id for file_path, doc_id in zip(unique, resolved) if doc_id
    }
    if not file_path_to_doc_id:
        return
    for source in sources:
        if source.get("doc_id"):
            continue
        candidate = _source_file_path_candidate(source)
        if candidate and candidate in file_path_to_doc_id:
            source["doc_id"] = file_path_to_doc_id[candidate]


async def _source_matches_tag_filter(
    source: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> bool:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    doc_id = source.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        # If we cannot read ``TAGGED_WITH`` because doc_id is unavailable, we do
        # not assert a match. We keep only explicit matches from resolvable doc
        # ids and fail the source otherwise.
        return False
    if doc_id not in tags_cache:
        tags_cache[doc_id] = await fetch_doc_tags(doc_id, folder)
    return _doc_tags_match_filter(tags_cache[doc_id], tag_filter)


async def _filter_sources_by_advanced_filters(
    sources: list[dict[str, Any]],
    *,
    tag_filter: dict[str, list[str]] | None,
    doc_filter: dict[str, list[str]] | None,
    folder: str,
    fetch_doc_tags: Any,
) -> tuple[list[dict[str, Any]], bool]:
    tag_required, tag_optional = _tag_filter_terms(tag_filter)
    doc_required, doc_optional = _doc_filter_terms(doc_filter)
    if not tag_required and not tag_optional and not doc_required and not doc_optional:
        return sources, False

    tags_cache: dict[str, set[str]] = {}
    kept: list[dict[str, Any]] = []
    has_unverified = False
    for source in sources:
        if not _source_matches_doc_filter(source, doc_filter):
            if not _source_doc_candidates(source):
                has_unverified = True
            continue
        if not await _source_matches_tag_filter(
            source, tag_filter, folder, tags_cache, fetch_doc_tags
        ):
            if not source.get("doc_id"):
                has_unverified = True
            continue
        kept.append(source)
    return kept, has_unverified


async def _build_envelope_sources(
    rag: Any,
    body: Any,
    folder: str,
    envelope: Any,
    fetch_doc_tags: Any,
) -> tuple[list, bool]:
    """Project + filter sources from an aquery_llm envelope."""
    chunk_ids = collect_chunk_ids(envelope or {})
    chunk_to_doc = await _resolve_chunk_to_doc_id(rag, chunk_ids)
    try:
        sources = build_sources_from_raw_data(envelope or {}, chunk_to_doc)
    except GraphAnswerEnvelopeError as exc:
        logger.warning(
            "twin_query: aquery_llm references unprojectable, surfacing empty "
            "sources + source_projection_failed status rather than "
            "reconstructing from a second vector pass: %s",
            exc,
        )
        return [], False
    await _enrich_sources_doc_ids_from_file_path(rag, sources)
    sources = _filter_sources_by_min_score(sources, body.min_score)
    filtered, filter_projection_incomplete = await _filter_sources_by_advanced_filters(
        sources,
        tag_filter=body.tag_filter,
        doc_filter=body.doc_filter,
        folder=folder,
        fetch_doc_tags=fetch_doc_tags,
    )
    return filtered, not filter_projection_incomplete


async def _build_sources_legacy_fallback(
    rag: Any, query: str, top_k: int
) -> list[dict[str, Any]]:
    """LEGACY: separate vector pass to assemble a sources list.

    DEPRECATED on the nominal /query and /stream paths since TR-RET-02
    step 2 / audit C3. Kept ONLY as a compat reference for tests in
    isolation; it MUST NOT be invoked from a successful aquery_llm
    response path because that reintroduces the structural lie this
    PR is closing (the displayed sources used to be the result of a
    second retrieval, not the chunks LightRAG actually grounded on).

    The nominal source-of-truth now lives in
    :func:`server._lightrag_compat.build_sources_from_raw_data` which
    maps ``data.references`` from the aquery_llm envelope.
    """
    try:
        chunks_vdb = getattr(rag, "chunks_vdb", None)
        if chunks_vdb is None:
            return []
        raw = await chunks_vdb.query(query, top_k=top_k)
    except Exception:
        logger.exception("twin_query: chunks_vdb.query failed - empty sources")
        return []

    if not isinstance(raw, list):
        raw = []

    sources: list[dict[str, Any]] = []
    total = len(raw)
    for rank, chunk in enumerate(raw[:top_k]):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("id") or chunk.get("chunk_id") or ""
        file_path = (
            chunk.get("file_path")
            or chunk.get("source")
            or chunk_id
            or UNKNOWN_SOURCE_NAME
        )
        doc_id = await _resolve_doc_for_chunk(rag, str(chunk_id))
        sources.append(
            {
                "n": rank + 1,
                "type": "file",
                "name": str(file_path),
                "meta": _chunk_to_meta(chunk),
                "score": _safe_get_score(chunk, rank, total),
                "doc_id": doc_id,
                "chunk_id": str(chunk_id) or None,
            }
        )
    return sources


__all__ = [
    "_build_envelope_sources",
    "_build_sources_legacy_fallback",
    "_enrich_sources_doc_ids_from_file_path",
    "_filter_sources_by_advanced_filters",
    "_filter_sources_by_min_score",
    "_public_sources",
    "_source_matches_tag_filter",
]
