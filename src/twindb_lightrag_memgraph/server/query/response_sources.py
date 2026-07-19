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
    """Validate explicit source metrics against ``min_score``.

    The retrieval scope applies ``min_score`` before the answer prompt is
    assembled.  This projection-time check is therefore only a consistency
    guard for metrics LightRAG actually returns.  A missing metric is retained:
    absence of a display score is not evidence that the already-scoped chunk
    failed the threshold, and manufacturing a proxy merely to re-filter it
    would be scientifically invalid.
    """
    if min_score <= 0:
        return sources
    return [
        source
        for source in sources
        if not isinstance(source.get("score"), (int, float))
        or float(source["score"]) >= min_score
    ]


def _same_reference_projection(
    original: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> bool:
    """Whether a validation pass preserved every reference, in order."""
    return [source.get("n") for source in original] == [
        source.get("n") for source in candidate
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


def _doc_filter_pass(
    sources: list[dict[str, Any]],
    *,
    doc_filter: dict[str, list[str]] | None,
    tag_active: bool,
) -> tuple[list[bool], list[str], bool]:
    """First pass: per-source doc-filter verdict + tag prefetch doc ids.

    Returns the parallel match verdicts, the deduped doc ids to prefetch tags
    for, and whether any excluded source could not be tied back to a doc
    (unverified projection).
    """
    doc_required, doc_optional = _doc_filter_terms(doc_filter)
    doc_filter_enabled = bool(doc_required or doc_optional)
    matches: list[bool] = []
    prefetch_doc_ids: list[str] = []
    prefetch_seen: set[str] = set()
    has_unverified = False

    for source in sources:
        source_match = _source_matches_doc_filter(source, doc_filter)
        matches.append(source_match)
        if not source_match:
            if doc_filter_enabled and not _source_doc_candidates(source):
                has_unverified = True
            continue
        if not tag_active:
            continue
        doc_id = source.get("doc_id")
        if isinstance(doc_id, str) and doc_id and doc_id not in prefetch_seen:
            prefetch_seen.add(doc_id)
            prefetch_doc_ids.append(doc_id)
    return matches, prefetch_doc_ids, has_unverified


def _apply_tag_filter(
    sources: list[dict[str, Any]],
    matches: list[bool],
    *,
    tag_active: bool,
    tags_cache: dict[str, set[str]],
    tag_filter: dict[str, list[str]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Second pass: keep doc-matched sources whose tags also match."""
    kept: list[dict[str, Any]] = []
    has_unverified = False
    for source, source_match in zip(sources, matches):
        if not source_match:
            continue
        if tag_active:
            doc_id = source.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                has_unverified = True
                continue
            if not _doc_tags_match_filter(tags_cache.get(doc_id, set()), tag_filter):
                continue
        kept.append(source)
    return kept, has_unverified


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

    tag_active = bool(tag_required or tag_optional)
    matches, prefetch_doc_ids, has_unverified_docs = _doc_filter_pass(
        sources, doc_filter=doc_filter, tag_active=tag_active
    )

    tags_cache: dict[str, set[str]] = {}
    if tag_active and prefetch_doc_ids:
        resolved_tags = await asyncio.gather(
            *(fetch_doc_tags(doc_id, folder) for doc_id in prefetch_doc_ids)
        )
        tags_cache.update(zip(prefetch_doc_ids, resolved_tags))

    kept, has_unverified_tags = _apply_tag_filter(
        sources,
        matches,
        tag_active=tag_active,
        tags_cache=tags_cache,
        tag_filter=tag_filter,
    )
    return kept, has_unverified_docs or has_unverified_tags


async def _build_envelope_sources(
    rag: Any,
    body: Any,
    folder: str,
    envelope: Any,
    fetch_doc_tags: Any,
) -> tuple[list, bool]:
    """Project sources from the answer envelope and validate them fail-closed.

    ``min_score`` / document / tag constraints already ran inside
    ``retrieval_scope`` before LightRAG assembled the LLM prompt.  The checks
    here are not a second retrieval filter and must never publish a subset of
    the references the model saw.  If a check would remove or cannot verify a
    reference, the caller receives ``([], False)`` and surfaces
    ``source_projection_failed``.  This preserves the one-to-one
    ``reference_id -> source.n`` contract instead of silently relabelling a
    partially filtered answer as grounded.
    """
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
    if not sources:
        logger.warning(
            "twin_query: successful nominal answer has no projectable "
            "references; surfacing source_projection_failed"
        )
        return [], False
    await _enrich_sources_doc_ids_from_file_path(rag, sources)

    score_validated = _filter_sources_by_min_score(sources, body.min_score)
    if not _same_reference_projection(sources, score_validated):
        logger.warning(
            "twin_query: min_score post-validation would remove references "
            "already used for answer synthesis; surfacing "
            "source_projection_failed"
        )
        return [], False

    validated, filter_projection_incomplete = await _filter_sources_by_advanced_filters(
        score_validated,
        tag_filter=body.tag_filter,
        doc_filter=body.doc_filter,
        folder=folder,
        fetch_doc_tags=fetch_doc_tags,
    )
    if filter_projection_incomplete or not _same_reference_projection(
        sources, validated
    ):
        logger.warning(
            "twin_query: document/tag post-validation could not preserve all "
            "references used for answer synthesis; surfacing "
            "source_projection_failed"
        )
        return [], False
    return sources, True


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

    total = len(raw)
    projected_chunks: list[tuple[int, dict[str, Any], str, str]] = []
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
        projected_chunks.append((rank, chunk, str(chunk_id), str(file_path)))

    if not projected_chunks:
        return []

    doc_ids = await asyncio.gather(
        *(
            _resolve_doc_for_chunk(rag, chunk_id)
            for _, _, chunk_id, _ in projected_chunks
        )
    )

    sources: list[dict[str, Any]] = []
    for (rank, chunk, chunk_id, file_path), doc_id in zip(projected_chunks, doc_ids):
        sources.append(
            {
                "n": rank + 1,
                "type": "file",
                "name": file_path,
                "meta": _chunk_to_meta(chunk),
                "score": _safe_get_score(chunk, rank, total),
                "doc_id": doc_id,
                "chunk_id": chunk_id or None,
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
