"""Source projection filtering helpers for Twin query responses."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from .._lightrag_compat import (
    GraphAnswerEnvelopeError,
    _index_chunks_by_ref,
    _parse_envelope_references,
    build_sources_from_raw_data,
    collect_chunk_ids,
)
from .doc_lookup import (
    _chunk_to_meta,
    _resolve_chunk_to_doc_id,
    _resolve_doc_for_chunk,
    _resolve_file_paths_to_doc_ids,
    _safe_get_score,
)
from .paragraph_anchor import (
    CitationEvidence,
    compute_best_anchor,
    compute_best_structural_anchor,
)
from .source_filters import (
    UNKNOWN_SOURCE_NAME,
    TagFilter,
    _doc_filter_terms,
    _doc_tags_match_filter,
    _source_doc_candidates,
    _source_file_path_candidate,
    _source_matches_doc_filter,
    _tag_filter_active,
)

logger = logging.getLogger(__name__)

_PUBLIC_SOURCE_KEYS = frozenset(("_lightrag_reference_name_fallback",))


def _public_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal source markers before returning public responses."""
    from ..upload_paths import display_upload_file_path

    public = [
        {key: value for key, value in source.items() if key not in _PUBLIC_SOURCE_KEYS}
        for source in sources
    ]
    for source in public:
        name = source.get("name")
        if isinstance(name, str):
            source["name"] = display_upload_file_path(name)
    return public


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


def _sort_sources_by_score(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return sources by descending measured relevance, stably.

    Reference numbers are intentionally left untouched: answer citations keep
    pointing at the same ``source.n`` even when the display list is reordered.
    Sources without a real metric stay visible after scored sources, in their
    original order.
    """

    def key(source: dict[str, Any]) -> tuple[bool, float]:
        score = source.get("score")
        measured = (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
        )
        return (not measured, -float(score) if measured else 0.0)

    return sorted(sources, key=key)


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

    file_path_to_doc_id = await _resolve_file_paths_to_doc_ids(rag, file_paths)
    if not file_path_to_doc_id:
        return
    for source in sources:
        if source.get("doc_id"):
            continue
        candidate = _source_file_path_candidate(source)
        if candidate and candidate in file_path_to_doc_id:
            source["doc_id"] = file_path_to_doc_id[candidate]


async def _enrich_sources_with_source_links(
    sources: list[dict[str, Any]], folder: str
) -> None:
    """Attach the parent document's canonical provenance links in one batch."""
    doc_ids = [
        str(source["doc_id"])
        for source in sources
        if isinstance(source.get("doc_id"), str) and source.get("doc_id")
    ]
    if not doc_ids:
        return
    try:
        from ..webui.store import get_store

        links_by_doc = await get_store(folder).source_links.list_for_documents(doc_ids)
    except Exception:  # enrichment must not turn a grounded answer into a 500
        logger.exception("twin_query: source_links enrichment failed")
        return
    for source in sources:
        doc_id = source.get("doc_id")
        source["source_links"] = links_by_doc.get(str(doc_id), []) if doc_id else []


async def _source_matches_tag_filter(
    source: dict[str, Any],
    tag_filter: TagFilter | None,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> bool:
    if not _tag_filter_active(tag_filter):
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
    tag_filter: TagFilter | None,
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
    tag_filter: TagFilter | None,
    doc_filter: dict[str, list[str]] | None,
    folder: str,
    fetch_doc_tags: Any,
    fetch_doc_tags_batch: Any = None,
) -> tuple[list[dict[str, Any]], bool]:
    tag_active = _tag_filter_active(tag_filter)
    doc_required, doc_optional = _doc_filter_terms(doc_filter)
    if not tag_active and not doc_required and not doc_optional:
        return sources, False

    matches, prefetch_doc_ids, has_unverified_docs = _doc_filter_pass(
        sources, doc_filter=doc_filter, tag_active=tag_active
    )

    tags_cache: dict[str, set[str]] = {}
    if tag_active and prefetch_doc_ids:
        if callable(fetch_doc_tags_batch):
            resolved_tags = await fetch_doc_tags_batch(prefetch_doc_ids, folder)
            resolved_tags = resolved_tags if isinstance(resolved_tags, dict) else {}
            tags_cache.update(
                (doc_id, set(resolved_tags.get(doc_id) or set()))
                for doc_id in prefetch_doc_ids
            )
        else:
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


def _anchor_candidates(
    matching_chunks: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Every ``(chunk_id, content)`` pair behind one reference, in order.

    LightRAG reference ids are per *file*, so a single reference routinely
    covers several chunks — the anchor election must see all of them, not
    just the one ``_first_chunk_id`` happens to project (PR #418 review,
    finding 1).
    """
    candidates: list[tuple[str, str]] = []
    for chunk in matching_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        if (
            isinstance(chunk_id, str)
            and chunk_id
            and isinstance(content, str)
            and content
        ):
            candidates.append((chunk_id, content))
    return candidates


async def _fetch_chunk_boundaries(
    rag: Any, chunk_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Batch-read ``twin_block_boundaries`` for the envelope's chunks.

    Phase B1: ONE ``text_chunks.get_by_ids`` call over the already-elected
    chunk ids — never a second retrieval. Chunks ingested before the
    preconverted-parse seam (or on the 1.4.x line) simply have no
    boundaries; any storage hiccup degrades to an empty mapping and the
    lexical anchors of phase A still apply.
    """
    if not chunk_ids:
        return {}
    try:
        rows = await rag.text_chunks.get_by_ids(chunk_ids)
        if isinstance(rows, dict):
            rows = [rows.get(chunk_id) for chunk_id in chunk_ids]
        boundaries_by_chunk: dict[str, list[dict[str, Any]]] = {}
        for chunk_id, row in zip(chunk_ids, rows or []):
            if not isinstance(row, dict):
                continue
            boundaries = row.get("twin_block_boundaries")
            if isinstance(boundaries, list) and boundaries:
                boundaries_by_chunk[chunk_id] = boundaries
        return boundaries_by_chunk
    except Exception:  # noqa: BLE001 - enrichment-only data, fail-soft
        logger.exception(
            "twin_query: twin_block_boundaries batch read failed; structural "
            "anchors skipped, lexical anchors unaffected"
        )
        return {}


def _enrich_sources_with_anchors(
    sources: list[dict[str, Any]],
    envelope: Any,
    citation_evidence: dict[int, CitationEvidence] | None,
    boundaries_by_chunk: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Attach intra-chunk paragraph anchors, in place, fail-soft.

    docs/adr/008-paragraph-citation-anchor.md: pure enrichment AFTER the fail-closed
    validations — an exception here must leave every source intact and must
    never flip the projection verdict, so the whole pass is wrapped and any
    failure is logged and swallowed. Sources without evidence, without a
    chunk to anchor into, or scored below the confidence floor simply carry
    no ``anchor`` key (the wire model defaults it to null).

    The election runs across every chunk behind the reference and the
    winning ``chunk_id`` is published together with its ``anchor`` — the
    two fields move atomically, so a citation grounded in the second chunk
    of a file repoints the source at that chunk instead of anchoring a
    lookalike paragraph in the first.
    """
    if not citation_evidence:
        return
    try:
        parsed = _parse_envelope_references(envelope or {})
        if parsed is None:
            return
        _, chunks = parsed
        chunks_by_ref = _index_chunks_by_ref(chunks)
        for source in sources:
            n = source.get("n")
            evidence = citation_evidence.get(n) if isinstance(n, int) else None
            if evidence is None or not source.get("chunk_id"):
                continue
            candidates = _anchor_candidates(chunks_by_ref.get(str(n), []))
            if not candidates:
                continue
            # Phase B1: structural election over ingestion-persisted block
            # boundaries first; anything short of a confident structural
            # anchor falls back to the phase A lexical election unchanged.
            elected = None
            if boundaries_by_chunk:
                structural_candidates = [
                    (chunk_id, content, boundaries_by_chunk[chunk_id])
                    for chunk_id, content in candidates
                    if chunk_id in boundaries_by_chunk
                ]
                if structural_candidates:
                    elected = compute_best_structural_anchor(
                        structural_candidates, evidence
                    )
            if elected is None:
                elected = compute_best_anchor(candidates, evidence)
            if elected is not None:
                winning_chunk_id, anchor = elected
                source["chunk_id"] = winning_chunk_id
                source["anchor"] = anchor
    except Exception:
        logger.exception(
            "twin_query: paragraph-anchor enrichment failed; publishing "
            "sources without anchors (fail-soft, projection verdict untouched)"
        )


async def _build_envelope_sources(
    rag: Any,
    body: Any,
    folder: str,
    envelope: Any,
    fetch_doc_tags: Any,
    fetch_doc_tags_batch: Any = None,
    citation_evidence: dict[int, CitationEvidence] | None = None,
    retrieval_scores: dict[str, float] | None = None,
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
        sources = build_sources_from_raw_data(
            envelope or {},
            chunk_to_doc,
            retrieval_scores,
        )
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
        tag_filter=body.tag_filter_payload,
        doc_filter=body.doc_filter,
        folder=folder,
        fetch_doc_tags=fetch_doc_tags,
        fetch_doc_tags_batch=fetch_doc_tags_batch,
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
    boundaries_by_chunk = (
        await _fetch_chunk_boundaries(rag, chunk_ids) if citation_evidence else {}
    )
    _enrich_sources_with_anchors(
        sources, envelope, citation_evidence, boundaries_by_chunk
    )
    await _enrich_sources_with_source_links(sources, folder)
    return _sort_sources_by_score(sources), True


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
    return _sort_sources_by_score(sources)


__all__ = [
    "_build_envelope_sources",
    "_build_sources_legacy_fallback",
    "_enrich_sources_doc_ids_from_file_path",
    "_enrich_sources_with_source_links",
    "_enrich_sources_with_anchors",
    "_filter_sources_by_advanced_filters",
    "_filter_sources_by_min_score",
    "_public_sources",
    "_source_matches_tag_filter",
    "_sort_sources_by_score",
]
