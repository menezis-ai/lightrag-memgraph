"""Source projection filtering helpers for Twin query responses."""

from __future__ import annotations

import asyncio
from typing import Any

from .doc_lookup import _resolve_doc_for_file_path
from .source_filters import (
    _doc_filter_terms,
    _doc_tags_match_filter,
    _source_doc_candidates,
    _source_file_path_candidate,
    _source_matches_doc_filter,
    _tag_filter_terms,
)

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


__all__ = [
    "_enrich_sources_doc_ids_from_file_path",
    "_filter_sources_by_advanced_filters",
    "_filter_sources_by_min_score",
    "_public_sources",
    "_source_matches_tag_filter",
]
