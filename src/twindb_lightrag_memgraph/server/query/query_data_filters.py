"""Tag filtering helpers for structured ``/query/data`` responses."""

from __future__ import annotations

from typing import Any

from .doc_lookup import _resolve_doc_for_chunk, _resolve_doc_for_file_path
from .source_filters import (
    UNKNOWN_SOURCE_NAME,
    _doc_tags_match_filter,
    _split_source_ids,
    _tag_filter_terms,
)


def _direct_doc_ids_for_query_data_row(row: dict[str, Any]) -> set[str]:
    return {
        str(row[key])
        for key in ("doc_id", "full_doc_id")
        if isinstance(row.get(key), str) and row.get(key)
    }


def _chunk_ids_for_query_data_row(row: dict[str, Any]) -> set[str]:
    chunk_ids = {
        str(row[key])
        for key in ("chunk_id", "id")
        if isinstance(row.get(key), str) and row.get(key)
    }
    chunk_ids.update(_split_source_ids(row.get("source_id")))
    return chunk_ids


def _file_path_candidates_for_query_data_row(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("file_path", "source", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value != UNKNOWN_SOURCE_NAME:
            candidates.append(value.strip())
    return candidates


async def _doc_ids_from_chunk_ids(rag: Any, chunk_ids: set[str]) -> set[str]:
    doc_ids = set()
    for chunk_id in chunk_ids:
        doc_id = await _resolve_doc_for_chunk(rag, chunk_id)
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


async def _doc_ids_from_file_paths(rag: Any, file_paths: list[str]) -> set[str]:
    doc_ids = set()
    for file_path in file_paths:
        doc_id = await _resolve_doc_for_file_path(rag, file_path)
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


async def _doc_ids_for_query_data_row(rag: Any, row: dict[str, Any]) -> set[str]:
    doc_ids = _direct_doc_ids_for_query_data_row(row)
    doc_ids.update(
        await _doc_ids_from_chunk_ids(rag, _chunk_ids_for_query_data_row(row))
    )
    doc_ids.update(
        await _doc_ids_from_file_paths(
            rag,
            _file_path_candidates_for_query_data_row(row),
        )
    )
    return doc_ids


async def _row_matches_tag_filter(
    rag: Any,
    row: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> bool:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    doc_ids = await _doc_ids_for_query_data_row(rag, row)
    if not doc_ids:
        # Active tag filters are an inclusion contract. If a row cannot be tied
        # back to a DocStatus node (direct doc id, source chunk, or file path),
        # we cannot prove it belongs to the tagged corpus, so reject it.
        return False
    for doc_id in doc_ids:
        # Per-request cache: chunks/references rows often repeat the same
        # doc_id; one Cypher round-trip per unique doc suffices.
        if doc_id not in tags_cache:
            tags_cache[doc_id] = await fetch_doc_tags(doc_id, folder)
        if _doc_tags_match_filter(tags_cache[doc_id], tag_filter):
            return True
    return False


async def _filter_rows_by_tags(
    rag: Any,
    rows: list,
    tag_filter,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> tuple[list, set[str]]:
    """Keep rows whose doc tags match the filter; collect their reference_ids."""
    kept_rows = []
    kept_reference_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if await _row_matches_tag_filter(
            rag, row, tag_filter, folder, tags_cache, fetch_doc_tags
        ):
            kept_rows.append(row)
            ref_id = row.get("reference_id")
            if isinstance(ref_id, str) and ref_id:
                kept_reference_ids.add(ref_id)
    return kept_rows, kept_reference_ids


async def _filter_query_data_by_tags(
    rag: Any,
    response: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
    fetch_doc_tags: Any,
) -> dict[str, Any]:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return response

    data = response.get("data")
    if not isinstance(data, dict):
        return response

    # Cache shared across all rows in this single request, bounded by the number
    # of unique doc_ids in the result set.
    tags_cache: dict[str, set[str]] = {}

    filtered_data = dict(data)
    kept_reference_ids: set[str] = set()
    for key in ("chunks", "entities", "relationships"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        kept_rows, ref_ids = await _filter_rows_by_tags(
            rag, rows, tag_filter, folder, tags_cache, fetch_doc_tags
        )
        filtered_data[key] = kept_rows
        kept_reference_ids |= ref_ids

    references = data.get("references")
    if isinstance(references, list):
        filtered_data["references"] = [
            ref
            for ref in references
            if not isinstance(ref, dict)
            or ref.get("reference_id") in kept_reference_ids
        ]

    filtered = dict(response)
    filtered["data"] = filtered_data
    metadata = dict(response.get("metadata") or {})
    metadata["tag_filter"] = tag_filter
    filtered["metadata"] = metadata
    return filtered


__all__ = [
    "_filter_query_data_by_tags",
]
