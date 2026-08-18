"""Tag filtering helpers for structured ``/query/data`` responses."""

from __future__ import annotations

import asyncio
from typing import Any

from .doc_lookup import (
    _resolve_chunk_to_doc_id,
    _resolve_doc_for_chunk,
    _resolve_doc_for_file_path,
    _resolve_file_paths_to_doc_ids,
)
from .source_filters import (
    UNKNOWN_SOURCE_NAME,
    TagFilter,
    _doc_tags_match_filter,
    _split_source_ids,
    _tag_filter_active,
)

# Budget for overlapping the two resolver groups. The per-id fan-out is inside
# each resolver; this only avoids doubling that fan-out for larger mixed rows.
_PARALLEL_RESOLVE_ID_BUDGET = 16


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


def _query_data_row_doc_candidates(
    row: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    chunk_ids = set(_chunk_ids_for_query_data_row(row))
    file_paths = set(_file_path_candidates_for_query_data_row(row))
    return (
        _direct_doc_ids_for_query_data_row(row),
        chunk_ids,
        file_paths,
    )


async def _resolve_doc_ids_for_chunk_ids(
    rag: Any, chunk_ids: set[str]
) -> dict[str, str]:
    if not chunk_ids:
        return {}
    return await _resolve_chunk_to_doc_id(rag, list(chunk_ids))


async def _resolve_doc_ids_for_file_paths(
    rag: Any, file_paths: set[str]
) -> dict[str, str]:
    if not file_paths:
        return {}
    return await _resolve_file_paths_to_doc_ids(rag, list(file_paths))


async def _doc_ids_from_chunk_ids(rag: Any, chunk_ids: set[str]) -> set[str]:
    if not chunk_ids:
        return set()
    resolved = await asyncio.gather(
        *(_resolve_doc_for_chunk(rag, chunk_id) for chunk_id in chunk_ids)
    )
    return {doc_id for doc_id in resolved if isinstance(doc_id, str) and doc_id}


async def _doc_ids_from_file_paths(rag: Any, file_paths: list[str]) -> set[str]:
    if not file_paths:
        return set()
    unique = set(file_paths)
    resolved = await asyncio.gather(
        *(_resolve_doc_for_file_path(rag, file_path) for file_path in unique)
    )
    return {doc_id for doc_id in resolved if isinstance(doc_id, str) and doc_id}


async def _doc_ids_for_query_data_row(rag: Any, row: dict[str, Any]) -> set[str]:
    doc_ids = _direct_doc_ids_for_query_data_row(row)
    chunk_doc_ids, file_doc_ids = await asyncio.gather(
        _doc_ids_from_chunk_ids(rag, _chunk_ids_for_query_data_row(row)),
        _doc_ids_from_file_paths(
            rag,
            _file_path_candidates_for_query_data_row(row),
        ),
    )
    doc_ids.update(chunk_doc_ids)
    doc_ids.update(file_doc_ids)
    return doc_ids


async def _row_matches_tag_filter(
    rag: Any,
    row: dict[str, Any],
    tag_filter: TagFilter | None,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> bool:
    if not _tag_filter_active(tag_filter):
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


def _combine_row_doc_ids(
    row_direct_doc_ids: list[set[str]],
    row_chunk_ids: list[set[str]],
    row_file_paths: list[set[str]],
    chunk_docs: dict[str, str],
    file_docs: dict[str, str],
) -> tuple[list[set[str]], set[str]]:
    """Merge direct / chunk-resolved / file-resolved doc ids per row.

    Returns the per-row doc-id sets (parallel to the input rows) plus the union
    of every doc id seen, used to drive the tag prefetch.
    """
    row_doc_ids: list[set[str]] = []
    all_doc_ids: set[str] = set()
    for idx, direct_doc_ids in enumerate(row_direct_doc_ids):
        doc_ids = set(direct_doc_ids)
        doc_ids.update(
            doc_id
            for chunk_id in row_chunk_ids[idx]
            if (doc_id := chunk_docs.get(chunk_id))
        )
        doc_ids.update(
            doc_id
            for file_path in row_file_paths[idx]
            if (doc_id := file_docs.get(file_path))
        )
        row_doc_ids.append(doc_ids)
        all_doc_ids.update(doc_ids)
    return row_doc_ids, all_doc_ids


def _collect_row_doc_candidates(
    rows: list,
) -> tuple[list[dict[str, Any]], list[set[str]], list[set[str]], list[set[str]]]:
    """Gather per-row doc-id candidates (direct, chunk, file) for dict rows."""
    rows_for_filter: list[dict[str, Any]] = []
    row_direct_doc_ids: list[set[str]] = []
    row_chunk_ids: list[set[str]] = []
    row_file_paths: list[set[str]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        direct_doc_ids, row_chunks, row_paths = _query_data_row_doc_candidates(row)
        rows_for_filter.append(row)
        row_direct_doc_ids.append(direct_doc_ids)
        row_chunk_ids.append(row_chunks)
        row_file_paths.append(row_paths)
    return rows_for_filter, row_direct_doc_ids, row_chunk_ids, row_file_paths


async def _resolve_row_doc_maps(
    rag: Any, chunk_ids: set[str], file_paths: set[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve chunk-id and file-path doc maps, in parallel when both are small."""
    if (
        chunk_ids
        and file_paths
        and len(chunk_ids) + len(file_paths) <= _PARALLEL_RESOLVE_ID_BUDGET
    ):
        chunk_docs, file_docs = await asyncio.gather(
            _resolve_doc_ids_for_chunk_ids(rag, chunk_ids),
            _resolve_doc_ids_for_file_paths(rag, file_paths),
        )
    else:
        chunk_docs = await _resolve_doc_ids_for_chunk_ids(rag, chunk_ids)
        file_docs = await _resolve_doc_ids_for_file_paths(rag, file_paths)
    return chunk_docs, file_docs


def _keep_rows_matching_tags(
    rows_for_filter: list[dict[str, Any]],
    row_doc_ids: list[set[str]],
    tag_filter: TagFilter | None,
    tags_cache: dict[str, set[str]],
) -> tuple[list, set[str]]:
    """Keep rows with at least one tag-matching doc; collect reference_ids."""
    kept_rows = []
    kept_reference_ids: set[str] = set()
    for row, doc_ids in zip(rows_for_filter, row_doc_ids):
        if not doc_ids:
            continue
        if not any(
            _doc_tags_match_filter(tags_cache[doc_id], tag_filter) for doc_id in doc_ids
        ):
            continue
        kept_rows.append(row)
        ref_id = row.get("reference_id")
        if isinstance(ref_id, str) and ref_id:
            kept_reference_ids.add(ref_id)
    return kept_rows, kept_reference_ids


async def _filter_rows_by_tags(
    rag: Any,
    rows: list,
    tag_filter: TagFilter | None,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
    fetch_doc_tags_batch: Any = None,
    chunk_docs_cache: dict[str, str] | None = None,
    file_docs_cache: dict[str, str] | None = None,
) -> tuple[list, set[str]]:
    """Keep rows whose doc tags match the filter; collect their reference_ids."""
    if not _tag_filter_active(tag_filter):
        return rows, set()

    rows_for_filter, row_direct_doc_ids, row_chunk_ids, row_file_paths = (
        _collect_row_doc_candidates(rows)
    )
    if not rows_for_filter:
        return [], set()

    chunk_ids: set[str] = set().union(*row_chunk_ids)
    file_paths: set[str] = set().union(*row_file_paths)

    chunk_docs = chunk_docs_cache if chunk_docs_cache is not None else {}
    file_docs = file_docs_cache if file_docs_cache is not None else {}
    resolved_chunks, resolved_files = await _resolve_row_doc_maps(
        rag,
        chunk_ids - chunk_docs.keys(),
        file_paths - file_docs.keys(),
    )
    chunk_docs.update(resolved_chunks)
    file_docs.update(resolved_files)

    row_doc_ids, all_doc_ids = _combine_row_doc_ids(
        row_direct_doc_ids,
        row_chunk_ids,
        row_file_paths,
        chunk_docs,
        file_docs,
    )

    unresolved_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in tags_cache]
    if unresolved_doc_ids:
        if callable(fetch_doc_tags_batch):
            resolved_tags = await fetch_doc_tags_batch(unresolved_doc_ids, folder)
            resolved_tags = resolved_tags if isinstance(resolved_tags, dict) else {}
            tags_cache.update(
                (doc_id, set(resolved_tags.get(doc_id) or set()))
                for doc_id in unresolved_doc_ids
            )
        else:
            resolved_tags = await asyncio.gather(
                *(fetch_doc_tags(doc_id, folder) for doc_id in unresolved_doc_ids)
            )
            tags_cache.update(zip(unresolved_doc_ids, resolved_tags))

    return _keep_rows_matching_tags(
        rows_for_filter, row_doc_ids, tag_filter, tags_cache
    )


async def _filter_query_data_by_tags(
    rag: Any,
    response: dict[str, Any],
    tag_filter: TagFilter | None,
    folder: str,
    fetch_doc_tags: Any,
    fetch_doc_tags_batch: Any = None,
) -> dict[str, Any]:
    if not _tag_filter_active(tag_filter):
        return response

    data = response.get("data")
    if not isinstance(data, dict):
        return response

    # Cache shared across all rows in this single request, bounded by the number
    # of unique doc_ids in the result set.
    tags_cache: dict[str, set[str]] = {}
    chunk_docs_cache: dict[str, str] = {}
    file_docs_cache: dict[str, str] = {}
    chunk_rows = data.get("chunks")
    if isinstance(chunk_rows, list):
        for row in chunk_rows:
            if not isinstance(row, dict):
                continue
            direct_doc_ids = _direct_doc_ids_for_query_data_row(row)
            if len(direct_doc_ids) != 1:
                continue
            doc_id = next(iter(direct_doc_ids))
            for chunk_key in ("chunk_id", "id"):
                chunk_id = row.get(chunk_key)
                if isinstance(chunk_id, str) and chunk_id:
                    chunk_docs_cache.setdefault(chunk_id, doc_id)
            file_path = row.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            file_docs_cache.setdefault(file_path, doc_id)

    filtered_data = dict(data)
    kept_reference_ids: set[str] = set()
    for key in ("chunks", "entities", "relationships"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        kept_rows, ref_ids = await _filter_rows_by_tags(
            rag,
            rows,
            tag_filter,
            folder,
            tags_cache,
            fetch_doc_tags,
            fetch_doc_tags_batch,
            chunk_docs_cache,
            file_docs_cache,
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
