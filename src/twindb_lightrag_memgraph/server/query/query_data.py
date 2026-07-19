"""Structured ``/query/data`` enrichment helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .doc_lookup import _resolve_doc_for_chunk, _safe_get_score
from .source_filters import _split_source_ids

logger = logging.getLogger(__name__)


def _iter_kg_rows(data: dict[str, Any]):
    """Yield the dict rows of a structured payload's KG sections in order."""
    for key in ("entities", "relationships"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                yield row


def _query_data_source_chunk_ids(data: dict[str, Any]) -> list[str]:
    """Collect chunk ids referenced by KG rows in a structured data payload."""
    chunk_ids, _scores = _query_data_source_chunk_projection(data)
    return chunk_ids


def _query_data_existing_chunk_ids(data: dict[str, Any]) -> set[str]:
    rows = data.get("chunks")
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("chunk_id", "id", "_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                out.add(value)
    return out


def _query_data_source_chunk_scores(data: dict[str, Any]) -> dict[str, float]:
    """Best-effort score per chunk id from KG rows that reference it."""
    _chunk_ids, scores = _query_data_source_chunk_projection(data)
    return scores


def _query_data_source_chunk_projection(
    data: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    """Collect referenced chunk ids and their best KG-derived scores."""
    kg_rows = list(_iter_kg_rows(data))
    total = len(kg_rows)
    if total == 0:
        return [], {}

    chunk_ids: list[str] = []
    seen: set[str] = set()
    scores: dict[str, float] = {}
    for rank, row in enumerate(kg_rows):
        row_chunk_ids = _split_source_ids(row.get("source_id"))
        if not row_chunk_ids:
            continue
        score = _safe_get_score(row, rank, total)
        for chunk_id in row_chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
            scores[chunk_id] = max(scores.get(chunk_id, score), score)
    return chunk_ids, scores


def _query_data_chunk_id(row: dict[str, Any]) -> str | None:
    for key in ("chunk_id", "id", "_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _annotate_query_data_chunk_scores(
    data: dict[str, Any],
    source_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = data.get("chunks")
    if not isinstance(rows, list):
        return data

    if source_scores is None:
        source_scores = _query_data_source_chunk_scores(data)
    total = len(rows)
    annotated_rows: list[Any] = []
    changed = False
    for rank, row in enumerate(rows):
        if not isinstance(row, dict):
            annotated_rows.append(row)
            continue
        annotated = dict(row)
        chunk_id = _query_data_chunk_id(annotated)
        if not isinstance(annotated.get("score"), (int, float)):
            if chunk_id and chunk_id in source_scores:
                annotated["score"] = source_scores[chunk_id]
            else:
                annotated["score"] = _safe_get_score(annotated, rank, total)
            changed = True
        annotated_rows.append(annotated)

    if not changed:
        return data
    annotated_data = dict(data)
    annotated_data["chunks"] = annotated_rows
    return annotated_data


def _next_query_data_reference_id(data: dict[str, Any]) -> int:
    references = data.get("references")
    if not isinstance(references, list):
        return 1
    numeric_ids: list[int] = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        try:
            numeric_ids.append(int(str(ref.get("reference_id") or "")))
        except ValueError:
            continue
    return max(numeric_ids, default=0) + 1


def _query_data_record_id(raw: dict[str, Any], requested_id: str) -> str | None:
    record_id = raw.get("chunk_id") or raw.get("id") or raw.get("_id") or requested_id
    if isinstance(record_id, str) and record_id:
        return record_id
    return None


async def _fetch_chunk_records_from_store(
    rag: Any,
    attr: str,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    store = getattr(rag, attr, None)
    get_by_ids = getattr(store, "get_by_ids", None)
    if not callable(get_by_ids):
        return {}
    try:
        raw_list = await get_by_ids(chunk_ids)
    except Exception:
        logger.exception("twin_query: %s.get_by_ids failed for query/data", attr)
        return {}
    if not isinstance(raw_list, list):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for requested_id, raw in zip(chunk_ids, raw_list):
        if not isinstance(raw, dict):
            continue
        record_id = _query_data_record_id(raw, requested_id)
        if record_id is not None:
            records.setdefault(record_id, dict(raw))
    return records


async def _fetch_chunk_records_by_id(
    rag: Any,
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch exact chunk records by id without issuing semantic retrieval."""
    if not chunk_ids:
        return {}
    unique = list(dict.fromkeys(chunk_ids))
    out: dict[str, dict[str, Any]] = {}

    for attr in ("text_chunks", "chunks_vdb"):
        records = await _fetch_chunk_records_from_store(rag, attr, unique)
        out.update({key: value for key, value in records.items() if key not in out})
        if len(out) == len(unique):
            break
    return out


async def _doc_file_path(rag: Any, doc_id: str | None) -> str | None:
    if not doc_id:
        return None
    try:
        get_by_id = getattr(rag.doc_status, "get_by_id", None)
        if not callable(get_by_id):
            return None
        doc = await get_by_id(doc_id)
    except Exception:
        logger.exception("twin_query: doc lookup failed for doc %s", doc_id)
        return None
    if isinstance(doc, dict):
        file_path = doc.get("file_path")
        if isinstance(file_path, str) and file_path:
            return file_path
    return None


async def _query_data_row_doc_id(
    rag: Any,
    row: dict[str, Any],
    chunk_id: str,
) -> str | None:
    doc_id = row.get("full_doc_id") or row.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    resolved = await _resolve_doc_for_chunk(rag, chunk_id)
    if resolved:
        row["full_doc_id"] = resolved
    return resolved


async def _query_data_row_file_path(
    rag: Any,
    row: dict[str, Any],
    doc_id: str | None,
) -> str | None:
    file_path = row.get("file_path") or row.get("source")
    if isinstance(file_path, str) and file_path:
        return file_path
    resolved = await _doc_file_path(rag, doc_id)
    if resolved:
        row["file_path"] = resolved
    return resolved


async def _query_data_chunk_row(
    rag: Any,
    chunk_id: str,
    raw: dict[str, Any],
    reference_id: str,
    score: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = dict(raw)
    row["chunk_id"] = str(
        row.get("chunk_id") or row.get("id") or row.get("_id") or chunk_id
    )
    doc_id = await _query_data_row_doc_id(rag, row, chunk_id)
    file_path = await _query_data_row_file_path(rag, row, doc_id)
    row["reference_id"] = reference_id
    if score is not None and not isinstance(row.get("score"), (int, float)):
        row["score"] = score
    return row, {
        "reference_id": reference_id,
        "file_path": str(file_path or chunk_id),
    }


async def _enrich_query_data_chunks_from_source_ids(
    rag: Any,
    response: dict[str, Any],
) -> dict[str, Any]:
    """Materialize chunks already referenced by KG rows.

    ``aquery_data(mode="hybrid")`` can return entities/relationships with
    ``source_id`` provenance while leaving ``data.chunks`` and
    ``data.references`` empty. For filtered API calls that makes the visible
    payload look sourceless even though LightRAG gave us exact chunk ids. This
    helper fetches those exact chunks by id; it never runs a new vector query.
    """
    data = response.get("data")
    if not isinstance(data, dict):
        return response

    source_chunk_ids, source_scores = _query_data_source_chunk_projection(data)
    existing_chunk_ids = _query_data_existing_chunk_ids(data)
    missing = [
        chunk_id for chunk_id in source_chunk_ids if chunk_id not in existing_chunk_ids
    ]
    if not missing:
        enriched = dict(response)
        enriched["data"] = _annotate_query_data_chunk_scores(data, source_scores)
        return enriched

    records = await _fetch_chunk_records_by_id(rag, missing)
    if not records:
        enriched = dict(response)
        enriched["data"] = _annotate_query_data_chunk_scores(data, source_scores)
        return enriched

    enriched_data = dict(data)
    raw_chunks = data.get("chunks")
    raw_references = data.get("references")
    chunks = list(raw_chunks) if isinstance(raw_chunks, list) else []
    references = list(raw_references) if isinstance(raw_references, list) else []
    next_ref = _next_query_data_reference_id(data)
    chunk_rows_to_fetch: list[Any] = []
    for chunk_id in missing:
        raw = records.get(chunk_id)
        if not isinstance(raw, dict):
            continue
        ref_id = str(next_ref)
        next_ref += 1
        chunk_rows_to_fetch.append(
            _query_data_chunk_row(
                rag,
                chunk_id,
                raw,
                ref_id,
                source_scores.get(chunk_id),
            )
        )
    if chunk_rows_to_fetch:
        for chunk_row, reference in await asyncio.gather(*chunk_rows_to_fetch):
            chunks.append(chunk_row)
            references.append(reference)

    enriched_data["chunks"] = chunks
    enriched_data["references"] = references
    enriched_data = _annotate_query_data_chunk_scores(enriched_data, source_scores)
    enriched = dict(response)
    enriched["data"] = enriched_data
    return enriched


__all__ = [
    "_annotate_query_data_chunk_scores",
    "_enrich_query_data_chunks_from_source_ids",
]
