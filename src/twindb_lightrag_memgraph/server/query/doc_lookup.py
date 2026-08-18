"""DocStatus lookup and source-row projection helpers for Twin query routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _chunk_record_doc_id(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("full_doc_id", "doc_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _chunk_record_id(record: Any, requested_id: str) -> str:
    if isinstance(record, dict):
        for key in ("chunk_id", "id", "_id"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return requested_id


def _safe_get_score(result: dict[str, Any], rank: int, total: int) -> float:
    """Best-effort score for a retrieval row.

    The LightRAG ``MemgraphVectorDBStorage`` projection includes the cosine
    score under ``__metrics__`` / ``score`` depending on the backend version.
    When neither is present we synthesise a smooth rank-based value so the UI
    can sort consistently.
    """
    for key in ("score", "similarity", "cosine_similarity"):
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])
    metrics = result.get("__metrics__")
    if isinstance(metrics, dict):
        for key in ("score", "similarity", "cosine_similarity"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                return float(metrics[key])
    if total <= 0:
        return 0.5
    # Smooth descent from 0.95 (rank 0) to 0.50 (rank total-1).
    return round(0.95 - 0.45 * (rank / max(total - 1, 1)), 3)


def _chunk_to_meta(chunk: dict[str, Any]) -> str | None:
    """Cheap meta string: chunk order index when present, else short id suffix."""
    idx = chunk.get("chunk_order_index")
    if isinstance(idx, int):
        return f"chunk {idx}"
    cid = chunk.get("chunk_id") or chunk.get("id") or ""
    if isinstance(cid, str) and "-" in cid:
        return f"chunk {cid.split('-')[-1][:8]}"
    return None


async def _resolve_doc_for_chunk(rag: Any, chunk_id: str) -> str | None:
    """Best-effort chunk_id -> doc_id resolution through DocStatus."""
    if not chunk_id:
        return None
    try:
        get_by_chunks = getattr(rag.doc_status, "get_docs_by_chunks", None)
        if callable(get_by_chunks):
            result = await get_by_chunks([chunk_id])
            if isinstance(result, dict) and result:
                return next(iter(result.keys()))
    except Exception:
        logger.exception("twin_query: doc lookup failed for chunk %s", chunk_id)
    return None


async def _resolve_doc_for_file_path(rag: Any, file_path: str) -> str | None:
    """Best-effort file_path -> doc_id resolution for projected references."""
    if not file_path:
        return None
    try:
        get_by_file_path = getattr(rag.doc_status, "get_doc_by_file_path", None)
        if callable(get_by_file_path):
            result = await get_by_file_path(file_path)
            if isinstance(result, dict):
                doc_id = result.get("id") or result.get("doc_id")
                if isinstance(doc_id, str) and doc_id:
                    return doc_id
    except Exception:
        logger.exception("twin_query: doc lookup failed for file_path %s", file_path)
    return None


async def _resolve_chunk_to_doc_id(rag: Any, chunk_ids: list[str]) -> dict[str, str]:
    """Batch chunk_id -> doc_id resolution for the aquery_llm path."""
    if not chunk_ids:
        return {}

    unique = list(dict.fromkeys(chunk_ids))
    out: dict[str, str] = {}

    # LightRAG already stores ``full_doc_id`` on chunk records. Prefer its
    # set-based ``get_by_ids`` API over one ``get_docs_by_chunks([id])`` call
    # per reference. Try both exact-record stores because older runtimes expose
    # only one of them.
    for attr in ("text_chunks", "chunks_vdb"):
        unresolved = [chunk_id for chunk_id in unique if chunk_id not in out]
        if not unresolved:
            break
        store = getattr(rag, attr, None)
        get_by_ids = getattr(store, "get_by_ids", None)
        if not callable(get_by_ids):
            continue
        try:
            records = await get_by_ids(unresolved)
        except Exception:
            logger.exception("twin_query: %s.get_by_ids failed for doc lookup", attr)
            continue
        if not isinstance(records, list):
            continue
        for requested_id, record in zip(unresolved, records):
            doc_id = _chunk_record_doc_id(record)
            if doc_id:
                out[_chunk_record_id(record, requested_id)] = doc_id

    # Alternate DocStatus implementations may expose only the legacy
    # singleton lookup. Keep that compatibility path for unresolved ids.
    unresolved = [chunk_id for chunk_id in unique if chunk_id not in out]
    if unresolved:
        resolved = await asyncio.gather(
            *(_resolve_doc_for_chunk(rag, chunk_id) for chunk_id in unresolved),
            return_exceptions=False,
        )
        for chunk_id, doc_id in zip(unresolved, resolved):
            if doc_id:
                out[chunk_id] = doc_id
    return out


async def _resolve_file_paths_to_doc_ids(
    rag: Any, file_paths: list[str]
) -> dict[str, str]:
    """Resolve unique source paths with one set-based read when supported."""
    unique = list(dict.fromkeys(path for path in file_paths if path))
    if not unique:
        return {}

    doc_status = getattr(rag, "doc_status", None)
    get_many = getattr(doc_status, "get_docs_by_file_paths", None)
    if callable(get_many):
        try:
            records = await get_many(unique)
        except Exception:
            logger.exception("twin_query: batch file_path doc lookup failed")
        else:
            if isinstance(records, dict):
                out: dict[str, str] = {}
                for file_path, record in records.items():
                    if not isinstance(file_path, str) or not isinstance(record, dict):
                        continue
                    doc_id = record.get("id") or record.get("doc_id")
                    if isinstance(doc_id, str) and doc_id:
                        out[file_path] = doc_id
                return out

    resolved = await asyncio.gather(
        *(_resolve_doc_for_file_path(rag, file_path) for file_path in unique),
        return_exceptions=False,
    )
    return {
        file_path: doc_id
        for file_path, doc_id in zip(unique, resolved)
        if isinstance(doc_id, str) and doc_id
    }


__all__ = [
    "_chunk_to_meta",
    "_resolve_chunk_to_doc_id",
    "_resolve_doc_for_chunk",
    "_resolve_doc_for_file_path",
    "_resolve_file_paths_to_doc_ids",
    "_safe_get_score",
]
