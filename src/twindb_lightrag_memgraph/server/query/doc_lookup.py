"""DocStatus lookup and source-row projection helpers for Twin query routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    resolved = await asyncio.gather(
        *(_resolve_doc_for_chunk(rag, chunk_id) for chunk_id in unique),
        return_exceptions=False,
    )
    out: dict[str, str] = {}
    for chunk_id, doc_id in zip(unique, resolved):
        if doc_id:
            out[chunk_id] = doc_id
    return out


__all__ = [
    "_chunk_to_meta",
    "_resolve_chunk_to_doc_id",
    "_resolve_doc_for_chunk",
    "_resolve_doc_for_file_path",
    "_safe_get_score",
]
