"""Custom chunk & document routes for DocumentProvider support.

Exposes chunk-to-document lookups and neighbouring-chunk retrieval
on top of LightRAG's internal KV stores (text_chunks, full_docs,
doc_status).

Endpoints
---------
GET /chunks/{chunk_id}/context?window=3
    Neighbouring chunks around a chunk (same document).

GET /chunks/{chunk_id}/document
    All chunks of the parent document for a given chunk.

GET /documents/{doc_id}/chunks?start=0&end=10
    Fetch a positional range of chunks from a document.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from lightrag import LightRAG
from pydantic import BaseModel

from .auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chunks"], dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ChunkItem(BaseModel):
    chunk_id: str
    content: str
    full_doc_id: str
    file_path: str
    chunk_order_index: int
    tokens: int


class ChunkContextResponse(BaseModel):
    chunks: list[ChunkItem]
    doc_id: str
    file_path: str
    total_chunks_in_doc: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _resolve_chunk(rag: LightRAG, chunk_id: str) -> dict[str, Any]:
    """Fetch a chunk record from ``text_chunks`` KV, raise 404 if missing."""
    chunk_data: dict[str, Any] | None = await rag.text_chunks.get_by_id(chunk_id)
    if not chunk_data:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
    return chunk_data


async def _get_ordered_chunk_ids(rag: LightRAG, doc_id: str) -> list[str]:
    """Return the ordered list of chunk IDs for a document.

    Prefers ``doc_status.chunks_list`` (populated during ingestion).
    Falls back to 404 when unavailable.
    """
    doc_status_data = await rag.doc_status.get_by_id(doc_id)

    if doc_status_data is not None:
        chunks_list: list[str] | None = None
        if hasattr(doc_status_data, "chunks_list"):
            chunks_list = getattr(doc_status_data, "chunks_list", None)
        elif isinstance(doc_status_data, dict):
            chunks_list = doc_status_data.get("chunks_list")

        if chunks_list:
            return chunks_list

    raise HTTPException(
        status_code=404,
        detail=f"No chunk ordering found for document '{doc_id}'",
    )


async def _fetch_chunks_by_ids(
    rag: LightRAG,
    chunk_ids: list[str],
) -> list[ChunkItem]:
    """Batch-fetch chunk records and return as ``ChunkItem`` list, preserving order."""
    raw_list: list[dict[str, Any]] = await rag.text_chunks.get_by_ids(chunk_ids)

    # get_by_ids may return items in arbitrary order -- reindex by chunk_id
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_list:
        if raw and isinstance(raw, dict):
            cid = raw.get("_id") or raw.get("chunk_id", "")
            if cid:
                by_id[cid] = raw

    items: list[ChunkItem] = []
    for idx, cid in enumerate(chunk_ids):
        raw = by_id.get(cid)
        if raw is None:
            continue
        items.append(
            ChunkItem(
                chunk_id=cid,
                content=raw.get("content", ""),
                full_doc_id=raw.get("full_doc_id", ""),
                file_path=raw.get("file_path", ""),
                chunk_order_index=raw.get("chunk_order_index", idx),
                tokens=raw.get("tokens", 0),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def create_chunk_routes(rag: LightRAG | Callable[[], LightRAG]) -> None:
    """Register chunk & document routes against the module-level ``router``."""
    router.routes.clear()

    def current_rag() -> LightRAG:
        return rag() if callable(rag) and not hasattr(rag, "text_chunks") else rag

    @router.get(
        "/chunks/{chunk_id}/context",
        response_model=ChunkContextResponse,
        operation_id="get_chunk_context",
        summary="Neighbouring chunks around a given chunk",
    )
    async def get_chunk_context(
        chunk_id: str,
        window: int = Query(
            default=3,
            ge=1,
            le=50,
            description="Chunks before/after to include",
        ),
    ) -> ChunkContextResponse:
        active_rag = current_rag()
        anchor = await _resolve_chunk(active_rag, chunk_id)
        doc_id: str = anchor.get("full_doc_id", "")
        if not doc_id:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk '{chunk_id}' has no parent document",
            )

        ordered_ids = await _get_ordered_chunk_ids(active_rag, doc_id)
        try:
            idx = ordered_ids.index(chunk_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk '{chunk_id}' not found in document '{doc_id}' chunk ordering",
            ) from exc

        start = max(0, idx - window)
        end = min(len(ordered_ids), idx + window + 1)
        window_ids = ordered_ids[start:end]

        items = await _fetch_chunks_by_ids(active_rag, window_ids)
        return ChunkContextResponse(
            chunks=items,
            doc_id=doc_id,
            file_path=anchor.get("file_path", ""),
            total_chunks_in_doc=len(ordered_ids),
        )

    @router.get(
        "/chunks/{chunk_id}/document",
        response_model=ChunkContextResponse,
        operation_id="get_chunk_document",
        summary="All chunks of the parent document for a given chunk",
    )
    async def get_chunk_document(chunk_id: str) -> ChunkContextResponse:
        active_rag = current_rag()
        anchor = await _resolve_chunk(active_rag, chunk_id)
        doc_id: str = anchor.get("full_doc_id", "")
        if not doc_id:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk '{chunk_id}' has no parent document",
            )

        ordered_ids = await _get_ordered_chunk_ids(active_rag, doc_id)
        items = await _fetch_chunks_by_ids(active_rag, ordered_ids)
        return ChunkContextResponse(
            chunks=items,
            doc_id=doc_id,
            file_path=anchor.get("file_path", ""),
            total_chunks_in_doc=len(ordered_ids),
        )

    @router.get(
        "/documents/{doc_id}/chunks",
        response_model=ChunkContextResponse,
        operation_id="get_document_chunks",
        summary="Fetch a range (or all) chunks from a document by doc_id",
    )
    async def get_document_chunks(
        doc_id: str,
        start: int | None = Query(
            default=None, ge=0, description="Start index (inclusive)"
        ),
        end: int | None = Query(
            default=None, ge=0, description="End index (inclusive)"
        ),
    ) -> ChunkContextResponse:
        active_rag = current_rag()
        ordered_ids = await _get_ordered_chunk_ids(active_rag, doc_id)
        total = len(ordered_ids)

        if start is not None or end is not None:
            s = start or 0
            e = (end or total - 1) + 1  # inclusive end -> slice end
            ordered_ids = ordered_ids[s:e]

        items = await _fetch_chunks_by_ids(active_rag, ordered_ids)
        file_path = items[0].file_path if items else ""

        return ChunkContextResponse(
            chunks=items,
            doc_id=doc_id,
            file_path=file_path,
            total_chunks_in_doc=total,
        )
