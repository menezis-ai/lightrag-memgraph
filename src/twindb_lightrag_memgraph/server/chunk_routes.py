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
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from lightrag import LightRAG
from pydantic import BaseModel

from .auth import require_auth
from .folder import bind_request_folder, current_folder_id, load_folder_catalog

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["chunks"],
    dependencies=[Depends(require_auth), Depends(bind_request_folder)],
)


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
    return _ordered_chunk_ids(doc_status_data, doc_id)


def _ordered_chunk_ids(doc_status_data: Any, doc_id: str) -> list[str]:
    """Extract chunk ordering from an already-loaded DocStatus record."""

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


def _doc_field(doc_status_data: Any, key: str, default: Any = None) -> Any:
    if isinstance(doc_status_data, dict):
        return doc_status_data.get(key, default)
    return getattr(doc_status_data, key, default)


def _doc_metadata(doc_status_data: Any) -> dict[str, Any]:
    metadata = _doc_field(doc_status_data, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _doc_matches_active_folder_legacy(doc_status_data: Any) -> bool:
    metadata = _doc_metadata(doc_status_data)
    default_folder = load_folder_catalog().default_folder_id
    folder = (
        _doc_field(doc_status_data, "folder")
        or metadata.get("folder")
        or default_folder
    )
    return folder == current_folder_id()


async def _require_doc_in_active_folder(rag: LightRAG, doc_id: str) -> Any:
    """Return DocStatus only when ``doc_id`` is visible in the active folder."""
    doc_status_data = await rag.doc_status.get_by_id(doc_id)
    if doc_status_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found",
        )

    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    if get_folders is not None:
        folders = await get_folders(doc_id)
        in_folder = current_folder_id() in (folders or [])
    else:
        in_folder = _doc_matches_active_folder_legacy(doc_status_data)

    if not in_folder:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found",
        )
    return doc_status_data


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


def _parent_doc_id(anchor: dict[str, Any], chunk_id: str) -> str:
    doc_id: str = anchor.get("full_doc_id", "")
    if not doc_id:
        raise HTTPException(
            status_code=404,
            detail=f"Chunk '{chunk_id}' has no parent document",
        )
    return doc_id


def _chunk_context_window(
    ordered_ids: list[str],
    chunk_id: str,
    doc_id: str,
    window: int,
) -> list[str]:
    try:
        idx = ordered_ids.index(chunk_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Chunk '{chunk_id}' not found in document '{doc_id}' chunk ordering",
        ) from exc

    start = max(0, idx - window)
    end = min(len(ordered_ids), idx + window + 1)
    return ordered_ids[start:end]


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
        operation_id="get_chunk_context",
        summary="Neighbouring chunks around a given chunk",
        responses={404: {"description": "Chunk or parent document not found"}},
    )
    async def get_chunk_context(
        chunk_id: Annotated[
            str,
            Path(
                description="Chunk identifier returned in a query source.",
                examples=["chunk-4f1a2b"],
            ),
        ],
        window: Annotated[
            int,
            Query(
                ge=1,
                le=50,
                description="Number of chunks before and after to include (1-50).",
            ),
        ] = 3,
    ) -> ChunkContextResponse:
        """Return the chunk's neighbours inside its document — `window`
        chunks before and after, in document order. Useful to read the
        surrounding context of a chunk cited as an answer source. Chunk
        ids come from the `sources` of `POST /twin/api/query`."""
        active_rag = current_rag()
        anchor = await _resolve_chunk(active_rag, chunk_id)
        doc_id = _parent_doc_id(anchor, chunk_id)
        doc_status_data = await _require_doc_in_active_folder(active_rag, doc_id)
        ordered_ids = _ordered_chunk_ids(doc_status_data, doc_id)
        window_ids = _chunk_context_window(ordered_ids, chunk_id, doc_id, window)
        items = await _fetch_chunks_by_ids(active_rag, window_ids)
        return ChunkContextResponse(
            chunks=items,
            doc_id=doc_id,
            file_path=anchor.get("file_path", ""),
            total_chunks_in_doc=len(ordered_ids),
        )

    @router.get(
        "/chunks/{chunk_id}/document",
        operation_id="get_chunk_document",
        summary="All chunks of the parent document for a given chunk",
        responses={404: {"description": "Chunk or parent document not found"}},
    )
    async def get_chunk_document(
        chunk_id: Annotated[
            str,
            Path(
                description="Chunk identifier whose parent document is requested.",
                examples=["chunk-4f1a2b"],
            ),
        ],
    ) -> ChunkContextResponse:
        """Return every chunk of the document the given chunk belongs to,
        in document order — the full text a cited chunk was taken from."""
        active_rag = current_rag()
        anchor = await _resolve_chunk(active_rag, chunk_id)
        doc_id = _parent_doc_id(anchor, chunk_id)
        doc_status_data = await _require_doc_in_active_folder(active_rag, doc_id)
        ordered_ids = _ordered_chunk_ids(doc_status_data, doc_id)
        items = await _fetch_chunks_by_ids(active_rag, ordered_ids)
        return ChunkContextResponse(
            chunks=items,
            doc_id=doc_id,
            file_path=anchor.get("file_path", ""),
            total_chunks_in_doc=len(ordered_ids),
        )

    @router.get(
        "/documents/{doc_id}/chunks",
        operation_id="get_document_chunks",
        summary="Fetch a range (or all) chunks from a document by doc_id",
        responses={404: {"description": "Document chunk ordering not found"}},
    )
    async def get_document_chunks(
        doc_id: Annotated[
            str,
            Path(
                description="Document identifier returned by the document list.",
                examples=["doc-7c91e2"],
            ),
        ],
        start: Annotated[
            int | None,
            Query(ge=0, description="First chunk position to return (inclusive)."),
        ] = None,
        end: Annotated[
            int | None,
            Query(ge=0, description="Last chunk position to return (inclusive)."),
        ] = None,
    ) -> ChunkContextResponse:
        """Return the document's chunks in order. Without `start`/`end`
        the whole document is returned; with them, only the requested
        positional range. `total_chunks_in_doc` always reports the full
        count."""
        active_rag = current_rag()
        doc_status_data = await _require_doc_in_active_folder(active_rag, doc_id)
        ordered_ids = _ordered_chunk_ids(doc_status_data, doc_id)
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
