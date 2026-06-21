"""Document endpoints for the Twin WebUI.

This module intentionally calls the legacy helper symbols through
``server.webui_router`` at runtime. Several tests and adjacent modules still
patch those helpers there; keeping that lookup dynamic preserves compatibility
while the large router is split incrementally.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from ..webui_models import Document, ListEnvelope
from .events import _make_event
from .store import get_store

router = APIRouter()


@router.get("/documents", response_model=ListEnvelope[Document])
async def list_documents(
    status: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
    tag: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    from .. import webui_router as legacy

    store = get_store()
    with store._lock:  # noqa: SLF001 - route/store coordination
        has_seed_documents = bool(store._documents)  # noqa: SLF001
    if has_seed_documents:
        items = store.list_documents(status=status, q=q, tag=tag)
        return {"items": items, "total": len(items)}
    try:
        items = await legacy._list_documents_from_doc_status(
            status=status, q=q, tag=tag
        )
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        items = store.list_documents(status=status, q=q, tag=tag)
    return {"items": items, "total": len(items)}


@router.get("/documents/{doc_id}/metadata")
async def get_document_metadata(doc_id: str) -> dict[str, Any]:
    from .. import webui_router as legacy

    doc = await legacy._get_doc_for_active_folder(doc_id)
    metadata = doc.get("metadata") or {}
    graph_tags = await legacy._graph_tags_for_doc(doc_id)
    tags = graph_tags or list(metadata.get("tags") or doc.get("tags") or [])
    folder = doc.get("folder") or metadata.get("folder") or legacy.current_folder_id()
    return {
        "tags": tags,
        "folder": folder,
        "review": metadata.get("review"),
        "classification": metadata.get("classification"),
        "metadata": metadata,
    }


@router.post("/documents/bulk-delete")
async def bulk_delete_documents(body: dict[str, Any]) -> dict[str, Any]:
    from .. import webui_router as legacy

    doc_ids = body.get("doc_ids")
    actor = body.get("actor") or "system"
    if not isinstance(doc_ids, list) or not doc_ids:
        raise HTTPException(
            status_code=400,
            detail="doc_ids must be a non-empty list of document ids.",
        )
    if len(doc_ids) > 500:
        raise HTTPException(
            status_code=413,
            detail="bulk-delete accepts at most 500 target documents.",
        )

    rag = legacy._get_rag()
    deleted = 0
    failed: list[str] = []
    for doc_id in doc_ids:
        if not isinstance(doc_id, str) or not doc_id:
            failed.append(str(doc_id))
            continue
        try:
            doc = await legacy._get_doc_for_active_folder(doc_id)
            await legacy._delete_doc_from_rag(rag, doc_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                failed.append(doc_id)
                continue
            raise
        except Exception:
            failed.append(doc_id)
            continue

        deleted += 1
        event = _make_event(
            kind="doc-deleted",
            sev="info",
            actor=actor,
            target_label=doc.get("file_path") or doc_id,
            summary=f"deleted by {actor}",
            meta={"doc_id": doc_id, "operation": "bulk-delete"},
            target_type="document",
        )
        await get_store().record_activity(event)

    return {"deleted": deleted, "failed": failed}
