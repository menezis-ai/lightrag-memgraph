"""Document endpoints for the Twin WebUI.

This module intentionally calls the legacy helper symbols through
``server.webui_router`` at runtime. Several tests and adjacent modules still
patch those helpers there; keeping that lookup dynamic preserves compatibility
while the large router is split incrementally.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..idp_jwt import require_admin_user
from ..webui_models import Document, ListEnvelope
from .events import _make_event
from .store import get_store

router = APIRouter()

# Per-document lock serialising membership add/remove so a concurrent add cannot
# slip in between "is this the last membership?" and the physical delete
# (architect review P1 race).
#
# LIMITATION (architect review P2): this lock is **process-local**. It closes
# the race within one ASGI worker but NOT across multiple workers/processes. The
# production runtime (gunicorn/uvicorn import-string per worker, see asgi.py) is
# likely multi-worker, so a cross-worker last-membership race window remains
# (narrow, and the routes are admin-gated). A proper fix moves the
# decide-and-delete into a storage-level atomic op or a workspace/distributed
# lock — tracked as a follow-up in FOLDER-MEMBERSHIP-REFACTOR.md. The dict is
# also unbounded; bound it (TTL / WeakValueDictionary) when this leaves batch 1.
_membership_locks: dict[str, asyncio.Lock] = {}


def _membership_lock(doc_id: str) -> asyncio.Lock:
    lock = _membership_locks.get(doc_id)
    if lock is None:
        lock = asyncio.Lock()
        _membership_locks[doc_id] = lock
    return lock


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


@router.get(
    "/documents/{doc_id}/metadata",
    responses={
        404: {"description": "Document not found"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
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


# ── Folder membership (one document, many folders, stored once) ──────────
# Explicit membership endpoints are the PRIMARY contract for sharing a document
# across folders without duplicating its data. See FOLDER-MEMBERSHIP-REFACTOR.md.
# folder_id is validated against the provisioned catalog so a typo cannot create
# an orphan Folder node or punch a cloisonnement hole.
#
# Documented residual risks (architect reviews; accepted for this batch):
#   - AUTHZ: the mutation routes are gated by ``require_admin_user``. That gate
#     is two-tier: with an IdP active (``TWIN_IDP_JWKS_URL`` set) it requires the
#     ``admin:folders`` scope; with the IdP DORMANT it degrades to "authenticated"
#     (idp_jwt.py palier 1). The BNP target runs MyAccess + Broadcom SSO (JWKS
#     active), so the strict admin gate applies there; a no-IdP dev/local run is
#     only "authenticated". Per-user source-doc + target-folder RBAC is owned by
#     MyAccess/SSO, not implemented here.
#   - CONCURRENCY: the per-doc lock is process-local (see ``_membership_locks``).
#   - LEGACY READS: other folder-scoped reads (chunks routes, native shims) still
#     consult the legacy ``folder`` property and are migrated in a later batch.


def _require_known_folder(folder_id: str) -> str:
    """Validate folder_id is a non-empty, provisioned (catalog) folder."""
    from .. import folder as folder_mod

    fid = (folder_id or "").strip()
    if not fid:
        raise HTTPException(status_code=400, detail="folder_id is required.")
    known = {s.id for s in folder_mod.load_folder_catalog().folders}
    if fid not in known:
        raise HTTPException(status_code=404, detail=f"Unknown folder '{fid}'.")
    return fid


@router.get(
    "/documents/{doc_id}/folders",
    responses={
        404: {"description": "Document not found / not visible in active folder"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def list_document_folders(doc_id: str) -> dict[str, Any]:
    from .. import webui_router as legacy

    # Cloisonnement (architect review P1): only a caller who can already see the
    # doc in their ACTIVE folder may enumerate its memberships. Otherwise a known
    # doc_id would leak existence + cross-folder membership. _get_doc_for_active_
    # folder 404s when the doc is absent or not a member of the active folder.
    await legacy._get_doc_for_active_folder(doc_id)
    rag = legacy._get_rag()
    folders = await rag.doc_status.get_folders_for_doc(doc_id)
    return {"doc_id": doc_id, "folders": folders or []}


@router.post(
    "/documents/{doc_id}/folders",
    # Admin-gated INTERIM authorization (architect review P1): until per-user
    # source-doc + target-folder RBAC lands (spec §3.4), only operators with the
    # admin gateway scope may mutate membership — never broadly exposed.
    dependencies=[Depends(require_admin_user)],
    responses={
        400: {"description": "Missing folder_id"},
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document or folder not found"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def add_document_to_folder(doc_id: str, body: dict[str, Any]) -> dict[str, Any]:
    from .. import webui_router as legacy

    folder_id = _require_known_folder(str(body.get("folder_id") or ""))
    actor = str(body.get("actor") or "system")

    rag = legacy._get_rag()
    async with _membership_lock(doc_id):
        ok = await rag.doc_status.add_to_folder(doc_id, folder_id)
        if not ok:
            raise HTTPException(
                status_code=404, detail=f"Document '{doc_id}' not found."
            )
        folders = await rag.doc_status.get_folders_for_doc(doc_id)

    event = _make_event(
        kind="doc-folder-added",
        sev="info",
        actor=actor,
        target_label=doc_id,
        summary=f"added to folder {folder_id} by {actor}",
        meta={"doc_id": doc_id, "folder_id": folder_id, "operation": "add-membership"},
        target_type="document",
    )
    await get_store().record_activity(event)
    return {"doc_id": doc_id, "folders": folders}


@router.delete(
    "/documents/{doc_id}/folders/{folder_id}",
    # Admin-gated interim authorization — see add_document_to_folder.
    dependencies=[Depends(require_admin_user)],
    responses={
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document or folder not found"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def remove_document_from_folder(
    doc_id: str,
    folder_id: str,
    actor: Annotated[str, Query()] = "system",
) -> dict[str, Any]:
    """Remove a doc from one folder. When this was its LAST membership, the
    document is physically deleted.

    Two correctness guards from the architect reviews:
    - *Ordering (P1.2):* the physical delete runs WHILE the membership edge still
      exists — native ``DETACH DELETE`` removes the node and its edges together,
      so a failed delete leaves the doc intact and still MEMBER_OF this folder
      (recoverable), never orphaned and invisible to membership reads.
    - *Race (P1):* the read-decide-delete runs under a per-doc lock shared with
      ``add_document_to_folder``, so a concurrent add cannot turn a last-folder
      removal into a physical delete of a doc that just gained a folder.
    """
    from .. import webui_router as legacy

    folder_id = _require_known_folder(folder_id)
    rag = legacy._get_rag()

    async with _membership_lock(doc_id):
        folders = await rag.doc_status.get_folders_for_doc(doc_id)
        if folders is None:
            raise HTTPException(
                status_code=404, detail=f"Document '{doc_id}' not found."
            )
        if folder_id not in folders:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document '{doc_id}' is not a member of folder '{folder_id}'."
                ),
            )

        physically_deleted = False
        if folders == [folder_id]:
            # Last membership: physically delete (node + edge removed together).
            await legacy._delete_doc_from_rag(rag, doc_id)
            physically_deleted = True
            remaining = 0
        else:
            await rag.doc_status.remove_from_folder(doc_id, folder_id)
            remaining = len(folders) - 1

    event = _make_event(
        kind="doc-deleted" if physically_deleted else "doc-folder-removed",
        sev="info",
        actor=actor,
        target_label=doc_id,
        summary=(
            f"removed from folder {folder_id} by {actor}"
            + (" (last folder → physically deleted)" if physically_deleted else "")
        ),
        meta={
            "doc_id": doc_id,
            "folder_id": folder_id,
            "operation": "remove-membership",
            "physically_deleted": physically_deleted,
            "remaining_folders": remaining,
        },
        target_type="document",
    )
    await get_store().record_activity(event)
    return {
        "doc_id": doc_id,
        "remaining_folders": remaining,
        "physically_deleted": physically_deleted,
    }


def _parse_bulk_delete_body(body: dict[str, Any]) -> tuple[list[Any], str]:
    doc_ids = body.get("doc_ids")
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
    return doc_ids, str(body.get("actor") or "system")


async def _delete_one_document(legacy: Any, rag: Any, doc_id: Any, actor: str) -> bool:
    """Delete a doc from the bulk-delete surface, ref-counted (architect P1).

    "Delete" here means **remove from the active folder**: a doc shared across
    folders is only un-shared from the current one, and its physical data
    (node/chunks/vectors) is removed only when this was its LAST membership.
    Backends without membership fall back to the legacy hard delete.
    """
    if not isinstance(doc_id, str) or not doc_id:
        return False
    active = legacy.current_folder_id()
    try:
        doc = await legacy._get_doc_for_active_folder(doc_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        return False
    except Exception:
        return False

    physically_deleted = False
    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    try:
        if get_folders is None:
            await legacy._delete_doc_from_rag(rag, doc_id)
            physically_deleted = True
        else:
            async with _membership_lock(doc_id):
                folders = await get_folders(doc_id)
                if folders is None or active not in folders:
                    return False
                if folders == [active]:
                    # Last membership → physical delete (edge intact until it
                    # succeeds, per the ordering guard).
                    await legacy._delete_doc_from_rag(rag, doc_id)
                    physically_deleted = True
                else:
                    await rag.doc_status.remove_from_folder(doc_id, active)
    except Exception:
        return False

    event = _make_event(
        kind="doc-deleted" if physically_deleted else "doc-folder-removed",
        sev="info",
        actor=actor,
        target_label=doc.get("file_path") or doc_id,
        summary=(
            f"deleted by {actor}"
            if physically_deleted
            else f"removed from folder {active} by {actor}"
        ),
        meta={
            "doc_id": doc_id,
            "operation": "bulk-delete",
            "folder": active,
            "physically_deleted": physically_deleted,
        },
        target_type="document",
    )
    await get_store().record_activity(event)
    return True


@router.post(
    "/documents/bulk-delete",
    responses={
        400: {"description": "Invalid bulk delete payload"},
        413: {"description": "Bulk delete payload too large"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def bulk_delete_documents(body: dict[str, Any]) -> dict[str, Any]:
    from .. import webui_router as legacy

    doc_ids, actor = _parse_bulk_delete_body(body)
    rag = legacy._get_rag()
    deleted = 0
    failed: list[str] = []
    for doc_id in doc_ids:
        if await _delete_one_document(legacy, rag, doc_id, actor):
            deleted += 1
        else:
            failed.append(str(doc_id))

    return {"deleted": deleted, "failed": failed}
