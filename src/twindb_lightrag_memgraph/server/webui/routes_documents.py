"""Document endpoints for the Twin WebUI.

This module intentionally calls the legacy helper symbols through
``server.webui_router`` at runtime. Several tests and adjacent modules still
patch those helpers there; keeping that lookup dynamic preserves compatibility
while the large router is split incrementally.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..idp_jwt import require_admin_user
from ..webui_models import Document, ListEnvelope
from .events import _make_event, _request_actor
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
    graph_tags = await legacy._graph_tags_for_doc_or_none(doc_id)
    tags_available = graph_tags is not None
    tags = graph_tags if tags_available else []
    folder = doc.get("folder") or metadata.get("folder") or legacy.current_folder_id()
    return {
        "tags": tags,
        "tags_source": "tagged_with",
        "tags_status": "ok" if tags_available else "unavailable",
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
    # Admin-gated (architect review P2): the full membership list is a
    # cloisonnement surface. Active-folder visibility alone is not enough — a
    # caller scoped to one folder must not learn the doc's OTHER folders. Until
    # per-user scope filtering is wired through MyAccess/SSO (which owns RBAC),
    # gate it like the mutations rather than leak cross-folder membership.
    dependencies=[Depends(require_admin_user)],
    responses={
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document not found / not visible in active folder"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def list_document_folders(doc_id: str) -> dict[str, Any]:
    from .. import webui_router as legacy

    # Even gated, keep the active-folder visibility check so a missing/unrelated
    # id 404s rather than confirming existence.
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
async def add_document_to_folder(
    doc_id: str, body: dict[str, Any], request: Request
) -> dict[str, Any]:
    from .. import webui_router as legacy

    folder_id = _require_known_folder(str(body.get("folder_id") or ""))
    actor = _request_actor(request)

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
        target_id=doc_id,
    )
    await get_store().record_activity(event)
    return {"doc_id": doc_id, "folders": folders}


@router.delete(
    "/documents/{doc_id}/folders/{folder_id}",
    # Admin-gated interim authorization — see add_document_to_folder.
    dependencies=[Depends(require_admin_user)],
    responses={
        400: {"description": "Invalid folder_id"},
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document or folder not found"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def remove_document_from_folder(
    doc_id: str,
    folder_id: str,
    request: Request,
    actor: Annotated[str | None, Query()] = None,
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
    audit_actor = _request_actor(request)
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
            remaining_folders: list[str] = []
        else:
            await rag.doc_status.remove_from_folder(doc_id, folder_id)
            remaining_folders = [folder for folder in folders if folder != folder_id]

    event = _make_event(
        kind="doc-deleted" if physically_deleted else "doc-folder-removed",
        sev="info",
        actor=audit_actor,
        target_label=doc_id,
        summary=(
            f"removed from folder {folder_id} by {audit_actor}"
            + (" (last folder → physically deleted)" if physically_deleted else "")
        ),
        meta={
            "doc_id": doc_id,
            "folder_id": folder_id,
            "operation": "remove-membership",
            "physically_deleted": physically_deleted,
            "remaining_folders": remaining_folders,
        },
        target_type="document",
        target_id=doc_id,
    )
    await get_store().record_activity(event)
    return {
        "ok": True,
        "doc_id": doc_id,
        "removed_folder": folder_id,
        "remaining_folders": remaining_folders,
        "physically_deleted": physically_deleted,
    }


def _parse_bulk_delete_body(body: dict[str, Any]) -> list[Any]:
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
    return doc_ids


async def _apply_membership_delete(
    legacy: Any, rag: Any, doc_id: str, active: str
) -> bool | None:
    """Un-share ``doc_id`` from ``active``; physically delete on last membership.

    Returns ``True`` when the node was physically deleted, ``False`` when it was
    only removed from the active folder, or ``None`` when the delete must be
    skipped (doc not in the active folder, or any backend error). The physical
    delete runs while the membership edge still exists (ordering guard)."""
    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    try:
        if get_folders is None:
            await legacy._delete_doc_from_rag(rag, doc_id)
            return True
        async with _membership_lock(doc_id):
            folders = await get_folders(doc_id)
            if folders is None or active not in folders:
                return None
            if folders == [active]:
                await legacy._delete_doc_from_rag(rag, doc_id)
                return True
            await rag.doc_status.remove_from_folder(doc_id, active)
            return False
    except Exception:
        return None


async def _delete_one_document(
    legacy: Any, rag: Any, doc_id: Any
) -> dict[str, Any] | None:
    """Delete a doc from the bulk-delete surface, ref-counted (architect P1).

    "Delete" here means **remove from the active folder**: a doc shared across
    folders is only un-shared from the current one, and its physical data
    (node/chunks/vectors) is removed only when this was its LAST membership.
    Backends without membership fall back to the legacy hard delete.
    """
    if not isinstance(doc_id, str) or not doc_id:
        return None
    active = legacy.current_folder_id()
    try:
        doc = await legacy._get_doc_for_active_folder(doc_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        return None
    except Exception:
        return None

    physically_deleted = await _apply_membership_delete(legacy, rag, doc_id, active)
    if physically_deleted is None:
        return None

    return {
        "doc_id": doc_id,
        "label": doc.get("file_path") or doc_id,
        "folder": active,
        "physically_deleted": physically_deleted,
    }


async def _emit_bulk_delete_activity(
    *,
    actor: str,
    results: list[dict[str, Any]],
    failed: list[str],
) -> None:
    """Emit one audit event for a bulk-delete request that changed documents."""
    if not results:
        return

    doc_ids = [str(result["doc_id"]) for result in results]
    deleted = len(results)
    active = str(results[0]["folder"])
    physically_deleted_count = sum(1 for r in results if r["physically_deleted"])
    unshared_count = deleted - physically_deleted_count
    cascade_summary = (
        "physical delete cascades document data, chunks, vectors and graph links"
    )
    if physically_deleted_count and unshared_count:
        summary = (
            f"Bulk delete by {actor}: {deleted} documents affected "
            f"({physically_deleted_count} physically deleted with cascade, "
            f"{unshared_count} unshared from folder {active})"
        )
    elif physically_deleted_count:
        summary = (
            f"Bulk delete by {actor}: {deleted} documents physically deleted; "
            f"{cascade_summary}"
        )
    else:
        summary = (
            f"Bulk delete by {actor}: {deleted} documents unshared "
            f"from folder {active}; no physical cascade"
        )

    meta: dict[str, Any] = {
        "operation": "bulk-delete",
        "folder": active,
        "doc_count": deleted,
        "doc_ids": doc_ids,
        "failed": failed,
        "failed_count": len(failed),
        "physically_deleted_count": physically_deleted_count,
        "unshared_count": unshared_count,
        "cascade": cascade_summary,
    }
    target_id = None
    target_label = f"{deleted} documents"
    target_type = "bulk"
    if deleted == 1:
        target_id = doc_ids[0]
        target_label = str(results[0]["label"])
        target_type = "document"
        meta["doc_id"] = doc_ids[0]

    event = _make_event(
        kind="doc-deleted" if physically_deleted_count else "doc-folder-removed",
        sev="info",
        actor=actor,
        target_label=target_label,
        summary=summary,
        meta=meta,
        target_type=target_type,
        target_id=target_id,
    )
    await get_store().record_activity(event)


@router.post(
    "/documents/bulk-delete",
    responses={
        400: {"description": "Invalid bulk delete payload"},
        413: {"description": "Bulk delete payload too large"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def bulk_delete_documents(body: dict[str, Any], request: Request) -> dict[str, Any]:
    from .. import webui_router as legacy

    doc_ids = _parse_bulk_delete_body(body)
    actor = _request_actor(request)
    rag = legacy._get_rag()
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    for doc_id in doc_ids:
        result = await _delete_one_document(legacy, rag, doc_id)
        if result is not None:
            results.append(result)
        else:
            failed.append(str(doc_id))

    await _emit_bulk_delete_activity(actor=actor, results=results, failed=failed)
    return {"deleted": len(results), "failed": failed}
