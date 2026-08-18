"""Document endpoints for the Twin WebUI.

This module intentionally calls the legacy helper symbols through
``server.webui_router`` at runtime. Several tests and adjacent modules still
patch those helpers there; keeping that lookup dynamic preserves compatibility
while the large router is split incrementally.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path as FilePath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from .._lightrag_compat import PipelineBusyDeletionError
from ..idp_jwt import require_admin_user
from ..webui_models import Document, ListEnvelope
from .events import _make_event, _request_actor
from .store import get_store

# Operator-facing 423 detail for a deletion the ingestion pipeline refused.
# The WebUI keys its "pipeline busy" copy on this prose (errorMessages.ts
# isPipelineBusyDetail) — keep "ingestion pipeline" + "busy" in any rewording.
_PIPELINE_BUSY_DELETE_DETAIL = (
    "Deletion not started: the ingestion pipeline is busy processing "
    "documents. The selected documents were not deleted; retry once the "
    "current processing finishes."
)


class _BulkDeleteRecoveryRequired(RuntimeError):
    """Carry committed partial work across a recovery-fenced bulk delete."""

    def __init__(
        self,
        detail: str,
        *,
        results: list[dict[str, Any]],
        failed: list[str],
        busy: list[str],
        unattempted: list[str],
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.results = list(results)
        self.failed = list(failed)
        self.busy = list(busy)
        self.unattempted = list(unattempted)


router = APIRouter(tags=["documents"])

_DOC_ID_PATH = Path(
    description="Document id, as returned by `GET /documents`.",
    examples=["doc-a1b2c3d4"],
)
logger = logging.getLogger("twindb_lightrag_memgraph")

# Per-document lock serialising membership add/remove so a concurrent add cannot
# slip in between "is this the last membership?" and the physical delete
# (architect review P1 race).
#
# This lock remains useful for avoiding duplicate work inside one process. The
# authoritative cross-worker guard is the storage-level delete claim acquired
# immediately before a last-membership physical cascade.
_MAX_MEMBERSHIP_LOCKS = 2048
_MEMBERSHIP_LOCK_CLEANUP_EVERY = 1024
_membership_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()
_membership_lock_accesses = 0


def _evict_membership_locks() -> None:
    """Drop stale, unlocked locks when map capacity is exceeded."""
    if len(_membership_locks) <= _MAX_MEMBERSHIP_LOCKS:
        return

    # Snapshot the unlocked, evictable keys first (oldest-first LRU order) so we
    # never mutate the mapping while iterating it. In-flight (locked) locks are
    # kept — they are still serializing membership writes.
    removable = [
        doc_id for doc_id, lock in _membership_locks.items() if not lock.locked()
    ]
    for doc_id in removable:
        if len(_membership_locks) <= _MAX_MEMBERSHIP_LOCKS:
            return
        _membership_locks.pop(doc_id, None)


def _membership_lock(doc_id: str) -> asyncio.Lock:
    # Callers must enter the returned lock immediately, without an intervening
    # await; eviction assumes unlocked cached locks have no pending user.
    global _membership_lock_accesses
    lock = _membership_locks.get(doc_id)
    if lock is None:
        lock = asyncio.Lock()
        _membership_locks[doc_id] = lock
    else:
        # Re-order recently-used locks so cleanup can preferentially evict cold
        # entries when the map exceeds its bounded size.
        _membership_locks.move_to_end(doc_id)

    _membership_lock_accesses += 1
    if _membership_lock_accesses % _MEMBERSHIP_LOCK_CLEANUP_EVERY == 0:
        _evict_membership_locks()
    return lock


async def _delete_with_last_membership_claim(
    delete_doc_from_rag: Callable[[Any, str], Awaitable[None]],
    rag: Any,
    doc_id: str,
    folder: str,
) -> None:
    """Claim and physically delete a doc, releasing on cascade failure."""
    claim_last = getattr(rag.doc_status, "claim_last_membership_delete", None)
    release_claim = getattr(rag.doc_status, "release_delete_claim", None)
    if claim_last is None or release_claim is None:
        # Falling back to the process-local lock here would re-open the
        # multi-worker data-loss race, so membership-aware backends fail closed.
        raise RuntimeError(
            "Membership backend lacks the atomic last-membership delete claim"
        )

    claim = secrets.token_urlsafe(24)
    if not await claim_last(doc_id, folder, claim):
        raise HTTPException(
            status_code=409,
            detail="Membership changed concurrently; retry the operation.",
        )
    try:
        await delete_doc_from_rag(rag, doc_id)
    except BaseException:
        try:
            await release_claim(doc_id, claim)
        except Exception:
            # Preserve the primary failure/cancellation (cancellation raised by
            # the cleanup itself still propagates). A claim deliberately fails
            # closed if Memgraph is unavailable during cleanup.
            logger.exception("Failed to release delete claim for %s", doc_id)
        raise


@router.get(
    "/documents",
    response_model=ListEnvelope[Document],
    summary="List documents in the active folder",
)
async def list_documents(
    status: Annotated[
        str | None,
        Query(
            description=(
                "Only return documents with this ingestion status "
                "(e.g. `processed`, `pending`, `failed`)."
            ),
            examples=["processed"],
        ),
    ] = None,
    q: Annotated[
        str | None,
        Query(
            description="Case-insensitive substring match on the document name.",
            examples=["onboarding"],
        ),
    ] = None,
    tag: Annotated[
        str | None,
        Query(
            description="Only return documents carrying this tag.",
            examples=["policy"],
        ),
    ] = None,
) -> dict[str, Any]:
    """List the documents of the folder selected by `X-Twin-Folder`
    (default folder when the header is omitted), with their status, tags
    and metadata. Filters combine with AND."""
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
    summary="Get a document's tags, folder and metadata",
    responses={
        404: {"description": "Document not found"},
        503: {"description": "Backend unavailable"},
    },
)
async def get_document_metadata(
    doc_id: Annotated[str, _DOC_ID_PATH],
) -> dict[str, Any]:
    """Return the document's tags (with their provenance), its folder,
    review state, sensitivity classification and free-form metadata."""
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
#   - AUTHZ: the mutation routes are gated by ``require_admin_user``. With an
#     active IdP this requires ``admin:folders``; with the IdP dormant only the
#     separately managed infrastructure root key is authoritative. Local JWTs
#     and generated ``twk_`` keys intentionally remain non-admin. Per-user
#     source-doc + target-folder RBAC is owned by MyAccess/SSO, not implemented
#     here.
#   - CONCURRENCY: the local lock is backed by a Memgraph CAS delete claim for
#     the last-membership path (see ``_delete_with_last_membership_claim``).
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


def _validate_upload_file_name(file_name: str) -> str:
    """Validate the literal basename accepted by LightRAG's upload route."""
    if not file_name or not file_name.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    if file_name != file_name.strip():
        raise HTTPException(status_code=400, detail="Invalid filename")
    if any(ord(character) < 32 or character == "\x7f" for character in file_name):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if any(separator in file_name for separator in ("/", "\\", ":")):
        raise HTTPException(status_code=400, detail="Unsafe filename detected")
    if file_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Unsafe filename detected")

    # Defense in depth for platform-specific path parsing. The preflight never
    # writes this path; matching the native route's invariant prevents the two
    # endpoints from resolving different document identities.
    try:
        input_root = FilePath.cwd().resolve()
        resolved = (input_root / file_name).resolve()
        if resolved.parent != input_root or not resolved.is_relative_to(input_root):
            raise HTTPException(status_code=400, detail="Unsafe filename detected")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid filename") from None
    return file_name


@router.post(
    "/documents/resolve-upload",
    summary="Resolve an upload to a new ingestion or folder membership",
    responses={
        400: {"description": "Missing or unsafe file_name"},
        409: {"description": "The canonical source has conflicting primaries"},
        503: {"description": "Backend unavailable"},
    },
)
async def resolve_document_upload(
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Share a known source into the active folder before native upload.

    LightRAG treats the canonical filename as the source identity and refuses
    a second upload with that name globally. The WebUI's folder contract is
    different: selecting that known source while another folder is active
    means "add the existing document here". Resolve that intent before the
    multipart endpoint so no duplicate row, content or ingestion job is made.
    """
    from lightrag.base import SourceAbsent, SourceConflict, SourceUnique
    from lightrag.utils_pipeline import resolve_existing_doc_source

    from .. import webui_router as legacy

    file_name = str(body.get("file_name") or "")
    # Mirror native-upload basename validation rather than letting this JSON
    # preflight accept an identity that the multipart route would reject. A
    # direct import is intentionally avoided: importing LightRAG's API router
    # initializes its CLI/config module as a side effect.
    safe_file_name = _validate_upload_file_name(file_name)
    rag = legacy._get_rag()
    resolution = await resolve_existing_doc_source(rag.doc_status, safe_file_name)
    if isinstance(resolution, SourceAbsent):
        return {"action": "upload"}
    if isinstance(resolution, SourceConflict):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Several documents claim the source '{safe_file_name}'. "
                "Resolve the source conflict before uploading it into a folder."
            ),
        )
    if not isinstance(resolution, SourceUnique):
        raise HTTPException(status_code=503, detail="Unsupported source resolution.")

    folder_id = legacy.current_folder_id()
    doc_id = resolution.doc_id
    async with _membership_lock(doc_id):
        folders = await rag.doc_status.get_folders_for_doc(doc_id)
        if folder_id in (folders or []):
            action = "already_present"
        else:
            added = await rag.doc_status.add_to_folder(doc_id, folder_id)
            if not added:
                # The source was deleted or claimed between resolution and the
                # membership write. Let native upload re-evaluate current state.
                return {"action": "upload"}
            action = "shared"

    track_id = resolution.doc.track_id or ""
    if action == "shared":
        actor = _request_actor(request)
        event = _make_event(
            kind="doc-folder-added",
            sev="info",
            actor=actor,
            target_label=doc_id,
            summary=f"added to folder {folder_id} by upload selection ({actor})",
            meta={
                "doc_id": doc_id,
                "folder_id": folder_id,
                "operation": "resolve-upload-membership",
                "file_name": safe_file_name,
            },
            target_type="document",
            target_id=doc_id,
        )
        await get_store().record_activity(event)

    return {
        "action": action,
        "doc_id": doc_id,
        "track_id": track_id,
        "message": (
            f"'{safe_file_name}' was added to folder '{folder_id}'."
            if action == "shared"
            else f"'{safe_file_name}' is already present in folder '{folder_id}'."
        ),
    }


@router.get(
    "/documents/{doc_id}/folders",
    # Admin-gated (architect review P2): the full membership list is a
    # cloisonnement surface. Active-folder visibility alone is not enough — a
    # caller scoped to one folder must not learn the doc's OTHER folders. Until
    # per-user scope filtering is wired through MyAccess/SSO (which owns RBAC),
    # gate it like the mutations rather than leak cross-folder membership.
    dependencies=[Depends(require_admin_user)],
    summary="List the folders a document belongs to (admin)",
    responses={
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document not found / not visible in active folder"},
        503: {"description": "Backend unavailable"},
    },
)
async def list_document_folders(
    doc_id: Annotated[str, _DOC_ID_PATH],
) -> dict[str, Any]:
    """Return every folder this document is a member of. A document lives
    once in storage and can be shared into several folders; this is the
    full membership list, hence admin-only."""
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
    summary="Share a document into a folder (admin)",
    responses={
        400: {"description": "Missing folder_id"},
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document or folder not found"},
        503: {"description": "Backend unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["folder_id"],
                        "properties": {
                            "folder_id": {
                                "type": "string",
                                "description": (
                                    "Target folder id (must exist in the "
                                    "folder catalog)."
                                ),
                            }
                        },
                    },
                    "example": {"folder_id": "general"},
                }
            },
        }
    },
)
async def add_document_to_folder(
    doc_id: Annotated[str, _DOC_ID_PATH],
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Add the document to another folder without duplicating its data.
    The response returns the updated membership list."""
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
    summary="Remove a document from a folder (admin)",
    responses={
        400: {"description": "Invalid folder_id"},
        401: {"description": "Unauthenticated"},
        403: {"description": "Not an admin"},
        404: {"description": "Document or folder not found"},
        409: {"description": "Membership changed concurrently; retry"},
        423: {
            "description": (
                "Ingestion pipeline busy; the document was not deleted — "
                "retry after processing finishes"
            )
        },
        503: {"description": "Backend unavailable"},
    },
)
async def remove_document_from_folder(
    doc_id: Annotated[str, _DOC_ID_PATH],
    folder_id: Annotated[
        str,
        Path(description="Folder to remove the document from.", examples=["general"]),
    ],
    request: Request,
    actor: Annotated[
        str | None,
        Query(
            description=(
                "Accepted for backward compatibility and ignored: the audit "
                "trail records the authenticated identity."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Remove the document from one folder. When this was its **last**
    membership, the document and all its derived data (chunks, vectors,
    graph links) are physically deleted; otherwise it simply stops being
    visible in that folder. The response says which of the two happened
    (`physically_deleted`)."""
    # Two correctness guards from the architect reviews:
    # - Ordering (P1.2): the physical delete runs WHILE the membership edge
    #   still exists — native DETACH DELETE removes the node and its edges
    #   together, so a failed delete leaves the doc intact and still
    #   MEMBER_OF this folder (recoverable), never orphaned and invisible
    #   to membership reads.
    # - Race (P1): the read-decide-delete runs under a per-doc lock shared
    #   with add_document_to_folder, so a concurrent add cannot turn a
    #   last-folder removal into a physical delete of a doc that just
    #   gained a folder.
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
            try:
                await _delete_with_last_membership_claim(
                    legacy._delete_doc_from_rag, rag, doc_id, folder_id
                )
            except PipelineBusyDeletionError as exc:
                raise HTTPException(
                    status_code=423, detail=_PIPELINE_BUSY_DELETE_DETAIL
                ) from exc
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
    skipped (doc not in the active folder). Backend errors deliberately bubble
    to the route so the API never reports a successful bulk operation while the
    storage layer failed. The physical delete runs while the membership edge
    still exists (ordering guard)."""
    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    if get_folders is None:
        await legacy._delete_doc_from_rag(rag, doc_id)
        return True
    async with _membership_lock(doc_id):
        folders = await get_folders(doc_id)
        if folders is None or active not in folders:
            return None
        if folders == [active]:
            await _delete_with_last_membership_claim(
                legacy._delete_doc_from_rag, rag, doc_id, active
            )
            return True
        await rag.doc_status.remove_from_folder(doc_id, active)
        return False


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
    busy: list[str] | None = None,
    actor_role: str | None = None,
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
        # Purple-team (audit 2026-08-06 §Purple rec 3): the role marker
        # makes the credential class of a destructive action visible in one
        # feed query — post-R-03b every delete is admin, so anything else
        # here is itself a signal.
        "actor_role": actor_role or "unknown",
        "doc_count": deleted,
        "doc_ids": doc_ids,
        "failed": failed,
        "failed_count": len(failed),
        "busy": list(busy or []),
        "busy_count": len(busy or []),
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


# Reintroduce bounded parallelism only after measuring the fixed per-document
# cost on the target deployment.
async def _run_bulk_delete_batch(
    legacy: Any,
    rag: Any,
    doc_ids: list[Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Delete docs sequentially and report any partial side effects honestly.

    Memgraph's membership removal is a read/decide/write transaction. Fanning
    hundreds of those transactions out concurrently caused write conflicts
    under operator load: one item was committed, then the whole
    request surfaced as a 503.
    Serial execution is deliberately conservative and deterministic. Bulk
    delete remains non-atomic, so failures after a success are returned as 207.

    Returns ``(results, failed, busy)``. ``busy`` lists the docs whose
    physical cascade LightRAG refused because its pipeline is held by an
    ingestion job — those docs are untouched and retryable, which is a
    different operator situation from ``failed`` (real error or unknown id).
    A batch where nothing was deleted and everything was busy raises 423 so
    the client can distinguish "wait and retry" from "backend broken".
    """

    results: list[dict[str, Any]] = []
    failed: list[str] = []
    busy: list[str] = []
    first_error: tuple[Any, Exception] | None = None
    for index, doc_id in enumerate(doc_ids):
        try:
            outcome = await _delete_one_document(legacy, rag, doc_id)
        except PipelineBusyDeletionError:
            busy.append(str(doc_id))
            logger.warning(
                "bulk delete deferred for document %s: ingestion pipeline busy",
                doc_id,
            )
            continue
        except Exception as exc:  # one failure must not hide committed siblings
            failed.append(str(doc_id))
            if isinstance(exc, HTTPException) and exc.status_code == 503:
                # A recovery_required fence is not an ordinary per-document
                # failure and cannot be cleared by retrying. Stop the batch,
                # but carry every already-committed mutation to the route so
                # its 503 response and audit event remain honest.
                raise _BulkDeleteRecoveryRequired(
                    str(exc.detail),
                    results=results,
                    failed=failed,
                    busy=busy,
                    unattempted=[str(item) for item in doc_ids[index + 1 :]],
                ) from exc
            if first_error is None:
                first_error = (doc_id, exc)
            logger.exception("bulk delete failed for document %s", doc_id)
            continue
        if outcome is None:
            failed.append(str(doc_id))
            continue
        results.append(outcome)

    if busy and not results and not failed:
        raise HTTPException(status_code=423, detail=_PIPELINE_BUSY_DELETE_DETAIL)

    if first_error is not None and not results:
        doc_id, error = first_error
        if isinstance(error, HTTPException):
            raise error
        raise HTTPException(
            status_code=503,
            detail=f"Bulk delete failed while deleting '{doc_id}': {error}",
        ) from error

    if failed or busy:
        logger.warning(
            "bulk delete completed partially: deleted=%d failed=%d busy=%d",
            len(results),
            len(failed),
            len(busy),
        )
    return results, failed, busy


@router.post(
    "/documents/bulk-delete",
    response_model=None,
    summary="Delete several documents from the active folder",
    # Audit 2026-08-06, R-03b: destructive document mutations are admin-only,
    # aligned with the tag/graph/folder gates. Palier 1 fails closed to the
    # infrastructure root key; palier 2 requires the IdP admin scope.
    dependencies=[Depends(require_admin_user)],
    responses={
        200: {"description": "All targets deleted / unshared"},
        207: {
            "description": (
                "Partial success; `failed` lists ids that errored, `busy` "
                "lists ids deferred because the ingestion pipeline was busy "
                "(untouched, retryable)"
            )
        },
        400: {"description": "Invalid bulk delete payload"},
        409: {"description": "Membership changed concurrently; retry"},
        413: {"description": "More than 500 target documents"},
        423: {
            "description": (
                "Ingestion pipeline busy; nothing was deleted — retry the "
                "same request after the current processing finishes"
            )
        },
        503: {
            "description": (
                "Backend unavailable or workspace recovery required; recovery "
                "responses include already-committed document mutations"
            )
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["doc_ids"],
                        "properties": {
                            "doc_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Ids of the documents to delete " "(1 to 500)."
                                ),
                            }
                        },
                    },
                    "example": {"doc_ids": ["doc-a1b2c3d4", "doc-e5f6a7b8"]},
                }
            },
        }
    },
)
async def bulk_delete_documents(body: dict[str, Any], request: Request) -> Any:
    """Remove up to 500 documents from the folder selected by
    `X-Twin-Folder`. Documents shared into other folders are only
    un-shared; documents whose last folder this was are physically
    deleted with their chunks, vectors and graph links. Not atomic: on
    partial completion the response is 207, with erroring ids in `failed`
    and ids the busy ingestion pipeline deferred in `busy` (those are
    untouched — retry them once processing finishes). When the pipeline
    defers every target the response is 423 and nothing was deleted.

    Deletes are deliberately serial to avoid Memgraph write-conflict storms.
    A very large batch can therefore outlive the reverse-proxy timeout while
    the server continues committing its non-atomic work; clients must refetch
    after an ambiguous gateway failure.
    """
    from .. import webui_router as legacy
    from ..auth import is_infrastructure_root_request

    doc_ids = _parse_bulk_delete_body(body)
    actor = _request_actor(request)
    # Purple-team rec 3 (audit 2026-08-06): stamp the credential class on
    # the destructive event — infra root key vs IdP admin scope.
    actor_role = (
        "infrastructure-root"
        if is_infrastructure_root_request(request)
        else "idp-admin"
    )
    rag = legacy._get_rag()
    try:
        results, failed, busy = await _run_bulk_delete_batch(legacy, rag, doc_ids)
    except _BulkDeleteRecoveryRequired as recovery:
        await _emit_bulk_delete_activity(
            actor=actor,
            results=recovery.results,
            failed=recovery.failed,
            busy=recovery.busy,
            actor_role=actor_role,
        )
        committed_ids = [str(result["doc_id"]) for result in recovery.results]
        if committed_ids:
            sample = ", ".join(committed_ids[:3])
            if len(committed_ids) > 3:
                sample += f", +{len(committed_ids) - 3} more"
            progress = (
                f"{len(committed_ids)} earlier document change"
                f"{' was' if len(committed_ids) == 1 else 's were'} already "
                f"committed ({sample})."
            )
        else:
            progress = "No selected document change was committed."
        return JSONResponse(
            {
                "detail": f"{recovery.detail} {progress}",
                "recovery_required": True,
                "deleted": len(recovery.results),
                "committed_doc_ids": committed_ids,
                "failed": recovery.failed,
                "busy": recovery.busy,
                "unattempted": recovery.unattempted,
            },
            status_code=503,
        )

    await _emit_bulk_delete_activity(
        actor=actor, results=results, failed=failed, busy=busy, actor_role=actor_role
    )
    payload = {"deleted": len(results), "failed": failed, "busy": busy}
    if failed or busy:
        return JSONResponse(payload, status_code=207)
    return payload
