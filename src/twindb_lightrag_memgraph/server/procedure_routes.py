"""Approval-workflow routes for the procedure ingestion profile (PR 2).

The PR 1 seam parks procedure documents as bundles (``_procedure_store``)
instead of enqueueing them; this router is the human gate that releases (or
refuses) them — the embryo of the Vihn/Fabrice document-validation workflow:

- ``GET  /procedures``                 — folder-bound summaries (auth).
- ``GET  /procedures/{id}``            — full bundle, PNGs included (admin).
- ``POST /procedures/{id}/approve``    — compose markdown (full text +
  informed schematic descriptions) and enqueue it under the ORIGINAL file
  name with the upload-time contexts rebound: primary folder, then a
  ``doc_status.add_to_folder`` membership per duplicate-request folder, and
  the STRICTEST operator classification so the MIP enqueue gate reproduces
  the upload policy exactly (admin).
- ``POST /procedures/{id}/reject``     — terminal until retry (admin).
- ``POST /procedures/{id}/retry``      — the ONLY relaunch path (admin).
- ``POST /procedures/{id}/reroute-standard`` — detection false positive:
  push the ORIGINAL file through the standard pipeline (admin).
- ``GET  /procedures/store/health`` / ``POST /procedures/store/recover`` —
  degraded-store inspection and the explicit quarantine recovery (admin).

Mounted in BOTH server surfaces — ``server/app.py:create_app`` AND the
overlay ``_mount_twin_subapp`` (lesson re-paid in #381; guard test in
``tests/test_server/test_overlay_procedures.py``). The rag instance is
injected via a ``get_rag`` factory argument, same family as
``twin_query_routes``/``native_shims``.

State machine notes: transitions go through the store's atomic
``transition_bundle`` (optimistic lock — two admins racing surface a 409,
never a double action). A degraded store raises ``StoreDegradedError``
everywhere → 503 with the recovery hint, list included: an honest outage
beats an empty list that hides parked documents.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path as FastapiPath,
    Query,
    Request,
)
from pydantic import BaseModel, Field

from .. import _procedure, _procedure_store
from .._constants import (
    doc_type_context,
    get_active_storage_folder,
    operator_classification_context,
    storage_folder_context,
)
from .auth import require_auth
from .folder import bind_request_folder
from .idp_jwt import require_admin_user

logger = logging.getLogger("twindb_lightrag_memgraph")

_UNKNOWN_BUNDLE = "unknown bundle"


class BundleSummary(BaseModel):
    """Folder-bound list projection: NO paths, NO PNGs, NO full text —
    a bundle visible through a folder must not leak another folder's
    request context (review finding, PR #384)."""

    id: str
    file_name: str
    state: str
    reason: str
    source: str
    # Opaque ingestion track id: lets the WebUI reconcile a parked upload
    # (the upload response only carries a track id; a parked document never
    # lands in /documents, so the optimistic row must resolve against the
    # procedure list instead).
    track_id: str | None = None
    schematics_total: int = 0
    schematics_described: int = 0
    classification: dict[str, Any] | None = None
    operator_classification: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ApproveRequest(BaseModel):
    folder: str | None = Field(
        default=None,
        description=(
            "Target folder override. Required when the bundle is folderless "
            "(scan-created) and carries no operator duplicate request."
        ),
    )


class RejectRequest(BaseModel):
    comment: str | None = None


def _degraded_503() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "procedure store degraded: the bundle claim index was "
            "quarantined — inspect the .corrupt-* files then POST "
            "/twin/api/procedures/store/recover"
        ),
    )


async def _store_call(func, *args, **kwargs):
    """Run a sync store call in a worker thread; degraded -> 503."""
    try:
        return await asyncio.to_thread(lambda: func(*args, **kwargs))
    except _procedure_store.StoreDegradedError as exc:
        raise _degraded_503() from exc


def _summary(bundle: dict) -> BundleSummary:
    schematics = [s for s in bundle.get("schematics") or [] if isinstance(s, dict)]
    return BundleSummary(
        id=str(bundle.get("id")),
        file_name=str(bundle.get("file_name") or ""),
        state=str(bundle.get("state") or ""),
        reason=str(bundle.get("reason") or ""),
        source=str(bundle.get("source") or ""),
        track_id=bundle.get("track_id"),
        schematics_total=int(bundle.get("schematics_total") or 0),
        schematics_described=sum(
            1 for s in schematics if isinstance(s.get("informed"), dict)
        ),
        classification=bundle.get("classification"),
        operator_classification=_procedure.strictest_operator_classification(bundle),
        created_at=bundle.get("created_at"),
        updated_at=bundle.get("updated_at"),
    )


def _visible_in_folder(bundle: dict, folder: str) -> bool:
    """Folder-bound visibility, with one deliberate exception.

    A bundle claimed by NO folder (scan-created, no operator request yet)
    would otherwise be reachable from no list at all — unreviewable in
    production. Unassigned bundles carry no cross-folder context (that is
    the whole point of ``folder=None`` for scans), so they surface in every
    folder's list until an operator claims them via approve/reroute with an
    explicit target folder.
    """
    folders = _procedure.bundle_folders(bundle)
    return not folders or folder in folders


def _actor(request: Request) -> str:
    from .webui.events import _request_actor

    return _request_actor(request)


async def _emit(
    *,
    kind: str,
    actor: str,
    bundle: dict | None,
    summary: str,
    sev: str = "info",
    meta: dict[str, Any] | None = None,
    notify: dict[str, Any] | None = None,
) -> None:
    """Best-effort activity (+ optional notification). Never raises."""
    try:
        from . import webui_router

        event = webui_router._make_event(
            kind=kind,
            sev=sev,
            actor=actor,
            target_label=str((bundle or {}).get("file_name") or "procedure-store"),
            summary=summary,
            meta={
                **({"bundle_id": bundle["id"]} if bundle else {}),
                **(meta or {}),
            },
            target_type="document",
            target_id=(bundle or {}).get("id"),
        )
        store = webui_router.get_store()
        await store.record_activity(event)
        if notify is not None:
            from .webui.events import _make_notification

            await store.push_notification(_make_notification(**notify))
    except Exception:  # noqa: BLE001 — audit must never break the request
        logger.exception("[ProcedureRoutes] activity emission failed (kind=%s)", kind)


def _resolve_primary_folder(bundle: dict, override: str | None) -> str | None:
    folders = _procedure.bundle_folders(bundle)
    return override or (folders[0] if folders else None)


async def _enqueue_approved_document(
    rag: Any, bundle: dict, markdown: str, primary: str, strictest: str | None
) -> None:
    with (
        storage_folder_context(primary),
        operator_classification_context(strictest),
    ):
        await rag.apipeline_enqueue_documents(
            markdown,
            file_paths=bundle.get("file_name"),
            track_id=bundle.get("track_id") or None,
        )


async def _apply_extra_memberships(
    rag: Any, bundle: dict, doc_id: str | None, primary: str
) -> list[str]:
    extra_folders = [f for f in _procedure.bundle_folders(bundle) if f != primary]
    if not extra_folders or doc_id is None:
        return extra_folders
    add_to_folder = getattr(getattr(rag, "doc_status", None), "add_to_folder", None)
    if not callable(add_to_folder):
        return extra_folders
    failures: list[str] = []
    for extra in extra_folders:
        try:
            added = await add_to_folder(doc_id, extra)
        except Exception:
            added = False
        if not added:
            failures.append(extra)
    return failures


async def _record_mip_refusal(bundle_id: str, bundle: dict, request: Request) -> dict:
    refused = await _store_call(
        _procedure_store.transition_bundle,
        bundle_id,
        ("pending",),
        state="failed",
        reason="mip-rejected-at-enqueue: the classification gate refused "
        "the document (see the FAILED document row)",
    )
    await _emit(
        kind="procedure-failed",
        actor=_actor(request),
        bundle=bundle,
        sev="warning",
        summary=f"Procedure '{bundle.get('file_name')}' refused by the "
        "classification gate at approve time",
    )
    return refused or bundle


class _ProcedureRouteHandlers:
    def __init__(self, get_rag: Callable[[], Any]) -> None:
        self.get_rag = get_rag

    async def store_health(self) -> dict[str, Any]:
        return {
            "degraded": await asyncio.to_thread(_procedure_store.is_degraded),
            "quarantine_files": await asyncio.to_thread(
                _procedure_store.quarantine_files
            ),
        }

    async def store_recover(self, request: Request) -> dict[str, Any]:
        removed = await asyncio.to_thread(_procedure_store.recover_store)
        await _emit(
            kind="procedure-store-recovered",
            actor=_actor(request),
            bundle=None,
            sev="warning",
            summary=f"Procedure store recovered — {len(removed)} quarantine file(s) removed",
            meta={"removed": removed},
        )
        return {"removed": removed, "degraded": _procedure_store.is_degraded()}

    async def list_procedures(
        self,
        folder: Annotated[str, Depends(bind_request_folder)],
        state: Annotated[
            str | None,
            Query(
                description=(
                    "Return only bundles in this workflow state. Omit to "
                    "return every state visible in the active folder."
                ),
                examples=["pending"],
            ),
        ] = None,
    ) -> list[BundleSummary]:
        bundles = await _store_call(_procedure_store.list_bundles, state=state)
        return [_summary(b) for b in bundles if _visible_in_folder(b, folder)]

    async def get_procedure(
        self,
        bundle_id: Annotated[
            str,
            FastapiPath(
                description="Opaque procedure bundle identifier from the list route.",
                examples=["procedure-7c91e2"],
            ),
        ],
    ) -> dict[str, Any]:
        bundle = await _store_call(_procedure_store.get_bundle, bundle_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail=_UNKNOWN_BUNDLE)
        return bundle

    async def approve_procedure(
        self,
        bundle_id: Annotated[
            str,
            FastapiPath(
                description="Identifier of the pending procedure bundle to approve.",
                examples=["procedure-7c91e2"],
            ),
        ],
        request: Request,
        body: ApproveRequest | None = None,
    ) -> dict[str, Any]:
        bundle = await _store_call(_procedure_store.get_bundle, bundle_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail=_UNKNOWN_BUNDLE)
        if bundle.get("state") != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"bundle is {bundle.get('state')}; only pending bundles can be approved",
            )
        markdown = _procedure.compose_approved_markdown(bundle)
        if not markdown.strip():
            raise HTTPException(
                status_code=409, detail="bundle has no content to enqueue"
            )
        primary = _resolve_primary_folder(bundle, body.folder if body else None)
        if primary is None:
            raise HTTPException(
                status_code=422,
                detail="bundle has no requesting folder (scan-created): pass "
                "'folder' in the request body to choose the target folder",
            )
        rag = self.get_rag()
        strictest = _procedure.strictest_operator_classification(bundle)
        try:
            await _enqueue_approved_document(rag, bundle, markdown, primary, strictest)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"enqueue failed: {exc}"
            ) from exc

        doc_id = _approved_doc_id(markdown, bundle)
        if doc_id is not None and await _doc_was_rejected(rag, doc_id):
            return await _record_mip_refusal(bundle_id, bundle, request)
        failures = await _apply_extra_memberships(rag, bundle, doc_id, primary)
        if failures:
            logger.error(
                "twindb procedure: approve of %s — could not apply membership "
                "for folder(s) %s (use the document 'Add to folder' action)",
                bundle_id,
                ", ".join(failures),
            )
        updated = await _store_call(
            _procedure_store.transition_bundle,
            bundle_id,
            ("pending",),
            state="approved",
            reason="approved",
            approved_doc_id=doc_id,
            approved_folder=primary,
            membership_failures=failures,
        )
        if updated is None:
            logger.error(
                "twindb procedure: approve of %s enqueued but the bundle state changed concurrently",
                bundle_id,
            )
            raise HTTPException(
                status_code=409,
                detail="document enqueued but the bundle state changed concurrently — inspect the bundle",
            )
        await _emit(
            kind="procedure-approved",
            actor=_actor(request),
            bundle=bundle,
            summary=f"Procedure '{bundle.get('file_name')}' approved and enqueued in folder '{primary}'",
            meta={"folder": primary, "doc_id": doc_id, "membership_failures": failures},
        )
        return updated

    async def reject_procedure(
        self,
        bundle_id: Annotated[
            str,
            FastapiPath(
                description="Identifier of the procedure bundle to reject.",
                examples=["procedure-7c91e2"],
            ),
        ],
        request: Request,
        body: RejectRequest | None = None,
    ) -> dict[str, Any]:
        comment = (body.comment if body else None) or ""
        updated = await _store_call(
            _procedure_store.transition_bundle,
            bundle_id,
            ("pending", "failed"),
            state="rejected",
            reason=f"rejected: {comment}" if comment else "rejected",
        )
        if updated is None:
            raise HTTPException(
                status_code=409,
                detail="bundle unknown or not in a rejectable state",
            )
        await _emit(
            kind="procedure-rejected",
            actor=_actor(request),
            bundle=updated,
            sev="warning",
            summary=f"Procedure '{updated.get('file_name')}' rejected",
            meta={"comment": comment} if comment else {},
        )
        return updated

    async def retry_procedure(
        self,
        bundle_id: Annotated[
            str,
            FastapiPath(
                description="Identifier of the failed or rejected bundle to re-process.",
                examples=["procedure-7c91e2"],
            ),
        ],
        request: Request,
    ) -> dict[str, Any]:
        if not _procedure.is_available():
            raise HTTPException(
                status_code=409,
                detail=(
                    "procedure ingestion prerequisites are unavailable; "
                    "check Settings > Vision before retrying"
                ),
            )
        if not await _procedure.is_effectively_enabled():
            raise HTTPException(
                status_code=409,
                detail=(
                    "procedure ingestion is disabled; enable it in "
                    "Settings > Vision before retrying"
                ),
            )
        outcome = await _procedure.aretry_bundle(bundle_id)
        if outcome is None:
            raise HTTPException(
                status_code=409,
                detail="bundle unknown or not retryable (failed/rejected only)",
            )
        if outcome.state == "error":
            raise HTTPException(status_code=502, detail=outcome.reason)
        bundle = await _store_call(_procedure_store.get_bundle, bundle_id)
        await _emit(
            kind="procedure-retried",
            actor=_actor(request),
            bundle=bundle,
            summary=f"Procedure '{(bundle or {}).get('file_name')}' re-processed (now {outcome.state})",
            meta={"state": outcome.state, "reason": outcome.reason},
        )
        return bundle or {"id": bundle_id, "state": outcome.state}

    async def reroute_standard(
        self,
        bundle_id: Annotated[
            str,
            FastapiPath(
                description=(
                    "Identifier of the bundle to send through the standard "
                    "ingestion pipeline."
                ),
                examples=["procedure-7c91e2"],
            ),
        ],
        request: Request,
        body: ApproveRequest | None = None,
    ) -> dict[str, Any]:
        bundle = await _store_call(_procedure_store.get_bundle, bundle_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail=_UNKNOWN_BUNDLE)
        if bundle.get("state") not in ("pending", "failed", "rejected"):
            raise HTTPException(
                status_code=409,
                detail=f"bundle is {bundle.get('state')}; cannot reroute",
            )
        original = Path(str(bundle.get("original_path") or ""))
        if not original.is_file():
            raise HTTPException(
                status_code=409,
                detail="original file left the input directory — re-upload",
            )
        primary = _resolve_primary_folder(bundle, body.folder if body else None)
        if primary is None:
            raise HTTPException(
                status_code=422,
                detail="bundle has no requesting folder: choose a target folder",
            )
        strictest = _procedure.strictest_operator_classification(bundle)
        import lightrag.api.routers.document_routes as dr

        try:
            with (
                doc_type_context("standard"),
                storage_folder_context(primary),
                operator_classification_context(strictest),
            ):
                ok, track_id = await dr.pipeline_enqueue_file(self.get_rag(), original)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"standard enqueue failed: {exc}"
            ) from exc
        if not ok:
            raise HTTPException(
                status_code=502,
                detail=f"standard enqueue refused (track {track_id})",
            )
        updated = await _store_call(
            _procedure_store.transition_bundle,
            bundle_id,
            ("pending", "failed", "rejected"),
            state="rerouted",
            reason=f"rerouted-standard into folder '{primary}'",
            rerouted_track_id=track_id,
        )
        await _emit(
            kind="procedure-rerouted",
            actor=_actor(request),
            bundle=updated or bundle,
            summary=f"Procedure '{bundle.get('file_name')}' rerouted to the standard pipeline (folder '{primary}')",
            meta={"folder": primary, "track_id": track_id},
        )
        return updated or bundle


def _responses(*codes: int) -> dict[int, dict[str, str]]:
    return {
        code: {"description": f"Procedure operation returned HTTP {code}"}
        for code in codes
    }


def build_procedure_router(get_rag: Callable[[], Any]) -> APIRouter:
    """Build the /procedures router bound to a rag-instance factory."""
    router = APIRouter(
        prefix="/procedures",
        tags=["procedures"],
        dependencies=[Depends(require_auth)],
    )
    admin = [Depends(require_admin_user)]
    handlers = _ProcedureRouteHandlers(get_rag)
    # Health routes stay before /{bundle_id}, so "store" is never captured.
    router.add_api_route(
        "/store/health",
        handlers.store_health,
        methods=["GET"],
        dependencies=admin,
        summary="Procedure store health (admin)",
        description=(
            "Report whether the procedure approval store is degraded and "
            "list any quarantined store files. A degraded store refuses "
            "procedure operations until recovered."
        ),
    )
    router.add_api_route(
        "/store/recover",
        handlers.store_recover,
        methods=["POST"],
        dependencies=admin,
        summary="Recover a degraded procedure store (admin)",
        description=(
            "Remove the quarantined store files and return the store to "
            "service. The response lists what was removed."
        ),
    )
    router.add_api_route(
        "",
        handlers.list_procedures,
        methods=["GET"],
        summary="List procedure bundles awaiting review",
        description=(
            "List the procedure-document bundles visible in the active "
            "folder, with their review state (`processing`, `pending`, "
            "`failed`, `approved`, `rejected`). Filter with the `state` "
            "query parameter. Procedure documents are parked here at "
            "upload time and reach the knowledge base only after an "
            "explicit approval."
        ),
    )
    router.add_api_route(
        "/{bundle_id}",
        handlers.get_procedure,
        methods=["GET"],
        dependencies=admin,
        responses=_responses(404),
        summary="Read a procedure bundle (admin)",
        description=(
            "Return the full bundle: extracted text, per-schematic vision "
            "descriptions, classification, and review history."
        ),
    )
    router.add_api_route(
        "/{bundle_id}/approve",
        handlers.approve_procedure,
        methods=["POST"],
        dependencies=admin,
        responses=_responses(404, 409, 422, 502),
        summary="Approve a bundle and ingest it (admin)",
        description=(
            "Release a `pending` bundle into the knowledge base: its "
            "reviewed markdown is ingested into the target folder "
            "(optionally overridden in the body). 409 when the bundle is "
            "not in a pending state; 502 when the ingestion enqueue fails "
            "(the bundle stays approved-but-unqueued for retry)."
        ),
    )
    router.add_api_route(
        "/{bundle_id}/reject",
        handlers.reject_procedure,
        methods=["POST"],
        dependencies=admin,
        responses=_responses(409),
        summary="Reject a bundle (admin)",
        description=(
            "Refuse a bundle with a reason. A rejected bundle never "
            "reaches the knowledge base and is not re-processed unless "
            "explicitly retried."
        ),
    )
    router.add_api_route(
        "/{bundle_id}/retry",
        handlers.retry_procedure,
        methods=["POST"],
        dependencies=admin,
        responses=_responses(409, 502),
        summary="Re-run the analysis of a failed bundle (admin)",
        description=(
            "Re-process the bundle's source file through the procedure "
            "analysis (text extraction + vision passes) and park the new "
            "result for review. Procedure ingestion must be enabled in "
            "Settings > Vision."
        ),
    )
    router.add_api_route(
        "/{bundle_id}/reroute-standard",
        handlers.reroute_standard,
        methods=["POST"],
        dependencies=admin,
        responses=_responses(404, 409, 422, 502),
        summary="Ingest a bundle through the standard pipeline (admin)",
        description=(
            "Bypass the procedure profile for this document: ingest the "
            "original file through the standard pipeline instead "
            "(useful when a file was wrongly detected as a procedure)."
        ),
    )
    return router


def _approved_doc_id(markdown: str, bundle: dict) -> str | None:
    """The doc id LightRAG derives for the enqueued markdown (guarded)."""
    try:
        from .._classification_hook import _doc_id_for_insert

        return _doc_id_for_insert(
            markdown,
            explicit_id=None,
            file_path=str(bundle.get("file_name") or ""),
        )
    except Exception as exc:
        logger.warning(
            "twindb procedure: cannot derive the approved doc id (%s: %s) — "
            "extra-folder memberships must be applied manually",
            type(exc).__name__,
            exc,
        )
        return None


async def _doc_was_rejected(rag: Any, doc_id: str) -> bool:
    """Best-effort post-enqueue check for a MIP-gate rejection row."""
    try:
        get_by_id = getattr(getattr(rag, "doc_status", None), "get_by_id", None)
        if not callable(get_by_id):
            return False
        row = await get_by_id(doc_id)
        if not isinstance(row, dict):
            return False
        status = str(row.get("status") or "").lower()
        return status == "failed" and "classification" in str(
            row.get("error_msg") or row.get("error") or ""
        )
    except Exception:  # noqa: BLE001 — best-effort probe
        return False


# ---------------------------------------------------------------------------
# Seam event sink: PR 1 parks bundles from the ingestion seam and emits
# through _procedure.set_event_sink — wire it to activity + notifications.
# ---------------------------------------------------------------------------


def install_procedure_event_sink() -> None:
    """Bridge seam events (procedure-parked/-failed) into the overlay stores.

    The sink is called from async ingestion code but must stay sync and
    non-blocking: it schedules the emission on the running loop. Installed
    by BOTH server surfaces (create_app and _mount_twin_subapp) —
    idempotent by construction (set_event_sink replaces).
    """

    def _sink(kind: str, payload: dict) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (sync test context): logs already cover it
        loop.create_task(_emit_seam_event(kind, payload))

    _procedure.set_event_sink(_sink)


async def _emit_seam_event(kind: str, payload: dict) -> None:
    file_name = str(payload.get("file_name") or "?")
    state = str(payload.get("state") or "")
    parked = kind == "procedure-parked"
    await _emit(
        kind=(
            kind
            if kind in ("procedure-parked", "procedure-failed")
            else ("procedure-failed")
        ),
        actor="system",
        bundle={"id": payload.get("bundle_id"), "file_name": file_name},
        sev="info" if parked else "warning",
        summary=(
            f"Procedure '{file_name}' parked for approval"
            if parked
            else f"Procedure '{file_name}' processing failed ({state})"
        ),
        meta={k: v for k, v in payload.items() if k != "bundle_id"},
        notify={
            "title": (
                "Procedure awaiting approval"
                if parked
                else "Procedure processing failed"
            ),
            "tagname": None,
            "suffix": file_name,
            "sub": str(payload.get("reason") or ""),
            "kind": "procedure-review",
        },
    )
