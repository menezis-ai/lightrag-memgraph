"""Activity endpoints for the Twin WebUI."""

from __future__ import annotations

from typing import Annotated, Any
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..idp_jwt import require_admin_user
from ..webui_models import AckResponse, ActivityEnvelope
from .events import _make_event, _request_actor
from .store import get_store

router = APIRouter(tags=["activity"])


@router.post(
    "/documents/uploads/activity",
    response_model=AckResponse,
    summary="Record an upload in the audit feed (admin only)",
    # Audit 2026-08-06, R-03a: client-declared audit writes are admin-only.
    # The authoritative `source-uploaded` event is emitted server-side by the
    # ingestion pipeline (registry ``_patch_upload_activity_emission``) with
    # ``emitted_by: server`` — this route remains for admin tooling/backfill
    # and is stamped ``emitted_by: client`` so a forensics query can tell the
    # two apart. Non-admin WebUI uploads rely on the server-side event; the
    # frontend swallows this call's 403 (Promise.allSettled).
    dependencies=[Depends(require_admin_user)],
    responses={400: {"description": "Missing upload source"}},
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["source"],
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Uploaded file name.",
                            },
                            "track_id": {
                                "type": "string",
                                "description": (
                                    "Ingestion tracking id returned by the "
                                    "upload endpoint."
                                ),
                            },
                            "status": {
                                "type": "string",
                                "description": 'Upload outcome (default "accepted").',
                            },
                        },
                    },
                    "example": {
                        "source": "employee-handbook.pdf",
                        "track_id": "upload-20260728-0001",
                        "status": "accepted",
                    },
                }
            },
        }
    },
)
async def record_source_uploaded(
    body: dict[str, Any],
    request: Request,
) -> dict[str, bool]:
    """Record a durable `source-uploaded` audit event (admin tooling only).

    Normal uploads are traced server-side by the ingestion pipeline; this
    route only remains for admin backfill/tooling and stamps the event
    `emitted_by: client` so it stays distinguishable from the probative
    server-side records."""
    source = str(body.get("source") or "").strip()
    if not source:
        raise HTTPException(
            status_code=400,
            detail="record_source_uploaded requires a non-empty source.",
        )
    actor = _request_actor(request)
    track_id = str(body.get("track_id") or "").strip()
    status = str(body.get("status") or "accepted").strip() or "accepted"
    event = _make_event(
        kind="source-uploaded",
        sev="info",
        actor=actor,
        target_label=source,
        summary=f"uploaded by {actor}",
        meta={
            "source": source,
            "track_id": track_id,
            "status": status,
            "emitted_by": "client",
        },
        target_type="source",
    )
    await get_store().record_activity(event)
    return {"ok": True}


@router.get(
    "/activity",
    response_model=ActivityEnvelope,
    summary="List audit-feed events",
)
async def list_activity(
    kind: Annotated[
        str | None,
        Query(
            description=(
                "Only events of this kind (e.g. `source-uploaded`, "
                "`doc-approved`, `tag-created`)."
            ),
            examples=["doc-approved"],
        ),
    ] = None,
    sev: Annotated[
        str | None,
        Query(
            description="Only events with this severity (`info`, `warning`, `error`).",
            examples=["warning"],
        ),
    ] = None,
    actor: Annotated[
        str | None,
        Query(description="Only events performed by this actor."),
    ] = None,
    q: Annotated[
        str | None,
        Query(description="Case-insensitive substring match on the event summary."),
    ] = None,
    activity_range: Annotated[
        Literal["24h", "7d", "30d", "all"] | None,
        Query(
            alias="range",
            description="Time window to search (default: all).",
        ),
    ] = None,
    resource_id: Annotated[
        str | None,
        Query(
            alias="resource.id",
            description="Only events targeting this resource (e.g. a document id).",
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of events returned."),
    ] = 200,
) -> dict[str, Any]:
    """Return the audit feed of the active folder, most recent first.
    Filters combine with AND. `total` counts the matches before `limit`
    is applied."""
    items, total, now_ms = await get_store().list_activity(
        kind=kind,
        sev=sev,
        actor=actor,
        q=q,
        range=activity_range,
        resource_id=resource_id,
        limit=limit,
    )
    return {"items": items, "total": total, "nowMs": now_ms}
