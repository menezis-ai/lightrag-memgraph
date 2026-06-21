"""Activity endpoints for the Twin WebUI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..webui_models import AckResponse, ActivityEnvelope
from .events import _make_event
from .store import get_store

router = APIRouter()


@router.post("/documents/uploads/activity", response_model=AckResponse)
async def record_source_uploaded(
    body: dict[str, Any],
) -> dict[str, bool]:
    """Record Activity for LightRAG-native upload accepts.

    The actual upload endpoint is the native LightRAG
    ``/documents/upload`` route, outside this Twin router. The WebUI
    calls this route only after that native endpoint accepts a file so
    the audit feed still has a durable ``source-uploaded`` event.
    """
    source = str(body.get("source") or "").strip()
    if not source:
        raise HTTPException(
            status_code=400,
            detail="record_source_uploaded requires a non-empty source.",
        )
    actor = str(body.get("actor") or "system").strip() or "system"
    track_id = str(body.get("track_id") or "").strip()
    status = str(body.get("status") or "accepted").strip() or "accepted"
    event = _make_event(
        kind="source-uploaded",
        sev="info",
        actor=actor,
        target_label=source,
        summary=f"uploaded by {actor}",
        meta={"source": source, "track_id": track_id, "status": status},
        target_type="source",
    )
    await get_store().record_activity(event)
    return {"ok": True}


@router.get("/activity", response_model=ActivityEnvelope)
async def list_activity(
    kind: str | None = Query(default=None),
    sev: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    items, now_ms = await get_store().list_activity(
        kind=kind, sev=sev, actor=actor, q=q, limit=limit
    )
    return {"items": items, "total": len(items), "nowMs": now_ms}
