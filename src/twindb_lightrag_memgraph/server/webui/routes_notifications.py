"""Notification endpoints for the Twin WebUI."""

from __future__ import annotations

from fastapi import APIRouter

from ..webui_models import AckResponse, Notification
from .store import get_store

router = APIRouter()


@router.get("/notifications", response_model=list[Notification])
async def list_notifications() -> list[dict[str, object]]:
    return await get_store().list_notifications()


@router.post("/notifications/read-all", response_model=AckResponse)
async def mark_all_notifications_read() -> dict[str, bool]:
    await get_store().mark_all_notifications_read()
    return {"ok": True}


@router.delete("/notifications", response_model=AckResponse)
async def clear_notifications() -> dict[str, bool]:
    await get_store().clear_notifications()
    return {"ok": True}
