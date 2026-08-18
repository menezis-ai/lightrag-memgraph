"""Notification endpoints for the Twin WebUI."""

from __future__ import annotations

from fastapi import APIRouter

from ..webui_models import AckResponse, Notification
from .store import get_store

router = APIRouter(tags=["notifications"])


@router.get(
    "/notifications",
    response_model=list[Notification],
    summary="List notifications",
)
async def list_notifications() -> list[dict[str, object]]:
    """Return the operator notifications of the active folder, most
    recent first, each with its read/unread state."""
    return await get_store().list_notifications()


@router.post(
    "/notifications/read-all",
    response_model=AckResponse,
    summary="Mark every notification as read",
)
async def mark_all_notifications_read() -> dict[str, bool]:
    """Mark all notifications of the active folder as read. The entries
    stay listed; only their unread flag changes."""
    await get_store().mark_all_notifications_read()
    return {"ok": True}


@router.delete(
    "/notifications",
    response_model=AckResponse,
    summary="Clear all notifications",
)
async def clear_notifications() -> dict[str, bool]:
    """Delete every notification of the active folder. The audit trail in
    `GET /activity` is not affected."""
    await get_store().clear_notifications()
    return {"ok": True}
