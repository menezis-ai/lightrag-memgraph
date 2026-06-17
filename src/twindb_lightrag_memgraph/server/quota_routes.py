"""``GET /twin/api/quota`` — instance storage quota snapshot.

Public endpoint (no auth gate). The banner in the WebUI polls it every
30 s so an anonymous operator browsing the read-only surface still
sees the warning banner.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from . import quota

router = APIRouter(prefix="/quota", tags=["quota"])


class QuotaSnapshot(BaseModel):
    used_bytes: int | None = None
    limit_bytes: int | None = None
    used_pct: float | None = None
    status: str = "ok"
    warn_threshold: float = quota.WARN_THRESHOLD
    configured: bool = False


@router.get("", response_model=QuotaSnapshot)
async def get_quota_snapshot() -> dict[str, Any]:
    return await quota.snapshot()


__all__ = ["router"]
