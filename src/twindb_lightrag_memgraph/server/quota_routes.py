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
    # Headline = the binding wall (license or ram).
    used_bytes: int | None = None
    limit_bytes: int | None = None
    used_pct: float | None = None
    status: str = "ok"
    warn_threshold: float = quota.WARN_THRESHOLD
    configured: bool = False
    budget_enforce: str = "reject"
    binding: str | None = None  # "license" | "ram"
    # Data footprint — the billed components.
    graph_bytes: int | None = None
    vector_bytes: int | None = None
    vectors_billed: bool = True
    # License / billed wall — what Memgraph charges for.
    billed_bytes: int | None = None
    license_limit_bytes: int | None = None
    billed_pct: float | None = None
    # RAM wall — Memgraph's --memory-limit.
    ram_used_bytes: int | None = None
    ram_limit_bytes: int | None = None
    ram_pct: float | None = None
    ram_basis: str | None = None  # "tracked" | "rss"


@router.get("", response_model=QuotaSnapshot)
async def get_quota_snapshot() -> dict[str, Any]:
    return await quota.snapshot()


__all__ = ["router"]
