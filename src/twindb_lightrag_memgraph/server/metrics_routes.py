"""Authenticated operational metrics shared by standalone and overlay apps."""

from __future__ import annotations

from fastapi import APIRouter

from .metrics import metrics_snapshot


def build_metrics_router() -> APIRouter:
    router = APIRouter(tags=["system"])

    @router.get("/ops/metrics", summary="Operational metrics JSON snapshot")
    def operational_metrics() -> dict[str, int]:
        """Return stable process-local counters retained for operators."""
        return metrics_snapshot()

    return router


__all__ = ["build_metrics_router"]
