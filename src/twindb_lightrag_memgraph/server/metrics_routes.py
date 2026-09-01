"""Authenticated operational metrics shared by standalone and overlay apps."""

from __future__ import annotations

from fastapi import APIRouter, Response

from .metrics import metrics_snapshot, prometheus_content_type, render_prometheus


def build_metrics_router() -> APIRouter:
    router = APIRouter(tags=["system"])

    @router.get("/ops/metrics", summary="Operational metrics JSON snapshot")
    def operational_metrics() -> dict[str, int]:
        """Return stable aggregate counters retained for existing operators."""
        return metrics_snapshot()

    @router.get(
        "/ops/metrics/prometheus",
        summary="Prometheus operational metrics exposition",
        response_class=Response,
    )
    def prometheus_metrics() -> Response:
        """Return bounded runtime metrics in Prometheus text format."""
        return Response(
            content=render_prometheus(),
            media_type=prometheus_content_type(),
            headers={"Cache-Control": "no-store"},
        )

    return router


__all__ = ["build_metrics_router"]
