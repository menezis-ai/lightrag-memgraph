"""Process-local operational counters shared by both Twin server surfaces.

The standalone factory and the production ``register(mount_server=True)``
overlay share this module. The stable JSON snapshot deliberately uses only the
standard library so mounting the server cannot introduce an optional metrics
dependency into the BNP runtime image.
"""

from __future__ import annotations

import threading
from typing import Final

_LOCK = threading.RLock()
_ROUTE_GROUPS: Final = frozenset(
    {
        "admin",
        "documents",
        "graph",
        "health",
        "ingestion",
        "metrics",
        "other",
        "query",
        "twin",
    }
)
_LEGACY_COUNTER_NAMES: Final = (
    "auth_rejects_total",
    "body_limit_rejects_total",
    "ingestion_failures_total",
    "query_failures_total",
    "quota_rejects_total",
)


def _empty_snapshot() -> dict[str, int]:
    return {
        **{name: 0 for name in _LEGACY_COUNTER_NAMES},
        "requests_total": 0,
        "storage_writes_total": 0,
        "storage_errors_total": 0,
        "audit_invalid_total": 0,
        "audit_dropped_total": 0,
        "audit_queue_depth": 0,
    }


_SNAPSHOT = _empty_snapshot()


def _install_storage_callback() -> None:
    """Connect the dependency-light storage pool to the server counters."""
    try:
        from .. import _pool

        _pool.set_storage_metric_recorder(record_storage_write)
    except (AttributeError, ImportError):
        # Older public storage slices can still import the server package. The
        # callback is additive and therefore degrades to missing storage counts.
        return


def _bounded_route_group(value: str) -> str:
    return value if value in _ROUTE_GROUPS else "other"


def increment_metric(name: str, amount: int = 1) -> None:
    """Increment one stable operational counter.

    Unknown names are refused so the JSON response remains a closed contract.
    """
    if amount < 1:
        return
    if name not in _LEGACY_COUNTER_NAMES:
        raise ValueError(f"Unknown operational metric: {name}")
    with _LOCK:
        _SNAPSHOT[name] += amount


def record_http_request(
    *, route_group: str, method: str, status_code: int, duration_seconds: float
) -> None:
    """Record one completed request in the stable aggregate snapshot.

    ``method`` and ``duration_seconds`` remain in the shared callback contract
    used by both server surfaces. The JSON snapshot intentionally does not
    expose a route/method matrix or latency histogram.
    """
    del method, duration_seconds
    group = _bounded_route_group(route_group)
    with _LOCK:
        _SNAPSHOT["requests_total"] += 1

    if status_code in {401, 403}:
        increment_metric("auth_rejects_total")
    if status_code == 507:
        increment_metric("quota_rejects_total")
    if group == "query" and status_code >= 500:
        increment_metric("query_failures_total")
    if group == "ingestion" and status_code >= 500:
        increment_metric("ingestion_failures_total")


def record_storage_write(outcome: str) -> None:
    """Record one write-slot operation from :mod:`._pool`."""
    is_error = outcome != "success"
    with _LOCK:
        _SNAPSHOT["storage_writes_total"] += 1
        if is_error:
            _SNAPSHOT["storage_errors_total"] += 1


def record_audit_event(outcome: str) -> None:
    """Hook for #89/#122 audit validation and queue-drop paths."""
    if outcome not in {"invalid", "dropped"}:
        raise ValueError("Audit metric outcome must be 'invalid' or 'dropped'")
    with _LOCK:
        _SNAPSHOT[f"audit_{outcome}_total"] += 1


def set_audit_queue_depth(depth: int) -> None:
    """Set the #122 queue gauge without allowing a negative depth."""
    bounded = max(0, int(depth))
    with _LOCK:
        _SNAPSHOT["audit_queue_depth"] = bounded


def metrics_snapshot() -> dict[str, int]:
    """Return the stable process-local JSON snapshot."""
    with _LOCK:
        return dict(_SNAPSHOT)


def reset_metrics() -> None:
    """Reset process-local counters for deterministic tests."""
    global _SNAPSHOT
    with _LOCK:
        _SNAPSHOT = _empty_snapshot()
        _install_storage_callback()


_install_storage_callback()

__all__ = [
    "increment_metric",
    "metrics_snapshot",
    "record_audit_event",
    "record_http_request",
    "record_storage_write",
    "reset_metrics",
    "set_audit_queue_depth",
]
