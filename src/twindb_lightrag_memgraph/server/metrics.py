"""Bounded-cardinality runtime metrics for both Twin server surfaces.

The standalone factory and the production ``register(mount_server=True)``
overlay share this module.  Metric objects are process-local by default.  When
``PROMETHEUS_MULTIPROC_DIR`` is configured before Python imports this module,
the official prometheus-client multiprocess collector aggregates worker files
at scrape time.
"""

from __future__ import annotations

import os
import threading
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest, multiprocess
from prometheus_client.exposition import CONTENT_TYPE_LATEST

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
_METHODS: Final = frozenset(
    {"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"}
)
_LEGACY_COUNTER_SPECS: Final = {
    "auth_rejects_total": (
        "twin_auth_rejects_total",
        "Authentication and authorization rejections.",
    ),
    "body_limit_rejects_total": (
        "twin_body_limit_rejects_total",
        "Requests rejected by a Twin body-size ceiling.",
    ),
    "ingestion_failures_total": (
        "twin_ingestion_failures_total",
        "Ingestion requests that returned a server error.",
    ),
    "query_failures_total": (
        "twin_query_failures_total",
        "Query requests that returned a server error.",
    ),
    "quota_rejects_total": (
        "twin_quota_rejects_total",
        "Requests rejected because an instance quota was reached.",
    ),
}


def _new_metric_state() -> None:
    global _REGISTRY, _HTTP_REQUESTS, _HTTP_LATENCY, _LEGACY_COUNTERS
    global _STORAGE_WRITES, _AUDIT_EVENTS, _AUDIT_QUEUE_DEPTH, _SNAPSHOT

    registry = CollectorRegistry(auto_describe=True)
    _REGISTRY = registry
    _HTTP_REQUESTS = Counter(
        "twin_http_requests_total",
        "HTTP requests completed by the Twin runtime.",
        ("route_group", "method", "status_class"),
        registry=registry,
    )
    _HTTP_LATENCY = Histogram(
        "twin_http_request_duration_seconds",
        "HTTP response-header latency grouped by bounded route family.",
        ("route_group", "method"),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        registry=registry,
    )
    _LEGACY_COUNTERS = {
        key: Counter(name, help_text, registry=registry)
        for key, (name, help_text) in _LEGACY_COUNTER_SPECS.items()
    }
    _STORAGE_WRITES = Counter(
        "twin_storage_writes_total",
        "Memgraph write-slot operations by terminal outcome.",
        ("outcome",),
        registry=registry,
    )
    _AUDIT_EVENTS = Counter(
        "twin_audit_events_total",
        "Regulatory audit events rejected or dropped before durable export.",
        ("outcome",),
        registry=registry,
    )
    _AUDIT_QUEUE_DEPTH = Gauge(
        "twin_audit_queue_depth",
        "Current bounded audit-export queue depth (zero until issue #122 is active).",
        registry=registry,
        multiprocess_mode="livesum",
    )
    _SNAPSHOT = {
        **{key: 0 for key in _LEGACY_COUNTER_SPECS},
        "requests_total": 0,
        "storage_writes_total": 0,
        "storage_errors_total": 0,
        "audit_invalid_total": 0,
        "audit_dropped_total": 0,
        "audit_queue_depth": 0,
    }


def _install_storage_callback() -> None:
    """Connect the dependency-light storage pool to this server-only exporter."""
    try:
        from .. import _pool

        _pool.set_storage_metric_recorder(record_storage_write)
    except (AttributeError, ImportError):
        # Older public storage slices can still import the server package.  The
        # callback is additive and therefore degrades to missing storage counts.
        return


def _bounded_route_group(value: str) -> str:
    return value if value in _ROUTE_GROUPS else "other"


def _bounded_method(value: str) -> str:
    upper = value.upper()
    return upper if upper in _METHODS else "OTHER"


def increment_metric(name: str, amount: int = 1) -> None:
    """Increment one legacy operational counter.

    Unknown names are refused so a request-derived value can never create an
    unbounded Prometheus time series.
    """
    if amount < 1:
        return
    try:
        counter = _LEGACY_COUNTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown operational metric: {name}") from exc
    with _LOCK:
        counter.inc(amount)
        _SNAPSHOT[name] += amount


def record_http_request(
    *, route_group: str, method: str, status_code: int, duration_seconds: float
) -> None:
    """Record one completed request with bounded labels and status counters."""
    group = _bounded_route_group(route_group)
    bounded_method = _bounded_method(method)
    status_class = f"{status_code // 100}xx" if 100 <= status_code <= 599 else "other"
    with _LOCK:
        _HTTP_REQUESTS.labels(group, bounded_method, status_class).inc()
        _HTTP_LATENCY.labels(group, bounded_method).observe(max(0.0, duration_seconds))
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
    bounded = "success" if outcome == "success" else "error"
    with _LOCK:
        _STORAGE_WRITES.labels(bounded).inc()
        _SNAPSHOT["storage_writes_total"] += 1
        if bounded == "error":
            _SNAPSHOT["storage_errors_total"] += 1


def record_audit_event(outcome: str) -> None:
    """Hook for #89/#122 audit validation and queue-drop paths."""
    if outcome not in {"invalid", "dropped"}:
        raise ValueError("Audit metric outcome must be 'invalid' or 'dropped'")
    with _LOCK:
        _AUDIT_EVENTS.labels(outcome).inc()
        _SNAPSHOT[f"audit_{outcome}_total"] += 1


def set_audit_queue_depth(depth: int) -> None:
    """Set the #122 queue gauge without allowing a negative depth."""
    bounded = max(0, int(depth))
    with _LOCK:
        _AUDIT_QUEUE_DEPTH.set(bounded)
        _SNAPSHOT["audit_queue_depth"] = bounded


def metrics_snapshot() -> dict[str, int]:
    """Return the stable JSON snapshot retained for existing operators."""
    with _LOCK:
        return dict(_SNAPSHOT)


def _multiprocess_enabled() -> bool:
    return bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))


def render_prometheus() -> bytes:
    """Render the current registry in the Prometheus text exposition format."""
    if _multiprocess_enabled():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(_REGISTRY)


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


def reset_metrics() -> None:
    """Reset process-local instruments for deterministic tests.

    Multiprocess files are owned by the process manager and must be cleaned
    before workers start, never from a live application or an individual test.
    """
    if _multiprocess_enabled():
        raise RuntimeError("Cannot reset metrics while PROMETHEUS_MULTIPROC_DIR is set")
    with _LOCK:
        _new_metric_state()
        _install_storage_callback()


_new_metric_state()
_install_storage_callback()

__all__ = [
    "increment_metric",
    "metrics_snapshot",
    "prometheus_content_type",
    "record_audit_event",
    "record_http_request",
    "record_storage_write",
    "render_prometheus",
    "reset_metrics",
    "set_audit_queue_depth",
]
