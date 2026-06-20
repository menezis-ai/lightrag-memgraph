"""Startup sanity checks for the Twin API route surface.

These checks are intentionally route-table based and ignore ``Mount`` /
``StaticFiles`` entries. A static ``/twin`` mount can match
``/twin/api/...`` and return 404/405 at request time; that is exactly the
miswiring this module must not consider healthy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiWiringProbe:
    method: str
    path: str
    label: str


def api_wiring_probes(api_prefix: str = "/twin/api") -> tuple[ApiWiringProbe, ...]:
    prefix = api_prefix.rstrip("/")
    return (
        ApiWiringProbe("GET", f"{prefix}/health", "health"),
        ApiWiringProbe("GET", f"{prefix}/folders", "folders:list"),
        ApiWiringProbe("GET", f"{prefix}/documents", "documents:list"),
        ApiWiringProbe(
            "GET",
            f"{prefix}/documents/{{doc_id}}/metadata",
            "documents:metadata",
        ),
        ApiWiringProbe("GET", f"{prefix}/tags", "tags:list"),
        ApiWiringProbe("GET", f"{prefix}/activity", "activity:list"),
        ApiWiringProbe("GET", f"{prefix}/notifications", "notifications:list"),
        ApiWiringProbe("GET", f"{prefix}/graph/entities", "graph:entities"),
        ApiWiringProbe("GET", f"{prefix}/graph/relations", "graph:relations"),
        ApiWiringProbe("POST", f"{prefix}/query", "query"),
        ApiWiringProbe("POST", f"{prefix}/query/data", "query:data"),
        ApiWiringProbe("POST", f"{prefix}/query/stream", "query:stream"),
        ApiWiringProbe("GET", f"{prefix}/settings/api-keys", "api-keys:list"),
        ApiWiringProbe("POST", f"{prefix}/settings/api-keys", "api-keys:create"),
        ApiWiringProbe(
            "DELETE",
            f"{prefix}/settings/api-keys/{{key_id}}",
            "api-keys:revoke",
        ),
        ApiWiringProbe("GET", f"{prefix}/quota", "quota:snapshot"),
    )


DEFAULT_API_WIRING_PROBES: tuple[ApiWiringProbe, ...] = api_wiring_probes()


def _route_matches(route: object, *, method: str, path: str) -> bool:
    methods = getattr(route, "methods", None)
    if not methods or method not in methods:
        return False
    route_path = getattr(route, "path", None)
    if route_path == path:
        return True
    path_regex = getattr(route, "path_regex", None)
    if path_regex is not None:
        return path_regex.match(path) is not None
    return False


def _openapi_pairs(app: object) -> set[tuple[str, str]]:
    """``(METHOD, templated_path)`` pairs from the live OpenAPI schema.

    OpenAPI is the version-robust source of truth. FastAPI 0.137 wraps
    ``include_router`` results in ``_IncludedRouter`` objects that expose
    neither ``.path`` nor ``.routes``, so a raw ``app.routes`` scan misses
    *every* included route — the exact blindness that let this check report
    api-keys/quota as missing while they were mounted. The schema also
    naturally excludes ``Mount``/``StaticFiles``, which is precisely the
    surface this sanity check must not count as healthy.
    """
    openapi = getattr(app, "openapi", None)
    if not callable(openapi):
        return set()
    try:
        paths = (openapi() or {}).get("paths", {}) or {}
    except Exception:  # pragma: no cover - schema build is best-effort here
        return set()
    return {
        (str(m).upper(), str(path))
        for path, ops in paths.items()
        if isinstance(ops, dict)
        for m in ops
    }


def _has_api_route(
    app: object,
    *,
    method: str,
    path: str,
    openapi_pairs: set[tuple[str, str]] | None = None,
) -> bool:
    if openapi_pairs is None:
        openapi_pairs = _openapi_pairs(app)
    if (method.upper(), path) in openapi_pairs:
        return True
    # Fallback: raw route-table scan (flattened routes / older FastAPI /
    # in-schema=False routes the OpenAPI schema omits).
    router = getattr(app, "router", None)
    routes = getattr(router, "routes", getattr(app, "routes", ()))
    return any(_route_matches(route, method=method, path=path) for route in routes)


def log_api_wiring_sanity(
    app: object,
    *,
    probes: Iterable[ApiWiringProbe] = DEFAULT_API_WIRING_PROBES,
    surface: str = "twin-api",
) -> list[ApiWiringProbe]:
    """Log whether critical Twin API endpoints are mounted.

    Returns the missing probes so tests can assert the same contract that
    operators see in Elastic.
    """

    openapi_pairs = _openapi_pairs(app)
    present: list[ApiWiringProbe] = []
    missing: list[ApiWiringProbe] = []
    for probe in probes:
        if _has_api_route(
            app, method=probe.method, path=probe.path, openapi_pairs=openapi_pairs
        ):
            present.append(probe)
        else:
            missing.append(probe)

    fmt_present = ", ".join(f"{p.method} {p.path} ({p.label})" for p in present)
    if missing:
        fmt_missing = ", ".join(f"{p.method} {p.path} ({p.label})" for p in missing)
        logger.warning(
            "twindb: 🚨 API CHECK FAILED ❌ surface=%s missing=[%s] present=[%s] "
            "action=route_wiring_broken",
            surface,
            fmt_missing,
            fmt_present,
        )
    else:
        logger.info(
            "twindb: All API Check passes ✅☀️ surface=%s routes=[%s]",
            surface,
            fmt_present,
        )
    return missing
