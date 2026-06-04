"""Route parity contract for the Twin WebUI surface.

The React client, MSW handlers, and FastAPI routers must not drift silently.
This test deliberately keeps a short allow-list of known backend gaps so CI
can catch new Disneyland routes without blocking the current documented
Couche 3 backlog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fastapi.routing import APIRoute

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.native_shims import (
    build_health_shim,
    build_native_shims_router,
)
from twindb_lightrag_memgraph.server.twin_query_routes import (
    build_twin_query_router,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_TS = REPO_ROOT / "lightrag_webui_twin/src/api/resources.ts"
MSW_HANDLERS_TS = REPO_ROOT / "lightrag_webui_twin/src/mocks/handlers.ts"


@dataclass(frozen=True, order=True)
class Route:
    method: str
    path: str


def _normalize_path(path: str) -> str:
    path = path.replace("${ANY}", "")
    path = path.replace("${TWIN}", "/twin/api")
    path = path.replace("//", "/")
    path = re.sub(r":[A-Za-z_][A-Za-z0-9_]*", "{param}", path)
    path = re.sub(r"\{[^}/]+\}", "{param}", path)
    return path


def _fmt(routes: set[Route]) -> str:
    return "\n".join(f"- {route.method} {route.path}" for route in sorted(routes))


def _fastapi_routes_from_router(router, *, prefix: str = "") -> set[Route]:
    routes: set[Route] = set()
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or set():
            if method not in {"GET", "POST", "PATCH", "DELETE"}:
                continue
            routes.add(Route(method, _normalize_path(f"{prefix}{route.path}")))
    return routes


def _backend_routes() -> set[Route]:
    def _fake_rag():
        raise RuntimeError("route parity only introspects paths")

    return (
        _fastapi_routes_from_router(webui_router.router, prefix="/twin/api")
        | _fastapi_routes_from_router(build_native_shims_router(_fake_rag))
        | _fastapi_routes_from_router(build_health_shim(_fake_rag))
        | _fastapi_routes_from_router(
            build_twin_query_router(_fake_rag), prefix="/twin/api"
        )
    )


def _msw_routes() -> set[Route]:
    text = MSW_HANDLERS_TS.read_text(encoding="utf-8")
    routes: set[Route] = set()
    pattern = re.compile(r"http\.(get|post|patch|delete)\(\s*`([^`]+)`", re.S)
    for method, raw_path in pattern.findall(text):
        path = _normalize_path(raw_path)
        if path.startswith("/__e2e/"):
            continue
        routes.add(Route(method.upper(), path))
    return routes


# Mirrors lightrag_webui_twin/src/api/resources.ts. Keep this list focused on
# production client paths, not test-only MSW controls.
FRONTEND_PRODUCTION_ROUTES: set[Route] = {
    Route("GET", "/documents"),
    Route("GET", "/documents/{param}/chunks"),
    Route("GET", "/documents/track_status/{param}"),
    Route("POST", "/documents/{param}/scan"),
    Route("POST", "/documents/reprocess_failed"),
    Route("POST", "/documents/upload"),
    Route("DELETE", "/documents/{param}"),
    Route("GET", "/health"),
    Route("GET", "/pipeline_status"),
    Route("GET", "/openapi"),
    Route("POST", "/query"),
    Route("POST", "/twin/api/query"),
    Route("POST", "/twin/api/query/stream"),
    Route("GET", "/twin/api/workspaces"),
    Route("GET", "/twin/api/spaces"),
    Route("POST", "/twin/api/spaces"),
    Route("PATCH", "/twin/api/spaces/{param}"),
    Route("DELETE", "/twin/api/spaces/{param}"),
    Route("GET", "/twin/api/notifications"),
    Route("POST", "/twin/api/notifications/read-all"),
    Route("DELETE", "/twin/api/notifications"),
    Route("GET", "/twin/api/health"),
    Route("GET", "/twin/api/thesaurus"),
    Route("GET", "/twin/api/tags"),
    Route("GET", "/twin/api/tags/categories"),
    Route("GET", "/twin/api/tags/categories/template"),
    Route("POST", "/twin/api/tags/categories/_import"),
    Route("POST", "/twin/api/tags"),
    Route("POST", "/twin/api/tags/{param}/approve"),
    Route("POST", "/twin/api/tags/{param}/reject"),
    Route("PATCH", "/twin/api/tags/{param}"),
    Route("POST", "/twin/api/tags/{param}/deprecate"),
    Route("POST", "/twin/api/tags/{param}/synonyms"),
    Route("DELETE", "/twin/api/tags/{param}"),
    Route("GET", "/twin/api/activity"),
    Route("GET", "/twin/api/documents/{param}/metadata"),
    Route("POST", "/twin/api/documents/{param}/approve"),
    Route("POST", "/twin/api/documents/{param}/reject"),
    Route("POST", "/twin/api/documents/bulk-delete"),
    Route("POST", "/twin/api/documents/_bulk-retag"),
    Route("POST", "/twin/api/auth/logout"),
    Route("GET", "/twin/api/graph/entities"),
    Route("GET", "/twin/api/graph/relations"),
    Route("PATCH", "/twin/api/graph/entities/{param}"),
    Route("PATCH", "/twin/api/graph/relations/{param}"),
    Route("POST", "/twin/api/graph/entities"),
    Route("DELETE", "/twin/api/graph/entities/{param}"),
    Route("POST", "/twin/api/graph/relations"),
    Route("DELETE", "/twin/api/graph/relations/{param}"),
}

# Owned by native LightRAG routes in production, not by our shims/router.
LIGHTRAG_NATIVE_PASSTHROUGH: set[Route] = {
    Route("GET", "/documents/track_status/{param}"),
    Route("POST", "/documents/reprocess_failed"),
    Route("POST", "/documents/upload"),
    Route("POST", "/query"),
    # FastAPI auto-exposes the live OpenAPI 3.1 spec on /openapi.json.
    # The Twin React Settings → API tab hits this directly so it stays
    # ISO with the LightRAG WebUI by construction (mock-kill F2).
    Route("GET", "/openapi.json"),
}

# Known Couche 3 gaps. If one of these starts passing, remove it from this set
# and update WEBUI-WIRING-PLAN.md.
KNOWN_BACKEND_GAPS: set[Route] = set()


def test_frontend_contract_paths_are_declared_in_resources_ts():
    text = RESOURCES_TS.read_text(encoding="utf-8")
    markers = {
        "/documents/track_status/",
        "/documents/reprocess_failed",
        "/documents/upload",
        "/documents/bulk-delete",
        "/documents/_bulk-retag",
        "/documents/${encodeURIComponent(docId)}/metadata",
        "/spaces/${encodeURIComponent(id)}",
        "/graph/entities",
        "/graph/relations",
        "/tags/categories/_import",
        "/auth/logout",
    }
    missing = {marker for marker in markers if marker not in text}
    assert not missing, "resources.ts no longer declares:\n" + "\n".join(
        sorted(missing)
    )


def test_frontend_routes_are_covered_by_msw_handlers():
    missing = FRONTEND_PRODUCTION_ROUTES - _msw_routes()
    assert not missing, "MSW is missing frontend route(s):\n" + _fmt(missing)


def test_frontend_routes_are_real_backend_or_known_gap():
    covered = _backend_routes() | LIGHTRAG_NATIVE_PASSTHROUGH
    missing = FRONTEND_PRODUCTION_ROUTES - covered
    unexpected = missing - KNOWN_BACKEND_GAPS
    assert not unexpected, (
        "Frontend route(s) are not exposed by backend and are not listed as "
        "known Couche 3 gaps:\n" + _fmt(unexpected)
    )
    stale_known_gaps = KNOWN_BACKEND_GAPS - missing
    assert not stale_known_gaps, (
        "Known route gap(s) are now covered; remove them from "
        "KNOWN_BACKEND_GAPS and update WEBUI-WIRING-PLAN.md:\n"
        + _fmt(stale_known_gaps)
    )


def test_msw_routes_do_not_exceed_backend_by_accident():
    covered = _backend_routes() | LIGHTRAG_NATIVE_PASSTHROUGH
    production_msw = {
        route for route in _msw_routes() if not route.path.startswith("/__e2e/")
    }
    extra = production_msw - covered - KNOWN_BACKEND_GAPS
    assert not extra, (
        "MSW route(s) have no backend implementation and no known-gap entry:\n"
        + _fmt(extra)
    )
