"""Standalone server route surface.

The React port assumes every Twin endpoint lives under `/twin/api/...`.
The standalone `create_app` (used by `python -m twindb_lightrag_memgraph.server`)
must therefore expose the same prefixed surface as the plugin
topology (`register(mount_server=True, ...)`). Without these routes a
React frontend pointed at the standalone server would 404 on every
Twin call — that gap was the 2026-06-08 review finding before this
guard was added.
"""

from __future__ import annotations

from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        memgraph_uri="bolt://localhost:7687",
        host="127.0.0.1",
        port=9622,
        api_key="t",
        llm_binding_api_key="t",
        embedding_binding_api_key="t",
    )


def _paths(app, methods: set[str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        method_set = getattr(route, "methods", None)
        if not path or not method_set:
            continue
        for m in methods & method_set:
            out.add((m, path))
    return out


class TestStandaloneTwinSurface:
    def test_twin_query_endpoints_are_exposed_under_prefix(self):
        app = create_app(_settings())
        routes = _paths(app, {"POST"})
        assert ("POST", "/twin/api/query") in routes
        assert ("POST", "/twin/api/query/stream") in routes

    def test_native_legacy_query_route_still_present(self):
        # The simple `/query` route remains for legacy callers that
        # were using the standalone before the Twin overlay landed.
        app = create_app(_settings())
        routes = _paths(app, {"POST"})
        assert ("POST", "/query") in routes

    def test_webui_router_mounted_both_unprefixed_and_under_twin_api(self):
        # Backwards-compat: existing pytest suite hits `/documents`,
        # `/spaces`, etc. directly. The React port hits `/twin/api/...`.
        # Both must work against the same server.
        app = create_app(_settings())
        routes = _paths(app, {"GET"})
        assert ("GET", "/documents") in routes
        assert ("GET", "/twin/api/documents") in routes
