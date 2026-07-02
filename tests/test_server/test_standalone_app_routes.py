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

from fastapi.testclient import TestClient

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


def _client() -> TestClient:
    return TestClient(create_app(_settings()), raise_server_exceptions=False)


class TestStandaloneTwinSurface:
    def test_twin_query_endpoints_are_exposed_under_prefix(self):
        client = _client()
        headers = {"Authorization": "Bearer t"}

        query = client.post(
            "/twin/api/query",
            json={"query": "route probe"},
            headers=headers,
        )
        stream = client.post(
            "/twin/api/query/stream",
            json={"query": "route probe"},
            headers=headers,
        )

        # The test app does not run lifespan, so the handler returns 500
        # when it reaches _get_rag(). A 404/405 would mean the route is absent.
        assert query.status_code == 500
        assert query.json()["detail"] == "LightRAG not initialized"
        assert stream.status_code == 500
        assert stream.json()["detail"] == "LightRAG not initialized"

    def test_native_legacy_query_route_still_present(self):
        # The simple `/query` route remains for legacy callers that
        # were using the standalone before the Twin overlay landed.
        response = _client().post(
            "/query",
            json={"query": "route probe"},
            headers={"Authorization": "Bearer t"},
        )

        assert response.status_code == 500

    def test_webui_router_mounted_both_unprefixed_and_under_twin_api(self):
        # Backwards-compat: existing pytest suite hits `/documents`,
        # `/spaces`, etc. directly. The React port hits `/twin/api/...`.
        # Both must work against the same server.
        client = _client()
        headers = {"Authorization": "Bearer t"}

        assert client.get("/documents", headers=headers).status_code == 200
        assert client.get("/twin/api/documents", headers=headers).status_code == 200
