"""HTTP-level tests for the quota surface.

Covers ``GET /twin/api/quota`` (snapshot payload, all 3 status states)
and the middleware that gates ingestion endpoints with 507 when
blocked. Memgraph is mocked via monkeypatching ``quota.snapshot``;
no real DB needed.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import quota, webui_router
from twindb_lightrag_memgraph.server.app import create_app


@pytest.fixture()
async def client_factory(monkeypatch):
    """Yields a builder that takes a status fixture and returns the client."""
    webui_router.reset_store()
    monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "2GiB")
    app = create_app()

    def _patch(status: str, used: int, limit: int) -> None:
        async def _fake_snapshot():
            return {
                "status": status,
                "used_bytes": used,
                "limit_bytes": limit,
                "used_pct": (used / limit) if limit else None,
                "warn_threshold": quota.WARN_THRESHOLD,
                "configured": True,
            }

        monkeypatch.setattr(quota, "snapshot", _fake_snapshot)

    async def _build():
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    yield _patch, _build
    webui_router.reset_store()


class TestQuotaSnapshotEndpoint:
    async def test_ok_state(self, client_factory):
        patch, build = client_factory
        patch("ok", 100 * 1024 ** 2, 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.get("/twin/api/quota")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["used_bytes"] == 100 * 1024 ** 2
        assert body["limit_bytes"] == 2 * 1024 ** 3
        assert 0 < body["used_pct"] < 0.85
        assert body["configured"] is True
        assert body["warn_threshold"] == 0.85

    async def test_warning_state(self, client_factory):
        patch, build = client_factory
        patch("warning", int(0.9 * 2 * 1024 ** 3), 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.get("/twin/api/quota")
        assert r.status_code == 200
        assert r.json()["status"] == "warning"

    async def test_blocked_state(self, client_factory):
        patch, build = client_factory
        patch("blocked", 2 * 1024 ** 3, 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.get("/twin/api/quota")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "blocked"
        assert body["used_pct"] >= 1.0


class TestQuotaMiddlewareGate:
    async def test_upload_blocked_returns_507(self, client_factory):
        patch, build = client_factory
        patch("blocked", 2 * 1024 ** 3 + 1, 2 * 1024 ** 3)
        async with await build() as c:
            # The middleware short-circuits BEFORE the native upload
            # handler runs, so the multipart body shape is irrelevant.
            r = await c.post("/documents/upload")
        assert r.status_code == 507
        body = r.json()
        assert "quota reached" in body["detail"]
        assert "GiB" in body["detail"]

    async def test_reprocess_failed_blocked_returns_507(self, client_factory):
        patch, build = client_factory
        patch("blocked", 2 * 1024 ** 3 + 1, 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.post("/documents/reprocess_failed")
        assert r.status_code == 507
        assert "quota reached" in r.json()["detail"]

    async def test_scan_blocked_returns_507(self, client_factory):
        patch, build = client_factory
        patch("blocked", 2 * 1024 ** 3 + 1, 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.post("/documents/abc-123/scan")
        assert r.status_code == 507

    async def test_warning_does_not_block(self, client_factory):
        patch, build = client_factory
        patch("warning", int(0.9 * 2 * 1024 ** 3), 2 * 1024 ** 3)
        async with await build() as c:
            # Will be 404 or another error from the downstream handler
            # — the only thing we assert is NOT 507.
            r = await c.post("/documents/reprocess_failed")
        assert r.status_code != 507

    async def test_ok_does_not_block(self, client_factory):
        patch, build = client_factory
        patch("ok", 100 * 1024 ** 2, 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.post("/documents/reprocess_failed")
        assert r.status_code != 507

    async def test_get_endpoints_never_gated(self, client_factory):
        patch, build = client_factory
        patch("blocked", 2 * 1024 ** 3 + 1, 2 * 1024 ** 3)
        async with await build() as c:
            r = await c.get("/twin/api/quota")
            # Snapshot endpoint must always answer (blocked status is
            # exactly the info the operator needs).
            assert r.status_code == 200


class TestQuotaUnconfiguredBypass:
    async def test_no_env_var_yields_ok_and_no_gating(self, monkeypatch):
        """Dev posture: MEMGRAPH_MEMORY_LIMIT unset → guard inert."""
        webui_router.reset_store()
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)

        async def _fake_probe():
            return 100

        monkeypatch.setattr(quota, "get_used_bytes", _fake_probe)
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/twin/api/quota")
            assert r.status_code == 200
            body = r.json()
            assert body["configured"] is False
            assert body["status"] == "ok"
            assert body["limit_bytes"] is None

            # Even an upload attempt must NOT see 507 — the guard is
            # disabled when no quota is configured.
            r = await c.post("/documents/reprocess_failed")
            assert r.status_code != 507
        webui_router.reset_store()
