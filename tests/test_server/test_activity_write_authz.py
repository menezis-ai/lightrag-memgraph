"""Regression: audit-feed write route is admin-only (audit 2026-08-06, R-03a).

Before the fix, any authenticated caller — including a non-admin ``twk_``
key or local JWT — could forge ``source-uploaded`` entries in the Activity
audit feed, destroying its probative value. The route is now gated by
``require_admin_user``; the authoritative event is emitted server-side by
the ingestion pipeline (covered in ``tests/test_prompt_security_ingestion.py``).
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import _create_jwt
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings

ROOT_KEY = "test-infra-root"
JWT_SECRET = "x" * 48


def _make_settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        working_dir="/tmp/lightrag_activity_authz_test",
        workspace="cib",
        enable_langsmith_tracing=False,
        api_key=ROOT_KEY,
        jwt_secret=JWT_SECRET,
        enable_webui_routes=True,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    webui_router.reset_store()
    yield
    webui_router.reset_store()


@pytest.fixture
async def app():
    return create_app(_make_settings())


def _non_admin_headers() -> dict[str, str]:
    """A valid local JWT — authenticated, but NOT infrastructure root."""
    return {"Authorization": f"Bearer {_create_jwt({'sub': 'operator'})}"}


class TestUploadActivityWriteGate:
    async def test_non_admin_authenticated_gets_403(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers=_non_admin_headers(),
        ) as client:
            resp = await client.post(
                "/documents/uploads/activity",
                json={"source": "forged-C4.pdf", "status": "uploaded"},
            )
        assert resp.status_code == 403

    async def test_infrastructure_root_can_still_write(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {ROOT_KEY}"},
        ) as client:
            resp = await client.post(
                "/documents/uploads/activity",
                json={"source": "backfill.pdf", "status": "accepted"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    async def test_admin_client_events_are_stamped_client(self, app):
        """Even an admin-written event is marked non-probative vs server ones."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {ROOT_KEY}"},
        ) as client:
            await client.post(
                "/documents/uploads/activity", json={"source": "backfill.pdf"}
            )
            feed = await client.get("/activity", params={"kind": "source-uploaded"})
        items = [
            e for e in feed.json()["items"] if e["target"]["label"] == "backfill.pdf"
        ]
        assert len(items) == 1
        assert items[0]["meta"]["emitted_by"] == "client"

    async def test_anonymous_gets_401_when_auth_configured(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/documents/uploads/activity", json={"source": "x.pdf"}
            )
        assert resp.status_code == 401
