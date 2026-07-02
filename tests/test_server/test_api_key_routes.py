"""HTTP-level tests for /twin/api/settings/api-keys.

Covers the create → list → revoke flow, including:
- the one-time-reveal contract (``full_value`` only in POST response);
- ``hash`` never returned;
- 404 on unknown id;
- activity event emission on create/revoke.

Memgraph is required (integration marker).
"""

from __future__ import annotations

import secrets

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import api_key_store, webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.api_key_routes import router as api_key_router
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp


@pytest.fixture(autouse=True)
def _reset_auth_state():
    configure_idp(None)
    yield
    configure_idp(None)
    configure_auth(api_key=None, jwt_secret=None)


def test_api_key_router_rejects_anonymous_when_mounted_directly():
    """The admin router must not rely on create_app's outer auth wrapper.

    ``require_admin_user`` intentionally allows authenticated users while
    the IdP is dormant. If this router carries only that dependency, a direct
    include can expose API-key listing/mutation before any auth check runs.
    """
    configure_auth(api_key="root-secret")
    app = FastAPI()
    app.include_router(api_key_router, prefix="/twin/api")

    r = TestClient(app).get("/twin/api/settings/api-keys")
    assert r.status_code == 401


@pytest.fixture()
async def client(monkeypatch):
    monkeypatch.setenv("WORKSPACE", f"apikey_routes_{secrets.token_hex(4)}")
    webui_router.reset_store()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    # Cleanup the workspace label
    from twindb_lightrag_memgraph._constants import resolve_workspace

    try:
        await api_key_store.reset_workspace(resolve_workspace())
    except Exception:
        pass
    webui_router.reset_store()


@pytest.mark.integration
class TestApiKeyRoutes:
    async def test_empty_list_on_fresh_workspace(self, client):
        r = await client.get("/twin/api/settings/api-keys")
        assert r.status_code == 200
        assert r.json() == []

    async def test_create_returns_full_value_then_list_strips_it(self, client):
        r = await client.post(
            "/twin/api/settings/api-keys",
            json={"name": "ingestion-agent"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "ingestion-agent"
        assert body["full_value"].startswith(api_key_store.KEY_PREFIX)
        assert body["prefix"].startswith(api_key_store.KEY_PREFIX)
        assert "hash" not in body
        assert body["revoked_at"] is None

        # List exposes prefix only.
        r = await client.get("/twin/api/settings/api-keys")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert "full_value" not in rows[0]
        assert "hash" not in rows[0]
        assert rows[0]["prefix"] == body["prefix"]
        assert rows[0]["id"] == body["id"]

    async def test_revoke_marks_revoked_at_and_persists(self, client):
        created = (
            await client.post(
                "/twin/api/settings/api-keys",
                json={"name": "throwaway"},
            )
        ).json()
        key_id = created["id"]

        r = await client.delete(f"/twin/api/settings/api-keys/{key_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == key_id
        assert body["revoked_at"] is not None
        assert "hash" not in body

        # The key stays listed (audit trail) but with revoked_at set.
        listing = (await client.get("/twin/api/settings/api-keys")).json()
        assert len(listing) == 1
        assert listing[0]["revoked_at"] == body["revoked_at"]

    async def test_revoke_unknown_returns_404(self, client):
        r = await client.delete("/twin/api/settings/api-keys/no-such-id")
        assert r.status_code == 404

    async def test_create_emits_activity_event(self, client):
        before = (await client.get("/twin/api/activity")).json()
        before_count = len(
            before.get("items", before if isinstance(before, list) else [])
        )
        await client.post("/twin/api/settings/api-keys", json={"name": "trace-test"})
        after_raw = (await client.get("/twin/api/activity")).json()
        after = after_raw.get("items", after_raw if isinstance(after_raw, list) else [])
        assert len(after) == before_count + 1
        evt = after[0]  # newest first
        assert evt["kind"] == "api-key-created"
        assert evt["target"]["type"] == "api-key"
        assert evt["meta"]["operation"] == "created"
        # Audit meta MUST NOT contain the full secret — only the prefix
        meta_blob = repr(evt["meta"])
        # the prefix ends with the ellipsis char, the full value never would
        assert "…" in evt["meta"]["prefix"]
        assert "hash" not in meta_blob

    async def test_revoke_emits_activity_event(self, client):
        created = (
            await client.post(
                "/twin/api/settings/api-keys", json={"name": "revoke-test"}
            )
        ).json()
        await client.delete(f"/twin/api/settings/api-keys/{created['id']}")
        after_raw = (await client.get("/twin/api/activity")).json()
        after = after_raw.get("items", after_raw if isinstance(after_raw, list) else [])
        # The newest event is the revoke (created is right behind).
        revoke_evt = after[0]
        assert revoke_evt["kind"] == "api-key-revoked"
        assert revoke_evt["meta"]["operation"] == "revoked"
        assert revoke_evt["sev"] == "warning"

    async def test_create_rejects_blank_name(self, client):
        r = await client.post("/twin/api/settings/api-keys", json={"name": ""})
        # Pydantic min_length=1 → 422
        assert r.status_code == 422

    async def test_create_rejects_long_name(self, client):
        r = await client.post("/twin/api/settings/api-keys", json={"name": "x" * 200})
        assert r.status_code == 422
