"""Auth chain integration for per-operator API keys.

Verifies the order documented in ``require_auth``:

  1. IdP JWT
  2. LIGHTRAG_API_KEY  (unchanged)
  3. NEW: per-operator key hash-lookup
  4. Legacy local JWT
  5. fallback 401

Specifically:
- LIGHTRAG_API_KEY still authenticates (no regression).
- A minted per-operator key authenticates and produces ``user="api_key:<id>"``.
- Once revoked, the same key rejects.
- A random ``twk_`` prefix that does not match any stored hash falls
  through to branch 4 (JWT) — which 401s with ``Invalid token``.
- Non-``twk_`` bearers bypass the API-key branch entirely.

Requires Memgraph (integration).
"""

from __future__ import annotations

import secrets

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import (
    api_key_store,
    auth as auth_module,
    webui_router,
)
from twindb_lightrag_memgraph.server.app import create_app


@pytest.fixture()
async def client(monkeypatch):
    # Static key + JWT secret both set: we exercise branch ordering.
    monkeypatch.setenv("WORKSPACE", f"apikey_auth_{secrets.token_hex(4)}")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "static-root-key-XYZ")
    monkeypatch.setenv("LIGHTRAG_JWT_SECRET", "test-secret-shhh")
    webui_router.reset_store()
    auth_module._reset_for_tests() if hasattr(auth_module, "_reset_for_tests") else None
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    from twindb_lightrag_memgraph._constants import resolve_workspace

    try:
        await api_key_store.reset_workspace(resolve_workspace())
    except Exception:
        pass
    webui_router.reset_store()


@pytest.mark.integration
class TestAuthChainWithApiKeys:
    async def test_static_api_key_still_works(self, client):
        # branch 2: LIGHTRAG_API_KEY is honoured unchanged.
        r = await client.get(
            "/auth-status",
            headers={"Authorization": "Bearer static-root-key-XYZ"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is True
        assert body["user"] == "api_key"
        assert body["identity"] is None

    async def test_minted_per_operator_key_authenticates(self, client):
        created = (
            await client.post(
                "/twin/api/settings/api-keys",
                headers={"Authorization": "Bearer static-root-key-XYZ"},
                json={"name": "auth-chain-test"},
            )
        ).json()
        full = created["full_value"]
        r = await client.get(
            "/auth-status", headers={"Authorization": f"Bearer {full}"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is True
        assert body["user"] == f"api_key:{created['id']}"
        assert body["identity"] is None

    async def test_revoked_per_operator_key_rejects(self, client):
        created = (
            await client.post(
                "/twin/api/settings/api-keys",
                headers={"Authorization": "Bearer static-root-key-XYZ"},
                json={"name": "to-revoke"},
            )
        ).json()
        full = created["full_value"]
        await client.delete(
            f"/twin/api/settings/api-keys/{created['id']}",
            headers={"Authorization": "Bearer static-root-key-XYZ"},
        )
        # Hit a protected route directly to see the 401
        r = await client.get(
            "/twin/api/health", headers={"Authorization": f"Bearer {full}"}
        )
        assert r.status_code == 401

    async def test_unknown_twk_falls_through_to_jwt_and_401s(self, client):
        # A twk_-prefixed bearer that does NOT match any stored hash
        # should NOT short-circuit at branch 3; it falls through to
        # branch 4 (JWT decode) which 401s with "Invalid token".
        r = await client.get(
            "/twin/api/health",
            headers={"Authorization": "Bearer twk_unknown_random_value_here"},
        )
        assert r.status_code == 401
        # Detail must be the JWT decode error, NOT "Invalid credentials"
        assert "Invalid token" in r.json().get("detail", "")

    async def test_non_twk_bearer_skips_api_key_branch(self, client):
        # A bearer not starting with twk_ should be JWT-decoded immediately.
        r = await client.get(
            "/twin/api/health",
            headers={"Authorization": "Bearer not.a.real.jwt"},
        )
        assert r.status_code == 401
        assert "Invalid token" in r.json().get("detail", "")


@pytest.fixture()
async def open_access_client(monkeypatch):
    """Open-access mode (no LIGHTRAG_API_KEY, no JWT secret).

    Validates the reviewer fix: a deployment running with no static
    backend should still be able to mint AND consume twk_ API keys via
    Settings → API keys.
    """
    monkeypatch.setenv("WORKSPACE", f"open_access_{secrets.token_hex(4)}")
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    monkeypatch.delenv("LIGHTRAG_JWT_SECRET", raising=False)
    monkeypatch.delenv("TOKEN_SECRET", raising=False)
    monkeypatch.delenv("AUTH_ACCOUNTS", raising=False)
    webui_router.reset_store()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    from twindb_lightrag_memgraph._constants import resolve_workspace

    try:
        await api_key_store.reset_workspace(resolve_workspace())
    except Exception:
        pass
    webui_router.reset_store()


@pytest.mark.integration
class TestOpenAccessTwkOptIn:
    async def test_anonymous_still_passes_in_open_access(self, open_access_client):
        # No bearer presented → request goes through (open-access default).
        r = await open_access_client.get("/twin/api/health")
        assert r.status_code == 200

    async def test_minted_twk_authenticates_in_open_access(self, open_access_client):
        # Open access does not imply administration: without an IdP, only the
        # separately managed root key may mint operator credentials.
        denied = await open_access_client.post(
            "/twin/api/settings/api-keys", json={"name": "open-test"}
        )
        assert denied.status_code == 403

        # Provision one out of band to keep exercising the independent runtime
        # contract: a valid twk_ bearer authenticates even when ordinary
        # anonymous reads remain enabled.
        from twindb_lightrag_memgraph._constants import resolve_workspace

        created = await api_key_store.create_key(
            resolve_workspace(), name="open-test", created_by="test-provisioner"
        )
        full = created["full_value"]
        # And use that key on a protected route.
        r = await open_access_client.get(
            "/auth-status", headers={"Authorization": f"Bearer {full}"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is True
        assert body["user"] == f"api_key:{created['id']}"

    async def test_unknown_twk_rejects_in_open_access(self, open_access_client):
        # A twk_-prefixed bearer that does NOT match must 401 even in
        # open-access mode — opt-in via prefix means strict validation.
        r = await open_access_client.get(
            "/twin/api/health",
            headers={"Authorization": "Bearer twk_unknown_open_access"},
        )
        assert r.status_code == 401
        assert "Invalid credentials" in r.json().get("detail", "")
