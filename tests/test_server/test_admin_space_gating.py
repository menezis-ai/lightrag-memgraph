"""Admin-only Space CRUD — IdP-active integration tests.

Covers the gating that ``test_spaces_crud.py`` does NOT exercise:
when the IdP middleware is active, ``POST/PATCH/DELETE /spaces`` must
return:

- 401 if no bearer token is presented;
- 403 if the token's user has no ``admin:spaces`` gateway scope;
- 200/201/204 only when the user is in one of ``IdpConfig.admin_groups``.

``GET /spaces`` stays open in both modes (no admin gate on read).

The dormant-IdP regression (everything open) is owned by
``test_spaces_crud.py``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import idp_jwt, space_store, webui_router
from twindb_lightrag_memgraph.server.app import create_app


# ---------------------------------------------------------------------------
# Local copies of the test helpers from test_idp_jwt.py. Duplicating keeps
# this file self-contained — the shared bits are tiny (~30 lines) and would
# require a conftest plugin to share otherwise.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_key, public_pem


@pytest.fixture()
def fake_jwks(rsa_keypair):
    _, public_pem = rsa_keypair

    class _FakeKey:
        key = public_pem

    class _FakeClient:
        def get_signing_key_from_jwt(self, _token: str):
            return _FakeKey()

    return _FakeClient()


def _make_token(rsa_keypair, *, groups: list[str]) -> str:
    private_key, _ = rsa_keypair
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "https://idp.example/realms/twin",
        "aud": "twin",
        "sub": f"user-{groups[0] if groups else 'anon'}",
        "email": "user@example.com",
        "name": "Test User",
        "groups": groups,
        "scope": "read:documents",
        "iat": now,
        "exp": now + 600,
    }
    return pyjwt.encode(claims, private_key, algorithm="RS256")


def _activate_idp(fake_jwks, *, admin_groups: frozenset[str]) -> None:
    cfg = idp_jwt.IdpConfig(
        jwks_url="https://idp.example/jwks",
        issuer="https://idp.example/realms/twin",
        audience="twin",
        admin_groups=admin_groups,
    )
    idp_jwt._active_config = cfg  # type: ignore[attr-defined]
    idp_jwt._active_cache = idp_jwt.JwksCache(  # type: ignore[attr-defined]
        cfg, fetcher=lambda _url: fake_jwks
    )


# ---------------------------------------------------------------------------
# App + client fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(monkeypatch, tmp_path, fake_jwks):
    monkeypatch.setenv("TWIN_DEFAULT_SPACE", "default")
    monkeypatch.setenv(
        "TWIN_SPACES_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default", "kind": "primary"},
            ]
        ),
    )
    monkeypatch.setenv("TWIN_MAX_SPACES", "3")
    monkeypatch.setenv(
        "TWIN_SPACES_RUNTIME_FILE", str(tmp_path / "twin-spaces.json")
    )
    space_store.reset_runtime_store()
    webui_router.reset_store()

    # Active IdP with default admin_groups (twin-admin + twin-steward).
    _activate_idp(fake_jwks, admin_groups=frozenset({"twin-admin", "twin-steward"}))

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c

    space_store.reset_runtime_store()
    webui_router.reset_store()
    idp_jwt.configure_idp(None)


# ---------------------------------------------------------------------------
# POST /spaces
# ---------------------------------------------------------------------------


class TestCreateSpaceGating:
    async def test_no_token_returns_401(self, client):
        r = await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "S", "kind": "sandbox"},
        )
        assert r.status_code == 401
        assert 'error="missing_token"' in r.headers["www-authenticate"]

    async def test_non_admin_token_returns_403(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        r = await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "S"},
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 403
        assert idp_jwt.ADMIN_SPACES_SCOPE in r.json()["detail"]

    async def test_admin_token_returns_201(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 201
        assert r.json()["id"] == "sandbox"

    async def test_folder_route_uses_same_admin_gate(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 201
        assert r.json()["id"] == "sandbox"


# ---------------------------------------------------------------------------
# PATCH /spaces/{id}
# ---------------------------------------------------------------------------


class TestUpdateSpaceGating:
    async def _provision_runtime_space(self, client, rsa_keypair) -> None:
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox"},
            cookies={"twin_idp_token": admin},
        )
        assert r.status_code == 201

    async def test_no_token_returns_401(self, client, rsa_keypair):
        await self._provision_runtime_space(client, rsa_keypair)
        # Re-issue without cookie:
        client.cookies.clear()
        r = await client.patch(
            "/spaces/sandbox", json={"label": "x"}
        )
        assert r.status_code == 401

    async def test_non_admin_returns_403(self, client, rsa_keypair):
        await self._provision_runtime_space(client, rsa_keypair)
        client.cookies.clear()
        token = _make_token(rsa_keypair, groups=["twin-contributor"])
        r = await client.patch(
            "/spaces/sandbox",
            json={"label": "Renamed"},
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 403

    async def test_admin_updates_label(self, client, rsa_keypair):
        await self._provision_runtime_space(client, rsa_keypair)
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await client.patch(
            "/spaces/sandbox",
            json={"label": "Sandbox v2"},
            cookies={"twin_idp_token": admin},
        )
        assert r.status_code == 200
        assert r.json()["kb"] == "Sandbox v2"


# ---------------------------------------------------------------------------
# DELETE /spaces/{id}
# ---------------------------------------------------------------------------


class TestDeleteSpaceGating:
    async def _provision_runtime_space(self, client, rsa_keypair) -> None:
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox"},
            cookies={"twin_idp_token": admin},
        )
        assert r.status_code == 201

    async def test_no_token_returns_401(self, client, rsa_keypair):
        await self._provision_runtime_space(client, rsa_keypair)
        client.cookies.clear()
        r = await client.delete("/spaces/sandbox")
        assert r.status_code == 401

    async def test_non_admin_returns_403(self, client, rsa_keypair):
        await self._provision_runtime_space(client, rsa_keypair)
        client.cookies.clear()
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        r = await client.delete(
            "/spaces/sandbox",
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 403

    async def test_admin_deletes_204(self, client, rsa_keypair):
        await self._provision_runtime_space(client, rsa_keypair)
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await client.delete(
            "/spaces/sandbox",
            cookies={"twin_idp_token": admin},
        )
        assert r.status_code == 204


# ---------------------------------------------------------------------------
# GET /spaces — read stays open under active IdP
# ---------------------------------------------------------------------------


class TestListSpacesNoGate:
    async def test_get_works_without_admin_scope(self, client, rsa_keypair):
        # A pure reader can still enumerate spaces (needed by the
        # space-switcher dropdown). Auth is required (router-level
        # ``require_auth`` dep), but the admin gate isn't.
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        r = await client.get(
            "/spaces", cookies={"twin_idp_token": token}
        )
        assert r.status_code == 200
        assert any(s["id"] == "default" for s in r.json())


# ---------------------------------------------------------------------------
# Custom admin_groups env: steward is no longer admin
# ---------------------------------------------------------------------------


class TestCustomAdminGroupsEnv:
    @pytest.fixture()
    async def custom_client(self, monkeypatch, tmp_path, fake_jwks):
        monkeypatch.setenv("TWIN_DEFAULT_SPACE", "default")
        monkeypatch.setenv(
            "TWIN_SPACES_JSON",
            json.dumps([{"id": "default", "label": "Default", "kind": "primary"}]),
        )
        monkeypatch.setenv("TWIN_MAX_SPACES", "3")
        monkeypatch.setenv(
            "TWIN_SPACES_RUNTIME_FILE", str(tmp_path / "twin-spaces.json")
        )
        space_store.reset_runtime_store()
        webui_router.reset_store()
        # Custom admin set: ONLY bnp.kb-admin grants admin:spaces.
        _activate_idp(fake_jwks, admin_groups=frozenset({"bnp.kb-admin"}))
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
        space_store.reset_runtime_store()
        webui_router.reset_store()
        idp_jwt.configure_idp(None)

    async def test_steward_no_longer_admin(self, custom_client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-steward"])
        r = await custom_client.post(
            "/spaces",
            json={"id": "sandbox", "label": "S"},
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 403

    async def test_custom_admin_group_allowed(
        self, custom_client, rsa_keypair
    ):
        token = _make_token(rsa_keypair, groups=["bnp.kb-admin"])
        r = await custom_client.post(
            "/spaces",
            json={"id": "sandbox", "label": "S"},
            cookies={"twin_idp_token": token},
        )
        assert r.status_code == 201
