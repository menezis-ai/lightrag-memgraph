"""Admin-only Folder CRUD — IdP-active integration tests.

Covers the gating that ``test_folders_crud.py`` does NOT exercise:
when the IdP middleware is active, ``POST/PATCH/DELETE /folders`` must
return:

- 401 if no bearer token is presented;
- 403 if the token's user has no ``admin:folders`` gateway scope;
- 200/201/204 only when the user is in one of ``IdpConfig.admin_groups``.

``GET /folders`` stays open in both modes (no admin gate on read).

The dormant-IdP regression (everything open) is owned by
``test_folders_crud.py``.
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

from twindb_lightrag_memgraph.server import idp_jwt, folder_store, webui_router
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


def _set_idp_cookie(client: AsyncClient, token: str) -> None:
    client.cookies.set("twin_idp_token", token)


# ---------------------------------------------------------------------------
# App + client fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(monkeypatch, tmp_path, fake_jwks):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default", "kind": "primary"},
            ]
        ),
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "3")
    monkeypatch.setenv(
        "TWIN_FOLDERS_RUNTIME_FILE", str(tmp_path / "twin-folders.json")
    )
    folder_store.reset_runtime_store()
    webui_router.reset_store()

    # ``create_app`` now also wires the IdP from env (audit 2026-06-10 H1).
    # Build the app first so ``configure_idp(None)`` doesn't clobber the
    # JWKS cache we are about to install, then activate the test IdP with
    # the mocked fetcher.
    app = create_app()
    _activate_idp(fake_jwks, admin_groups=frozenset({"twin-admin", "twin-steward"}))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c

    folder_store.reset_runtime_store()
    webui_router.reset_store()
    idp_jwt.configure_idp(None)


# ---------------------------------------------------------------------------
# POST /folders
# ---------------------------------------------------------------------------


class TestCreateFolderGating:
    async def _latest_auth_event(self):
        store = webui_router.get_store()
        items, _, _ = await store.list_activity(kind="auth", limit=1)
        assert items
        return items[0]

    async def test_no_token_returns_401(self, client):
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "S", "kind": "sandbox"},
        )
        assert r.status_code == 401
        assert 'error="missing_token"' in r.headers["www-authenticate"]

    async def test_no_token_emits_auth_activity_401(self, client):
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "S", "kind": "sandbox"},
        )
        assert r.status_code == 401

        event = await self._latest_auth_event()
        assert event["kind"] == "auth"
        assert event["sev"] == "warning"
        assert event["actor"]["user"] == "anonymous"
        assert event["target"] == {"type": "route", "label": "/folders"}
        assert event["meta"] == {
            "operation": "access_denied",
            "method": "POST",
            "path": "/folders",
            "status_code": 401,
            "reason": "unauthorized",
        }

    async def test_non_admin_token_returns_403(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        _set_idp_cookie(client, token)
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "S"},
        )
        assert r.status_code == 403
        assert idp_jwt.ADMIN_FOLDERS_SCOPE in r.json()["detail"]

    async def test_non_admin_emits_auth_activity_403(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        _set_idp_cookie(client, token)
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "S"},
        )
        assert r.status_code == 403

        event = await self._latest_auth_event()
        assert event["kind"] == "auth"
        assert event["sev"] == "warning"
        assert event["actor"]["user"] == "user-twin-reader"
        assert event["target"] == {"type": "route", "label": "/folders"}
        assert event["meta"]["operation"] == "access_denied"
        assert event["meta"]["method"] == "POST"
        assert event["meta"]["path"] == "/folders"
        assert event["meta"]["status_code"] == 403
        assert event["meta"]["reason"] == "forbidden"

    async def test_admin_token_returns_201(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(client, token)
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
        )
        assert r.status_code == 201
        assert r.json()["id"] == "sandbox"

    async def test_folder_route_uses_same_admin_gate(self, client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(client, token)
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
        )
        assert r.status_code == 201
        assert r.json()["id"] == "sandbox"


# ---------------------------------------------------------------------------
# PATCH /folders/{id}
# ---------------------------------------------------------------------------


class TestUpdateFolderGating:
    async def _provision_runtime_folder(self, client, rsa_keypair) -> None:
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(client, admin)
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox"},
        )
        assert r.status_code == 201

    async def test_no_token_returns_401(self, client, rsa_keypair):
        await self._provision_runtime_folder(client, rsa_keypair)
        # Re-issue without cookie:
        client.cookies.clear()
        r = await client.patch(
            "/folders/sandbox", json={"label": "x"}
        )
        assert r.status_code == 401

    async def test_non_admin_returns_403(self, client, rsa_keypair):
        await self._provision_runtime_folder(client, rsa_keypair)
        client.cookies.clear()
        token = _make_token(rsa_keypair, groups=["twin-contributor"])
        _set_idp_cookie(client, token)
        r = await client.patch(
            "/folders/sandbox",
            json={"label": "Renamed"},
        )
        assert r.status_code == 403

    async def test_admin_updates_label(self, client, rsa_keypair):
        await self._provision_runtime_folder(client, rsa_keypair)
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(client, admin)
        r = await client.patch(
            "/folders/sandbox",
            json={"label": "Sandbox v2"},
        )
        assert r.status_code == 200
        assert r.json()["kb"] == "Sandbox v2"


# ---------------------------------------------------------------------------
# DELETE /folders/{id}
# ---------------------------------------------------------------------------


class TestDeleteFolderGating:
    async def _provision_runtime_folder(self, client, rsa_keypair) -> None:
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(client, admin)
        r = await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox"},
        )
        assert r.status_code == 201

    async def test_no_token_returns_401(self, client, rsa_keypair):
        await self._provision_runtime_folder(client, rsa_keypair)
        client.cookies.clear()
        r = await client.delete("/folders/sandbox")
        assert r.status_code == 401

    async def test_non_admin_returns_403(self, client, rsa_keypair):
        await self._provision_runtime_folder(client, rsa_keypair)
        client.cookies.clear()
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        _set_idp_cookie(client, token)
        r = await client.delete("/folders/sandbox")
        assert r.status_code == 403

    async def test_admin_deletes_204(self, client, rsa_keypair):
        await self._provision_runtime_folder(client, rsa_keypair)
        admin = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(client, admin)
        r = await client.delete("/folders/sandbox")
        assert r.status_code == 204


# ---------------------------------------------------------------------------
# GET /folders — read stays open under active IdP
# ---------------------------------------------------------------------------


class TestListFoldersNoGate:
    async def test_get_works_without_admin_scope(self, client, rsa_keypair):
        # A pure reader can still enumerate folders (needed by the
        # folder-switcher dropdown). Auth is required (router-level
        # ``require_auth`` dep), but the admin gate isn't.
        token = _make_token(rsa_keypair, groups=["twin-reader"])
        _set_idp_cookie(client, token)
        r = await client.get("/folders")
        assert r.status_code == 200
        assert any(s["id"] == "default" for s in r.json())


# ---------------------------------------------------------------------------
# Custom admin_groups env: steward is no longer admin
# ---------------------------------------------------------------------------


class TestCustomAdminGroupsEnv:
    @pytest.fixture()
    async def custom_client(self, monkeypatch, tmp_path, fake_jwks):
        monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
        monkeypatch.setenv(
            "TWIN_FOLDERS_JSON",
            json.dumps([{"id": "default", "label": "Default", "kind": "primary"}]),
        )
        monkeypatch.setenv("TWIN_MAX_FOLDERS", "3")
        monkeypatch.setenv(
            "TWIN_FOLDERS_RUNTIME_FILE", str(tmp_path / "twin-folders.json")
        )
        folder_store.reset_runtime_store()
        webui_router.reset_store()
        # ``create_app`` calls ``configure_idp(None)`` (audit 2026-06-10
        # H1). Build the app first, then activate the custom IdP after.
        app = create_app()
        _activate_idp(fake_jwks, admin_groups=frozenset({"corp.kb-admin"}))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
        folder_store.reset_runtime_store()
        webui_router.reset_store()
        idp_jwt.configure_idp(None)

    async def test_steward_no_longer_admin(self, custom_client, rsa_keypair):
        token = _make_token(rsa_keypair, groups=["twin-steward"])
        _set_idp_cookie(custom_client, token)
        r = await custom_client.post(
            "/folders",
            json={"id": "sandbox", "label": "S"},
        )
        assert r.status_code == 403

    async def test_custom_admin_group_allowed(
        self, custom_client, rsa_keypair
    ):
        token = _make_token(rsa_keypair, groups=["corp.kb-admin"])
        _set_idp_cookie(custom_client, token)
        r = await custom_client.post(
            "/folders",
            json={"id": "sandbox", "label": "S"},
        )
        assert r.status_code == 201
