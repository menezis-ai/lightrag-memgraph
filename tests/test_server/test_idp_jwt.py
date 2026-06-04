"""Unit + middleware tests for the IdP JWT module.

We generate an RSA keypair in-test, plug a fake ``PyJWKClient`` into
``JwksCache`` via the ``fetcher`` injection point, and verify every
documented behaviour: claim mapping, palier derivation, cookie vs
bearer priority, 401 reasons, JWKS cache TTL.
"""

from __future__ import annotations

import json
import time
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse

from twindb_lightrag_memgraph.server import idp_jwt


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[Any, str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_key, private_pem, public_pem


@pytest.fixture()
def fake_jwks(rsa_keypair):
    """A ``signing_key_for(token)``-compatible stand-in for PyJWKClient."""

    _, _, public_pem = rsa_keypair

    class _FakeKey:
        key = public_pem

    class _FakeClient:
        def __init__(self, *_a, **_kw) -> None:
            self.calls = 0

        def get_signing_key_from_jwt(self, _token: str):
            self.calls += 1
            return _FakeKey()

    return _FakeClient()


def _make_token(rsa_keypair, **overrides) -> str:
    private_key, _, _ = rsa_keypair
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "https://idp.example/realms/twin",
        "aud": "twin",
        "sub": "user-42",
        "email": "claire@example.com",
        "name": "Claire B.",
        "groups": ["twin-steward"],
        "twin_spaces": ["default", "sandbox"],
        "scope": "read:documents write:documents",
        "iat": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    return pyjwt.encode(claims, private_key, algorithm="RS256")


def _config_for(jwks_client) -> idp_jwt.IdpConfig:
    return idp_jwt.IdpConfig(
        jwks_url="https://idp.example/jwks",
        issuer="https://idp.example/realms/twin",
        audience="twin",
    )


def _activate(config: idp_jwt.IdpConfig, jwks_client) -> None:
    """Reset module-level state to a known-good config + fake JWKS."""
    idp_jwt._active_config = config  # type: ignore[attr-defined]
    idp_jwt._active_cache = idp_jwt.JwksCache(  # type: ignore[attr-defined]
        config, fetcher=lambda _url: jwks_client
    )


@pytest.fixture(autouse=True)
def _reset_idp():
    yield
    idp_jwt.configure_idp(None)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestIdpConfigFromEnv:
    def test_returns_none_when_jwks_url_unset(self):
        assert idp_jwt.IdpConfig.from_env(env={}) is None

    def test_minimal_config(self):
        cfg = idp_jwt.IdpConfig.from_env(env={"TWIN_IDP_JWKS_URL": "https://idp/jwks"})
        assert cfg is not None
        assert cfg.jwks_url == "https://idp/jwks"
        assert cfg.algorithms == ("RS256",)
        assert cfg.cookie_name == "twin_idp_token"
        # built-in palier mapping
        assert cfg.group_to_palier["twin-steward"] == 3

    def test_overrides_algorithms_and_cookie(self):
        cfg = idp_jwt.IdpConfig.from_env(
            env={
                "TWIN_IDP_JWKS_URL": "https://idp/jwks",
                "TWIN_IDP_ALGORITHMS": "RS256, ES256",
                "TWIN_IDP_COOKIE_NAME": "myaccess_session",
                "TWIN_IDP_NAME": "myaccess",
                "TWIN_IDP_REALM": "bnp-cib",
            }
        )
        assert cfg is not None
        assert cfg.algorithms == ("RS256", "ES256")
        assert cfg.cookie_name == "myaccess_session"
        assert cfg.idp_name == "myaccess"
        assert cfg.idp_realm == "bnp-cib"

    def test_custom_group_palier_map(self):
        cfg = idp_jwt.IdpConfig.from_env(
            env={
                "TWIN_IDP_JWKS_URL": "https://idp/jwks",
                "TWIN_IDP_GROUP_TO_PALIER_JSON": json.dumps(
                    {"bnp.cib.steward": 3, "bnp.cib.viewer": 1}
                ),
            }
        )
        assert cfg is not None
        assert cfg.group_to_palier == {
            "bnp.cib.steward": 3,
            "bnp.cib.viewer": 1,
        }

    def test_invalid_group_palier_json_falls_back_to_default(self):
        cfg = idp_jwt.IdpConfig.from_env(
            env={
                "TWIN_IDP_JWKS_URL": "https://idp/jwks",
                "TWIN_IDP_GROUP_TO_PALIER_JSON": "not json",
            }
        )
        assert cfg is not None
        assert cfg.group_to_palier == {
            "twin-steward": 3,
            "twin-contributor": 2,
            "twin-reader": 1,
        }


# ---------------------------------------------------------------------------
# Claims → AuthenticatedUser
# ---------------------------------------------------------------------------


class TestClaimsMapping:
    def _cfg(self) -> idp_jwt.IdpConfig:
        return idp_jwt.IdpConfig(jwks_url="https://idp/jwks")

    def test_full_projection(self):
        cfg = self._cfg()
        user = idp_jwt.claims_to_user(
            {
                "sub": "user-42",
                "email": "claire@example.com",
                "name": "Claire B.",
                "groups": ["twin-steward"],
                "twin_spaces": ["default", "sandbox"],
                "scope": "read:documents write:documents",
                "exp": 4102444800,  # 2100-01-01
            },
            cfg,
        )
        assert user["sso_subject"] == "user-42"
        assert user["email"] == "claire@example.com"
        assert user["name"] == "Claire B."
        assert user["palier"] == {
            "level": 3,
            "label": "Steward",
            "scopes": ["twin:read", "twin:write", "twin:approve"],
        }
        assert user["workspaces"] == ["default", "sandbox"]
        assert user["gateway_scopes"] == [
            "read:documents",
            "write:documents",
        ]
        assert user["session_expires"].startswith("2100-01-01T")

    def test_no_groups_defaults_to_reader(self):
        cfg = self._cfg()
        user = idp_jwt.claims_to_user({"sub": "u", "groups": []}, cfg)
        assert user["palier"]["level"] == 1
        assert user["palier"]["label"] == "Reader"

    def test_highest_group_wins(self):
        cfg = self._cfg()
        user = idp_jwt.claims_to_user(
            {"sub": "u", "groups": ["twin-reader", "twin-steward"]}, cfg
        )
        assert user["palier"]["level"] == 3

    def test_string_scope_split_on_space(self):
        cfg = self._cfg()
        user = idp_jwt.claims_to_user(
            {"sub": "u", "scope": "a b c"}, cfg
        )
        assert user["gateway_scopes"] == ["a", "b", "c"]

    def test_groups_supports_string_list(self):
        cfg = self._cfg()
        user = idp_jwt.claims_to_user(
            {"sub": "u", "groups": "twin-contributor"}, cfg
        )
        assert user["palier"]["level"] == 2

    def test_falls_back_through_email_then_subject(self):
        cfg = self._cfg()
        user = idp_jwt.claims_to_user(
            {"sub": "u-99", "email": "a@b"}, cfg
        )
        assert user["sso_subject"] == "u-99"
        assert user["name"] == "a@b"  # name → email fallback


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def _stub_request(*, cookies=None, headers=None) -> Request:
    """Build a starlette ``Request`` with the given cookies/headers."""
    cookies = cookies or {}
    headers = headers or {}
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    if cookie_header:
        raw_headers.append((b"cookie", cookie_header.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers,
        "query_string": b"",
    }
    return Request(scope)


class TestExtractBearer:
    def test_returns_cookie_value_first(self):
        cfg = idp_jwt.IdpConfig(jwks_url="https://idp/jwks")
        req = _stub_request(
            cookies={"twin_idp_token": "from-cookie"},
            headers={"authorization": "Bearer from-header"},
        )
        assert idp_jwt.extract_bearer_token(req, cfg) == "from-cookie"

    def test_returns_bearer_when_no_cookie(self):
        cfg = idp_jwt.IdpConfig(jwks_url="https://idp/jwks")
        req = _stub_request(headers={"authorization": "Bearer t"})
        assert idp_jwt.extract_bearer_token(req, cfg) == "t"

    def test_returns_none_when_neither(self):
        cfg = idp_jwt.IdpConfig(jwks_url="https://idp/jwks")
        assert idp_jwt.extract_bearer_token(_stub_request(), cfg) is None

    def test_ignores_non_bearer_authorization(self):
        cfg = idp_jwt.IdpConfig(jwks_url="https://idp/jwks")
        req = _stub_request(headers={"authorization": "Basic dXNlcjpwd2Q="})
        assert idp_jwt.extract_bearer_token(req, cfg) is None


# ---------------------------------------------------------------------------
# Decode + verify
# ---------------------------------------------------------------------------


class TestDecode:
    def test_valid_token(self, rsa_keypair, fake_jwks):
        cfg = _config_for(fake_jwks)
        cache = idp_jwt.JwksCache(cfg, fetcher=lambda _u: fake_jwks)
        token = _make_token(rsa_keypair)
        claims = idp_jwt.decode_idp_token(token, cfg, cache)
        assert claims["sub"] == "user-42"

    def test_expired_token_raises_with_expired_marker(
        self, rsa_keypair, fake_jwks
    ):
        cfg = _config_for(fake_jwks)
        cache = idp_jwt.JwksCache(cfg, fetcher=lambda _u: fake_jwks)
        now = int(time.time())
        token = _make_token(rsa_keypair, iat=now - 1200, exp=now - 600)
        with pytest.raises(idp_jwt.IdpAuthError) as exc:
            idp_jwt.decode_idp_token(token, cfg, cache)
        assert exc.value.status_code == 401
        assert 'error="expired"' in exc.value.headers["WWW-Authenticate"]

    def test_wrong_audience(self, rsa_keypair, fake_jwks):
        cfg = _config_for(fake_jwks)
        cache = idp_jwt.JwksCache(cfg, fetcher=lambda _u: fake_jwks)
        token = _make_token(rsa_keypair, aud="other-twin")
        with pytest.raises(idp_jwt.IdpAuthError) as exc:
            idp_jwt.decode_idp_token(token, cfg, cache)
        assert exc.value.status_code == 401
        assert "Wrong audience" in exc.value.detail

    def test_wrong_issuer(self, rsa_keypair, fake_jwks):
        cfg = _config_for(fake_jwks)
        cache = idp_jwt.JwksCache(cfg, fetcher=lambda _u: fake_jwks)
        token = _make_token(rsa_keypair, iss="https://evil/idp")
        with pytest.raises(idp_jwt.IdpAuthError) as exc:
            idp_jwt.decode_idp_token(token, cfg, cache)
        assert "Wrong issuer" in exc.value.detail

    def test_audience_skipped_when_unset(self, rsa_keypair, fake_jwks):
        cfg = idp_jwt.IdpConfig(
            jwks_url="https://idp/jwks",
            issuer="https://idp.example/realms/twin",
            audience=None,
        )
        cache = idp_jwt.JwksCache(cfg, fetcher=lambda _u: fake_jwks)
        # Token has aud="twin" but config doesn't enforce it
        token = _make_token(rsa_keypair, aud="anything")
        claims = idp_jwt.decode_idp_token(token, cfg, cache)
        assert claims["sub"] == "user-42"


# ---------------------------------------------------------------------------
# JWKS cache
# ---------------------------------------------------------------------------


class TestJwksCache:
    def test_lazy_initial_fetch(self, fake_jwks):
        cfg = _config_for(fake_jwks)
        calls = []

        def fetcher(url):
            calls.append(url)
            return fake_jwks

        cache = idp_jwt.JwksCache(cfg, fetcher=fetcher)
        # No fetch until first lookup
        assert calls == []
        cache.signing_key_for("doesnt-matter")
        assert calls == ["https://idp.example/jwks"]

    def test_no_refetch_within_ttl(self, fake_jwks):
        cfg = idp_jwt.IdpConfig(
            jwks_url="https://idp/jwks", jwks_cache_ttl=600
        )
        calls = []

        def fetcher(url):
            calls.append(url)
            return fake_jwks

        cache = idp_jwt.JwksCache(cfg, fetcher=fetcher)
        cache.signing_key_for("t1")
        cache.signing_key_for("t2")
        cache.signing_key_for("t3")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/me")
    async def me(request: Request):
        user = idp_jwt.require_idp_user(request)
        if user is None:
            return JSONResponse({"identity": None})
        return JSONResponse({"user": user})

    return app


class TestRequireIdpUser:
    async def test_dormant_returns_none(self):
        idp_jwt.configure_idp(None)
        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/me")
        assert r.json() == {"identity": None}

    async def test_valid_cookie_returns_user(self, rsa_keypair, fake_jwks):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        token = _make_token(rsa_keypair)
        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies={"twin_idp_token": token},
        ) as c:
            r = await c.get("/me")
        assert r.status_code == 200
        body = r.json()
        assert body["user"]["sso_subject"] == "user-42"
        assert body["user"]["palier"]["label"] == "Steward"

    async def test_valid_bearer_returns_user(self, rsa_keypair, fake_jwks):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        token = _make_token(rsa_keypair)
        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as c:
            r = await c.get("/me")
        assert r.status_code == 200
        assert r.json()["user"]["sso_subject"] == "user-42"

    async def test_missing_token_401(self, fake_jwks):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/me")
        assert r.status_code == 401
        assert 'error="missing_token"' in r.headers["www-authenticate"]

    async def test_expired_token_401_with_marker(
        self, rsa_keypair, fake_jwks
    ):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        now = int(time.time())
        token = _make_token(rsa_keypair, iat=now - 1200, exp=now - 600)
        app = _build_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies={"twin_idp_token": token},
        ) as c:
            r = await c.get("/me")
        assert r.status_code == 401
        assert 'error="expired"' in r.headers["www-authenticate"]


# ---------------------------------------------------------------------------
# Wiring in auth.require_auth
# ---------------------------------------------------------------------------


class TestRequireAuthIntegration:
    async def test_idp_jwt_authenticates_when_legacy_auth_disabled(
        self, rsa_keypair, fake_jwks
    ):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        from twindb_lightrag_memgraph.server.auth import (
            configure_auth,
            require_auth,
        )

        configure_auth()  # no api key, no jwt secret
        token = _make_token(rsa_keypair)
        app = FastAPI()

        @app.get("/who")
        async def who(identity=__import__("fastapi").Depends(require_auth)):
            return {"identity": identity}

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies={"twin_idp_token": token},
        ) as c:
            r = await c.get("/who")
        assert r.status_code == 200
        assert r.json()["identity"] == "user-42"

    async def test_idp_only_missing_token_rejects_anonymous_request(self, fake_jwks):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        from twindb_lightrag_memgraph.server.auth import (
            configure_auth,
            require_auth,
        )

        configure_auth()  # no api key, no jwt secret
        app = FastAPI()

        @app.get("/who")
        async def who(identity=__import__("fastapi").Depends(require_auth)):
            return {"identity": identity}

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.get("/who")
        assert r.status_code == 401
        assert 'error="missing_token"' in r.headers["www-authenticate"]

    async def test_idp_only_non_jwt_bearer_rejects_request(self, fake_jwks):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        from twindb_lightrag_memgraph.server.auth import (
            configure_auth,
            require_auth,
        )

        configure_auth()  # no api key, no jwt secret
        app = FastAPI()

        @app.get("/who")
        async def who(identity=__import__("fastapi").Depends(require_auth)):
            return {"identity": identity}

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer not-a-jwt"},
        ) as c:
            r = await c.get("/who")
        assert r.status_code == 401

    async def test_idp_invalid_token_rejects_request_even_with_static_key(
        self, rsa_keypair, fake_jwks
    ):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        from twindb_lightrag_memgraph.server.auth import (
            configure_auth,
            require_auth,
        )

        configure_auth(api_key="secret-key")
        now = int(time.time())
        expired = _make_token(rsa_keypair, iat=now - 1200, exp=now - 600)
        app = FastAPI()

        @app.get("/who")
        async def who(identity=__import__("fastapi").Depends(require_auth)):
            return {"identity": identity}

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies={"twin_idp_token": expired},
            headers={"Authorization": "Bearer secret-key"},
        ) as c:
            r = await c.get("/who")
        # The IdP cookie is present → must validate or 401. The
        # legacy static-key bearer is NOT a silent escape hatch.
        assert r.status_code == 401
        assert 'error="expired"' in r.headers["www-authenticate"]

    async def test_static_key_still_works_when_no_idp_cookie(
        self, fake_jwks
    ):
        cfg = _config_for(fake_jwks)
        _activate(cfg, fake_jwks)
        from twindb_lightrag_memgraph.server.auth import (
            configure_auth,
            require_auth,
        )

        configure_auth(api_key="secret-key")
        app = FastAPI()

        @app.get("/who")
        async def who(identity=__import__("fastapi").Depends(require_auth)):
            return {"identity": identity}

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer secret-key"},
        ) as c:
            r = await c.get("/who")
        assert r.status_code == 200
        assert r.json()["identity"] == "api_key"


# ---------------------------------------------------------------------------
# Runtime config stripping
# ---------------------------------------------------------------------------


class TestRuntimeConfigDebugUserStripped:
    def test_debug_user_present_when_idp_dormant(self, monkeypatch):
        for key in (
            "TWIN_IDP_JWKS_URL",
            "TWIN_IDP_ISSUER",
            "TWIN_IDP_AUDIENCE",
        ):
            monkeypatch.delenv(key, raising=False)

        from twindb_lightrag_memgraph import _build_runtime_config

        cfg = _build_runtime_config()
        assert "debugUser" in cfg

    def test_debug_user_stripped_when_idp_active(self, monkeypatch):
        monkeypatch.setenv("TWIN_IDP_JWKS_URL", "https://idp/jwks")
        from twindb_lightrag_memgraph import _build_runtime_config

        cfg = _build_runtime_config()
        assert "debugUser" not in cfg
