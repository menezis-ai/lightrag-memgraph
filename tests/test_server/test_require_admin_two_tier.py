"""Regression coverage for ``require_admin_user`` two-tier behaviour.

Without an IdP only the infrastructure root key is authoritative. Legacy JWTs
have identity but no RBAC claims and must not inherit administrator power.
With an IdP, the verified ``admin:folders`` scope remains mandatory.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from twindb_lightrag_memgraph.server import idp_jwt
from twindb_lightrag_memgraph.server.auth import _create_jwt, configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import (
    ADMIN_FOLDERS_SCOPE,
    IdpConfig,
    configure_idp,
    require_admin_user,
)


def _make_request(
    cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> Request:
    """Build a starlette Request stub with cookies + headers."""
    scope: dict[str, Any] = {
        "type": "http",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
    }
    request = Request(scope)
    # Override cookies (starlette parses lazily from headers; easier to
    # mock the property directly).
    if cookies is not None:
        object.__setattr__(request, "_cookies", cookies)

        class _CookieDict(dict):
            pass

        request.scope.setdefault("type", "http")
    return request


@pytest.fixture(autouse=True)
def _reset_idp_state():
    configure_idp(None)
    configure_auth()
    yield
    configure_idp(None)
    configure_auth()


# ---------------------------------------------------------------------------
# IdP dormant
# ---------------------------------------------------------------------------


def test_dormant_allows_only_infrastructure_root_key():
    configure_auth(api_key="infra-root")
    request = _make_request(headers={"Authorization": "Bearer infra-root"})
    user = require_admin_user(request)
    assert user["sso_subject"] == "api_key"
    assert user["idp_validated"] is False
    assert user["gateway_scopes"] == [ADMIN_FOLDERS_SCOPE]


def test_dormant_allows_native_x_api_key_transport():
    configure_auth(api_key="infra-root")
    request = _make_request(headers={"X-API-Key": "infra-root"})

    user = require_admin_user(request)

    assert user["sso_subject"] == "api_key"
    assert user["gateway_scopes"] == [ADMIN_FOLDERS_SCOPE]


def test_dormant_rejects_legacy_jwt_without_rbac_claims():
    configure_auth(jwt_secret="legacy-secret", jwt_password="not-default")
    token = _create_jwt({"sub": "legacy-reader"})
    request = _make_request(headers={"Authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc:
        require_admin_user(request)
    assert exc.value.status_code == 403
    assert "infrastructure root key" in exc.value.detail


def test_dormant_rejects_legacy_jwt_subject_named_api_key():
    """An audit-label collision must never become a root capability."""
    configure_auth(jwt_secret="legacy-secret", jwt_password="not-default")
    token = _create_jwt({"sub": "api_key"})
    request = _make_request(headers={"Authorization": f"Bearer {token}"})

    with pytest.raises(HTTPException) as exc:
        require_admin_user(request)

    assert exc.value.status_code == 403
    assert "infrastructure root key" in exc.value.detail


def test_dormant_rejects_request_without_root_credentials():
    with pytest.raises(HTTPException) as exc:
        require_admin_user(_make_request())
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# Palier 2 — IdP active
# ---------------------------------------------------------------------------


def _active_config() -> IdpConfig:
    return IdpConfig(
        jwks_url="https://idp.test/.well-known/jwks.json",
        issuer="https://idp.test",
        audience="twin",
        algorithms=("RS256",),
        admin_groups=frozenset({"twin-admin"}),
    )


def test_palier2_active_no_token_raises_401(monkeypatch):
    configure_idp(_active_config())
    request = _make_request()

    with pytest.raises(HTTPException) as exc:
        require_admin_user(request)
    assert exc.value.status_code == 401


def test_palier2_active_without_scope_raises_403(monkeypatch):
    """User authenticated but missing admin:folders scope → 403."""
    configure_idp(_active_config())

    monkeypatch.setattr(
        idp_jwt,
        "require_idp_user",
        lambda req: {
            "sso_subject": "alice",
            "gateway_scopes": ["twin:read"],
            "folders": ["default"],
        },
    )

    with pytest.raises(HTTPException) as exc:
        require_admin_user(_make_request())
    assert exc.value.status_code == 403
    assert ADMIN_FOLDERS_SCOPE in exc.value.detail


def test_palier2_active_with_scope_returns_user(monkeypatch):
    configure_idp(_active_config())

    expected = {
        "sso_subject": "alice",
        "gateway_scopes": ["twin:read", ADMIN_FOLDERS_SCOPE],
        "folders": ["default"],
    }
    monkeypatch.setattr(idp_jwt, "require_idp_user", lambda req: expected)

    result = require_admin_user(_make_request())
    assert result is expected
