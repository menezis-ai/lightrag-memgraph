"""Audit 2026-06-10 P0 — ``require_admin_user`` two-tier behaviour (H4).

The previous version of ``require_admin_user`` returned ``None`` when
the IdP middleware was dormant, which meant any caller — including
unauthenticated ones — could hit folder CRUD routes. The fix:

* **Palier 1 — IdP dormant**: returns a placeholder user dict
  (``idp_validated=False``). The route-level ``require_auth`` has
  already filtered anonymous, so a real identity is required to reach
  this branch. A single boot warning + a single INFO log on the first
  admin hit document the "no RBAC yet" posture.
* **Palier 2 — IdP active**: requires the ``admin:folders`` gateway
  scope (already projected by ``claims_to_user``). 401 on missing
  token, 403 on scope-missing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from twindb_lightrag_memgraph.server import idp_jwt
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
    yield
    configure_idp(None)


# ---------------------------------------------------------------------------
# Palier 1 — IdP dormant
# ---------------------------------------------------------------------------


def test_palier1_dormant_returns_placeholder(caplog):
    caplog.set_level("WARNING")
    configure_idp(None)
    request = _make_request()
    user = require_admin_user(request)
    assert user is not None
    assert user["idp_validated"] is False
    assert user["gateway_scopes"] == []
    # The dormant warning should fire from configure_idp.
    assert any("palier 1" in r.getMessage() for r in caplog.records)


def test_palier1_dormant_logs_info_once_per_process(caplog):
    """The per-call INFO log is rate-limited to once per (configure_idp,
    require_admin_user) cycle to avoid log flooding."""
    caplog.set_level("INFO")
    configure_idp(None)
    request = _make_request()

    require_admin_user(request)
    require_admin_user(request)
    require_admin_user(request)

    info_hits = [
        r
        for r in caplog.records
        if r.levelname == "INFO" and "palier 1" in r.getMessage()
    ]
    assert len(info_hits) == 1


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
