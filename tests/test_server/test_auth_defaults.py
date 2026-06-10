"""LightRAG-parity auth defaults (product decision 2026-06-10).

Supersedes the audit-P0 fail-closed posture from the same morning:
BNP deployments crash-looped on the boot refusal (H1) and the
unconditional ``changeme`` rejection (H2) because LightRAG-native env
sets don't carry ``LIGHTRAG_JWT_PASSWORD`` / ``LIGHTRAG_API_KEY``.

New contract — identical to LightRAG native:

* No backend configured → server boots, ``require_auth`` allows
  anonymous (returns ``None``), one loud WARNING in the logs.
* ``changeme`` (default password) → accepted, loud SECURITY warning,
  never a raise.
* IdP path is untouched: with ``TWIN_IDP_JWKS_URL`` set the middleware
  still fails closed (palier 2) — that posture only activates by
  explicit env opt-in.
"""

from __future__ import annotations

import logging

import pytest

from twindb_lightrag_memgraph.server import auth as auth_module
from twindb_lightrag_memgraph.server.auth import (
    DEFAULT_JWT_PASSWORD,
    configure_auth,
    require_auth,
)
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp


@pytest.fixture(autouse=True)
def _reset_state():
    configure_idp(None)
    yield
    configure_idp(None)
    configure_auth(api_key=None, jwt_secret=None)


# ---------------------------------------------------------------------------
# Open access when nothing is configured (LightRAG parity)
# ---------------------------------------------------------------------------


async def test_no_backend_allows_anonymous():
    configure_auth(api_key=None, jwt_secret=None)
    assert await require_auth(credentials=None) is None


def test_no_backend_logs_warning(caplog):
    caplog.set_level(logging.WARNING)
    configure_auth(api_key=None, jwt_secret=None)
    assert any("auth DISABLED" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 'changeme' tolerated with a SECURITY warning, never a raise
# ---------------------------------------------------------------------------


def test_changeme_jwt_password_warns_but_boots(caplog):
    caplog.set_level(logging.WARNING)
    configure_auth(jwt_secret="some-secret", jwt_password=DEFAULT_JWT_PASSWORD)
    assert auth_module._auth_enabled is True
    assert any(
        "SECURITY" in r.getMessage() and "changeme" in r.getMessage()
        for r in caplog.records
    )


def test_changeme_in_auth_accounts_warns_but_boots(caplog):
    caplog.set_level(logging.WARNING)
    configure_auth(
        jwt_secret="some-secret",
        jwt_password="real-password",
        auth_accounts={"alice": DEFAULT_JWT_PASSWORD, "bob": "ok"},
    )
    assert auth_module._auth_enabled is True
    warning = next(
        r.getMessage()
        for r in caplog.records
        if "SECURITY" in r.getMessage() and "AUTH_ACCOUNTS" in r.getMessage()
    )
    assert "alice" in warning
    assert "bob" not in warning


def test_non_default_password_no_security_warning(caplog):
    caplog.set_level(logging.WARNING)
    configure_auth(jwt_secret="some-secret", jwt_password="real-password")
    assert not any("SECURITY" in r.getMessage() for r in caplog.records)


def test_no_jwt_secret_means_password_not_checked(caplog):
    caplog.set_level(logging.WARNING)
    configure_auth(
        api_key="some-key",
        jwt_secret=None,
        jwt_password=DEFAULT_JWT_PASSWORD,
    )
    assert auth_module._jwt_secret is None
    assert not any("SECURITY" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Configured backends still gate anonymous (unchanged behaviour)
# ---------------------------------------------------------------------------


async def test_api_key_configured_rejects_anonymous():
    from fastapi import HTTPException

    configure_auth(api_key="some-key")
    with pytest.raises(HTTPException) as exc:
        await require_auth(credentials=None)
    assert exc.value.status_code == 401
