"""Audit 2026-06-10 P0 — boot-time fail-closed posture (H1 + H2).

Two checks:

* H1 — ``ensure_auth_backend_configured`` refuses to start when neither
  ``LIGHTRAG_API_KEY``, ``LIGHTRAG_JWT_SECRET``/``TOKEN_SECRET``, nor
  ``TWIN_IDP_JWKS_URL`` is set, unless ``TWIN_ALLOW_OPEN_ACCESS=1`` is
  explicitly opted into. LightRAG natively boots wide open; Twin
  refuses that posture by default.
* H2 — ``configure_auth`` refuses the literal ``"changeme"`` default
  password unconditionally — both as ``LIGHTRAG_JWT_PASSWORD`` and as
  any value in ``AUTH_ACCOUNTS``.

These tests run with the ``no_default_auth`` marker so the conftest's
default ``LIGHTRAG_API_KEY`` injection is skipped.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server import auth as auth_module
from twindb_lightrag_memgraph.server.auth import (
    DEFAULT_JWT_PASSWORD,
    configure_auth,
    ensure_auth_backend_configured,
)


pytestmark = pytest.mark.no_default_auth


# ---------------------------------------------------------------------------
# H1 — boot refuses without auth backend
# ---------------------------------------------------------------------------


def test_h1_boot_refuses_when_no_backend_configured():
    with pytest.raises(RuntimeError, match="no auth backend configured"):
        ensure_auth_backend_configured(
            api_key=None,
            jwt_secret=None,
            idp_configured=False,
            allow_open_access=False,
        )


def test_h1_boot_passes_with_api_key():
    ensure_auth_backend_configured(
        api_key="some-key",
        jwt_secret=None,
        idp_configured=False,
        allow_open_access=False,
    )


def test_h1_boot_passes_with_jwt_secret():
    ensure_auth_backend_configured(
        api_key=None,
        jwt_secret="some-secret",
        idp_configured=False,
        allow_open_access=False,
    )


def test_h1_boot_passes_with_idp_configured():
    ensure_auth_backend_configured(
        api_key=None,
        jwt_secret=None,
        idp_configured=True,
        allow_open_access=False,
    )


def test_h1_open_access_opt_in_allows_boot(caplog):
    caplog.set_level("WARNING")
    ensure_auth_backend_configured(
        api_key=None,
        jwt_secret=None,
        idp_configured=False,
        allow_open_access=True,
    )
    assert any(
        "WIDE OPEN" in record.getMessage() for record in caplog.records
    )


# ---------------------------------------------------------------------------
# H2 — 'changeme' refused unconditionally
# ---------------------------------------------------------------------------


def test_h2_changeme_jwt_password_refused():
    with pytest.raises(ValueError, match="changeme"):
        configure_auth(
            jwt_secret="some-secret",
            jwt_password=DEFAULT_JWT_PASSWORD,
        )


def test_h2_changeme_jwt_password_refused_even_when_accounts_configured():
    """Audit 2026-06-10 H2: previous version skipped the ``changeme`` check
    when ``AUTH_ACCOUNTS`` was non-empty."""
    with pytest.raises(ValueError, match="changeme"):
        configure_auth(
            jwt_secret="some-secret",
            jwt_password=DEFAULT_JWT_PASSWORD,
            auth_accounts={"alice": "alice-pwd"},
        )


def test_h2_changeme_in_auth_accounts_refused():
    with pytest.raises(ValueError, match="changeme"):
        configure_auth(
            jwt_secret="some-secret",
            jwt_password="not-the-default",
            auth_accounts={"alice": DEFAULT_JWT_PASSWORD},
        )


def test_h2_no_jwt_secret_means_password_not_checked():
    """No /login endpoint => password value irrelevant."""
    configure_auth(
        api_key="some-key",
        jwt_secret=None,
        jwt_password=DEFAULT_JWT_PASSWORD,
    )
    assert auth_module._jwt_secret is None
