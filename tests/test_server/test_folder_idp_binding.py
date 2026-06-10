"""Audit 2026-06-10 P0 — bind X-Twin-Folder to identity (C2).

Two-tier behaviour:

* **Palier 1 — IdP dormant**: ``resolve_folder_for_request`` reproduces
  the pure header/catalog logic (back-compat for OVH standalone, dev,
  smoke runner). No identity check.
* **Palier 2 — IdP active**: the resolved folder must be in the
  caller's ``twin_folders`` claim. Empty claim → only default folder
  reachable (user-chosen fallback for the MyAccess rollout period).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi import HTTPException, Request

from twindb_lightrag_memgraph.server import idp_jwt
from twindb_lightrag_memgraph.server.folder import resolve_folder_for_request
from twindb_lightrag_memgraph.server.idp_jwt import IdpConfig, configure_idp


def _make_request(folder_header: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if folder_header is not None:
        headers.append((b"x-twin-folder", folder_header.encode()))
    scope: dict[str, Any] = {
        "type": "http",
        "headers": headers,
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Pin the folder catalog so the tests don't depend on host env."""
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"kb"},'
        '{"id":"alpha","label":"Alpha","kind":"kb"},'
        '{"id":"beta","label":"Beta","kind":"kb"}]',
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "5")
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    configure_idp(None)
    yield
    configure_idp(None)


# ---------------------------------------------------------------------------
# Palier 1 — IdP dormant
# ---------------------------------------------------------------------------


def test_palier1_default_folder_when_header_absent():
    assert resolve_folder_for_request(_make_request()) == "default"


def test_palier1_known_folder_accepted():
    assert resolve_folder_for_request(_make_request("alpha")) == "alpha"


def test_palier1_unknown_folder_rejected():
    with pytest.raises(HTTPException) as exc:
        resolve_folder_for_request(_make_request("ghost"))
    assert exc.value.status_code == 403


def test_palier1_invalid_identifier_rejected():
    with pytest.raises(HTTPException) as exc:
        resolve_folder_for_request(_make_request("not a valid id"))
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Palier 2 — IdP active
# ---------------------------------------------------------------------------


def _active_config() -> IdpConfig:
    return IdpConfig(
        jwks_url="https://idp.test/.well-known/jwks.json",
        issuer="https://idp.test",
        audience="twin",
    )


def test_palier2_folder_in_user_scope_returned(monkeypatch):
    configure_idp(_active_config())
    monkeypatch.setattr(
        idp_jwt,
        "require_idp_user",
        lambda req: {"folders": ["alpha", "beta"]},
    )
    assert resolve_folder_for_request(_make_request("alpha")) == "alpha"


def test_palier2_folder_out_of_user_scope_403(monkeypatch):
    configure_idp(_active_config())
    monkeypatch.setattr(
        idp_jwt,
        "require_idp_user",
        lambda req: {"folders": ["alpha"]},
    )
    with pytest.raises(HTTPException) as exc:
        resolve_folder_for_request(_make_request("beta"))
    assert exc.value.status_code == 403
    assert "not in user scope" in exc.value.detail


def test_palier2_empty_claim_allows_only_default(monkeypatch):
    """User-chosen fallback: empty user.folders → only the catalog
    default folder is reachable. Covers the MyAccess rollout window
    where the ``twin_folders`` claim isn't yet emitted."""
    configure_idp(_active_config())
    monkeypatch.setattr(idp_jwt, "require_idp_user", lambda req: {"folders": []})

    # Default folder OK.
    assert resolve_folder_for_request(_make_request()) == "default"
    assert resolve_folder_for_request(_make_request("default")) == "default"

    # Any non-default folder → 403.
    with pytest.raises(HTTPException) as exc:
        resolve_folder_for_request(_make_request("alpha"))
    assert exc.value.status_code == 403


def test_palier2_claim_missing_key_treated_as_empty(monkeypatch):
    configure_idp(_active_config())
    monkeypatch.setattr(idp_jwt, "require_idp_user", lambda req: {})
    assert resolve_folder_for_request(_make_request()) == "default"
    with pytest.raises(HTTPException) as exc:
        resolve_folder_for_request(_make_request("alpha"))
    assert exc.value.status_code == 403
