"""Admin gate coverage for tag and graph governance mutations."""

from __future__ import annotations

import json
import time
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import folder_store, idp_jwt, webui_router
from twindb_lightrag_memgraph.server.app import create_app


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


def _activate_idp(fake_jwks) -> None:
    cfg = idp_jwt.IdpConfig(
        jwks_url="https://idp.example/jwks",
        issuer="https://idp.example/realms/twin",
        audience="twin",
        admin_groups=frozenset({"twin-admin", "twin-steward"}),
    )
    idp_jwt._active_config = cfg  # type: ignore[attr-defined]
    idp_jwt._active_cache = idp_jwt.JwksCache(  # type: ignore[attr-defined]
        cfg, fetcher=lambda _url: fake_jwks
    )


@pytest.fixture()
async def non_admin_client(monkeypatch, tmp_path, fake_jwks, rsa_keypair):
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

    app = create_app()
    _activate_idp(fake_jwks)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.cookies.set(
            "twin_idp_token",
            _make_token(rsa_keypair, groups=["twin-reader"]),
        )
        yield client

    folder_store.reset_runtime_store()
    webui_router.reset_store()
    idp_jwt.configure_idp(None)


GOVERNANCE_MUTATIONS: list[tuple[str, str, Any]] = [
    (
        "POST",
        "/twin/api/tags/categories/_import",
        [{"id": "risk", "label": "Risk", "color": "#123456"}],
    ),
    (
        "POST",
        "/twin/api/documents/_bulk-retag",
        {"targets": ["doc-1"], "adds": ["risk"], "removes": []},
    ),
    (
        "POST",
        "/twin/api/tags",
        {"tag": "risk", "def": "Risk marker", "category": "governance"},
    ),
    ("POST", "/twin/api/tags/rman/approve", {"actor": "reader"}),
    ("POST", "/twin/api/tags/rman/reject", {"reason": "duplicate"}),
    ("POST", "/twin/api/tags/rman/suggest-edit", {"def": "Updated"}),
    ("PATCH", "/twin/api/tags/rman", {"def": "Updated"}),
    ("POST", "/twin/api/tags/rman/deprecate", {"reason": "stale"}),
    ("POST", "/twin/api/tags/rman/reactivate", {"actor": "reader"}),
    ("POST", "/twin/api/tags/rman/synonyms", {"aliases": ["backup"]}),
    ("DELETE", "/twin/api/tags/rman", {"strategy": "untag"}),
    (
        "POST",
        "/twin/api/graph/entities",
        {"name": "Manual Entity", "type": "PRODUCT"},
    ),
    ("PATCH", "/twin/api/graph/entities/kg_manual", {"summary": "Updated"}),
    ("DELETE", "/twin/api/graph/entities/kg_manual", None),
    (
        "POST",
        "/twin/api/graph/relations",
        {"source": "kg_a", "target": "kg_b", "label": "USES"},
    ),
    ("PATCH", "/twin/api/graph/relations/kr_manual", {"strength": 0.9}),
    ("DELETE", "/twin/api/graph/relations/kr_manual", None),
]


@pytest.mark.parametrize(("method", "path", "payload"), GOVERNANCE_MUTATIONS)
async def test_governance_mutations_require_admin_scope(
    non_admin_client: AsyncClient,
    method: str,
    path: str,
    payload: Any,
):
    response = await non_admin_client.request(method, path, json=payload)

    assert response.status_code == 403
    assert idp_jwt.ADMIN_FOLDERS_SCOPE in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    [
        "/twin/api/tags",
        "/twin/api/tags/categories",
        "/twin/api/graph/entities",
        "/twin/api/graph/relations",
    ],
)
async def test_governance_reads_do_not_require_admin_scope(
    non_admin_client: AsyncClient,
    path: str,
):
    response = await non_admin_client.get(path)

    assert response.status_code == 200
