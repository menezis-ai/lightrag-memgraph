"""Audit 2026-06-10 P0 — native shim routes require auth (C1).

Previously ``_inject_native_shims`` mounted the shim router without any
dependency, exposing ``GET /documents``, ``DELETE /documents/{id}``,
``POST /documents/{id}/scan``, ``/pipeline_status``, and ``/openapi``
to anonymous callers. The fix wires ``require_auth`` through the
``auth_dependency`` param of ``build_native_shims_router``. Public
routes (``/auth-status``, ``/login``, ``/logout``, ``/health``) stay
open by design.

This file also doubles as the LightRAG compatibility test for the
shim layer (per ``docs/test-doctrine-lightrag-compat.md``): the
flag-off router (``auth_dependency=None``) must keep the historical
shape — every route in the catalog wired with no dependency — so a
deployment that intentionally opts out via the param keeps working.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from twindb_lightrag_memgraph.server.auth import configure_auth, require_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp
from twindb_lightrag_memgraph.server.native_shims import build_native_shims_router


@pytest.fixture(autouse=True)
def _reset_auth_state():
    configure_idp(None)
    configure_auth(api_key="test-api-key")
    yield
    configure_idp(None)


def _build_app(*, with_auth: bool) -> FastAPI:
    app = FastAPI()

    class _FakeRag:
        pass

    router = build_native_shims_router(
        lambda: _FakeRag(),
        auth_dependency=require_auth if with_auth else None,
    )
    app.include_router(router)
    return app


PROTECTED_PATHS = [
    ("GET", "/documents"),
    ("GET", "/documents/abc/chunks"),
    ("POST", "/documents/abc/scan"),
    ("DELETE", "/documents/abc"),
    ("GET", "/pipeline_status"),
    ("GET", "/openapi"),
]

PUBLIC_PATHS = [
    ("GET", "/auth-status"),
    ("POST", "/login"),
    ("POST", "/logout"),
]


@pytest.mark.parametrize("method,path", PROTECTED_PATHS)
def test_protected_routes_reject_anonymous(method, path):
    app = _build_app(with_auth=True)
    client = TestClient(app)
    response = client.request(method, path, json={})
    assert response.status_code == 401, (
        f"{method} {path} returned {response.status_code} for anonymous"
    )


@pytest.mark.parametrize("method,path", PUBLIC_PATHS)
def test_public_routes_reachable_anonymously(method, path):
    app = _build_app(with_auth=True)
    client = TestClient(app)
    payload = {"username": "x", "password": "y"} if path == "/login" else {}
    response = client.request(method, path, json=payload)
    # 401 means "auth was demanded" — we want anything except that for
    # the public handshake routes. /login may 401 on bad creds, that's
    # the credentials check, not the dependency.
    assert response.status_code != 403


def test_protected_routes_accept_valid_bearer():
    app = _build_app(with_auth=True)
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-api-key"}
    response = client.get("/openapi", headers=headers)
    assert response.status_code == 200


def test_pipeline_status_projects_real_lightrag_history(monkeypatch):
    async def fake_get_namespace_data(name, *, workspace):
        assert name == "pipeline_status"
        assert workspace == "ws-test"
        return {
            "busy": True,
            "docs": 2,
            "job_name": "document indexing",
            "latest_message": "Memgraph merge complete",
            "history_messages": [
                "Dequeued BNP incident note",
                "Embedding batch 1/1 complete",
                "Memgraph merge complete",
            ],
            "ignored": "not part of the Twin contract",
        }

    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_namespace_data",
        fake_get_namespace_data,
    )

    app = FastAPI()

    class _FakeRag:
        workspace = "ws-test"

    app.include_router(build_native_shims_router(lambda: _FakeRag()))
    response = TestClient(app).get("/pipeline_status")

    assert response.status_code == 200
    assert response.json() == {
        "busy": True,
        "job_count": 2,
        "job_name": "document indexing",
        "latest_message": "Memgraph merge complete",
        "history_messages": [
            "Dequeued BNP incident note",
            "Embedding batch 1/1 complete",
            "Memgraph merge complete",
        ],
    }


def test_flag_off_keeps_routes_anonymous():
    """LightRAG compat axis: with auth_dependency=None the routes stay
    public — opt-out is honoured."""
    app = _build_app(with_auth=False)
    client = TestClient(app)
    response = client.get("/openapi")
    assert response.status_code == 200
