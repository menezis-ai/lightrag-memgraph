"""Regression: destructive document mutations are admin-only (audit 2026-08-06, R-03b).

During the audit, a non-admin ``twk_`` key bulk-deleted a document and its
source file on disk. The four destructive surfaces — bulk-delete,
approve, reject, and the native DELETE shim — are now gated by
``require_admin_user`` (uniform with the tag/graph/folder gates; product
decision 2026-08-06: no separate reviewer role for approve/reject).
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import (
    _create_jwt,
    configure_auth,
    require_auth,
)
from twindb_lightrag_memgraph.server.idp_jwt import require_admin_user
from twindb_lightrag_memgraph.server.native_shims import build_native_shims_router
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings

ROOT_KEY = "test-infra-root"
JWT_SECRET = "x" * 48


def _make_settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        working_dir="/tmp/lightrag_doc_mutation_authz_test",
        workspace="cib",
        enable_langsmith_tracing=False,
        api_key=ROOT_KEY,
        jwt_secret=JWT_SECRET,
        enable_webui_routes=True,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    webui_router.reset_store()
    yield
    webui_router.reset_store()


@pytest.fixture
async def app():
    return create_app(_make_settings())


def _non_admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_create_jwt({'sub': 'operator'})}"}


class _FakeDocStatus:
    async def get_by_id(self, doc_id):
        return {"id": doc_id, "file_path": "/kb/a.pdf", "metadata": {}}

    async def upsert(self, _docs):
        return None


class _FakeRag:
    def __init__(self):
        self.doc_status = _FakeDocStatus()


class TestOverlayMutationGates:
    """Non-admin → 403 on every destructive overlay route."""

    async def test_bulk_delete_requires_admin(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers=_non_admin_headers(),
        ) as client:
            resp = await client.post(
                "/documents/bulk-delete", json={"doc_ids": ["doc-a"]}
            )
        assert resp.status_code == 403

    async def test_approve_requires_admin(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers=_non_admin_headers(),
        ) as client:
            resp = await client.post("/documents/doc-a/approve", json={})
        assert resp.status_code == 403

    async def test_reject_requires_admin(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers=_non_admin_headers(),
        ) as client:
            resp = await client.post(
                "/documents/doc-a/reject", json={"reason": "stale"}
            )
        assert resp.status_code == 403

    async def test_approve_still_works_for_infra_root(self, app):
        """The gate must not regress the legitimate admin path."""
        from twindb_lightrag_memgraph import _twindb_state

        _twindb_state["rag"] = _FakeRag()
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": f"Bearer {ROOT_KEY}"},
            ) as client:
                resp = await client.post("/documents/doc-a/approve", json={})
        finally:
            _twindb_state.pop("rag", None)
        assert resp.status_code == 200
        assert resp.json()["review"]["state"] == "approved"


class TestNativeShimDeleteGate:
    """The DELETE /documents/{id} shim takes the admin dependency."""

    @pytest.fixture
    async def shim_app(self):
        configure_auth(api_key=ROOT_KEY, jwt_secret=JWT_SECRET)
        yield
        configure_auth(api_key=None, jwt_secret=None)

    def _build(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(
            build_native_shims_router(
                lambda: _FakeRag(),
                auth_dependency=require_auth,
                admin_dependency=require_admin_user,
            )
        )
        return app

    async def test_non_admin_delete_gets_403(self, shim_app):
        app = self._build()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers=_non_admin_headers(),
        ) as client:
            resp = await client.delete("/documents/doc-a")
        assert resp.status_code == 403

    async def test_anonymous_delete_gets_401(self, shim_app):
        app = self._build()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/documents/doc-a")
        assert resp.status_code == 401
