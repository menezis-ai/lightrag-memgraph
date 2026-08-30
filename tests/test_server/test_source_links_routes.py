from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from twindb_lightrag_memgraph.server.idp_jwt import require_admin_user
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.query.response_sources import (
    _enrich_sources_with_source_links,
)
from twindb_lightrag_memgraph.server.webui import routes_source_links
from twindb_lightrag_memgraph.server.webui import router as webui_router
from twindb_lightrag_memgraph.server.webui.store import get_store, reset_store


@pytest.fixture(autouse=True)
def _clean_store():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    async def visible(_doc_id: str) -> None:
        return None

    async def admin() -> None:
        return None

    monkeypatch.setattr(routes_source_links, "_require_visible_document", visible)
    app = FastAPI()
    app.include_router(routes_source_links.router, prefix="/twin/api")
    app.dependency_overrides[require_admin_user] = admin
    return app


async def test_source_links_crud_version_tombstone_and_audit(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        created_response = await client.post(
            "/twin/api/documents/doc-1/source-links",
            json={"url": "HTTPS://Example.COM:443/runbook", "label": " Runbook "},
        )
        assert created_response.status_code == 201
        created = created_response.json()
        assert created["url"] == "https://example.com/runbook"
        assert created["label"] == "Runbook"
        assert created["version"] == 1

        listed = await client.get("/twin/api/documents/doc-1/source-links")
        assert listed.json() == [created]

        updated_response = await client.patch(
            f"/twin/api/documents/doc-1/source-links/{created['id']}",
            json={"version": 1, "label": None},
        )
        assert updated_response.status_code == 200
        updated = updated_response.json()
        assert updated["label"] is None
        assert updated["version"] == 2

        stale = await client.patch(
            f"/twin/api/documents/doc-1/source-links/{created['id']}",
            json={"version": 1, "label": "stale"},
        )
        assert stale.status_code == 409

        deleted_response = await client.delete(
            f"/twin/api/documents/doc-1/source-links/{created['id']}",
            params={"version": 2},
        )
        assert deleted_response.status_code == 200
        assert deleted_response.json()["deleted"] is True
        assert (await client.get("/twin/api/documents/doc-1/source-links")).json() == []

    events, _, _ = await get_store().list_activity()
    link_events = [
        event for event in events if event["kind"].startswith("source-link-")
    ]
    chronological = list(reversed(link_events))
    assert [event["kind"] for event in chronological] == [
        "source-link-created",
        "source-link-updated",
        "source-link-deleted",
    ]
    assert chronological[-1]["meta"]["before"]["deleted"] is False
    assert chronological[-1]["meta"]["after"]["deleted"] is True


async def test_source_link_rejects_non_web_url(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/twin/api/documents/doc-1/source-links",
            json={"url": "javascript:alert(1)"},
        )
        whitespace = await client.post(
            "/twin/api/documents/doc-1/source-links",
            json={"url": "https://exa mple.test/source"},
        )
    assert response.status_code == 422
    assert whitespace.status_code == 422


async def test_retrieval_sources_inherit_document_links():
    row = {
        "id": "slink-1",
        "doc_id": "doc-1",
        "url": "https://example.test/source",
        "label": "Source",
        "created_by": "tester",
        "created_at": "2026-08-19T00:00:00Z",
        "updated_by": "tester",
        "updated_at": "2026-08-19T00:00:00Z",
        "version": 1,
        "deleted": False,
        "deleted_by": None,
        "deleted_at": None,
    }
    await get_store("default").source_links.create(row)
    sources = [{"doc_id": "doc-1"}, {"doc_id": None}]

    await _enrich_sources_with_source_links(sources, "default")

    assert sources[0]["source_links"] == [row]
    assert sources[1]["source_links"] == []


async def test_cross_folder_document_is_indistinguishable_from_absence(
    app, monkeypatch
):
    async def hidden(_doc_id: str) -> None:
        raise HTTPException(status_code=404, detail="Document not found")

    monkeypatch.setattr(routes_source_links, "_require_visible_document", hidden)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        read = await client.get("/twin/api/documents/other/source-links")
        write = await client.post(
            "/twin/api/documents/other/source-links",
            json={"url": "https://example.com/"},
        )
    assert read.status_code == 404
    assert write.status_code == 404


async def test_real_folder_membership_contract_guards_source_links(monkeypatch):
    """Exercise the production folder resolver, not a visibility test double."""
    from twindb_lightrag_memgraph import _twindb_state

    class DocStatus:
        async def get_by_id(self, doc_id: str):
            if doc_id != "doc-visible-default":
                return None
            return {
                "id": doc_id,
                "status": "processed",
                "metadata": {"folder": "default"},
            }

        async def get_folders_for_doc(self, doc_id: str) -> list[str]:
            assert doc_id == "doc-visible-default"
            return ["default"]

    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default", "kind": "primary"},
                {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            ]
        ),
    )
    previous_rag = _twindb_state.get("rag")
    _twindb_state["rag"] = SimpleNamespace(doc_status=DocStatus())
    configure_auth(api_key="folder-contract-root", jwt_secret=None)
    scoped_app = FastAPI()
    scoped_app.include_router(webui_router.router, prefix="/twin/api")
    headers = {"X-API-Key": "folder-contract-root"}

    try:
        async with AsyncClient(
            transport=ASGITransport(app=scoped_app), base_url="http://test"
        ) as client:
            visible = await client.get(
                "/twin/api/documents/doc-visible-default/source-links",
                headers=headers,
            )
            hidden_read = await client.get(
                "/twin/api/documents/doc-visible-default/source-links",
                headers={**headers, "X-Twin-Folder": "sandbox"},
            )
            hidden_write = await client.post(
                "/twin/api/documents/doc-visible-default/source-links",
                headers={**headers, "X-Twin-Folder": "sandbox"},
                json={"url": "https://example.test/private"},
            )
    finally:
        configure_auth()
        if previous_rag is None:
            _twindb_state.pop("rag", None)
        else:
            _twindb_state["rag"] = previous_rag

    assert visible.status_code == 200
    assert hidden_read.status_code == 404
    assert hidden_write.status_code == 404
