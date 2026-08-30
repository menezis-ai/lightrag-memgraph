from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _rag(folder: str = "f1"):
    rag = MagicMock()
    rag.text_chunks.get_by_id = AsyncMock(
        return_value={
            "_id": "c1",
            "content": "one",
            "full_doc_id": "d1",
            "file_path": "runbook.md",
            "chunk_order_index": 0,
            "tokens": 1,
        }
    )
    rag.text_chunks.get_by_ids = AsyncMock(
        return_value=[
            {
                "_id": "c1",
                "content": "one",
                "full_doc_id": "d1",
                "file_path": "runbook.md",
                "chunk_order_index": 0,
                "tokens": 1,
            }
        ]
    )
    rag.doc_status.get_by_id = AsyncMock(
        return_value={"chunks_list": ["c1"], "file_path": "runbook.md"}
    )
    rag.doc_status.get_folders_for_doc = AsyncMock(return_value=[folder])
    return rag


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "f1")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "f1", "label": "Folder 1", "kind": "kb"},
                {"id": "f2", "label": "Folder 2", "kind": "kb"},
            ]
        ),
    )


def _overlay_client(rag) -> TestClient:
    import twindb_lightrag_memgraph as t
    from twindb_lightrag_memgraph.server.auth import configure_auth

    t._twindb_state["rag"] = rag
    app = FastAPI()
    t._mount_twin_subapp(app, "/twin/api", webui_stores="seed")
    configure_auth(api_key="root")
    return TestClient(
        app,
        headers={"Authorization": "Bearer root", "X-Twin-Folder": "f1"},
    )


def test_production_overlay_exercises_all_prefixed_chunk_routes(monkeypatch):
    from twindb_lightrag_memgraph.server import chunk_routes

    source_link = {"id": "slink-1", "doc_id": "d1", "url": "https://example.test/"}
    monkeypatch.setattr(
        chunk_routes,
        "_source_links_for_doc",
        AsyncMock(return_value=[source_link]),
    )
    client = _overlay_client(_rag())

    context = client.get("/twin/api/chunks/c1/context", params={"window": 3})
    document = client.get("/twin/api/chunks/c1/document")
    chunks = client.get("/twin/api/documents/d1/chunks", params={"start": 0, "end": 10})

    assert context.status_code == document.status_code == chunks.status_code == 200
    assert context.json()["chunks"][0]["chunk_id"] == "c1"
    assert context.json()["source_links"] == [source_link]
    assert document.json()["doc_id"] == "d1"
    assert document.json()["source_links"] == [source_link]
    assert chunks.json()["total_chunks_in_doc"] == 1
    assert chunks.json()["source_links"] == [source_link]


def test_standalone_factory_exercises_all_prefixed_chunk_routes(monkeypatch):
    from twindb_lightrag_memgraph.server import app as app_module
    from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings

    rag = _rag()
    monkeypatch.setattr(app_module, "_get_rag", lambda: rag)
    app = app_module.create_app(
        LightRAGServerSettings(
            workspace="chunk_route_test",
            api_key="root",
            enable_langsmith_tracing=False,
            graph_relation_id_backfill_on_startup=False,
            webui_tag_backend="memory",
            webui_activity_backend="memory",
            webui_notifications_backend="memory",
        )
    )
    client = TestClient(
        app,
        headers={"Authorization": "Bearer root", "X-Twin-Folder": "f1"},
    )

    context = client.get("/twin/api/chunks/c1/context")
    document = client.get("/twin/api/chunks/c1/document")
    chunks = client.get("/twin/api/documents/d1/chunks")

    assert context.status_code == document.status_code == chunks.status_code == 200
    assert context.json()["doc_id"] == "d1"
    assert document.json()["chunks"][0]["chunk_id"] == "c1"
    assert chunks.json()["total_chunks_in_doc"] == 1


def test_overlay_chunk_route_hides_cross_folder_document():
    client = _overlay_client(_rag(folder="f2"))

    response = client.get("/twin/api/chunks/c1/document")

    assert response.status_code == 404


def test_chunk_routes_survive_source_link_backend_failure():
    from twindb_lightrag_memgraph.server.webui.store import get_store

    class BrokenSourceLinks:
        async def list_for_document(self, _doc_id):
            raise RuntimeError("source-link graph unavailable")

    client = _overlay_client(_rag())
    get_store("f1")._source_link_backend = BrokenSourceLinks()  # noqa: SLF001

    context = client.get("/twin/api/chunks/c1/context")
    document = client.get("/twin/api/chunks/c1/document")
    chunks = client.get("/twin/api/documents/d1/chunks")

    assert context.status_code == document.status_code == chunks.status_code == 200
    assert context.json()["source_links"] == []
    assert document.json()["source_links"] == []
    assert chunks.json()["source_links"] == []
