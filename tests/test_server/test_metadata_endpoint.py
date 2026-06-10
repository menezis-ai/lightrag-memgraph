"""Contract tests for GET /documents/{id}/metadata on the Twin overlay."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import webui_router


class FakeDocStatus:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {
            "doc-c2": {
                "id": "doc-c2",
                "file_path": "/kb/c2.docx",
                "metadata": {
                    "folder": "default",
                    "tags": ["oracle", "rman"],
                    "review": {"state": "approved", "actor": "steward"},
                    "classification": {
                        "class_id": "C2",
                        "raw_name": "C2 Confidentiel",
                    },
                },
            },
            "doc-sandbox": {
                "id": "doc-sandbox",
                "file_path": "/kb/sandbox.docx",
                "metadata": {"folder": "sandbox", "tags": ["sandbox"]},
            },
        }

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()


@pytest.fixture()
async def client(monkeypatch):
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
    webui_router.reset_store()
    _twindb_state["rag"] = FakeRag()
    app = FastAPI()
    app.include_router(webui_router.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c
    _twindb_state.pop("rag", None)
    webui_router.reset_store()


class TestDocumentMetadataEndpoint:
    async def test_returns_overlay_metadata_and_classification(self, client):
        r = await client.get("/documents/doc-c2/metadata")

        assert r.status_code == 200
        body = r.json()
        assert body["tags"] == ["oracle", "rman"]
        assert body["folder"] == "default"
        assert body["review"] == {"state": "approved", "actor": "steward"}
        assert body["classification"]["class_id"] == "C2"
        assert body["metadata"]["classification"]["raw_name"] == "C2 Confidentiel"

    async def test_hides_document_from_another_folder(self, client):
        r = await client.get("/documents/doc-sandbox/metadata")

        assert r.status_code == 404

    async def test_accepts_document_in_requested_folder(self, client):
        r = await client.get(
            "/documents/doc-sandbox/metadata",
            headers={"X-Twin-Folder": "sandbox"},
        )

        assert r.status_code == 200
        assert r.json()["folder"] == "sandbox"
