"""Contract tests for POST /documents/bulk-delete on the Twin overlay."""

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
            "doc-a": {
                "id": "doc-a",
                "file_path": "/kb/a.pdf",
                "metadata": {"folder": "default"},
            },
            "doc-b": {
                "id": "doc-b",
                "file_path": "/kb/b.pdf",
                "metadata": {"folder": "default"},
            },
            "doc-sandbox": {
                "id": "doc-sandbox",
                "file_path": "/kb/sandbox.pdf",
                "metadata": {"folder": "sandbox"},
            },
        }

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self.docs.pop(doc_id, None)


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()
        self.deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        await self.doc_status.delete([doc_id])


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
    rag = FakeRag()
    _twindb_state["rag"] = rag
    app = FastAPI()
    app.include_router(webui_router.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        c._test_rag = rag
        yield c
    _twindb_state.pop("rag", None)
    webui_router.reset_store()


class TestBulkDeleteEndpoint:
    async def test_deletes_documents_and_emits_activity(self, client):
        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-b"], "actor": "operator"},
        )

        assert r.status_code == 200
        assert r.json() == {"deleted": 2, "failed": []}
        assert client._test_rag.deleted == ["doc-a", "doc-b"]

        activity = await client.get("/activity")
        events = activity.json()["items"]
        deletes = [e for e in events if e["kind"] == "doc-deleted"]
        assert len(deletes) == 1
        event = deletes[0]
        assert event["target"]["type"] == "bulk"
        assert event["target"]["label"] == "2 documents"
        assert event["target"]["id"] is None
        assert "2 documents physically deleted" in event["summary"]
        assert "cascade" in event["summary"]
        assert event["meta"]["doc_count"] == 2
        assert set(event["meta"]["doc_ids"]) == {"doc-a", "doc-b"}
        assert event["meta"]["physically_deleted_count"] == 2
        assert event["meta"]["unshared_count"] == 0
        assert event["meta"]["failed"] == []

        doc_a_activity = await client.get(
            "/activity", params={"resource.id": "doc-a"}
        )
        doc_b_activity = await client.get(
            "/activity", params={"resource.id": "doc-b"}
        )
        missing_activity = await client.get(
            "/activity", params={"resource.id": "missing"}
        )
        assert doc_a_activity.json()["total"] == 1
        assert doc_a_activity.json()["items"][0]["id"] == event["id"]
        assert doc_b_activity.json()["total"] == 1
        assert doc_b_activity.json()["items"][0]["id"] == event["id"]
        assert missing_activity.json()["total"] == 0

    async def test_reports_missing_or_cross_folder_ids_as_failed(self, client):
        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-sandbox", "missing"]},
        )

        assert r.status_code == 200
        assert r.json() == {
            "deleted": 1,
            "failed": ["doc-sandbox", "missing"],
        }
        assert client._test_rag.deleted == ["doc-a"]

        activity = await client.get("/activity")
        events = activity.json()["items"]
        deletes = [e for e in events if e["kind"] == "doc-deleted"]
        assert len(deletes) == 1
        assert deletes[0]["target"]["id"] == "doc-a"
        assert deletes[0]["meta"]["doc_id"] == "doc-a"
        assert deletes[0]["meta"]["failed"] == ["doc-sandbox", "missing"]

    async def test_rejects_empty_target_list(self, client):
        r = await client.post("/documents/bulk-delete", json={"doc_ids": []})

        assert r.status_code == 400
        assert "doc_ids" in r.json()["detail"]
