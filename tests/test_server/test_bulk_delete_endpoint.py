"""Contract tests for POST /documents/bulk-delete on the Twin overlay."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.auth import configure_auth


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


class _DeletionResult:
    """Shape-compatible stand-in for LightRAG's DeletionResult."""

    def __init__(self, status: str, message: str, status_code: int = 200) -> None:
        self.status = status
        self.message = message
        self.status_code = status_code


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()
        self.deleted: list[str] = []
        self.fail_delete_for: set[str] = set()
        self.busy_delete_for: set[str] = set()
        self.recovery_delete_for: set[str] = set()

    async def adelete_by_doc_id(self, doc_id: str) -> Any:
        if doc_id in self.fail_delete_for:
            raise RuntimeError("backend refused delete")
        if doc_id in self.recovery_delete_for:
            return _DeletionResult(
                "not_allowed",
                "Pipeline recovery is required for this workspace.",
                status_code=503,
            )
        if doc_id in self.busy_delete_for:
            return _DeletionResult(
                "not_allowed",
                "Deletion not allowed: current job 'ingest [3 files]' "
                "is not a document deletion job",
                status_code=403,
            )
        self.deleted.append(doc_id)
        await self.doc_status.delete([doc_id])


@pytest.fixture()
async def client(monkeypatch):
    configure_auth(api_key="bulk-delete-test-root")
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
        headers={"Authorization": "Bearer bulk-delete-test-root"},
    ) as c:
        c._test_rag = rag
        yield c
    _twindb_state.pop("rag", None)
    webui_router.reset_store()
    configure_auth()


class TestBulkDeleteEndpoint:
    async def test_deletes_documents_and_emits_activity(self, client):
        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-b"], "actor": "operator"},
        )

        assert r.status_code == 200
        assert r.json() == {"deleted": 2, "failed": [], "busy": []}
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

        doc_a_activity = await client.get("/activity", params={"resource.id": "doc-a"})
        doc_b_activity = await client.get("/activity", params={"resource.id": "doc-b"})
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

        assert r.status_code == 207
        assert r.json() == {
            "deleted": 1,
            "failed": ["doc-sandbox", "missing"],
            "busy": [],
        }
        assert client._test_rag.deleted == ["doc-a"]

        activity = await client.get("/activity")
        events = activity.json()["items"]
        deletes = [e for e in events if e["kind"] == "doc-deleted"]
        assert len(deletes) == 1
        assert deletes[0]["target"]["id"] == "doc-a"
        assert deletes[0]["meta"]["doc_id"] == "doc-a"
        assert deletes[0]["meta"]["failed"] == ["doc-sandbox", "missing"]

    async def test_backend_delete_error_returns_503_not_failed_success(self, client):
        client._test_rag.fail_delete_for.add("doc-a")

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a"]},
        )

        assert r.status_code == 503
        assert "doc-a" in r.json()["detail"]
        assert "backend refused delete" in r.json()["detail"]
        assert client._test_rag.deleted == []

    async def test_mid_batch_backend_error_returns_honest_partial_success(self, client):
        client._test_rag.fail_delete_for.add("doc-b")

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-b"]},
        )

        assert r.status_code == 207
        assert r.json() == {"deleted": 1, "failed": ["doc-b"], "busy": []}
        assert client._test_rag.deleted == ["doc-a"]

        activity = await client.get("/activity")
        event = next(
            item for item in activity.json()["items"] if item["kind"] == "doc-deleted"
        )
        assert event["meta"]["doc_ids"] == ["doc-a"]
        assert event["meta"]["failed"] == ["doc-b"]

    async def test_rejects_empty_target_list(self, client):
        r = await client.post("/documents/bulk-delete", json={"doc_ids": []})

        assert r.status_code == 400
        assert "doc_ids" in r.json()["detail"]

    async def test_pipeline_busy_on_every_target_returns_423(self, client):
        """LightRAG's not_allowed refusal is transient: answer 423, not 5xx.

        The detail prose is a UI contract — errorMessages.ts keys its
        pipeline-busy copy on "ingestion pipeline" + "busy" (see
        isPipelineBusyDetail); rewording it here must update that matcher.
        """
        client._test_rag.busy_delete_for.update({"doc-a", "doc-b"})

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-b"]},
        )

        assert r.status_code == 423
        detail = r.json()["detail"]
        assert "ingestion pipeline" in detail
        assert "busy" in detail
        # Nothing was deleted and both docs remain retryable.
        assert client._test_rag.deleted == []
        assert await client._test_rag.doc_status.get_by_id("doc-a") is not None
        assert await client._test_rag.doc_status.get_by_id("doc-b") is not None

    async def test_pipeline_busy_subset_reported_in_busy_not_failed(self, client):
        client._test_rag.busy_delete_for.add("doc-b")

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-b"]},
        )

        assert r.status_code == 207
        assert r.json() == {"deleted": 1, "failed": [], "busy": ["doc-b"]}
        assert client._test_rag.deleted == ["doc-a"]
        # The deferred doc is untouched, so the client can retry it verbatim.
        assert await client._test_rag.doc_status.get_by_id("doc-b") is not None

        activity = await client.get("/activity")
        event = next(
            item for item in activity.json()["items"] if item["kind"] == "doc-deleted"
        )
        assert event["meta"]["busy"] == ["doc-b"]
        assert event["meta"]["busy_count"] == 1

    async def test_recovery_required_not_allowed_remains_actionable_503(self, client):
        client._test_rag.recovery_delete_for.add("doc-a")

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a"]},
        )

        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "recovery is required" in detail.lower()
        assert "documented recovery procedure" in detail.lower()
        assert "ingestion pipeline is busy" not in detail.lower()
        assert client._test_rag.deleted == []

    async def test_pipeline_busy_after_real_failure_still_reports_error(self, client):
        """A real failure must keep its 5xx signal even when busy docs exist."""
        client._test_rag.fail_delete_for.add("doc-a")
        client._test_rag.busy_delete_for.add("doc-b")

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-b"]},
        )

        assert r.status_code == 503
        assert "backend refused delete" in r.json()["detail"]
