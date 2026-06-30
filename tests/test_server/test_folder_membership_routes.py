"""Endpoint contract tests for folder membership routes.

Covers the new /documents/{id}/folders surface and the ref-counted bulk-delete,
against an in-memory FakeDocStatus that models membership. The admin gate is
dormant here (no TWIN_IDP_JWKS_URL) so require_admin_user returns a placeholder
and the routes are reachable — the gating itself is exercised in the IdP tests.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import webui_router


class FakeDocStatus:
    """In-memory doc store with membership (mirrors the Memgraph contract)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {
            "doc-a": {"id": "doc-a", "file_path": "/kb/a.pdf", "metadata": {}},
            "doc-default": {"id": "doc-default", "file_path": "/kb/d.pdf", "metadata": {}},
            "doc-sandbox": {"id": "doc-sandbox", "file_path": "/kb/s.pdf", "metadata": {}},
        }
        # membership: doc_id -> set of folder ids
        self.members: dict[str, set[str]] = {
            "doc-a": {"default"},
            "doc-default": {"default"},
            "doc-sandbox": {"sandbox"},
        }

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)

    async def get_folders_for_doc(self, doc_id: str):
        if doc_id not in self.docs:
            return None
        return sorted(self.members.get(doc_id, set()))

    async def add_to_folder(self, doc_id: str, folder: str) -> bool:
        if doc_id not in self.docs:
            return False
        self.members.setdefault(doc_id, set()).add(folder)
        return True

    async def remove_from_folder(self, doc_id: str, folder: str):
        if doc_id not in self.docs:
            return None
        self.members.get(doc_id, set()).discard(folder)
        return len(self.members.get(doc_id, set()))


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()
        self.deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.doc_status.docs.pop(doc_id, None)
        self.doc_status.members.pop(doc_id, None)


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
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        c._test_rag = rag
        yield c
    _twindb_state.pop("rag", None)
    webui_router.reset_store()


# ── POST add membership ──────────────────────────────────────────────────


class TestAddMembership:
    async def test_add_shares_doc_into_a_second_folder(self, client):
        r = await client.post(
            "/documents/doc-a/folders", json={"folder_id": "sandbox"}
        )
        assert r.status_code == 200
        assert r.json()["folders"] == ["default", "sandbox"]
        # one physical doc, no copy
        assert client._test_rag.deleted == []

    async def test_unknown_folder_is_rejected(self, client):
        r = await client.post("/documents/doc-a/folders", json={"folder_id": "nope"})
        assert r.status_code == 404

    async def test_missing_folder_id_is_400(self, client):
        r = await client.post("/documents/doc-a/folders", json={})
        assert r.status_code == 400

    async def test_unknown_doc_is_404(self, client):
        r = await client.post("/documents/ghost/folders", json={"folder_id": "sandbox"})
        assert r.status_code == 404


# ── GET memberships (folder-scoped, no cross-folder leak) ────────────────


class TestListMembership:
    async def test_lists_folders_for_visible_doc(self, client):
        await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
        r = await client.get("/documents/doc-a/folders")
        assert r.status_code == 200
        assert r.json()["folders"] == ["default", "sandbox"]

    async def test_does_not_leak_doc_outside_active_folder(self, client):
        # active folder is "default"; doc-sandbox is only in "sandbox" → 404,
        # never reveal its existence/membership.
        r = await client.get("/documents/doc-sandbox/folders")
        assert r.status_code == 404

    async def test_unknown_doc_is_404(self, client):
        r = await client.get("/documents/ghost/folders")
        assert r.status_code == 404


# ── DELETE membership (ref-counted) ──────────────────────────────────────


class TestRemoveMembership:
    async def test_remove_shared_doc_keeps_data(self, client):
        await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
        r = await client.delete("/documents/doc-a/folders/default")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["removed_folder"] == "default"
        assert body["physically_deleted"] is False
        assert body["remaining_folders"] == ["sandbox"]
        assert client._test_rag.deleted == []  # data untouched

    async def test_remove_last_membership_physically_deletes(self, client):
        r = await client.delete("/documents/doc-default/folders/default")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["removed_folder"] == "default"
        assert body["physically_deleted"] is True
        assert body["remaining_folders"] == []
        assert client._test_rag.deleted == ["doc-default"]

    async def test_remove_non_member_folder_is_404(self, client):
        r = await client.delete("/documents/doc-a/folders/sandbox")
        assert r.status_code == 404
        assert client._test_rag.deleted == []

    async def test_remove_unknown_doc_is_404(self, client):
        r = await client.delete("/documents/ghost/folders/default")
        assert r.status_code == 404


class TestMembershipActivity:
    async def test_folder_membership_events_are_serializable(self, client):
        add = await client.post(
            "/documents/doc-a/folders", json={"folder_id": "sandbox"}
        )
        assert add.status_code == 200

        remove = await client.delete("/documents/doc-a/folders/default")
        assert remove.status_code == 200

        activity = await client.get("/activity")
        assert activity.status_code == 200
        kinds = {event["kind"] for event in activity.json()["items"]}
        assert "doc-folder-added" in kinds
        assert "doc-folder-removed" in kinds


# ── Bulk-delete routed through ref-count ─────────────────────────────────


class TestBulkDeleteRefcount:
    async def test_bulk_delete_unshares_shared_doc(self, client):
        # doc-a in {default, sandbox}; bulk-delete from active "default" must
        # only un-share it (still in sandbox), never nuke the physical data.
        await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
        r = await client.post("/documents/bulk-delete", json={"doc_ids": ["doc-a"]})
        assert r.status_code == 200
        assert r.json() == {"deleted": 1, "failed": []}
        assert client._test_rag.deleted == []  # NOT physically deleted
        assert client._test_rag.doc_status.members["doc-a"] == {"sandbox"}

    async def test_bulk_delete_last_membership_physically_deletes(self, client):
        r = await client.post(
            "/documents/bulk-delete", json={"doc_ids": ["doc-default"]}
        )
        assert r.status_code == 200
        assert r.json() == {"deleted": 1, "failed": []}
        assert client._test_rag.deleted == ["doc-default"]
