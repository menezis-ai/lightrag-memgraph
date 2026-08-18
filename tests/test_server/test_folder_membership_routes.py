"""Endpoint contract tests for folder membership routes.

Covers the new /documents/{id}/folders surface and the ref-counted bulk-delete,
against an in-memory FakeDocStatus that models membership. With no IdP, the
fixture authenticates using the explicit infrastructure root key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lightrag.base import (
    DocSchedulingRecord,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.webui import routes_documents
from twindb_lightrag_memgraph.server.auth import configure_auth


class FakeDocStatus:
    """In-memory doc store with membership (mirrors the Memgraph contract)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {
            "doc-a": {"id": "doc-a", "file_path": "/kb/a.pdf", "metadata": {}},
            "doc-default": {
                "id": "doc-default",
                "file_path": "/kb/d.pdf",
                "metadata": {},
            },
            "doc-sandbox": {
                "id": "doc-sandbox",
                "file_path": "/kb/s.pdf",
                "metadata": {},
            },
        }
        # membership: doc_id -> set of folder ids
        self.members: dict[str, set[str]] = {
            "doc-a": {"default"},
            "doc-default": {"default"},
            "doc-sandbox": {"sandbox"},
        }
        self.claims: dict[str, str] = {}

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)

    async def get_folders_for_doc(self, doc_id: str):
        if doc_id not in self.docs:
            return None
        return sorted(self.members.get(doc_id, set()))

    async def add_to_folder(self, doc_id: str, folder: str) -> bool:
        if doc_id not in self.docs or doc_id in self.claims:
            return False
        self.members.setdefault(doc_id, set()).add(folder)
        return True

    async def resolve_doc_source_strict(self, canonical_source_key: str):
        if canonical_source_key == "conflict.pdf":
            return SourceConflict(
                candidate_count=2,
                sample_doc_ids=("doc-conflict-a", "doc-conflict-b"),
            )
        for doc_id, doc in self.docs.items():
            if Path(doc["file_path"]).name != canonical_source_key:
                continue
            return SourceUnique(
                doc_id=doc_id,
                doc=DocSchedulingRecord(
                    id=doc_id,
                    status=DocStatus.PROCESSED,
                    created_at="2026-08-06T00:00:00Z",
                    updated_at="2026-08-06T00:00:00Z",
                    file_path=canonical_source_key,
                    track_id=f"track-{doc_id}",
                    has_custom_chunk_journal=False,
                ),
            )
        return SourceAbsent()

    async def remove_from_folder(self, doc_id: str, folder: str):
        if doc_id not in self.docs:
            return None
        self.members.get(doc_id, set()).discard(folder)
        return len(self.members.get(doc_id, set()))

    async def claim_last_membership_delete(
        self, doc_id: str, folder: str, claim: str
    ) -> bool:
        if doc_id in self.claims or self.members.get(doc_id) != {folder}:
            return False
        self.claims[doc_id] = claim
        return True

    async def release_delete_claim(self, doc_id: str, claim: str) -> None:
        if self.claims.get(doc_id) == claim:
            self.claims.pop(doc_id)


class _BusyDeletionResult:
    """LightRAG DeletionResult shape for the pipeline-reserved refusal."""

    status = "not_allowed"
    status_code = 403
    message = (
        "Deletion not allowed: current job 'ingest [3 files]' "
        "is not a document deletion job"
    )


class _RecoveryRequiredDeletionResult:
    status = "not_allowed"
    status_code = 503
    message = "Pipeline recovery is required for this workspace."


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()
        self.deleted: list[str] = []
        self.cache_cleared = 0
        self.aclear_cache_raises = False
        self.delete_raises = False
        self.pipeline_busy = False
        self.recovery_required = False

    async def adelete_by_doc_id(self, doc_id: str):
        if self.delete_raises:
            raise RuntimeError("delete cascade failed")
        if self.recovery_required:
            return _RecoveryRequiredDeletionResult()
        if self.pipeline_busy:
            return _BusyDeletionResult()
        self.deleted.append(doc_id)
        self.doc_status.docs.pop(doc_id, None)
        self.doc_status.members.pop(doc_id, None)
        self.doc_status.claims.pop(doc_id, None)

    async def aclear_cache(self) -> None:
        # LightRAG's LLM-response-cache purge, invoked by _delete_doc_from_rag
        # after a physical delete so cached answers stop citing the dead doc.
        self.cache_cleared += 1
        if self.aclear_cache_raises:
            raise RuntimeError("cache backend down")


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
    configure_auth(api_key="test-infra-root")
    rag = FakeRag()
    _twindb_state["rag"] = rag
    app = FastAPI()
    app.include_router(webui_router.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    ) as c:
        c._test_rag = rag
        yield c
    _twindb_state.pop("rag", None)
    webui_router.reset_store()
    configure_auth()


# ── Resolve upload intent ────────────────────────────────────────────────


class TestResolveUpload:
    async def test_existing_source_is_shared_into_active_folder(self, client):
        r = await client.post(
            "/documents/resolve-upload",
            json={"file_name": "a.pdf"},
            headers={"X-Twin-Folder": "sandbox"},
        )

        assert r.status_code == 200
        assert r.json() == {
            "action": "shared",
            "doc_id": "doc-a",
            "track_id": "track-doc-a",
            "message": "'a.pdf' was added to folder 'sandbox'.",
        }
        assert client._test_rag.doc_status.members["doc-a"] == {
            "default",
            "sandbox",
        }

    async def test_existing_source_in_active_folder_is_idempotent(self, client):
        r = await client.post(
            "/documents/resolve-upload",
            json={"file_name": "a.pdf"},
        )

        assert r.status_code == 200
        assert r.json()["action"] == "already_present"
        assert client._test_rag.doc_status.members["doc-a"] == {"default"}

    async def test_unknown_source_continues_to_native_upload(self, client):
        r = await client.post(
            "/documents/resolve-upload",
            json={"file_name": "new.pdf"},
            headers={"X-Twin-Folder": "sandbox"},
        )

        assert r.status_code == 200
        assert r.json() == {"action": "upload"}

    async def test_unsafe_filename_is_rejected_like_native_upload(self, client):
        r = await client.post(
            "/documents/resolve-upload",
            json={"file_name": "../a.pdf"},
        )

        assert r.status_code == 400

    async def test_conflicting_primary_sources_fail_closed(self, client):
        r = await client.post(
            "/documents/resolve-upload",
            json={"file_name": "conflict.pdf"},
            headers={"X-Twin-Folder": "sandbox"},
        )

        assert r.status_code == 409
        assert "source conflict" in r.json()["detail"]


# ── POST add membership ──────────────────────────────────────────────────


class TestAddMembership:
    async def test_add_shares_doc_into_a_second_folder(self, client):
        r = await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
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

    async def test_last_membership_cas_conflict_is_409_not_hard_delete(self, client):
        doc_status = client._test_rag.doc_status

        async def concurrent_share(_doc_id: str, _folder: str, _claim: str) -> bool:
            doc_status.members["doc-default"].add("sandbox")
            return False

        doc_status.claim_last_membership_delete = concurrent_share
        r = await client.delete("/documents/doc-default/folders/default")

        assert r.status_code == 409
        assert client._test_rag.deleted == []
        assert doc_status.members["doc-default"] == {"default", "sandbox"}

    async def test_failed_cascade_releases_claim(self, client):
        client._test_rag.delete_raises = True

        with pytest.raises(RuntimeError, match="delete cascade failed"):
            await routes_documents._delete_with_last_membership_claim(
                webui_router._delete_doc_from_rag,
                client._test_rag,
                "doc-default",
                "default",
            )

        assert "doc-default" not in client._test_rag.doc_status.claims


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
        assert r.json() == {"deleted": 1, "failed": [], "busy": []}
        assert client._test_rag.deleted == []  # NOT physically deleted
        assert client._test_rag.doc_status.members["doc-a"] == {"sandbox"}

    async def test_bulk_delete_last_membership_physically_deletes(self, client):
        r = await client.post(
            "/documents/bulk-delete", json={"doc_ids": ["doc-default"]}
        )
        assert r.status_code == 200
        assert r.json() == {"deleted": 1, "failed": [], "busy": []}
        assert client._test_rag.deleted == ["doc-default"]

    async def test_bulk_delete_busy_pipeline_unshares_but_defers_last_membership(
        self, client
    ):
        """The 2026-08-06 OVH incident shape: during an ingestion, un-sharing
        keeps working (no LightRAG call) while last-membership physical
        deletes are deferred as `busy` — with the delete claim released so
        the retry is not stranded."""
        await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
        client._test_rag.pipeline_busy = True

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-default"]},
        )

        assert r.status_code == 207
        assert r.json() == {"deleted": 1, "failed": [], "busy": ["doc-default"]}
        assert client._test_rag.deleted == []
        assert client._test_rag.doc_status.members["doc-a"] == {"sandbox"}
        # Deferred doc untouched and unclaimed → the retry can proceed.
        assert "doc-default" in client._test_rag.doc_status.docs
        assert "doc-default" not in client._test_rag.doc_status.claims

        # Pipeline drained → the same retry payload now completes.
        client._test_rag.pipeline_busy = False
        retry = await client.post(
            "/documents/bulk-delete", json={"doc_ids": ["doc-default"]}
        )
        assert retry.status_code == 200
        assert retry.json() == {"deleted": 1, "failed": [], "busy": []}
        assert client._test_rag.deleted == ["doc-default"]

    async def test_bulk_partial_unshare_then_recovery_fence_returns_structured_503(
        self, client
    ):
        await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
        client._test_rag.recovery_required = True

        r = await client.post(
            "/documents/bulk-delete",
            json={"doc_ids": ["doc-a", "doc-default"]},
        )

        assert r.status_code == 503
        body = r.json()
        assert body["recovery_required"] is True
        assert body["deleted"] == 1
        assert body["committed_doc_ids"] == ["doc-a"]
        assert body["failed"] == ["doc-default"]
        assert body["busy"] == []
        assert body["unattempted"] == []
        assert "operator recovery is required" in body["detail"].lower()
        assert (
            "1 earlier document change was already committed (doc-a)" in body["detail"]
        )
        # The first mutation really committed; the fenced physical delete did
        # not, and its last-membership claim was released.
        assert client._test_rag.doc_status.members["doc-a"] == {"sandbox"}
        assert "doc-default" in client._test_rag.doc_status.docs
        assert "doc-default" not in client._test_rag.doc_status.claims

        activity = await client.get("/activity")
        event = next(
            item
            for item in activity.json()["items"]
            if item["kind"] == "doc-folder-removed"
        )
        assert event["meta"]["doc_ids"] == ["doc-a"]
        assert event["meta"]["failed"] == ["doc-default"]

    async def test_single_folder_remove_busy_pipeline_returns_423(self, client):
        client._test_rag.pipeline_busy = True

        r = await client.delete("/documents/doc-default/folders/default")

        assert r.status_code == 423
        detail = r.json()["detail"]
        assert "ingestion pipeline" in detail
        assert "busy" in detail
        assert "doc-default" in client._test_rag.doc_status.docs
        assert "doc-default" not in client._test_rag.doc_status.claims


# ── Query-cache purge on physical delete ─────────────────────────────────
#
# LightRAG caches query answers keyed only on (query, mode, params), never on
# corpus state. Without an explicit purge, a physically-deleted doc keeps being
# cited in the *answer text* of any repeated question even though folder-scoped
# retrieval no longer returns its chunks. _delete_doc_from_rag must drop the
# cache on physical delete — and only on physical delete.
class TestQueryCachePurgeOnDelete:
    async def test_physical_delete_purges_query_cache(self, client):
        r = await client.delete("/documents/doc-default/folders/default")
        assert r.status_code == 200
        assert r.json()["physically_deleted"] is True
        assert client._test_rag.deleted == ["doc-default"]
        assert client._test_rag.cache_cleared == 1

    async def test_bulk_physical_delete_purges_query_cache(self, client):
        r = await client.post(
            "/documents/bulk-delete", json={"doc_ids": ["doc-default"]}
        )
        assert r.status_code == 200
        assert client._test_rag.deleted == ["doc-default"]
        # Exactly one purge for one physical delete — a double global flush
        # would be needlessly costly and is a regression this pins down.
        assert client._test_rag.cache_cleared == 1

    async def test_unshare_does_not_purge_query_cache(self, client):
        # Shared doc removed from one folder is NOT physically deleted, so the
        # global answer cache must be left intact (the doc still exists).
        await client.post("/documents/doc-a/folders", json={"folder_id": "sandbox"})
        r = await client.delete("/documents/doc-a/folders/default")
        assert r.status_code == 200
        assert r.json()["physically_deleted"] is False
        assert client._test_rag.deleted == []
        assert client._test_rag.cache_cleared == 0

    async def test_cache_purge_failure_does_not_break_delete(self, client):
        # Cache purge is best-effort: a backend failure must never fail the
        # primary delete (side-effect-never-breaks-primary-op rule).
        client._test_rag.aclear_cache_raises = True
        r = await client.delete("/documents/doc-default/folders/default")
        assert r.status_code == 200
        assert r.json()["physically_deleted"] is True
        assert client._test_rag.deleted == ["doc-default"]
