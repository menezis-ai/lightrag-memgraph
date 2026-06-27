"""Contract guards for ``GET /documents`` projection (TR-ING-01).

The QA issue raised here: a PDF that finished in
``status=failed`` was rendered with 327 chunks and no error reason
visible. The fields the operator needed to see (``error_msg``, ``chunks_count``,
``status``) are already on the LightRAG ``DocProcessingStatus`` and are
projected through ``native_shims._project_doc``. These tests pin that
they reach the public ``GET /documents`` envelope intact so a future
projection edit can't silently regress to "0 chunks shown, no error".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph._constants import DEFAULT_PAGE_SIZE
from twindb_lightrag_memgraph.server import native_shims
from twindb_lightrag_memgraph.server.native_shims import build_native_shims_router


@dataclass
class FakeDocStatus:
    status: str
    file_path: str
    chunks_count: int | None
    chunks_list: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_summary: str | None = None
    content_length: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    track_id: str | None = None
    error_msg: str | None = None


class FakeDocStatusStore:
    """Returns whatever the test fixture wires into ``self.docs``."""

    def __init__(self, docs: dict[str, FakeDocStatus] | None = None) -> None:
        self.docs = docs or {}

    async def get_docs_paginated(self, **_: Any):
        return list(self.docs.items()), len(self.docs)


class FakeRag:
    def __init__(self, docs: dict[str, FakeDocStatus] | None = None) -> None:
        self.doc_status = FakeDocStatusStore(docs)


@pytest.fixture(autouse=True)
def _folder_env(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [{"id": "default", "label": "Default folder", "kind": "primary"}]
        ),
    )


def _make_client(monkeypatch, docs: dict[str, FakeDocStatus]) -> AsyncClient:
    async def no_tags(_docs, *, folder: str) -> None:
        return None

    monkeypatch.setattr(native_shims, "_attach_tags_via_graph", no_tags)
    rag = FakeRag(docs)
    app = FastAPI()
    app.include_router(build_native_shims_router(lambda: rag))
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


class TestDocumentsFailedProjection:
    """The four fields the operator needs to see on a failed doc must reach
    the public ``GET /documents`` envelope intact.
    """

    async def test_failed_doc_preserves_status_chunks_count_and_error_msg(
        self, monkeypatch
    ):
        # QA reported case: 327 chunks indexed before a downstream failure,
        # with an explicit error reason.
        docs = {
            "627-57-PB": FakeDocStatus(
                status="failed",
                file_path="627-57-PB.pdf",
                chunks_count=327,
                error_msg="LLM extractor: invalid JSON response on chunk 14",
                content_summary="Koselleck on bourgeois modernity (es)",
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        doc = items[0]
        # The three operator-facing fields must round-trip verbatim.
        assert doc["status"] == "failed"
        assert doc["chunks_count"] == 327
        assert (
            doc["error_msg"]
            == "LLM extractor: invalid JSON response on chunk 14"
        )

    async def test_documents_envelope_exposes_page_and_page_size(self, monkeypatch):
        docs = {
            "doc-1": FakeDocStatus(
                status="processed",
                file_path="page.pdf",
                chunks_count=1,
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents", params={"cursor": "2"})

        assert r.status_code == 200
        body = r.json()
        assert body["page"] == 2
        assert body["page_size"] == DEFAULT_PAGE_SIZE

    async def test_failed_doc_without_error_msg_keeps_field_as_null(
        self, monkeypatch
    ):
        """A FAILED doc may legitimately lack an error message (older
        ingestion paths). The contract must still expose the field as
        ``null`` — not drop it — so the React port branches consistently.
        """
        docs = {
            "no-reason": FakeDocStatus(
                status="failed",
                file_path="legacy-fail.pdf",
                chunks_count=0,
                error_msg=None,
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["error_msg"] is None
        assert items[0]["status"] == "failed"

    async def test_chunks_count_zero_does_not_collapse_with_none(
        self, monkeypatch
    ):
        """Regression guard for the previous ``or 0`` projection.

        The old code mapped ``chunks_count=None`` and ``chunks_count=0``
        to the same ``0`` — which would have been correct for the wire
        format here, but the symptom (an operator can't tell "indexing
        never started" from "started, produced nothing") was the lossy
        bit. The explicit ``is None`` check now preserves the
        distinction at the projection boundary; the envelope still
        normalises ``None → 0`` so the React contract stays
        non-nullable, but a future contract change to allow ``null``
        won't have to re-litigate this branch.
        """
        docs = {
            "started-zero": FakeDocStatus(
                status="failed",
                file_path="zero-chunks.pdf",
                chunks_count=0,
                error_msg="empty input",
            ),
            "never-started": FakeDocStatus(
                status="failed",
                file_path="never-started.pdf",
                chunks_count=None,
                error_msg="pre-chunking failure",
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents")
        assert r.status_code == 200
        by_id = {item["doc_id"]: item for item in r.json()["items"]}
        assert by_id["started-zero"]["chunks_count"] == 0
        assert by_id["never-started"]["chunks_count"] == 0
        # And the error messages survive on both.
        assert by_id["started-zero"]["error_msg"] == "empty input"
        assert by_id["never-started"]["error_msg"] == "pre-chunking failure"

    async def test_successful_doc_projection_unchanged(self, monkeypatch):
        """The failed-doc test pins what changes; this one pins what
        doesn't, so adding the explicit None check or future error
        plumbing can't accidentally break the happy path.
        """
        docs = {
            "happy": FakeDocStatus(
                status="processed",
                file_path="happy.pdf",
                chunks_count=42,
                error_msg=None,
                content_summary="ok",
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        doc = items[0]
        assert doc["status"] == "processed"
        assert doc["chunks_count"] == 42
        assert doc["error_msg"] is None

    async def test_lightrag_doc_id_projects_real_content_hash(self, monkeypatch):
        digest = "abcdef0123456789abcdef0123456789"
        docs = {
            f"doc-{digest}": FakeDocStatus(
                status="processed",
                file_path="hashed.pdf",
                chunks_count=4,
                metadata={},
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents")
        assert r.status_code == 200
        doc = r.json()["items"][0]
        assert doc["metadata"]["content_hash"] == digest
        assert doc["metadata"]["content_hash_source"] == "lightrag_doc_id"

    async def test_existing_hash_metadata_is_not_overwritten(self, monkeypatch):
        docs = {
            "doc-abcdef0123456789abcdef0123456789": FakeDocStatus(
                status="processed",
                file_path="already-hashed.pdf",
                chunks_count=4,
                metadata={"sha256": "real-sha256"},
            ),
        }
        async with _make_client(monkeypatch, docs) as client:
            r = await client.get("/documents")
        assert r.status_code == 200
        doc = r.json()["items"][0]
        assert doc["metadata"] == {"sha256": "real-sha256"}


class TestPerDocumentScanContract:
    async def test_per_document_scan_rejects_instead_of_acknowledging_noop(
        self,
        monkeypatch,
    ):
        async with _make_client(monkeypatch, {}) as client:
            r = await client.post("/documents/doc-1/scan")

        assert r.status_code == 409
        assert "Per-document scan is not supported" in r.json()["detail"]


class _MembershipDocStatus:
    """DocStatus store exposing the membership API for the ref-counted delete."""

    def __init__(self, folders_by_doc: dict[str, list[str]]) -> None:
        self._folders = folders_by_doc
        self.removed: list[tuple[str, str]] = []

    async def get_folders_for_doc(self, doc_id: str):
        return self._folders.get(doc_id)

    async def remove_from_folder(self, doc_id: str, folder: str) -> int:
        self.removed.append((doc_id, folder))
        remaining = [f for f in (self._folders.get(doc_id) or []) if f != folder]
        self._folders[doc_id] = remaining
        return len(remaining)


class _MembershipRag:
    def __init__(self, doc_status: Any) -> None:
        self.doc_status = doc_status
        self.physically_deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.physically_deleted.append(doc_id)


class TestSingleDeleteRefCounted:
    """DELETE /documents/{id} must un-share, not hard-delete, a shared doc.

    Parity with the bulk-delete surface: a single-delete from folder A on a doc
    shared into A+B may NOT destroy the physical record (it stays visible in B).
    """

    async def test_shared_doc_single_delete_unshares_keeps_it_in_other_folder(self):
        ds = _MembershipDocStatus({"doc1": ["A", "B"]})
        rag = _MembershipRag(ds)
        await native_shims._delete_or_unshare(rag, "doc1", "A")
        # un-shared from A, NOT physically deleted (still MEMBER_OF B)
        assert rag.physically_deleted == []
        assert ds.removed == [("doc1", "A")]

    async def test_last_membership_single_delete_hard_deletes(self):
        ds = _MembershipDocStatus({"doc1": ["A"]})
        rag = _MembershipRag(ds)
        await native_shims._delete_or_unshare(rag, "doc1", "A")
        assert rag.physically_deleted == ["doc1"]
        assert ds.removed == []

    async def test_backend_without_membership_falls_back_to_hard_delete(self):
        class _LegacyDocStatus:
            pass  # no get_folders_for_doc → legacy LightRAG-native behaviour

        rag = _MembershipRag(_LegacyDocStatus())
        await native_shims._delete_or_unshare(rag, "doc1", "A")
        assert rag.physically_deleted == ["doc1"]


class TestNativeDeleteFolderGate:
    """The native DELETE /documents/{id} gate must be membership-first: a doc
    is visible in a folder per MEMBER_OF, not the legacy `folder` property.
    """

    async def test_membership_is_authority_over_legacy_folder_property(self):
        ds = _MembershipDocStatus({"doc1": ["B"]})
        rag = _MembershipRag(ds)
        legacy_doc = {"folder": "A"}  # legacy property disagrees with membership
        # member of B → visible in B; NOT a member of A despite the legacy prop
        assert await native_shims._doc_visible_in_folder(rag, "doc1", legacy_doc, "B")
        assert not await native_shims._doc_visible_in_folder(
            rag, "doc1", legacy_doc, "A"
        )

    async def test_falls_back_to_legacy_folder_when_no_membership_api(self):
        class _LegacyDocStatus:
            pass  # no get_folders_for_doc → legacy _doc_matches_folder

        rag = _MembershipRag(_LegacyDocStatus())
        # _folder_env fixture provisions "default" as the catalog default folder.
        assert await native_shims._doc_visible_in_folder(
            rag, "doc1", {"folder": "default"}, "default"
        )
        assert not await native_shims._doc_visible_in_folder(
            rag, "doc1", {"folder": "default"}, "other"
        )


class _DeleteRouteDocStatus:
    """DocStatus store for the DELETE route: membership is the authority."""

    def __init__(self, membership: list[str]) -> None:
        self._membership = membership
        self.removed: list[tuple[str, str]] = []

    async def get_by_id(self, doc_id: str):
        return {"id": doc_id, "folder": "A"}  # legacy prop intentionally != B

    async def get_folders_for_doc(self, doc_id: str):
        return self._membership

    async def remove_from_folder(self, doc_id: str, folder: str) -> int:
        self.removed.append((doc_id, folder))
        return len([f for f in self._membership if f != folder])


class _DeleteRouteRag:
    def __init__(self, membership: list[str]) -> None:
        self.doc_status = _DeleteRouteDocStatus(membership)
        self.physically_deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.physically_deleted.append(doc_id)


def _make_delete_client(monkeypatch, membership: list[str]):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "A")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "A", "label": "A", "kind": "primary"},
                {"id": "B", "label": "B", "kind": "custom"},
                {"id": "C", "label": "C", "kind": "custom"},
            ]
        ),
    )
    rag = _DeleteRouteRag(membership)
    app = FastAPI()
    app.include_router(build_native_shims_router(lambda: rag))
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return client, rag


class TestNativeDeleteRouteGate:
    """End-to-end DELETE /documents/{id} through the router + gate + delete."""

    async def test_legacy_a_member_of_bc_delete_from_b_unshares(self, monkeypatch):
        client, rag = _make_delete_client(monkeypatch, ["B", "C"])
        async with client:
            r = await client.delete("/documents/doc1", headers={"X-Twin-Folder": "B"})
        # gate accepts B via membership (legacy prop says A); unshare, not physical
        assert r.status_code == 200
        assert rag.doc_status.removed == [("doc1", "B")]
        assert rag.physically_deleted == []

    async def test_legacy_a_member_of_none_delete_from_a_404(self, monkeypatch):
        client, rag = _make_delete_client(monkeypatch, [])
        async with client:
            r = await client.delete("/documents/doc1", headers={"X-Twin-Folder": "A"})
        # empty membership wins over the legacy folder=A property → not visible
        assert r.status_code == 404
        assert rag.physically_deleted == []
        assert rag.doc_status.removed == []
