"""Duplicate upload compatibility: duplicate means add folder membership.

These unit tests avoid Memgraph and pin the control flow introduced for
docs/adr/006-folder-membership-relation.md, decision 4:

- LightRAG 1.4.9.x filename duplicates go through ``get_doc_by_file_path``;
- LightRAG 1.4.11/1.4.12 content duplicates emit ``metadata.is_duplicate``
  DocStatus records that should become memberships on the original doc;
- newer LightRAG duplicate checks use ``get_doc_by_file_basename`` and
  ``get_doc_by_content_hash``;
- filename-only checks are folder-local and non-mutating;
- content-hash or duplicate-metadata confirmation may share membership;
- ordinary folder-scoped reads must not mutate memberships.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from twindb_lightrag_memgraph._constants import (
    confirmed_content_doc_ids_context,
    duplicate_share_folder_context,
    storage_folder_context,
)
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


class _SingleRecordResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record

    async def consume(self) -> None:
        return None


class _ListResult:
    def __init__(self, rows=None):
        self.rows = rows or []

    async def consume(self) -> None:
        return None

    def __aiter__(self):
        async def _iter():
            for row in self.rows:
                yield row

        return _iter()


_ORIGINAL_RECORD = {
    "id": "doc-original",
    "props": {
        "id": "doc-original",
        "file_path": "shared.pdf",
        "status": "processed",
    },
}


class _ReadSession:
    def __init__(self, *, scoped_record=None, global_record=_ORIGINAL_RECORD):
        self.scoped_record = scoped_record
        self.global_record = global_record
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query: str, **params):
        self.calls.append((query, params))
        record = (
            self.scoped_record if params.get("active_folder") else self.global_record
        )
        return _SingleRecordResult(record)


class _WriteSession:
    def __init__(self, *, shared_duplicate_ids: list[str] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.shared_duplicate_ids = shared_duplicate_ids

    async def run(self, query: str, **params):
        self.calls.append((query, params))
        if memberships := params.get("memberships"):
            shared = (
                [item["duplicate_id"] for item in memberships]
                if self.shared_duplicate_ids is None
                else self.shared_duplicate_ids
            )
            return _SingleRecordResult({"duplicate_ids": shared})
        return _ListResult()


def _read_sessions(*, scoped_record=None, global_record=_ORIGINAL_RECORD):
    session = _ReadSession(
        scoped_record=scoped_record,
        global_record=global_record,
    )

    @asynccontextmanager
    async def _read_session():
        yield session

    return _read_session, session


@asynccontextmanager
async def _write_slot():
    yield None


def _store(monkeypatch) -> MemgraphDocStatusStorage:
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "dup_share_unit")
    return MemgraphDocStatusStorage(
        namespace="doc_status",
        global_config={},
        embedding_func=None,
    )


async def test_filename_cross_folder_lookup_is_non_mutating(
    monkeypatch,
):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    read_sessions, session = _read_sessions(scoped_record=None)
    monkeypatch.setattr(_pool, "get_read_session", read_sessions)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_file_path("shared.pdf")

    assert found is None
    assert added == []
    assert len(session.calls) == 1
    assert session.calls[0][1]["active_folder"] == "B"
    assert "MEMBER_OF" in session.calls[0][0]


async def test_basename_cross_folder_lookup_is_non_mutating(
    monkeypatch,
):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    read_sessions, _ = _read_sessions(scoped_record=None)
    monkeypatch.setattr(_pool, "get_read_session", read_sessions)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_file_basename("shared.pdf")

    assert found is None
    assert added == []


async def test_content_hash_cross_folder_lookup_adds_membership_without_duplicate(
    monkeypatch,
):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    read_sessions, session = _read_sessions(scoped_record=None)
    monkeypatch.setattr(_pool, "get_read_session", read_sessions)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_content_hash("abc123")

    # 1.5.5 contract: the original is returned even after a successful share
    # (returning None made the post-parse dedup check read "no duplicate" and
    # fully re-ingest); the membership share still happens eagerly.
    assert found is not None
    assert found[0] == "doc-original"
    assert added == [("doc-original", "B")]
    assert len(session.calls) == 2
    assert session.calls[0][1]["active_folder"] == "B"
    assert "active_folder" not in session.calls[1][1]


async def test_content_hash_share_failure_preserves_duplicate_result(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)

    async def vanished_or_claimed(_doc_id: str, _folder: str) -> bool:
        return False

    read_sessions, _ = _read_sessions(scoped_record=None)
    monkeypatch.setattr(_pool, "get_read_session", read_sessions)
    monkeypatch.setattr(store, "add_to_folder", vanished_or_claimed)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_content_hash("abc123")

    assert found is not None
    assert found[0] == "doc-original"
    assert found[1]["file_path"] == "shared.pdf"


async def test_legacy_filter_shares_only_content_confirmed_existing_id(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    doc_id = "doc-0123456789abcdef0123456789abcdef"
    added: list[tuple[str, str]] = []

    @asynccontextmanager
    async def read_session():
        class _Session:
            async def run(self, _query: str, **_params):
                # No missing row means the key already exists.
                return _ListResult()

        yield _Session()

    async def fake_add_to_folder(found_id: str, folder: str) -> bool:
        added.append((found_id, folder))
        return True

    monkeypatch.setattr(_pool, "get_read_session", read_session)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with (
        duplicate_share_folder_context("B"),
        confirmed_content_doc_ids_context({doc_id}),
    ):
        missing = await store.filter_keys({doc_id})

    assert missing == set()
    assert added == [(doc_id, "B")]


async def test_same_folder_lookup_remains_a_duplicate(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    read_sessions, _ = _read_sessions(scoped_record=_ORIGINAL_RECORD)
    monkeypatch.setattr(_pool, "get_read_session", read_sessions)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_file_basename("shared.pdf")

    assert found is not None
    doc_id, props = found
    assert doc_id == "doc-original"
    assert props["file_path"] == "shared.pdf"


async def test_scoped_read_context_alone_does_not_add_membership(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    read_sessions, _ = _read_sessions()
    monkeypatch.setattr(_pool, "get_read_session", read_sessions)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with storage_folder_context("B"):
        found = await store.get_doc_by_file_path("shared.pdf")

    assert found is not None
    assert found["id"] == "doc-original"
    assert added == []


async def test_content_duplicate_upsert_members_original_without_dup_node(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    write_session = _WriteSession()

    @asynccontextmanager
    async def write_session_ctx():
        yield write_session

    monkeypatch.setattr(_pool, "acquire_write_slot", _write_slot)
    monkeypatch.setattr(_pool, "get_session", write_session_ctx)

    with storage_folder_context("B"):
        await store.upsert(
            {
                "dup-upload": {
                    "status": "failed",
                    "file_path": "copy.pdf",
                    "metadata": {
                        "is_duplicate": True,
                        "duplicate_kind": "content_hash",
                        "original_doc_id": "doc-original",
                    },
                }
            }
        )

    assert len(write_session.calls) == 1
    query, params = write_session.calls[0]
    assert "MERGE (n:`DocStatus_dup_share_unit` {id: e.id})" not in query
    assert params["memberships"][0]["doc_id"] == "doc-original"
    assert params["memberships"][0]["folder"] == "B"
    assert params["memberships"][0]["membership_updated_at"]


async def test_content_duplicate_upsert_persists_dup_when_original_cannot_share(
    monkeypatch,
):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    write_session = _WriteSession(shared_duplicate_ids=[])

    @asynccontextmanager
    async def write_session_ctx():
        yield write_session

    monkeypatch.setattr(_pool, "acquire_write_slot", _write_slot)
    monkeypatch.setattr(_pool, "get_session", write_session_ctx)

    with storage_folder_context("B"):
        await store.upsert(
            {
                "dup-upload": {
                    "status": "failed",
                    "file_path": "copy.pdf",
                    "metadata": {
                        "is_duplicate": True,
                        "duplicate_kind": "content_hash",
                        "original_doc_id": "doc-original",
                    },
                }
            }
        )

    assert len(write_session.calls) == 3
    fallback_property_query, fallback_params = write_session.calls[1]
    assert "MERGE (n:`DocStatus_dup_share_unit` {id: e.id})" in (
        fallback_property_query
    )
    assert fallback_params["entries"][0]["id"] == "dup-upload"
    assert fallback_params["entries"][0]["folder"] == "B"


async def test_filename_duplicate_metadata_does_not_share_original(monkeypatch):
    """Filename identity alone keeps the FAILED attempt visible."""
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    write_session = _WriteSession()

    @asynccontextmanager
    async def write_session_ctx():
        yield write_session

    monkeypatch.setattr(_pool, "acquire_write_slot", _write_slot)
    monkeypatch.setattr(_pool, "get_session", write_session_ctx)

    with storage_folder_context("B"):
        await store.upsert(
            {
                "dup-upload": {
                    "status": "failed",
                    "file_path": "shared.pdf",
                    "track_id": "upload_20260806",
                    "metadata": {
                        "is_duplicate": True,
                        "duplicate_kind": "filename",
                        "original_doc_id": "doc-original",
                    },
                }
            }
        )

    assert len(write_session.calls) == 2
    property_query, property_params = write_session.calls[0]
    assert "MERGE (n:`DocStatus_dup_share_unit` {id: e.id})" in property_query
    assert property_params["entries"][0]["id"] == "dup-upload"
    assert property_params["entries"][0]["folder"] == "B"


async def test_filename_duplicate_with_explicit_hash_uses_atomic_equality_guard(
    monkeypatch,
):
    """A future explicit candidate hash may share, but only through equality."""
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    write_session = _WriteSession()

    @asynccontextmanager
    async def write_session_ctx():
        yield write_session

    monkeypatch.setattr(_pool, "acquire_write_slot", _write_slot)
    monkeypatch.setattr(_pool, "get_session", write_session_ctx)

    with storage_folder_context("test_3"):
        await store.upsert(
            {
                "dup-explicit-hash": {
                    "status": "failed",
                    "file_path": "documento_1.txt",
                    "track_id": "upload_20260806",
                    "content_hash": "sha256-explicit-a",
                    "metadata": {
                        "is_duplicate": True,
                        "duplicate_kind": "filename",
                        "original_doc_id": "doc-original",
                    },
                }
            }
        )

    assert len(write_session.calls) == 1
    query, params = write_session.calls[0]
    assert "MERGE (n:`DocStatus_dup_share_unit` {id: e.id})" not in query
    assert "n.content_hash = m.content_hash" in query
    assert params["memberships"][0]["doc_id"] == "doc-original"
    assert params["memberships"][0]["folder"] == "test_3"
    assert params["memberships"][0]["content_hash"] == "sha256-explicit-a"
    assert params["memberships"][0]["requires_hash_match"] is True


async def test_filename_duplicate_hash_mismatch_preserves_visible_attempt(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    # The real Cypher equality guard returns no duplicate id when the original
    # row carries a different hash; model that result explicitly.
    write_session = _WriteSession(shared_duplicate_ids=[])

    @asynccontextmanager
    async def write_session_ctx():
        yield write_session

    monkeypatch.setattr(_pool, "acquire_write_slot", _write_slot)
    monkeypatch.setattr(_pool, "get_session", write_session_ctx)

    with storage_folder_context("test_3"):
        await store.upsert(
            {
                "dup-explicit-hash": {
                    "status": "failed",
                    "file_path": "documento_1.txt",
                    "track_id": "upload_20260806",
                    "content_hash": "sha256-candidate-b",
                    "metadata": {
                        "is_duplicate": True,
                        "duplicate_kind": "filename",
                        "original_doc_id": "doc-original",
                    },
                }
            }
        )

    assert len(write_session.calls) == 3
    share_query, share_params = write_session.calls[0]
    assert "n.content_hash = m.content_hash" in share_query
    assert share_params["memberships"][0]["content_hash"] == "sha256-candidate-b"
    fallback_query, fallback_params = write_session.calls[1]
    assert "MERGE (n:`DocStatus_dup_share_unit` {id: e.id})" in fallback_query
    assert fallback_params["entries"][0]["id"] == "dup-explicit-hash"
