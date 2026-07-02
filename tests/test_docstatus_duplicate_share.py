"""Duplicate upload compatibility: duplicate means add folder membership.

These unit tests avoid Memgraph and pin the control flow introduced for
FOLDER-MEMBERSHIP-REFACTOR step 5:

- LightRAG 1.4.9.x filename duplicates go through ``get_doc_by_file_path``;
- LightRAG 1.4.11/1.4.12 content duplicates emit ``metadata.is_duplicate``
  DocStatus records that should become memberships on the original doc;
- newer LightRAG duplicate checks use ``get_doc_by_file_basename`` and
  ``get_doc_by_content_hash``;
- ordinary folder-scoped reads must not mutate memberships.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from twindb_lightrag_memgraph._constants import (
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


class _ReadSession:
    async def run(self, _query: str, **_params):
        return _SingleRecordResult(
            {
                "id": "doc-original",
                "props": {
                    "id": "doc-original",
                    "file_path": "shared.pdf",
                    "status": "processed",
                },
            }
        )


class _WriteSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query: str, **params):
        self.calls.append((query, params))
        return _ListResult()


@asynccontextmanager
async def _read_session():
    yield _ReadSession()


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


async def test_filename_duplicate_lookup_adds_membership(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    monkeypatch.setattr(_pool, "get_read_session", _read_session)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_file_path("shared.pdf")

    assert found is not None
    assert found["id"] == "doc-original"
    assert added == [("doc-original", "B")]


async def test_basename_duplicate_lookup_adds_membership(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    monkeypatch.setattr(_pool, "get_read_session", _read_session)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_file_basename("shared.pdf")

    assert found is not None
    doc_id, props = found
    assert doc_id == "doc-original"
    assert props["file_path"] == "shared.pdf"
    assert added == [("doc-original", "B")]


async def test_content_hash_duplicate_lookup_adds_membership(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    monkeypatch.setattr(_pool, "get_read_session", _read_session)
    monkeypatch.setattr(store, "add_to_folder", fake_add_to_folder)

    with duplicate_share_folder_context("B"):
        found = await store.get_doc_by_content_hash("abc123")

    assert found is not None
    doc_id, props = found
    assert doc_id == "doc-original"
    assert props["file_path"] == "shared.pdf"
    assert added == [("doc-original", "B")]


async def test_scoped_read_context_alone_does_not_add_membership(monkeypatch):
    from twindb_lightrag_memgraph import _pool

    store = _store(monkeypatch)
    added: list[tuple[str, str]] = []

    async def fake_add_to_folder(doc_id: str, folder: str) -> bool:
        added.append((doc_id, folder))
        return True

    monkeypatch.setattr(_pool, "get_read_session", _read_session)
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
                        "original_doc_id": "doc-original",
                    },
                }
            }
        )

    assert len(write_session.calls) == 1
    query, params = write_session.calls[0]
    assert "MERGE (n:`DocStatus_dup_share_unit` {id: e.id})" not in query
    assert params["memberships"] == [{"doc_id": "doc-original", "folder": "B"}]
