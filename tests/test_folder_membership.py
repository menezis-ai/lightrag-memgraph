"""Folder membership contract — one document, many folders, stored once.

Integration tests (need a running Memgraph; auto-skipped when MEMGRAPH_URI is
unset). Pins the core contract of FOLDER-MEMBERSHIP-REFACTOR.md batch 1:

- a document `MEMBER_OF` folder A AND folder B is **one physical node** (no
  content duplication);
- it is visible from both folder-scoped reads;
- removal is **ref-counted** — the membership edge is dropped, and the doc is
  only physically deletable when its last membership is gone.
"""

import pytest

from twindb_lightrag_memgraph import _pool, register
from twindb_lightrag_memgraph._constants import storage_folder_context
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

register()

pytestmark = pytest.mark.integration

_WS = "folder_member_test"


def _doc(doc_id: str) -> dict:
    return {
        "id": doc_id,
        "status": "processed",
        "content_summary": "x",
        "content_length": 1,
        "file_path": "faq.md",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "content_hash": "h1",
    }


@pytest.fixture
async def store(monkeypatch):
    # The storage resolves its workspace from the env (resolve_workspace), not a
    # constructor kwarg — pin it so this test owns an isolated label namespace.
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", _WS)
    st = MemgraphDocStatusStorage(
        namespace="doc_status",
        global_config={},
        embedding_func=None,
    )
    assert st.workspace == _WS
    await st.initialize()
    # clean slate (doc + folder nodes)
    async with _pool.get_session() as s:
        await (await s.run(f"MATCH (n:`DocStatus_{_WS}`) DETACH DELETE n")).consume()
        await (await s.run(f"MATCH (f:`Folder_{_WS}`) DETACH DELETE f")).consume()
    yield st
    async with _pool.get_session() as s:
        await (await s.run(f"MATCH (n:`DocStatus_{_WS}`) DETACH DELETE n")).consume()
        await (await s.run(f"MATCH (f:`Folder_{_WS}`) DETACH DELETE f")).consume()


async def _physical_doc_count() -> int:
    async with _pool.get_read_session() as s:
        result = await s.run(f"MATCH (n:`DocStatus_{_WS}`) RETURN count(n) AS c")
        rec = await result.single()
        await result.consume()
        return rec["c"]


async def test_upsert_creates_membership_for_active_folder(store):
    with storage_folder_context("A"):
        await store.upsert({"doc-1": _doc("doc-1")})
    assert await store.get_folders_for_doc("doc-1") == ["A"]


async def test_same_document_in_two_folders_is_one_node(store):
    with storage_folder_context("A"):
        await store.upsert({"doc-1": _doc("doc-1")})
    assert await store.add_to_folder("doc-1", "B") is True

    # member of both, stored once
    assert await store.get_folders_for_doc("doc-1") == ["A", "B"]
    assert await _physical_doc_count() == 1

    # visible from both folder-scoped reads
    docs_a, _ = await store.get_docs_paginated(folder="A")
    docs_b, _ = await store.get_docs_paginated(folder="B")
    assert [d for d, _ in docs_a] == ["doc-1"]
    assert [d for d, _ in docs_b] == ["doc-1"]
    assert (await store.get_status_counts(folder="A")).get("processed") == 1
    assert (await store.get_status_counts(folder="B")).get("processed") == 1


async def test_remove_is_ref_counted(store):
    with storage_folder_context("A"):
        await store.upsert({"doc-1": _doc("doc-1")})
    await store.add_to_folder("doc-1", "B")

    # remove from A → still a member of B, doc untouched
    assert await store.remove_from_folder("doc-1", "A") == 1
    assert await store.get_folders_for_doc("doc-1") == ["B"]
    assert await _physical_doc_count() == 1
    docs_a, _ = await store.get_docs_paginated(folder="A")
    assert [d for d, _ in docs_a] == []

    # remove from B → last membership gone, ref-count 0 (safe to physically delete)
    assert await store.remove_from_folder("doc-1", "B") == 0
    assert await store.get_folders_for_doc("doc-1") == []


async def test_add_to_folder_unknown_doc_returns_false(store):
    assert await store.add_to_folder("nope", "A") is False


async def test_remove_unknown_doc_returns_none(store):
    assert await store.remove_from_folder("nope", "A") is None


async def test_get_folders_distinguishes_absent_from_no_membership(store):
    # absent doc → None (so callers can 404); existing doc with no membership
    # would be [], but here every upserted doc has at least one membership.
    assert await store.get_folders_for_doc("nope") is None
    with storage_folder_context("A"):
        await store.upsert({"doc-1": _doc("doc-1")})
    assert await store.get_folders_for_doc("doc-1") == ["A"]


async def test_add_to_folder_is_idempotent(store):
    with storage_folder_context("A"):
        await store.upsert({"doc-1": _doc("doc-1")})
    await store.add_to_folder("doc-1", "B")
    await store.add_to_folder("doc-1", "B")  # again
    assert await store.get_folders_for_doc("doc-1") == ["A", "B"]
