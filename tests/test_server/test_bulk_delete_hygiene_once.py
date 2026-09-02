"""Bulk delete owes the post-delete hygiene ONCE per batch, not once per doc.

``_run_bulk_delete_batch`` deletes serially (write-conflict storms, 28afb6c).
Every physical delete used to purge the query LLM cache and run the
workspace-global source_id sweep inline — N cache drops and N three-scan
sweeps per request, the sweep's coalescing never engaging inside one request.
The batch now defers that hygiene (``hygiene=False``) and settles it once
after its last delete, in a ``finally`` so a recovery fence or a cancellation
mid-batch still settles what the committed deletes owe. It also hands the
DocStatus record it already read down to the physical delete instead of
reading it a third time.

Unit cases drive the real batch runner with recording fakes; the integration
case drives the real batch → real deferred hygiene → real sweep on a live
Memgraph (auto-skipped without ``MEMGRAPH_URI``).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from twindb_lightrag_memgraph.server import graph_reader
from twindb_lightrag_memgraph.server.webui import router
from twindb_lightrag_memgraph.server.webui import routes_documents as rd

FOLDER = "default"


class _DocStatus:
    def __init__(self, memberships: dict[str, list[str]]) -> None:
        self.memberships = {k: list(v) for k, v in memberships.items()}
        self.claims: list[str] = []

    async def get_folders_for_doc(self, doc_id: str):
        folders = self.memberships.get(doc_id)
        return list(folders) if folders is not None else None

    async def claim_last_membership_delete(self, doc_id, folder, claim) -> bool:
        self.claims.append(doc_id)
        return doc_id in self.memberships

    async def release_delete_claim(self, doc_id, claim) -> None:
        return None

    async def remove_from_folder(self, doc_id: str, folder: str) -> None:
        self.memberships[doc_id] = [f for f in self.memberships[doc_id] if f != folder]


class _Rag:
    def __init__(self, memberships: dict[str, list[str]]) -> None:
        self.doc_status = _DocStatus(memberships)


class _Legacy:
    """Recording stand-in for the ``webui_router`` shim."""

    def __init__(
        self,
        *,
        fail_with: dict[str, Exception] | None = None,
        block_on: str | None = None,
    ) -> None:
        self.deletes: list[tuple[str, bool, str | None]] = []
        self.hygiene_calls = 0
        self.deletes_before_hygiene: list[str] | None = None
        self.fail_with = fail_with or {}
        # ``block_on``: the physical delete of that doc parks forever, so a test
        # can cancel the batch while it is in flight (client disconnect).
        self.block_on = block_on
        self.blocked = asyncio.Event()

    def current_folder_id(self) -> str:
        return FOLDER

    async def _get_doc_for_active_folder(self, doc_id: str) -> dict:
        return {"id": doc_id, "file_path": f"/kb/{doc_id}.md"}

    async def _delete_doc_from_rag(
        self, rag, doc_id, *, hygiene=True, source_path=None
    ):
        if doc_id in self.fail_with:
            raise self.fail_with[doc_id]
        if doc_id == self.block_on:
            self.blocked.set()
            await asyncio.Event().wait()  # never set: parked until cancelled
        self.deletes.append((doc_id, hygiene, source_path))
        rag.doc_status.memberships.pop(doc_id, None)

    async def _post_delete_hygiene(self, rag) -> None:
        self.hygiene_calls += 1
        self.deletes_before_hygiene = [d for d, _, _ in self.deletes]


async def test_batch_defers_hygiene_and_settles_it_once_after_the_last_delete():
    rag = _Rag({"a": [FOLDER], "b": [FOLDER], "c": [FOLDER]})
    legacy = _Legacy()
    results, failed, busy = await rd._run_bulk_delete_batch(
        legacy, rag, ["a", "b", "c"]
    )
    assert (failed, busy) == ([], [])
    assert [r["doc_id"] for r in results] == ["a", "b", "c"]
    assert all(r["physically_deleted"] for r in results)
    # Every physical delete ran WITHOUT its own hygiene and with the source
    # path the visibility check already had — no third DocStatus read.
    assert legacy.deletes == [
        ("a", False, "/kb/a.md"),
        ("b", False, "/kb/b.md"),
        ("c", False, "/kb/c.md"),
    ]
    assert legacy.hygiene_calls == 1
    assert legacy.deletes_before_hygiene == ["a", "b", "c"]


async def test_unshare_only_batch_owes_no_hygiene():
    rag = _Rag({"a": [FOLDER, "other"], "b": [FOLDER, "other"]})
    legacy = _Legacy()
    results, _, _ = await rd._run_bulk_delete_batch(legacy, rag, ["a", "b"])
    assert [r["physically_deleted"] for r in results] == [False, False]
    assert legacy.deletes == []
    assert legacy.hygiene_calls == 0
    assert rag.doc_status.memberships == {"a": ["other"], "b": ["other"]}


async def test_recovery_fence_mid_batch_still_settles_the_committed_deletes():
    """The 503 fence stops the loop after 'a' was physically deleted: the
    batch raises, but 'a' still gets the hygiene it owes (finally)."""
    rag = _Rag({"a": [FOLDER], "b": [FOLDER], "c": [FOLDER]})
    legacy = _Legacy(fail_with={"b": HTTPException(status_code=503, detail="fence")})
    with pytest.raises(rd._BulkDeleteRecoveryRequired) as excinfo:
        await rd._run_bulk_delete_batch(legacy, rag, ["a", "b", "c"])
    assert [r["doc_id"] for r in excinfo.value.results] == ["a"]
    assert excinfo.value.unattempted == ["c"]
    assert legacy.hygiene_calls == 1
    assert legacy.deletes_before_hygiene == ["a"]


async def test_cancellation_mid_batch_still_settles_the_committed_deletes():
    """Client disconnect while 'b' is being deleted, after 'a' was committed:
    ``CancelledError`` is a ``BaseException`` and bypasses the loop's
    ``except Exception`` — only the ``finally`` can settle what 'a' owes. The
    cancellation still propagates afterwards."""
    rag = _Rag({"a": [FOLDER], "b": [FOLDER], "c": [FOLDER]})
    legacy = _Legacy(block_on="b")
    task = asyncio.create_task(rd._run_bulk_delete_batch(legacy, rag, ["a", "b", "c"]))
    await asyncio.wait_for(legacy.blocked.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert legacy.hygiene_calls == 1
    assert legacy.deletes_before_hygiene == ["a"]
    assert [d for d, _, _ in legacy.deletes] == ["a"]  # 'b' never completed
    assert "c" in rag.doc_status.memberships  # 'c' never started


async def test_busy_only_batch_is_423_without_hygiene():
    from twindb_lightrag_memgraph.server._lightrag_compat import (
        PipelineBusyDeletionError,
    )

    rag = _Rag({"a": [FOLDER]})
    legacy = _Legacy(fail_with={"a": PipelineBusyDeletionError("busy")})
    with pytest.raises(HTTPException) as excinfo:
        await rd._run_bulk_delete_batch(legacy, rag, ["a"])
    assert excinfo.value.status_code == 423
    assert legacy.hygiene_calls == 0


class _HygieneRag:
    """Fake RAG for the per-document helper: counts every hygiene touch."""

    workspace = "ws-hyg"

    def __init__(self) -> None:
        self.get_by_id_calls = 0
        self.cache_cleared = 0
        self.deleted: list[str] = []
        self.doc_status = self

    async def get_by_id(self, doc_id: str):
        self.get_by_id_calls += 1
        return {"id": doc_id, "file_path": f"/kb/{doc_id}.md"}

    async def adelete_by_doc_id(self, doc_id: str):
        self.deleted.append(doc_id)
        return None

    async def aclear_cache(self) -> None:
        self.cache_cleared += 1


@pytest.fixture
def sweeps(monkeypatch):
    calls: list[str] = []

    async def _sweep(workspace: str):
        calls.append(workspace)
        return {}

    async def _no_cleanup(paths):
        return None

    from twindb_lightrag_memgraph import _import_cleanup

    monkeypatch.setattr(graph_reader, "request_source_ref_sweep", _sweep)
    monkeypatch.setattr(_import_cleanup, "cleanup_import_paths", _no_cleanup)
    return calls


async def test_single_delete_keeps_inline_hygiene_and_reads_its_own_source_path(sweeps):
    rag = _HygieneRag()
    await router._delete_doc_from_rag(rag, "d1")
    assert rag.deleted == ["d1"]
    assert rag.get_by_id_calls == 1
    assert rag.cache_cleared == 1 and sweeps == ["ws-hyg"]


async def test_deferred_delete_skips_hygiene_and_the_extra_read(sweeps):
    rag = _HygieneRag()
    await router._delete_doc_from_rag(rag, "d1", hygiene=False, source_path="/kb/d1.md")
    assert rag.deleted == ["d1"]
    assert rag.get_by_id_calls == 0
    assert rag.cache_cleared == 0 and sweeps == []
    # The deferred hygiene is what the batch calls afterwards.
    await router._post_delete_hygiene(rag)
    assert rag.cache_cleared == 1 and sweeps == ["ws-hyg"]


# ---------------------------------------------------------------------------
# Integration — real batch → real deferred hygiene → real sweep on Memgraph
# ---------------------------------------------------------------------------

SEP = graph_reader._GRAPH_FIELD_SEP


class _LiveDocStatus(_DocStatus):
    def __init__(self, memberships, chunks: dict[str, str]) -> None:
        super().__init__(memberships)
        self.chunks = chunks

    async def get_by_id(self, doc_id: str):
        if doc_id not in self.memberships:
            return None
        return {"id": doc_id, "file_path": f"/kb/{doc_id}.md", "metadata": {}}


class _LiveRag:
    """LightRAG stand-in whose physical delete removes the doc's chunk node in
    Memgraph and — like the real rebuild — leaves the entity refs behind."""

    def __init__(self, workspace: str, memberships, chunks) -> None:
        self.workspace = workspace
        self.doc_status = _LiveDocStatus(memberships, chunks)
        self.cache_cleared = 0

    async def adelete_by_doc_id(self, doc_id: str):
        from twindb_lightrag_memgraph import _pool

        chunk = self.doc_status.chunks[doc_id]
        async with _pool.get_session() as session:
            result = await session.run(
                f"MATCH (k:`KV_{self.workspace}_text_chunks` {{id: $id}}) DETACH DELETE k",
                id=chunk,
            )
            await result.consume()
        self.doc_status.memberships.pop(doc_id, None)
        return None

    async def aclear_cache(self) -> None:
        self.cache_cleared += 1


@pytest.mark.integration
async def test_bulk_batch_settles_dead_refs_with_one_real_sweep(monkeypatch):
    from twindb_lightrag_memgraph import _import_cleanup, _pool, _twindb_state
    from twindb_lightrag_memgraph.server import folder as folder_mod
    from twindb_lightrag_memgraph.server import webui_router

    ws = f"bulkhyg_{uuid.uuid4().hex[:8]}"
    kv = f"KV_{ws}_text_chunks"

    async def _run(session, query, **params):
        result = await session.run(query, **params)
        await result.consume()

    async with _pool.get_session() as session:
        await _run(
            session,
            f"CREATE (:`{kv}` {{id: 'c-a'}}), (:`{kv}` {{id: 'c-b'}}), (:`{kv}` {{id: 'c-keep'}})",
        )
        await _run(session, f"CREATE (:`{ws}` {{entity_id: 'E-a', source_id: 'c-a'}})")
        await _run(
            session, f"CREATE (:`{ws}` {{entity_id: 'E-ab', source_id: 'c-a{SEP}c-b'}})"
        )
        await _run(
            session,
            f"CREATE (:`{ws}` {{entity_id: 'E-mixed', source_id: 'c-b{SEP}c-keep'}})",
        )
        await _run(
            session, f"CREATE (:`{ws}` {{entity_id: 'E-keep', source_id: 'c-keep'}})"
        )

    rag = _LiveRag(
        ws, {"doc-a": [FOLDER], "doc-b": [FOLDER]}, {"doc-a": "c-a", "doc-b": "c-b"}
    )
    real_sweep = graph_reader.sweep_stale_source_refs
    sweeps: list[dict] = []

    async def _counting_sweep(workspace: str):
        counters = await real_sweep(workspace)
        sweeps.append(counters)
        return counters

    async def _no_cleanup(paths):
        return None

    monkeypatch.setattr(graph_reader, "sweep_stale_source_refs", _counting_sweep)
    monkeypatch.setattr(_import_cleanup, "cleanup_import_paths", _no_cleanup)
    _twindb_state["rag"] = rag
    token = folder_mod._active_folder_id.set(FOLDER)
    try:
        results, failed, busy = await rd._run_bulk_delete_batch(
            webui_router, rag, ["doc-a", "doc-b"]
        )
        assert (failed, busy) == ([], [])
        assert [r["physically_deleted"] for r in results] == [True, True]
        # ONE sweep and ONE cache purge for the whole batch …
        assert len(sweeps) == 1 and rag.cache_cleared == 1
        # … and it settled everything both deletes left behind: the entities
        # whose refs ALL died are gone, the mixed one is rewritten, the
        # untouched one untouched.
        assert sweeps[0]["entities_removed"] == 2
        assert sweeps[0]["entities_rewritten"] == 1
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{ws}`) RETURN n.entity_id AS id, n.source_id AS s ORDER BY id"
            )
            rows = {r["id"]: r["s"] async for r in result}
            await result.consume()
        assert rows == {"E-keep": "c-keep", "E-mixed": "c-keep"}
    finally:
        folder_mod._active_folder_id.reset(token)
        _twindb_state.pop("rag", None)
        async with _pool.get_session() as session:
            for label in (kv, ws):
                await _run(session, f"MATCH (n:`{label}`) DETACH DELETE n")
