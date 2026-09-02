"""source_id hygiene sweep (OVH audit 2026-07-28 §4) — unit + integration.

LightRAG's post-delete rebuild leaves dead chunk refs in entity/relation
``source_id`` forever; the Twin sweep purges them after every single-document
delete, and once at the end of a bulk batch that physically deleted at least
one document. Unit tier: the pure partition and the fail-soft wiring (a sweep
failure must never fail the user's delete). Integration tier (real
Memgraph): rewrite / removal / untouched entities and relations, with the
MG-2 vector-row cascade.
"""

from __future__ import annotations

import uuid

import pytest

from twindb_lightrag_memgraph.server import graph_reader

SEP = "<SEP>"


class TestPartitionSourceRefs:
    def test_dead_refs_pruned_order_preserved(self):
        kept, changed = graph_reader._partition_source_refs(
            f"a{SEP}dead{SEP}b", {"a", "b"}, SEP
        )
        assert (kept, changed) == (["a", "b"], True)

    def test_untouched_when_all_live(self):
        kept, changed = graph_reader._partition_source_refs(f"a{SEP}b", {"a", "b"}, SEP)
        assert (kept, changed) == (["a", "b"], False)

    def test_duplicates_collapse_and_flag_change(self):
        kept, changed = graph_reader._partition_source_refs(
            f"a{SEP}a{SEP}b", {"a", "b"}, SEP
        )
        assert (kept, changed) == (["a", "b"], True)

    def test_all_dead_returns_empty(self):
        kept, changed = graph_reader._partition_source_refs(f"x{SEP}y", {"a"}, SEP)
        assert (kept, changed) == ([], True)

    def test_empty_segments_ignored_without_rewrite(self):
        # Empty segments are separator noise, not dead refs: they are
        # dropped from the comparison basis, so a source_id whose only
        # oddity is empty segments triggers NO write.
        kept, changed = graph_reader._partition_source_refs(
            f"{SEP}a{SEP}{SEP}", {"a"}, SEP
        )
        assert (kept, changed) == (["a"], False)


class _FakeStatus:
    status = "success"
    message = None


class _FakeRag:
    workspace = "hygiene_unit_ws"

    async def adelete_by_doc_id(self, doc_id):
        return _FakeStatus()


async def test_delete_helper_survives_sweep_failure(monkeypatch):
    """OVH §4 wiring contract: best-effort — the delete must succeed even
    when the sweep blows up, and the sweep must have been attempted."""
    from twindb_lightrag_memgraph.server.webui import router

    async def _noop_purge(rag):
        return None

    calls: list[str] = []

    async def _boom(workspace):
        calls.append(workspace)
        raise RuntimeError("memgraph down mid-sweep")

    monkeypatch.setattr(router, "_purge_query_llm_cache", _noop_purge)
    monkeypatch.setattr(graph_reader, "sweep_stale_source_refs", _boom)

    await router._delete_doc_from_rag(_FakeRag(), "doc-1")
    assert calls == ["hygiene_unit_ws"]


async def test_physical_delete_cleans_captured_source_path(monkeypatch):
    """The filename must disappear after the authoritative delete succeeds."""
    from twindb_lightrag_memgraph import _import_cleanup
    from twindb_lightrag_memgraph.server.webui import router

    class _DocStatus:
        async def get_by_id(self, doc_id):
            assert doc_id == "doc-1"
            return {"id": doc_id, "file_path": "/inputs/demo/demo.qa.pdf"}

    rag = _FakeRag()
    rag.doc_status = _DocStatus()
    cleaned: list[str] = []

    async def _capture(paths):
        cleaned.extend(paths)

    async def _noop(_rag):
        return None

    async def _noop_sweep(_workspace):
        return None

    monkeypatch.setattr(_import_cleanup, "cleanup_import_paths", _capture)
    monkeypatch.setattr(router, "_purge_query_llm_cache", _noop)
    monkeypatch.setattr(graph_reader, "request_source_ref_sweep", _noop_sweep)

    await router._delete_doc_from_rag(rag, "doc-1")

    assert cleaned == ["/inputs/demo/demo.qa.pdf"]


async def test_concurrent_sweep_requests_coalesce(monkeypatch):
    """A burst of N delete-triggered requests costs at most 2 sweeps: the
    holder's initial run plus ONE dirty re-run — never N full scans."""
    import asyncio

    runs: list[str] = []

    async def _slow_sweep(workspace):
        runs.append(workspace)
        await asyncio.sleep(0.05)
        return {"entities_rewritten": 0}

    monkeypatch.setattr(graph_reader, "sweep_stale_source_refs", _slow_sweep)
    ws = f"coalesce_{id(monkeypatch)}"

    results = await asyncio.gather(
        *(graph_reader.request_source_ref_sweep(ws) for _ in range(25))
    )
    assert len(runs) == 2
    # Exactly one caller held the lock and got counters; the rest
    # returned immediately with None after marking the workspace dirty.
    assert sum(1 for r in results if r is not None) == 1


@pytest.mark.integration
async def test_sweep_purges_dead_refs_end_to_end():
    """Real Memgraph: rewrite, removal (+ MG-2 vector cascade), untouched."""
    ws = f"hyg_{uuid.uuid4().hex[:8]}"
    # Raw-Cypher seed with explicit labels; _pool self-initializes from
    # MEMGRAPH_URI — no register()/workspace env needed here.
    from twindb_lightrag_memgraph import _pool

    kv = f"KV_{ws}_text_chunks"
    vec_e = f"Vec_{ws}_entities"
    vec_r = f"Vec_{ws}_relationships"

    async def _run(session, query, **params):
        result = await session.run(query, **params)
        await result.consume()

    async with _pool.get_session() as session:
        await _run(session, f"CREATE (:`{kv}` {{id: 'c-live'}})")
        await _run(
            session,
            f"CREATE (:`{ws}` {{entity_id: 'E1', source_id: 'c-live{SEP}c-dead'}})",
        )
        await _run(
            session, f"CREATE (:`{ws}` {{entity_id: 'E2', source_id: 'c-dead'}})"
        )
        await _run(
            session, f"CREATE (:`{ws}` {{entity_id: 'E3', source_id: 'c-live'}})"
        )
        await _run(
            session, f"CREATE (:`{ws}` {{entity_id: 'E4', source_id: 'c-live'}})"
        )
        await _run(
            session,
            f"MATCH (a:`{ws}` {{entity_id: 'E1'}}), (b:`{ws}` {{entity_id: 'E3'}}) "
            f"CREATE (a)-[:DIRECTED {{source_id: 'c-live{SEP}c-dead'}}]->(b)",
        )
        await _run(
            session,
            f"MATCH (a:`{ws}` {{entity_id: 'E3'}}), (b:`{ws}` {{entity_id: 'E4'}}) "
            "CREATE (a)-[:DIRECTED {source_id: 'c-dead'}]->(b)",
        )
        # M2: an edge touching the to-be-removed E2 must be skipped by the
        # relation pass (dropped by E2's DETACH DELETE) and its vector row
        # covered by the entity cascade's relationship statement.
        await _run(
            session,
            f"MATCH (a:`{ws}` {{entity_id: 'E2'}}), (b:`{ws}` {{entity_id: 'E3'}}) "
            "CREATE (a)-[:DIRECTED {source_id: 'c-dead'}]->(b)",
        )
        await _run(session, f"CREATE (:`{vec_r}` {{src_id: 'E2', tgt_id: 'E3'}})")
        await _run(session, f"CREATE (:`{vec_e}` {{entity_name: 'E2'}})")
        await _run(session, f"CREATE (:`{vec_e}` {{entity_name: 'E1'}})")
        await _run(session, f"CREATE (:`{vec_r}` {{src_id: 'E3', tgt_id: 'E4'}})")

    try:
        counters = await graph_reader.sweep_stale_source_refs(ws)
        assert counters == {
            "entities_rewritten": 1,
            "entities_removed": 1,
            "relations_rewritten": 1,
            "relations_removed": 1,
            "relations_dropped_with_entities": 1,
        }

        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{ws}`) RETURN n.entity_id AS id, "
                "n.source_id AS source_id ORDER BY id"
            )
            rows = {r["id"]: r["source_id"] async for r in result}
            assert rows == {"E1": "c-live", "E3": "c-live", "E4": "c-live"}

            result = await session.run(
                f"MATCH (:`{ws}`)-[r]->(:`{ws}`) RETURN r.source_id AS s"
            )
            edges = [r["s"] async for r in result]
            assert edges == ["c-live"]

            result = await session.run(
                f"MATCH (v:`{vec_e}`) RETURN v.entity_name AS name"
            )
            vec_names = {r["name"] async for r in result}
            assert vec_names == {"E1"}

            result = await session.run(f"MATCH (v:`{vec_r}`) RETURN count(v) AS c")
            record = await result.single()
            assert record["c"] == 0

        # Idempotence: a second pass finds nothing to do.
        assert await graph_reader.sweep_stale_source_refs(ws) == {
            "entities_rewritten": 0,
            "entities_removed": 0,
            "relations_rewritten": 0,
            "relations_removed": 0,
            "relations_dropped_with_entities": 0,
        }
    finally:
        async with _pool.get_session() as session:
            for label in (ws, kv, vec_e, vec_r):
                await _run(session, f"MATCH (n:`{label}`) DETACH DELETE n")
