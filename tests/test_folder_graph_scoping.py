"""Folder cloisonnement of the KG graph reads — completes batch 2.

Scoping the vector *selection* (``vector_impl.query``) is not enough: LightRAG
re-expands the graph after selection via the patched batch methods on
``MemgraphStorage``. Without scoping, a folder-A entity pulls edges / neighbours /
descriptions from folder B back into the prompt context. These tests pin the
graph-read cloisonnement so "no cross-folder context enters the prompt" holds for
the KG modes (hybrid/local/global).

Product-owner decisions verified here:

- ``get_nodes_batch`` returns only nodes with ≥1 source chunk in the folder.
- ``get_edges_batch`` returns only edges with ≥1 source chunk in the folder.
- ``get_nodes_edges_batch`` keeps only in-folder edges → only neighbours reached
  by them.
- degrees are **folder-scoped** (count in-folder edges), not global.
- with no active folder the reads are the legacy global ones (strict compat).

Integration only (real Memgraph; auto-skipped when ``MEMGRAPH_URI`` is unset).
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph import _pool, register

register()

from twindb_lightrag_memgraph._constants import storage_folder_context  # noqa: E402
from twindb_lightrag_memgraph.patches.registry import (  # noqa: E402
    _GRAPH_FIELD_SEP,
    _patched_edge_degrees_batch,
    _patched_get_edges_batch,
    _patched_get_edges_with_degrees_batch,
    _patched_get_nodes_batch,
    _patched_get_nodes_edges_batch,
    _patched_get_nodes_with_degrees_batch,
    _patched_node_degrees_batch,
)
from twindb_lightrag_memgraph.docstatus_impl import (  # noqa: E402
    MemgraphDocStatusStorage,
)

pytestmark = pytest.mark.integration

_WS = "folder_graph_test"
SEP = _GRAPH_FIELD_SEP


class _GraphStub:
    """Minimal stand-in carrying just what the patched methods read."""

    def __init__(self, driver, database):
        self._driver = driver
        self._DATABASE = database

    def _get_workspace_label(self) -> str:
        return _WS

    get_nodes_batch = _patched_get_nodes_batch
    node_degrees_batch = _patched_node_degrees_batch
    edge_degrees_batch = _patched_edge_degrees_batch
    get_edges_batch = _patched_get_edges_batch
    get_nodes_edges_batch = _patched_get_nodes_edges_batch
    get_nodes_with_degrees_batch = _patched_get_nodes_with_degrees_batch
    get_edges_with_degrees_batch = _patched_get_edges_with_degrees_batch


def _doc(doc_id: str) -> dict:
    return {
        "id": doc_id,
        "status": "processed",
        "content_summary": "x",
        "content_length": 1,
        "file_path": f"{doc_id}.md",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "content_hash": doc_id,
    }


@pytest.fixture
async def graph(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", _WS)
    ds = MemgraphDocStatusStorage(
        namespace="doc_status", global_config={}, embedding_func=None
    )
    await ds.initialize()

    async def _wipe():
        async with _pool.get_session() as s:
            for lbl in (
                f"DocStatus_{_WS}",
                f"Folder_{_WS}",
                f"Vec_{_WS}_chunks",
                _WS,
            ):
                await (await s.run(f"MATCH (n:`{lbl}`) DETACH DELETE n")).consume()

    await _wipe()

    # docs: doc-a ∈ folder A, doc-b ∈ folder B
    with storage_folder_context("A"):
        await ds.upsert({"doc-a": _doc("doc-a")})
    with storage_folder_context("B"):
        await ds.upsert({"doc-b": _doc("doc-b")})

    async with _pool.get_session() as s:
        # chunks: chunk-a owned by doc-a, chunk-b owned by doc-b
        await (
            await s.run(
                f"CREATE (:`Vec_{_WS}_chunks` {{id: 'chunk-a', full_doc_id: 'doc-a'}}) "
                f"CREATE (:`Vec_{_WS}_chunks` {{id: 'chunk-b', full_doc_id: 'doc-b'}})"
            )
        ).consume()
        # graph entities: ent-A (src chunk-a), ent-B (src chunk-b),
        # ent-shared (src chunk-a<SEP>chunk-b)
        await (
            await s.run(
                f"CREATE (:`{_WS}` {{entity_id: 'ent-A', source_id: 'chunk-a'}}) "
                f"CREATE (:`{_WS}` {{entity_id: 'ent-B', source_id: 'chunk-b'}}) "
                f"CREATE (:`{_WS}` {{entity_id: 'ent-shared', "
                f"source_id: 'chunk-a{SEP}chunk-b'}})"
            )
        ).consume()
        # edges: r_ab (src chunk-a) A-side, r_b (src chunk-b) B-side
        await (
            await s.run(
                f"MATCH (a:`{_WS}` {{entity_id:'ent-A'}}), "
                f"(sh:`{_WS}` {{entity_id:'ent-shared'}}) "
                f"CREATE (a)-[:REL {{source_id:'chunk-a'}}]->(sh)"
            )
        ).consume()
        await (
            await s.run(
                f"MATCH (b:`{_WS}` {{entity_id:'ent-B'}}), "
                f"(sh:`{_WS}` {{entity_id:'ent-shared'}}) "
                f"CREATE (b)-[:REL {{source_id:'chunk-b'}}]->(sh)"
            )
        ).consume()

    driver, database = await _pool.get_driver()
    yield _GraphStub(driver, database)
    await _wipe()


# ── get_nodes_batch ───────────────────────────────────────────────────────


async def test_get_nodes_batch_scopes_by_member_source_chunk(graph):
    ids = ["ent-A", "ent-B", "ent-shared"]

    # No folder → all nodes (legacy global behaviour).
    assert set((await graph.get_nodes_batch(ids)).keys()) == set(ids)

    with storage_folder_context("A"):
        nodes = await graph.get_nodes_batch(ids)
    assert set(nodes) == {"ent-A", "ent-shared"}  # ent-B is B-only

    with storage_folder_context("B"):
        nodes = await graph.get_nodes_batch(ids)
    assert set(nodes) == {"ent-B", "ent-shared"}

    with storage_folder_context("C"):
        assert await graph.get_nodes_batch(ids) == {}


# ── get_edges_batch ───────────────────────────────────────────────────────


async def test_get_edges_batch_scopes_by_member_source_chunk(graph):
    pairs = [
        {"src": "ent-A", "tgt": "ent-shared"},
        {"src": "ent-B", "tgt": "ent-shared"},
    ]
    with storage_folder_context("A"):
        edges = await graph.get_edges_batch(pairs)
    assert ("ent-A", "ent-shared") in edges
    assert ("ent-B", "ent-shared") not in edges  # B-only edge excluded

    with storage_folder_context("B"):
        edges = await graph.get_edges_batch(pairs)
    assert ("ent-B", "ent-shared") in edges
    assert ("ent-A", "ent-shared") not in edges


# ── get_nodes_edges_batch (neighbours via in-folder edges only) ───────────


async def test_get_nodes_edges_batch_only_in_folder_neighbours(graph):
    with storage_folder_context("A"):
        edges = await graph.get_nodes_edges_batch(["ent-shared"])
    # ent-shared reaches ent-A via the chunk-a edge (in A) but NOT ent-B
    # (its edge is chunk-b, folder B).
    neighbours = {pair[1] for pair in edges["ent-shared"]}
    assert "ent-A" in neighbours
    assert "ent-B" not in neighbours

    # No folder → both neighbours visible.
    edges = await graph.get_nodes_edges_batch(["ent-shared"])
    neighbours = {pair[1] for pair in edges["ent-shared"]}
    assert {"ent-A", "ent-B"} <= neighbours


# ── degrees are folder-scoped ─────────────────────────────────────────────


async def test_node_degrees_are_folder_scoped(graph):
    # ent-shared has 2 edges globally (to A and B); scoped to A only 1 counts.
    assert (await graph.node_degrees_batch(["ent-shared"]))["ent-shared"] == 2
    with storage_folder_context("A"):
        assert (await graph.node_degrees_batch(["ent-shared"]))["ent-shared"] == 1
    with storage_folder_context("C"):
        assert (await graph.node_degrees_batch(["ent-shared"]))["ent-shared"] == 0


# ── fused variants stay consistent ────────────────────────────────────────


async def test_fused_nodes_with_degrees_scoped(graph):
    with storage_folder_context("A"):
        nodes, degrees = await graph.get_nodes_with_degrees_batch(
            ["ent-A", "ent-B", "ent-shared"]
        )
    assert set(nodes) == {"ent-A", "ent-shared"}
    assert degrees["ent-shared"] == 1  # scoped degree


async def test_fused_edges_with_degrees_scoped(graph):
    pairs = [
        {"src": "ent-A", "tgt": "ent-shared"},
        {"src": "ent-B", "tgt": "ent-shared"},
    ]
    with storage_folder_context("A"):
        edge_data, edge_degrees = await graph.get_edges_with_degrees_batch(pairs)
    assert ("ent-A", "ent-shared") in edge_data
    assert ("ent-B", "ent-shared") not in edge_data
    # scoped node degrees: ent-A=1 (its only edge is in A), ent-shared=1
    assert edge_degrees[("ent-A", "ent-shared")] == 2
