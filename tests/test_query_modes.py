"""Smoke tests for all 4 query modes (local, global, hybrid, mix).

Verifies that operate._perform_kg_search runs without AttributeError
for each mode, using mocked graph with our batch patches.
No Memgraph, no LLM required.
"""

from unittest.mock import AsyncMock

import pytest
from lightrag.base import QueryParam

import twindb_lightrag_memgraph

twindb_lightrag_memgraph.register()

from lightrag import operate

# ── Helpers ───────────────────────────────────────────────────────────


def _mock_graph():
    """Mock graph with all batch methods our patches add."""
    graph = AsyncMock()
    graph.workspace = "test_ws"

    # Batch read methods
    graph.get_nodes_batch = AsyncMock(return_value={})
    graph.node_degrees_batch = AsyncMock(return_value={})
    graph.get_edges_batch = AsyncMock(return_value={})
    graph.edge_degrees_batch = AsyncMock(return_value={})
    graph.get_nodes_edges_batch = AsyncMock(return_value={})

    # Fused methods
    graph.get_nodes_with_degrees_batch = AsyncMock(return_value=({}, {}))
    graph.get_edges_with_degrees_batch = AsyncMock(return_value=({}, {}))

    return graph


def _mock_vdb(results=None):
    """Mock vector DB returning pre-set results."""
    vdb = AsyncMock()
    vdb.query = AsyncMock(return_value=results or [])
    vdb.cosine_better_than_threshold = 0.2
    return vdb


def _mock_text_chunks():
    """Mock text_chunks KV storage."""
    kv = AsyncMock()
    kv.global_config = {"kg_chunk_pick_method": "ORDER"}
    kv.embedding_func = None
    return kv


async def _search(mode, graph=None, entities_vdb=None, rels_vdb=None, chunks_vdb=None):
    """Shortcut to call _perform_kg_search with defaults."""
    graph = graph or _mock_graph()
    return await operate._perform_kg_search(
        query="test query",
        ll_keywords="test" if mode in ("local", "hybrid", "mix") else "",
        hl_keywords="test" if mode in ("global", "hybrid", "mix") else "",
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb or _mock_vdb(),
        relationships_vdb=rels_vdb or _mock_vdb(),
        text_chunks_db=_mock_text_chunks(),
        query_param=QueryParam(mode=mode),
        chunks_vdb=chunks_vdb,
    )


# ── Tests ─────────────────────────────────────────────────────────────


class TestLocalMode:
    async def test_local_empty(self):
        """Local mode with no VDB hits should return empty without error."""
        result = await _search("local")
        assert result["final_entities"] == []
        assert result["final_relations"] == []

    async def test_local_with_entities(self):
        """Local mode should traverse the full node→edge batch path."""
        graph = _mock_graph()
        graph.get_nodes_with_degrees_batch = AsyncMock(
            return_value=(
                {"ALICE": {"entity_type": "Person", "description": "A person"}},
                {"ALICE": 5},
            )
        )
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={"ALICE": [("ALICE", "BOB")]}
        )
        graph.get_edges_with_degrees_batch = AsyncMock(
            return_value=(
                {("ALICE", "BOB"): {"weight": "1.0", "description": "knows"}},
                {("ALICE", "BOB"): 8},
            )
        )

        entities_vdb = _mock_vdb(
            results=[{"entity_name": "ALICE", "created_at": "2024-01-01"}]
        )

        result = await _search("local", graph=graph, entities_vdb=entities_vdb)

        assert len(result["final_entities"]) == 1
        assert result["final_entities"][0]["entity_name"] == "ALICE"
        assert len(result["final_relations"]) == 1


class TestGlobalMode:
    async def test_global_empty(self):
        """Global mode with no VDB hits should return empty without error."""
        result = await _search("global")
        assert result["final_entities"] == []
        assert result["final_relations"] == []

    async def test_global_with_edges(self):
        """Global mode should call get_edges_batch and get_nodes_batch."""
        graph = _mock_graph()
        graph.get_edges_batch = AsyncMock(
            return_value={("ALICE", "BOB"): {"weight": "1.0", "description": "knows"}}
        )
        graph.get_nodes_batch = AsyncMock(
            return_value={
                "ALICE": {"entity_type": "Person", "description": "A"},
                "BOB": {"entity_type": "Person", "description": "B"},
            }
        )

        rels_vdb = _mock_vdb(
            results=[{"src_id": "ALICE", "tgt_id": "BOB", "created_at": "2024-01-01"}]
        )

        result = await _search("global", graph=graph, rels_vdb=rels_vdb)

        assert len(result["final_relations"]) == 1
        assert len(result["final_entities"]) == 2
        graph.get_edges_batch.assert_awaited()
        graph.get_nodes_batch.assert_awaited()


class TestHybridMode:
    async def test_hybrid_empty(self):
        """Hybrid mode with no hits should return empty without error."""
        result = await _search("hybrid")
        assert result["final_entities"] == []

    async def test_hybrid_calls_both_paths(self):
        """Hybrid mode should exercise both local and global paths."""
        graph = _mock_graph()
        graph.get_nodes_with_degrees_batch = AsyncMock(
            return_value=(
                {"ALICE": {"entity_type": "Person", "description": "A"}},
                {"ALICE": 3},
            )
        )
        graph.get_nodes_edges_batch = AsyncMock(return_value={"ALICE": []})

        entities_vdb = _mock_vdb(
            results=[{"entity_name": "ALICE", "created_at": "2024-01-01"}]
        )

        result = await _search(
            "hybrid", graph=graph, entities_vdb=entities_vdb, rels_vdb=_mock_vdb()
        )

        assert len(result["final_entities"]) == 1
        graph.get_nodes_edges_batch.assert_awaited()


class TestMixMode:
    async def test_mix_empty(self):
        """Mix mode with no hits should return empty without error."""
        result = await _search("mix", chunks_vdb=_mock_vdb())
        assert result["final_entities"] == []

    async def test_mix_vector_chunks(self):
        """Mix mode should include vector chunks."""
        chunks_vdb = _mock_vdb(
            results=[{"id": "c1", "content": "some text", "chunk_id": "c1"}]
        )
        result = await _search("mix", chunks_vdb=chunks_vdb)
        chunks_vdb.query.assert_awaited_once()


class TestBatchMethodsExist:
    """Verify that register() patches all required batch methods."""

    def test_memgraph_storage_has_batch_methods(self):
        from lightrag.kg.memgraph_impl import MemgraphStorage

        for method in [
            "get_nodes_batch",
            "node_degrees_batch",
            "get_edges_batch",
            "edge_degrees_batch",
            "get_nodes_edges_batch",
            "get_nodes_with_degrees_batch",
            "get_edges_with_degrees_batch",
        ]:
            assert hasattr(MemgraphStorage, method), f"Missing {method}"
