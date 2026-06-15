"""Integration tests for batch graph methods (get_nodes_batch, get_edges_batch, etc.).

Verifies that the patched MemgraphStorage batch methods correctly retrieve
nodes and edges from Memgraph. Requires running Memgraph instance.
"""

import os
import pytest

# Skip if no Memgraph URI
pytestmark = pytest.mark.skipif(
    not os.environ.get("MEMGRAPH_URI"),
    reason="MEMGRAPH_URI not set",
)

import twindb_lightrag_memgraph

twindb_lightrag_memgraph.register()

from lightrag.kg.memgraph_impl import MemgraphStorage
from lightrag.kg.shared_storage import initialize_share_data

initialize_share_data(workers=1)


@pytest.fixture
async def graph():
    """Create a MemgraphStorage instance and seed test data."""
    g = MemgraphStorage(
        namespace="test_batch",
        global_config={"workspace": "test_batch"},
        embedding_func=None,
    )
    await g.initialize()

    ws = g._get_workspace_label()

    # Clean up any prior test data
    async with g._driver.session() as session:
        await session.run(f"MATCH (n:`{ws}`) DETACH DELETE n")

    # Seed nodes
    await g.upsert_node("ALICE", {
        "entity_id": "ALICE",
        "entity_type": "Person",
        "description": "A researcher",
        "source_id": "chunk1",
    })
    await g.upsert_node("BOB", {
        "entity_id": "BOB",
        "entity_type": "Person",
        "description": "A developer",
        "source_id": "chunk1",
    })
    await g.upsert_node("PROJECT_X", {
        "entity_id": "PROJECT_X",
        "entity_type": "Project",
        "description": "A secret project",
        "source_id": "chunk2",
    })

    # Seed edges
    await g.upsert_edge("ALICE", "BOB", {
        "weight": "1.0",
        "description": "colleagues",
        "keywords": "work",
        "source_id": "chunk1",
    })
    await g.upsert_edge("ALICE", "PROJECT_X", {
        "weight": "0.8",
        "description": "works on",
        "keywords": "project",
        "source_id": "chunk2",
    })

    yield g

    # Cleanup
    async with g._driver.session() as session:
        await session.run(f"MATCH (n:`{ws}`) DETACH DELETE n")


class TestGetNodesBatch:
    async def test_returns_all_nodes(self, graph):
        result = await graph.get_nodes_batch(["ALICE", "BOB", "PROJECT_X"])
        assert len(result) == 3
        assert "ALICE" in result
        assert result["ALICE"]["entity_type"] == "Person"

    async def test_missing_node_excluded(self, graph):
        result = await graph.get_nodes_batch(["ALICE", "NONEXISTENT"])
        assert len(result) == 1
        assert "ALICE" in result

    async def test_empty_input(self, graph):
        result = await graph.get_nodes_batch([])
        assert result == {}


class TestGetEdgesBatch:
    async def test_returns_edge(self, graph):
        pairs = [{"src": "ALICE", "tgt": "BOB"}]
        result = await graph.get_edges_batch(pairs)
        assert len(result) == 1
        key = ("ALICE", "BOB")
        assert key in result
        assert result[key]["description"] == "colleagues"

    async def test_reverse_direction_also_matches(self, graph):
        """Undirected match: (BOB)-[r]-(ALICE) should also find the edge."""
        pairs = [{"src": "BOB", "tgt": "ALICE"}]
        result = await graph.get_edges_batch(pairs)
        assert len(result) == 1
        assert ("BOB", "ALICE") in result

    async def test_missing_edge(self, graph):
        pairs = [{"src": "BOB", "tgt": "PROJECT_X"}]
        result = await graph.get_edges_batch(pairs)
        assert len(result) == 0

    async def test_empty_input(self, graph):
        result = await graph.get_edges_batch([])
        assert result == {}


class TestNodeDegreesBatch:
    async def test_degrees(self, graph):
        result = await graph.node_degrees_batch(["ALICE", "BOB", "PROJECT_X"])
        # ALICE has 2 edges (to BOB and PROJECT_X)
        assert result["ALICE"] == 2
        # BOB has 1 edge (to ALICE)
        assert result["BOB"] == 1
        # PROJECT_X has 1 edge (to ALICE)
        assert result["PROJECT_X"] == 1

    async def test_missing_node_gets_zero(self, graph):
        result = await graph.node_degrees_batch(["ALICE", "NONEXISTENT"])
        assert result["NONEXISTENT"] == 0


class TestGetNodesEdgesBatch:
    async def test_returns_connected_pairs(self, graph):
        result = await graph.get_nodes_edges_batch(["ALICE"])
        assert "ALICE" in result
        edges = result["ALICE"]
        assert len(edges) == 2

    async def test_missing_node_gets_empty(self, graph):
        result = await graph.get_nodes_edges_batch(["NONEXISTENT"])
        assert result["NONEXISTENT"] == []


class TestGetNodesWithDegreesBatch:
    async def test_fused_query(self, graph):
        nodes, degrees = await graph.get_nodes_with_degrees_batch(
            ["ALICE", "BOB"]
        )
        assert len(nodes) == 2
        assert nodes["ALICE"]["entity_type"] == "Person"
        assert degrees["ALICE"] == 2
        assert degrees["BOB"] == 1


class TestGetEdgesWithDegreesBatch:
    async def test_fused_query(self, graph):
        pairs = [{"src": "ALICE", "tgt": "BOB"}]
        edges, degrees = await graph.get_edges_with_degrees_batch(pairs)
        assert len(edges) == 1
        assert ("ALICE", "BOB") in edges
        assert edges[("ALICE", "BOB")]["description"] == "colleagues"
        # degree = degree(ALICE) + degree(BOB) = 2 + 1 = 3
        assert degrees[("ALICE", "BOB")] == 3
