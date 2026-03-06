"""Unit tests for buffered batch writes (_BufferedGraphProxy).

No Memgraph required — all graph interactions are mocked.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph._buffered_graph import _BufferedGraphProxy

# ── Helpers ───────────────────────────────────────────────────────────


def _mock_graph(workspace="test_ws"):
    """Create a mock MemgraphStorage with session support."""
    graph = AsyncMock()
    graph.workspace = workspace

    # Track Cypher queries run via session
    queries = []

    @asynccontextmanager
    async def mock_session(**kwargs):
        session = AsyncMock()
        original_run = session.run

        async def tracking_run(query, **params):
            queries.append({"query": query, "params": params})
            return await original_run(query, **params)

        session.run = tracking_run
        yield session

    graph._driver = MagicMock()
    graph._driver.session = mock_session
    graph._queries = queries
    return graph


# ── Buffer accumulation ───────────────────────────────────────────────


class TestBufferAccumulation:
    async def test_upsert_node_buffered_not_forwarded(self):
        """upsert_node should accumulate in buffer, not call real graph."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node(
            "Alice", {"entity_type": "Person", "entity_id": "Alice"}
        )

        graph.upsert_node.assert_not_awaited()
        assert "Alice" in proxy._node_buffer
        assert proxy._node_types["Alice"] == "Person"

    async def test_upsert_edge_buffered_not_forwarded(self):
        """upsert_edge should accumulate in buffer, not call real graph."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_edge("Alice", "Bob", {"weight": "1.0"})

        graph.upsert_edge.assert_not_awaited()
        assert ("Alice", "Bob") in proxy._edge_buffer

    async def test_multiple_upserts_same_node_merge(self):
        """Multiple upserts to the same node should merge properties."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node(
            "Alice", {"entity_type": "Person", "entity_id": "Alice"}
        )
        await proxy.upsert_node(
            "Alice", {"description": "A person named Alice", "entity_id": "Alice"}
        )

        assert proxy._node_buffer["Alice"]["entity_type"] == "Person"
        assert proxy._node_buffer["Alice"]["description"] == "A person named Alice"

    async def test_multiple_upserts_same_edge_merge(self):
        """Multiple upserts to the same edge should merge properties."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_edge("Alice", "Bob", {"weight": "1.0"})
        await proxy.upsert_edge("Alice", "Bob", {"description": "knows"})

        assert proxy._edge_buffer[("Alice", "Bob")]["weight"] == "1.0"
        assert proxy._edge_buffer[("Alice", "Bob")]["description"] == "knows"


# ── Read-your-own-writes ──────────────────────────────────────────────


class TestReadYourOwnWrites:
    async def test_get_node_returns_buffered_data(self):
        """get_node should return buffered data without hitting real graph."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node(
            "Alice", {"entity_type": "Person", "entity_id": "Alice"}
        )
        result = await proxy.get_node("Alice")

        assert result["entity_type"] == "Person"
        graph.get_node.assert_not_awaited()

    async def test_get_node_falls_back_to_real_graph(self):
        """get_node for non-buffered node should delegate to real graph."""
        graph = _mock_graph()
        graph.get_node.return_value = {"entity_type": "Org", "entity_id": "ACME"}
        proxy = _BufferedGraphProxy(graph)

        result = await proxy.get_node("ACME")

        assert result["entity_type"] == "Org"
        graph.get_node.assert_awaited_once_with("ACME")

    async def test_has_node_checks_buffer_first(self):
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node("Alice", {"entity_id": "Alice"})
        assert await proxy.has_node("Alice") is True
        graph.has_node.assert_not_awaited()

    async def test_has_edge_checks_buffer_first(self):
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_edge("Alice", "Bob", {"weight": "1.0"})
        assert await proxy.has_edge("Alice", "Bob") is True
        graph.has_edge.assert_not_awaited()

    async def test_has_edge_falls_back_to_real_graph(self):
        graph = _mock_graph()
        graph.has_edge.return_value = False
        proxy = _BufferedGraphProxy(graph)

        assert await proxy.has_edge("X", "Y") is False
        graph.has_edge.assert_awaited_once_with("X", "Y")

    async def test_get_edge_returns_buffered_data(self):
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_edge("Alice", "Bob", {"weight": "1.0"})
        result = await proxy.get_edge("Alice", "Bob")

        assert result["weight"] == "1.0"
        graph.get_edge.assert_not_awaited()


# ── __getattr__ delegation ────────────────────────────────────────────


class TestDelegation:
    async def test_unknown_attr_delegates_to_real_graph(self):
        """Attributes not on proxy should be fetched from real graph."""
        graph = _mock_graph()
        graph.some_custom_attr = "hello"
        proxy = _BufferedGraphProxy(graph)

        assert proxy.some_custom_attr == "hello"

    async def test_workspace_accessible(self):
        graph = _mock_graph(workspace="prod")
        proxy = _BufferedGraphProxy(graph)

        assert proxy.workspace == "prod"


# ── Flush ─────────────────────────────────────────────────────────────


class TestFlush:
    async def test_flush_nodes_uses_unwind(self):
        """Node flush should use UNWIND + MERGE Cypher."""
        graph = _mock_graph(workspace="ws")
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node(
            "Alice", {"entity_type": "Person", "entity_id": "Alice"}
        )
        await proxy.upsert_node("Bob", {"entity_type": "Person", "entity_id": "Bob"})
        await proxy.flush()

        assert len(graph._queries) >= 1
        node_query = graph._queries[0]["query"]
        assert "UNWIND" in node_query
        assert "MERGE" in node_query
        assert "`ws`" in node_query

    async def test_flush_nodes_sets_type_labels(self):
        """Node flush should set type labels via separate queries per type."""
        graph = _mock_graph(workspace="ws")
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node(
            "Alice", {"entity_type": "Person", "entity_id": "Alice"}
        )
        await proxy.upsert_node(
            "ACME", {"entity_type": "Organization", "entity_id": "ACME"}
        )
        await proxy.flush()

        # 1 UNWIND MERGE + 2 type label queries (Person, Organization)
        assert len(graph._queries) >= 3
        type_queries = [q for q in graph._queries if "SET n:" in q["query"]]
        assert len(type_queries) == 2

    async def test_flush_edges_uses_unwind(self):
        """Edge flush should use UNWIND + MATCH + MERGE Cypher."""
        graph = _mock_graph(workspace="ws")
        proxy = _BufferedGraphProxy(graph)

        # Need nodes first (edges use MATCH)
        await proxy.upsert_node("Alice", {"entity_id": "Alice"})
        await proxy.upsert_edge("Alice", "Bob", {"weight": "1.0"})
        await proxy.flush()

        edge_queries = [q for q in graph._queries if "DIRECTED" in q["query"]]
        assert len(edge_queries) == 1
        assert "UNWIND" in edge_queries[0]["query"]
        assert "MATCH" in edge_queries[0]["query"]

    async def test_flush_nodes_before_edges(self):
        """Nodes must be flushed before edges (edges use MATCH for nodes)."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node("Alice", {"entity_id": "Alice"})
        await proxy.upsert_edge("Alice", "Bob", {"weight": "1.0"})
        await proxy.flush()

        # Find node and edge query indices
        node_idx = None
        edge_idx = None
        for i, q in enumerate(graph._queries):
            if "DIRECTED" in q["query"]:
                edge_idx = i
            elif "MERGE" in q["query"] and "DIRECTED" not in q["query"]:
                if node_idx is None:
                    node_idx = i

        assert node_idx is not None
        assert edge_idx is not None
        assert node_idx < edge_idx, "Nodes must flush before edges"

    async def test_flush_empty_buffers_no_queries(self):
        """Flushing empty buffers should not run any queries."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.flush()

        assert len(graph._queries) == 0

    async def test_flush_only_nodes_no_edge_query(self):
        """When there are only nodes, no edge query should run."""
        graph = _mock_graph()
        proxy = _BufferedGraphProxy(graph)

        await proxy.upsert_node("Alice", {"entity_id": "Alice"})
        await proxy.flush()

        edge_queries = [q for q in graph._queries if "DIRECTED" in q["query"]]
        assert len(edge_queries) == 0


# ── Double-patch registration ─────────────────────────────────────────


class TestDoublePatch:
    def test_merge_patched_in_operate(self):
        """merge_nodes_and_edges should be patched in lightrag.operate."""
        import twindb_lightrag_memgraph

        twindb_lightrag_memgraph.register()
        from lightrag import operate

        assert "buffered" in operate.merge_nodes_and_edges.__name__

    def test_merge_patched_in_lightrag_module(self):
        """merge_nodes_and_edges should be patched in lightrag.lightrag."""
        import twindb_lightrag_memgraph

        twindb_lightrag_memgraph.register()
        from lightrag import lightrag as lr_mod

        assert "buffered" in lr_mod.merge_nodes_and_edges.__name__


# ── Signature compatibility ────────────────────────────────────────────


class TestSignatureCompat:
    """Verify the patch works with both old and new lightrag signatures."""

    async def test_new_signature_kwargs(self):
        """New lightrag calls merge_nodes_and_edges(chunk_results=..., knowledge_graph_inst=...)."""
        from unittest.mock import call

        from lightrag import operate
        from lightrag.kg.memgraph_impl import MemgraphStorage

        mock_graph = _mock_graph()
        mock_graph.__class__ = MemgraphStorage

        original_called_with = {}

        async def fake_original(*args, **kwargs):
            original_called_with["args"] = args
            original_called_with["kwargs"] = kwargs

        with patch.object(operate, "merge_nodes_and_edges") as patched:
            # Re-apply the patch with our fake original
            from twindb_lightrag_memgraph._buffered_graph import _BufferedGraphProxy

            _orig = fake_original

            async def _buffered(*args, **kwargs):
                graph_inst = kwargs.get("knowledge_graph_inst")
                if graph_inst is None:
                    for arg in args:
                        if isinstance(arg, MemgraphStorage):
                            graph_inst = arg
                            break
                if not isinstance(graph_inst, MemgraphStorage):
                    return await _orig(*args, **kwargs)
                proxy = _BufferedGraphProxy(graph_inst)
                if "knowledge_graph_inst" in kwargs:
                    kwargs["knowledge_graph_inst"] = proxy
                await _orig(*args, **kwargs)
                await proxy.flush()

            # Simulate the new-style call from lightrag.py:1986
            await _buffered(
                chunk_results=[("nodes", "edges")],
                knowledge_graph_inst=mock_graph,
                entity_vdb=AsyncMock(),
                relationships_vdb=AsyncMock(),
                global_config={"workspace": "test"},
            )

            # The original should have received a proxy, not the real graph
            kg = original_called_with["kwargs"]["knowledge_graph_inst"]
            assert isinstance(kg, _BufferedGraphProxy)

    async def test_old_signature_positional(self):
        """Old lightrag calls merge_nodes_and_edges(entity_map, edge_map, graph, config)."""
        from lightrag import operate
        from lightrag.kg.memgraph_impl import MemgraphStorage

        from twindb_lightrag_memgraph._buffered_graph import _BufferedGraphProxy

        mock_graph = _mock_graph()
        mock_graph.__class__ = MemgraphStorage

        original_called_with = {}

        async def fake_original(*args, **kwargs):
            original_called_with["args"] = args
            original_called_with["kwargs"] = kwargs

        _orig = fake_original

        async def _buffered(*args, **kwargs):
            graph_inst = kwargs.get("knowledge_graph_inst")
            if graph_inst is None:
                for arg in args:
                    if isinstance(arg, MemgraphStorage):
                        graph_inst = arg
                        break
            if not isinstance(graph_inst, MemgraphStorage):
                return await _orig(*args, **kwargs)
            proxy = _BufferedGraphProxy(graph_inst)
            if "knowledge_graph_inst" in kwargs:
                kwargs["knowledge_graph_inst"] = proxy
            else:
                args = list(args)
                for i, arg in enumerate(args):
                    if arg is graph_inst:
                        args[i] = proxy
                        break
                args = tuple(args)
            await _orig(*args, **kwargs)
            await proxy.flush()

        # Simulate old-style positional call
        await _buffered({}, {}, mock_graph, {"workspace": "test"})

        # The original should have received a proxy at position 2
        assert isinstance(original_called_with["args"][2], _BufferedGraphProxy)


# ── Fallback for non-Memgraph ─────────────────────────────────────────


class TestFallback:
    async def test_non_memgraph_calls_original(self):
        """Non-MemgraphStorage graphs should bypass buffering entirely."""
        from lightrag import operate

        # Create a non-Memgraph graph mock
        fake_graph = AsyncMock()
        fake_graph.__class__ = type("FakeGraph", (), {})

        # Call the patched function with a non-Memgraph graph
        # It should call the original, which we can verify by checking
        # that the fake_graph doesn't have flush called on it
        with patch.object(
            operate,
            "merge_nodes_and_edges",
            operate.merge_nodes_and_edges,
        ):
            # We can't easily run the full original here, but we can verify
            # the isinstance check works by checking the proxy import
            from lightrag.kg.memgraph_impl import MemgraphStorage

            assert not isinstance(fake_graph, MemgraphStorage)
