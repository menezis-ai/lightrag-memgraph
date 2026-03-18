"""
Tests for the batch graph operation monkey-patches.

Offline tests — mock the neo4j driver, no Memgraph connection needed.
Verifies that register() patches all 5 batch methods on MemgraphStorage
and that each returns the correct types from a single UNWIND query.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import twindb_lightrag_memgraph


def _reset():
    twindb_lightrag_memgraph._registered = False


def _make_storage(workspace="test_ws"):
    """Return a MemgraphStorage instance with a mocked driver."""
    from lightrag.kg.memgraph_impl import MemgraphStorage

    _reset()
    twindb_lightrag_memgraph.register()

    storage = MemgraphStorage.__new__(MemgraphStorage)
    storage.workspace = workspace
    storage._DATABASE = "memgraph"
    storage._driver = MagicMock()
    return storage


class _FakeRecord(dict):
    """Dict-like record returned by async iteration over neo4j results."""

    def __getitem__(self, key):
        return dict.__getitem__(self, key)


class _AsyncRecordIterator:
    """Async iterator wrapper around a list of records."""

    def __init__(self, records):
        self._records = list(records)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record


def _mock_session_with_records(records):
    """Create a mock driver.session() context manager yielding canned records.

    ``records`` is a list of dicts — each becomes a _FakeRecord yielded
    by async iteration, and returned by ``fetch()``.
    """
    fake_records = [_FakeRecord(r) for r in records]

    result = _AsyncRecordIterator(fake_records)
    result.consume = AsyncMock()
    result.fetch = AsyncMock(return_value=fake_records)

    session = AsyncMock()
    session.run = AsyncMock(return_value=result)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session = MagicMock(return_value=ctx)
    return driver, session


class TestBatchMethodsPatched:
    """Verify that register() actually replaces the 5 batch methods."""

    def test_get_nodes_batch_patched(self):
        from lightrag.base import BaseGraphStorage

        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        assert MemgraphStorage.get_nodes_batch is not BaseGraphStorage.get_nodes_batch

    def test_node_degrees_batch_patched(self):
        from lightrag.base import BaseGraphStorage

        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        assert (
            MemgraphStorage.node_degrees_batch
            is not BaseGraphStorage.node_degrees_batch
        )

    def test_get_edges_batch_patched(self):
        from lightrag.base import BaseGraphStorage

        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        assert MemgraphStorage.get_edges_batch is not BaseGraphStorage.get_edges_batch

    def test_edge_degrees_batch_patched(self):
        from lightrag.base import BaseGraphStorage

        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        assert (
            MemgraphStorage.edge_degrees_batch
            is not BaseGraphStorage.edge_degrees_batch
        )

    def test_get_nodes_edges_batch_patched(self):
        from lightrag.base import BaseGraphStorage

        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        assert (
            MemgraphStorage.get_nodes_edges_batch
            is not BaseGraphStorage.get_nodes_edges_batch
        )


class TestGetNodesBatch:
    async def test_returns_dict_of_nodes(self):
        storage = _make_storage()
        node_data = {"entity_id": "alice", "type": "person", "labels": ["test_ws"]}
        driver, session = _mock_session_with_records([{"eid": "alice", "n": node_data}])
        storage._driver = driver

        result = await storage.get_nodes_batch(["alice", "bob"])

        assert isinstance(result, dict)
        assert "alice" in result
        assert result["alice"]["type"] == "person"
        # Workspace label should be stripped from labels
        assert "test_ws" not in result["alice"]["labels"]
        # "bob" was not returned by the query => missing from result
        assert "bob" not in result

    async def test_empty_input(self):
        storage = _make_storage()
        result = await storage.get_nodes_batch([])
        assert result == {}

    async def test_single_query_executed(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records([])
        storage._driver = driver

        await storage.get_nodes_batch(["a", "b", "c"])

        # Exactly one session.run() call (single UNWIND query)
        session.run.assert_awaited_once()
        query = session.run.call_args[0][0]
        assert "UNWIND" in query


class TestNodeDegreesBatch:
    async def test_returns_degrees(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records(
            [
                {"eid": "alice", "degree": 3},
                {"eid": "bob", "degree": 1},
            ]
        )
        storage._driver = driver

        result = await storage.node_degrees_batch(["alice", "bob", "missing"])

        assert result["alice"] == 3
        assert result["bob"] == 1
        assert result["missing"] == 0  # default for missing nodes

    async def test_empty_input(self):
        storage = _make_storage()
        result = await storage.node_degrees_batch([])
        assert result == {}


class TestGetEdgesBatch:
    async def test_returns_edge_properties(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records(
            [
                {
                    "src": "alice",
                    "tgt": "bob",
                    "props": {"weight": 0.8, "description": "knows"},
                },
            ]
        )
        storage._driver = driver

        pairs = [{"src": "alice", "tgt": "bob"}]
        result = await storage.get_edges_batch(pairs)

        assert isinstance(result, dict)
        key = ("alice", "bob")
        assert key in result
        assert result[key]["weight"] == 0.8
        assert result[key]["description"] == "knows"
        # Default properties filled in
        assert result[key]["source_id"] is None
        assert result[key]["keywords"] is None

    async def test_fills_all_defaults_when_props_empty(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records(
            [{"src": "a", "tgt": "b", "props": {}}]
        )
        storage._driver = driver

        result = await storage.get_edges_batch([{"src": "a", "tgt": "b"}])

        edge = result[("a", "b")]
        assert edge["weight"] == 1.0
        assert edge["source_id"] is None
        assert edge["description"] is None
        assert edge["keywords"] is None

    async def test_empty_input(self):
        storage = _make_storage()
        result = await storage.get_edges_batch([])
        assert result == {}


class TestEdgeDegreesBatch:
    async def test_sums_node_degrees(self):
        storage = _make_storage()
        # Mock node_degrees_batch (already patched, so we mock it directly)
        storage.node_degrees_batch = AsyncMock(
            return_value={"alice": 3, "bob": 1, "carol": 5}
        )

        result = await storage.edge_degrees_batch([("alice", "bob"), ("bob", "carol")])

        assert result[("alice", "bob")] == 4
        assert result[("bob", "carol")] == 6

    async def test_empty_input(self):
        storage = _make_storage()
        result = await storage.edge_degrees_batch([])
        assert result == {}


class TestGetNodesEdgesBatch:
    async def test_returns_edge_tuples(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records(
            [
                {
                    "eid": "alice",
                    "edges": [["alice", "bob"], ["alice", "carol"]],
                },
                {
                    "eid": "bob",
                    "edges": [["bob", "alice"]],
                },
            ]
        )
        storage._driver = driver

        result = await storage.get_nodes_edges_batch(["alice", "bob", "missing"])

        assert ("alice", "bob") in result["alice"]
        assert ("alice", "carol") in result["alice"]
        assert ("bob", "alice") in result["bob"]
        assert result["missing"] == []  # default for missing nodes

    async def test_filters_none_edges(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records(
            [
                {
                    "eid": "alice",
                    "edges": [[None, None], ["alice", "bob"]],
                },
            ]
        )
        storage._driver = driver

        result = await storage.get_nodes_edges_batch(["alice"])

        assert len(result["alice"]) == 1
        assert result["alice"][0] == ("alice", "bob")

    async def test_empty_input(self):
        storage = _make_storage()
        result = await storage.get_nodes_edges_batch([])
        assert result == {}


class TestGetNodesWithDegreesBatch:
    """Fused get_nodes_batch + node_degrees_batch."""

    async def test_returns_nodes_and_degrees(self):
        storage = _make_storage()
        node_data = {"entity_id": "alice", "type": "person", "labels": ["test_ws"]}
        driver, session = _mock_session_with_records(
            [{"eid": "alice", "n": node_data, "degree": 3}]
        )
        storage._driver = driver

        nodes, degrees = await storage.get_nodes_with_degrees_batch(
            ["alice", "missing"]
        )

        assert "alice" in nodes
        assert nodes["alice"]["type"] == "person"
        assert "test_ws" not in nodes["alice"]["labels"]
        assert degrees["alice"] == 3
        assert degrees["missing"] == 0  # default for missing
        assert "missing" not in nodes

    async def test_empty_input(self):
        storage = _make_storage()
        nodes, degrees = await storage.get_nodes_with_degrees_batch([])
        assert nodes == {}
        assert degrees == {}

    async def test_single_query(self):
        storage = _make_storage()
        driver, session = _mock_session_with_records([])
        storage._driver = driver

        await storage.get_nodes_with_degrees_batch(["a", "b"])

        session.run.assert_awaited_once()
        query = session.run.call_args[0][0]
        assert "UNWIND" in query
        assert "count(r) AS degree" in query


class TestGetEdgesWithDegreesBatch:
    """Fused get_edges_batch + edge_degrees_batch (2 queries in 1 session)."""

    async def test_returns_edges_and_degrees(self):
        storage = _make_storage()

        # Two sequential query results in the same session
        edge_records = [
            _FakeRecord(
                {
                    "src": "alice",
                    "tgt": "bob",
                    "props": {"weight": 0.8, "description": "knows"},
                }
            )
        ]
        degree_records = [
            _FakeRecord({"eid": "alice", "degree": 3}),
            _FakeRecord({"eid": "bob", "degree": 4}),
        ]

        # Build two separate result iterators
        edge_result = _AsyncRecordIterator(edge_records)
        edge_result.consume = AsyncMock()

        degree_result = _AsyncRecordIterator(degree_records)
        degree_result.consume = AsyncMock()

        session = AsyncMock()
        session.run = AsyncMock(side_effect=[edge_result, degree_result])

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        driver = MagicMock()
        driver.session = MagicMock(return_value=ctx)
        storage._driver = driver

        edges, degrees = await storage.get_edges_with_degrees_batch(
            [{"src": "alice", "tgt": "bob"}]
        )

        key = ("alice", "bob")
        assert key in edges
        assert edges[key]["weight"] == 0.8
        assert edges[key]["source_id"] is None  # default filled
        assert degrees[key] == 3 + 4  # sum of node degrees

    async def test_empty_input(self):
        storage = _make_storage()
        edges, degrees = await storage.get_edges_with_degrees_batch([])
        assert edges == {}
        assert degrees == {}


class TestOperatePatches:
    """Verify operate.py functions are monkey-patched after register()."""

    def test_get_node_data_patched(self):
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        # The patched version should NOT be the original
        assert "fused" in operate._get_node_data.__name__

    def test_find_edges_patched(self):
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        assert "fused" in operate._find_most_related_edges_from_entities.__name__


class TestFusedGetNodeData:
    """End-to-end tests for the patched _get_node_data function."""

    async def test_fused_path_with_results(self):
        """Full pipeline: VDB query -> fused batch -> node enrichment."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        # Mock VDB returning 2 entity results
        entities_vdb = AsyncMock()
        entities_vdb.cosine_better_than_threshold = 0.2
        entities_vdb.query = AsyncMock(
            return_value=[
                {"entity_name": "alice", "created_at": "2025-01-01"},
                {"entity_name": "bob", "created_at": "2025-01-02"},
            ]
        )

        # Mock graph storage with fused method
        graph = AsyncMock()
        graph.get_nodes_with_degrees_batch = AsyncMock(
            return_value=(
                {
                    "alice": {"description": "A person", "entity_type": "person"},
                    "bob": {"description": "Another person", "entity_type": "person"},
                },
                {"alice": 3, "bob": 1},
            )
        )
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={
                "alice": [],
                "bob": [],
            }
        )
        graph.get_edges_with_degrees_batch = AsyncMock(return_value=({}, {}))

        query_param = MagicMock()
        query_param.top_k = 10

        node_datas, relations = await operate._get_node_data(
            "test query", graph, entities_vdb, query_param
        )

        assert len(node_datas) == 2
        assert node_datas[0]["entity_name"] == "alice"
        assert node_datas[0]["rank"] == 3
        assert node_datas[1]["entity_name"] == "bob"
        assert node_datas[1]["rank"] == 1
        # Fused method was called, not individual batch methods
        graph.get_nodes_with_degrees_batch.assert_awaited_once_with(["alice", "bob"])
        graph.get_nodes_batch.assert_not_awaited()
        graph.node_degrees_batch.assert_not_awaited()

    async def test_empty_vdb_results(self):
        """Returns empty when VDB finds nothing."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        entities_vdb = AsyncMock()
        entities_vdb.cosine_better_than_threshold = 0.2
        entities_vdb.query = AsyncMock(return_value=[])

        graph = AsyncMock()
        query_param = MagicMock()
        query_param.top_k = 10

        node_datas, relations = await operate._get_node_data(
            "test", graph, entities_vdb, query_param
        )

        assert node_datas == []
        assert relations == []

    async def test_fallback_without_fused_method(self):
        """Falls back to gather() when graph has no fused method."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        entities_vdb = AsyncMock()
        entities_vdb.cosine_better_than_threshold = 0.2
        entities_vdb.query = AsyncMock(
            return_value=[
                {"entity_name": "alice"},
            ]
        )

        # Graph WITHOUT fused method
        graph = AsyncMock(
            spec=[
                "get_nodes_batch",
                "node_degrees_batch",
                "get_nodes_edges_batch",
                "get_edges_batch",
                "edge_degrees_batch",
            ]
        )
        graph.get_nodes_batch = AsyncMock(
            return_value={
                "alice": {"description": "A person"},
            }
        )
        graph.node_degrees_batch = AsyncMock(return_value={"alice": 2})
        graph.get_nodes_edges_batch = AsyncMock(return_value={"alice": []})
        graph.get_edges_batch = AsyncMock(return_value={})
        graph.edge_degrees_batch = AsyncMock(return_value={})

        query_param = MagicMock()
        query_param.top_k = 10

        node_datas, relations = await operate._get_node_data(
            "test", graph, entities_vdb, query_param
        )

        assert len(node_datas) == 1
        # Fallback: individual batch methods were called
        graph.get_nodes_batch.assert_awaited_once()
        graph.node_degrees_batch.assert_awaited_once()

    async def test_missing_nodes_filtered(self):
        """Nodes not found in graph are excluded from results."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        entities_vdb = AsyncMock()
        entities_vdb.cosine_better_than_threshold = 0.2
        entities_vdb.query = AsyncMock(
            return_value=[
                {"entity_name": "alice"},
                {"entity_name": "ghost"},  # not in graph
            ]
        )

        graph = AsyncMock()
        graph.get_nodes_with_degrees_batch = AsyncMock(
            return_value=(
                {"alice": {"description": "exists"}},  # ghost missing
                {"alice": 1, "ghost": 0},
            )
        )
        graph.get_nodes_edges_batch = AsyncMock(return_value={"alice": []})
        graph.get_edges_with_degrees_batch = AsyncMock(return_value=({}, {}))

        query_param = MagicMock()
        query_param.top_k = 10

        node_datas, _ = await operate._get_node_data(
            "test", graph, entities_vdb, query_param
        )

        assert len(node_datas) == 1
        assert node_datas[0]["entity_name"] == "alice"


class TestFusedFindEdges:
    """End-to-end tests for the patched _find_most_related_edges_from_entities."""

    async def test_fused_path_with_edges(self):
        """Full pipeline: get edges -> fused props+degrees -> sorted output."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        graph = AsyncMock()
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={
                "alice": [("alice", "bob"), ("alice", "carol")],
                "bob": [("alice", "bob")],  # duplicate, should be deduped
            }
        )
        graph.get_edges_with_degrees_batch = AsyncMock(
            return_value=(
                {
                    ("alice", "bob"): {
                        "weight": 0.9,
                        "description": "friends",
                        "source_id": "s1",
                        "keywords": "social",
                    },
                    ("alice", "carol"): {
                        "weight": 0.5,
                        "description": "colleagues",
                        "source_id": "s2",
                        "keywords": "work",
                    },
                },
                {
                    ("alice", "bob"): 7,
                    ("alice", "carol"): 4,
                },
            )
        )

        node_datas = [
            {"entity_name": "alice"},
            {"entity_name": "bob"},
        ]
        query_param = MagicMock()

        result = await operate._find_most_related_edges_from_entities(
            node_datas, query_param, graph
        )

        assert len(result) == 2
        # Sorted by (rank, weight) descending — alice-bob (rank=7) first
        assert result[0]["src_tgt"] == ("alice", "bob")
        assert result[0]["rank"] == 7
        assert result[0]["weight"] == 0.9
        assert result[1]["src_tgt"] == ("alice", "carol")
        # Fused method called, not individual
        graph.get_edges_with_degrees_batch.assert_awaited_once()
        graph.get_edges_batch.assert_not_awaited()
        graph.edge_degrees_batch.assert_not_awaited()

    async def test_edge_deduplication(self):
        """Edges (a,b) and (b,a) are treated as the same edge."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        graph = AsyncMock()
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={
                "alice": [("alice", "bob")],
                "bob": [("bob", "alice")],  # reverse of same edge
            }
        )
        graph.get_edges_with_degrees_batch = AsyncMock(
            return_value=(
                {
                    ("alice", "bob"): {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                },
                {("alice", "bob"): 5},
            )
        )

        node_datas = [{"entity_name": "alice"}, {"entity_name": "bob"}]
        result = await operate._find_most_related_edges_from_entities(
            node_datas, MagicMock(), graph
        )

        assert len(result) == 1  # deduped

    async def test_fallback_without_fused_method(self):
        """Falls back to gather() when graph has no fused method."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        graph = AsyncMock(
            spec=["get_nodes_edges_batch", "get_edges_batch", "edge_degrees_batch"]
        )
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={
                "alice": [("alice", "bob")],
            }
        )
        graph.get_edges_batch = AsyncMock(
            return_value={
                ("alice", "bob"): {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                },
            }
        )
        graph.edge_degrees_batch = AsyncMock(
            return_value={
                ("alice", "bob"): 3,
            }
        )

        node_datas = [{"entity_name": "alice"}]
        result = await operate._find_most_related_edges_from_entities(
            node_datas, MagicMock(), graph
        )

        assert len(result) == 1
        graph.get_edges_batch.assert_awaited_once()
        graph.edge_degrees_batch.assert_awaited_once()

    async def test_missing_weight_gets_default(self):
        """Edges without 'weight' property get default 1.0."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        graph = AsyncMock()
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={
                "a": [("a", "b")],
            }
        )
        graph.get_edges_with_degrees_batch = AsyncMock(
            return_value=(
                {
                    ("a", "b"): {
                        "description": "test",
                        "source_id": None,
                        "keywords": None,
                    }
                },  # no weight
                {("a", "b"): 2},
            )
        )

        result = await operate._find_most_related_edges_from_entities(
            [{"entity_name": "a"}], MagicMock(), graph
        )

        assert result[0]["weight"] == 1.0

    async def test_no_edges(self):
        """Nodes with no connections return empty."""
        _reset()
        twindb_lightrag_memgraph.register()
        import lightrag.operate as operate

        graph = AsyncMock()
        graph.get_nodes_edges_batch = AsyncMock(
            return_value={
                "alice": [],
            }
        )
        graph.get_edges_with_degrees_batch = AsyncMock(return_value=({}, {}))

        result = await operate._find_most_related_edges_from_entities(
            [{"entity_name": "alice"}], MagicMock(), graph
        )

        assert result == []


# ── _SafeDriverWrapper tests ─────────────────────────────────────────


def _init_helper():
    """Build a mock async context manager for get_data_init_lock."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_lock():
        yield

    return _fake_lock


class TestSafeDriverWrapper:
    """Cover _SafeDriverWrapper: session routing, close, __getattr__.

    We construct the wrapper directly (extracted from the patched initialize
    closure) to avoid fighting with closure-captured imports.
    """

    @staticmethod
    def _get_wrapper_class():
        """Extract _SafeDriverWrapper from the patched initialize closure."""
        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        # The class is stored in _patched_initialize's closure
        for cell in MemgraphStorage.initialize.__code__.co_consts:
            pass  # not accessible via co_consts
        # Simpler: look in the closure cells
        for cell in MemgraphStorage.initialize.__closure__ or []:
            val = cell.cell_contents
            if isinstance(val, type) and val.__name__ == "_SafeDriverWrapper":
                return val
        raise RuntimeError("Could not find _SafeDriverWrapper in closure")

    def _make_raw_driver(self, mock_session):
        mock_raw = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_raw.session = MagicMock(return_value=ctx)
        mock_raw.close = AsyncMock()
        return mock_raw

    async def test_bolt_session_issues_use_database(self):
        """bolt:// with database set -> issues USE DATABASE."""
        WrapperCls = self._get_wrapper_class()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_raw = self._make_raw_driver(mock_session)

        wrapper = WrapperCls(mock_raw, database="mydb", use_routing=False)

        async with wrapper.session() as session:
            pass

        mock_session.run.assert_any_call("USE DATABASE mydb")

    async def test_routing_session_passes_database_kwarg(self):
        """neo4j+s:// with database -> database= passed to session()."""
        WrapperCls = self._get_wrapper_class()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_raw = self._make_raw_driver(mock_session)

        wrapper = WrapperCls(mock_raw, database="mydb", use_routing=True)

        async with wrapper.session() as session:
            pass

        mock_raw.session.assert_called_with(database="mydb")
        # Should NOT have issued USE DATABASE
        use_db_calls = [
            c for c in mock_session.run.call_args_list if "USE DATABASE" in str(c)
        ]
        assert len(use_db_calls) == 0

    async def test_wrapper_close(self):
        """close() delegates to the real driver."""
        WrapperCls = self._get_wrapper_class()
        mock_raw = MagicMock()
        mock_raw.close = AsyncMock()

        wrapper = WrapperCls(mock_raw, database="", use_routing=False)
        await wrapper.close()
        mock_raw.close.assert_awaited_once()

    async def test_wrapper_getattr_delegates(self):
        """__getattr__ delegates unknown attributes to real driver."""
        WrapperCls = self._get_wrapper_class()
        mock_raw = MagicMock()
        mock_raw.some_custom_attr = "hello"

        wrapper = WrapperCls(mock_raw, database="", use_routing=False)
        assert wrapper.some_custom_attr == "hello"


# ── _patched_initialize tests ────────────────────────────────────────


class TestPatchedInitialize:
    """Cover _patched_initialize: index creation, error paths."""

    async def _init_with_mocks(
        self, session_run_side_effect=None, use_routing=False, database="memgraph"
    ):
        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage
        from lightrag.kg.shared_storage import initialize_share_data

        initialize_share_data(workers=1)

        mock_session = AsyncMock()
        if session_run_side_effect:
            mock_session.run = AsyncMock(side_effect=session_run_side_effect)
        else:
            mock_session.run = AsyncMock()

        mock_raw = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_raw.session = MagicMock(return_value=ctx)

        storage = MemgraphStorage.__new__(MemgraphStorage)
        storage.workspace = "test_ws"

        uri = "neo4j+s://remote:7687" if use_routing else "bolt://localhost:7687"

        with (
            patch(
                "twindb_lightrag_memgraph._pool._read_connection_config",
                return_value=(uri, database, {"auth": ("", "")}),
            ),
            patch(
                "twindb_lightrag_memgraph._pool._uses_routing_protocol",
                return_value=use_routing,
            ),
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_raw),
        ):
            await storage.initialize()

        return storage, mock_session

    async def test_creates_index(self):
        """Initialize creates an index on the workspace label."""
        storage, mock_session = await self._init_with_mocks()

        assert storage._DATABASE == "memgraph"
        calls = [str(c) for c in mock_session.run.call_args_list]
        assert any("CREATE INDEX" in c for c in calls)
        assert any("RETURN 1" in c for c in calls)

    async def test_index_already_exists_ignored(self):
        """'already exists' error during index creation is silently ignored."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is CREATE INDEX (USE DATABASE skipped for "memgraph")
                raise Exception("Index already exists")
            return AsyncMock()

        storage, _ = await self._init_with_mocks(session_run_side_effect=side_effect)
        assert storage._driver is not None

    async def test_connection_failure_raises(self):
        """Connection failure propagates the exception."""
        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage
        from lightrag.kg.shared_storage import initialize_share_data

        initialize_share_data(workers=1)

        mock_raw = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_raw.session = MagicMock(return_value=ctx)

        storage = MemgraphStorage.__new__(MemgraphStorage)
        storage.workspace = "test_ws"

        with (
            patch(
                "twindb_lightrag_memgraph._pool._read_connection_config",
                return_value=("bolt://localhost:7687", "memgraph", {"auth": ("", "")}),
            ),
            patch(
                "twindb_lightrag_memgraph._pool._uses_routing_protocol",
                return_value=False,
            ),
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_raw),
        ):
            with pytest.raises(ConnectionError):
                await storage.initialize()

    async def test_empty_database_defaults_to_memgraph(self):
        """Empty database env var defaults to 'memgraph'."""
        storage, _ = await self._init_with_mocks(database="")
        assert storage._DATABASE == "memgraph"


# ── RuntimeError guard tests ─────────────────────────────────────────


class TestDriverNoneGuards:
    """Cover the 'driver is None' RuntimeError branches."""

    def _make_uninit_storage(self):
        _reset()
        twindb_lightrag_memgraph.register()
        from lightrag.kg.memgraph_impl import MemgraphStorage

        storage = MemgraphStorage.__new__(MemgraphStorage)
        storage.workspace = "test"
        storage._DATABASE = "memgraph"
        storage._driver = None
        return storage

    async def test_get_nodes_batch_driver_none(self):
        storage = self._make_uninit_storage()
        with pytest.raises(RuntimeError, match="not initialized"):
            await storage.get_nodes_batch(["a"])

    async def test_node_degrees_batch_driver_none(self):
        storage = self._make_uninit_storage()
        with pytest.raises(RuntimeError, match="not initialized"):
            await storage.node_degrees_batch(["a"])

    async def test_get_edges_batch_driver_none(self):
        storage = self._make_uninit_storage()
        with pytest.raises(RuntimeError, match="not initialized"):
            await storage.get_edges_batch([{"src": "a", "tgt": "b"}])

    async def test_get_nodes_edges_batch_driver_none(self):
        storage = self._make_uninit_storage()
        with pytest.raises(RuntimeError, match="not initialized"):
            await storage.get_nodes_edges_batch(["a"])

    async def test_get_nodes_with_degrees_batch_driver_none(self):
        storage = self._make_uninit_storage()
        with pytest.raises(RuntimeError, match="not initialized"):
            await storage.get_nodes_with_degrees_batch(["a"])

    async def test_get_edges_with_degrees_batch_driver_none(self):
        storage = self._make_uninit_storage()
        with pytest.raises(RuntimeError, match="not initialized"):
            await storage.get_edges_with_degrees_batch([{"src": "a", "tgt": "b"}])
