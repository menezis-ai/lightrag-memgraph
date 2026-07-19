"""Scientific contracts for the bounded GraphRAG neighbourhood expansion."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import twindb_lightrag_memgraph.patches.registry as registry


class _Records:
    def __init__(self, rows):
        self._rows = iter(rows)
        self.consume = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._rows)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _graph_with_rows(rows):
    records = _Records(rows)
    session = AsyncMock()
    session.run = AsyncMock(return_value=records)
    context = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session = MagicMock(return_value=context)

    graph = SimpleNamespace(
        _driver=driver,
        _DATABASE="memgraph",
        _get_workspace_label=lambda: "test_ws",
    )
    return graph, session


class TestGraphTraversalConfiguration:
    def test_defaults_are_small_and_bounded(self, monkeypatch):
        monkeypatch.delenv(registry._GRAPH_MAX_HOPS_ENV, raising=False)
        monkeypatch.delenv(registry._GRAPH_PATHS_PER_SEED_ENV, raising=False)
        monkeypatch.delenv(registry._GRAPH_HOP_PENALTY_ENV, raising=False)

        assert registry._graph_traversal_config() == (2, 20, 0.15)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            (registry._GRAPH_MAX_HOPS_ENV, "0"),
            (registry._GRAPH_MAX_HOPS_ENV, "4"),
            (registry._GRAPH_MAX_HOPS_ENV, "2.5"),
            (registry._GRAPH_PATHS_PER_SEED_ENV, "0"),
            (registry._GRAPH_PATHS_PER_SEED_ENV, "101"),
            (registry._GRAPH_HOP_PENALTY_ENV, "nan"),
            (registry._GRAPH_HOP_PENALTY_ENV, "1.1"),
        ],
    )
    def test_invalid_environment_values_fail_closed(self, monkeypatch, name, value):
        monkeypatch.setenv(name, value)

        with pytest.raises(ValueError, match=name):
            registry._graph_traversal_config()

    def test_equal_mean_weight_is_strictly_hop_penalized(self):
        one_hop = registry._graph_path_score([0.9], 0.15)
        two_hops = registry._graph_path_score([0.9, 0.9], 0.15)
        three_hops = registry._graph_path_score([0.9, 0.9, 0.9], 0.15)

        assert one_hop == pytest.approx(0.9)
        assert two_hops == pytest.approx(0.75)
        assert three_hops == pytest.approx(0.6)

    def test_mean_does_not_reward_duplicate_edge_weight(self):
        assert registry._graph_path_score([0.8], 0.0) == pytest.approx(
            registry._graph_path_score([0.8, 0.8, 0.8], 0.0)
        )


class TestGraphPathBatchCypher:
    async def test_bounded_query_flattens_real_path_edges_and_passes_cap(
        self, monkeypatch
    ):
        graph, session = _graph_with_rows(
            [
                {
                    "eid": "seed",
                    "traversals": [
                        {
                            "edge": ["seed", "bridge"],
                            "discovery_hop": 1,
                            "path_hops": 2,
                            "path_score": 0.65,
                            "path_key": "|seed|bridge|target",
                        },
                        {
                            "edge": ["bridge", "target"],
                            "discovery_hop": 2,
                            "path_hops": 2,
                            "path_score": 0.65,
                            "path_key": "|seed|bridge|target",
                        },
                    ],
                }
            ]
        )
        monkeypatch.setattr(
            registry, "_twin_member_chunks", AsyncMock(return_value=None)
        )

        result = await registry._patched_get_nodes_edges_paths_batch(
            graph,
            ["seed", "missing"],
            max_hops=2,
            paths_per_seed=7,
            hop_penalty=0.2,
        )

        query = session.run.call_args.args[0]
        params = session.run.call_args.kwargs
        assert (
            "MATCH path=(seed)-[__rels:DIRECTED *BFS 1..2 "
            "(__rel, __next | 'test_ws' IN labels(__next))]"
            "-(reached:`test_ws`)" in query
        )
        assert "all(__node IN nodes(path) WHERE 'test_ws' IN labels(__node))" in query
        assert "weight_sum / toFloat(hops)" in query
        assert "$hop_penalty * toFloat(hops - 1)" in query
        assert "[..$paths_per_seed] AS selected_paths" in query
        assert "path_nodes[edge_index + 1].entity_id" in query
        assert "ORDER BY eid ASC, path_score DESC, hops ASC" in query
        assert params == {
            "ids": ["seed", "missing"],
            "paths_per_seed": 7,
            "hop_penalty": 0.2,
        }
        assert [item["edge"] for item in result["seed"]] == [
            ("seed", "bridge"),
            ("bridge", "target"),
        ]
        assert result["seed"][1]["discovery_hop"] == 2
        assert result["missing"] == []

    async def test_folder_scope_applies_to_every_relationship_in_path(
        self, monkeypatch
    ):
        graph, session = _graph_with_rows([])
        monkeypatch.setattr(
            registry,
            "_twin_member_chunks",
            AsyncMock(return_value=["chunk-a", "chunk-b"]),
        )

        await registry._patched_get_nodes_edges_paths_batch(graph, ["seed"])

        query = session.run.call_args.args[0]
        params = session.run.call_args.kwargs
        assert "[__rels:DIRECTED *BFS 1..2" in query
        assert "__next | 'test_ws' IN labels(__next) AND any(" in query
        assert "all(__rel IN relationships(path)" in query
        assert "split(coalesce(__rel.source_id, ''), $sep)" in query
        assert "WHERE _cid IN $mchunks" in query
        assert params["mchunks"] == ["chunk-a", "chunk-b"]
        assert params["sep"] == registry._GRAPH_FIELD_SEP

    async def test_one_hop_configuration_keeps_edge_shape_compatible(self, monkeypatch):
        graph, session = _graph_with_rows(
            [
                {
                    "eid": "seed",
                    "traversals": [
                        {
                            "edge": ["seed", "neighbour"],
                            "discovery_hop": 1,
                            "path_hops": 1,
                            "path_score": 0.9,
                            "path_key": "|seed|neighbour",
                        }
                    ],
                }
            ]
        )
        monkeypatch.setattr(
            registry, "_twin_member_chunks", AsyncMock(return_value=None)
        )

        result = await registry._patched_get_nodes_edges_paths_batch(
            graph, ["seed"], max_hops=1
        )

        assert "[__rels:DIRECTED *BFS 1..1" in session.run.call_args.args[0]
        assert result["seed"][0]["edge"] == ("seed", "neighbour")
        assert result["seed"][0]["discovery_hop"] == 1
        assert result["seed"][0]["path_hops"] == 1


class TestFusedMultiHopRetrieval:
    async def test_best_discovery_path_controls_metadata_and_ranking(self):
        graph = SimpleNamespace(
            get_nodes_edges_paths_batch=AsyncMock(
                return_value={
                    "seed-a": [
                        {
                            "edge": ("a", "b"),
                            "seed": "seed-a",
                            "discovery_hop": 1,
                            "path_hops": 1,
                            "path_score": 0.4,
                            "path_key": "|seed-a|a|b",
                        },
                        {
                            "edge": ("c", "d"),
                            "seed": "seed-a",
                            "discovery_hop": 2,
                            "path_hops": 2,
                            "path_score": 0.8,
                            "path_key": "|seed-a|c|d",
                        },
                    ],
                    "seed-b": [
                        {
                            "edge": ("b", "a"),
                            "seed": "seed-b",
                            "discovery_hop": 2,
                            "path_hops": 2,
                            "path_score": 0.7,
                            "path_key": "|seed-b|b|a",
                        }
                    ],
                }
            ),
            get_edges_with_degrees_batch=AsyncMock(
                return_value=(
                    {
                        ("a", "b"): {"weight": 0.9},
                        ("c", "d"): {"weight": 0.5},
                    },
                    {("a", "b"): 100, ("c", "d"): 1},
                )
            ),
        )

        result = await registry._fused_find_edges(
            [{"entity_name": "seed-a"}, {"entity_name": "seed-b"}],
            MagicMock(),
            graph,
        )

        # Explicit path score is primary: a high global degree cannot erase
        # the bounded semantic traversal evidence.
        assert [row["src_tgt"] for row in result] == [("c", "d"), ("a", "b")]
        ab = result[1]
        assert ab["graph_seed"] == "seed-b"
        assert ab["graph_path_score"] == pytest.approx(0.7)
        assert ab["graph_hops"] == 2
        assert ab["graph_path_hops"] == 2

    async def test_backend_without_dedicated_method_keeps_legacy_one_hop(self):
        graph = SimpleNamespace(
            get_nodes_edges_batch=AsyncMock(
                return_value={"seed": [("seed", "neighbour")]}
            ),
            get_edges_batch=AsyncMock(
                return_value={
                    ("neighbour", "seed"): {"weight": 0.8},
                }
            ),
            edge_degrees_batch=AsyncMock(return_value={("neighbour", "seed"): 4}),
        )

        result = await registry._fused_find_edges(
            [{"entity_name": "seed"}], MagicMock(), graph
        )

        graph.get_nodes_edges_batch.assert_awaited_once_with(["seed"])
        assert result[0]["src_tgt"] == ("neighbour", "seed")
        assert "graph_path_score" not in result[0]


@pytest.mark.integration
async def test_live_memgraph_executes_bounded_two_hop_query(monkeypatch):
    """Execute the generated BFS query against the supported Memgraph image."""
    import twindb_lightrag_memgraph
    from lightrag.kg.memgraph_impl import MemgraphStorage
    from lightrag.kg.shared_storage import initialize_share_data

    monkeypatch.delenv("MEMGRAPH_WORKSPACE", raising=False)
    initialize_share_data(workers=1)
    twindb_lightrag_memgraph.register()

    suffix = uuid4().hex
    workspace = f"graph_multihop_{suffix}"
    foreign_workspace = f"graph_multihop_foreign_{suffix}"
    graph = MemgraphStorage(
        namespace="test_graph_multihop_live",
        global_config={"workspace": workspace},
        embedding_func=None,
        workspace=workspace,
    )
    await graph.initialize()
    ws = graph._get_workspace_label()

    try:
        async with graph._driver.session(database=graph._DATABASE) as session:
            records = await session.run(
                f"CREATE (a:`{ws}` {{entity_id: 'A'}}), "
                f"(b:`{ws}` {{entity_id: 'B'}}), "
                f"(c:`{ws}` {{entity_id: 'C'}}), "
                f"(x:`{foreign_workspace}` {{entity_id: 'X'}}), "
                "(a)-[:DIRECTED {weight: 0.9, source_id: 'chunk-ab'}]->(b), "
                "(b)-[:DIRECTED {weight: 0.7, source_id: 'chunk-bc'}]->(c), "
                "(b)-[:DIRECTED {weight: 1.0, source_id: 'chunk-bx'}]->(x)"
            )
            await records.consume()

        result = await graph.get_nodes_edges_paths_batch(
            ["A", "MISSING"],
            max_hops=2,
            paths_per_seed=20,
            hop_penalty=0.1,
        )

        traversals = result["A"]
        two_hop_edges = {
            row["edge"]
            for row in traversals
            if row["path_hops"] == 2 and row["path_score"] == pytest.approx(0.7)
        }
        assert ("A", "B") in two_hop_edges
        assert ("B", "C") in two_hop_edges
        assert all("X" not in row["edge"] for row in traversals)
        assert result["MISSING"] == []
    finally:
        async with graph._driver.session(database=graph._DATABASE) as session:
            records = await session.run(
                f"MATCH (n) WHERE '{ws}' IN labels(n) "
                f"OR '{foreign_workspace}' IN labels(n) DETACH DELETE n"
            )
            await records.consume()
        await graph.finalize()
