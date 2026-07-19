"""Tests for ontology storage layer (Memgraph persistence)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph.intelligence.ontology.storage import (
    OntologyEdge,
    OntologyNode,
    OntologyStorage,
    _query_term_candidates,
)


class AsyncRecordResult:
    def __init__(self, records):
        self._records = records
        self._iterator = iter(())
        self.consume = AsyncMock()

    def __aiter__(self):
        self._iterator = iter(self._records)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.fixture
def mock_session():
    session = AsyncMock()
    result = AsyncMock()
    result.consume = AsyncMock()
    result.single = AsyncMock(return_value={"cnt": 5})
    session.run = AsyncMock(return_value=result)
    return session


@pytest.fixture
def mock_driver(mock_session):
    driver = AsyncMock()
    driver.session = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return driver


@pytest.fixture
def storage(mock_driver):
    s = OntologyStorage("test_ws")
    s._driver = mock_driver
    s._database = "memgraph"
    return s


class TestOntologyStorage:
    def test_label(self, storage):
        assert storage._label() == "Onto_test_ws"

    @patch("twindb_lightrag_memgraph.intelligence.ontology.storage._pool")
    async def test_initialize_creates_indexes(self, mock_pool):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.consume = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_session),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        mock_pool.get_driver = AsyncMock(return_value=(mock_driver, "memgraph"))

        storage = OntologyStorage("init_ws")
        await storage.initialize()

        # Should have run CREATE INDEX queries + seed data
        assert mock_session.run.call_count > 2

    async def test_upsert_nodes(self, storage, mock_session):
        nodes = [
            OntologyNode(
                name="ORA-04030",
                node_type="Term",
                confidence=0.95,
                source_doc="doc_0",
            ),
            OntologyNode(
                name="PGA",
                node_type="Term",
                confidence=0.90,
                source_doc="doc_0",
            ),
        ]

        await storage.upsert_nodes(nodes)

        mock_session.run.assert_called()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "UNWIND" in query
        assert "MERGE" in query
        assert "Onto_test_ws" in query

    async def test_upsert_edges_grouped_by_type(self, storage, mock_session):
        edges = [
            OntologyEdge(
                source_name="ORA-04030",
                source_type="Term",
                target_name="PGA",
                target_type="Term",
                relation_type="RELATED_TO",
                confidence=0.9,
            ),
            OntologyEdge(
                source_name="PGA",
                source_type="Term",
                target_name="SGA",
                target_type="Term",
                relation_type="CO_OCCURS",
                confidence=0.85,
            ),
            OntologyEdge(
                source_name="ORA-04030",
                source_type="Term",
                target_name="SGA",
                target_type="Term",
                relation_type="RELATED_TO",
                confidence=0.8,
            ),
        ]

        await storage.upsert_edges(edges)

        # Should have 2 calls: one for RELATED_TO, one for CO_OCCURS
        assert mock_session.run.call_count == 2

        calls = mock_session.run.call_args_list
        queries = [c[0][0] for c in calls]

        rel_types_in_queries = set()
        for q in queries:
            if "RELATED_TO" in q:
                rel_types_in_queries.add("RELATED_TO")
            if "CO_OCCURS" in q:
                rel_types_in_queries.add("CO_OCCURS")

        assert rel_types_in_queries == {"RELATED_TO", "CO_OCCURS"}

    async def test_query_expansion_uses_ngram_seeds_and_weighted_paths(
        self, storage, mock_session
    ):
        result = AsyncRecordResult(
            [
                {"name": "PGA memory"},
                {"name": "heap allocation"},
            ]
        )
        mock_session.run = AsyncMock(return_value=result)

        names = await storage.query_expansion(
            "Pourquoi ORA-04030 survient-il après allocation PGA ?",
            max_hops=2,
        )

        assert names == ["PGA memory", "heap allocation"]
        query = mock_session.run.call_args.args[0]
        params = mock_session.run.call_args.kwargs
        assert "toLower(start.name) IN $candidate_terms" in query
        assert "SYNONYM|RELATED_TO|CO_OCCURS*1..2" in query
        assert "relation.confidence" in query
        assert "max(confidence_product / toFloat(hops)) AS path_score" in query
        assert "collect({score: path_score, hops: hops})[0] AS best_path" in query
        assert "ORDER BY best_path.score DESC" in query
        assert "best_path.hops ASC" in query
        assert "toLower(name) ASC" in query
        assert "name ASC" in query
        assert "ora-04030" in params["candidate_terms"]
        assert "allocation pga" in params["candidate_terms"]
        assert "pourquoi" not in params["candidate_terms"]
        assert "term" not in params
        result.consume.assert_awaited_once()

    def test_query_candidates_are_bounded_and_deterministic(self):
        question = "Pourquoi ORA-04030 survient-il après allocation PGA ?"

        first = _query_term_candidates(question)
        second = _query_term_candidates(question)

        assert first == second
        assert len(first) <= 64
        assert first[:2] == ["ora-04030", "pga"]
        assert "allocation pga" in first
        assert "pourquoi" not in first

    async def test_query_expansion_skips_generic_only_question(
        self, storage, mock_session
    ):
        names = await storage.query_expansion("Pourquoi une erreur ?")

        assert names == []
        mock_session.run.assert_not_called()

    @pytest.mark.parametrize("max_hops", [0, 4, True, 1.5, "2"])
    async def test_query_expansion_rejects_unbounded_hops(
        self, storage, mock_session, max_hops
    ):
        with pytest.raises(ValueError, match="between 1 and 3"):
            await storage.query_expansion("ORA-04030", max_hops=max_hops)

        mock_session.run.assert_not_called()

    async def test_has_data_true(self, storage, mock_session):
        result = await storage.has_data()
        assert result is True

    async def test_has_data_false(self, storage, mock_session):
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"cnt": 0})
        mock_result.consume = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)

        result = await storage.has_data()
        assert result is False

    async def test_seed_normative_data(self, storage, mock_session):
        from twindb_lightrag_memgraph.intelligence.ontology.schema import (
            SEED_ENVIRONMENTS,
            SEED_METHODOLOGIES,
            SEED_SLAS,
        )

        await storage._seed_normative(mock_session, "Onto_test_ws")

        expected_calls = (
            len(SEED_METHODOLOGIES) + len(SEED_SLAS) + len(SEED_ENVIRONMENTS)
        )
        assert mock_session.run.call_count == expected_calls

    async def test_drop(self, storage, mock_session):
        result = await storage.drop()
        assert result["status"] == "success"
        assert "Onto_test_ws" in result["message"]

        mock_session.run.assert_called()
        query = mock_session.run.call_args[0][0]
        assert "DETACH DELETE" in query

    async def test_upsert_empty_nodes(self, storage, mock_session):
        await storage.upsert_nodes([])
        mock_session.run.assert_not_called()

    async def test_upsert_empty_edges(self, storage, mock_session):
        await storage.upsert_edges([])
        mock_session.run.assert_not_called()


class TestOntologyStorageCypherInjection:
    """Defense-in-depth: storage must reject Cypher-injection payloads
    that slip past the parser allow-lists upstream.

    Symmetric to the fix in ``_buffered_graph.py`` (commit 1fa47dd) which
    closed the same vector on entity_type for the chunk graph.
    """

    def test_workspace_with_backtick_is_rejected(self):
        with pytest.raises(ValueError):
            OntologyStorage("base`) DETACH DELETE n //")

    def test_workspace_with_space_is_rejected(self):
        with pytest.raises(ValueError):
            OntologyStorage("ws with space")

    async def test_malicious_relation_type_is_dropped(self, storage, mock_session):
        edges = [
            OntologyEdge(
                source_name="alpha",
                source_type="Term",
                target_name="beta",
                target_type="Term",
                relation_type="REL`]->(x) WITH x MATCH (n) DETACH DELETE n //",
                confidence=0.9,
            ),
        ]

        await storage.upsert_edges(edges)

        mock_session.run.assert_not_called()

    async def test_safe_relation_type_still_runs(self, storage, mock_session):
        edges = [
            OntologyEdge(
                source_name="alpha",
                source_type="Term",
                target_name="beta",
                target_type="Term",
                relation_type="RELATED_TO",
                confidence=0.9,
            ),
        ]

        await storage.upsert_edges(edges)

        assert mock_session.run.call_count == 1
        query = mock_session.run.call_args[0][0]
        assert "`RELATED_TO`" in query
        assert "DETACH DELETE" not in query

    async def test_mixed_edges_only_safe_one_runs(self, storage, mock_session):
        edges = [
            OntologyEdge(
                source_name="alpha",
                source_type="Term",
                target_name="beta",
                target_type="Term",
                relation_type="RELATED_TO",
                confidence=0.9,
            ),
            OntologyEdge(
                source_name="alpha",
                source_type="Term",
                target_name="beta",
                target_type="Term",
                relation_type="REL`]->(x) WITH x MATCH (n) DETACH DELETE n //",
                confidence=0.9,
            ),
        ]

        await storage.upsert_edges(edges)

        assert mock_session.run.call_count == 1
        query = mock_session.run.call_args[0][0]
        assert "`RELATED_TO`" in query
