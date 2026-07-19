"""Tests for QueryExpander v2 (graph-based ontology expansion)."""

from unittest.mock import AsyncMock, patch

import pytest

from twindb_lightrag_memgraph.intelligence.config import TwinRAGConfig
from twindb_lightrag_memgraph.intelligence.features.query_expander import QueryExpander


@pytest.fixture
def config():
    return TwinRAGConfig(
        llm_api_key="test-key",
        llm_api_base="http://mock:8080",
    )


@pytest.fixture
def expander(config):
    return QueryExpander(config)


class TestExpandV2:
    async def test_expand_v2_with_graph(self, expander):
        """When ontology has data and returns terms, use graph expansion."""
        mock_storage = AsyncMock()
        mock_storage.has_data = AsyncMock(return_value=True)
        mock_storage.query_expansion = AsyncMock(
            return_value=["PGA memory", "heap allocation", "SGA"]
        )

        mock_driver = AsyncMock()
        mock_get_driver = AsyncMock(return_value=(mock_driver, "memgraph"))

        with patch(
            "twindb_lightrag_memgraph.intelligence.ontology.storage.OntologyStorage",
            return_value=mock_storage,
        ):
            with patch(
                "twindb_lightrag_memgraph._pool.get_driver",
                mock_get_driver,
            ):
                result = await expander.expand_v2(
                    "Pourquoi ORA-04030 survient-il après allocation PGA ?",
                    workspace="oracle_ws",
                )

        assert len(result.added_terms) == 3
        assert "PGA memory" in result.added_terms
        assert "ORA-04030" in result.expanded_query
        mock_storage.query_expansion.assert_awaited_once_with(
            "Pourquoi ORA-04030 survient-il après allocation PGA ?",
            max_hops=2,
        )

    async def test_expand_v2_applies_domain_filter_in_rank_order(self, expander):
        mock_storage = AsyncMock()
        mock_storage.has_data = AsyncMock(return_value=True)
        mock_storage.query_expansion = AsyncMock(
            return_value=["PGA memory", "heap allocation", "SGA", "sga"]
        )
        mock_storage.get_domain_terms = AsyncMock(return_value=["sga", "PGA MEMORY"])

        with patch(
            "twindb_lightrag_memgraph.intelligence.ontology.storage.OntologyStorage",
            return_value=mock_storage,
        ):
            with patch(
                "twindb_lightrag_memgraph._pool.get_driver",
                AsyncMock(return_value=(AsyncMock(), "memgraph")),
            ):
                result = await expander.expand_v2(
                    "ORA-04030 allocation PGA",
                    workspace="oracle_ws",
                    domain_hint="oracle",
                )

        assert result.added_terms == ["PGA memory", "SGA"]
        mock_storage.get_domain_terms.assert_awaited_once_with("oracle")

    async def test_expand_v2_domain_miss_falls_back_to_filtered_thesaurus(
        self, expander
    ):
        mock_storage = AsyncMock()
        mock_storage.has_data = AsyncMock(return_value=True)
        mock_storage.query_expansion = AsyncMock(return_value=["PGA memory"])
        mock_storage.get_domain_terms = AsyncMock(return_value=["VLAN"])

        with patch(
            "twindb_lightrag_memgraph.intelligence.ontology.storage.OntologyStorage",
            return_value=mock_storage,
        ):
            with patch(
                "twindb_lightrag_memgraph._pool.get_driver",
                AsyncMock(return_value=(AsyncMock(), "memgraph")),
            ):
                result = await expander.expand_v2(
                    "ORA-04030",
                    workspace="oracle_ws",
                    domain_hint="network",
                )

        assert result.expanded_query == "ORA-04030"
        assert result.added_terms == []

    async def test_expand_v2_fallback_to_v1(self, expander):
        """When graph expansion fails, fall back to v1 (JSON thesaurus)."""
        # Make _pool.get_driver raise to trigger the except branch
        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            AsyncMock(side_effect=Exception("connection failed")),
        ):
            result = await expander.expand_v2(
                "ORA-04030 probleme memoire", workspace="oracle_ws"
            )

        # Should fall back to v1 and find ORA-04030 in thesaurus
        assert result.original_query == "ORA-04030 probleme memoire"

    async def test_expand_v2_no_ontology_data(self, expander):
        """When ontology has no data, fall back to v1."""
        mock_storage = AsyncMock()
        mock_storage.has_data = AsyncMock(return_value=False)

        mock_driver = AsyncMock()
        mock_get_driver = AsyncMock(return_value=(mock_driver, "memgraph"))

        with patch(
            "twindb_lightrag_memgraph.intelligence.ontology.storage.OntologyStorage",
            return_value=mock_storage,
        ):
            with patch(
                "twindb_lightrag_memgraph._pool.get_driver",
                mock_get_driver,
            ):
                result = await expander.expand_v2("ORA-04030", workspace="oracle_ws")

        # Falls back to v1 thesaurus
        assert result.original_query == "ORA-04030"

    async def test_expand_v2_empty_graph_result(self, expander):
        """When graph returns no terms, fall back to v1."""
        mock_storage = AsyncMock()
        mock_storage.has_data = AsyncMock(return_value=True)
        mock_storage.query_expansion = AsyncMock(return_value=[])

        mock_driver = AsyncMock()
        mock_get_driver = AsyncMock(return_value=(mock_driver, "memgraph"))

        with patch(
            "twindb_lightrag_memgraph.intelligence.ontology.storage.OntologyStorage",
            return_value=mock_storage,
        ):
            with patch(
                "twindb_lightrag_memgraph._pool.get_driver",
                mock_get_driver,
            ):
                result = await expander.expand_v2(
                    "ORA-04030 probleme memoire", workspace="oracle_ws"
                )

        # v1 fallback should find thesaurus match
        assert result.original_query == "ORA-04030 probleme memoire"
