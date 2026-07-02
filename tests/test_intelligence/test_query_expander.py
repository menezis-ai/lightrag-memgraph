"""Tests for F03: Query Expander."""

import pytest

from twindb_lightrag_memgraph.intelligence.features.query_expander import QueryExpander


class TestQueryExpander:
    """F03: Query expansion tests."""

    @pytest.fixture
    def expander(self, config):
        return QueryExpander(config)

    def test_expand_with_match(self, expander):
        result = expander.expand("Probleme ORA-04030 sur le serveur")
        assert result.expanded_query != result.original_query
        assert len(result.added_terms) > 0
        assert any("memory" in t.lower() or "PGA" in t for t in result.added_terms)

    def test_expand_no_match(self, expander):
        result = expander.expand("question sans terme technique reconnu xyz123")
        assert result.expanded_query == result.original_query
        assert len(result.added_terms) == 0

    def test_expand_with_domain_filter(self, expander):
        result = expander.expand("VLAN configuration", domain_hint="network")
        assert len(result.added_terms) > 0
        # All matched entries should be network domain
        for entry in result.matched_entries:
            assert entry["domaine"] == "network"

    def test_expand_domain_filter_excludes(self, expander):
        """Filtering by wrong domain should not match."""
        result = expander.expand("ORA-04030", domain_hint="network")
        assert result.expanded_query == result.original_query
        assert len(result.added_terms) == 0

    def test_expand_multi_term(self, expander):
        result = expander.expand("ORA-04030 et RMAN backup")
        assert len(result.matched_entries) >= 2

    def test_expand_max_limit(self, expander):
        """Should respect max_total_synonyms limit."""
        result = expander.expand("ORA-04030 et RMAN et Data Guard")
        assert len(result.added_terms) <= expander.config.max_total_synonyms

    def test_expand_empty_query(self, expander):
        result = expander.expand("")
        assert result.expanded_query == ""
        assert len(result.added_terms) == 0

    def test_expand_case_insensitive(self, expander):
        result = expander.expand("ora-04030 probleme memoire")
        assert len(result.added_terms) > 0
