"""
twin_rag_intelligence/features/query_expander.py
==================================================
F03: Query Expansion with IT/Ops thesaurus.

Impact measured (CNCAC production):
  +30% recall on domain terminology
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..config import TwinRAGConfig
from ..thesaurus.loader import ThesaurusLoader

logger = logging.getLogger("twin_rag_intelligence.expander")


@dataclass
class ExpansionResult:
    original_query: str
    expanded_query: str
    added_terms: list[str] = field(default_factory=list)
    matched_entries: list[dict] = field(default_factory=list)


class QueryExpander:
    """
    Query expansion with IT/Ops thesaurus.

    Strategy (identical to ChatDocAI CNCAC):
    1. Detect technical terms in the query
    2. Add up to max_synonyms_per_term (3) synonyms per matched term
    3. Maximum max_total_synonyms (5) total synonyms per query
    4. Optional filtering by domain (oracle, network, linux...)
    """

    def __init__(self, config: TwinRAGConfig) -> None:
        self.config = config
        self._loader = ThesaurusLoader()

    def expand(
        self,
        query: str,
        domain_hint: Optional[str] = None,
    ) -> ExpansionResult:
        """
        Expand the query with technical synonyms.

        Args:
            query: Query to expand.
            domain_hint: Optional domain to filter synonyms.

        Returns:
            ExpansionResult with expanded query and added terms.
        """
        thesaurus = self._loader.load()
        glossaire = thesaurus.get("glossaire", [])
        query_lower = query.lower()

        matched = []
        for entry in glossaire:
            if domain_hint and entry.get("domaine") != domain_hint:
                continue
            term_lower = entry["terme"].lower()
            if term_lower in query_lower:
                matched.append(entry)

        if not matched:
            return ExpansionResult(original_query=query, expanded_query=query)

        all_synonyms: list[str] = []
        for entry in matched:
            synonyms = entry.get("synonymes", [])[: self.config.max_synonyms_per_term]
            all_synonyms.extend(synonyms)

        # Deduplicate and limit
        unique = list(dict.fromkeys(all_synonyms))[: self.config.max_total_synonyms]

        if unique:
            expanded = f"{query} {' '.join(unique)}"
            return ExpansionResult(
                original_query=query,
                expanded_query=expanded,
                added_terms=unique,
                matched_entries=matched,
            )

        return ExpansionResult(original_query=query, expanded_query=query)

    async def expand_v2(
        self,
        query: str,
        workspace: str,
        domain_hint: Optional[str] = None,
    ) -> ExpansionResult:
        """Graph-based expansion via Memgraph ontology.

        Checks if ontology data exists for the workspace via a lightweight
        count query. Falls back to v1 (JSON thesaurus) on failure or if
        no ontology data is present.

        Args:
            query: Query to expand.
            workspace: Workspace name for ontology lookup.
            domain_hint: Optional domain to filter.

        Returns:
            ExpansionResult with expanded query and added terms.
        """
        try:
            from ..ontology.storage import OntologyStorage
            from ... import _pool

            storage = OntologyStorage(workspace)
            storage._driver, storage._database = await _pool.get_driver()

            if not await storage.has_data():
                return self.expand(query, domain_hint)

            graph_terms = await storage.query_expansion(query, max_hops=2)

            if domain_hint:
                domain_terms = await storage.get_domain_terms(domain_hint)
                graph_terms = [
                    t for t in graph_terms if t in domain_terms
                ] or graph_terms

            if graph_terms:
                unique = list(dict.fromkeys(graph_terms))[
                    : self.config.max_total_synonyms
                ]
                expanded = f"{query} {' '.join(unique)}"
                return ExpansionResult(
                    original_query=query,
                    expanded_query=expanded,
                    added_terms=unique,
                )
        except Exception:
            logger.debug("Ontology expansion failed, falling back to v1")

        return self.expand(query, domain_hint)
