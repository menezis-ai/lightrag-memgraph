"""
intelligence/ontology/storage.py
=================================
Memgraph persistence layer for the ontology graph.

Follows the same patterns as kv_impl.py:
- Uses _pool.get_driver() for shared connection
- Label: Onto_{workspace}
- MERGE for upserts, DETACH DELETE for drops
- Always await result.consume()
"""

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ... import _pool
from ..._constants import (
    DEFAULT_WORKSPACE,
    MEMGRAPH_WORKSPACE_ENV,
    validate_identifier,
)
from .schema import (
    SEED_ENVIRONMENTS,
    SEED_METHODOLOGIES,
    SEED_SLAS,
)

logger = logging.getLogger("twin_rag_intelligence.ontology.storage")

_MIN_EXPANSION_HOPS = 1
_MAX_EXPANSION_HOPS = 3
_MAX_QUERY_NGRAM = 4
_MAX_QUERY_CANDIDATES = 64
_QUERY_TOKEN_RE = re.compile(r"[\w]+(?:[-./:][\w]+)*", re.UNICODE)

# These words may occur in almost every support question and are not useful
# ontology seeds on their own.  Technical identifiers remain eligible even
# when short because punctuation, digits, or acronym casing mark them as such.
_GENERIC_QUERY_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "after",
        "apres",
        "après",
        "are",
        "at",
        "au",
        "aux",
        "avec",
        "avant",
        "before",
        "by",
        "cause",
        "causes",
        "ce",
        "ces",
        "comment",
        "dans",
        "de",
        "des",
        "du",
        "en",
        "erreur",
        "error",
        "est",
        "et",
        "for",
        "from",
        "how",
        "in",
        "is",
        "issue",
        "la",
        "le",
        "les",
        "lorsque",
        "of",
        "on",
        "or",
        "out",
        "ou",
        "pour",
        "pourquoi",
        "problem",
        "probleme",
        "problème",
        "quel",
        "quelle",
        "quelles",
        "quels",
        "qui",
        "se",
        "son",
        "sont",
        "sur",
        "the",
        "that",
        "this",
        "to",
        "un",
        "une",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }
)


def _is_technical_token(raw: str) -> bool:
    """True for identifier-like tokens (digits, separators, acronyms)."""
    return (
        any(character.isdigit() for character in raw)
        or any(separator in raw for separator in "./:_")
        or (len(raw) >= 2 and raw.isupper())
    )


def _candidate_class(technical: bool, size: int) -> int:
    """Technical identifiers first, then more specific phrases."""
    if technical and size == 1:
        return 0
    if technical:
        return 1
    if size > 1:
        return 2
    return 3


def _ranked_query_phrases(
    tokens: list[str], raw_tokens: list[str]
) -> list[tuple[tuple[int, int, int, int, str], str]]:
    """Rank informative n-grams with a deterministic sort key."""
    ranked: list[tuple[tuple[int, int, int, int, str], str]] = []
    max_size = min(_MAX_QUERY_NGRAM, len(tokens))
    for size in range(1, max_size + 1):
        for start in range(0, len(tokens) - size + 1):
            end = start + size
            phrase_tokens = tokens[start:end]
            technical = any(
                _is_technical_token(raw_tokens[index]) for index in range(start, end)
            )
            informative = [
                token
                for token in phrase_tokens
                if token not in _GENERIC_QUERY_TERMS and len(token) >= 3
            ]
            if not technical and not informative:
                continue

            phrase = " ".join(phrase_tokens)
            # Source position and lexical value make the bound deterministic.
            key = (
                _candidate_class(technical, size),
                -size,
                -len(phrase),
                start,
                phrase,
            )
            ranked.append((key, phrase))
    return ranked


def _query_term_candidates(query: str) -> list[str]:
    """Return bounded, deterministic ontology seeds extracted from *query*.

    Exact matching is intentional: it prevents a broad lexical match from
    activating unrelated ontology branches.  The candidates include
    informative unigrams and contiguous n-grams, while generic question words
    cannot become seeds by themselves.
    """

    normalized = unicodedata.normalize("NFKC", query)
    raw_tokens = _QUERY_TOKEN_RE.findall(normalized)
    if not raw_tokens:
        return []

    tokens = [token.casefold() for token in raw_tokens]
    ranked = _ranked_query_phrases(tokens, raw_tokens)

    candidates: list[str] = []
    seen: set[str] = set()
    for _key, phrase in sorted(ranked):
        if phrase in seen:
            continue
        seen.add(phrase)
        candidates.append(phrase)
        if len(candidates) == _MAX_QUERY_CANDIDATES:
            break
    return candidates


@dataclass
class OntologyNode:
    name: str
    node_type: str
    properties: dict = field(default_factory=dict)
    confidence: float = 1.0
    source_doc: str = ""


@dataclass
class OntologyEdge:
    source_name: str
    source_type: str
    target_name: str
    target_type: str
    relation_type: str
    confidence: float = 1.0
    source_doc: str = ""


class OntologyStorage:
    """Memgraph persistence for the ontology graph."""

    def __init__(self, workspace: str | None = None) -> None:
        raw_workspace = workspace or os.environ.get(
            MEMGRAPH_WORKSPACE_ENV, DEFAULT_WORKSPACE
        )
        self.workspace = validate_identifier(raw_workspace, "workspace")
        self._driver = None
        self._database = None

    def _label(self) -> str:
        return f"Onto_{self.workspace}"

    async def initialize(self) -> None:
        """Create indexes and seed normative data."""
        self._driver, self._database = await _pool.get_driver()
        label = self._label()

        async with self._driver.session(database=self._database) as session:
            # Index on name + node_type
            try:
                await session.run(f"CREATE INDEX ON :`{label}`(name)")
            except Exception:
                pass

            try:
                await session.run(f"CREATE INDEX ON :`{label}`(node_type)")
            except Exception:
                pass

            # Seed normative data
            await self._seed_normative(session, label)

    async def _seed_normative(self, session, label: str) -> None:
        """Seed Methodology, SLA, and Environment nodes (idempotent)."""
        now = datetime.now(timezone.utc).isoformat()

        # Methodologies
        for m in SEED_METHODOLOGIES:
            result = await session.run(
                f"""
                MERGE (n:`{label}` {{name: $name, node_type: 'Methodology'}})
                ON CREATE SET n.version = $version,
                              n.framework = $framework,
                              n.confidence = 1.0,
                              n.created_at = $ts
                """,
                name=m["name"],
                version=m["version"],
                framework=m["framework"],
                ts=now,
            )
            await result.consume()

        # SLAs
        for s in SEED_SLAS:
            result = await session.run(
                f"""
                MERGE (n:`{label}` {{name: $priority, node_type: 'SLA'}})
                ON CREATE SET n.priority = $priority,
                              n.gtr_hours = $gtr_hours,
                              n.description = $description,
                              n.confidence = 1.0,
                              n.created_at = $ts
                """,
                priority=s["priority"],
                gtr_hours=s["gtr_hours"],
                description=s["description"],
                ts=now,
            )
            await result.consume()

        # Environments
        for e in SEED_ENVIRONMENTS:
            result = await session.run(
                f"""
                MERGE (n:`{label}` {{name: $name, node_type: 'Environment'}})
                ON CREATE SET n.tier = $tier,
                              n.confidence = 1.0,
                              n.created_at = $ts
                """,
                name=e["name"],
                tier=e["tier"],
                ts=now,
            )
            await result.consume()

        logger.info(
            "[Ontology:%s] Seeded %d methodologies, %d SLAs, %d environments",
            self.workspace,
            len(SEED_METHODOLOGIES),
            len(SEED_SLAS),
            len(SEED_ENVIRONMENTS),
        )

    async def upsert_nodes(self, nodes: list[OntologyNode]) -> None:
        """Upsert ontology nodes via UNWIND + MERGE."""
        if not nodes:
            return

        label = self._label()
        now = datetime.now(timezone.utc).isoformat()

        entries = [
            {
                "name": n.name,
                "node_type": n.node_type,
                "confidence": n.confidence,
                "source_doc": n.source_doc,
                "props": n.properties,
                "ts": now,
            }
            for n in nodes
        ]

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                UNWIND $entries AS e
                MERGE (n:`{label}` {{name: e.name, node_type: e.node_type}})
                ON CREATE SET n.created_at = e.ts
                SET n.confidence = e.confidence,
                    n.source_doc = e.source_doc,
                    n.updated_at = e.ts
                """,
                entries=entries,
            )
            await result.consume()

        logger.info("[Ontology:%s] Upserted %d nodes", self.workspace, len(nodes))

    async def upsert_edges(self, edges: list[OntologyEdge]) -> None:
        """Upsert ontology edges, grouped by relation_type.

        Memgraph requires static relationship types in MERGE,
        so we group edges by type and run one UNWIND per group.
        """
        if not edges:
            return

        label = self._label()
        now = datetime.now(timezone.utc).isoformat()

        # Group by relation_type
        groups: dict[str, list[dict]] = {}
        for e in edges:
            entry = {
                "src_name": e.source_name,
                "src_type": e.source_type,
                "tgt_name": e.target_name,
                "tgt_type": e.target_type,
                "confidence": e.confidence,
                "source_doc": e.source_doc,
                "ts": now,
            }
            groups.setdefault(e.relation_type, []).append(entry)

        async with self._driver.session(database=self._database) as session:
            for rel_type, entries in groups.items():
                try:
                    safe_rel_type = validate_identifier(str(rel_type), "relation_type")
                except ValueError:
                    logger.warning(
                        "[Ontology:%s] Dropping %d edge(s) with unsafe "
                        "relation_type %r (failed identifier validation)",
                        self.workspace,
                        len(entries),
                        rel_type,
                    )
                    continue
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MATCH (src:`{label}` {{name: e.src_name, node_type: e.src_type}})
                    MATCH (tgt:`{label}` {{name: e.tgt_name, node_type: e.tgt_type}})
                    MERGE (src)-[r:`{safe_rel_type}`]->(tgt)
                    SET r.confidence = e.confidence,
                        r.source_doc = e.source_doc,
                        r.created_at = e.ts
                    """,
                    entries=entries,
                )
                await result.consume()

        logger.info(
            "[Ontology:%s] Upserted %d edges across %d relation types",
            self.workspace,
            len(edges),
            len(groups),
        )

    async def has_data(self) -> bool:
        """Check if any ontology data exists (lightweight count query)."""
        label = self._label()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS cnt LIMIT 1"
            )
            record = await result.single()
            await result.consume()
            return record["cnt"] > 0 if record else False

    async def query_expansion(self, query: str, max_hops: int = 2) -> list[str]:
        """Return deterministically ranked terms related to query candidates.

        Candidate terms are exact, bounded n-grams extracted from the natural
        language query.  For every reachable term, the best path score is::

            max_path(product(clamp(relation.confidence, 0, 1)) / path_hops)

        This score uses stored evidence confidence and explicitly penalizes
        longer paths.  It is a ranking heuristic, not a measured recall gain.
        """
        if (
            isinstance(max_hops, bool)
            or not isinstance(max_hops, int)
            or not _MIN_EXPANSION_HOPS <= max_hops <= _MAX_EXPANSION_HOPS
        ):
            raise ValueError(
                "max_hops must be an integer between "
                f"{_MIN_EXPANSION_HOPS} and {_MAX_EXPANSION_HOPS}"
            )

        candidate_terms = _query_term_candidates(query)
        if not candidate_terms:
            return []

        label = self._label()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (start:`{label}` {{node_type: 'Term'}})
                WHERE toLower(start.name) IN $candidate_terms
                MATCH path=(start)-[:SYNONYM|RELATED_TO|CO_OCCURS*1..{max_hops}]-(related:`{label}`)
                WHERE related.node_type = 'Term' AND related.name <> start.name
                WITH related.name AS name,
                     length(path) AS hops,
                     reduce(
                         confidence_product = 1.0,
                         relation IN relationships(path) |
                         confidence_product *
                         CASE
                             WHEN relation.confidence IS NULL THEN 0.0
                             WHEN toFloat(relation.confidence) < 0.0 THEN 0.0
                             WHEN toFloat(relation.confidence) > 1.0 THEN 1.0
                             ELSE toFloat(relation.confidence)
                         END
                     ) AS confidence_product
                WITH name,
                     hops,
                     max(confidence_product / toFloat(hops)) AS path_score
                ORDER BY name ASC, path_score DESC, hops ASC
                WITH name,
                     collect({{score: path_score, hops: hops}})[0] AS best_path
                RETURN name
                ORDER BY best_path.score DESC,
                         best_path.hops ASC,
                         toLower(name) ASC,
                         name ASC
                LIMIT 20
                """,
                candidate_terms=candidate_terms,
            )
            raw_names = [record["name"] async for record in result]
            names = list(dict.fromkeys(raw_names))
            await result.consume()
            return names

    async def get_domain_terms(self, domain: str) -> list[str]:
        """Get all terms belonging to a domain."""
        label = self._label()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (d:`{label}` {{name: $domain, node_type: 'Domain'}})
                      <-[:PART_OF]-(t:`{label}` {{node_type: 'Term'}})
                RETURN t.name AS name
                """,
                domain=domain,
            )
            names = [record["name"] async for record in result]
            await result.consume()
            return names

    async def drop(self) -> dict[str, str]:
        """Drop all ontology data for this workspace."""
        label = self._label()
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                await result.consume()
            return {"status": "success", "message": f"Ontology {label} dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
