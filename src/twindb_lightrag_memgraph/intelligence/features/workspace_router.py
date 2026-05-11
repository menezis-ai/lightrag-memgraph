"""F06 -- Workspace Router (Nexus Router embarque).

Resolves which Memgraph workspaces to query for a given question.
Cascade: L4 override > TopologyContext > Ontology > Keyword > Default.

No external dependencies. Pure function (input -> output).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("twindb_lightrag_memgraph.intelligence.router")


# --- Data contracts ---


@dataclass
class TopologyContext:
    """Contract between the Topology Agent (TigerGraph) and Knowledge Agent (Memgraph).

    Provided by L4 when the Topology Agent has resolved Shadow Nodes.
    Can be None if the query has no topological context.
    """

    servers: list[str] = field(default_factory=list)
    workspaces: list[str] = field(default_factory=list)
    workspaces_publics: list[str] = field(default_factory=list)
    topology_path: str = ""
    topology_context: str | None = None


@dataclass
class RoutingResult:
    """F06 routing result."""

    workspaces: list[str]
    workspaces_publics: list[str]
    strategy: str  # "l4_override" | "topology" | "ontology" | "keyword" | "default"
    confidence: float = 1.0
    matched_keywords: list[str] = field(default_factory=list)


# --- Routing rules ---


@dataclass
class RoutingRule:
    """A keyword -> workspace routing rule."""

    keywords: list[str]
    target_workspace: str
    workspace_type: str = "public"  # "public" or "private"
    confidence: float = 0.9

    def match(self, query_lower: str) -> list[str]:
        """Return matched keywords found in the query."""
        return [kw for kw in self.keywords if kw.lower() in query_lower]


# --- Main router ---


class WorkspaceRouter:
    """F06 -- Nexus Router embedded in the ReAct pipeline.

    Usage MVP:
        router = WorkspaceRouter.from_json("routing_rules.json")
        result = await router.route(query="Probleme RMAN Oracle")

    Usage with L4 override:
        result = await router.route(
            query="...",
            provided_workspaces=["cib"],
            provided_workspaces_publics=["commons"],
        )
    """

    def __init__(
        self,
        rules: list[RoutingRule],
        default_workspace: str = "commons",
    ) -> None:
        self._rules = rules
        self._default_workspace = default_workspace
        # Pre-compile regex patterns for hot-path performance
        self._compiled: list[tuple[RoutingRule, re.Pattern]] = []
        for rule in rules:
            pattern = "|".join(re.escape(kw) for kw in rule.keywords)
            self._compiled.append((rule, re.compile(pattern, re.IGNORECASE)))

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        default_workspace: str = "commons",
    ) -> WorkspaceRouter:
        """Load routing rules from a JSON file."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        rules = [
            RoutingRule(
                keywords=r["keywords"],
                target_workspace=r["target_workspace"],
                workspace_type=r.get("workspace_type", "public"),
                confidence=r.get("confidence", 0.9),
            )
            for r in data.get("rules", [])
        ]
        return cls(
            rules=rules,
            default_workspace=data.get("default_workspace", default_workspace),
        )

    async def route(
        self,
        query: str,
        *,
        provided_workspaces: list[str] | None = None,
        provided_workspaces_publics: list[str] | None = None,
        topology_context: TopologyContext | None = None,
    ) -> RoutingResult:
        """Resolve workspaces via waterfall cascade.

        Priority:
            1. L4 override (provided_workspaces)
            2. TopologyContext (Shadow Nodes resolved by TigerGraph)
            3. Keyword match (routing_rules.json)
            4. Default fallback (commons)
        """
        # Priority 1: L4 Override
        if provided_workspaces is not None:
            logger.debug(
                "F06: L4 override -> %s + %s",
                provided_workspaces,
                provided_workspaces_publics,
            )
            return RoutingResult(
                workspaces=provided_workspaces,
                workspaces_publics=provided_workspaces_publics
                or [self._default_workspace],
                strategy="l4_override",
                confidence=1.0,
            )

        # Priority 2: Topology Context (Phase 3)
        if topology_context is not None and topology_context.workspaces:
            logger.debug(
                "F06: topology -> %s + %s",
                topology_context.workspaces,
                topology_context.workspaces_publics,
            )
            return RoutingResult(
                workspaces=topology_context.workspaces,
                workspaces_publics=topology_context.workspaces_publics
                or [self._default_workspace],
                strategy="topology",
                confidence=1.0,
            )

        # Priority 3: Ontology Traversal (Phase 2 -- not yet implemented)
        # Will use QueryExpander v2 Cypher 2-hop traversal in Onto_{ws}.

        # Priority 4: Keyword Match (MVP)
        result = self._keyword_match(query)
        if result is not None:
            logger.debug(
                "F06: keyword -> %s (matched: %s)",
                result.workspaces_publics,
                result.matched_keywords,
            )
            return result

        # Fallback: Default
        logger.debug("F06: default fallback -> %s", self._default_workspace)
        return RoutingResult(
            workspaces=[],
            workspaces_publics=[self._default_workspace],
            strategy="default",
            confidence=0.5,
        )

    def _keyword_match(self, query: str) -> RoutingResult | None:
        """Strategy 4: regex match against routing_rules.json."""
        matched_private: list[str] = []
        matched_public: list[str] = []
        all_matched_keywords: list[str] = []
        max_confidence = 0.0

        for rule, pattern in self._compiled:
            matches = pattern.findall(query)
            if matches:
                if rule.workspace_type == "private":
                    matched_private.append(rule.target_workspace)
                else:
                    matched_public.append(rule.target_workspace)
                all_matched_keywords.extend(matches)
                max_confidence = max(max_confidence, rule.confidence)

        if not matched_private and not matched_public:
            return None

        # Always include default workspace in publics
        if self._default_workspace not in matched_public:
            matched_public.append(self._default_workspace)

        return RoutingResult(
            workspaces=list(dict.fromkeys(matched_private)),
            workspaces_publics=list(dict.fromkeys(matched_public)),
            strategy="keyword",
            confidence=max_confidence,
            matched_keywords=all_matched_keywords,
        )
