"""F06 -- Folder Router (Nexus Router embarque).

Resolves which Twin folders / Memgraph workspaces to query for a given question.
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


@dataclass(init=False)
class FolderRoutingResult:
    """F06 routing result."""

    folders: list[str]
    public_folders: list[str]
    strategy: str  # "l4_override" | "topology" | "ontology" | "keyword" | "default"
    confidence: float = 1.0
    matched_keywords: list[str] = field(default_factory=list)

    def __init__(
        self,
        folders: list[str] | None = None,
        public_folders: list[str] | None = None,
        strategy: str = "default",
        confidence: float = 1.0,
        matched_keywords: list[str] | None = None,
        workspaces: list[str] | None = None,
        workspaces_publics: list[str] | None = None,
    ) -> None:
        self.folders = folders if folders is not None else workspaces or []
        self.public_folders = (
            public_folders if public_folders is not None else workspaces_publics or []
        )
        self.strategy = strategy
        self.confidence = confidence
        self.matched_keywords = matched_keywords or []

    @property
    def workspaces(self) -> list[str]:
        """Deprecated compatibility alias for internal LightRAG naming."""
        return self.folders

    @property
    def workspaces_publics(self) -> list[str]:
        """Deprecated compatibility alias for internal LightRAG naming."""
        return self.public_folders


# --- Routing rules ---


@dataclass
class FolderRoutingRule:
    """A keyword -> folder routing rule."""

    keywords: list[str]
    target_folder: str
    folder_type: str = "public"  # "public" or "private"
    confidence: float = 0.9

    def match(self, query_lower: str) -> list[str]:
        """Return matched keywords found in the query."""
        return [kw for kw in self.keywords if kw.lower() in query_lower]


# --- Main router ---


class FolderRouter:
    """F06 -- Nexus Router embedded in the ReAct pipeline.

    Usage MVP:
        router = FolderRouter.from_json("routing_rules.json")
        result = await router.route(query="Probleme RMAN Oracle")

    Usage with L4 override:
        result = await router.route(
            query="...",
            provided_folders=["cib"],
            provided_public_folders=["commons"],
        )
    """

    def __init__(
        self,
        rules: list[FolderRoutingRule],
        default_folder: str = "commons",
    ) -> None:
        self._rules = rules
        self._default_folder = default_folder
        self._default_workspace = default_folder
        # Pre-compile regex patterns for hot-path performance
        self._compiled: list[tuple[FolderRoutingRule, re.Pattern]] = []
        for rule in rules:
            pattern = "|".join(re.escape(kw) for kw in rule.keywords)
            self._compiled.append((rule, re.compile(pattern, re.IGNORECASE)))

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        default_folder: str = "commons",
        default_workspace: str | None = None,
    ) -> FolderRouter:
        """Load routing rules from a JSON file."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        rules = [
            FolderRoutingRule(
                keywords=r["keywords"],
                target_folder=r.get("target_folder") or r["target_workspace"],
                folder_type=r.get("folder_type") or r.get("workspace_type", "public"),
                confidence=r.get("confidence", 0.9),
            )
            for r in data.get("rules", [])
        ]
        fallback_folder = default_workspace or default_folder
        return cls(
            rules=rules,
            default_folder=data.get("default_folder")
            or data.get("default_workspace", fallback_folder),
        )

    async def route(  # NOSONAR - async contract.
        self,
        query: str,
        *,
        provided_folders: list[str] | None = None,
        provided_public_folders: list[str] | None = None,
        provided_workspaces: list[str] | None = None,
        provided_workspaces_publics: list[str] | None = None,
        topology_context: TopologyContext | None = None,
    ) -> FolderRoutingResult:
        """Resolve folders via waterfall cascade.

        Priority:
            1. L4 override (provided_folders)
            2. TopologyContext (Shadow Nodes resolved by TigerGraph)
            3. Keyword match (routing_rules.json)
            4. Default fallback (commons)
        """
        # Priority 1: L4 Override
        folders_override = provided_folders
        public_folders_override = provided_public_folders
        if folders_override is None and provided_workspaces is not None:
            folders_override = provided_workspaces
        if public_folders_override is None and provided_workspaces_publics is not None:
            public_folders_override = provided_workspaces_publics

        if folders_override is not None:
            logger.debug(
                "F06: L4 override -> %s + %s",
                folders_override,
                public_folders_override,
            )
            return FolderRoutingResult(
                folders=folders_override,
                public_folders=public_folders_override or [self._default_folder],
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
            return FolderRoutingResult(
                folders=topology_context.workspaces,
                public_folders=topology_context.workspaces_publics
                or [self._default_folder],
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
                result.public_folders,
                result.matched_keywords,
            )
            return result

        # Fallback: Default
        logger.debug("F06: default fallback -> %s", self._default_folder)
        return FolderRoutingResult(
            folders=[],
            public_folders=[self._default_folder],
            strategy="default",
            confidence=0.5,
        )

    def _keyword_match(self, query: str) -> FolderRoutingResult | None:
        """Strategy 4: regex match against routing_rules.json."""
        matched_private: list[str] = []
        matched_public: list[str] = []
        all_matched_keywords: list[str] = []
        max_confidence = 0.0

        for rule, pattern in self._compiled:
            matches = pattern.findall(query)
            if matches:
                if rule.folder_type == "private":
                    matched_private.append(rule.target_folder)
                else:
                    matched_public.append(rule.target_folder)
                all_matched_keywords.extend(matches)
                max_confidence = max(max_confidence, rule.confidence)

        if not matched_private and not matched_public:
            return None

        # Always include default folder in publics.
        if self._default_folder not in matched_public:
            matched_public.append(self._default_folder)

        return FolderRoutingResult(
            folders=list(dict.fromkeys(matched_private)),
            public_folders=list(dict.fromkeys(matched_public)),
            strategy="keyword",
            confidence=max_confidence,
            matched_keywords=all_matched_keywords,
        )


# Backwards-compatible names for imports that still use internal LightRAG wording.
RoutingResult = FolderRoutingResult
RoutingRule = FolderRoutingRule
WorkspaceRouter = FolderRouter
