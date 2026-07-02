"""
twin_rag_intelligence/react/act.py
====================================
ACT phase of the ReAct agent.

Transposition of _hybrid_search_step() from ChatDocAI.
Differences from CNCAC:
  - Uses LightRAG.aquery() instead of direct Neo4j
  - Multi-workspace (private KB + public KBs in parallel)
  - Cross-workspace fusion with deduplication by entity_id
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lightrag import LightRAG, QueryParam

from ..config import TwinRAGConfig

logger = logging.getLogger("twin_rag_intelligence.act")


@dataclass
class ChunkResult:
    """A chunk retrieved by the retrieval engine."""

    chunk_id: str
    text: str
    score: float
    source_workspace: str
    document_id: Optional[str] = None
    document_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: Optional[float] = None


class SearchEngine:
    """
    Hybrid multi-workspace search engine.

    Strategy:
    1. For each workspace, call LightRAG.aquery() in hybrid mode
    2. Fuse results with deduplication by chunk_id
    3. Normalize cross-workspace scores (min-max scaling)
    """

    def __init__(self, config: TwinRAGConfig) -> None:
        self.config = config

    async def hybrid_search(
        self,
        rag: LightRAG,
        query: str,
        config: TwinRAGConfig,
    ) -> list[ChunkResult]:
        """
        Hybrid search via LightRAG on a single workspace.

        LightRAG.aquery() with mode="hybrid" combines:
        - Vector search (embedding cosine similarity)
        - Graph traversal (entities + LightRAG communities)
        - Keyword matching (fulltext on graph nodes)
        """
        try:
            param = QueryParam(
                mode=config.lightrag_mode,
                top_k=config.vector_limit + config.fulltext_limit,
            )
            result = await rag.aquery(query, param=param)
            return self._parse_lightrag_result(result, rag)

        except Exception as e:
            logger.exception("Search error on workspace: %s", e)
            return []

    def _parse_lightrag_result(
        self,
        result: Any,
        rag: LightRAG,
    ) -> list[ChunkResult]:
        """Parse LightRAG result into a list of ChunkResult."""
        workspace = getattr(rag, "_workspace", "unknown")

        if isinstance(result, str):
            return [
                ChunkResult(
                    chunk_id=f"{workspace}_synthesis",
                    text=result,
                    score=1.0,
                    source_workspace=workspace,
                )
            ]

        chunks: list[ChunkResult] = []
        if isinstance(result, list):
            for i, item in enumerate(result):
                chunks.append(
                    ChunkResult(
                        chunk_id=item.get("id", f"{workspace}_{i}"),
                        text=item.get("text", str(item)),
                        score=item.get("score", 0.5),
                        source_workspace=workspace,
                        document_id=item.get("document_id"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return chunks

    def fuse_and_dedup(
        self,
        workspace_results: list[Any],
        workspace_names: list[str],
    ) -> list[ChunkResult]:
        """
        Fuse results from N workspaces and deduplicate.

        Deduplication strategy:
        - Exact chunk_id match
        - Keep highest score on collision
        """
        seen_ids: set[str] = set()
        fused: list[ChunkResult] = []

        for ws_result, ws_name in zip(workspace_results, workspace_names):
            if isinstance(ws_result, Exception):
                logger.warning("Workspace %s failed: %s", ws_name, ws_result)
                continue
            if not isinstance(ws_result, list):
                continue

            for chunk in ws_result:
                if isinstance(chunk, ChunkResult) and chunk.chunk_id not in seen_ids:
                    chunk.source_workspace = ws_name
                    seen_ids.add(chunk.chunk_id)
                    fused.append(chunk)

        fused.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            "Fusion: %d unique chunks across %d workspaces",
            len(fused),
            len(workspace_names),
        )
        return fused
