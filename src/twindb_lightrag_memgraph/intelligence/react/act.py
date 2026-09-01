"""
twin_rag_intelligence/react/act.py
====================================
ACT phase of the ReAct agent.

Transposition of _hybrid_search_step() from ChatDocAI.
Differences from CNCAC:
  - Uses LightRAG.aquery_data() instead of direct Neo4j
  - Multi-workspace (private KB + public KBs in parallel)
  - Rank-based cross-workspace fusion with deduplication by chunk_id
"""

import logging
import math
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
    1. Retrieve structured LightRAG data independently in each workspace
    2. Extract only document chunks with their upstream provenance
    3. Fuse workspace rankings with Reciprocal Rank Fusion (RRF)

    RRF deliberately consumes ranks rather than raw retrieval scores. Scores
    emitted by different workspaces or retrieval channels need not be
    calibrated, whereas their within-workspace ordering has a stable meaning.
    """

    RRF_K = 60

    def __init__(self, config: TwinRAGConfig) -> None:
        self.config = config

    async def hybrid_search(
        self,
        rag: LightRAG,
        query: str,
        config: TwinRAGConfig,
    ) -> list[ChunkResult]:
        """
        Structured retrieval via LightRAG on a single workspace.

        ``aquery_data`` stops before answer generation and returns the chunks
        that LightRAG selected for context. The configured LightRAG mode still
        controls whether retrieval uses vector, graph, or mixed evidence.
        Generated answers from ``aquery`` are never accepted as passages.
        """
        try:
            param = QueryParam(
                mode=config.lightrag_mode,
                top_k=config.vector_limit + config.fulltext_limit,
            )
            query_data = getattr(rag, "aquery_data", None)
            if not callable(query_data):
                logger.error(
                    "LightRAG structured retrieval is unavailable; "
                    "refusing to treat a generated answer as passage evidence"
                )
                return []
            result = await query_data(query, param=param)
            return self._parse_lightrag_result(result, rag)

        except Exception as e:
            logger.exception("Search error on workspace: %s", e)
            return []

    def _parse_lightrag_result(
        self,
        result: Any,
        rag: LightRAG,
        source_folder: str | None = None,
    ) -> list[ChunkResult]:
        """Parse structured LightRAG retrieval data into document chunks.

        LightRAG versions supported by this project return an envelope with a
        ``data.chunks`` list. Direct ``chunks`` dictionaries and lists remain
        accepted for compatibility with earlier structured adapters. Strings
        are intentionally rejected because they are generated answers, not
        retrieval evidence.
        """
        # The server runtime uses one physical LightRAG workspace and scopes
        # logical folders through MEMBER_OF.  In that path the source folder
        # must come from the already-authorised request scope, never from
        # ``rag.workspace`` (which names the physical namespace).
        workspace = source_folder or self._workspace_of(rag)
        raw_chunks = self._extract_raw_chunks(result)
        if raw_chunks is None:
            return []

        chunks: list[ChunkResult] = []
        for rank, item in enumerate(raw_chunks, start=1):
            chunk = self._chunk_from_item(item, rank, workspace)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _workspace_of(rag: LightRAG) -> str:
        """Resolve the workspace name, falling back to the legacy attribute."""
        workspace = getattr(rag, "workspace", None)
        if isinstance(workspace, str) and workspace:
            return workspace
        legacy_workspace = getattr(rag, "_workspace", None)
        if isinstance(legacy_workspace, str) and legacy_workspace:
            return legacy_workspace
        return "unknown"

    @staticmethod
    def _extract_raw_chunks(result: Any) -> Optional[list]:
        """Locate the chunk list in a structured retrieval envelope, if any."""
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            if result.get("status") == "failure":
                return None
            data = result.get("data")
            if isinstance(data, dict):
                raw_chunks = data.get("chunks")
            else:
                raw_chunks = result.get("chunks")
            if isinstance(raw_chunks, list):
                return raw_chunks
        return None

    def _chunk_from_item(
        self, item: Any, rank: int, workspace: str
    ) -> Optional[ChunkResult]:
        """Build one ChunkResult from a raw chunk dict; None when malformed."""
        if not isinstance(item, dict):
            return None

        chunk_id = item.get("chunk_id") or item.get("id") or item.get("_id")
        text = item.get("content") or item.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            return None
        if not isinstance(text, str) or not text.strip():
            return None

        score = self._raw_score(item)
        raw_metadata = item.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata["retrieval_rank"] = rank
        reference_id = item.get("reference_id")
        if reference_id is not None:
            metadata["reference_id"] = str(reference_id)

        return ChunkResult(
            chunk_id=chunk_id,
            text=text,
            score=score,
            source_workspace=str(workspace),
            document_id=self._first_text(item, "full_doc_id", "doc_id", "document_id"),
            document_path=self._first_text(
                item, "file_path", "document_path", "source"
            ),
            metadata=metadata,
        )

    @staticmethod
    def _first_text(item: dict[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _raw_score(item: dict[str, Any]) -> float:
        """Preserve an upstream numeric score without inventing one."""
        for key in ("score", "similarity", "cosine_similarity"):
            value = item.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                return float(value)
        return 0.0

    def fuse_and_dedup(
        self,
        workspace_results: list[Any],
        workspace_names: list[str],
    ) -> list[ChunkResult]:
        """
        Fuse workspace rankings with deterministic Reciprocal Rank Fusion.

        Each workspace contributes at most ``1 / (RRF_K + rank)`` per exact
        chunk id. A chunk retrieved from several workspaces accumulates those
        contributions. Raw scores never cross workspace boundaries. Ties are
        resolved by chunk id so repeated runs over the same rankings are
        deterministic.
        """
        rankings = self._collect_rankings(workspace_results, workspace_names)

        fused = [self._fused_chunk(by_workspace) for by_workspace in rankings.values()]

        fused.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))

        logger.info(
            "Fusion: %d unique chunks across %d workspaces",
            len(fused),
            len(workspace_names),
        )
        return fused

    def _collect_rankings(
        self,
        workspace_results: list[Any],
        workspace_names: list[str],
    ) -> dict[str, dict[str, tuple[int, ChunkResult]]]:
        """Index each chunk id by workspace with its best (lowest) rank."""
        rankings: dict[str, dict[str, tuple[int, ChunkResult]]] = {}

        for ws_result, ws_name in zip(workspace_results, workspace_names):
            if isinstance(ws_result, Exception):
                logger.warning("Workspace %s failed: %s", ws_name, ws_result)
                continue
            if not isinstance(ws_result, list):
                continue

            best_in_workspace = self._best_ranked_in_workspace(ws_result)
            for chunk_id, ranked_chunk in best_in_workspace.items():
                by_workspace = rankings.setdefault(chunk_id, {})
                current = by_workspace.get(ws_name)
                if current is None or ranked_chunk[0] < current[0]:
                    by_workspace[ws_name] = ranked_chunk
        return rankings

    @staticmethod
    def _best_ranked_in_workspace(
        ws_result: list,
    ) -> dict[str, tuple[int, ChunkResult]]:
        """Keep the best (lowest) rank per chunk id within one workspace."""
        best_in_workspace: dict[str, tuple[int, ChunkResult]] = {}
        for rank, chunk in enumerate(ws_result, start=1):
            if not isinstance(chunk, ChunkResult) or not chunk.chunk_id:
                continue
            current = best_in_workspace.get(chunk.chunk_id)
            if current is None or rank < current[0]:
                best_in_workspace[chunk.chunk_id] = (rank, chunk)
        return best_in_workspace

    def _fused_chunk(
        self, by_workspace: dict[str, tuple[int, ChunkResult]]
    ) -> ChunkResult:
        """Fuse one chunk's per-workspace rankings into a single RRF result."""
        occurrences = sorted(
            (
                (workspace, rank, chunk)
                for workspace, (rank, chunk) in by_workspace.items()
            ),
            key=lambda occurrence: (occurrence[1], occurrence[0]),
        )
        representative_workspace, _rank, representative = occurrences[0]
        source_workspaces = sorted(by_workspace)
        rrf_ranks = {
            workspace: by_workspace[workspace][0] for workspace in source_workspaces
        }
        retrieval_scores = {
            workspace: by_workspace[workspace][1].score
            for workspace in source_workspaces
        }
        fused_score = math.fsum(
            1.0 / (self.RRF_K + rank) for _workspace, rank, _chunk in occurrences
        )
        metadata = dict(representative.metadata)
        metadata.update(
            {
                "source_workspaces": source_workspaces,
                "rrf_k": self.RRF_K,
                "rrf_ranks": rrf_ranks,
                "retrieval_scores": retrieval_scores,
            }
        )
        return ChunkResult(
            chunk_id=representative.chunk_id,
            text=representative.text,
            score=fused_score,
            source_workspace=representative_workspace,
            document_id=representative.document_id,
            document_path=representative.document_path,
            metadata=metadata,
            rerank_score=representative.rerank_score,
        )
