"""Opt-in bridge from the Twin query surface to the L3 intelligence engine.

The default ``l2`` mode deliberately imports no intelligence module.  In
``l3`` mode the engine itself is still constructed lazily on the first query,
after the host application has captured its fully initialised LightRAG
instance.  Retrieval always reuses that instance under the canonical
``MEMBER_OF``/advanced-filter request scope; this bridge never creates a
per-folder LightRAG workspace.

See ``docs/qa/intelligence-l3-degradation-contract.md`` and Forgejo #117.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Literal, Mapping

from .models import TwinQueryBody
from .doc_lookup import _resolve_chunk_to_doc_id
from .params import _make_query_param, _query_param_kwargs
from .paragraph_anchor import (
    collect_citation_evidence,
    compute_best_anchor,
    compute_best_structural_anchor,
)
from .request_scope import _retrieval_scope
from .response_sources import (
    _enrich_sources_doc_ids_from_file_path,
    _enrich_sources_with_source_links,
    _public_sources,
)

QueryEngineMode = Literal["l2", "l3"]
_QUERY_ENGINE_ENV = "TWIN_RAG_QUERY_ENGINE"
logger = logging.getLogger(__name__)


async def _load_chunk_records(rag: Any, chunk_ids: list[str]) -> dict[str, dict]:
    """Read cited chunks once for provenance and full-content anchors."""
    unique = list(dict.fromkeys(chunk_ids))
    if not unique:
        return {}
    get_by_ids = getattr(getattr(rag, "text_chunks", None), "get_by_ids", None)
    if not callable(get_by_ids):
        return {}
    try:
        rows = await get_by_ids(unique)
    except Exception:  # enrichment-only lookup, fail-soft
        logger.exception("twin_query: L3 cited-chunk enrichment failed")
        return {}
    if isinstance(rows, dict):
        return {
            chunk_id: row
            for chunk_id in unique
            if isinstance((row := rows.get(chunk_id)), dict)
        }
    if not isinstance(rows, list):
        return {}
    return {
        chunk_id: row for chunk_id, row in zip(unique, rows) if isinstance(row, dict)
    }


def query_engine_mode(environ: Mapping[str, str] | None = None) -> QueryEngineMode:
    """Return the fail-closed query-engine selection (default: ``l2``)."""
    raw = (environ if environ is not None else os.environ).get(_QUERY_ENGINE_ENV, "l2")
    mode = raw.strip().lower()
    if mode not in {"l2", "l3"}:
        raise RuntimeError(f"{_QUERY_ENGINE_ENV} must be 'l2' or 'l3'")
    return mode  # type: ignore[return-value]


class L3QueryRuntime:
    """Lazy L3 engine bound to the host application's LightRAG getter."""

    def __init__(self, get_rag) -> None:
        self._get_rag = get_rag
        self._engine: Any | None = None

    def engine(self):
        """Construct L3 lazily without enabling per-folder workspace routing."""
        if self._engine is None:
            # Imports stay inside the opt-in hot path.  A normal L2 process
            # neither imports nor constructs the L3 stack.
            from ...intelligence.config import TwinRAGConfig
            from ...intelligence.engine import TwinRAGEngine

            self._engine = TwinRAGEngine(
                TwinRAGConfig(
                    enable_folder_routing=False,
                    enable_workspace_routing=False,
                )
            )
        return self._engine

    async def retrieve(
        self,
        *,
        body: TwinQueryBody,
        folder: str,
        query: str,
    ):
        """Retrieve L3 chunks through the captured, initialised host RAG.

        ``folder`` has already passed the HTTP policy boundary.  Binding it
        here applies the exact same storage-level MEMBER_OF, tag, document and
        score filters as the L2 route before any chunk can enter an L3 prompt.
        """
        rag = self._get_rag()
        engine = self.engine()

        from lightrag.base import QueryParam

        param = _make_query_param(QueryParam, _query_param_kwargs(body))
        with _retrieval_scope(folder, body) as retrieval_scores:
            result = await rag.aquery_data(query, param=param)
            measured_scores = dict(retrieval_scores)
        chunks = engine.search._parse_lightrag_result(
            result,
            rag,
            source_folder=folder,
        )
        for chunk in chunks:
            score = measured_scores.get(chunk.chunk_id)
            if score is not None:
                chunk.metadata["measured_retrieval_score"] = score
        return chunks

    async def aquery(
        self,
        *,
        body: TwinQueryBody,
        folder: str,
        on_token=None,
        on_stage=None,
    ):
        """Run L3 over exactly the already-authorised active folder."""
        engine = self.engine()

        async def retrieve_scoped(requested_folder: str, query: str):
            if requested_folder != folder:
                raise PermissionError(
                    "L3 attempted to leave the authorised folder scope"
                )
            return await self.retrieve(body=body, folder=folder, query=query)

        return await engine.aquery(
            body.query,
            conversation_history=[
                message.model_dump() for message in body.conversation_history
            ],
            user_id=body.actor,
            folder=folder,
            # Explicitly empty means no implicit/default second folder.  The
            # engine's runtime config also disables autonomous folder routing.
            folders_publics=[],
            authorized_folders={folder},
            retrieval_provider=retrieve_scoped,
            enable_rerank=body.enable_rerank,
            on_synthesis_token=on_token,
            on_stage=on_stage,
            response_type=body.response_type,
        )

    async def project(self, result, *, folder: str) -> dict[str, Any]:
        """Map QueryResult through the applicable L2 source enrichments."""
        sources = []
        for citation in result.citations:
            score = citation.retrieval_score
            if not (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and 0.0 <= float(score) <= 1.0
            ):
                score = None
            name = (
                citation.document_path
                or citation.document_id
                or citation.source_workspace
                or "unknown"
            )
            source = {
                "n": citation.passage_index + 1,
                "type": "file",
                "name": name,
                "meta": citation.source_workspace,
                "score": score,
            }
            if citation.document_id:
                source["doc_id"] = citation.document_id
            if citation.chunk_id:
                source["chunk_id"] = citation.chunk_id
            sources.append(source)

        rag = self._get_rag()
        chunk_ids = [
            str(source["chunk_id"]) for source in sources if source.get("chunk_id")
        ]
        chunk_records = await _load_chunk_records(rag, chunk_ids)
        for source in sources:
            chunk_id = str(source.get("chunk_id") or "")
            record = chunk_records.get(chunk_id)
            if source.get("doc_id") or not isinstance(record, dict):
                continue
            for key in ("full_doc_id", "doc_id"):
                doc_id = record.get(key)
                if isinstance(doc_id, str) and doc_id:
                    source["doc_id"] = doc_id
                    break
        unresolved_chunk_ids = [
            str(source["chunk_id"])
            for source in sources
            if source.get("chunk_id") and not source.get("doc_id")
        ]
        chunk_to_doc = await _resolve_chunk_to_doc_id(rag, unresolved_chunk_ids)
        for source in sources:
            chunk_id = str(source.get("chunk_id") or "")
            if not source.get("doc_id") and chunk_id in chunk_to_doc:
                source["doc_id"] = chunk_to_doc[chunk_id]
        await _enrich_sources_doc_ids_from_file_path(rag, sources)

        citation_evidence = collect_citation_evidence(result.answer)
        for citation, source in zip(result.citations, sources):
            evidence = citation_evidence.get(source["n"])
            if evidence is None or not citation.chunk_id:
                continue
            record = chunk_records.get(str(citation.chunk_id))
            content = record.get("content") if isinstance(record, dict) else None
            if not isinstance(content, str) or not content:
                continue
            elected = None
            boundaries = record.get("twin_block_boundaries")
            if isinstance(boundaries, list) and boundaries:
                elected = compute_best_structural_anchor(
                    [(str(citation.chunk_id), content, boundaries)], evidence
                )
            if elected is None:
                elected = compute_best_anchor(
                    [(str(citation.chunk_id), content)], evidence
                )
            if elected is not None:
                source["chunk_id"], source["anchor"] = elected

        await _enrich_sources_with_source_links(sources, folder)
        sources = _public_sources(sources)

        trace = result.trace
        fallbacks = list(trace.fallbacks if trace is not None else [])[:4]
        return {
            "response": result.answer,
            "sources": sources,
            "answer_status": result.answer_status.value,
            "trace": {
                "engine": "l3",
                "degraded": bool(fallbacks),
                "fallbacks": fallbacks,
                "early_exit": trace.early_exit if trace is not None else None,
            },
        }


def build_l3_query_runtime(
    get_rag,
    *,
    environ: Mapping[str, str] | None = None,
) -> L3QueryRuntime | None:
    """Build the lazy adapter only when explicitly selected by the operator."""
    if query_engine_mode(environ) == "l2":
        return None
    return L3QueryRuntime(get_rag)


__all__ = [
    "L3QueryRuntime",
    "QueryEngineMode",
    "build_l3_query_runtime",
    "query_engine_mode",
]
