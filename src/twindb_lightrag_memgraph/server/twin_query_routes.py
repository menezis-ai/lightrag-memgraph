"""Twin overlay POST /twin/api/query — structured retrieval response.

LightRAG's native ``POST /query`` returns just ``{"response": str}``
which gives the React port no way to render clickable citations or a
sources panel. This wrapper:

1. Runs the standard ``aquery()`` to get the synthesised answer (one
   LLM call, same as before).
2. Issues a cheap vector-only retrieval against ``chunks_vdb`` to
   surface the top-k chunks the response was grounded on, then looks
   them up against ``DocStatus`` to enrich each source with its
   parent document's display path.
3. Returns ``{response, sources}`` where each source carries the
   minimal contract the React ``RetrievalSource`` type expects:
   ``n, type, name, meta, score, doc_id?, chunk_id?``.

Deliberate trade-offs:
- The vector retrieval reuses LightRAG's existing chunks_vdb, so the
  cost is the embedding-search round-trip (no extra LLM call).
- Sources reflect *retrieval*, not necessarily *citation*. LightRAG
  does not currently emit ``{cite:N}`` markers in the response — the
  React port surfaces sources in a sidebar; inline highlighting
  remains a follow-up once the prompt is augmented to cite by index.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TwinQueryBody(BaseModel):
    query: str
    mode: str = Field(default="mix")
    top_k: int = Field(default=10, ge=1, le=200)
    max_total_tokens: int | None = Field(default=None, ge=1)
    only_need_context: bool = Field(default=False)
    only_need_prompt: bool = Field(default=False)


class TwinRetrievalSource(BaseModel):
    n: int
    type: str = "file"
    name: str
    meta: str | None = None
    score: float = 0.0
    doc_id: str | None = None
    chunk_id: str | None = None


class TwinQueryResponse(BaseModel):
    response: str
    sources: list[TwinRetrievalSource] = Field(default_factory=list)


def _safe_get_score(result: dict[str, Any], rank: int, total: int) -> float:
    """Best-effort score for a retrieval row.

    The LightRAG ``MemgraphVectorDBStorage`` projection includes the
    cosine score under ``__metrics__`` / ``score`` depending on the
    backend version. When neither is present we synthesise a smooth
    rank-based value so the UI can sort consistently.
    """
    for key in ("score", "similarity", "cosine_similarity"):
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])
    metrics = result.get("__metrics__")
    if isinstance(metrics, dict):
        for key in ("score", "similarity", "cosine_similarity"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                return float(metrics[key])
    if total <= 0:
        return 0.5
    # Smooth descent from 0.95 (rank 0) to 0.50 (rank total-1).
    return round(0.95 - 0.45 * (rank / max(total - 1, 1)), 3)


def _chunk_to_meta(chunk: dict[str, Any]) -> str | None:
    """Cheap meta string — chunk order index when present, else the
    short id suffix."""
    idx = chunk.get("chunk_order_index")
    if isinstance(idx, int):
        return f"chunk {idx}"
    cid = chunk.get("chunk_id") or chunk.get("id") or ""
    if isinstance(cid, str) and "-" in cid:
        return f"chunk {cid.split('-')[-1][:8]}"
    return None


async def _resolve_doc_for_chunk(rag: Any, chunk_id: str) -> str | None:
    """Best-effort: walk DocStatus to find which doc owns this chunk
    so the source surfaces a real ``doc_id`` instead of just the
    file path."""
    if not chunk_id:
        return None
    try:
        # `get_docs_by_chunks` is the modern LightRAG signature when
        # available; fall back to a scan otherwise.
        get_by_chunks = getattr(rag.doc_status, "get_docs_by_chunks", None)
        if callable(get_by_chunks):
            result = await get_by_chunks([chunk_id])
            if isinstance(result, dict) and result:
                return next(iter(result.keys()))
    except Exception:
        logger.exception("twin_query: doc lookup failed for chunk %s", chunk_id)
    return None


def build_twin_query_router(get_rag) -> APIRouter:
    """Mount the Twin overlay query endpoint.

    Args:
        get_rag: zero-arg callable returning the captured ``LightRAG``
            instance. Raises a 500 if the host bootstrap didn't capture
            one (same pattern as the native shims).
    """
    router = APIRouter(tags=["twin-query"])

    @router.post("/query", response_model=TwinQueryResponse)
    async def query_endpoint(
        body: TwinQueryBody, request: Request
    ) -> dict[str, Any]:
        del request  # currently unused; kept for future X-Twin-Space scoping
        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        param_kwargs: dict[str, Any] = {
            "mode": body.mode,
            "top_k": body.top_k,
            "only_need_context": body.only_need_context,
            "only_need_prompt": body.only_need_prompt,
        }
        if body.max_total_tokens is not None:
            param_kwargs["max_total_tokens"] = body.max_total_tokens

        # --- 1) Synthesised response ----------------------------------
        try:
            answer = await rag.aquery(body.query, param=QueryParam(**param_kwargs))
        except Exception as exc:
            logger.exception("twin_query: aquery failed")
            raise HTTPException(500, f"Query failed: {exc}") from exc

        if not isinstance(answer, str):
            answer = str(answer)

        # If the operator asked for context-only or prompt-only the
        # answer body already carries everything they wanted — skip
        # the source enrichment to avoid a second retrieval round-trip.
        if body.only_need_context or body.only_need_prompt:
            return {"response": answer, "sources": []}

        # --- 2) Retrieval-anchored sources ----------------------------
        sources: list[dict[str, Any]] = []
        try:
            chunks_vdb = getattr(rag, "chunks_vdb", None)
            if chunks_vdb is None:
                return {"response": answer, "sources": []}
            raw = await chunks_vdb.query(body.query, top_k=body.top_k)
        except Exception:
            logger.exception("twin_query: chunks_vdb.query failed — empty sources")
            return {"response": answer, "sources": []}

        if not isinstance(raw, list):
            raw = []

        total = len(raw)
        for rank, chunk in enumerate(raw[: body.top_k]):
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or ""
            file_path = (
                chunk.get("file_path")
                or chunk.get("source")
                or chunk_id
                or "unknown source"
            )
            doc_id = await _resolve_doc_for_chunk(rag, str(chunk_id))
            sources.append(
                {
                    "n": rank + 1,
                    "type": "file",
                    "name": str(file_path),
                    "meta": _chunk_to_meta(chunk),
                    "score": _safe_get_score(chunk, rank, total),
                    "doc_id": doc_id,
                    "chunk_id": str(chunk_id) or None,
                }
            )

        return {"response": answer, "sources": sources}

    return router


__all__ = ["TwinQueryBody", "TwinQueryResponse", "build_twin_query_router"]
