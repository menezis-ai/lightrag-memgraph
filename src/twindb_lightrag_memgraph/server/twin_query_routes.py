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

import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class TwinQueryBody(BaseModel):
    query: str
    actor: str | None = Field(default=None, max_length=200)
    mode: str = Field(default="mix")
    response_type: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)
    chunk_top_k: int | None = Field(default=None, ge=1, le=200)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    only_need_context: bool = Field(default=False)
    only_need_prompt: bool = Field(default=False)
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    history_turns: int | None = Field(default=None, ge=0, le=20)
    user_prompt: str | None = Field(default=None, max_length=4000)
    enable_rerank: bool | None = Field(default=None)
    tag_filter: dict[str, list[str]] | None = Field(default=None)

    @field_validator("tag_filter")
    @classmethod
    def _validate_tag_filter(
        cls, value: dict[str, list[str]] | None
    ) -> dict[str, list[str]] | None:
        if value is None:
            return None
        allowed_keys = {"all", "any"}
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            raise ValueError(
                "tag_filter keys must be a subset of {'all', 'any'}"
            )
        return value


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


class TwinQueryDataResponse(BaseModel):
    status: str = "success"
    message: str = "Query executed successfully"
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _query_param_kwargs(body: TwinQueryBody, *, stream: bool = False) -> dict[str, Any]:
    param_kwargs: dict[str, Any] = {
        "mode": body.mode,
        "top_k": body.top_k,
        "only_need_context": body.only_need_context,
        "only_need_prompt": body.only_need_prompt,
        "stream": stream,
    }
    if body.response_type is not None:
        param_kwargs["response_type"] = body.response_type
    if body.chunk_top_k is not None:
        param_kwargs["chunk_top_k"] = body.chunk_top_k
    if body.max_entity_tokens is not None:
        param_kwargs["max_entity_tokens"] = body.max_entity_tokens
    if body.max_relation_tokens is not None:
        param_kwargs["max_relation_tokens"] = body.max_relation_tokens
    if body.max_total_tokens is not None:
        param_kwargs["max_total_tokens"] = body.max_total_tokens
    if body.hl_keywords:
        param_kwargs["hl_keywords"] = body.hl_keywords
    if body.ll_keywords:
        param_kwargs["ll_keywords"] = body.ll_keywords
    if body.conversation_history:
        param_kwargs["conversation_history"] = body.conversation_history
    if body.history_turns is not None:
        param_kwargs["history_turns"] = body.history_turns
    if body.user_prompt is not None and body.user_prompt.strip():
        param_kwargs["user_prompt"] = body.user_prompt.strip()
    if body.enable_rerank is not None:
        param_kwargs["enable_rerank"] = body.enable_rerank
    if body.tag_filter is not None:
        param_kwargs["tag_filter"] = body.tag_filter
    return param_kwargs


def _make_query_param(query_param_cls: Any, param_kwargs: dict[str, Any]) -> Any:
    try:
        return query_param_cls(**param_kwargs)
    except TypeError:
        if "tag_filter" not in param_kwargs:
            raise
        # LightRAG versions before the tag-filter constructor field can still
        # carry the runtime attribute for downstream code that understands it.
        fallback_kwargs = dict(param_kwargs)
        tag_filter = fallback_kwargs.pop("tag_filter")
        param = query_param_cls(**fallback_kwargs)
        setattr(param, "tag_filter", tag_filter)
        return param


def _answer_chunk_to_text(chunk: Any) -> str:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in ("response", "content", "text", "delta"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value
        return ""
    return str(chunk)


async def _iter_answer_text(answer: Any) -> AsyncIterator[str]:
    if isinstance(answer, str):
        yield answer
        return
    if hasattr(answer, "__aiter__"):
        async for chunk in answer:
            text = _answer_chunk_to_text(chunk)
            if text:
                yield text
        return
    if isinstance(answer, Iterable) and not isinstance(answer, (bytes, dict)):
        for chunk in answer:
            text = _answer_chunk_to_text(chunk)
            if text:
                yield text
        return
    text = _answer_chunk_to_text(answer)
    if text:
        yield text


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


def _split_source_ids(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return []
    return [
        item.strip()
        for item in raw.replace("<SEP>", ",").split(",")
        if item.strip()
    ]


def _doc_metadata(status: Any) -> dict[str, Any]:
    if status is None:
        return {}
    if isinstance(status, dict):
        metadata = status.get("metadata")
    else:
        metadata = getattr(status, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _tag_filter_terms(
    tag_filter: dict[str, list[str]] | None,
) -> tuple[set[str], set[str]]:
    if not tag_filter:
        return set(), set()
    required = {
        tag.strip().lower()
        for tag in tag_filter.get("all", [])
        if isinstance(tag, str) and tag.strip()
    }
    optional = {
        tag.strip().lower()
        for tag in tag_filter.get("any", [])
        if isinstance(tag, str) and tag.strip()
    }
    return required, optional


def _status_matches_tag_filter(
    status: Any, tag_filter: dict[str, list[str]] | None
) -> bool:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    metadata = _doc_metadata(status)
    tags = {
        str(tag).strip().lower()
        for tag in metadata.get("tags", [])
        if str(tag).strip()
    }
    if required and not required.issubset(tags):
        return False
    if optional and tags.isdisjoint(optional):
        return False
    return True


async def _get_doc_status(rag: Any, doc_id: str) -> Any:
    get_by_id = getattr(getattr(rag, "doc_status", None), "get_by_id", None)
    if callable(get_by_id):
        return await get_by_id(doc_id)
    aget_docs = getattr(rag, "aget_docs_by_ids", None)
    if callable(aget_docs):
        docs = await aget_docs([doc_id])
        if isinstance(docs, dict):
            return docs.get(doc_id)
    return None


async def _doc_ids_for_query_data_row(rag: Any, row: dict[str, Any]) -> set[str]:
    doc_ids = {
        str(row[key])
        for key in ("doc_id", "full_doc_id")
        if isinstance(row.get(key), str) and row.get(key)
    }
    chunk_ids = set()
    for key in ("chunk_id", "id"):
        if isinstance(row.get(key), str) and row.get(key):
            chunk_ids.add(str(row[key]))
    chunk_ids.update(_split_source_ids(row.get("source_id")))
    for chunk_id in chunk_ids:
        doc_id = await _resolve_doc_for_chunk(rag, chunk_id)
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


async def _row_matches_tag_filter(
    rag: Any, row: dict[str, Any], tag_filter: dict[str, list[str]] | None
) -> bool:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    doc_ids = await _doc_ids_for_query_data_row(rag, row)
    if not doc_ids:
        return False
    for doc_id in doc_ids:
        status = await _get_doc_status(rag, doc_id)
        if _status_matches_tag_filter(status, tag_filter):
            return True
    return False


async def _filter_query_data_by_tags(
    rag: Any,
    response: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
) -> dict[str, Any]:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return response

    data = response.get("data")
    if not isinstance(data, dict):
        return response

    filtered_data = dict(data)
    kept_reference_ids: set[str] = set()
    for key in ("chunks", "entities", "relationships"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        kept_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if await _row_matches_tag_filter(rag, row, tag_filter):
                kept_rows.append(row)
                ref_id = row.get("reference_id")
                if isinstance(ref_id, str) and ref_id:
                    kept_reference_ids.add(ref_id)
        filtered_data[key] = kept_rows

    references = data.get("references")
    if isinstance(references, list):
        filtered_data["references"] = [
            ref
            for ref in references
            if not isinstance(ref, dict)
            or ref.get("reference_id") in kept_reference_ids
        ]

    filtered = dict(response)
    filtered["data"] = filtered_data
    metadata = dict(response.get("metadata") or {})
    metadata["tag_filter"] = tag_filter
    filtered["metadata"] = metadata
    return filtered


async def _build_sources(
    rag: Any, query: str, top_k: int
) -> list[dict[str, Any]]:
    """Query chunks_vdb + DocStatus to build the WebUI RetrievalSource list.

    Shared between the non-stream `/query` endpoint and the NDJSON
    `/query/stream` endpoint so both routes return the same shape. On
    any failure returns ``[]`` — the caller still has the answer text
    and the sources panel just stays empty in the UI.
    """
    try:
        chunks_vdb = getattr(rag, "chunks_vdb", None)
        if chunks_vdb is None:
            return []
        raw = await chunks_vdb.query(query, top_k=top_k)
    except Exception:
        logger.exception("twin_query: chunks_vdb.query failed — empty sources")
        return []

    if not isinstance(raw, list):
        raw = []

    sources: list[dict[str, Any]] = []
    total = len(raw)
    for rank, chunk in enumerate(raw[:top_k]):
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
    return sources


def _actor_from_request(body: TwinQueryBody, request: Request) -> str:
    if body.actor and body.actor.strip():
        return body.actor.strip()
    for header in (
        "x-auth-request-email",
        "x-forwarded-user",
        "x-auth-request-user",
    ):
        value = request.headers.get(header)
        if value and value.strip():
            return value.strip()
    return "system"


async def _record_retrieval_activity(
    body: TwinQueryBody,
    request: Request,
    *,
    sources_count: int,
    stream: bool,
) -> None:
    """Best-effort Activity write for completed retrieval calls."""
    try:
        from .webui_router import _make_event, get_store

        actor = _actor_from_request(body, request)
        event = _make_event(
            kind="retrieval",
            sev="info",
            actor=actor,
            target_label=body.query[:120],
            summary=f"retrieval completed ({body.mode})",
            meta={
                "query": body.query,
                "mode": body.mode,
                "top_k": body.top_k,
                "sources_count": sources_count,
                "stream": stream,
                "tag_filter": body.tag_filter,
            },
            target_type="query",
        )
        await get_store().record_activity(event)
    except Exception:
        logger.exception("twin_query: failed to record retrieval activity")


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
        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        param_kwargs = _query_param_kwargs(body)
        param = _make_query_param(QueryParam, param_kwargs)

        # --- 1) Synthesised response ----------------------------------
        try:
            answer = await rag.aquery(body.query, param=param)
        except Exception as exc:
            logger.exception("twin_query: aquery failed")
            raise HTTPException(500, f"Query failed: {exc}") from exc

        if not isinstance(answer, str):
            answer = str(answer)

        # If the operator asked for context-only or prompt-only the
        # answer body already carries everything they wanted — skip
        # the source enrichment to avoid a second retrieval round-trip.
        if body.only_need_context or body.only_need_prompt:
            await _record_retrieval_activity(
                body, request, sources_count=0, stream=False
            )
            return {"response": answer, "sources": []}

        sources = await _build_sources(rag, body.query, body.top_k)
        await _record_retrieval_activity(
            body, request, sources_count=len(sources), stream=False
        )
        return {"response": answer, "sources": sources}

    @router.post("/query/data", response_model=TwinQueryDataResponse)
    async def query_data_endpoint(
        body: TwinQueryBody, request: Request
    ) -> dict[str, Any]:
        """Return structured LightRAG retrieval data through the Twin prefix.

        This mirrors LightRAG's native `/query/data` endpoint while keeping
        the Twin contract (`/twin/api/*`, folder header, tag_filter) on the
        same surface as `/query` and `/query/stream`.
        """
        del request
        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        param = _make_query_param(QueryParam, _query_param_kwargs(body))
        try:
            result = await rag.aquery_data(body.query, param=param)
        except Exception as exc:
            logger.exception("twin_query: aquery_data failed")
            raise HTTPException(500, f"Query data failed: {exc}") from exc

        if not isinstance(result, dict):
            return {
                "status": "failure",
                "message": "Invalid response type",
                "data": {},
                "metadata": {},
            }

        result = await _filter_query_data_by_tags(rag, result, body.tag_filter)
        return {
            "status": result.get("status", "success"),
            "message": result.get("message", "Query executed successfully"),
            "data": (
                result.get("data") if isinstance(result.get("data"), dict) else {}
            ),
            "metadata": (
                result.get("metadata")
                if isinstance(result.get("metadata"), dict)
                else {}
            ),
        }

    @router.post("/query/stream")
    async def query_stream_endpoint(
        body: TwinQueryBody, request: Request
    ) -> StreamingResponse:
        """Stream the LightRAG answer as NDJSON and emit a final sources event.

        Wire format (one JSON object per line):
          {"type":"token","value":"<chunk text>"}
          ... repeated for every LLM chunk ...
          {"type":"sources","value":[<RetrievalSource>, ...]}

        Client buffers tokens, calls onChunk for streaming UI, and uses
        the final sources event to render the structured sources panel.
        Strip of the `### References` / `### Références` block is the
        client's responsibility on the joined token stream (the
        per-chunk boundary can land inside the heading itself, so a
        server-side strip would require buffering and defeat streaming).

        Error contract (post-stream-open): once the response has
        started, an HTTP status flip is no longer possible — the
        client has already committed to a 200 reader loop. Failures
        from ``aquery`` are therefore surfaced as a final ``token``
        event carrying ``"[query failed: <exc>]"`` followed by an
        empty ``sources`` event. Callers MUST treat token events as
        possibly-error-bearing and render the text verbatim; the
        absence of a non-empty sources payload is the only signal
        that the run did not complete cleanly. Pre-stream failures
        (RAG bootstrap, body validation) still surface as real HTTP
        4xx/5xx like the non-stream `/query` route.
        """
        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        async def generate() -> AsyncIterator[str]:
            try:
                param = _make_query_param(
                    QueryParam, _query_param_kwargs(body, stream=True)
                )
                answer = await rag.aquery(body.query, param=param)
                async for text in _iter_answer_text(answer):
                    yield json.dumps({"type": "token", "value": text}) + "\n"
            except Exception as exc:
                logger.exception("twin_query: streaming aquery failed")
                yield json.dumps(
                    {"type": "token", "value": f"\n[query failed: {exc}]"}
                ) + "\n"
                yield json.dumps({"type": "sources", "value": []}) + "\n"
                return

            if body.only_need_context or body.only_need_prompt:
                await _record_retrieval_activity(
                    body, request, sources_count=0, stream=True
                )
                yield json.dumps({"type": "sources", "value": []}) + "\n"
                return

            sources = await _build_sources(rag, body.query, body.top_k)
            await _record_retrieval_activity(
                body, request, sources_count=len(sources), stream=True
            )
            yield json.dumps({"type": "sources", "value": sources}) + "\n"

        return StreamingResponse(
            generate(), media_type="application/x-ndjson"
        )

    return router


__all__ = [
    "TwinQueryBody",
    "TwinQueryDataResponse",
    "TwinQueryResponse",
    "build_twin_query_router",
]
