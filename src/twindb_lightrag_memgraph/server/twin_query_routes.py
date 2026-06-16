"""Twin overlay POST /twin/api/query — structured retrieval response.

LightRAG's native ``POST /query`` returns just ``{"response": str}``
which gives the React port no way to render clickable citations or a
sources panel. This wrapper consumes ``LightRAG.aquery_llm`` (single
call, answer + structured retrieval in one pass) and projects the
envelope into the Twin contract.

Wire shape returned by ``/query`` and ``/query/stream``:
``{response, sources, answer_status}`` where each source carries
``n, type, name, meta, score, doc_id?, chunk_id?``.

Doctrine (TR-RET-02 step 2 / audit C3):
- ``sources`` are projected from ``aquery_llm``'s ``data.references``
  — the chunks LightRAG actually used to ground the answer. We never
  re-issue a separate vector retrieval against ``chunks_vdb`` on the
  nominal path; that was the structural lie this module used to ship.
- ``n`` mirrors LightRAG's ``reference_id`` so the React port's
  ``parseAnswer`` (``[N]`` citation parser in
  ``lightrag_webui_twin/src/types/retrieval.ts``) stays aligned with
  the sources list. The mapping is intentionally non-deduplicating.
- ``answer_status`` is set from the envelope: ``failure_reason ==
  "no_results"`` → ``insufficient_information``; defense-in-depth via
  the ``[no-context]`` marker in the response content.
- Generic backend failures (failure with another reason, or an
  exception inside aquery_llm) surface as real HTTP 500 on
  ``/query`` and an ``[query failed: …]`` token + grounded status on
  ``/query/stream`` (the HTTP status cannot flip mid-stream). They
  are NEVER masked as ``insufficient_information``.
- The legacy ``aquery() + chunks_vdb`` path lives on as
  :func:`_build_sources_legacy_fallback`, kept for compat tests in
  isolation. It MUST NOT be invoked from the nominal route paths.

The ``only_need_context`` / ``only_need_prompt`` modes still use the
legacy ``aquery()`` because aquery_llm has no special-casing for
those — the operator gets the requested body and an empty sources
list (those modes never claimed grounded sources to begin with).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ._lightrag_compat import (
    ANSWER_STATUS_GROUNDED,
    ANSWER_STATUS_INSUFFICIENT,
    AnswerMarkerStripper,
    AnswerStatus,
    GraphAnswerEnvelopeError,
    build_sources_from_raw_data,
    classify_answer,
    classify_aquery_llm_result,
    collect_chunk_ids,
    is_streaming_envelope,
)

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
    # TR-RET-02: ``"insufficient_information"`` when LightRAG signalled
    # no usable retrieval context (canonical ``[no-context]`` marker in
    # the fail response). The React port uses this to suppress the
    # Sources panel honestly rather than parsing the LLM prose.
    # Typed as ``AnswerStatus`` so the generated OpenAPI schema
    # advertises the enum to clients/tooling instead of an open str.
    answer_status: AnswerStatus = Field(default=ANSWER_STATUS_GROUNDED)


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


def _query_param_ctor_fields(query_param_cls: Any) -> set[str] | None:
    """Constructor-accepted field names for the installed ``QueryParam``.

    Returns ``None`` when the fields cannot be introspected (non-dataclass),
    in which case callers fall back to passing every kwarg through.
    """
    try:
        return {f.name for f in dataclasses.fields(query_param_cls)}
    except TypeError:
        return None


def _make_query_param(query_param_cls: Any, param_kwargs: dict[str, Any]) -> Any:
    """Build a ``QueryParam`` that is resilient to upstream field churn.

    LightRAG renames/removes ``QueryParam`` fields between minor releases
    (e.g. ``history_turns`` was dropped in 1.5, and ``tag_filter`` is a Twin
    extension never present upstream). Passing such a kwarg straight to the
    constructor raises ``TypeError`` and 500s the whole query endpoint. We
    instead route only constructor-known kwargs through ``__init__`` and apply
    the rest as runtime attributes, so downstream code that understands them
    still sees them and the request never crashes.
    """
    fields = _query_param_ctor_fields(query_param_cls)
    if fields is None:
        return query_param_cls(**param_kwargs)

    ctor_kwargs = {k: v for k, v in param_kwargs.items() if k in fields}
    extra_kwargs = {k: v for k, v in param_kwargs.items() if k not in fields}
    param = query_param_cls(**ctor_kwargs)
    for key, value in extra_kwargs.items():
        setattr(param, key, value)
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


async def _resolve_chunk_to_doc_id(
    rag: Any, chunk_ids: list[str]
) -> dict[str, str]:
    """Batch chunk_id -> doc_id resolution for the aquery_llm path.

    LightRAG's ``get_docs_by_chunks`` signature returns
    ``{doc_id: status}`` keyed by doc, so we resolve each chunk
    individually and run the per-chunk lookups concurrently via
    ``asyncio.gather`` — N chunks resolve in roughly one round-trip
    instead of N. Failures degrade silently to "no doc_id"; the
    source still renders, just without the drill-down doc id.
    """
    if not chunk_ids:
        return {}
    import asyncio

    # De-dup so we don't ask the same chunk twice.
    unique = list(dict.fromkeys(chunk_ids))
    resolved = await asyncio.gather(
        *(_resolve_doc_for_chunk(rag, chunk_id) for chunk_id in unique),
        return_exceptions=False,
    )
    out: dict[str, str] = {}
    for chunk_id, doc_id in zip(unique, resolved):
        if doc_id:
            out[chunk_id] = doc_id
    return out


def _split_source_ids(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return []
    return [
        item.strip()
        for item in raw.replace("<SEP>", ",").split(",")
        if item.strip()
    ]


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


def _doc_tags_match_filter(
    doc_tags: set[str], tag_filter: dict[str, list[str]] | None
) -> bool:
    """Audit C2: doc tags come from the ``TAGGED_WITH`` graph relation,
    never from ``DocStatus.metadata.tags`` (which can lag the WebUI
    retag flow and produces a misleading filter result).
    """
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    if required and not required.issubset(doc_tags):
        return False
    if optional and doc_tags.isdisjoint(optional):
        return False
    return True


async def _fetch_doc_graph_tags(doc_id: str, folder: str) -> set[str]:
    """Read a document's tags via the canonical ``TAGGED_WITH`` edge.

    Audit C2 / fix: ``DocStatus.metadata.tags`` is not the source of
    truth — the WebUI retag flow MERGE-creates a
    ``(:DocStatus_{workspace})-[:TAGGED_WITH]->(:WebuiTag_{folder})``
    edge and does not touch the legacy property. Reading the property
    would silently disagree with the rest of the Twin overlay.

    ``folder`` is passed in explicitly by the route handler — no
    implicit ``current_folder_id()`` dependency in this low-level
    helper (Codex review on PR fix/query-data-tag-filter-graph).

    On any Memgraph failure the function returns ``set()`` and logs:
    with an active filter that conservatively rejects rather than
    fabricating a match, which is the honest degradation here.
    """
    if not doc_id or not folder:
        return set()
    try:
        from .. import _pool
        from .._constants import resolve_workspace

        workspace = resolve_workspace()
        doc_label = f"DocStatus_{workspace}"
        tag_label = f"WebuiTag_{folder}"
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (d:`{doc_label}` {{id: $doc_id}})
                OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:`{tag_label}`)
                RETURN collect(t.id) AS tags
                """,
                doc_id=doc_id,
            )
            record = await result.single()
            await result.consume()
        raw_tags = (record or {}).get("tags") or []
    except Exception:
        logger.exception(
            "twin_query: TAGGED_WITH lookup failed for doc=%s folder=%s",
            doc_id,
            folder,
        )
        return set()
    return {
        str(tag).strip().lower()
        for tag in raw_tags
        if isinstance(tag, str) and tag.strip()
    }


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
    rag: Any,
    row: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
    tags_cache: dict[str, set[str]],
) -> bool:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    doc_ids = await _doc_ids_for_query_data_row(rag, row)
    if not doc_ids:
        return False
    for doc_id in doc_ids:
        # Per-request cache: chunks/references rows often repeat the
        # same doc_id; one Cypher round-trip per unique doc suffices.
        if doc_id not in tags_cache:
            tags_cache[doc_id] = await _fetch_doc_graph_tags(doc_id, folder)
        if _doc_tags_match_filter(tags_cache[doc_id], tag_filter):
            return True
    return False


async def _filter_query_data_by_tags(
    rag: Any,
    response: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
) -> dict[str, Any]:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return response

    data = response.get("data")
    if not isinstance(data, dict):
        return response

    # Cache shared across all rows in this single request — bounded by
    # the number of unique doc_ids in the result set.
    tags_cache: dict[str, set[str]] = {}

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
            if await _row_matches_tag_filter(
                rag, row, tag_filter, folder, tags_cache
            ):
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


async def _build_sources_legacy_fallback(
    rag: Any, query: str, top_k: int
) -> list[dict[str, Any]]:
    """LEGACY: separate vector pass to assemble a sources list.

    DEPRECATED on the nominal /query and /stream paths since TR-RET-02
    step 2 / audit C3. Kept ONLY as a compat reference for tests in
    isolation; it MUST NOT be invoked from a successful aquery_llm
    response path because that reintroduces the structural lie this
    PR is closing (the displayed sources used to be the result of a
    second retrieval, not the chunks LightRAG actually grounded on).

    The nominal source-of-truth now lives in
    :func:`server._lightrag_compat.build_sources_from_raw_data` which
    maps ``data.references`` from the aquery_llm envelope.
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
        # TR-RET-02 step 3 / audit C1: ``tag_filter`` is NOT applied
        # to retrieval by LightRAG 1.4.x — its ``QueryParam`` has no
        # such field and the previous setattr-on-param fallback was
        # silently ignored downstream. Accepting the parameter here
        # used to lie to the operator about scoping the retrieval by
        # tags; reject it loudly instead.
        if body.tag_filter is not None:
            raise HTTPException(
                422,
                "tag_filter is not applied to retrieval by LightRAG 1.4.x; "
                "remove it from /query and /query/stream requests.",
            )

        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        param_kwargs = _query_param_kwargs(body)
        param = _make_query_param(QueryParam, param_kwargs)

        # only_need_context / only_need_prompt skip the LLM entirely,
        # so aquery_llm is overkill. Keep the legacy aquery() path here
        # — the operator gets the context/prompt body they asked for and
        # the sources panel stays empty (this branch never claimed
        # grounded sources to begin with).
        if body.only_need_context or body.only_need_prompt:
            try:
                answer_raw = await rag.aquery(body.query, param=param)
            except Exception as exc:
                logger.exception("twin_query: aquery failed")
                raise HTTPException(500, f"Query failed: {exc}") from exc
            answer_text = (
                answer_raw if isinstance(answer_raw, str) else str(answer_raw or "")
            )
            cleaned, _ = classify_answer(answer_text)
            await _record_retrieval_activity(
                body, request, sources_count=0, stream=False
            )
            return {
                "response": cleaned,
                "sources": [],
                "answer_status": ANSWER_STATUS_GROUNDED,
            }

        # --- Nominal path: aquery_llm gives us answer + grounding
        #     context in a single call. The sources panel is built
        #     from data.references — the chunks LightRAG actually
        #     used. No second vector retrieval (TR-RET-02 step 2 /
        #     audit C3). chunks_vdb is NEVER touched here.
        try:
            envelope = await rag.aquery_llm(body.query, param=param)
        except Exception as exc:
            logger.exception("twin_query: aquery_llm failed")
            raise HTTPException(500, f"Query failed: {exc}") from exc

        try:
            clean_answer, answer_status = classify_aquery_llm_result(envelope)
        except GraphAnswerEnvelopeError as exc:
            # Hard backend failure (status=failure, reason != no_results).
            # Surface as a real 500 — do NOT mask as insufficient
            # information, that would hide the issue from the operator.
            logger.error("twin_query: aquery_llm envelope failure: %s", exc)
            raise HTTPException(
                500, f"Query failed: {exc}"
            ) from exc

        if answer_status == ANSWER_STATUS_INSUFFICIENT:
            await _record_retrieval_activity(
                body, request, sources_count=0, stream=False
            )
            return {
                "response": clean_answer,
                "sources": [],
                "answer_status": answer_status,
            }

        # Build sources from the envelope. Unexpected shape → log +
        # empty sources. We DO NOT fall back to a second vector pass:
        # that path was the structural lie this PR is closing.
        chunk_ids = collect_chunk_ids(envelope)
        chunk_to_doc = await _resolve_chunk_to_doc_id(rag, chunk_ids)
        try:
            sources = build_sources_from_raw_data(envelope, chunk_to_doc)
        except GraphAnswerEnvelopeError as exc:
            logger.warning(
                "twin_query: aquery_llm references unprojectable, "
                "surfacing empty sources rather than reconstructing them "
                "from a second vector pass: %s",
                exc,
            )
            sources = []
        await _record_retrieval_activity(
            body, request, sources_count=len(sources), stream=False
        )
        return {
            "response": clean_answer,
            "sources": sources,
            "answer_status": answer_status,
        }

    @router.post("/query/data", response_model=TwinQueryDataResponse)
    async def query_data_endpoint(
        body: TwinQueryBody, request: Request
    ) -> dict[str, Any]:
        """Return structured LightRAG retrieval data through the Twin prefix.

        This mirrors LightRAG's native `/query/data` endpoint while keeping
        the Twin contract (`/twin/api/*`, folder header, tag_filter) on the
        same surface as `/query` and `/query/stream`.
        """
        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        # Audit C2 / fix: resolve the active folder once at the route
        # boundary and thread it explicitly through the tag-filter
        # helpers. The low-level helpers refuse to import the folder
        # ContextVar themselves — passing it down keeps the chain
        # testable without monkeypatching globals.
        from .folder import resolve_folder_for_request

        folder = resolve_folder_for_request(request)

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

        result = await _filter_query_data_by_tags(
            rag, result, body.tag_filter, folder
        )
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
          {"type":"status","value":"grounded"|"insufficient_information"}
          {"type":"sources","value":[<RetrievalSource>, ...]}

        The ``status`` event lands exactly once, before the final
        ``sources`` event, so the client can branch its rendering
        deterministically (TR-RET-02 step 1). ``sources`` are
        projected from ``aquery_llm``'s ``data.references`` — the
        chunks LightRAG actually used to ground the answer (step 2 /
        audit C3); we never re-issue a second ``chunks_vdb`` retrieval
        on this path.

        Client buffers tokens, calls onChunk for streaming UI, and uses
        the final ``sources`` event to render the structured panel.
        Strip of the ``### References`` / ``### Références`` block is
        the client's responsibility on the joined token stream (the
        per-chunk boundary can land inside the heading itself, so a
        server-side strip would require buffering and defeat streaming).

        Error contract (post-stream-open): once the response has
        started, an HTTP status flip is no longer possible — the
        client has already committed to a 200 reader loop.
        ``aquery_llm`` exceptions and structured backend failures
        (``status=failure`` for any reason other than ``no_results``)
        are therefore surfaced as a final ``token`` event carrying
        ``"[query failed: <reason>]"`` followed by ``status=grounded``
        and an empty ``sources`` event. Callers MUST treat token
        events as possibly-error-bearing and render the text verbatim;
        the absence of a non-empty sources payload is the only signal
        that the run did not complete cleanly. ``no_results`` is the
        only failure_reason mapped to ``insufficient_information`` —
        the rest must NOT be masked as such. Pre-stream failures
        (RAG bootstrap, body validation including the audit-C1
        ``tag_filter`` 422) still surface as real HTTP 4xx/5xx like
        the non-stream `/query` route.
        """
        # TR-RET-02 step 3 / audit C1: same honest rejection as /query
        # — LightRAG 1.4.x does not apply tag_filter to retrieval.
        if body.tag_filter is not None:
            raise HTTPException(
                422,
                "tag_filter is not applied to retrieval by LightRAG 1.4.x; "
                "remove it from /query and /query/stream requests.",
            )

        try:
            rag = get_rag()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

        from lightrag.base import QueryParam

        async def generate() -> AsyncIterator[str]:
            # TR-RET-02 step 2 / audit C3:
            # - aquery_llm streams tokens via ``llm_response.response_iterator``
            #   AND returns the structured ``data.references`` we need
            #   for the sources event — single call, real grounding context.
            # - chunks_vdb is NOT touched on this path. The sources panel
            #   only ever shows the chunks LightRAG actually used.
            # - The marker stripper runs on the streamed text so
            #   ``[no-context]`` never leaks to the operator even when
            #   it straddles a chunk boundary.
            stripper = AnswerMarkerStripper()
            envelope: dict[str, Any] | None = None
            try:
                param = _make_query_param(
                    QueryParam, _query_param_kwargs(body, stream=True)
                )
                envelope = await rag.aquery_llm(body.query, param=param)
                llm_response = (
                    envelope.get("llm_response")
                    if isinstance(envelope, dict)
                    else None
                ) or {}
                token_source: Any
                if is_streaming_envelope(envelope):
                    token_source = llm_response.get("response_iterator")
                else:
                    # aquery_llm produced the answer synchronously
                    # (failure path, bypass mode, or non-streaming
                    # backend). Treat content as a single-shot token.
                    token_source = llm_response.get("content") or ""
                async for text in _iter_answer_text(token_source):
                    for safe in stripper.feed(text):
                        if safe:
                            yield json.dumps(
                                {"type": "token", "value": safe}
                            ) + "\n"
                for safe in stripper.flush():
                    if safe:
                        yield json.dumps(
                            {"type": "token", "value": safe}
                        ) + "\n"
            except Exception as exc:
                logger.exception("twin_query: streaming aquery_llm failed")
                yield json.dumps(
                    {"type": "token", "value": f"\n[query failed: {exc}]"}
                ) + "\n"
                yield json.dumps(
                    {"type": "status", "value": ANSWER_STATUS_GROUNDED}
                ) + "\n"
                yield json.dumps({"type": "sources", "value": []}) + "\n"
                return

            # Determine answer_status: prefer the envelope's structured
            # failure_reason (set by LightRAG itself); fall back to the
            # marker (which we already detected via the stripper) for
            # defense in depth.
            envelope_status: AnswerStatus = ANSWER_STATUS_GROUNDED
            envelope_fatal_reason: str | None = None
            if isinstance(envelope, dict):
                metadata = envelope.get("metadata") or {}
                failure_reason = (
                    metadata.get("failure_reason")
                    if isinstance(metadata, dict)
                    else None
                )
                if envelope.get("status") == "failure":
                    if failure_reason == "no_results":
                        envelope_status = ANSWER_STATUS_INSUFFICIENT
                    else:
                        # Generic backend failure mid-stream. We can't
                        # flip the HTTP status now (200 already sent),
                        # but we MUST NOT mask it: emit an explicit
                        # ``[query failed: …]`` token so the operator
                        # sees the error rather than an empty grounded
                        # response (matches the ``except`` branch above).
                        envelope_fatal_reason = (
                            failure_reason
                            or str(envelope.get("message") or "backend failure")
                        )
            if stripper.detected and envelope_fatal_reason is None:
                envelope_status = ANSWER_STATUS_INSUFFICIENT

            if envelope_fatal_reason is not None:
                logger.error(
                    "twin_query stream: aquery_llm envelope failure "
                    "surfaced as in-stream error token: %s",
                    envelope_fatal_reason,
                )
                yield json.dumps(
                    {
                        "type": "token",
                        "value": f"\n[query failed: {envelope_fatal_reason}]",
                    }
                ) + "\n"
                yield json.dumps(
                    {"type": "status", "value": ANSWER_STATUS_GROUNDED}
                ) + "\n"
                await _record_retrieval_activity(
                    body, request, sources_count=0, stream=True
                )
                yield json.dumps({"type": "sources", "value": []}) + "\n"
                return

            yield json.dumps(
                {"type": "status", "value": envelope_status}
            ) + "\n"

            if (
                body.only_need_context
                or body.only_need_prompt
                or envelope_status == ANSWER_STATUS_INSUFFICIENT
            ):
                await _record_retrieval_activity(
                    body, request, sources_count=0, stream=True
                )
                yield json.dumps({"type": "sources", "value": []}) + "\n"
                return

            chunk_ids = collect_chunk_ids(envelope or {})
            chunk_to_doc = await _resolve_chunk_to_doc_id(rag, chunk_ids)
            try:
                sources = build_sources_from_raw_data(
                    envelope or {}, chunk_to_doc
                )
            except GraphAnswerEnvelopeError as exc:
                # Honest degradation: empty sources rather than a
                # silent second-pass reconstruction.
                logger.warning(
                    "twin_query stream: aquery_llm references "
                    "unprojectable, emitting empty sources: %s",
                    exc,
                )
                sources = []
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
