"""Twin overlay POST /twin/api/query — structured retrieval response.

LightRAG's native ``POST /query`` returns just ``{"response": str}``
which gives the React port no way to render clickable citations or a
sources panel. This wrapper consumes ``LightRAG.aquery_llm`` (single
call, answer + structured retrieval in one pass) and projects the
envelope into the Twin contract.

Wire shape returned by ``/query`` and ``/query/stream``:
``{response, sources, answer_status, model?}`` where each source carries
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
  ``/query`` and an ``[query failed: …]`` token + ``query_failed``
  status on ``/query/stream`` (the HTTP status cannot flip
  mid-stream, so the failure is carried in the status event rather
  than masked as ``grounded``). They are NEVER masked as
  ``insufficient_information``.
- The legacy ``aquery() + chunks_vdb`` path lives on as
  :func:`_build_sources_legacy_fallback`, kept for compat tests in
  isolation. It MUST NOT be invoked from the nominal route paths.

The ``only_need_context`` mode still uses legacy ``aquery()`` because
``aquery_llm`` returns a bare context string for that flag rather than
the structured envelope this module projects. It reports
``answer_status = no_retrieval``. The external model rejects
``only_need_prompt`` (privileged prompt disclosure), raw
``user_prompt`` overrides, and ``bypass`` (ungrounded direct LLM).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from .._lightrag_compat import (
    ANSWER_STATUS_INSUFFICIENT,
    ANSWER_STATUS_NO_RETRIEVAL,
    ANSWER_STATUS_SOURCE_PROJECTION_FAILED,
    AnswerMarkerStripper,
    GraphAnswerEnvelopeError,
    classify_answer,
    classify_aquery_llm_result,
)
from .activity import _record_retrieval_activity
from .models import TwinQueryBody, TwinQueryDataResponse, TwinQueryResponse
from .paragraph_anchor import CitationEvidenceCollector, collect_citation_evidence
from .params import _make_query_param, _query_param_kwargs
from .query_data import (
    _annotate_query_data_chunk_scores,
    _enrich_query_data_chunks_from_source_ids,
)
from .query_data_filters import (
    _filter_query_data_by_tags as _filter_query_data_by_tags_impl,
)
from .request_scope import (
    _annotate_query_data_fallback,
    _has_advanced_filter,
    _is_no_retrieval_mode,
    _query_data_failure_reason,
    _query_data_fallback_mode,
    _retrieval_scope,
)
from .response_sources import (
    _build_envelope_sources as _build_envelope_sources_impl,
    _build_sources_legacy_fallback as _build_sources_legacy_fallback_impl,
    _enrich_sources_doc_ids_from_file_path as _enrich_sources_doc_ids_from_file_path_impl,
    _filter_sources_by_advanced_filters as _filter_sources_by_advanced_filters_impl,
    _public_sources,
    _source_matches_tag_filter as _source_matches_tag_filter_impl,
)
from .source_filters import TagFilter, _source_matches_doc_filter
from .streaming import (
    _determine_stream_status,
    _emit_answer_tokens,
    _query_stream_empty_sources_events,
    _query_stream_failure_events,
    _query_stream_fatal_events,
    _query_stream_grounded_events,
)

logger = logging.getLogger(__name__)


class ClientDisconnectedDuringQuery(Exception):
    """Raised when the HTTP client drops a query while retrieval is running."""


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def _await_query_or_disconnect(awaitable, request: Request):
    """Await a costly query call while polling for client disconnect.

    FastAPI does not automatically cancel an in-flight Python coroutine when
    the browser aborts a fetch. Retrieval can still be expensive for a very
    narrow filter (the ANN window may need to cover most global neighbours); if
    the operator abandons the thread, stop waiting and cancel the underlying
    task instead of letting it run to a late response.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if task in done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise ClientDisconnectedDuringQuery
    except Exception:
        if not task.done():
            task.cancel()
        raise


async def _enrich_sources_doc_ids_from_file_path(
    rag: Any,
    sources: list[dict[str, Any]],
) -> None:
    await _enrich_sources_doc_ids_from_file_path_impl(rag, sources)


async def _source_matches_tag_filter(
    source: dict[str, Any],
    tag_filter: TagFilter | None,
    folder: str,
    tags_cache: dict[str, set[str]],
) -> bool:
    return await _source_matches_tag_filter_impl(
        source,
        tag_filter,
        folder,
        tags_cache,
        _fetch_doc_graph_tags,
    )


async def _filter_sources_by_advanced_filters(
    sources: list[dict[str, Any]],
    *,
    tag_filter: TagFilter | None,
    doc_filter: dict[str, list[str]] | None,
    folder: str,
) -> tuple[list[dict[str, Any]], bool]:
    return await _filter_sources_by_advanced_filters_impl(
        sources,
        tag_filter=tag_filter,
        doc_filter=doc_filter,
        folder=folder,
        fetch_doc_tags=_fetch_doc_graph_tags,
        fetch_doc_tags_batch=_active_tag_batch_fetcher(),
    )


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
        from ... import _pool
        from ..._constants import resolve_workspace

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


_DEFAULT_FETCH_DOC_GRAPH_TAGS = _fetch_doc_graph_tags


async def _fetch_doc_graph_tags_batch(
    doc_ids: list[str], folder: str
) -> dict[str, set[str]]:
    """Read canonical tags for many documents in one set-based query."""
    unique = list(dict.fromkeys(doc_id for doc_id in doc_ids if doc_id))
    tags_by_doc = {doc_id: set() for doc_id in unique}
    if not unique or not folder:
        return tags_by_doc

    try:
        from ... import _pool
        from ..._constants import resolve_workspace

        workspace = resolve_workspace()
        doc_label = f"DocStatus_{workspace}"
        tag_label = f"WebuiTag_{folder}"
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $doc_ids AS doc_id
                MATCH (d:`{doc_label}` {{id: doc_id}})
                OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:`{tag_label}`)
                RETURN doc_id, collect(t.id) AS tags
                """,
                doc_ids=unique,
            )
            async for record in result:
                tags_by_doc[record["doc_id"]] = {
                    str(tag).strip().lower()
                    for tag in (record["tags"] or [])
                    if isinstance(tag, str) and tag.strip()
                }
            await result.consume()
    except Exception:
        logger.exception(
            "twin_query: batch TAGGED_WITH lookup failed for %d docs folder=%s",
            len(unique),
            folder,
        )
    return tags_by_doc


def _active_tag_batch_fetcher():
    """Preserve custom/monkeypatched per-doc fetch semantics."""
    if _fetch_doc_graph_tags is not _DEFAULT_FETCH_DOC_GRAPH_TAGS:
        return None
    return _fetch_doc_graph_tags_batch


async def _filter_query_data_by_tags(
    rag: Any,
    response: dict[str, Any],
    tag_filter: TagFilter | None,
    folder: str,
) -> dict[str, Any]:
    return await _filter_query_data_by_tags_impl(
        rag,
        response,
        tag_filter,
        folder,
        _fetch_doc_graph_tags,
        _active_tag_batch_fetcher(),
    )


async def _build_sources_legacy_fallback(
    rag: Any, query: str, top_k: int
) -> list[dict[str, Any]]:
    return await _build_sources_legacy_fallback_impl(rag, query, top_k)


async def _twin_query(get_rag, body: TwinQueryBody, request: Request) -> dict[str, Any]:
    """Body of ``POST /twin/api/query`` (non-streaming, answer + sources)."""
    try:
        rag = get_rag()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    from lightrag.base import QueryParam
    from ..folder import resolve_folder_for_request

    folder = resolve_folder_for_request(request)
    param_kwargs = _query_param_kwargs(body)
    param = _make_query_param(QueryParam, param_kwargs)

    # QA RET-V5-001: per-phase wall-clock, surfaced in the retrieval Activity
    # event (`meta.timings`, ms) and the server log so a latency report can be
    # characterized from the ledger (volumetry? filter shape? backend stall?)
    # instead of re-reproduced blind.
    route_started = time.perf_counter()

    if body.only_need_context:
        try:
            # Folder-scoped retrieval: the storage layer constrains candidate
            # chunks/entities/relations to docs MEMBER_OF the active folder, so
            # no cross-folder context can enter the returned body.
            with _retrieval_scope(folder, body):
                answer_raw = await rag.aquery(body.query, param=param)
        except Exception as exc:
            logger.exception("twin_query: aquery failed")
            raise HTTPException(500, f"Query failed: {exc}") from exc
        answer_text = (
            answer_raw if isinstance(answer_raw, str) else str(answer_raw or "")
        )
        cleaned, _ = classify_answer(answer_text)
        await _record_retrieval_activity(
            body,
            request,
            folder=folder,
            sources_count=0,
            stream=False,
            timings={"total_ms": _elapsed_ms(route_started)},
        )
        return {
            "response": cleaned,
            "sources": [],
            "answer_status": ANSWER_STATUS_NO_RETRIEVAL,
        }

    # --- Nominal path: aquery_llm gives answer + grounding context in a single
    #     call. The sources panel is built from data.references — the chunks
    #     LightRAG actually used. No second vector retrieval (audit C3).
    retrieval_started = time.perf_counter()
    try:
        # Folder-scoped grounding: every vector retrieval LightRAG issues inside
        # aquery_llm (chunks + entity/relation vdb) is filtered to the active
        # folder's membership at the storage layer (batch-2 cloisonnement).
        with _retrieval_scope(folder, body) as retrieval_scores:
            envelope = await _await_query_or_disconnect(
                rag.aquery_llm(body.query, param=param),
                request,
            )
    except ClientDisconnectedDuringQuery as exc:
        logger.info("twin_query: client disconnected while aquery_llm was running")
        raise HTTPException(499, "Client closed request") from exc
    except Exception as exc:
        logger.exception("twin_query: aquery_llm failed")
        raise HTTPException(500, f"Query failed: {exc}") from exc
    retrieval_llm_ms = _elapsed_ms(retrieval_started)

    try:
        clean_answer, answer_status = classify_aquery_llm_result(envelope)
    except GraphAnswerEnvelopeError as exc:
        # Hard backend failure (status=failure, reason != no_results). Surface
        # as a real 500 — do NOT mask as insufficient information.
        logger.exception("twin_query: aquery_llm envelope failure: %s", exc)
        raise HTTPException(500, f"Query failed: {exc}") from exc

    # bypass mode answered directly from the LLM with no retrieval -- the answer
    # is real but never grounded in sources, so report no_retrieval (not the
    # grounded default the envelope would otherwise imply) and skip projection.
    if body.mode == "bypass":
        await _record_retrieval_activity(
            body,
            request,
            folder=folder,
            sources_count=0,
            stream=False,
            timings={
                "retrieval_llm_ms": retrieval_llm_ms,
                "total_ms": _elapsed_ms(route_started),
            },
        )
        return {
            "response": clean_answer,
            "sources": [],
            "answer_status": ANSWER_STATUS_NO_RETRIEVAL,
        }

    if answer_status == ANSWER_STATUS_INSUFFICIENT:
        await _record_retrieval_activity(
            body,
            request,
            folder=folder,
            sources_count=0,
            stream=False,
            timings={
                "retrieval_llm_ms": retrieval_llm_ms,
                "total_ms": _elapsed_ms(route_started),
            },
        )
        return {
            "response": clean_answer,
            "sources": [],
            "answer_status": answer_status,
        }

    # Keep the sources panel consistent with the grounded answer.
    # _build_envelope_sources projects from the aquery_llm envelope's
    # data.references (no second chunks_vdb pass — that structural lie was
    # removed in audit C3). The _retrieval_scope wrapper is retained
    # defensively so any future retrieval added here stays folder-scoped.
    # clean_answer is available before projection on this path, so the
    # paragraph-anchor evidence is collected from the exact text the
    # operator receives (PARAGRAPH-CITATION-PLAN.md §5.1, /query leg).
    projection_started = time.perf_counter()
    with _retrieval_scope(folder, body):
        sources, projection_ok = await _build_envelope_sources(
            rag,
            body,
            folder,
            envelope,
            citation_evidence=collect_citation_evidence(clean_answer),
            retrieval_scores=retrieval_scores,
        )
    sources_projection_ms = _elapsed_ms(projection_started)
    # A grounded answer whose references could not be projected is surfaced
    # honestly (answer kept, sources empty, explicit status) — never silently as
    # grounded + [] (which reads as "no sources") nor as a 500 (which would hide
    # a usable answer).
    if not projection_ok:
        answer_status = ANSWER_STATUS_SOURCE_PROJECTION_FAILED
        sources_for_activity = []
    else:
        sources_for_activity = sources

    timings = {
        "retrieval_llm_ms": retrieval_llm_ms,
        "sources_projection_ms": sources_projection_ms,
        "total_ms": _elapsed_ms(route_started),
    }
    logger.info(
        "twin_query timings: total=%sms retrieval_llm=%sms sources_projection=%sms "
        "sources=%s mode=%s folder=%s filtered=%s",
        timings["total_ms"],
        retrieval_llm_ms,
        sources_projection_ms,
        len(sources_for_activity),
        body.mode,
        folder,
        _has_advanced_filter(body),
    )
    await _record_retrieval_activity(
        body,
        request,
        folder=folder,
        sources_count=len(sources_for_activity),
        stream=False,
        timings=timings,
    )
    return {
        "response": clean_answer,
        "sources": _public_sources(sources_for_activity),
        "answer_status": answer_status,
    }


async def _twin_query_data(
    get_rag, body: TwinQueryBody, request: Request
) -> dict[str, Any]:
    """Body of ``POST /twin/api/query/data`` (structured retrieval data)."""
    try:
        rag = get_rag()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    from lightrag.base import QueryParam

    # Audit C2 / fix: resolve the active folder once at the route boundary and
    # thread it explicitly through the tag-filter helpers (no ContextVar import
    # in the low-level helpers — keeps the chain testable).
    from ..folder import resolve_folder_for_request

    folder = resolve_folder_for_request(request)

    param = _make_query_param(QueryParam, _query_param_kwargs(body))
    try:
        # Folder-scoped retrieval (see _twin_query): the structured data path
        # grounds on the same vdb queries, so it must be scoped identically.
        with _retrieval_scope(folder, body):
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

    fallback_mode = _query_data_fallback_mode(body)
    if fallback_mode is not None and _query_data_failure_reason(result) == "no_results":
        fallback_kwargs = _query_param_kwargs(body)
        fallback_kwargs["mode"] = fallback_mode
        fallback_param = _make_query_param(QueryParam, fallback_kwargs)
        try:
            with _retrieval_scope(folder, body):
                fallback = await rag.aquery_data(body.query, param=fallback_param)
        except Exception as exc:
            logger.exception("twin_query: fallback aquery_data failed")
            raise HTTPException(500, f"Query data fallback failed: {exc}") from exc
        if isinstance(fallback, dict):
            result = _annotate_query_data_fallback(
                fallback,
                requested_mode=body.mode,
                fallback_mode=fallback_mode,
            )

    if _has_advanced_filter(body):
        result = await _enrich_query_data_chunks_from_source_ids(rag, result)
    elif isinstance(result.get("data"), dict):
        result = dict(result)
        result["data"] = _annotate_query_data_chunk_scores(result["data"])
    result = await _filter_query_data_by_tags(
        rag, result, body.tag_filter_payload, folder
    )
    return {
        "status": result.get("status", "success"),
        "message": result.get("message", "Query executed successfully"),
        "data": result.get("data") if isinstance(result.get("data"), dict) else {},
        "metadata": (
            result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        ),
    }


async def _build_envelope_sources(
    rag,
    body: TwinQueryBody,
    folder,
    envelope,
    citation_evidence=None,
    retrieval_scores=None,
) -> tuple[list, bool]:
    """Project + filter sources from an aquery_llm envelope (shared by /query
    and /query/stream).

    Returns ``(sources, projection_ok)``. When ``data.references`` cannot be
    projected into the Twin contract (LightRAG envelope shape broke), returns
    ``([], False)`` — the caller then surfaces ``answer_status =
    source_projection_failed`` rather than a silent ``grounded`` + ``[]`` (which
    would look like a genuinely source-less answer). No second vector pass — the
    structural lie this path deliberately avoids.

    ``citation_evidence`` (per-citation lexical evidence from
    ``paragraph_anchor``) drives the optional intra-chunk ``anchor``
    enrichment — additive and fail-soft, applied only after the fail-closed
    validations passed."""
    return await _build_envelope_sources_impl(
        rag,
        body,
        folder,
        envelope,
        _fetch_doc_graph_tags,
        _active_tag_batch_fetcher(),
        citation_evidence=citation_evidence,
        retrieval_scores=retrieval_scores,
    )


async def _generate_twin_query_stream(
    rag: Any,
    body: TwinQueryBody,
    request: Request,
    folder: str,
    query_param_cls: Any,
) -> AsyncIterator[str]:
    route_started = time.perf_counter()
    stripper = AnswerMarkerStripper()
    # Fed the same stripper-safe text the client receives; finalized lazily
    # inside the grounded-events closure below, so the insufficient/disconnect
    # paths never pay for (nor trigger) anchor scoring (§5.1).
    evidence_collector = CitationEvidenceCollector()
    envelope: dict[str, Any] | None = None
    try:
        # Flush an explicit stage before the expensive filtered lookup. Besides
        # powering current/future UI feedback, this commits the stream quickly
        # so a slow vector/graph resolution no longer looks like a dead click.
        yield json.dumps({"type": "stage", "value": "retrieval"}) + "\n"
        param = _make_query_param(
            query_param_cls, _query_param_kwargs(body, stream=True)
        )
        retrieval_started = time.perf_counter()
        with _retrieval_scope(folder, body) as retrieval_scores:
            envelope = await _await_query_or_disconnect(
                rag.aquery_llm(body.query, param=param),
                request,
            )
        retrieval_context_ms = _elapsed_ms(retrieval_started)
        yield json.dumps({"type": "stage", "value": "generation"}) + "\n"
        generation_started = time.perf_counter()
        async for line in _emit_answer_tokens(
            envelope, stripper, collector=evidence_collector
        ):
            yield line
        generation_ms = _elapsed_ms(generation_started)
    except ClientDisconnectedDuringQuery:
        logger.info(
            "twin_query stream: client disconnected while aquery_llm was running"
        )
        return
    except Exception as exc:
        async for line in _query_stream_failure_events(exc):
            yield line
        return

    status, fatal_reason = _determine_stream_status(envelope, stripper)
    timings = {
        "retrieval_context_ms": retrieval_context_ms,
        "generation_ms": generation_ms,
    }
    if fatal_reason is not None:
        timings["total_ms"] = _elapsed_ms(route_started)
        async for line in _query_stream_fatal_events(
            body, request, folder, fatal_reason, timings
        ):
            yield line
        return

    no_retrieval = _is_no_retrieval_mode(body)
    if no_retrieval:
        status = ANSWER_STATUS_NO_RETRIEVAL
    if no_retrieval or status == ANSWER_STATUS_INSUFFICIENT:
        timings["total_ms"] = _elapsed_ms(route_started)
        async for line in _query_stream_empty_sources_events(
            body, request, folder, status, timings
        ):
            yield line
        return

    async def _build_sources_with_anchor_evidence(rag_, body_, folder_, envelope_):
        # Same callable contract _query_stream_grounded_events always had;
        # the collector's evidence is bound here so anchors ride the nominal
        # grounded path only, and finalize() runs exactly once, lazily.
        return await _build_envelope_sources(
            rag_,
            body_,
            folder_,
            envelope_,
            citation_evidence=evidence_collector.finalize(),
            retrieval_scores=retrieval_scores,
        )

    yield json.dumps({"type": "stage", "value": "sources"}) + "\n"
    async for line in _query_stream_grounded_events(
        rag,
        body,
        request,
        folder,
        envelope,
        status,
        _build_sources_with_anchor_evidence,
        timings,
        route_started,
    ):
        yield line


def _twin_query_stream(
    get_rag,
    body: TwinQueryBody,
    request: Request,
    *,
    llm_model: str | None = None,
):
    """Body of ``POST /twin/api/query/stream`` (NDJSON tokens + sources event)."""
    try:
        rag = get_rag()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    from lightrag.base import QueryParam
    from ..folder import resolve_folder_for_request

    folder = resolve_folder_for_request(request)
    events = _generate_twin_query_stream(rag, body, request, folder, QueryParam)
    resolved_llm_model = llm_model or getattr(rag, "llm_model_name", None)

    async def _events_with_metadata():
        if resolved_llm_model:
            yield json.dumps(
                {"type": "meta", "value": {"model": resolved_llm_model}}
            ) + "\n"
        async for event in events:
            yield event

    return StreamingResponse(_events_with_metadata(), media_type="application/x-ndjson")


def build_twin_query_router(
    get_rag,
    *,
    auth_dependency=None,
    llm_model: str | None = None,
) -> APIRouter:
    """Mount the Twin overlay query endpoints.

    Args:
        get_rag: zero-arg callable returning the captured ``LightRAG``
            instance. Raises a 500 if the host bootstrap didn't capture
            one (same pattern as the native shims).
    """
    from ..folder import document_folder_header

    # ``document_folder_header`` is documentation-only: the handlers resolve
    # the folder themselves via ``resolve_folder_for_request(request)``, but
    # without the declared dependency the spec would advertise no parameters
    # while the routes honour ``X-Twin-Folder``.
    dependencies = [Depends(document_folder_header)]
    if auth_dependency is not None:
        dependencies.insert(0, Depends(auth_dependency))
    router = APIRouter(tags=["twin-query"], dependencies=dependencies)

    @router.post(
        "/query",
        response_model=TwinQueryResponse,
        summary="Ask a question, get a grounded answer with sources",
        responses={
            499: {"description": "Client closed request"},
            500: {"description": "Query backend error"},
        },
    )
    async def query_endpoint(body: TwinQueryBody, request: Request) -> dict[str, Any]:
        """Answer a question from the knowledge base. The response pairs
        the generated answer with `sources` — the chunks that actually
        grounded it, each with its source document, relevance score and
        `chunk_id` (fetch the chunk text and its surroundings via
        `GET /chunks/{chunk_id}/context`) — so every claim can be traced
        back to its origin. Scope the search with the `X-Twin-Folder`
        header and the optional `tag_filter` / `doc_filter` / `min_score`
        body fields. For token-by-token delivery use
        `POST /query/stream`."""
        result = await _twin_query(get_rag, body, request)
        resolved_llm_model = llm_model or getattr(get_rag(), "llm_model_name", None)
        if resolved_llm_model:
            result["model"] = resolved_llm_model
        return result

    @router.post(
        "/query/data",
        response_model=TwinQueryDataResponse,
        responses={500: {"description": "Query backend error"}},
    )
    async def query_data_endpoint(
        body: TwinQueryBody, request: Request
    ) -> dict[str, Any]:
        """Return structured LightRAG retrieval data through the Twin prefix.

        This mirrors LightRAG's native `/query/data` endpoint while keeping
        the Twin contract (`/twin/api/*`, folder header, tag_filter) on the
        same surface as `/query` and `/query/stream`.
        """
        return await _twin_query_data(get_rag, body, request)

    @router.post(
        "/query/stream",
        responses={500: {"description": "Query backend error"}},
    )
    def query_stream_endpoint(
        body: TwinQueryBody, request: Request
    ) -> StreamingResponse:
        """Stream the LightRAG answer as NDJSON and emit a final sources event.

        Nominal grounded wire format (one JSON object per line):
          {"type":"meta","value":{"model":"<configured model>"}}
          {"type":"stage","value":"retrieval"}
          {"type":"stage","value":"generation"}
          {"type":"token","value":"<chunk text>"}
          ... repeated for every LLM chunk ...
          {"type":"stage","value":"sources"}
          {"type":"status","value":"grounded"|"insufficient_information"
                                    |"source_projection_failed"|"no_retrieval"
                                    |"query_failed"}
          {"type":"sources","value":[<RetrievalSource>, ...]}

        ``meta`` is optional when no model name is configured. ``retrieval``
        is the first query-stage event, ``generation`` precedes every answer
        token, and ``sources`` follows all answer tokens immediately before
        grounded-source projection. Failure, disconnect, no-retrieval and
        insufficient-information paths may terminate before the ``sources``
        stage, but their terminal ``status`` still precedes the final
        ``sources`` payload.

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
        ``"[query failed: <reason>]"`` followed by
        ``status=query_failed`` and an empty ``sources`` event — the
        status carries the failure rather than pretending the error
        notice is a ``grounded`` answer. Callers MUST treat token
        events as possibly-error-bearing and render the text verbatim;
        a ``query_failed`` status (or the absence of a non-empty
        sources payload) signals the run did not complete cleanly.
        ``no_results`` is the
        only failure_reason mapped to ``insufficient_information`` —
        the rest must NOT be masked as such. Pre-stream failures
        (RAG bootstrap, body validation) still surface as real HTTP
        4xx/5xx like the non-stream `/query` route.
        """
        return _twin_query_stream(get_rag, body, request, llm_model=llm_model)

    return router


__all__ = [
    "TwinQueryBody",
    "TwinQueryDataResponse",
    "TwinQueryResponse",
    "_make_query_param",
    "_query_param_kwargs",
    "_source_matches_doc_filter",
    "build_twin_query_router",
]
