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
  ``/query`` and an ``[query failed: …]`` token + ``query_failed``
  status on ``/query/stream`` (the HTTP status cannot flip
  mid-stream, so the failure is carried in the status event rather
  than masked as ``grounded``). They are NEVER masked as
  ``insufficient_information``.
- The legacy ``aquery() + chunks_vdb`` path lives on as
  :func:`_build_sources_legacy_fallback`, kept for compat tests in
  isolation. It MUST NOT be invoked from the nominal route paths.

The ``only_need_context`` / ``only_need_prompt`` modes still use the
legacy ``aquery()``: ``aquery_llm`` delegates to ``kg_query`` /
``naive_query``, which DO honor those flags but return a bare
context / prompt string rather than the structured ``{data,
llm_response}`` envelope this module projects — so routing them
through ``aquery_llm`` would mishandle the return shape. The operator
gets the requested body and an empty sources list. These modes, and
``bypass`` (direct LLM, no retrieval), report
``answer_status = no_retrieval`` rather than ``grounded``: they are
sourceless by design, so the empty sources panel is the contract, not
a failure. Distinct from ``insufficient_information`` (retrieval ran,
found nothing usable). See :func:`_is_no_retrieval_mode`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .._lightrag_compat import (
    ANSWER_STATUS_GROUNDED,
    ANSWER_STATUS_INSUFFICIENT,
    ANSWER_STATUS_NO_RETRIEVAL,
    ANSWER_STATUS_QUERY_FAILED,
    ANSWER_STATUS_SOURCE_PROJECTION_FAILED,
    AnswerMarkerStripper,
    AnswerStatus,
    GraphAnswerEnvelopeError,
    build_sources_from_raw_data,
    classify_answer,
    classify_aquery_llm_result,
    collect_chunk_ids,
    is_streaming_envelope,
)
from .doc_lookup import (
    _chunk_to_meta,
    _resolve_chunk_to_doc_id,
    _resolve_doc_for_chunk,
    _resolve_doc_for_file_path,
    _safe_get_score,
)
from .models import TwinQueryBody, TwinQueryDataResponse, TwinQueryResponse
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
from .source_filters import (
    UNKNOWN_SOURCE_NAME,
    _doc_filter_terms,
    _doc_tags_match_filter,
    _source_doc_candidates,
    _source_file_path_candidate,
    _source_matches_doc_filter,
    _tag_filter_terms,
)
from .streaming import _iter_answer_text

_PUBLIC_SOURCE_KEYS = frozenset(("_lightrag_reference_name_fallback",))


def _public_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal source markers before returning public responses."""
    return [
        {key: value for key, value in source.items() if key not in _PUBLIC_SOURCE_KEYS}
        for source in sources
    ]


logger = logging.getLogger(__name__)


class ClientDisconnectedDuringQuery(Exception):
    """Raised when the HTTP client drops a query while retrieval is running."""


async def _await_query_or_disconnect(awaitable, request: Request):
    """Await a costly query call while polling for client disconnect.

    FastAPI does not automatically cancel an in-flight Python coroutine when
    the browser aborts a fetch. Filtered retrieval can be expensive because the
    storage layer performs exact cosine scoring over the pre-filtered corpus; if
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


def _filter_sources_by_min_score(
    sources: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    if min_score <= 0:
        return sources
    return [
        source
        for source in sources
        if isinstance(source.get("score"), (int, float))
        and float(source["score"]) >= min_score
    ]


async def _enrich_sources_doc_ids_from_file_path(
    rag: Any,
    sources: list[dict[str, Any]],
) -> None:
    """Fill missing ``doc_id`` values in-place from projected source paths."""
    file_paths = [
        candidate
        for source in sources
        if not source.get("doc_id")
        for candidate in [_source_file_path_candidate(source)]
        if candidate
    ]
    if not file_paths:
        return
    import asyncio

    unique = list(dict.fromkeys(file_paths))
    resolved = await asyncio.gather(
        *(_resolve_doc_for_file_path(rag, file_path) for file_path in unique),
        return_exceptions=False,
    )
    file_path_to_doc_id = {
        file_path: doc_id for file_path, doc_id in zip(unique, resolved) if doc_id
    }
    if not file_path_to_doc_id:
        return
    for source in sources:
        if source.get("doc_id"):
            continue
        candidate = _source_file_path_candidate(source)
        if candidate and candidate in file_path_to_doc_id:
            source["doc_id"] = file_path_to_doc_id[candidate]


async def _source_matches_tag_filter(
    source: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
    tags_cache: dict[str, set[str]],
) -> bool:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    doc_id = source.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        # If we cannot read ``TAGGED_WITH`` because doc_id is unavailable, we do
        # not assert a match. We keep only explicit matches from resolvable doc
        # ids and fail the source otherwise.
        return False
    if doc_id not in tags_cache:
        tags_cache[doc_id] = await _fetch_doc_graph_tags(doc_id, folder)
    return _doc_tags_match_filter(tags_cache[doc_id], tag_filter)


async def _filter_sources_by_advanced_filters(
    sources: list[dict[str, Any]],
    *,
    tag_filter: dict[str, list[str]] | None,
    doc_filter: dict[str, list[str]] | None,
    folder: str,
) -> tuple[list[dict[str, Any]], bool]:
    tag_required, tag_optional = _tag_filter_terms(tag_filter)
    doc_required, doc_optional = _doc_filter_terms(doc_filter)
    if not tag_required and not tag_optional and not doc_required and not doc_optional:
        return sources, False

    tags_cache: dict[str, set[str]] = {}
    kept: list[dict[str, Any]] = []
    has_unverified = False
    for source in sources:
        if not _source_matches_doc_filter(source, doc_filter):
            if not _source_doc_candidates(source):
                has_unverified = True
            continue
        if not await _source_matches_tag_filter(source, tag_filter, folder, tags_cache):
            if not source.get("doc_id"):
                has_unverified = True
            continue
        kept.append(source)
    return kept, has_unverified


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


async def _filter_query_data_by_tags(
    rag: Any,
    response: dict[str, Any],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
) -> dict[str, Any]:
    return await _filter_query_data_by_tags_impl(
        rag,
        response,
        tag_filter,
        folder,
        _fetch_doc_graph_tags,
    )


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
            or UNKNOWN_SOURCE_NAME
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
    folder: str,
    sources_count: int,
    stream: bool,
) -> None:
    """Best-effort Activity write for completed retrieval calls."""
    try:
        from ..webui_router import _make_event, get_store

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
                "doc_filter": body.doc_filter,
            },
            target_type="query",
        )
        await get_store(folder).record_activity(event)
    except Exception:
        logger.exception("twin_query: failed to record retrieval activity")


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

    # only_need_context / only_need_prompt skip the LLM entirely, so aquery_llm
    # is overkill. Keep the legacy aquery() path here — the operator gets the
    # context/prompt body they asked for and the sources panel stays empty
    # (this branch never claimed grounded sources to begin with).
    if body.only_need_context or body.only_need_prompt:
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
            body, request, folder=folder, sources_count=0, stream=False
        )
        return {
            "response": cleaned,
            "sources": [],
            "answer_status": ANSWER_STATUS_NO_RETRIEVAL,
        }

    # --- Nominal path: aquery_llm gives answer + grounding context in a single
    #     call. The sources panel is built from data.references — the chunks
    #     LightRAG actually used. No second vector retrieval (audit C3).
    try:
        # Folder-scoped grounding: every vector retrieval LightRAG issues inside
        # aquery_llm (chunks + entity/relation vdb) is filtered to the active
        # folder's membership at the storage layer (batch-2 cloisonnement).
        with _retrieval_scope(folder, body):
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
            body, request, folder=folder, sources_count=0, stream=False
        )
        return {
            "response": clean_answer,
            "sources": [],
            "answer_status": ANSWER_STATUS_NO_RETRIEVAL,
        }

    if answer_status == ANSWER_STATUS_INSUFFICIENT:
        await _record_retrieval_activity(
            body, request, folder=folder, sources_count=0, stream=False
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
    with _retrieval_scope(folder, body):
        sources, projection_ok = await _build_envelope_sources(
            rag, body, folder, envelope
        )
    # A grounded answer whose references could not be projected is surfaced
    # honestly (answer kept, sources empty, explicit status) — never silently as
    # grounded + [] (which reads as "no sources") nor as a 500 (which would hide
    # a usable answer).
    if not projection_ok:
        answer_status = ANSWER_STATUS_SOURCE_PROJECTION_FAILED
        sources_for_activity = []
    else:
        sources_for_activity = sources

    await _record_retrieval_activity(
        body,
        request,
        folder=folder,
        sources_count=len(sources_for_activity),
        stream=False,
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
    result = await _filter_query_data_by_tags(rag, result, body.tag_filter, folder)
    return {
        "status": result.get("status", "success"),
        "message": result.get("message", "Query executed successfully"),
        "data": result.get("data") if isinstance(result.get("data"), dict) else {},
        "metadata": (
            result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        ),
    }


async def _build_envelope_sources(
    rag, body: TwinQueryBody, folder, envelope
) -> tuple[list, bool]:
    """Project + filter sources from an aquery_llm envelope (shared by /query
    and /query/stream).

    Returns ``(sources, projection_ok)``. When ``data.references`` cannot be
    projected into the Twin contract (LightRAG envelope shape broke), returns
    ``([], False)`` — the caller then surfaces ``answer_status =
    source_projection_failed`` rather than a silent ``grounded`` + ``[]`` (which
    would look like a genuinely source-less answer). No second vector pass — the
    structural lie this path deliberately avoids."""
    chunk_ids = collect_chunk_ids(envelope or {})
    chunk_to_doc = await _resolve_chunk_to_doc_id(rag, chunk_ids)
    try:
        sources = build_sources_from_raw_data(envelope or {}, chunk_to_doc)
    except GraphAnswerEnvelopeError as exc:
        logger.warning(
            "twin_query: aquery_llm references unprojectable, surfacing empty "
            "sources + source_projection_failed status rather than "
            "reconstructing from a second vector pass: %s",
            exc,
        )
        return [], False
    await _enrich_sources_doc_ids_from_file_path(rag, sources)
    sources = _filter_sources_by_min_score(sources, body.min_score)
    filtered, filter_projection_incomplete = await _filter_sources_by_advanced_filters(
        sources,
        tag_filter=body.tag_filter,
        doc_filter=body.doc_filter,
        folder=folder,
    )
    # If advanced filters are active but could not be validated against all
    # projected sources (missing doc id/name candidates), keep a conservative
    # posture and mark the projection as incomplete.
    return filtered, not filter_projection_incomplete


def _select_token_source(envelope) -> Any:
    """Pick the streaming iterator (or single-shot content) from an envelope."""
    llm_response = (
        envelope.get("llm_response") if isinstance(envelope, dict) else None
    ) or {}
    if is_streaming_envelope(envelope):
        return llm_response.get("response_iterator")
    # Synchronous answer (failure path, bypass mode, non-streaming backend):
    # treat content as a single-shot token.
    return llm_response.get("content") or ""


async def _emit_answer_tokens(envelope, stripper) -> AsyncIterator[str]:
    """Yield NDJSON ``token`` events from the envelope, marker-stripped."""
    async for text in _iter_answer_text(_select_token_source(envelope)):
        for safe in stripper.feed(text):
            if safe:
                yield json.dumps({"type": "token", "value": safe}) + "\n"
    for safe in stripper.flush():
        if safe:
            yield json.dumps({"type": "token", "value": safe}) + "\n"


def _determine_stream_status(envelope, stripper) -> tuple[AnswerStatus, str | None]:
    """Resolve (answer_status, fatal_reason) from the envelope + marker state.

    fatal_reason is set only for a generic backend failure (status=failure,
    reason != no_results) that must be surfaced as an in-stream error token.
    """
    status: AnswerStatus = ANSWER_STATUS_GROUNDED
    fatal_reason: str | None = None
    if isinstance(envelope, dict):
        metadata = envelope.get("metadata") or {}
        failure_reason = (
            metadata.get("failure_reason") if isinstance(metadata, dict) else None
        )
        if envelope.get("status") == "failure":
            if failure_reason == "no_results":
                status = ANSWER_STATUS_INSUFFICIENT
            else:
                fatal_reason = failure_reason or str(
                    envelope.get("message") or "backend failure"
                )
    if stripper.detected and fatal_reason is None:
        status = ANSWER_STATUS_INSUFFICIENT
    return status, fatal_reason


async def _query_stream_failure_events(exc: Exception) -> AsyncIterator[str]:
    logger.exception("twin_query: streaming aquery_llm failed")
    yield json.dumps({"type": "token", "value": f"\n[query failed: {exc}]"}) + "\n"
    yield json.dumps({"type": "status", "value": ANSWER_STATUS_QUERY_FAILED}) + "\n"
    yield json.dumps({"type": "sources", "value": []}) + "\n"


async def _query_stream_fatal_events(
    body: TwinQueryBody,
    request: Request,
    folder: str,
    fatal_reason: str,
) -> AsyncIterator[str]:
    logger.error(
        "twin_query stream: aquery_llm envelope failure surfaced as "
        "in-stream error token: %s",
        fatal_reason,
    )
    yield json.dumps(
        {"type": "token", "value": f"\n[query failed: {fatal_reason}]"}
    ) + "\n"
    yield json.dumps({"type": "status", "value": ANSWER_STATUS_QUERY_FAILED}) + "\n"
    await _record_retrieval_activity(
        body, request, folder=folder, sources_count=0, stream=True
    )
    yield json.dumps({"type": "sources", "value": []}) + "\n"


async def _query_stream_empty_sources_events(
    body: TwinQueryBody,
    request: Request,
    folder: str,
    status: AnswerStatus,
) -> AsyncIterator[str]:
    yield json.dumps({"type": "status", "value": status}) + "\n"
    await _record_retrieval_activity(
        body, request, folder=folder, sources_count=0, stream=True
    )
    yield json.dumps({"type": "sources", "value": []}) + "\n"


async def _query_stream_grounded_events(
    rag: Any,
    body: TwinQueryBody,
    request: Request,
    folder: str,
    envelope: dict[str, Any] | None,
    status: AnswerStatus,
) -> AsyncIterator[str]:
    with _retrieval_scope(folder, body):
        sources, projection_ok = await _build_envelope_sources(
            rag, body, folder, envelope
        )
    if not projection_ok:
        status = ANSWER_STATUS_SOURCE_PROJECTION_FAILED
        sources = []
    yield json.dumps({"type": "status", "value": status}) + "\n"
    await _record_retrieval_activity(
        body, request, folder=folder, sources_count=len(sources), stream=True
    )
    yield json.dumps({"type": "sources", "value": _public_sources(sources)}) + "\n"


async def _generate_twin_query_stream(
    rag: Any,
    body: TwinQueryBody,
    request: Request,
    folder: str,
    query_param_cls: Any,
) -> AsyncIterator[str]:
    stripper = AnswerMarkerStripper()
    envelope: dict[str, Any] | None = None
    try:
        param = _make_query_param(
            query_param_cls, _query_param_kwargs(body, stream=True)
        )
        with _retrieval_scope(folder, body):
            envelope = await _await_query_or_disconnect(
                rag.aquery_llm(body.query, param=param),
                request,
            )
        async for line in _emit_answer_tokens(envelope, stripper):
            yield line
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
    if fatal_reason is not None:
        async for line in _query_stream_fatal_events(
            body, request, folder, fatal_reason
        ):
            yield line
        return

    no_retrieval = _is_no_retrieval_mode(body)
    if no_retrieval:
        status = ANSWER_STATUS_NO_RETRIEVAL
    if no_retrieval or status == ANSWER_STATUS_INSUFFICIENT:
        async for line in _query_stream_empty_sources_events(
            body, request, folder, status
        ):
            yield line
        return

    async for line in _query_stream_grounded_events(
        rag, body, request, folder, envelope, status
    ):
        yield line


def _twin_query_stream(get_rag, body: TwinQueryBody, request: Request):
    """Body of ``POST /twin/api/query/stream`` (NDJSON tokens + sources event)."""
    try:
        rag = get_rag()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc

    from lightrag.base import QueryParam
    from ..folder import resolve_folder_for_request

    folder = resolve_folder_for_request(request)
    return StreamingResponse(
        _generate_twin_query_stream(rag, body, request, folder, QueryParam),
        media_type="application/x-ndjson",
    )


def build_twin_query_router(get_rag) -> APIRouter:
    """Mount the Twin overlay query endpoints.

    Args:
        get_rag: zero-arg callable returning the captured ``LightRAG``
            instance. Raises a 500 if the host bootstrap didn't capture
            one (same pattern as the native shims).
    """
    router = APIRouter(tags=["twin-query"])

    @router.post(
        "/query",
        response_model=TwinQueryResponse,
        responses={
            499: {"description": "Client closed request"},
            500: {"description": "Query backend error"},
        },
    )
    async def query_endpoint(body: TwinQueryBody, request: Request) -> dict[str, Any]:
        return await _twin_query(get_rag, body, request)

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

        Wire format (one JSON object per line):
          {"type":"token","value":"<chunk text>"}
          ... repeated for every LLM chunk ...
          {"type":"status","value":"grounded"|"insufficient_information"
                                    |"source_projection_failed"|"no_retrieval"
                                    |"query_failed"}
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
        return _twin_query_stream(get_rag, body, request)

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
