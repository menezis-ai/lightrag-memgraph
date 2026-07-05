"""Streaming answer text normalization helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from .._lightrag_compat import (
    ANSWER_STATUS_GROUNDED,
    ANSWER_STATUS_INSUFFICIENT,
    ANSWER_STATUS_QUERY_FAILED,
    ANSWER_STATUS_SOURCE_PROJECTION_FAILED,
    AnswerStatus,
    is_streaming_envelope,
)
from .activity import _record_retrieval_activity
from .request_scope import _retrieval_scope
from .response_sources import _public_sources

logger = logging.getLogger(__name__)


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
    body: Any,
    request: Any,
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
    body: Any,
    request: Any,
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
    body: Any,
    request: Any,
    folder: str,
    envelope: dict[str, Any] | None,
    status: AnswerStatus,
    build_envelope_sources: Any,
) -> AsyncIterator[str]:
    with _retrieval_scope(folder, body):
        sources, projection_ok = await build_envelope_sources(
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


__all__ = [
    "_answer_chunk_to_text",
    "_determine_stream_status",
    "_emit_answer_tokens",
    "_iter_answer_text",
    "_query_stream_empty_sources_events",
    "_query_stream_failure_events",
    "_query_stream_fatal_events",
    "_query_stream_grounded_events",
    "_select_token_source",
]
