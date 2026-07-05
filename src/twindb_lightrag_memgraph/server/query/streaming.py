"""Streaming answer text normalization helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from .._lightrag_compat import (
    ANSWER_STATUS_GROUNDED,
    ANSWER_STATUS_INSUFFICIENT,
    AnswerStatus,
    is_streaming_envelope,
)


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


__all__ = [
    "_answer_chunk_to_text",
    "_determine_stream_status",
    "_emit_answer_tokens",
    "_iter_answer_text",
    "_select_token_source",
]
