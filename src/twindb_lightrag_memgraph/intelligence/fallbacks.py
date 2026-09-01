"""Request-local, bounded L3 degradation markers.

Only this closed vocabulary may cross the public Twin query boundary.  Raw
questions, prompts, reasoning text and exception messages remain internal.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

QueryFallback = Literal[
    "intent_fallback",
    "reason_fallback",
    "rerank_fallback",
    "synthesis_failed",
]

_active_fallbacks: ContextVar[list[QueryFallback] | None] = ContextVar(
    "twindb_l3_query_fallbacks",
    default=None,
)


@contextmanager
def query_fallback_scope() -> Iterator[list[QueryFallback]]:
    markers: list[QueryFallback] = []
    token = _active_fallbacks.set(markers)
    try:
        yield markers
    finally:
        _active_fallbacks.reset(token)


def record_query_fallback(marker: QueryFallback) -> None:
    markers = _active_fallbacks.get()
    if markers is not None and marker not in markers and len(markers) < 4:
        markers.append(marker)


__all__ = [
    "QueryFallback",
    "query_fallback_scope",
    "record_query_fallback",
]
