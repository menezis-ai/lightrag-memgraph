"""Transient error retry for Memgraph auto-commit transactions.

Memgraph uses MVCC with SNAPSHOT_ISOLATION.  Concurrent transactions
touching the same label index can cause ``TransientError`` (GQL 50N42:
"Cannot resolve conflicting transactions").  Memgraph docs prescribe:
"Retry this transaction when the conflicting transaction is finished."

Usage::

    async def _op():
        result = await session.run(query, **params)
        await result.consume()
    await retry_transient(_op)
"""

import asyncio
import logging
import os
import random
from typing import Any, Awaitable, Callable, TypeVar

from neo4j.exceptions import TransientError

from ._constants import (
    DEFAULT_RETRY_BASE_DELAY_MS,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    MEMGRAPH_RETRY_BASE_DELAY_MS_ENV,
    MEMGRAPH_RETRY_MAX_ATTEMPTS_ENV,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

T = TypeVar("T")

_MAX_DELAY_MS = 2000


def _read_max_attempts() -> int:
    raw = os.environ.get(MEMGRAPH_RETRY_MAX_ATTEMPTS_ENV, "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return DEFAULT_RETRY_MAX_ATTEMPTS


def _read_base_delay_ms() -> int:
    raw = os.environ.get(MEMGRAPH_RETRY_BASE_DELAY_MS_ENV, "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return DEFAULT_RETRY_BASE_DELAY_MS


async def retry_transient(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int | None = None,
    base_delay_ms: int | None = None,
) -> T:
    """Execute *func* with exponential-backoff retry on TransientError.

    Args:
        func: Zero-argument async callable.
        max_attempts: Override for ``MEMGRAPH_RETRY_MAX_ATTEMPTS`` (default 6).
        base_delay_ms: Override for ``MEMGRAPH_RETRY_BASE_DELAY_MS`` (default 50).

    Returns:
        Whatever *func* returns.

    Raises:
        TransientError: If all attempts are exhausted.
    """
    attempts = max_attempts if max_attempts is not None else _read_max_attempts()
    base_ms = base_delay_ms if base_delay_ms is not None else _read_base_delay_ms()

    last_exc: TransientError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except TransientError as exc:
            last_exc = exc
            if attempt == attempts:
                logger.error(
                    "TransientError persisted after %d attempts: %s",
                    attempts,
                    exc,
                )
                raise
            delay_ms = min(base_ms * (2 ** (attempt - 1)), _MAX_DELAY_MS)
            jittered_ms = random.uniform(0, delay_ms)
            logger.warning(
                "TransientError on attempt %d/%d, retrying in %.0fms: %s",
                attempt,
                attempts,
                jittered_ms,
                exc,
            )
            await asyncio.sleep(jittered_ms / 1000.0)

    # Unreachable — satisfies type checker.
    raise last_exc  # type: ignore[misc]
