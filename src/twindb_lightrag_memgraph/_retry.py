"""Transient error retry for Memgraph auto-commit transactions.

Memgraph uses MVCC with SNAPSHOT_ISOLATION.  Concurrent transactions
touching the same label index can cause ``TransientError`` (GQL 50N42:
"Cannot resolve conflicting transactions").  Memgraph docs prescribe:
"Retry this transaction when the conflicting transaction is finished."

SYNC replication can also cause 50N42 when a replica lags behind.
A dedicated retry profile (longer delays, more attempts) handles this.

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
    DEFAULT_REPLICA_RETRIES,
    DEFAULT_REPLICA_RETRY_DELAY_MS,
    DEFAULT_RETRY_BASE_DELAY_MS,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    MEMGRAPH_REPLICA_RETRIES_ENV,
    MEMGRAPH_REPLICA_RETRY_DELAY_MS_ENV,
    MEMGRAPH_RETRY_BASE_DELAY_MS_ENV,
    MEMGRAPH_RETRY_MAX_ATTEMPTS_ENV,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

T = TypeVar("T")

_MAX_DELAY_MS = 2000
_MAX_REPLICA_DELAY_MS = 30_000
_REPLICA_MARKER = "sync replica"


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


def _read_replica_retries() -> int:
    raw = os.environ.get(MEMGRAPH_REPLICA_RETRIES_ENV, "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return DEFAULT_REPLICA_RETRIES


def _read_replica_delay_ms() -> int:
    raw = os.environ.get(MEMGRAPH_REPLICA_RETRY_DELAY_MS_ENV, "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return DEFAULT_REPLICA_RETRY_DELAY_MS


def _is_replica_error(exc: TransientError) -> bool:
    """Detect SYNC replica lag errors (longer retry profile)."""
    return _REPLICA_MARKER in str(exc).lower()


async def retry_transient(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int | None = None,
    base_delay_ms: int | None = None,
) -> T:
    """Execute *func* with exponential-backoff retry on TransientError.

    Two retry profiles:
    - **MVCC conflict** (default): fast retry, short delays (50ms base, 6 attempts).
    - **SYNC replica lag**: detected by error message, longer retry
      (2s base, 20 attempts) configurable via ``MEMGRAPH_REPLICA_RETRIES``
      and ``MEMGRAPH_REPLICA_RETRY_DELAY_MS``.

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
    max_delay = _MAX_DELAY_MS
    replica_escalated = False

    last_exc: TransientError | None = None
    attempt = 0
    while attempt < attempts:
        attempt += 1
        try:
            return await func()
        except TransientError as exc:
            last_exc = exc

            # On first replica error, escalate to replica retry profile
            if not replica_escalated and _is_replica_error(exc):
                replica_escalated = True
                attempts = max(attempts, _read_replica_retries())
                base_ms = _read_replica_delay_ms()
                max_delay = _MAX_REPLICA_DELAY_MS
                logger.warning(
                    "SYNC replica error detected — switching to replica "
                    "retry profile (%d attempts, %dms base delay)",
                    attempts,
                    base_ms,
                )

            if attempt == attempts:
                logger.error(
                    "TransientError persisted after %d attempts: %s",
                    attempts,
                    exc,
                )
                raise

            delay_ms = min(base_ms * (2 ** (attempt - 1)), max_delay)
            jittered_ms = random.uniform(0, delay_ms)

            error_type = "replica lag" if replica_escalated else "MVCC conflict"
            logger.warning(
                "TransientError (%s) on attempt %d/%d, retrying in %.0fms: %s",
                error_type,
                attempt,
                attempts,
                jittered_ms,
                exc,
            )
            await asyncio.sleep(jittered_ms / 1000.0)

    # Unreachable — satisfies type checker.
    raise last_exc  # type: ignore[misc]
