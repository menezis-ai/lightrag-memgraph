"""TTL property management for Memgraph Enterprise native TTL.

Memgraph Enterprise can automatically delete nodes that have:
- The ``:TTL`` label
- A ``ttl`` integer property (Unix epoch timestamp)

Enable the background job once on the server::

    ENABLE TTL EVERY "1d" AT "02:00:00";

This module adds the TTL label and property to KV nodes at upsert time
when ``MEMGRAPH_TTL_SECONDS`` is set.  Which KV namespaces get TTL is
controlled by ``MEMGRAPH_TTL_LABELS`` (default: ``full_docs,text_chunks``).
"""

import logging
import os
import time

from ._constants import (
    DEFAULT_TTL_LABELS,
    MEMGRAPH_TTL_LABELS_ENV,
    MEMGRAPH_TTL_SECONDS_ENV,
)

logger = logging.getLogger("twindb_lightrag_memgraph")


def get_ttl_seconds() -> int | None:
    """Return configured TTL in seconds, or None if disabled."""
    raw = os.environ.get(MEMGRAPH_TTL_SECONDS_ENV, "")
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except (ValueError, TypeError):
        logger.warning("Invalid MEMGRAPH_TTL_SECONDS=%r, TTL disabled", raw)
        return None


def get_ttl_namespaces() -> set[str]:
    """Return set of KV namespace names that should get TTL labels."""
    raw = os.environ.get(MEMGRAPH_TTL_LABELS_ENV, DEFAULT_TTL_LABELS)
    return {ns.strip() for ns in raw.split(",") if ns.strip()}


def compute_ttl_timestamp(ttl_seconds: int | None = None) -> int | None:
    """Return Unix timestamp for node expiry, or None if TTL disabled."""
    secs = ttl_seconds if ttl_seconds is not None else get_ttl_seconds()
    if secs is None:
        return None
    return int(time.time()) + secs
