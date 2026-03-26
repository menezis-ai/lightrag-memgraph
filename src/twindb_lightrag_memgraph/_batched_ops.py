"""Batched Cypher operations to prevent OOM on large datasets.

Memgraph is an in-memory database. Unbounded ``MATCH (n) DETACH DELETE n``
on millions of nodes can spike memory beyond the license limit.  These
helpers split large operations into bounded transactions.
"""

import logging
import os

from . import _pool
from ._constants import DEFAULT_DELETE_BATCH_SIZE, MEMGRAPH_DELETE_BATCH_SIZE_ENV

logger = logging.getLogger("twindb_lightrag_memgraph")


def _resolve_batch_size(override: int | None = None) -> int:
    if override is not None:
        return override
    raw = os.environ.get(MEMGRAPH_DELETE_BATCH_SIZE_ENV, "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return DEFAULT_DELETE_BATCH_SIZE


async def batched_delete(label: str, *, batch_size: int | None = None) -> int:
    """Delete ALL nodes with *label* in batches.

    Returns total number of deleted nodes.
    """
    bs = _resolve_batch_size(batch_size)
    total = 0
    while True:
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"MATCH (n:`{label}`) "
                    f"WITH n LIMIT $batch "
                    f"DETACH DELETE n "
                    f"RETURN count(n) AS deleted",
                    batch=bs,
                )
                record = await result.single()
                await result.consume()
                deleted = record["deleted"] if record else 0
                total += deleted
                if deleted < bs:
                    break
    if total:
        logger.info("Batched delete on :%s — %d nodes removed", label, total)
    return total


async def batched_delete_by_ids(
    label: str, ids: list[str], *, batch_size: int | None = None
) -> int:
    """Delete nodes by ID in batches (for large ID lists)."""
    bs = _resolve_batch_size(batch_size) if batch_size else 5_000
    total = 0
    for i in range(0, len(ids), bs):
        chunk = ids[i : i + bs]
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"UNWIND $ids AS target_id "
                    f"MATCH (n:`{label}` {{id: target_id}}) "
                    f"DETACH DELETE n",
                    ids=chunk,
                )
                await result.consume()
        total += len(chunk)
    if total:
        logger.info("Batched ID delete on :%s — %d IDs processed", label, total)
    return total
