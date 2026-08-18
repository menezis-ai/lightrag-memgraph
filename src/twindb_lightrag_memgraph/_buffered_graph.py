"""Buffered graph proxy for batching upsert_node/upsert_edge into UNWIND queries.

Instead of 130+ individual Bolt round-trips per document (50 entities + 80 relations),
this proxy buffers all upserts and flushes them as 2-3 UNWIND queries.

Read operations (get_node, has_edge, get_edge) support read-your-own-writes
from the buffer, falling back to the real graph for data not yet buffered.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ._constants import validate_identifier
from ._pool import _is_closed_transport_error, acquire_write_slot, get_session
from ._retry import with_conflict_retry

logger = logging.getLogger("twindb_lightrag_memgraph")
_T = TypeVar("_T")
_flush_lock: asyncio.Lock | None = None
_flush_lock_loop_id: int | None = None


def _get_flush_lock() -> asyncio.Lock:
    """Return a loop-bound lock for buffered graph flushes.

    LightRAG may merge several documents concurrently in one pipeline run.
    Memgraph can reject overlapping MERGE-heavy graph transactions with
    ``Cannot resolve conflicting transactions`` even when each individual
    write is idempotent. Serializing just this buffered graph flush keeps the
    high-volume KV/vector writes throttled by the normal pool semaphore while
    making graph merge flushes deterministic.
    """
    global _flush_lock, _flush_lock_loop_id

    current_loop_id = id(asyncio.get_running_loop())
    if _flush_lock is None or _flush_lock_loop_id != current_loop_id:
        _flush_lock = asyncio.Lock()
        _flush_lock_loop_id = current_loop_id
    return _flush_lock


class _BufferedGraphProxy:
    """Wraps a MemgraphStorage, buffering upsert_node/upsert_edge calls.

    Reads (get_node, has_edge, get_edge) pass through to the real graph
    with read-your-own-writes from the buffer.
    All other attribute access delegates to the real graph via __getattr__.
    """

    def __init__(self, real_graph):
        self._real = real_graph
        self._node_buffer = {}  # entity_name -> node_data dict
        self._node_types = {}  # entity_name -> entity_type label
        self._edge_buffer = {}  # (src, tgt) -> edge_data dict

    async def _read_with_retry(
        self,
        op_name: str,
        fn: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Retry native graph reads once on stale Bolt transports.

        LightRAG's Memgraph graph backend owns an independent Neo4j
        driver. In long-lived UI runtimes that driver can hand out an
        idle pooled connection whose underlying TCP transport has already
        been closed by Memgraph. The failure is transient; a second call
        gets a fresh connection from the driver's pool. Without this
        guard a single stale read aborts the whole buffered merge.
        """
        try:
            return await fn()
        except Exception as exc:
            if not _is_closed_transport_error(exc):
                raise
            logger.warning(
                "Buffered graph read %s hit a closed Bolt transport; retrying once",
                op_name,
            )
            return await fn()

    # ── Intercepted write methods (buffered) ──────────────────────────

    async def upsert_node(  # NOSONAR - async contract.
        self, node_id: str, node_data: dict[str, str]
    ):
        """Buffer node upsert instead of firing a Bolt query."""
        if node_id in self._node_buffer:
            self._node_buffer[node_id].update(node_data)
        else:
            self._node_buffer[node_id] = dict(node_data)
        if "entity_type" in node_data:
            self._node_types[node_id] = node_data["entity_type"]

    async def upsert_edge(  # NOSONAR - async contract.
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Buffer edge upsert instead of firing a Bolt query."""
        key = (source_node_id, target_node_id)
        if key in self._edge_buffer:
            self._edge_buffer[key].update(edge_data)
        else:
            self._edge_buffer[key] = dict(edge_data)

    # ── Read-your-own-writes passthrough ──────────────────────────────

    async def get_node(self, entity_name: str):
        """Check buffer first, then delegate to real graph."""
        if entity_name in self._node_buffer:
            return self._node_buffer[entity_name]
        return await self._read_with_retry(
            "get_node",
            lambda: self._real.get_node(entity_name),
        )

    async def has_node(self, entity_name: str) -> bool:
        if entity_name in self._node_buffer:
            return True
        return await self._read_with_retry(
            "has_node",
            lambda: self._real.has_node(entity_name),
        )

    async def has_edge(self, src: str, tgt: str) -> bool:
        if (src, tgt) in self._edge_buffer:
            return True
        return await self._read_with_retry(
            "has_edge",
            lambda: self._real.has_edge(src, tgt),
        )

    async def get_edge(self, src: str, tgt: str):
        if (src, tgt) in self._edge_buffer:
            return self._edge_buffer[(src, tgt)]
        return await self._read_with_retry(
            "get_edge",
            lambda: self._real.get_edge(src, tgt),
        )

    # ── Delegate everything else ──────────────────────────────────────

    def __getattr__(self, name):
        return getattr(self._real, name)

    # ── Flush ─────────────────────────────────────────────────────────

    async def flush(self):
        """Flush buffered nodes then edges as UNWIND queries.

        Nodes must flush before edges because upsert_edge uses MATCH
        (not MERGE) for source/target nodes.
        """
        node_count = len(self._node_buffer)
        edge_count = len(self._edge_buffer)
        async with _get_flush_lock():
            try:
                await with_conflict_retry("Buffered flush", self._flush_once)
            except Exception:
                logger.error(
                    "Buffered flush FAILED: %d nodes, %d edges",
                    node_count,
                    edge_count,
                )
                raise
        self._node_buffer.clear()
        self._node_types.clear()
        self._edge_buffer.clear()
        logger.debug(
            "Buffered flush: %d nodes, %d edges",
            node_count,
            edge_count,
        )

    async def _flush_once(self):
        async with acquire_write_slot():
            if self._node_buffer:
                await self._flush_nodes(write_slot_acquired=True)
            if self._edge_buffer:
                await self._flush_edges(write_slot_acquired=True)

    async def _flush_nodes(self, *, write_slot_acquired: bool = False):
        """Single UNWIND query for all buffered nodes + per-type label queries."""
        workspace = validate_identifier(self._real.workspace, "workspace")
        entries = [
            {"entity_id": name, "properties": data}
            for name, data in self._node_buffer.items()
        ]

        async def _write():
            async with get_session() as session:
                await self._flush_nodes_with_session(session, workspace, entries)

        if write_slot_acquired:
            await _write()
        else:
            async with acquire_write_slot():
                await _write()

    async def _flush_nodes_with_session(self, session, workspace: str, entries: list):
        result = await session.run(
            f"""
            UNWIND $entries AS e
            MERGE (n:`{workspace}` {{entity_id: e.entity_id}})
            SET n += e.properties
            """,
            entries=entries,
        )
        await result.consume()

        # Set additional type labels — group by type to minimize queries.
        # Cypher can't do SET n:$dynamic, so one query per distinct type.
        by_type: dict[str, list[str]] = {}
        for name, node_type in self._node_types.items():
            by_type.setdefault(node_type, []).append(name)
        for node_type, names in by_type.items():
            try:
                safe_type = validate_identifier(str(node_type), "entity_type")
            except ValueError:
                logger.warning(
                    "Skipping unsafe buffered entity_type label: %r",
                    node_type,
                )
                continue
            result = await session.run(
                f"""
                UNWIND $names AS name
                MATCH (n:`{workspace}` {{entity_id: name}})
                SET n:`{safe_type}`
                """,
                names=names,
            )
            await result.consume()

    async def _flush_edges(self, *, write_slot_acquired: bool = False):
        """Single UNWIND query for all buffered edges."""
        workspace = validate_identifier(self._real.workspace, "workspace")
        entries = [
            {
                "source_entity_id": src,
                "target_entity_id": tgt,
                "properties": data,
            }
            for (src, tgt), data in self._edge_buffer.items()
        ]

        async def _write():
            async with get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MATCH (source:`{workspace}` {{entity_id: e.source_entity_id}})
                    MATCH (target:`{workspace}` {{entity_id: e.target_entity_id}})
                    MERGE (source)-[r:DIRECTED]-(target)
                    SET r += e.properties
                    """,
                    entries=entries,
                )
                await result.consume()

        if write_slot_acquired:
            await _write()
        else:
            async with acquire_write_slot():
                await _write()
