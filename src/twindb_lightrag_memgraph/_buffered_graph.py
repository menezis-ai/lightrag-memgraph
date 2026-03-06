"""Buffered graph proxy for batching upsert_node/upsert_edge into UNWIND queries.

Instead of 130+ individual Bolt round-trips per document (50 entities + 80 relations),
this proxy buffers all upserts and flushes them as 2-3 UNWIND queries.

Read operations (get_node, has_edge, get_edge) support read-your-own-writes
from the buffer, falling back to the real graph for data not yet buffered.
"""

import logging

logger = logging.getLogger("twindb_lightrag_memgraph")


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

    # ── Intercepted write methods (buffered) ──────────────────────────

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """Buffer node upsert instead of firing a Bolt query."""
        if node_id in self._node_buffer:
            self._node_buffer[node_id].update(node_data)
        else:
            self._node_buffer[node_id] = dict(node_data)
        if "entity_type" in node_data:
            self._node_types[node_id] = node_data["entity_type"]

    async def upsert_edge(
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
        return await self._real.get_node(entity_name)

    async def has_node(self, entity_name: str) -> bool:
        if entity_name in self._node_buffer:
            return True
        return await self._real.has_node(entity_name)

    async def has_edge(self, src: str, tgt: str) -> bool:
        if (src, tgt) in self._edge_buffer:
            return True
        return await self._real.has_edge(src, tgt)

    async def get_edge(self, src: str, tgt: str):
        if (src, tgt) in self._edge_buffer:
            return self._edge_buffer[(src, tgt)]
        return await self._real.get_edge(src, tgt)

    # ── Delegate everything else ──────────────────────────────────────

    def __getattr__(self, name):
        return getattr(self._real, name)

    # ── Flush ─────────────────────────────────────────────────────────

    async def flush(self):
        """Flush buffered nodes then edges as UNWIND queries.

        Nodes must flush before edges because upsert_edge uses MATCH
        (not MERGE) for source/target nodes.
        """
        if self._node_buffer:
            await self._flush_nodes()
        if self._edge_buffer:
            await self._flush_edges()
        logger.debug(
            "Buffered flush: %d nodes, %d edges",
            len(self._node_buffer),
            len(self._edge_buffer),
        )

    async def _flush_nodes(self):
        """Single UNWIND query for all buffered nodes + per-type label queries."""
        workspace = self._real.workspace
        entries = [
            {"entity_id": name, "properties": data}
            for name, data in self._node_buffer.items()
        ]

        async with self._real._driver.session() as session:
            await session.run(
                f"""
                UNWIND $entries AS e
                MERGE (n:`{workspace}` {{entity_id: e.entity_id}})
                SET n += e.properties
                """,
                entries=entries,
            )

            # Set additional type labels — group by type to minimize queries.
            # Cypher can't do SET n:$dynamic, so one query per distinct type.
            by_type: dict[str, list[str]] = {}
            for name, node_type in self._node_types.items():
                by_type.setdefault(node_type, []).append(name)
            for node_type, names in by_type.items():
                await session.run(
                    f"""
                    UNWIND $names AS name
                    MATCH (n:`{workspace}` {{entity_id: name}})
                    SET n:`{node_type}`
                    """,
                    names=names,
                )

    async def _flush_edges(self):
        """Single UNWIND query for all buffered edges."""
        workspace = self._real.workspace
        entries = [
            {
                "source_entity_id": src,
                "target_entity_id": tgt,
                "properties": data,
            }
            for (src, tgt), data in self._edge_buffer.items()
        ]

        async with self._real._driver.session() as session:
            await session.run(
                f"""
                UNWIND $entries AS e
                MATCH (source:`{workspace}` {{entity_id: e.source_entity_id}})
                MATCH (target:`{workspace}` {{entity_id: e.target_entity_id}})
                MERGE (source)-[r:DIRECTED]-(target)
                SET r += e.properties
                """,
                entries=entries,
            )
