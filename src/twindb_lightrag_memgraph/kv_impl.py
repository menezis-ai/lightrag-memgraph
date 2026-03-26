"""
KV Storage backend using Memgraph nodes as key-value pairs.

Each KV entry is a Cypher node:
  Label: :KV_{workspace}_{namespace}
  Properties: id (key), data (JSON string), __created_at, __updated_at

Index: CREATE INDEX ON :KV_{workspace}_{namespace}(id)
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lightrag.base import BaseKVStorage
from lightrag.utils import logger

from . import _pool
from ._constants import resolve_workspace, validate_identifier


@dataclass
class MemgraphKVStorage(BaseKVStorage):
    def __init__(self, namespace, global_config, embedding_func, **kwargs):
        workspace = resolve_workspace()
        validate_identifier(namespace, "namespace")
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )

    def _label(self) -> str:
        """Cypher label unique to this workspace + namespace."""
        return f"KV_{self.workspace}_{self.namespace}"

    async def initialize(self):
        label = self._label()
        _, database = await _pool.get_driver()
        logger.info(
            "[MemgraphKV:%s] Initializing KV storage on Memgraph (db=%s, label=%s)",
            self.workspace,
            database,
            label,
        )
        async with _pool.get_session() as session:
            try:
                result = await session.run(f"CREATE INDEX ON :`{label}`(id)")
                await result.consume()
                logger.info(f"[MemgraphKV:{self.workspace}] Index on :{label}(id)")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(
                        "[MemgraphKV:%s] Index already exists on :%s(id)",
                        self.workspace,
                        label,
                    )
                else:
                    logger.warning(
                        "[MemgraphKV:%s] Index creation failed: %s", self.workspace, e
                    )

    async def finalize(self):
        pass  # shared driver, closed globally

    async def index_done_callback(self):
        pass  # Memgraph persists automatically

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN n.data AS data",
                id=id,
            )
            record = await result.single()
            await result.consume()
            if record and record["data"]:
                return json.loads(record["data"])
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS target_id
                OPTIONAL MATCH (n:`{label}` {{id: target_id}})
                RETURN target_id, n.data AS data
                """,
                ids=ids,
            )
            # Preserve ordering + return None for missing keys
            records = {r["target_id"]: r["data"] async for r in result}
            await result.consume()
            out = []
            for key in ids:
                raw = records.get(key)
                out.append(json.loads(raw) if raw else None)
            return out

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that do NOT exist in storage."""
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $keys AS k
                OPTIONAL MATCH (n:`{label}` {{id: k}})
                WITH k, n WHERE n IS NULL
                RETURN k
                """,
                keys=list(keys),
            )
            missing = set()
            async for record in result:
                missing.add(record["k"])
            await result.consume()
            return missing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        from ._ttl import compute_ttl_timestamp, get_ttl_namespaces, get_ttl_seconds

        label = self._label()
        now = datetime.now(timezone.utc).isoformat()

        # Determine if this namespace gets TTL
        ttl_seconds = get_ttl_seconds()
        apply_ttl = ttl_seconds is not None and self.namespace in get_ttl_namespaces()
        ttl_ts = compute_ttl_timestamp(ttl_seconds) if apply_ttl else None

        entries = [
            {
                "id": k,
                "data": json.dumps(v, ensure_ascii=False, default=str),
                "ts": now,
            }
            for k, v in data.items()
        ]

        # Build SET clause — add ttl property when TTL is enabled
        set_clause = "SET n.data = e.data, n.__updated_at = e.ts"
        if apply_ttl:
            set_clause += ", n.ttl = $ttl_ts"

        params: dict[str, Any] = {"entries": entries}
        if apply_ttl:
            params["ttl_ts"] = ttl_ts

        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    ON CREATE SET n.__created_at = e.ts
                    {set_clause}
                    """,
                    **params,
                )
                await result.consume()

                # Add :TTL label for nodes that have ttl but not yet the label
                if apply_ttl:
                    result = await session.run(
                        f"""
                        MATCH (n:`{label}`)
                        WHERE n.ttl IS NOT NULL AND NOT n:TTL
                        SET n:TTL
                        """,
                    )
                    await result.consume()

    async def delete(self, ids: list[str]) -> None:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $ids AS target_id
                    MATCH (n:`{label}` {{id: target_id}})
                    DETACH DELETE n
                    """,
                    ids=list(ids),
                )
                await result.consume()

    async def is_empty(self) -> bool:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS cnt LIMIT 1"
            )
            record = await result.single()
            await result.consume()
            return record["cnt"] == 0 if record else True

    async def drop(self) -> dict[str, str]:
        from ._batched_ops import batched_delete

        label = self._label()
        total = await batched_delete(label)
        return {
            "status": "success",
            "message": f"KV namespace {label} dropped ({total} nodes)",
        }
