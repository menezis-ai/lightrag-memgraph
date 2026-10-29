"""
KV Storage backend using Memgraph nodes as key-value pairs.

Each KV entry is a Cypher node:
  Label: :LRMG2_KV_w{len(workspace)}_{workspace}_n{len(namespace)}_{namespace}
  Properties: id (key), data (JSON string), __created_at, __updated_at

Index: CREATE INDEX ON :LRMG2_KV_w{len(workspace)}_{workspace}_n{len(namespace)}_{namespace}(id)
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

from lightrag.base import BaseKVStorage
from lightrag.utils import logger

from . import _pool
from ._constants import resolve_workspace, validate_identifier
from ._identity import (
    StorageIdentityMixin,
    initialization_context,
    initialization_session,
    initialize_identity,
    invalidate_identity,
    mark_identity_ready,
    storage_label,
)


@dataclass
class MemgraphKVStorage(StorageIdentityMixin, BaseKVStorage):
    supports_strict_point_reads: ClassVar[bool] = True
    _storage_kind: ClassVar[str] = "kv"

    def __init__(
        self, namespace, global_config, embedding_func, *, workspace=None, **kwargs
    ):
        workspace = resolve_workspace(workspace, global_config)
        validate_identifier(namespace, "namespace")
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )

    def _label(self) -> str:
        """Cypher label unique to this workspace + namespace."""
        return storage_label(self._storage_kind, self.workspace, self.namespace)

    async def initialize(self):
        invalidate_identity(self)
        context = initialization_context(self, self._storage_kind)
        label = self._label()
        _, database = await _pool.get_driver()
        logger.info(
            "[MemgraphKV:%s] Initializing KV storage on Memgraph (db=%s, label=%s)",
            self.workspace,
            database,
            label,
        )
        async with initialization_session() as session:
            await initialize_identity(self, self._storage_kind, session)
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

        mark_identity_ready(self, self._storage_kind, context)

    async def finalize(self):
        invalidate_identity(self)

    async def index_done_callback(self):
        pass  # Memgraph persists automatically

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return await self._get_by_id(id, strict=False)

    async def _get_by_id(self, id: str, *, strict: bool) -> dict[str, Any] | None:
        label = self._label()
        async with self._read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN n.data AS data",
                id=id,
            )
            record = await result.single()
            await result.consume()
            if record:
                if strict:
                    data = json.loads(record["data"])
                    if not isinstance(data, dict):
                        raise ValueError("Stored KV data must be an object")
                    return data
                if record["data"]:
                    return json.loads(record["data"])
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        label = self._label()
        async with self._read_session() as session:
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

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        return await self._get_by_id(id, strict=True)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that do NOT exist in storage."""
        label = self._label()
        async with self._read_session() as session:
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
        label = self._label()
        now = datetime.now(timezone.utc).isoformat()
        entries = [
            {
                "id": k,
                "data": json.dumps(v, ensure_ascii=False, default=str),
                "ts": now,
            }
            for k, v in data.items()
        ]
        async with _pool.acquire_write_slot():
            async with self._write_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    ON CREATE SET n.__created_at = e.ts
                    SET n.data = e.data, n.__updated_at = e.ts
                    """,
                    entries=entries,
                )
                await result.consume()

    async def delete(self, ids: list[str]) -> None:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with self._write_session() as session:
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
        async with self._read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS cnt LIMIT 1"
            )
            record = await result.single()
            await result.consume()
            return record["cnt"] == 0 if record else True

    async def drop(self) -> dict[str, str]:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with self._write_session() as session:
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                await result.consume()
        return {"status": "success", "message": f"KV namespace {label} dropped"}
