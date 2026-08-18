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

try:  # LightRAG 1.5.5+
    from lightrag.exceptions import StorageControlPlaneError
except ImportError:  # pragma: no cover - pre-1.5.5 LightRAG

    class StorageControlPlaneError(RuntimeError):
        """Fallback shim so raise-sites stay importable pre-1.5.5."""


from . import _pool
from ._constants import resolve_workspace, validate_identifier
from ._prompt_security import neutralize_chunk_payloads
from ._retry import with_conflict_retry

# LightRAG's text-chunks KV namespace (``lightrag.namespace.NameSpace``);
# duplicated as a literal so this backend stays importable even if upstream
# renames the constant holder — the namespace string itself is the storage
# contract and cannot change without a data migration anyway.
_KV_TEXT_CHUNKS_NAMESPACE = "text_chunks"


@dataclass
class MemgraphKVStorage(BaseKVStorage):
    def __init__(self, namespace, global_config, embedding_func, **kwargs):
        workspace = validate_identifier(
            str(global_config.get("workspace") or resolve_workspace()), "workspace"
        )
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

    async def finalize(self):  # NOSONAR - async contract.
        pass  # shared driver, closed globally

    async def index_done_callback(self):  # NOSONAR - async contract.
        pass  # Memgraph persists automatically

    # LightRAG 1.5.5 strict point reads: the manual FAILED-retry protocol
    # gates on this to distinguish "content really absent" from "storage
    # failure" — without it every retry leaves FAILED docs untouched. This
    # read is one indexed query whose errors propagate, so the strict
    # contract (miss = confirmed absence, failure = raise) already holds.
    supports_strict_point_reads = True

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

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        """Point read, complete-or-raise (LightRAG 1.5.5 base contract).

        Unlike the lenient ``get_by_id`` (which reports a miss for an
        existing node whose ``data`` payload is empty or JSON null), the
        strict path distinguishes the two: ``None`` ONLY for a confirmed
        missing node; an existing node with an unusable payload raises —
        the manual FAILED-retry protocol treats ``None`` as "content really
        absent" and would silently drop the doc otherwise.
        """
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                OPTIONAL MATCH (n:`{label}` {{id: $id}})
                RETURN n IS NOT NULL AS found, n.data AS data
                """,
                id=id,
            )
            record = await result.single()
            await result.consume()
        if not record or not record["found"]:
            return None  # confirmed absent
        data = record["data"]
        if not data:
            raise StorageControlPlaneError(
                f"[MemgraphKV:{self.workspace}] strict read of {id!r}: node "
                "exists but its data payload is empty — unusable, not absent"
            )
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise StorageControlPlaneError(
                f"[MemgraphKV:{self.workspace}] strict read of {id!r}: node "
                f"exists but its data payload does not decode: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise StorageControlPlaneError(
                f"[MemgraphKV:{self.workspace}] strict read of {id!r}: node "
                f"exists but its payload decodes to {type(parsed).__name__}, "
                "not an object"
            )
        return parsed

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
        label = self._label()
        now = datetime.now(timezone.utc).isoformat()
        if self.namespace == _KV_TEXT_CHUNKS_NAMESPACE:
            # Audit 2026-08-06, R-06: chunk text is untrusted, stored content
            # that lands verbatim in LLM prompts — neutralize reserved prompt
            # delimiters at the storage boundary (see _prompt_security).
            data = neutralize_chunk_payloads(data)
        entries = [
            {
                "id": k,
                "data": json.dumps(v, ensure_ascii=False, default=str),
                "ts": now,
            }
            for k, v in data.items()
        ]

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
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

        # Re-runnable: MERGE + SET, and ON CREATE only fires on insert.
        await with_conflict_retry(f"MemgraphKV.upsert[{label}]", _write)

    async def delete(self, ids: list[str]) -> None:
        label = self._label()

        async def _write() -> None:
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

        # Re-runnable: deleting an already-deleted node matches nothing.
        await with_conflict_retry(f"MemgraphKV.delete[{label}]", _write)

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
        label = self._label()

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                    await result.consume()

        # Re-runnable: a second pass matches an empty label set.
        await with_conflict_retry(f"MemgraphKV.drop[{label}]", _write)
        return {"status": "success", "message": f"KV namespace {label} dropped"}
