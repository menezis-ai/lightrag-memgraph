"""
Document Status Storage backend using Memgraph.

Each doc status is a Cypher node:
  Label: :DocStatus_{workspace}
  Properties: id, status, created_at, updated_at, chunks_count,
              content_summary, content_length, error_msg,
              metadata (JSON), track_id, file_path, chunks_list (JSON)
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger

from . import _pool
from ._constants import resolve_workspace, validate_identifier


@dataclass
class MemgraphDocStatusStorage(DocStatusStorage):
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
        return f"DocStatus_{self.workspace}"

    async def initialize(self):
        label = self._label()
        _, database = await _pool.get_driver()
        logger.info(
            "[MemgraphDocStatus:%s] Initializing DocStatus storage on Memgraph "
            "(db=%s, label=%s)",
            self.workspace,
            database,
            label,
        )
        async with _pool.get_session() as session:
            for prop in ["id", "status", "file_path", "track_id"]:
                try:
                    await session.run(f"CREATE INDEX ON :`{label}`({prop})")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.debug(
                            "[MemgraphDocStatus:%s] Index on %s already exists",
                            self.workspace,
                            prop,
                        )
                    else:
                        logger.warning(
                            "[MemgraphDocStatus:%s] Index creation on %s failed: %s",
                            self.workspace,
                            prop,
                            e,
                        )
        logger.info(f"[MemgraphDocStatus:{self.workspace}] Indexes created on :{label}")

    async def finalize(self):
        pass  # Shared driver; closed globally via _pool.close_driver()

    async def index_done_callback(self):
        pass  # Memgraph persists automatically, no flush needed

    # ── Serialization helpers ──────────────────────────────────────────

    @staticmethod
    def _serialize_status(doc_id: str, status: DocProcessingStatus) -> dict:
        """Convert DocProcessingStatus to flat dict for Cypher properties."""
        d: dict[str, Any] = {
            "id": doc_id,
            "status": status.status.value,
            "created_at": status.created_at or datetime.now(timezone.utc).isoformat(),
            "updated_at": status.updated_at or datetime.now(timezone.utc).isoformat(),
        }
        for field_name in (
            "content_summary",
            "content_length",
            "chunks_count",
            "error_msg",
            "track_id",
            "file_path",
        ):
            val = getattr(status, field_name, None)
            if val is not None:
                d[field_name] = val
        if status.metadata:
            d["metadata"] = json.dumps(status.metadata, default=str)
        if status.chunks_list:
            d["chunks_list"] = json.dumps(status.chunks_list)
        if status.multimodal_processed is not None:
            d["multimodal_processed"] = status.multimodal_processed
        return d

    @staticmethod
    def _deserialize_status(props: dict) -> DocProcessingStatus:
        """Convert Cypher node properties back to DocProcessingStatus."""
        metadata = None
        if "metadata" in props and props["metadata"]:
            try:
                metadata = json.loads(props["metadata"])
            except json.JSONDecodeError:
                metadata = {}

        chunks_list = None
        if "chunks_list" in props and props["chunks_list"]:
            try:
                chunks_list = json.loads(props["chunks_list"])
            except json.JSONDecodeError:
                chunks_list = []

        raw_status = props.get("status", "pending")
        try:
            status = DocStatus(raw_status)
        except ValueError:
            logger.warning(
                f"Unknown doc status '{raw_status}', falling back to PENDING"
            )
            status = DocStatus.PENDING

        return DocProcessingStatus(
            content_summary=props.get("content_summary", ""),
            content_length=props.get("content_length", 0),
            file_path=props.get("file_path", ""),
            status=status,
            created_at=props.get("created_at", ""),
            updated_at=props.get("updated_at", ""),
            track_id=props.get("track_id"),
            chunks_count=props.get("chunks_count"),
            chunks_list=chunks_list,
            error_msg=props.get("error_msg"),
            metadata=metadata or {},
            multimodal_processed=props.get("multimodal_processed"),
        )

    # ── BaseKVStorage interface ────────────────────────────────────────

    @staticmethod
    def _deserialize_props(props: dict) -> dict[str, Any]:
        """Deserialize JSON-encoded fields back to Python objects.

        Fields like chunks_list and metadata are stored as JSON strings
        in Memgraph. LightRAG's adelete_by_doc_id expects chunks_list
        to be a real list (it does ``set(data["chunks_list"])``), so we
        must parse them here.
        """
        out = dict(props)
        for key in ("chunks_list", "metadata"):
            val = out.get(key)
            if isinstance(val, str) and val:
                try:
                    out[key] = json.loads(val)
                except json.JSONDecodeError:
                    pass
        return out

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN properties(n) AS props",
                id=id,
            )
            record = await result.single()
            await result.consume()
            if record:
                return self._deserialize_props(record["props"])
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS target_id
                MATCH (n:`{label}` {{id: target_id}})
                RETURN properties(n) AS props
                """,
                ids=ids,
            )
            out = []
            async for record in result:
                out.append(self._deserialize_props(record["props"]))
            await result.consume()
            return out

    async def filter_keys(self, keys: set[str]) -> set[str]:
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
        entries = []
        for doc_id, doc_data in data.items():
            if isinstance(doc_data, DocProcessingStatus):
                props = self._serialize_status(doc_id, doc_data)
            else:
                props = {"id": doc_id, **doc_data}
                props.setdefault("updated_at", now)
                props.setdefault("created_at", now)
                for k, v in props.items():
                    if isinstance(v, (dict, list)):
                        props[k] = json.dumps(v, default=str)
                    elif hasattr(v, "value"):  # Enum
                        props[k] = v.value
            entries.append({"id": doc_id, "props": props})

        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    SET n += e.props
                    """,
                    entries=entries,
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

    # ── DocStatusStorage-specific interface ─────────────────────────────

    async def get_status_counts(self) -> dict[str, int]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}`) RETURN n.status AS status, count(n) AS cnt"
            )
            counts = {}
            async for record in result:
                counts[record["status"]] = record["cnt"]
            await result.consume()
            return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        label = self._label()
        status_val = status.value
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{status: $status}})
                RETURN n.id AS id, properties(n) AS props
                """,
                status=status_val,
            )
            docs = {}
            async for record in result:
                docs[record["id"]] = self._deserialize_status(record["props"])
            await result.consume()
            return docs

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{track_id: $track_id}})
                RETURN n.id AS id, properties(n) AS props
                """,
                track_id=track_id,
            )
            docs = {}
            async for record in result:
                docs[record["id"]] = self._deserialize_status(record["props"])
            await result.consume()
            return docs

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        label = self._label()
        skip = (page - 1) * page_size
        order = "DESC" if sort_direction == "desc" else "ASC"

        # Whitelist sort fields to prevent injection
        allowed_sort = {"created_at", "updated_at", "id", "status"}
        if sort_field not in allowed_sort:
            sort_field = "updated_at"

        where_clause = ""
        params: dict[str, Any] = {}
        if status_filter is not None:
            where_clause = "WHERE n.status = $status"
            params["status"] = status_filter.value

        async with _pool.get_read_session() as session:
            # Count total
            count_result = await session.run(
                f"MATCH (n:`{label}`) {where_clause} RETURN count(n) AS total",
                **params,
            )
            count_record = await count_result.single()
            await count_result.consume()
            total = count_record["total"] if count_record else 0

            # Fetch page
            result = await session.run(
                f"""
                MATCH (n:`{label}`) {where_clause}
                RETURN n.id AS id, properties(n) AS props
                ORDER BY n.{sort_field} {order}
                SKIP $skip LIMIT $limit
                """,
                **params,
                skip=skip,
                limit=page_size,
            )
            docs = []
            async for record in result:
                docs.append((record["id"], self._deserialize_status(record["props"])))
            await result.consume()
            return docs, total

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{file_path: $file_path}})
                RETURN properties(n) AS props
                """,
                file_path=file_path,
            )
            record = await result.single()
            await result.consume()
            if record:
                return dict(record["props"])
            return None

    async def drop(self) -> dict[str, str]:
        label = self._label()
        try:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                    await result.consume()
            return {"status": "success", "message": f"DocStatus {label} dropped"}
        except Exception as e:
            logger.error("DocStatus drop failed for %s: %s", label, e)
            return {"status": "error", "message": "Drop operation failed"}
