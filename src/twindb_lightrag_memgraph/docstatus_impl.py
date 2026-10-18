"""
Document Status Storage backend using Memgraph.

Each doc status is a Cypher node:
  Label: :DocStatus_{workspace}
  Properties: id, status, created_at, updated_at, chunks_count,
              content_summary, content_length, error_msg,
              metadata (JSON), track_id, file_path, chunks_list (JSON)
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger

from . import _pool
from ._constants import DEFAULT_PAGE_SIZE, resolve_workspace, validate_identifier
from ._docstatus_v15 import DocStatusSchedulingMixin


@dataclass
class MemgraphDocStatusStorage(DocStatusSchedulingMixin, DocStatusStorage):
    supports_strict_point_reads: ClassVar[bool] = True

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
            for prop in [
                "id",
                "status",
                "file_path",
                "track_id",
                "updated_at",
                "created_at",
                "content_hash",
            ]:
                try:
                    result = await session.run(f"CREATE INDEX ON :`{label}`({prop})")
                    await result.consume()
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
            "content_hash",
        ):
            val = getattr(status, field_name, None)
            if val is not None:
                d[field_name] = val
        d["metadata"] = json.dumps(status.metadata or {}, default=str)
        d["chunks_list"] = json.dumps(status.chunks_list or [])
        multimodal = getattr(status, "multimodal_processed", None)
        if multimodal is not None:
            d["multimodal_processed"] = multimodal
        return d

    @staticmethod
    def _deserialize_status(
        props: dict, *, strict: bool = False
    ) -> DocProcessingStatus:
        """Convert Cypher node properties back to DocProcessingStatus."""
        if strict:
            for name in (
                "status",
                "created_at",
                "updated_at",
                "content_summary",
                "content_length",
            ):
                if name not in props:
                    raise ValueError(f"Missing document status field: {name}")
        metadata = None
        if "metadata" in props and props["metadata"]:
            try:
                metadata = json.loads(props["metadata"])
            except json.JSONDecodeError:
                if strict:
                    raise
                metadata = {}

        chunks_list = None
        if "chunks_list" in props and props["chunks_list"]:
            try:
                chunks_list = json.loads(props["chunks_list"])
            except json.JSONDecodeError:
                if strict:
                    raise
                chunks_list = []

        if strict and (
            (metadata is not None and not isinstance(metadata, dict))
            or (chunks_list is not None and not isinstance(chunks_list, list))
        ):
            raise ValueError("Invalid document metadata or chunks_list")

        raw_status = props.get("status", "pending")
        # LightRAG's DocStatus enum values are lowercase ("pending",
        # "processing", "processed", "failed"). Some seed/imported nodes
        # carry uppercase ("PROCESSED") — normalise so the cast doesn't
        # fall back to PENDING and mis-report finished docs as queued.
        try:
            status = DocStatus(str(raw_status).lower())
        except ValueError:
            if strict:
                raise
            logger.warning(
                f"Unknown doc status '{raw_status}', falling back to PENDING"
            )
            status = DocStatus.PENDING

        kwargs = dict(
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
        )
        if "multimodal_processed" in DocProcessingStatus.__dataclass_fields__:
            kwargs["multimodal_processed"] = props.get("multimodal_processed")
        if "content_hash" in DocProcessingStatus.__dataclass_fields__:
            kwargs["content_hash"] = props.get("content_hash")
        return DocProcessingStatus(**kwargs)

    # ── BaseKVStorage interface ────────────────────────────────────────

    @staticmethod
    def _deserialize_props(props: dict, *, strict: bool = True) -> dict[str, Any]:
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
                    if strict:
                        raise
                    pass
            if strict and val is not None:
                expected = dict if key == "metadata" else list
                if not isinstance(out[key], expected):
                    raise ValueError(f"Invalid document status field: {key}")
        return out

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        # LightRAG uses this ordinary point read during document deletion too.
        # Corrupt chunk lists must fail before any destructive operation.
        return await self._get_by_id(id, strict=True)

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        return await self._get_by_id(id, strict=True)

    async def _get_by_id(self, id: str, *, strict: bool) -> dict[str, Any] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN properties(n) AS props",
                id=id,
            )
            record = await result.single()
            await result.consume()
            if record:
                return self._deserialize_props(record["props"], strict=strict)
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
                props = {**doc_data, "id": doc_id}
                props.setdefault("updated_at", now)
                props.setdefault("created_at", now)
            props = self._encode_status_fields(props)
            created_at = props.pop("created_at") or now
            if not isinstance(created_at, str):
                raise ValueError("created_at must be a timestamp string")
            props["updated_at"] = props.get("updated_at") or now
            entries.append({"id": doc_id, "created_at": created_at, "props": props})

        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    ON CREATE SET n.created_at = e.created_at
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
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        label = self._label()
        status_values = [status.value for status in statuses]
        if not status_values:
            return {}
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)
                WHERE n.status IN $statuses
                RETURN n.id AS id, properties(n) AS props
                """,
                statuses=status_values,
            )
            docs = {}
            async for record in result:
                try:
                    docs[record["id"]] = self._deserialize_status(
                        record["props"], strict=strict
                    )
                except (ValueError, TypeError, KeyError):
                    if strict:
                        raise
                    logger.warning("Skipping unusable document status %s", record["id"])
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
        page_size: int = DEFAULT_PAGE_SIZE,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
        *,
        status_filters: list[DocStatus] | None = None,
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
        if status_filters:
            where_clause = "WHERE n.status IN $statuses"
            params["statuses"] = [status.value for status in status_filters]
        elif status_filter is not None:
            where_clause = "WHERE n.status = $status"
            params["status"] = status_filter.value

        # Run count and fetch in parallel on separate read sessions —
        # cuts round-trip time roughly in half on large collections, which
        # avoids nginx upstream timeouts (→ 502 Bad Gateway) on /documents/paginated.
        async def _count() -> int:
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"MATCH (n:`{label}`) {where_clause} RETURN count(n) AS total",
                    **params,
                )
                record = await result.single()
                await result.consume()
                return record["total"] if record else 0

        async def _fetch() -> list[tuple[str, DocProcessingStatus]]:
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"""
                    MATCH (n:`{label}`) {where_clause}
                    RETURN n.id AS id, properties(n) AS props
                    ORDER BY n.{sort_field} {order}, n.id {order}
                    SKIP $skip LIMIT $limit
                    """,
                    **params,
                    skip=skip,
                    limit=page_size,
                )
                out: list[tuple[str, DocProcessingStatus]] = []
                async for record in result:
                    out.append(
                        (record["id"], self._deserialize_status(record["props"]))
                    )
                await result.consume()
                return out

        docs, total = await asyncio.gather(_fetch(), _count())
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
                return self._deserialize_props(record["props"])
            return None

    async def drop(self) -> dict[str, str]:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                await result.consume()
        return {"status": "success", "message": f"DocStatus {label} dropped"}
