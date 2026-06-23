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
from typing import Any

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger

from . import _pool
from ._constants import (
    DEFAULT_PAGE_SIZE,
    default_twin_folder,
    get_active_storage_folder,
    resolve_workspace,
    validate_identifier,
)


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

    @staticmethod
    def _status_value(status: DocStatus | str) -> str:
        return status.value if hasattr(status, "value") else str(status)

    @staticmethod
    def _doc_status_supports(field_name: str) -> bool:
        return field_name in getattr(DocProcessingStatus, "__dataclass_fields__", {})

    @staticmethod
    def _folder_from_metadata(metadata: Any) -> str | None:
        if isinstance(metadata, str) and metadata:
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get("folder")
        if not raw:
            return None
        try:
            return validate_identifier(str(raw), "folder")
        except ValueError:
            logger.warning("Ignoring invalid DocStatus metadata.folder=%r", raw)
            return None

    @staticmethod
    def _resolve_folder_for_props(
        props: dict[str, Any],
        metadata: Any,
    ) -> str | None:
        raw = props.get("folder")
        if raw:
            try:
                return validate_identifier(str(raw), "folder")
            except ValueError:
                logger.warning("Ignoring invalid DocStatus folder=%r", raw)
        metadata_folder = MemgraphDocStatusStorage._folder_from_metadata(metadata)
        if metadata_folder:
            return metadata_folder
        active_folder = get_active_storage_folder()
        if active_folder:
            return active_folder
        return None

    @staticmethod
    def _folder_for_read_props(props: dict[str, Any]) -> str:
        raw = props.get("folder")
        if raw:
            try:
                return validate_identifier(str(raw), "folder")
            except ValueError:
                logger.warning("Ignoring invalid stored DocStatus folder=%r", raw)
        metadata_folder = MemgraphDocStatusStorage._folder_from_metadata(
            props.get("metadata")
        )
        return metadata_folder or default_twin_folder()

    @staticmethod
    def resolve_status_filter_values(
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
    ) -> set[str] | None:
        values: set[str] = set()
        if status_filter is not None:
            values.add(MemgraphDocStatusStorage._status_value(status_filter))
        for status in status_filters or []:
            values.add(MemgraphDocStatusStorage._status_value(status))
        return values or None

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
                "folder",
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
            await self._backfill_missing_folders(session, label)
        logger.info(f"[MemgraphDocStatus:{self.workspace}] Indexes created on :{label}")

    async def finalize(self):  # NOSONAR - async contract.
        pass  # Shared driver; closed globally via _pool.close_driver()

    async def index_done_callback(self):  # NOSONAR - async contract.
        pass  # Memgraph persists automatically, no flush needed

    # ── Serialization helpers ──────────────────────────────────────────

    async def _backfill_missing_folders(self, session, label: str) -> None:
        """Populate top-level ``folder`` on legacy DocStatus nodes."""
        default_folder = default_twin_folder()
        while True:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)
                WHERE n.folder IS NULL AND n.id IS NOT NULL
                RETURN n.id AS id, n.metadata AS metadata
                LIMIT 1000
                """
            )
            rows = []
            async for record in result:
                metadata_folder = self._folder_from_metadata(record["metadata"])
                rows.append(
                    {
                        "id": record["id"],
                        "folder": metadata_folder or default_folder,
                    }
                )
            await result.consume()
            if not rows:
                return
            update = await session.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:`{label}` {{id: row.id}})
                WHERE n.folder IS NULL
                SET n.folder = row.folder
                """,
                rows=rows,
            )
            await update.consume()

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
            "folder",
        ):
            val = getattr(status, field_name, None)
            if val is not None:
                d[field_name] = val
        folder = MemgraphDocStatusStorage._resolve_folder_for_props(d, status.metadata)
        if folder:
            d["folder"] = folder
        else:
            d.pop("folder", None)
        if status.metadata:
            d["metadata"] = json.dumps(status.metadata, default=str)
        if status.chunks_list:
            d["chunks_list"] = json.dumps(status.chunks_list)
        multimodal = getattr(status, "multimodal_processed", None)
        if multimodal is not None:
            d["multimodal_processed"] = multimodal
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
        # LightRAG's DocStatus enum values are lowercase ("pending",
        # "processing", "processed", "failed"). Some seed/imported nodes
        # carry uppercase ("PROCESSED") — normalise so the cast doesn't
        # fall back to PENDING and mis-report finished docs as queued.
        try:
            status = DocStatus(str(raw_status).lower())
        except ValueError:
            logger.warning(
                f"Unknown doc status '{raw_status}', falling back to PENDING"
            )
            status = DocStatus.PENDING

        kwargs = {
            "content_summary": props.get("content_summary", ""),
            "content_length": props.get("content_length", 0),
            "file_path": props.get("file_path", ""),
            "status": status,
            "created_at": props.get("created_at", ""),
            "updated_at": props.get("updated_at", ""),
            "track_id": props.get("track_id"),
            "chunks_count": props.get("chunks_count"),
            "chunks_list": chunks_list,
            "error_msg": props.get("error_msg"),
            "metadata": metadata or {},
        }
        if MemgraphDocStatusStorage._doc_status_supports("folder"):
            kwargs["folder"] = MemgraphDocStatusStorage._folder_for_read_props(props)
        if hasattr(DocProcessingStatus, "multimodal_processed"):
            kwargs["multimodal_processed"] = props.get("multimodal_processed")
        if MemgraphDocStatusStorage._doc_status_supports("content_hash"):
            kwargs["content_hash"] = props.get("content_hash")
        return DocProcessingStatus(**kwargs)

    # ── BaseKVStorage interface ────────────────────────────────────────

    @staticmethod
    def _deserialize_props(
        props: dict,
        *,
        include_default_folder: bool = True,
    ) -> dict[str, Any]:
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
        if include_default_folder or out.get("folder"):
            out["folder"] = MemgraphDocStatusStorage._folder_for_read_props(out)
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
                folder = props.get("folder")
                props.pop("folder", None)
            else:
                props = {"id": doc_id, **doc_data}
                props.setdefault("updated_at", now)
                props.setdefault("created_at", now)
                folder = self._resolve_folder_for_props(props, props.get("metadata"))
                props.pop("folder", None)
                for k, v in props.items():
                    if isinstance(v, (dict, list)):
                        props[k] = json.dumps(v, default=str)
                    elif hasattr(v, "value"):  # Enum
                        props[k] = v.value
            entries.append({"id": doc_id, "props": props, "folder": folder})

        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    SET n += e.props
                    WITH n, e
                    WHERE n.folder IS NULL
                    SET n.folder = coalesce(e.folder, $default_folder)
                    """,
                    entries=entries,
                    default_folder=default_twin_folder(),
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

    async def get_status_counts(self, folder: str | None = None) -> dict[str, int]:
        label = self._label()
        where_clause = ""
        params: dict[str, Any] = {}
        if folder:
            where_clause = "WHERE n.folder = $folder"
            params["folder"] = validate_identifier(folder, "folder")
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)
                {where_clause}
                RETURN n.status AS status, count(n) AS cnt
                """,
                **params,
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

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
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
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
        folder: str | None = None,
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        label = self._label()
        skip = (page - 1) * page_size
        order = "DESC" if sort_direction == "desc" else "ASC"

        # Whitelist sort fields to prevent injection
        allowed_sort = {"created_at", "updated_at", "id", "status"}
        if sort_field not in allowed_sort:
            sort_field = "updated_at"

        status_values = self.resolve_status_filter_values(status_filter, status_filters)
        filters: list[str] = []
        params: dict[str, Any] = {}
        if status_values is not None:
            filters.append("n.status IN $statuses")
            params["statuses"] = list(status_values)
        if folder:
            filters.append("n.folder = $folder")
            params["folder"] = validate_identifier(folder, "folder")
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

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
                    ORDER BY n.{sort_field} {order}
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

    async def get_all_status_counts(self, folder: str | None = None) -> dict[str, int]:
        return await self.get_status_counts(folder=folder)

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

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{file_path: $basename}})
                RETURN n.id AS id, properties(n) AS props
                """,
                basename=basename,
            )
            record = await result.single()
            await result.consume()
            if record:
                return record["id"], self._deserialize_props(
                    record["props"],
                    include_default_folder=False,
                )
            return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{content_hash: $content_hash}})
                RETURN n.id AS id, properties(n) AS props
                """,
                content_hash=content_hash,
            )
            record = await result.single()
            await result.consume()
            if record:
                return record["id"], self._deserialize_props(
                    record["props"],
                    include_default_folder=False,
                )
            return None

    async def drop(self) -> dict[str, str]:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                await result.consume()
        return {"status": "success", "message": f"DocStatus {label} dropped"}
