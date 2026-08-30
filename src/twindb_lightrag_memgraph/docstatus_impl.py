"""
Document Status Storage backend using Memgraph.

Each doc status is a Cypher node:
  Label: :DocStatus_{workspace}
  Properties: id, status, created_at, updated_at, chunks_count,
              content_summary, content_length, error_msg,
              metadata (JSON), track_id, file_path, chunks_list (JSON)
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger

# ── LightRAG 1.5.5+ scheduling contract (guarded import) ──────────────────
# 1.5.5 grew four abstract methods on DocStatusStorage (keyset sweep, batch
# strict reads, typed source resolution). The types below do not exist on
# 1.4.x / ≤1.5.4 — the guard keeps this module importable during the BNP
# transition window; the methods themselves are only ever CALLED by a 1.5.5+
# pipeline.
try:  # pragma: no cover - exercised implicitly by the version in the venv
    from lightrag.base import (
        CURSOR_END,
        CURSOR_START,
        CursorAfter,
        CursorPosition,
        DocSchedulingRecord,
        DocStatusPage,
        SourceAbsent,
        SourceConflict,
        SourceConflictPage,
        SourceConflictRepairResult,
        SourceConflictSummary,
        SourceResolution,
        SourceUnique,
    )
    from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
    from lightrag.exceptions import (
        SourceConflictRepairCASError,
        StorageControlPlaneError,
    )

    _HAS_155_SCHEDULING = True
except ImportError:  # pragma: no cover - pre-1.5.5 LightRAG
    _HAS_155_SCHEDULING = False
    CURSOR_END = CURSOR_START = None
    CursorAfter = CursorPosition = None
    DocSchedulingRecord = DocStatusPage = None
    SourceAbsent = SourceConflict = SourceResolution = SourceUnique = None
    SourceConflictPage = SourceConflictRepairResult = SourceConflictSummary = None
    CUSTOM_CHUNK_PATCH_METADATA_KEY = "custom_chunk_patch"

    class StorageControlPlaneError(RuntimeError):
        """Fallback shim so raise-sites stay importable pre-1.5.5."""

    class SourceConflictRepairCASError(StorageControlPlaneError):
        """Fallback shim so raise-sites stay importable pre-1.5.5."""


from . import _pool
from ._constants import (
    DEFAULT_PAGE_SIZE,
    default_twin_folder,
    get_active_duplicate_share_folder,
    get_active_upload_relative_path,
    get_active_storage_folder,
    get_confirmed_content_doc_ids,
    purge_llm_cache_on_failed_enabled,
    resolve_workspace,
    validate_identifier,
)
from ._import_cleanup import cleanup_processed_imports
from ._retry import with_conflict_retry
from ._upload_paths import canonical_upload_file_name


@dataclass
class MemgraphDocStatusStorage(DocStatusStorage):
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
        return f"DocStatus_{self.workspace}"

    def _folder_label(self) -> str:
        """Cypher label for (:Folder) membership nodes, workspace-scoped.

        Folders are a *logical* cloisonnement: a document is `MEMBER_OF` one or
        more folders, stored once. The label is workspace-scoped like the doc
        labels so the membership graph lives in the same physical namespace.
        See docs/adr/006-folder-membership-relation.md.
        """
        return f"Folder_{self.workspace}"

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
    def _metadata_dict(metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str) and metadata:
            try:
                decoded = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    @staticmethod
    def _duplicate_original_doc_id(props: dict[str, Any]) -> str | None:
        """Return the original only when duplicate evidence is content-safe.

        LightRAG 1.5.6 derives known-source document ids from ``file_path``.
        Consequently neither ``original_doc_id`` nor the derived duplicate
        record id proves content equality for a ``filename`` collision. Such a
        record is eligible for sharing only when it carries an explicit
        candidate ``content_hash``; the write query then requires equality
        with the original row's hash atomically. ``content_hash`` duplicate
        verdicts are already the result of LightRAG's explicit hash lookup.
        """
        metadata = MemgraphDocStatusStorage._metadata_dict(props.get("metadata"))
        duplicate_kind = str(metadata.get("duplicate_kind") or "").strip().lower()
        if metadata.get("is_duplicate") is not True or not metadata.get(
            "original_doc_id"
        ):
            return None
        original_doc_id = str(metadata["original_doc_id"])
        if duplicate_kind == "content_hash":
            return original_doc_id
        if duplicate_kind == "filename" and props.get("content_hash"):
            return original_doc_id
        return None

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
        await self._run_backfills(label)
        logger.info(f"[MemgraphDocStatus:{self.workspace}] Indexes created on :{label}")

    async def _run_backfills(self, label: str) -> None:
        """Run the two migration backfills under the write slot and retry.

        These are the only *data-mutating* writes in ``initialize()`` — the
        ``CREATE INDEX`` statements above are DDL, deliberately left outside
        this helper because they already swallow "already exists" and are not
        the write/write conflict shape. The backfills used to run on that same
        index session with no write slot and no retry, which is why the
        ``acquire_write_slot()`` inventory missed them. ``_backfill_membership``
        bumps ``__membership_epoch``, so a second worker booting (or a boot
        during ingestion) races exactly the conflict this module exists to
        absorb, and an unretried conflict aborts initialization.

        Deliberately OUTSIDE the index session: _retry's invariant is that each
        attempt gets a *fresh* session, so re-running on the session whose
        transaction was just aborted would defeat the retry. Index creation
        stays where it was — it is DDL, already tolerant of its own errors.

        Whole-operation re-run is safe: both backfills are guarded
        (``WHERE n.folder IS NULL``, ``NOT EXISTS((n)-[:MEMBER_OF]->(...))``),
        so rows completed by a losing attempt are skipped rather than redone.
        """

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    await self._backfill_missing_folders(session, label)
                    await self._backfill_membership(session, label)

        await with_conflict_retry(f"MemgraphDocStatus.backfills[{label}]", _write)

    async def _backfill_membership(self, session, label: str) -> None:
        """Create `MEMBER_OF` edges for legacy nodes that only carry the
        single-valued ``folder`` property, so membership-based reads see them.

        Idempotent (MERGE) and re-runnable on every boot. This is what lets the
        reads be membership-authoritative without a separate migration step.
        """
        flabel = self._folder_label()
        result = await session.run(f"""
            MATCH (n:`{label}`)
            WHERE n.folder IS NOT NULL
              AND n.__delete_claim IS NULL
              AND NOT EXISTS((n)-[:MEMBER_OF]->(:`{flabel}`))
            SET n.__membership_epoch = coalesce(n.__membership_epoch, 0) + 1
            MERGE (f:`{flabel}` {{id: n.folder}})
            MERGE (n)-[membership:MEMBER_OF]->(f)
            SET membership.updated_at = coalesce(
                membership.updated_at, n.updated_at, n.created_at)
            """)
        await result.consume()

    async def finalize(self):  # NOSONAR - async contract.
        pass  # Shared driver; closed globally via _pool.close_driver()

    async def index_done_callback(self):  # NOSONAR - async contract.
        pass  # Memgraph persists automatically, no flush needed

    # ── Serialization helpers ──────────────────────────────────────────

    async def _backfill_missing_folders(self, session, label: str) -> None:
        """Populate top-level ``folder`` on legacy DocStatus nodes."""
        default_folder = default_twin_folder()
        while True:
            result = await session.run(f"""
                MATCH (n:`{label}`)
                WHERE n.folder IS NULL AND n.id IS NOT NULL
                RETURN n.id AS id, n.metadata AS metadata
                LIMIT 1000
                """)
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

        # LightRAG 1.4.9.11 computes doc ids from normalized content, calls
        # this method, then silently drops ids that already exist. It has
        # neither the content-hash getter nor duplicate metadata used by newer
        # releases. The enqueue patch binds only ids computed from the actual
        # input body, so this is a content-confirmed share—not an arbitrary-id
        # or filename match.
        active_folder = get_active_duplicate_share_folder()
        confirmed_existing = (
            (set(keys) - missing) & set(get_confirmed_content_doc_ids())
            if active_folder
            else set()
        )
        for doc_id in sorted(confirmed_existing):
            shared = await self.add_to_folder(doc_id, active_folder)
            if shared:
                logger.info(
                    "[MemgraphDocStatus:%s] shared legacy content-identical "
                    "doc %s into folder %s before filter_keys suppressed it",
                    self.workspace,
                    doc_id,
                    active_folder,
                )
            else:
                logger.warning(
                    "[MemgraphDocStatus:%s] could not share legacy "
                    "content-identical doc %s into folder %s because the "
                    "document disappeared or is delete-claimed",
                    self.workspace,
                    doc_id,
                    active_folder,
                )
        return missing

    def _serialize_upsert_props(
        self, doc_id: str, doc_data: Any, now: str
    ) -> tuple[dict[str, Any], str | None]:
        """Serialize one upsert record into ``(props, folder)``.

        Pops the transient ``folder`` key out of ``props`` (it drives the
        MEMBER_OF relation, not a node property) and JSON/Enum-encodes plain
        dict payloads. ``DocProcessingStatus`` instances are already encoded by
        ``_serialize_status``."""
        if isinstance(doc_data, DocProcessingStatus):
            props = self._serialize_status(doc_id, doc_data)
            self._attach_active_relative_path(props)
            folder = props.get("folder")
            props.pop("folder", None)
            return props, folder
        props = {"id": doc_id, **doc_data}
        props.setdefault("updated_at", now)
        props.setdefault("created_at", now)
        self._attach_active_relative_path(props)
        folder = self._resolve_folder_for_props(props, props.get("metadata"))
        props.pop("folder", None)
        for k, v in props.items():
            if isinstance(v, (dict, list)):
                props[k] = json.dumps(v, default=str)
            elif hasattr(v, "value"):  # Enum
                props[k] = v.value
        return props, folder

    @staticmethod
    def _attach_active_relative_path(props: dict[str, Any]) -> None:
        """Persist folder-upload display provenance without changing file_path."""
        relative_path = get_active_upload_relative_path()
        if not relative_path:
            return
        file_path = str(props.get("file_path") or "")
        expected_file_path = canonical_upload_file_name(relative_path)
        if file_path and file_path != expected_file_path:
            raise ValueError(
                "relative upload path does not match the canonical stored file name"
            )
        metadata = MemgraphDocStatusStorage._metadata_dict(props.get("metadata"))
        metadata["relative_path"] = relative_path
        props["metadata"] = json.dumps(metadata, default=str)

    async def _run_upsert_writes(
        self,
        session,
        label: str,
        folder_label: str,
        entries: list[dict],
        duplicate_memberships: list[dict[str, Any]],
    ) -> tuple[list[dict], dict[str, Any]]:
        """Persist docs and fall back visibly when duplicate sharing misses.

        Returns ``(fallback_entries, old_statuses)`` where ``old_statuses``
        maps each written doc id to the ``status`` it carried *before* this
        write (``None`` on first insert) — the transition signal behind the
        ``source-ready`` / ``source-failed`` Activity events (QA ACT-V5-001).
        """
        old_statuses: dict[str, Any] = {}

        async def _write_entries(batch: list[dict]) -> None:
            if not batch:
                return
            result = await session.run(
                f"""
                UNWIND $entries AS e
                MERGE (n:`{label}` {{id: e.id}})
                WITH n, e, n.status AS old_status
                SET n += e.props
                RETURN e.id AS id, old_status
                """,
                entries=batch,
            )
            async for record in result:
                old_statuses[record["id"]] = record["old_status"]
            await result.consume()
            # Keep status/retry-state updates possible while a deletion owns the
            # claim, but never create another membership under that claim. This
            # second query also writes the node guard, so it conflicts with a
            # claim racing from an older snapshot.
            result = await session.run(
                f"""
                UNWIND $entries AS e
                MATCH (n:`{label}` {{id: e.id}})
                WHERE n.__delete_claim IS NULL
                SET n.__membership_epoch = coalesce(n.__membership_epoch, 0) + 1
                WITH n,
                     coalesce(e.folder, $default_folder) AS fid,
                     e.membership_updated_at AS membership_updated_at
                // Dual-write the legacy single-valued property (migration
                // safety net) only on first insert.
                FOREACH (_ IN CASE WHEN n.folder IS NULL THEN [1] ELSE [] END |
                    SET n.folder = fid)
                // The membership relation is the new source of truth: a doc
                // is MEMBER_OF one or more folders, stored once.
                MERGE (f:`{folder_label}` {{id: fid}})
                MERGE (n)-[membership:MEMBER_OF]->(f)
                // Normal ingestion and reprocessing refresh this folder-local
                // document clock. The duplicate-share, add, and backfill
                // writers use coalesce so joining another folder records when
                // that membership was created instead.
                SET membership.updated_at = membership_updated_at
                """,
                entries=batch,
                default_folder=default_twin_folder(),
            )
            await result.consume()

        await _write_entries(entries)
        fallback_entries: list[dict] = []
        if duplicate_memberships:
            result = await session.run(
                f"""
                UNWIND $memberships AS m
                MATCH (n:`{label}` {{id: m.doc_id}})
                WHERE n.__delete_claim IS NULL
                  AND (
                    NOT m.requires_hash_match OR
                    (m.content_hash IS NOT NULL AND n.content_hash = m.content_hash)
                  )
                SET n.__membership_epoch = coalesce(n.__membership_epoch, 0) + 1
                MERGE (f:`{folder_label}` {{id: m.folder}})
                MERGE (n)-[membership:MEMBER_OF]->(f)
                SET membership.updated_at = coalesce(
                    membership.updated_at, m.membership_updated_at)
                RETURN collect(m.duplicate_id) AS duplicate_ids
                """,
                memberships=duplicate_memberships,
            )
            record = await result.single()
            await result.consume()
            shared_ids = set(record["duplicate_ids"] or []) if record else set()
            for membership in duplicate_memberships:
                duplicate_id = membership["duplicate_id"]
                if duplicate_id in shared_ids:
                    logger.info(
                        "[MemgraphDocStatus:%s] duplicate record %s shared "
                        "original doc %s into folder %s; suppressing the "
                        "visible duplicate",
                        self.workspace,
                        duplicate_id,
                        membership["doc_id"],
                        membership["folder"],
                    )
                    continue
                logger.warning(
                    "[MemgraphDocStatus:%s] duplicate record %s could not share "
                    "original doc %s into folder %s because it disappeared or "
                    "is delete-claimed; preserving the visible duplicate",
                    self.workspace,
                    duplicate_id,
                    membership["doc_id"],
                    membership["folder"],
                )
                fallback_entries.append(
                    {
                        "id": duplicate_id,
                        "props": membership["duplicate_props"],
                        "folder": membership["folder"],
                        "membership_updated_at": membership["membership_updated_at"],
                    }
                )
            await _write_entries(fallback_entries)
        return fallback_entries, old_statuses

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        label = self._label()
        now = datetime.now(timezone.utc).isoformat()
        entries = []
        duplicate_memberships: list[dict[str, Any]] = []
        cleanup_props: list[dict[str, Any]] = []
        for doc_id, doc_data in data.items():
            props, folder = self._serialize_upsert_props(doc_id, doc_data, now)
            cleanup_props.append(props)
            original_doc_id = self._duplicate_original_doc_id(props)
            if original_doc_id and folder:
                metadata = self._metadata_dict(props.get("metadata"))
                duplicate_kind = (
                    str(metadata.get("duplicate_kind") or "").strip().lower()
                )
                duplicate_memberships.append(
                    {
                        "doc_id": original_doc_id,
                        "duplicate_id": doc_id,
                        "duplicate_props": props,
                        "folder": folder,
                        "content_hash": props.get("content_hash"),
                        "requires_hash_match": duplicate_kind == "filename",
                        "membership_updated_at": now,
                    }
                )
                continue
            entries.append(
                {
                    "id": doc_id,
                    "props": props,
                    "folder": folder,
                    "membership_updated_at": now,
                }
            )

        if not entries and not duplicate_memberships:
            return

        folder_label = self._folder_label()

        async def _write() -> tuple[list[dict], dict[str, Any]]:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    return await self._run_upsert_writes(
                        session, label, folder_label, entries, duplicate_memberships
                    )

        # This is the path that lost the race in CI: the membership-epoch guard
        # in _run_upsert_writes' second query is *designed* to conflict with a
        # delete claim racing from an older snapshot. Re-running the whole helper
        # is safe even though its three queries autocommit independently — each
        # is MERGE/SET shaped, and the epoch bump is write-only (see _retry).
        fallback_entries, old_statuses = await with_conflict_retry(
            f"MemgraphDocStatus.upsert[{label}]", _write
        ) or ([], {})
        await cleanup_processed_imports(cleanup_props)

        failed_doc_ids = [
            e["id"]
            for e in [*entries, *fallback_entries]
            if e["props"].get("status") == DocStatus.FAILED.value
        ]
        if failed_doc_ids:
            await self._purge_failed_doc_llm_cache(failed_doc_ids)

        await self._emit_source_status_activity(
            self._status_transitions(entries, old_statuses)
        )

    @staticmethod
    def _status_transitions(
        entries: list[dict[str, Any]], old_statuses: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Terminal-status *transitions* among the written entries.

        QA ACT-V5-001: the Activity ledger exposed ``source-ready`` /
        ``source-failed`` filters with no live emission behind them. A
        transition fires only when the stored status actually changed, so a
        re-upsert of an already-processed document emits nothing (backfills,
        metadata rewrites), while a genuine reprocess emits again — which is
        the honest audit trail.
        """
        transitions: list[dict[str, Any]] = []
        for entry in entries:
            props = entry.get("props") or {}
            new_status = str(props.get("status") or "").lower()
            if new_status not in ("processed", "failed"):
                continue
            old_status = old_statuses.get(entry["id"])
            if old_status is not None and str(old_status).lower() == new_status:
                continue
            transitions.append(
                {
                    "doc_id": entry["id"],
                    "status": new_status,
                    "folder": entry.get("folder"),
                    "file_path": props.get("file_path"),
                    "track_id": props.get("track_id"),
                    "chunks_count": props.get("chunks_count"),
                    "error_msg": props.get("error_msg"),
                }
            )
        return transitions

    async def _emit_source_status_activity(
        self, transitions: list[dict[str, Any]]
    ) -> None:
        """Best-effort ``source-ready`` / ``source-failed`` Activity emission.

        Mirrors the ``_signal_empty_extraction_merge`` pattern in
        ``patches/registry.py``: the overlay store is imported lazily and any
        failure is swallowed — the audit ledger must never break ingestion,
        and a storage-only deployment (no server overlay) simply skips it.
        """
        for transition in transitions:
            try:
                from .server.webui_router import _make_event, get_store

                ready = transition["status"] == "processed"
                doc_id = transition["doc_id"]
                label = str(transition.get("file_path") or doc_id)
                if ready:
                    summary = f"Source {label} processed and ready for retrieval"
                else:
                    reason = str(transition.get("error_msg") or "").strip()
                    summary = f"Source {label} failed during ingestion" + (
                        f" — {reason}" if reason else ""
                    )
                event = _make_event(
                    kind="source-ready" if ready else "source-failed",
                    sev="info" if ready else "error",
                    actor="system",
                    target_label=label,
                    summary=summary,
                    meta={
                        "doc_id": doc_id,
                        "path": transition.get("file_path"),
                        "track_id": transition.get("track_id"),
                        "chunks_count": transition.get("chunks_count"),
                        "error_msg": transition.get("error_msg"),
                        "workspace": self.workspace,
                        "folder": transition.get("folder"),
                    },
                    target_type="source",
                    target_id=doc_id,
                )
                await get_store(transition.get("folder")).record_activity(event)
            except Exception as exc:  # store absent/unreachable — never raises
                logger.debug("source-status activity event skipped: %s", exc)

    async def _purge_failed_doc_llm_cache(self, doc_ids: list[str]) -> None:
        """Purge LLM extraction-cache rows tied to docs that just FAILED.

        LightRAG persists the entity-extraction LLM cache even when the
        document itself fails — the extraction cache is gated by
        ``enable_llm_cache_for_entity_extract`` (default true, independent of
        ``enable_llm_cache``), and the FAILED handlers in ``process_document``
        explicitly flush the cache *before* writing the FAILED status row
        (verified on the 1.4.9.11 wheel). Left in place, a re-ingestion of the
        same document replays the cached — possibly truncated or imparsable —
        responses instead of re-calling the LLM (audit 2026-07-02 addendum,
        finding B).

        Cache rows are matched through the failed doc's chunk ids: extraction
        cache entries embed their ``chunk_id`` in the JSON ``data`` payload,
        and chunk rows embed ``full_doc_id``. Query-mode cache rows carry no
        chunk id and are never touched. Best-effort by design: any failure is
        logged and swallowed — this hygiene pass must never break the
        ingestion pipeline. Disable with ``TWIN_PURGE_LLM_CACHE_ON_FAILED=0``
        (LightRAG-native behavior: cache rows survive the failure).
        """
        if not purge_llm_cache_on_failed_enabled():
            return
        try:
            ws = validate_identifier(self.workspace, "workspace")
            chunks_label = f"KV_{ws}_text_chunks"
            cache_label = f"KV_{ws}_llm_response_cache"
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"""
                    MATCH (c:`{chunks_label}`)
                    WHERE any(d IN $doc_ids WHERE c.data CONTAINS d)
                    RETURN c.id AS id
                    """,
                    doc_ids=list(doc_ids),
                )
                chunk_ids = [record["id"] async for record in result]
                await result.consume()
            if not chunk_ids:
                return

            async def _write():
                async with _pool.acquire_write_slot():
                    async with _pool.get_session() as session:
                        result = await session.run(
                            f"""
                            MATCH (n:`{cache_label}`)
                            WHERE any(cid IN $chunk_ids WHERE n.data CONTAINS cid)
                            DETACH DELETE n
                            """,
                            chunk_ids=chunk_ids,
                        )
                        return await result.consume()

            # Re-runnable: a retry matches the rows the first attempt did not
            # delete. The whole method is best-effort (caller-side try/except),
            # so a retry only widens the window in which the purge succeeds.
            summary = await with_conflict_retry(
                f"MemgraphDocStatus._purge_failed_doc_llm_cache[{cache_label}]",
                _write,
            )
            deleted = getattr(getattr(summary, "counters", None), "nodes_deleted", None)
            if deleted:
                logger.info(
                    "[MemgraphDocStatus:%s] purged %s LLM extraction-cache "
                    "row(s) for FAILED doc(s) %s",
                    self.workspace,
                    deleted,
                    doc_ids,
                )
        except Exception:
            logger.warning(
                "[MemgraphDocStatus:%s] LLM extraction-cache purge after "
                "FAILED status failed for docs %s (ingestion unaffected)",
                self.workspace,
                doc_ids,
                exc_info=True,
            )

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
        await with_conflict_retry(f"MemgraphDocStatus.delete[{label}]", _write)

    # ── Folder membership (many-to-many; data stored once) ─────────────
    # A document is MEMBER_OF one or more folders. See
    # docs/adr/006-folder-membership-relation.md. These back the explicit membership
    # endpoints and the ref-counted delete.

    async def add_to_folder(self, doc_id: str, folder: str) -> bool:
        """Add an existing document to a folder. Idempotent; no content copy.

        Returns True if the document exists (membership ensured), False if no
        such doc_id.
        """
        label = self._label()
        flabel = self._folder_label()
        fid = validate_identifier(folder, "folder")
        now = datetime.now(timezone.utc).isoformat()

        async def _write() -> bool:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        MATCH (n:`{label}` {{id: $doc_id}})
                        WHERE n.__delete_claim IS NULL
                        SET n.__membership_epoch =
                            coalesce(n.__membership_epoch, 0) + 1
                        MERGE (f:`{flabel}` {{id: $fid}})
                        MERGE (n)-[membership:MEMBER_OF]->(f)
                        SET membership.updated_at = coalesce(
                            membership.updated_at, $now)
                        RETURN n.id AS id
                        """,
                        doc_id=doc_id,
                        fid=fid,
                        now=now,
                    )
                    record = await result.single()
                    await result.consume()
                    return record is not None

        # Re-runnable: MERGE on both node and relation, and the return value is
        # recomputed from the doc's existence, not from what this attempt wrote.
        return await with_conflict_retry(
            f"MemgraphDocStatus.add_to_folder[{label}]", _write
        )

    async def remove_from_folder(self, doc_id: str, folder: str) -> int | None:
        """Remove a doc from a folder (delete the membership edge only).

        Returns the number of REMAINING memberships (ref-count) so the caller
        can physically delete only when it reaches 0. Returns None if the doc
        does not exist.
        """
        label = self._label()
        flabel = self._folder_label()
        fid = validate_identifier(folder, "folder")

        async def _write() -> int | None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        MATCH (n:`{label}` {{id: $doc_id}})
                        WHERE n.__delete_claim IS NULL
                        SET n.__membership_epoch =
                            coalesce(n.__membership_epoch, 0) + 1
                        WITH n
                        OPTIONAL MATCH (n)-[r:MEMBER_OF]->(:`{flabel}` {{id: $fid}})
                        DELETE r
                        WITH n
                        OPTIONAL MATCH (n)-[rem:MEMBER_OF]->()
                        // n.id is a grouping key: zero input rows (doc not found)
                        // → zero output rows → None. Without it, count() would
                        // return a single 0-row and mask a missing document.
                        RETURN n.id AS id, count(rem) AS remaining
                        """,
                        doc_id=doc_id,
                        fid=fid,
                    )
                    record = await result.single()
                    await result.consume()
                    return record["remaining"] if record else None

        # Re-runnable: deleting an already-deleted edge is a no-op, and the
        # ref-count is recomputed fresh on each attempt rather than decremented,
        # so a retry returns the true current count instead of an off-by-one.
        return await with_conflict_retry(
            f"MemgraphDocStatus.remove_from_folder[{label}]", _write
        )

    async def claim_last_membership_delete(
        self, doc_id: str, folder: str, claim: str
    ) -> bool:
        """Claim a document for a last-membership physical delete.

        The claim and the exact-one-membership check happen in one Memgraph
        transaction. Every membership writer also updates
        ``__membership_epoch`` on the DocStatus node, creating a write/write
        conflict when two workers race from the same snapshot. Once committed,
        ``__delete_claim`` makes later membership writers fail closed.
        """
        label = self._label()
        flabel = self._folder_label()
        fid = validate_identifier(folder, "folder")
        if not claim:
            raise ValueError("delete claim must not be empty")

        async def _write() -> bool:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        MATCH (n:`{label}` {{id: $doc_id}})
                        WHERE n.__delete_claim IS NULL
                        OPTIONAL MATCH (n)-[:MEMBER_OF]->(f:`{flabel}`)
                        WITH n, collect(f.id) AS folders
                        WHERE size(folders) = 1 AND folders[0] = $fid
                        SET n.__membership_epoch =
                                coalesce(n.__membership_epoch, 0) + 1,
                            n.__delete_claim = $claim
                        RETURN n.id AS id
                        """,
                        doc_id=doc_id,
                        fid=fid,
                        claim=claim,
                    )
                    record = await result.single()
                    await result.consume()
                    return record is not None

        # ⚠️ THE path that makes _retry's narrow predicate load-bearing. This is a
        # compare-and-set: `WHERE __delete_claim IS NULL ... SET __delete_claim`.
        # Retrying is correct ONLY because Memgraph *aborts* a conflicting
        # transaction, so the losing attempt committed nothing and the guard is
        # still NULL on re-run. If with_conflict_retry were ever widened to
        # transport/timeout errors, a committed-but-unobserved claim would be
        # re-read as someone else's, this would return False for a claim that
        # actually succeeded, and the document would stay claimed forever with no
        # caller left to release it. Do not widen the predicate.
        return await with_conflict_retry(
            f"MemgraphDocStatus.claim_last_membership_delete[{label}]", _write
        )

    async def release_delete_claim(self, doc_id: str, claim: str) -> None:
        """Release this caller's claim after a failed physical cascade."""
        label = self._label()
        if not claim:
            raise ValueError("delete claim must not be empty")

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        MATCH (n:`{label}` {{id: $doc_id}})
                        WHERE n.__delete_claim = $claim
                        SET n.__membership_epoch =
                            coalesce(n.__membership_epoch, 0) + 1
                        REMOVE n.__delete_claim
                        """,
                        doc_id=doc_id,
                        claim=claim,
                    )
                    await result.consume()

        # Re-runnable: releasing an already-released claim matches nothing, and
        # the guard is keyed on *this* caller's claim so a retry can never clear
        # a claim someone else took in the meantime.
        await with_conflict_retry(
            f"MemgraphDocStatus.release_delete_claim[{label}]", _write
        )

    async def get_folders_for_doc(self, doc_id: str) -> list[str] | None:
        """List the folders a document is a member of (ordered).

        Returns ``None`` when the document does not exist, vs ``[]`` for an
        existing document with no membership — so callers can 404 a bad id
        rather than silently masking it. The ``n.id`` grouping key makes the
        aggregation yield zero rows (not a single empty one) for a missing doc.
        """
        label = self._label()
        flabel = self._folder_label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{id: $doc_id}})
                OPTIONAL MATCH (n)-[:MEMBER_OF]->(f:`{flabel}`)
                RETURN n.id AS id, collect(f.id) AS fids
                """,
                doc_id=doc_id,
            )
            record = await result.single()
            await result.consume()
            if record is None:
                return None
            return sorted(record["fids"])

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
        flabel = self._folder_label()
        where_clause = ""
        params: dict[str, Any] = {}
        if folder:
            # Membership is the source of truth (initialize() backfills it from
            # the legacy property for old nodes, so this is safe). The single
            # `folder` property is kept written for rollback only, never read.
            where_clause = (
                f"WHERE EXISTS((n)-[:MEMBER_OF]->(:`{flabel}` {{id: $folder}}))"
            )
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

    async def get_folder_counts(self, folders: list[str]) -> dict[str, int]:
        """Return total document counts for several folders in one read.

        ``get_status_counts(folder=...)`` remains the detailed single-folder
        API used by LightRAG.  Folder selectors only need totals, so issuing
        that query once per catalog entry wastes a database round-trip for
        every folder.  This projection traverses the same membership source of
        truth once and fills missing folders with zero.
        """
        folder_ids = list(
            dict.fromkeys(validate_identifier(folder, "folder") for folder in folders)
        )
        if not folder_ids:
            return {}

        label = self._label()
        flabel = self._folder_label()
        counts = dict.fromkeys(folder_ids, 0)
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)-[:MEMBER_OF]->(f:`{flabel}`)
                WHERE f.id IN $folders
                RETURN f.id AS folder, count(DISTINCT n) AS cnt
                """,
                folders=folder_ids,
            )
            async for record in result:
                counts[record["folder"]] = int(record["cnt"] or 0)
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
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        # ``strict`` (LightRAG 1.5.5): complete-or-raise. This implementation
        # is a single server-side query — a transport failure raises in both
        # modes, and ``_deserialize_status`` is total over the shapes this
        # backend itself writes, so strict needs no separate path.
        del strict
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

    # ── LightRAG 1.5.5+ scheduling contract ───────────────────────────────
    # Four abstract methods added by the 1.5.5 DocStatusStorage base. The
    # reference semantics are lightrag/kg/json_doc_status_impl.py; this
    # implementation is server-side (indexed Cypher, O(page) memory).

    def _scheduling_record_from_props(
        self, doc_id: str, props: dict[str, Any], *, strict: bool
    ):
        """Raw props → DocSchedulingRecord; relaxed mode skips unusable rows."""
        try:
            status = DocStatus(str(props.get("status") or "").lower())
            created_at = props.get("created_at")
            updated_at = props.get("updated_at")
            if not isinstance(created_at, str) or not isinstance(updated_at, str):
                raise TypeError("created_at/updated_at must be strings")
            metadata = self._metadata_dict(props.get("metadata"))
            return DocSchedulingRecord(
                id=doc_id,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                file_path=props.get("file_path") or "no-file-path",
                track_id=props.get("track_id"),
                has_custom_chunk_journal=isinstance(
                    metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(
                "[MemgraphDocStatus:%s] unusable scheduling row %s: %s",
                self.workspace,
                doc_id,
                exc,
            )
            if strict:
                raise
            return None

    @staticmethod
    def _decode_page_cursor(opaque: str) -> tuple[str, str]:
        """Opaque cursor → (created_key, id); malformed = control-plane error."""
        try:
            key = json.loads(opaque)
            if (
                not isinstance(key, list)
                or len(key) != 2
                or not all(isinstance(part, str) for part in key)
            ):
                raise ValueError("page cursor must be a [created_at, id] pair")
        except (ValueError, TypeError) as exc:
            raise StorageControlPlaneError(
                f"Malformed doc-status page cursor for MemgraphDocStatusStorage: {exc}"
            ) from exc
        return key[0], key[1]

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: "CursorPosition" = CURSOR_START,
        strict: bool = False,
    ) -> "DocStatusPage":
        """Bounded keyset page, sorted server-side on (created_at ASC, id ASC).

        The sort key coalesces a missing ``created_at`` to ``""`` so such rows
        sort FIRST and are *consumed* by the cursor (never returned — the
        record constructor rejects them, per the base contract note); they can
        never strand a sweep behind a real-timestamp cursor. Consumed-position
        contract: every fetched row advances the cursor, returned or skipped;
        fewer rows than ``limit`` proves exhaustion.
        """
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        after: tuple[str, str] | None = None
        if isinstance(position, CursorAfter):
            after = self._decode_page_cursor(position.opaque)

        label = self._label()
        status_values = [self._status_value(s) for s in statuses]
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)
                WHERE n.status IN $statuses
                  AND (
                    $after_created IS NULL
                    OR coalesce(n.created_at, "") > $after_created
                    OR (coalesce(n.created_at, "") = $after_created
                        AND n.id > $after_id)
                  )
                RETURN n.id AS id,
                       coalesce(n.created_at, "") AS created_key,
                       properties(n) AS props
                ORDER BY created_key ASC, id ASC
                LIMIT $limit
                """,
                statuses=status_values,
                after_created=after[0] if after else None,
                after_id=after[1] if after else None,
                limit=limit,
            )
            rows = [
                (record["id"], record["created_key"], record["props"])
                async for record in result
            ]
            await result.consume()

        docs: dict[str, Any] = {}
        for doc_id, _created_key, props in rows:
            record = self._scheduling_record_from_props(doc_id, props, strict=strict)
            if record is None:
                continue  # relaxed skip is still consumed by the cursor
            docs[doc_id] = record
        if len(rows) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            last_id, last_created_key, _ = rows[-1]
            next_position = CursorAfter(json.dumps([last_created_key, last_id]))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def _fetch_props_by_ids(
        self, doc_ids: "Sequence[str]"
    ) -> list[tuple[str, dict[str, Any]]]:
        """One UNWIND batch read; a transport error fails the whole call,
        which is exactly the strict=True guarantee (no partial mapping)."""
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS did
                MATCH (n:`{label}` {{id: did}})
                RETURN did AS id, properties(n) AS props
                """,
                ids=list(doc_ids),
            )
            rows = [(record["id"], record["props"]) async for record in result]
            await result.consume()
        return rows

    async def get_docs_by_ids(
        self,
        doc_ids: "Sequence[str]",
        *,
        strict: bool = False,
    ) -> dict[str, "DocSchedulingRecord"]:
        """Batch strict read of scheduling records (base contract)."""
        if not doc_ids:
            return {}
        out: dict[str, Any] = {}
        for doc_id, props in await self._fetch_props_by_ids(doc_ids):
            record = self._scheduling_record_from_props(doc_id, props, strict=strict)
            if record is not None:
                out[doc_id] = record
        return out

    async def get_full_docs_by_ids(
        self,
        doc_ids: "Sequence[str]",
        *,
        strict: bool = False,
    ) -> dict[str, DocProcessingStatus]:
        """Batch hydration to full DocProcessingStatus (base contract).

        Reuses ``_deserialize_status`` — the same raw → DocProcessingStatus
        normalisation as ``get_docs_by_statuses``.
        """
        if not doc_ids:
            return {}
        out: dict[str, DocProcessingStatus] = {}
        for doc_id, props in await self._fetch_props_by_ids(doc_ids):
            try:
                out[doc_id] = self._deserialize_status(props)
            except (KeyError, TypeError, ValueError) as exc:
                logger.error(
                    "[MemgraphDocStatus:%s] unusable doc_status row hydrating %s: %s",
                    self.workspace,
                    doc_id,
                    exc,
                )
                if strict:
                    raise
        return out

    # Bounded candidate fetch for source resolution: file_path is indexed, so
    # candidates for one basename are few; the bound only guards pathological
    # collision sets, and hitting it is an ambiguity → control-plane error
    # (fail-closed — SourceAbsent would mint duplicate rows downstream).
    _SOURCE_RESOLUTION_SCAN_LIMIT = 200

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> "SourceResolution":
        """Typed, fail-closed source resolution (base contract).

        Primary = ``metadata.is_duplicate != true``; duplicate markers are
        pointer rows created by the dedup-share flow and never count. A query
        failure propagates (never mapped to Absent).
        """
        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{file_path: $key}})
                RETURN n.id AS id, properties(n) AS props
                ORDER BY n.id ASC
                LIMIT $limit
                """,
                key=canonical_source_key,
                limit=self._SOURCE_RESOLUTION_SCAN_LIMIT,
            )
            rows = [(record["id"], record["props"]) async for record in result]
            await result.consume()

        primaries: list[tuple[str, dict[str, Any]]] = []
        for doc_id, props in rows:
            metadata = self._metadata_dict(props.get("metadata"))
            if metadata.get("is_duplicate") is True:
                continue
            primaries.append((doc_id, props))
            if len(primaries) >= 2:
                break
        if len(primaries) < 2 and len(rows) >= self._SOURCE_RESOLUTION_SCAN_LIMIT:
            raise StorageControlPlaneError(
                f"Source resolution for '{canonical_source_key}' is ambiguous: "
                f"{len(rows)} candidate rows reached the scan bound "
                f"({self._SOURCE_RESOLUTION_SCAN_LIMIT}) without confirming "
                "the primary set"
            )
        if not primaries:
            return SourceAbsent()
        if len(primaries) == 1:
            doc_id, props = primaries[0]
            record = self._scheduling_record_from_props(doc_id, props, strict=True)
            return SourceUnique(doc_id=doc_id, doc=record)
        return SourceConflict(
            candidate_count=None,
            sample_doc_ids=tuple(sorted(doc_id for doc_id, _ in primaries)),
        )

    # ── LightRAG 1.5.5+ strict capabilities ───────────────────────────────
    # Probed by lightrag.utils_pipeline.describe_doc_status_capabilities();
    # a missing one degrades the pipeline fail-closed (admission 503, 501 on
    # the conflict endpoints, stale FAILED stubs kept forever).

    # Strict point reads: one indexed lookup — a miss IS a confirmed absence
    # and any transport/server failure raises (never mapped to None).
    supports_strict_point_reads = True

    _CONFLICT_SAMPLE_CAP = 32

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        """Point read, complete-or-raise (base contract)."""
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN properties(n) AS props",
                id=id,
            )
            record = await result.single()
            await result.consume()
        if not record:
            return None
        props = self._deserialize_props(record["props"])
        props["id"] = id
        return props

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        """Fail-closed status count (server-side aggregate)."""
        del strict  # single aggregate query: an error raises in both modes
        if not statuses:
            return 0
        label = self._label()
        status_values = [self._status_value(s) for s in statuses]
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)
                WHERE n.status IN $statuses
                RETURN count(n) AS c
                """,
                statuses=status_values,
            )
            record = await result.single()
            await result.consume()
        return int(record["c"]) if record else 0

    @staticmethod
    def _conflict_fingerprint(sorted_doc_ids: list[str]) -> str:
        """Deterministic digest over candidate doc IDs in stable sort order
        (same construction as the upstream reference backend)."""
        digest = hashlib.sha256()
        for doc_id in sorted_doc_ids:
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    async def _primary_candidate_ids(self, canonical_source_key: str) -> list[str]:
        """Sorted primary (non-duplicate) doc ids for a canonical key.

        Bounded fetch, fail-closed on the bound (same rationale as
        ``resolve_doc_source_strict``: an ambiguous set must never look
        resolved)."""
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}` {{file_path: $key}})
                RETURN n.id AS id, n.metadata AS metadata
                ORDER BY n.id ASC
                LIMIT $limit
                """,
                key=canonical_source_key,
                limit=self._SOURCE_RESOLUTION_SCAN_LIMIT,
            )
            rows = [(record["id"], record["metadata"]) async for record in result]
            await result.consume()
        if len(rows) >= self._SOURCE_RESOLUTION_SCAN_LIMIT:
            raise StorageControlPlaneError(
                f"Primary-candidate read for '{canonical_source_key}' reached "
                f"the scan bound ({self._SOURCE_RESOLUTION_SCAN_LIMIT}); the "
                "candidate set is ambiguous"
            )
        return sorted(
            doc_id
            for doc_id, metadata in rows
            if self._metadata_dict(metadata).get("is_duplicate") is not True
        )

    async def _sample_primary_candidates(
        self, canonical_source_key: str
    ) -> tuple[list[str], bool]:
        """Primary (non-duplicate) ids for one key, paged past pointer rows.

        Pages the key's rows in bounded windows (id-keyset) and keeps
        collecting until the primary sample reaches ``_CONFLICT_SAMPLE_CAP``
        or the candidate set is exhausted. Returns ``(sorted_primaries,
        exhausted)`` — memory stays O(window + cap) however many pointer
        rows precede the primaries, and an all-pointer window can never
        masquerade as "no conflict" (that verdict requires exhaustion).
        """
        primaries: list[str] = []
        after_id: str | None = None
        label = self._label()
        while True:
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"""
                    MATCH (n:`{label}` {{file_path: $key}})
                    WHERE $after_id IS NULL OR n.id > $after_id
                    RETURN n.id AS id, n.metadata AS metadata
                    ORDER BY id ASC
                    LIMIT $limit
                    """,
                    key=canonical_source_key,
                    after_id=after_id,
                    limit=self._SOURCE_RESOLUTION_SCAN_LIMIT,
                )
                rows = [(record["id"], record["metadata"]) async for record in result]
                await result.consume()
            for doc_id, metadata in rows:
                if self._metadata_dict(metadata).get("is_duplicate") is not True:
                    primaries.append(doc_id)
            if len(rows) < self._SOURCE_RESOLUTION_SCAN_LIMIT:
                return sorted(primaries), True
            if len(primaries) >= self._CONFLICT_SAMPLE_CAP:
                return sorted(primaries[: self._CONFLICT_SAMPLE_CAP]), False
            after_id = rows[-1][0]

    @staticmethod
    def _decode_conflict_cursor(opaque: str) -> str:
        try:
            key = json.loads(opaque)
            if not isinstance(key, str):
                raise ValueError("conflict cursor must be a string")
        except (ValueError, TypeError) as exc:
            raise StorageControlPlaneError(
                "Malformed source-conflict cursor for "
                f"MemgraphDocStatusStorage: {exc}"
            ) from exc
        return key

    async def list_source_conflicts_page(
        self,
        *,
        limit: int,
        position: "CursorPosition" = CURSOR_START,
    ) -> "SourceConflictPage":
        """Page canonical source keys with >1 primary candidate (base contract).

        Bounded end to end: the grouping query is COUNT-only (no ``collect``
        of row payloads — one high-cardinality source must not materialize
        its whole candidate set), and the per-key primary filter
        (``is_duplicate`` in the JSON-encoded metadata) runs through
        :meth:`_sample_primary_candidates`, which PAGES the key's rows in
        bounded windows until the sample cap is met or the set is exhausted
        — review blocker: a window full of pointer rows must neither hide a
        genuine conflict behind it nor fail the whole listing.
        ``candidate_count`` is exact when the set was exhausted, ``None``
        when sampling stopped at the cap. Consumed-position: every fetched
        multi-row key advances the cursor, kept or (exhaustion-proven)
        filtered; fewer fetched keys than ``limit`` = exhaustion.
        """
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if position is CURSOR_END:
            return SourceConflictPage(conflicts=(), next_position=CURSOR_END)
        last_key: str | None = None
        if isinstance(position, CursorAfter):
            last_key = self._decode_conflict_cursor(position.opaque)

        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (n:`{label}`)
                WHERE n.file_path IS NOT NULL
                  AND NOT n.file_path IN ["", "unknown_source", "no-file-path"]
                  AND ($after IS NULL OR n.file_path > $after)
                WITH n.file_path AS key, count(n) AS total
                WHERE total >= 2
                RETURN key
                ORDER BY key ASC
                LIMIT $limit
                """,
                after=last_key,
                limit=limit,
            )
            keys = [record["key"] async for record in result]
            await result.consume()

        conflicts: list[Any] = []
        for key in keys:
            primaries, exhausted = await self._sample_primary_candidates(key)
            if len(primaries) < 2:
                # Only reachable when the key's rows were EXHAUSTED (the
                # sampler never stops early below the cap), so this skip is
                # proven, never a truncation artifact.
                continue
            conflicts.append(
                SourceConflictSummary(
                    canonical_source_key=key,
                    candidate_count=len(primaries) if exhausted else None,
                    sample_doc_ids=tuple(primaries[: self._CONFLICT_SAMPLE_CAP]),
                )
            )
        if len(keys) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            next_position = CursorAfter(json.dumps(keys[-1], ensure_ascii=False))
        return SourceConflictPage(
            conflicts=tuple(conflicts), next_position=next_position
        )

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> "SourceConflictRepairResult":
        """Demote all-but-one primary to duplicate, CAS-guarded (base contract).

        No storage-wide lock exists here (see the base note): the strict
        re-read + count/fingerprint CAS detects a concurrently changed
        candidate set and refuses with
        :class:`SourceConflictRepairCASError` instead of overwriting it.
        Content is never deleted — demoted rows only gain
        ``metadata.is_duplicate=true`` + ``original_doc_id``.
        """
        candidates = await self._primary_candidate_ids(canonical_source_key)
        count = len(candidates)
        fingerprint = self._conflict_fingerprint(candidates)
        if primary_doc_id not in candidates:
            raise ValueError(
                f"primary_doc_id {primary_doc_id!r} is not a current primary "
                f"candidate for {canonical_source_key!r}"
            )
        demoted = [d for d in candidates if d != primary_doc_id]
        if dry_run:
            return SourceConflictRepairResult(
                canonical_source_key=canonical_source_key,
                primary_doc_id=primary_doc_id,
                candidate_count=count,
                fingerprint=fingerprint,
                demoted_sample_doc_ids=tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
                committed=False,
            )
        if (
            count != expected_candidate_count
            or fingerprint != expected_candidate_fingerprint
        ):
            raise SourceConflictRepairCASError(
                f"[{self.workspace}] source-conflict repair CAS failed for "
                f"{canonical_source_key!r}: candidate set changed "
                f"(count {count} vs {expected_candidate_count})"
            )

        label = self._label()
        entries = []
        for doc_id in demoted:
            row = await self.get_by_id_strict(doc_id)
            metadata = dict((row or {}).get("metadata") or {})
            metadata["is_duplicate"] = True
            metadata["original_doc_id"] = primary_doc_id
            entries.append(
                {"id": doc_id, "metadata": json.dumps(metadata, default=str)}
            )

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        UNWIND $entries AS e
                        MATCH (n:`{label}` {{id: e.id}})
                        SET n.metadata = e.metadata
                        """,
                        entries=entries,
                    )
                    await result.consume()

        await with_conflict_retry(
            f"MemgraphDocStatus.repair_source_conflict[{label}]", _write
        )
        return SourceConflictRepairResult(
            canonical_source_key=canonical_source_key,
            primary_doc_id=primary_doc_id,
            candidate_count=count,
            fingerprint=fingerprint,
            demoted_sample_doc_ids=tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
            committed=True,
        )

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        # INTENTIONALLY workspace-global, NOT folder-scoped. This backs
        # LightRAG's post-upload polling (/documents/track_status/{id}); folder-
        # scoping it could break polling when the X-Twin-Folder header is not
        # propagated on every poll, or when a duplicate upload shared an existing
        # doc into the active folder under a different track. A track_id is an
        # opaque per-upload handle, not a cloisonnement surface — downstream
        # writes (bulk-retag, etc.) still enforce active-folder membership.
        # If a future tier needs per-folder track isolation, design it with a
        # dedicated route + test (follow-up ticket), do not silently scope here.
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
            # Membership is authoritative (backfilled at initialize); the legacy
            # `folder` property is written for rollback only, never read.
            flabel = self._folder_label()
            filters.append(f"EXISTS((n)-[:MEMBER_OF]->(:`{flabel}` {{id: $folder}}))")
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
                if folder:
                    # A source can be shared physically while remaining a
                    # folder-local upload.  Project the membership timestamp
                    # so a fresh add to folder B never inherits folder A's
                    # historical upload date.  max() also collapses corrupt
                    # legacy duplicate MEMBER_OF paths.
                    folder_sort = (
                        "coalesce(folder_updated_at, n.updated_at)"
                        if sort_field == "updated_at"
                        else f"n.{sort_field}"
                    )
                    result = await session.run(
                        f"""
                        MATCH (n:`{label}`) {where_clause}
                        OPTIONAL MATCH
                          (n)-[membership:MEMBER_OF]->
                          (:`{self._folder_label()}` {{id: $folder}})
                        WITH n, max(membership.updated_at) AS folder_updated_at
                        RETURN n.id AS id, properties(n) AS props,
                               folder_updated_at
                        ORDER BY {folder_sort} {order}
                        SKIP $skip LIMIT $limit
                        """,
                        **params,
                        skip=skip,
                        limit=page_size,
                    )
                else:
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
                    props = dict(record["props"])
                    if folder and record["folder_updated_at"]:
                        props["updated_at"] = record["folder_updated_at"]
                    out.append((record["id"], self._deserialize_status(props)))
                await result.consume()
                return out

        docs, total = await asyncio.gather(_fetch(), _count())
        return docs, total

    async def get_all_status_counts(self, folder: str | None = None) -> dict[str, int]:
        return await self.get_status_counts(folder=folder)

    # Duplicate-lookup getters: file_path / basename / content_hash are NOT
    # unique in the store (duplicate uploads, legacy data), so a bare
    # `result.single()` would pick a driver-arbitrary row (audit 2026-07-02
    # addendum, finding D). During an upload, filename-only checks are scoped
    # to the active folder and never mutate membership: equal names do not
    # prove equal content. Content-hash checks first prefer an in-folder match,
    # then share the oldest global match only after hash equality is known.

    # Page size for the oldest-first candidate sweep when an exclusion
    # applies. Pointer rows for one hash naming the same excluded doc are a
    # handful in practice, but the sweep PAGES past a full window of excluded
    # rows rather than declaring absence — a swallowed genuine holder beyond
    # the window would admit duplicate content as new.
    _DUPLICATE_EXCLUSION_SCAN_LIMIT = 25

    def _record_excluded(
        self, doc_id: str, props: dict[str, Any], exclude: str
    ) -> bool:
        """True when the row IS the excluded doc or merely points at it
        (``is_duplicate`` naming it as ``original_doc_id``) — LightRAG 1.5.5
        base contract: the duplicate check can neither be answered with the
        row being processed nor with a record of that row."""
        if doc_id == exclude:
            return True
        metadata = self._metadata_dict(props.get("metadata"))
        return (
            metadata.get("is_duplicate") is True
            and str(metadata.get("original_doc_id") or "") == exclude
        )

    async def _oldest_duplicate_record(
        self,
        property_name: str,
        value: str,
        *,
        folder: str | None = None,
        exclude_doc_id: str | None = None,
    ) -> Any | None:
        """Return the oldest matching record, optionally inside one folder,
        optionally excluding one doc id and its duplicate pointer rows."""
        if property_name not in {"file_path", "content_hash"}:
            raise ValueError(f"Unsupported duplicate property: {property_name}")

        label = self._label()
        node = f"(n:`{label}` {{{property_name}: $value}})"
        params: dict[str, Any] = {"value": value}
        if folder:
            folder_label = self._folder_label()
            params["active_folder"] = folder
            match = (
                f"MATCH {node}"
                f"-[:MEMBER_OF]->(:`{folder_label}` {{id: $active_folder}})"
            )
        else:
            match = f"MATCH {node}"

        if exclude_doc_id is None:
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"""
                    {match}
                    WITH DISTINCT n
                    ORDER BY n.created_at ASC, n.id ASC
                    LIMIT $limit
                    RETURN n.id AS id, properties(n) AS props
                    """,
                    **params,
                    limit=1,
                )
                record = await result.single()
                await result.consume()
                return record

        # Exclusion sweep: bounded keyset pages over (created_at, id) until a
        # non-excluded survivor appears or a short page proves exhaustion —
        # never "confirmed absent" off one window full of excluded rows.
        page_size = self._DUPLICATE_EXCLUSION_SCAN_LIMIT
        after_created: str | None = None
        after_id: str | None = None
        while True:
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"""
                    {match}
                    WITH DISTINCT n
                    WHERE $after_created IS NULL
                       OR coalesce(n.created_at, "") > $after_created
                       OR (coalesce(n.created_at, "") = $after_created
                           AND n.id > $after_id)
                    RETURN n.id AS id,
                           coalesce(n.created_at, "") AS created_key,
                           properties(n) AS props
                    ORDER BY created_key ASC, id ASC
                    LIMIT $limit
                    """,
                    **params,
                    limit=page_size,
                    after_created=after_created,
                    after_id=after_id,
                )
                records = [record async for record in result]
                await result.consume()
            for record in records:
                if not self._record_excluded(
                    record["id"], record["props"], exclude_doc_id
                ):
                    return record
            if len(records) < page_size:
                return None  # exhausted the candidate set: confirmed absent
            after_created = records[-1]["created_key"]
            after_id = records[-1]["id"]

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        record = await self._oldest_duplicate_record(
            "file_path",
            file_path,
            folder=get_active_duplicate_share_folder(),
        )
        if not record:
            return None
        props = self._deserialize_props(record["props"])
        props["id"] = record["id"]
        return props

    async def get_docs_by_file_paths(
        self, file_paths: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Resolve source paths in one query, retaining oldest-duplicate wins."""
        unique = list(dict.fromkeys(path for path in file_paths if path))
        if not unique:
            return {}

        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $file_paths AS file_path
                OPTIONAL MATCH (n:`{label}` {{file_path: file_path}})
                WITH file_path, n
                ORDER BY file_path, n.created_at ASC, n.id ASC
                WITH file_path, collect(n)[0] AS n
                RETURN file_path, n.id AS id, properties(n) AS props
                """,
                file_paths=unique,
            )
            records = [dict(record) async for record in result]
            await result.consume()

        out: dict[str, dict[str, Any]] = {}
        for record in records:
            if not record.get("id") or not isinstance(record.get("props"), dict):
                continue
            props = self._deserialize_props(record["props"])
            props["id"] = record["id"]
            out[record["file_path"]] = props
        return out

    async def _share_confirmed_content_duplicate(
        self, doc_id: str, active_folder: str
    ) -> bool:
        """Share a globally matched document after content equality is known."""
        shared = await self.add_to_folder(doc_id, active_folder)
        if shared:
            logger.info(
                "[MemgraphDocStatus:%s] shared content-identical doc %s into "
                "folder %s; suppressing the cross-folder duplicate match",
                self.workspace,
                doc_id,
                active_folder,
            )
        else:
            logger.warning(
                "[MemgraphDocStatus:%s] content-identical doc %s disappeared "
                "or is delete-claimed before it could be shared into folder "
                "%s; preserving the duplicate result",
                self.workspace,
                doc_id,
                active_folder,
            )
        return shared

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        record = await self._oldest_duplicate_record(
            "file_path",
            basename,
            folder=get_active_duplicate_share_folder(),
        )
        if not record:
            return None
        return record["id"], self._deserialize_props(
            record["props"],
            include_default_folder=False,
        )

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        # ``exclude_doc_id`` (LightRAG 1.5.5): the duplicate check must not be
        # answered with the row being processed nor with a pointer row of it.
        if not content_hash:
            return None
        active_folder = get_active_duplicate_share_folder()
        record = await self._oldest_duplicate_record(
            "content_hash",
            content_hash,
            folder=active_folder,
            exclude_doc_id=exclude_doc_id,
        )
        if record:
            return record["id"], self._deserialize_props(
                record["props"],
                include_default_folder=False,
            )

        if active_folder:
            record = await self._oldest_duplicate_record(
                "content_hash",
                content_hash,
                exclude_doc_id=exclude_doc_id,
            )
            if record:
                # Share the original into the active folder, then STILL
                # return it. Returning None after a successful share was the
                # 1.4.x contract ("shared, nothing visible to create") — on
                # 1.5.5 the post-parse dedup check reads None as "no
                # duplicate exists" and fully re-ingests the new doc. The
                # caller marks the current doc is_duplicate/original_doc_id,
                # which our upsert seam converts into the membership share
                # (idempotent with the eager share below).
                await self._share_confirmed_content_duplicate(
                    record["id"], active_folder
                )
                return record["id"], self._deserialize_props(
                    record["props"],
                    include_default_folder=False,
                )
        return None

    async def drop(self) -> dict[str, str]:
        label = self._label()

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                    await result.consume()

        # Re-runnable: a second pass matches an empty label set.
        await with_conflict_retry(f"MemgraphDocStatus.drop[{label}]", _write)
        return {"status": "success", "message": f"DocStatus {label} dropped"}
