"""Bounded LightRAG 1.5 scheduling and deduplication over native Cypher reads.

New upstream types are imported inside methods so LightRAG 1.4 stays usable.
All control-plane reads propagate database and decode failures.
"""

import json
from collections.abc import Sequence
from typing import Any

from lightrag.base import DocStatus
from lightrag.utils import logger

from . import _pool


class DocStatusSchedulingMixin:
    @staticmethod
    def _decode_position(position):
        from lightrag.base import CURSOR_START, CursorAfter
        from lightrag.exceptions import StorageControlPlaneError

        if position is None or position is CURSOR_START:
            return None
        if isinstance(position, CursorAfter):
            try:
                key = json.loads(position.opaque)
                if (
                    isinstance(key, list)
                    and len(key) == 2
                    and all(isinstance(x, str) for x in key)
                ):
                    return key
            except (ValueError, TypeError):
                pass
        raise StorageControlPlaneError("Malformed Memgraph scheduling cursor")

    async def _ordered_status_rows(
        self, where, params, *, limit, after=None, projection="properties(n)"
    ):
        # A legacy NULL timestamp sorts first and remains reachable on resume.
        key = "coalesce(toString(n.created_at), '')"
        if after is not None:
            where += (
                f" AND ({key} > $created OR ({key} = $created AND n.id > $last_id))"
            )
            params = {**params, "created": after[0], "last_id": after[1]}
        async with self._read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label()}`) WHERE {where} "
                f"RETURN {projection} AS props, {key} AS created, n.id AS id "
                f"ORDER BY created ASC, id ASC LIMIT $limit",
                **params,
                limit=limit,
            )
            rows = [
                (r["id"], r["props"], [r["created"], r["id"]]) async for r in result
            ]
            await result.consume()
            return rows

    @staticmethod
    def _scheduling_record(doc_id, props):
        from lightrag.base import DocSchedulingRecord

        raw_status = props["status"]
        status = (
            raw_status
            if isinstance(raw_status, DocStatus)
            else DocStatus(raw_status.lower())
        )
        created = props["created_at"]
        updated = props.get("updated_at", created)
        if not isinstance(created, str) or not isinstance(updated, str):
            raise ValueError("Scheduling timestamps must be strings")
        journal = props.get("_has_custom_chunk_journal")
        if journal is None:
            metadata = props.get("metadata") or "{}"
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if not isinstance(metadata, dict):
                raise ValueError("Document metadata must be an object")
            journal = isinstance(metadata.get("custom_chunk_patch"), dict)
        return DocSchedulingRecord(
            id=doc_id,
            status=status,
            created_at=created,
            updated_at=updated,
            file_path=props.get("file_path"),
            track_id=props.get("track_id"),
            has_custom_chunk_journal=bool(journal),
        )

    # Project only scheduler fields. New rows do not transfer metadata/chunk lists.
    _scheduling_projection = (
        "{status: n.status, created_at: n.created_at, updated_at: n.updated_at, "
        "file_path: n.file_path, track_id: n.track_id, "
        "_has_custom_chunk_journal: n._has_custom_chunk_journal, "
        "metadata: CASE WHEN n._has_custom_chunk_journal IS NULL THEN n.metadata ELSE null END}"
    )

    async def get_docs_by_statuses_page(
        self, statuses, *, limit, position=None, strict=False
    ):
        from lightrag.base import CURSOR_END, CursorAfter, DocStatusPage

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("page limit must be a positive integer")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        after = self._decode_position(position)
        rows = await self._ordered_status_rows(
            "n.status IN $statuses",
            {"statuses": [s.value for s in statuses]},
            limit=limit,
            after=after,
            projection=self._scheduling_projection,
        )
        docs = {}
        for doc_id, props, _ in rows:
            try:
                docs[doc_id] = self._scheduling_record(doc_id, props)
            except (ValueError, TypeError, KeyError, AttributeError):
                if strict:
                    raise
                logger.warning("Skipping unusable scheduling row %s", doc_id)
        # A skipped row is consumed too. Never terminate just because docs is empty.
        next_position = (
            CURSOR_END if len(rows) < limit else CursorAfter(json.dumps(rows[-1][2]))
        )
        return DocStatusPage(docs=docs, next_position=next_position)

    async def _batch_status_rows(
        self, doc_ids: Sequence[str], *, projection="properties(n)"
    ):
        # Chunk caller-provided batches as well as server-side scheduling sweeps.
        ids = list(dict.fromkeys(doc_ids))
        for offset in range(0, len(ids), 256):
            async with self._read_session() as session:
                result = await session.run(
                    f"UNWIND $ids AS target_id MATCH (n:`{self._label()}` {{id: target_id}}) "
                    f"RETURN n.id AS id, {projection} AS props",
                    ids=ids[offset : offset + 256],
                )
                rows = [(r["id"], r["props"]) async for r in result]
                await result.consume()
            for row in rows:
                yield row

    async def get_docs_by_ids(self, doc_ids: Sequence[str], *, strict=False):
        docs = {}
        async for doc_id, props in self._batch_status_rows(
            doc_ids, projection=self._scheduling_projection
        ):
            try:
                docs[doc_id] = self._scheduling_record(doc_id, props)
            except (ValueError, TypeError, KeyError, AttributeError):
                if strict:
                    raise
                logger.warning("Skipping unusable scheduling row %s", doc_id)
        return docs

    async def get_full_docs_by_ids(self, doc_ids: Sequence[str], *, strict=False):
        docs = {}
        async for doc_id, props in self._batch_status_rows(doc_ids):
            try:
                docs[doc_id] = self._deserialize_status(props, strict=strict)
            except (ValueError, TypeError, KeyError):
                if strict:
                    raise
                logger.warning("Skipping unusable document status %s", doc_id)
        return docs

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        if not statuses:
            return 0
        async with self._read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label()}`) WHERE n.status IN $statuses RETURN count(n) AS count",
                statuses=[s.value for s in statuses],
            )
            record = await result.single()
            await result.consume()
            return record["count"]

    async def update_doc_status_fields(
        self, doc_id: str, fields: dict[str, Any], *, missing_ok=False
    ):
        from lightrag.exceptions import StorageRecordNotFoundError

        if "created_at" in fields:
            raise ValueError("created_at is an immutable scheduling sort key")
        if "id" in fields or any(k.startswith("_") for k in fields):
            raise ValueError("Internal document status fields cannot be updated")
        props = self._encode_status_fields(fields)
        async with _pool.acquire_write_slot():
            async with self._write_session() as session:
                result = await session.run(
                    f"MATCH (n:`{self._label()}` {{id: $id}}) SET n += $props RETURN n.id AS id",
                    id=doc_id,
                    props=props,
                )
                record = await result.single()
                await result.consume()
        if record is None and not missing_ok:
            raise StorageRecordNotFoundError(doc_id)

    @staticmethod
    def _encode_status_fields(fields):
        props = dict(fields)
        if "metadata" in props:
            metadata = props["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            props["_has_custom_chunk_journal"] = isinstance(
                metadata, dict
            ) and isinstance(metadata.get("custom_chunk_patch"), dict)
        for key, value in props.items():
            if isinstance(value, (dict, list)):
                props[key] = json.dumps(value, default=str)
            elif hasattr(value, "value"):
                props[key] = value.value
        return props

    async def _source_candidates(self, where, params):
        after = None
        while True:
            rows = await self._ordered_status_rows(
                where, params, limit=128, after=after
            )
            for doc_id, props, _ in rows:
                yield doc_id, self._deserialize_props(props, strict=True)
            if len(rows) < 128:
                break
            after = rows[-1][2]

    @staticmethod
    def _is_duplicate(props):
        metadata = props.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Document metadata must be an object")
        return bool(metadata.get("is_duplicate"))

    async def get_doc_by_file_basename(self, basename: str):
        if not basename or basename == "unknown_source":
            return None
        async for doc_id, props in self._source_candidates(
            "n.file_path = $source", {"source": basename}
        ):
            if not self._is_duplicate(props):
                return doc_id, props
        return None

    async def get_doc_by_content_hash(self, content_hash: str, *, exclude_doc_id=None):
        if not content_hash:
            return None
        async for doc_id, props in self._source_candidates(
            "n.content_hash = $hash", {"hash": content_hash}
        ):
            if doc_id == exclude_doc_id:
                continue
            metadata = props.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError("Document metadata must be an object")
            if (
                exclude_doc_id is not None
                and metadata.get("is_duplicate")
                and metadata.get("original_doc_id") == exclude_doc_id
            ):
                continue
            return doc_id, props
        return None

    async def resolve_doc_source_strict(self, canonical_source_key: str):
        from lightrag.base import SourceAbsent, SourceConflict, SourceUnique

        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()
        candidates = []
        async for doc_id, props in self._source_candidates(
            "n.file_path = $source", {"source": canonical_source_key}
        ):
            if not self._is_duplicate(props):
                candidates.append((doc_id, props))
                if len(candidates) == 2:
                    return SourceConflict(
                        candidate_count=None,
                        sample_doc_ids=tuple(sorted(x[0] for x in candidates)),
                    )
        if candidates:
            doc_id, props = candidates[0]
            return SourceUnique(
                doc_id=doc_id, doc=self._scheduling_record(doc_id, props)
            )
        return SourceAbsent()
