"""Document-scoped, auditable source-link persistence.

Links are provenance attached to a document.  They deliberately do not alter
LightRAG's ``file_path`` source identity and are never fetched by the server.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Iterable
from typing import Any

from .. import _pool
from .._constants import validate_identifier
from .._retry import with_conflict_retry

logger = logging.getLogger(__name__)


def _public_source_link(props: dict[str, Any]) -> dict[str, Any]:
    """Return one backend-independent wire shape, including nullable fields."""
    return {
        "id": str(props.get("id") or ""),
        "doc_id": str(props.get("doc_id") or ""),
        "url": str(props.get("url") or ""),
        "label": props.get("label"),
        "created_by": str(props.get("created_by") or ""),
        "created_at": str(props.get("created_at") or ""),
        "updated_by": str(props.get("updated_by") or ""),
        "updated_at": str(props.get("updated_at") or ""),
        "version": int(props.get("version") or 1),
        "deleted": bool(props.get("deleted", False)),
        "deleted_by": props.get("deleted_by"),
        "deleted_at": props.get("deleted_at"),
    }


class SourceLinkNotFound(LookupError):
    pass


class SourceLinkVersionConflict(RuntimeError):
    pass


class InMemorySourceLinkStore:
    """Process-local backend used by seed/demo stores and unit tests."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def list_for_document(self, doc_id: str) -> list[dict[str, Any]]:
        rows = [
            _public_source_link(row)
            for (stored_doc_id, _), row in self._items.items()
            if stored_doc_id == doc_id and not row.get("deleted")
        ]
        return sorted(rows, key=lambda row: (row["created_at"], row["id"]))

    async def list_for_documents(
        self, doc_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        unique = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        return {doc_id: await self.list_for_document(doc_id) for doc_id in unique}

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = _public_source_link(row)
        key = (normalized["doc_id"], normalized["id"])
        async with self._lock:
            if key in self._items:
                raise SourceLinkVersionConflict("source link id already exists")
            self._items[key] = copy.deepcopy(normalized)
            return _public_source_link(normalized)

    async def update(
        self,
        doc_id: str,
        link_id: str,
        *,
        expected_version: int,
        url: str,
        label: str | None,
        actor: str,
        updated_at: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        key = (doc_id, link_id)
        async with self._lock:
            current = self._items.get(key)
            if current is None or current.get("deleted"):
                raise SourceLinkNotFound(link_id)
            if current["version"] != expected_version:
                raise SourceLinkVersionConflict(link_id)
            before = copy.deepcopy(current)
            current.update(
                url=url,
                label=label,
                updated_by=actor,
                updated_at=updated_at,
                version=expected_version + 1,
            )
            return _public_source_link(before), _public_source_link(current)

    async def delete(
        self,
        doc_id: str,
        link_id: str,
        *,
        expected_version: int,
        actor: str,
        deleted_at: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        key = (doc_id, link_id)
        async with self._lock:
            current = self._items.get(key)
            if current is None or current.get("deleted"):
                raise SourceLinkNotFound(link_id)
            if current["version"] != expected_version:
                raise SourceLinkVersionConflict(link_id)
            before = copy.deepcopy(current)
            current.update(
                deleted=True,
                deleted_by=actor,
                deleted_at=deleted_at,
                updated_by=actor,
                updated_at=deleted_at,
                version=expected_version + 1,
            )
            return _public_source_link(before), _public_source_link(current)


class MemgraphSourceLinkStore:
    """Persistent tombstone-backed source-link store."""

    def __init__(self, workspace: str) -> None:
        self.workspace = validate_identifier(workspace, "workspace")

    def _label(self) -> str:
        return f"TwinSourceLink_{self.workspace}"

    async def initialize(self) -> None:
        """Ensure lookup indexes under the shared write throttle."""
        label = self._label()
        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            for prop in ("id", "doc_id"):
                try:
                    result = await session.run(f"CREATE INDEX ON :`{label}`({prop})")
                    await result.consume()
                    logger.info(
                        "[SourceLinkStore] Index on :%s(%s) ensured",
                        label,
                        prop,
                    )
                except Exception as exc:
                    if "already exists" not in str(exc).lower():
                        raise
                    logger.debug(
                        "[SourceLinkStore] Index already exists on :%s(%s)",
                        label,
                        prop,
                    )

    @staticmethod
    def _public_props(props: dict[str, Any]) -> dict[str, Any]:
        return _public_source_link(props)

    async def list_for_document(self, doc_id: str) -> list[dict[str, Any]]:
        return (await self.list_for_documents([doc_id])).get(doc_id, [])

    async def list_for_documents(
        self, doc_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        unique = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        output = {doc_id: [] for doc_id in unique}
        if not unique:
            return output
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $doc_ids AS doc_id
                OPTIONAL MATCH (link:`{self._label()}` {{doc_id: doc_id}})
                WHERE coalesce(link.deleted, false) = false
                RETURN doc_id, properties(link) AS props
                ORDER BY doc_id, link.created_at, link.id
                """,
                doc_ids=unique,
            )
            async for record in result:
                props = record.get("props")
                if props:
                    output[record["doc_id"]].append(self._public_props(props))
            await result.consume()
        return output

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = self._public_props(row)

        async def _write():
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        CREATE (link:`{self._label()}`)
                        SET link = $props
                        RETURN properties(link) AS props
                        """,
                        props=normalized,
                    )
                    record = await result.single()
                    await result.consume()
                    return record

        record = await with_conflict_retry(
            f"source_links.create[{normalized['id']}]", _write
        )
        return self._public_props(record["props"])

    async def _current(self, doc_id: str, link_id: str) -> dict[str, Any] | None:
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (link:`{self._label()}` {{doc_id: $doc_id, id: $link_id}})
                RETURN properties(link) AS props
                """,
                doc_id=doc_id,
                link_id=link_id,
            )
            record = await result.single()
            await result.consume()
        return self._public_props(record["props"]) if record else None

    async def update(
        self,
        doc_id: str,
        link_id: str,
        *,
        expected_version: int,
        url: str,
        label: str | None,
        actor: str,
        updated_at: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        async def _write():
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        MATCH (link:`{self._label()}` {{doc_id: $doc_id, id: $link_id}})
                        WHERE coalesce(link.deleted, false) = false
                          AND link.version = $expected_version
                        WITH link, properties(link) AS before
                        SET link.url = $url,
                            link.label = $label,
                            link.updated_by = $actor,
                            link.updated_at = $updated_at,
                            link.version = $expected_version + 1
                        RETURN before, properties(link) AS after
                        """,
                        doc_id=doc_id,
                        link_id=link_id,
                        expected_version=expected_version,
                        url=url,
                        label=label,
                        actor=actor,
                        updated_at=updated_at,
                    )
                    record = await result.single()
                    await result.consume()
                    return record

        record = await with_conflict_retry(f"source_links.update[{link_id}]", _write)
        if record:
            return self._public_props(record["before"]), self._public_props(
                record["after"]
            )
        await self._raise_missing_or_conflict(doc_id, link_id)
        raise AssertionError("unreachable")

    async def delete(
        self,
        doc_id: str,
        link_id: str,
        *,
        expected_version: int,
        actor: str,
        deleted_at: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        async def _write():
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        MATCH (link:`{self._label()}` {{doc_id: $doc_id, id: $link_id}})
                        WHERE coalesce(link.deleted, false) = false
                          AND link.version = $expected_version
                        WITH link, properties(link) AS before
                        SET link.deleted = true,
                            link.deleted_by = $actor,
                            link.deleted_at = $deleted_at,
                            link.updated_by = $actor,
                            link.updated_at = $deleted_at,
                            link.version = $expected_version + 1
                        RETURN before, properties(link) AS after
                        """,
                        doc_id=doc_id,
                        link_id=link_id,
                        expected_version=expected_version,
                        actor=actor,
                        deleted_at=deleted_at,
                    )
                    record = await result.single()
                    await result.consume()
                    return record

        record = await with_conflict_retry(f"source_links.delete[{link_id}]", _write)
        if record:
            return self._public_props(record["before"]), self._public_props(
                record["after"]
            )
        await self._raise_missing_or_conflict(doc_id, link_id)
        raise AssertionError("unreachable")

    async def _raise_missing_or_conflict(self, doc_id: str, link_id: str) -> None:
        current = await self._current(doc_id, link_id)
        if current is None or current.get("deleted"):
            raise SourceLinkNotFound(link_id)
        raise SourceLinkVersionConflict(link_id)


SourceLinkStore = InMemorySourceLinkStore | MemgraphSourceLinkStore


__all__ = [
    "InMemorySourceLinkStore",
    "MemgraphSourceLinkStore",
    "SourceLinkNotFound",
    "SourceLinkStore",
    "SourceLinkVersionConflict",
]
