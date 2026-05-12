"""Tag governance storage backends for the WebUI router (S4c slice 1).

Defines a small protocol so the router can read tags + categories without
caring whether they come from the in-memory seed or a Memgraph KV namespace.
The Memgraph backend bootstraps from the seed on first init when its KV is
empty, so dev/demo experience is identical regardless of backend.

Why this slice is read-only:
- The WebUI's tag-action toasts (Approve / Edit / Deprecate / …) are still
  consumed client-side in App.tsx — the router does not yet expose
  POST/PATCH/DELETE on tags. Persisting reads now means the day mutation
  endpoints land, we don't have to swap the read path again.

KV schema (Memgraph backend):
  :WebuiTag_{workspace}        nodes — one per tag,      id=tag,        data=JSON
  :WebuiTagCategory_{workspace} nodes — one per category, id=category id, data=JSON

Reads issue a single Cypher query per resource. Bootstrap is also a single
UNWIND so booting a fresh workspace is one round-trip per resource.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Protocol

from .. import _pool
from .._constants import validate_identifier
from . import webui_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — minimal surface the WebuiStore depends on for tag governance
# ---------------------------------------------------------------------------


class TagStore(Protocol):
    """Tag governance store. Read + mutation surface used by the WebUI router.

    Mutations are async because the Memgraph implementation issues Cypher
    writes; the in-memory variant matches the signature with no-op awaits.
    """

    def list_tags(self) -> list[dict[str, Any]]: ...
    def list_categories(self) -> list[dict[str, Any]]: ...
    async def get_tag(self, tag: str) -> dict[str, Any] | None: ...
    async def upsert_tag(self, entry: dict[str, Any]) -> dict[str, Any]: ...
    async def delete_tag(self, tag: str) -> bool: ...


# ---------------------------------------------------------------------------
# In-memory backend (current behavior, default)
# ---------------------------------------------------------------------------


class InMemoryTagStore:
    """Tag store backed by a Python list seeded from ``webui_seed``."""

    def __init__(
        self,
        tags: list[dict[str, Any]] | None = None,
        categories: list[dict[str, Any]] | None = None,
    ) -> None:
        self._tags = copy.deepcopy(tags if tags is not None else webui_seed.TAGS)
        self._categories = copy.deepcopy(
            categories if categories is not None else webui_seed.TAG_CATEGORIES
        )

    def list_tags(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._tags)

    def list_categories(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._categories)

    async def get_tag(self, tag: str) -> dict[str, Any] | None:
        for entry in self._tags:
            if entry["tag"] == tag:
                return copy.deepcopy(entry)
        return None

    async def upsert_tag(self, entry: dict[str, Any]) -> dict[str, Any]:
        name = entry["tag"]
        for i, existing in enumerate(self._tags):
            if existing["tag"] == name:
                self._tags[i] = copy.deepcopy(entry)
                return copy.deepcopy(self._tags[i])
        self._tags.append(copy.deepcopy(entry))
        return copy.deepcopy(entry)

    async def delete_tag(self, tag: str) -> bool:
        before = len(self._tags)
        self._tags = [t for t in self._tags if t["tag"] != tag]
        return len(self._tags) < before


# ---------------------------------------------------------------------------
# Memgraph backend — persists tags + categories as KV nodes
# ---------------------------------------------------------------------------


class MemgraphTagStore:
    """Tag store backed by Memgraph KV nodes, per workspace.

    Calling ``await store.bootstrap()`` once at startup writes the seed if the
    workspace KV is empty; subsequent boots find existing data and skip.
    """

    def __init__(self, workspace: str = "default") -> None:
        # Memgraph labels are workspace-scoped and Cypher-interpolated as
        # identifiers (backticked, but validation here is the canonical
        # belt-and-suspenders against the Cassandre injection class).
        validate_identifier(workspace, "workspace")
        self._workspace = workspace

    # -- Labels -------------------------------------------------------

    @property
    def _tag_label(self) -> str:
        return f"WebuiTag_{self._workspace}"

    @property
    def _cat_label(self) -> str:
        return f"WebuiTagCategory_{self._workspace}"

    # -- Public surface ----------------------------------------------

    async def initialize(self) -> None:
        """Create id-indexes on the two labels. Idempotent."""
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                for label in (self._tag_label, self._cat_label):
                    try:
                        result = await session.run(
                            f"CREATE INDEX ON :`{label}`(id)"
                        )
                        await result.consume()
                        logger.info(
                            "[WebuiTagStore] Index on :%s(id) ensured", label
                        )
                    except Exception as e:  # noqa: BLE001 — narrow check below
                        if "already exists" in str(e).lower():
                            logger.debug(
                                "[WebuiTagStore] Index already exists on :%s(id)",
                                label,
                            )
                        else:
                            raise

    async def bootstrap_if_empty(
        self,
        tags: list[dict[str, Any]] | None = None,
        categories: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Seed the KV when both label sets are empty for this workspace.

        Returns True if a bootstrap happened (KV was empty), False otherwise.
        """
        seed_tags = tags if tags is not None else webui_seed.TAGS
        seed_cats = categories if categories is not None else webui_seed.TAG_CATEGORIES

        async with _pool.get_read_session() as session:
            res = await session.run(
                f"MATCH (n:`{self._tag_label}`) RETURN count(n) AS c"
            )
            record = await res.single()
            await res.consume()
            tag_count = record["c"] if record else 0
            res = await session.run(
                f"MATCH (n:`{self._cat_label}`) RETURN count(n) AS c"
            )
            record = await res.single()
            await res.consume()
            cat_count = record["c"] if record else 0

        if tag_count > 0 or cat_count > 0:
            logger.debug(
                "[WebuiTagStore] Bootstrap skipped (tags=%d, categories=%d)",
                tag_count,
                cat_count,
            )
            return False

        await self._write_many(self._tag_label, "tag", seed_tags)
        await self._write_many(self._cat_label, "id", seed_cats)
        logger.info(
            "[WebuiTagStore] Bootstrapped %d tags + %d categories",
            len(seed_tags),
            len(seed_cats),
        )
        return True

    async def list_tags(self) -> list[dict[str, Any]]:
        return await self._read_many(self._tag_label)

    async def list_categories(self) -> list[dict[str, Any]]:
        return await self._read_many(self._cat_label)

    async def get_tag(self, tag: str) -> dict[str, Any] | None:
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._tag_label}` {{id: $id}}) RETURN n.data AS data",
                id=tag,
            )
            record = await result.single()
            await result.consume()
        if not record or not record.get("data"):
            return None
        try:
            return json.loads(record["data"])
        except json.JSONDecodeError:
            return None

    async def upsert_tag(self, entry: dict[str, Any]) -> dict[str, Any]:
        if "tag" not in entry:
            raise ValueError("upsert_tag requires entry['tag']")
        await self._write_many(self._tag_label, "tag", [entry])
        return copy.deepcopy(entry)

    async def delete_tag(self, tag: str) -> bool:
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"MATCH (n:`{self._tag_label}` {{id: $id}}) "
                    "WITH n, count(n) AS c "
                    "DETACH DELETE n "
                    "RETURN c",
                    id=tag,
                )
                record = await result.single()
                await result.consume()
        return bool(record and record.get("c", 0) > 0)

    # -- Internals ---------------------------------------------------

    async def _read_many(self, label: str) -> list[dict[str, Any]]:
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}`) RETURN n.id AS id, n.data AS data "
                f"ORDER BY n.`__created_at`, n.id"
            )
            rows = await result.data()
            await result.consume()
        out: list[dict[str, Any]] = []
        for row in rows:
            data_raw = row.get("data")
            if not data_raw:
                continue
            try:
                out.append(json.loads(data_raw))
            except json.JSONDecodeError:
                logger.warning(
                    "[WebuiTagStore] Skipping non-JSON node on :%s id=%s",
                    label,
                    row.get("id"),
                )
        return out

    async def _write_many(
        self,
        label: str,
        id_key: str,
        items: list[dict[str, Any]],
    ) -> None:
        """UNWIND-batch insert; assigns __created_at on first MERGE."""
        rows = [
            {"id": str(item[id_key]), "data": json.dumps(item, sort_keys=True)}
            for item in items
        ]
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (n:`{label}` {{id: row.id}})
                    ON CREATE SET n.`__created_at` = timestamp()
                    SET n.data = row.data, n.`__updated_at` = timestamp()
                    """,
                    rows=rows,
                )
                await result.consume()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_in_memory_store() -> InMemoryTagStore:
    """Build the default in-memory store seeded from ``webui_seed``."""
    return InMemoryTagStore()


async def make_memgraph_store(workspace: str = "default") -> MemgraphTagStore:
    """Build a Memgraph-backed store, ensure indexes, and bootstrap-from-seed
    on the first call for a fresh workspace."""
    store = MemgraphTagStore(workspace=workspace)
    await store.initialize()
    await store.bootstrap_if_empty()
    return store
