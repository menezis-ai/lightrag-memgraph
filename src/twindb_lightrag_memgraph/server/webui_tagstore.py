"""Tag governance storage backends for the WebUI router (S4c slice 1).

Defines a small protocol so the router can read tags + categories without
caring whether they come from the in-memory seed or a Memgraph KV namespace.
The low-level Memgraph factory can explicitly bootstrap demo data for dev/test;
production app wiring instantiates the store directly so fresh folders start
without demo tags.

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
from .._constants import resolve_workspace, validate_identifier
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
    async def get_tag(
        self, tag: str
    ) -> dict[str, Any] | None: ...  # NOSONAR - async contract.
    async def upsert_tag(
        self, entry: dict[str, Any]
    ) -> dict[str, Any]: ...  # NOSONAR - async contract.
    async def delete_tag(self, tag: str) -> bool: ...  # NOSONAR - async contract.


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

    async def get_tag(
        self, tag: str
    ) -> dict[str, Any] | None:  # NOSONAR - async contract.
        for entry in self._tags:
            if entry["tag"] == tag:
                return copy.deepcopy(entry)
        return None

    async def upsert_tag(
        self, entry: dict[str, Any]
    ) -> dict[str, Any]:  # NOSONAR - async contract.
        name = entry["tag"]
        for i, existing in enumerate(self._tags):
            if existing["tag"] == name:
                self._tags[i] = copy.deepcopy(entry)
                return copy.deepcopy(self._tags[i])
        self._tags.append(copy.deepcopy(entry))
        return copy.deepcopy(entry)

    async def delete_tag(self, tag: str) -> bool:  # NOSONAR - async contract.
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
        # belt-and-suspenders against the Cypher-injection class).
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
                        result = await session.run(f"CREATE INDEX ON :`{label}`(id)")
                        await result.consume()
                        logger.info("[WebuiTagStore] Index on :%s(id) ensured", label)
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

    async def replace_categories_from_config(
        self,
        config_path: str,
    ) -> int:
        """Mirror the categories from an external JSON config file.

        Doctrine: categories are referentiel — a curated taxonomy
        decided by Knowledge governance, not user-generated. To respect
        that doctrine in production, the tenant admin defines the
        taxonomy in a JSON file (Kubernetes ConfigMap, NFS mount, …)
        and Twin enforces it on every boot.

        Semantics: **replace, not merge**. Every boot, the function:
          1. Reads the JSON file.
          2. Validates the shape (list of ``{id, label, color}``).
          3. Deletes all existing ``WebuiTagCategory_{workspace}``
             nodes for this workspace.
          4. Writes the file's categories.

        This is idempotent across reboots: editing the config file
        and restarting Twin is enough to publish a taxonomy change.

        Caveat: tags that reference a removed category keep their
        ``category`` field pointing at a now-orphan id. We log them
        as a warning so the admin can decide whether to migrate or
        purge. We do **not** auto-delete or auto-rename — destructive
        cleanup of user-generated data must always be explicit.

        Args:
            config_path: filesystem path to a JSON file shaped like
                ``[{"id": "oracle", "label": "Oracle", "color": "#..."}, …]``.

        Returns:
            The number of categories applied.

        Raises:
            FileNotFoundError: if the config path doesn't exist.
            ValueError: if the JSON shape is invalid (missing keys,
                duplicate ids, non-list root).
        """
        import json
        from pathlib import Path

        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"webui_categories_config: file not found at {path}. "
                "Either drop the flag (fallback to internal seed) or "
                "fix the mount path."
            )

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"webui_categories_config: {path} is not valid JSON: {exc}"
            ) from exc

        return await self.replace_categories_from_list(raw, source=str(path))

    async def replace_categories_from_list(
        self,
        raw: Any,
        source: str = "import",
    ) -> int:
        """Validate + mirror a list of category objects into Memgraph.

        Same semantics as :meth:`replace_categories_from_config` but
        consumes an in-memory ``list[dict]`` instead of a file. Used by
        the HTTP import endpoint (``POST /tags/categories/_import``) so
        an admin can upload a JSON taxonomy via the WebUI without
        needing shell access to the host.

        Validation matches ``docs/templates/twin-categories.schema.json``:
        a non-empty list of objects each carrying ``id``, ``label``,
        ``color``; ``id`` must be unique within the document and a
        non-empty string.

        Args:
            raw: parsed JSON content — expected to be a list of dicts.
            source: free-form label included in error messages so the
                caller knows which input was rejected
                (e.g. ``"import"``, the file path, ``"_self_check"``).

        Returns:
            The number of categories applied.

        Raises:
            ValueError: if ``raw`` doesn't match the schema.
        """
        if not isinstance(raw, list):
            raise ValueError(
                f"{source}: root must be a JSON array of category objects, "
                f"got {type(raw).__name__}."
            )
        if not raw:
            raise ValueError(
                f"{source}: array is empty — at least one category is required."
            )

        seen_ids: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{source}[{i}]: must be an object, " f"got {type(entry).__name__}."
                )
            for required in ("id", "label", "color"):
                if required not in entry:
                    raise ValueError(
                        f"{source}[{i}]: missing required key {required!r}. "
                        f"Got keys {list(entry.keys())}."
                    )
            cat_id = entry["id"]
            if not isinstance(cat_id, str) or not cat_id:
                raise ValueError(
                    f"{source}[{i}].id must be a non-empty string, " f"got {cat_id!r}."
                )
            if cat_id in seen_ids:
                raise ValueError(f"{source}[{i}]: duplicate id {cat_id!r}.")
            seen_ids.add(cat_id)
            normalized.append(
                {
                    "id": cat_id,
                    "label": str(entry["label"]),
                    "color": str(entry["color"]),
                }
            )

        # Mirror: drop existing then write fresh.
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"MATCH (n:`{self._cat_label}`) DETACH DELETE n"
                )
                await result.consume()
        await self._write_many(self._cat_label, "id", normalized)

        logger.info(
            "[WebuiTagStore] replace_categories_from_list: "
            "applied %d categories from %s",
            len(normalized),
            source,
        )
        return len(normalized)

    async def bootstrap_categories_if_empty(
        self,
        categories: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Seed JUST the categories KV when empty for this workspace.

        Variant of :meth:`bootstrap_if_empty` that touches **only** the
        ``WebuiTagCategory_{workspace}`` label, never the tag label.
        Used by ``register(webui_stores="memgraph")`` to ship a curated
        taxonomy (Oracle, Infrastructure, Network, Payment, Lifecycle,
        Governance) without polluting tags / activity / notifications
        with demo fixtures — the doctrine being that *categories are
        referentiel data* (governance taxonomy) whereas *tags are
        user-generated*.

        Eventually superseded by an external config (e.g. JSON file
        mounted via ConfigMap) — cf. the Option 3 doctrine. For now this
        bootstrap-from-seed is the pragmatic shim.

        Returns True if a bootstrap happened, False if categories were
        already present.
        """
        seed_cats = categories if categories is not None else webui_seed.TAG_CATEGORIES

        async with _pool.get_read_session() as session:
            res = await session.run(
                f"MATCH (n:`{self._cat_label}`) RETURN count(n) AS c"
            )
            record = await res.single()
            await res.consume()
            cat_count = record["c"] if record else 0

        if cat_count > 0:
            logger.debug(
                "[WebuiTagStore] Category bootstrap skipped (count=%d)",
                cat_count,
            )
            return False

        await self._write_many(self._cat_label, "id", seed_cats)
        logger.info(
            "[WebuiTagStore] Bootstrapped %d categories (governance taxonomy)",
            len(seed_cats),
        )
        return True

    async def list_tags(self) -> list[dict[str, Any]]:
        """Return tags with live document/chunk usage counters.

        Tag catalog entries are stored as JSON on ``WebuiTag_*`` nodes, while
        document membership lives in graph edges
        ``(:DocStatus_*)-[:TAGGED_WITH]->(:WebuiTag_*)``. Mutations can change
        the edges without rewriting every tag JSON blob, so derive the usage
        counters from the graph at read time.
        """
        doc_workspace = resolve_workspace()
        doc_label = f"DocStatus_{doc_workspace}"
        folder_label = f"Folder_{doc_workspace}"
        folder = self._workspace
        async with _pool.get_read_session() as session:
            result = await session.run(f"""
                MATCH (t:`{self._tag_label}`)
                RETURN
                  t.id AS id,
                  t.data AS data
                ORDER BY t.`__created_at`, t.id
                """)
            rows = await result.data()
            await result.consume()

            usage_result = await session.run(
                f"""
                MATCH (d:`{doc_label}`)-[:TAGGED_WITH]->(t:`{self._tag_label}`)
                WHERE EXISTS((d)-[:MEMBER_OF]->(:`{folder_label}` {{id: $folder}}))
                RETURN
                  t.id AS id,
                  count(DISTINCT d) AS sources_count,
                  sum(coalesce(d.chunks_count, 0)) AS chunks_count
                """,
                folder=folder,
            )
            usage_rows = await usage_result.data()
            await usage_result.consume()

        usage_by_id = {
            row.get("id"): {
                "sources_count": int(row.get("sources_count") or 0),
                "chunks_count": int(row.get("chunks_count") or 0),
            }
            for row in usage_rows
            if row.get("id")
        }

        out: list[dict[str, Any]] = []
        for row in rows:
            data_raw = row.get("data")
            if not data_raw:
                continue
            try:
                item = json.loads(data_raw)
            except json.JSONDecodeError:
                logger.warning(
                    "[WebuiTagStore] Skipping non-JSON node on :%s id=%s",
                    self._tag_label,
                    row.get("id"),
                )
                continue
            usage = usage_by_id.get(row.get("id"), {})
            item["sources_count"] = usage.get("sources_count", 0)
            item["chunks_count"] = usage.get("chunks_count", 0)
            out.append(item)
        return out

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
    """Build a Memgraph-backed store and explicitly bootstrap demo seed data.

    Production server wiring does not call this factory; it instantiates
    ``MemgraphTagStore`` directly and only bootstraps governance categories.
    """
    store = MemgraphTagStore(workspace=workspace)
    await store.initialize()
    await store.bootstrap_if_empty()
    return store
