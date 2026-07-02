"""State store backing the WebUI route module."""

from __future__ import annotations

import copy
import threading
from typing import Any

from .. import folder_store, webui_seed
from ..folder import current_folder_id, load_folder_catalog
from ..webui_activitystore import InMemoryActivityStore, MemgraphActivityStore
from ..webui_notificationstore import (
    InMemoryNotificationStore,
    MemgraphNotificationStore,
)
from ..webui_tagstore import InMemoryTagStore, MemgraphTagStore


class WebuiStore:
    """In-process + Memgraph-pluggable state for the WebUI endpoints.

    The three mutation-heavy resources (tags, activity, notifications) each
    accept an injected backend; the rest stays seeded in-memory. Reads on
    backed resources go through the backend; static accessors return deep
    copies of the seed.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        folders: list[dict[str, Any]],
        thesaurus: list[dict[str, Any]],
        tag_categories_seed: list[dict[str, Any]],
        tags_seed: list[dict[str, Any]],
        openapi_groups: list[dict[str, Any]],
        openapi_version: str,
        graph_entities: list[dict[str, Any]],
        graph_relations: list[dict[str, Any]],
        tag_backend: InMemoryTagStore | MemgraphTagStore | None = None,
        activity_backend: InMemoryActivityStore | MemgraphActivityStore | None = None,
        notification_backend: (
            InMemoryNotificationStore | MemgraphNotificationStore | None
        ) = None,
        mode: str = "seed",
    ) -> None:
        self._documents = documents
        self._folders = folders
        self._thesaurus = thesaurus
        self._openapi_groups = openapi_groups
        self._openapi_version = openapi_version
        self._graph_entities = graph_entities
        self._graph_relations = graph_relations
        # Audit C5: explicit mode used by route-level gates such as
        # ``_graph_seed_fallback_allowed``. ``"seed"`` means the store
        # carries demo fixtures and is safe to serve as a fallback in
        # dev/standalone; ``"memgraph"`` means a production deploy and
        # demo data must NEVER leak even if Memgraph is empty.
        self._mode = mode
        self._tag_backend: InMemoryTagStore | MemgraphTagStore = (
            tag_backend
            if tag_backend is not None
            else InMemoryTagStore(tags=tags_seed, categories=tag_categories_seed)
        )
        self._activity_backend: InMemoryActivityStore | MemgraphActivityStore = (
            activity_backend
            if activity_backend is not None
            else InMemoryActivityStore()
        )
        self._notification_backend: (
            InMemoryNotificationStore | MemgraphNotificationStore
        ) = (
            notification_backend
            if notification_backend is not None
            else InMemoryNotificationStore()
        )
        self._lock = threading.Lock()

    # -- Construction ---------------------------------------------------

    @classmethod
    def from_seed(cls) -> WebuiStore:
        return cls(
            documents=copy.deepcopy(webui_seed.DOCUMENTS),
            folders=copy.deepcopy(webui_seed.FOLDERS),
            thesaurus=copy.deepcopy(webui_seed.THESAURUS),
            tag_categories_seed=copy.deepcopy(webui_seed.TAG_CATEGORIES),
            tags_seed=copy.deepcopy(webui_seed.TAGS),
            openapi_groups=copy.deepcopy(webui_seed.OPENAPI_GROUPS),
            openapi_version=webui_seed.OPENAPI_VERSION,
            graph_entities=copy.deepcopy(webui_seed.GRAPH_ENTITIES),
            graph_relations=copy.deepcopy(webui_seed.GRAPH_RELATIONS),
            mode="seed",
        )

    @classmethod
    def for_folder(cls, folder: str, *, mode: str = "seed") -> WebuiStore:
        """Build a per-folder WebuiStore.

        ``mode``:

        - ``"seed"`` (default) — the default folder gets the full demo
          payload from :meth:`from_seed`; non-default folders start empty
          for user-generated stores (documents / tags / graph) while
          keeping reference data (folders / thesaurus / openapi).
          Useful for ``python -m twindb_lightrag_memgraph.server``
          standalone demo and CI.

        - ``"memgraph"`` — every folder, **including the default**, boots
          without demo user content or demo suggestion vocabulary. Reference
          catalog metadata required by the UI is still loaded.
        """
        if mode == "memgraph":
            return cls(
                documents=[],
                folders=copy.deepcopy(webui_seed.FOLDERS),
                thesaurus=[],
                tag_categories_seed=copy.deepcopy(webui_seed.TAG_CATEGORIES),
                tags_seed=[],
                openapi_groups=copy.deepcopy(webui_seed.OPENAPI_GROUPS),
                openapi_version=webui_seed.OPENAPI_VERSION,
                graph_entities=[],
                graph_relations=[],
                mode="memgraph",
            )
        default_folder = load_folder_catalog().default_folder_id
        if folder == default_folder:
            return cls.from_seed()
        return cls(
            documents=[],
            folders=copy.deepcopy(webui_seed.FOLDERS),
            thesaurus=copy.deepcopy(webui_seed.THESAURUS),
            tag_categories_seed=copy.deepcopy(webui_seed.TAG_CATEGORIES),
            tags_seed=[],
            openapi_groups=copy.deepcopy(webui_seed.OPENAPI_GROUPS),
            openapi_version=webui_seed.OPENAPI_VERSION,
            graph_entities=[],
            graph_relations=[],
            mode="seed",
        )

    # -- Backend accessors --------------------------------------------

    @property
    def mode(self) -> str:
        """Explicit ``webui_stores`` mode this store was built with.

        ``"seed"`` for in-memory demo content; ``"memgraph"`` for a
        production deploy backed by real storage. Audit C5 uses this
        to gate the graph-seed fallback at the route level.
        """
        return self._mode

    @property
    def tags(self) -> InMemoryTagStore | MemgraphTagStore:
        return self._tag_backend

    @property
    def activity(self) -> InMemoryActivityStore | MemgraphActivityStore:
        return self._activity_backend

    @property
    def notifications(
        self,
    ) -> InMemoryNotificationStore | MemgraphNotificationStore:
        return self._notification_backend

    # -- Documents ------------------------------------------------------

    def list_documents(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        default_folder = load_folder_catalog().default_folder_id
        active_folder = current_folder_id()
        items = [
            d
            for d in self._documents
            if (
                d.get("folder") or d.get("metadata", {}).get("folder") or default_folder
            )
            == active_folder
        ]
        if status and status != "all":
            items = [d for d in items if d["status"] == status]
        if q:
            needle = q.lower()
            items = [d for d in items if needle in str(d.get("source", "")).lower()]
        if tag:
            items = [d for d in items if tag in d.get("tags", [])]
        return copy.deepcopy(items)

    # -- Folders -------------------------------------------------------

    def list_folders(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._folders)

    # -- Notifications -------------------------------------------------

    async def list_notifications(self) -> list[dict[str, Any]]:
        return await self._notification_backend.list()

    async def mark_all_notifications_read(self) -> None:
        await self._notification_backend.mark_all_read()

    async def clear_notifications(self) -> None:
        await self._notification_backend.clear()

    async def push_notification(self, notification: dict[str, Any]) -> dict[str, Any]:
        return await self._notification_backend.push(notification)

    # -- Thesaurus + tags ---------------------------------------------

    async def list_thesaurus(self) -> list[dict[str, Any]]:
        """Legacy autocomplete endpoint, derived from the tag catalog.

        `/tags` is the canonical governance surface. `/thesaurus` remains
        only for older clients and must not carry a second, divergent
        vocabulary.
        """
        tags = await self.list_tags()
        return [
            {
                "tag": entry["tag"],
                "category": entry.get("category", "uncategorized"),
                "def": entry.get("def", ""),
            }
            for entry in tags
            if entry.get("tier") != "requested"
            and entry.get("status") not in {"deprecated", "rejected"}
        ]

    async def list_tags(self) -> list[dict[str, Any]]:
        backend = self._tag_backend
        if isinstance(backend, MemgraphTagStore):
            return await backend.list_tags()
        return backend.list_tags()

    async def list_tag_categories(self) -> list[dict[str, Any]]:
        backend = self._tag_backend
        if isinstance(backend, MemgraphTagStore):
            return await backend.list_categories()
        return backend.list_categories()

    # -- Activity ------------------------------------------------------

    async def list_activity(
        self,
        *,
        kind: str | None = None,
        sev: str | None = None,
        actor: str | None = None,
        q: str | None = None,
        range: str | None = None,
        resource_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        return await self._activity_backend.list(
            kind=kind,
            sev=sev,
            actor=actor,
            q=q,
            range=range,
            resource_id=resource_id,
            limit=limit,
        )

    async def record_activity(self, event: dict[str, Any]) -> dict[str, Any]:
        return await self._activity_backend.append(event)

    # -- OpenAPI -------------------------------------------------------

    def openapi(self) -> tuple[list[dict[str, Any]], str]:
        return copy.deepcopy(self._openapi_groups), self._openapi_version

    # -- Graph ---------------------------------------------------------

    def list_graph_entities(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._graph_entities)

    def list_graph_relations(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._graph_relations)


_stores: dict[str, WebuiStore] = {}


def get_store(folder: str | None = None) -> WebuiStore:
    folder_id = folder or current_folder_id()
    store = _stores.get(folder_id)
    if store is None:
        store = WebuiStore.for_folder(folder_id)
        _stores[folder_id] = store
    return store


def set_store(store: WebuiStore, folder: str | None = None) -> None:
    folder_id = folder or load_folder_catalog().default_folder_id
    _stores[folder_id] = store


def reset_store() -> None:
    _stores.clear()
    folder_store.reset_runtime_store()
    set_store(WebuiStore.from_seed())
