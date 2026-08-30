"""State store backing the WebUI route module."""

from __future__ import annotations

import asyncio
import copy
import logging
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
from ..source_links_store import (
    InMemorySourceLinkStore,
    MemgraphSourceLinkStore,
    SourceLinkStore,
)

logger = logging.getLogger(__name__)


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
        source_link_backend: SourceLinkStore | None = None,
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
        self._source_link_backend: SourceLinkStore = (
            source_link_backend
            if source_link_backend is not None
            else InMemorySourceLinkStore()
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

    @property
    def source_links(self) -> SourceLinkStore:
        return self._source_link_backend

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

# Strong references to in-flight backend-initialize tasks scheduled from the
# sync `get_store` path, so the event loop cannot garbage-collect them mid-run.
_pending_backend_inits: set[asyncio.Task] = set()


def _memgraph_template_store() -> WebuiStore | None:
    """The Memgraph-mode store the boot wiring registered, if any.

    MG-3 (audit 2026-07-02): the boot wiring
    (``server/app.py:_init_webui_backends`` and
    ``patches/registry.py:_init_overlay_memgraph_stores``) registers a
    ``mode="memgraph"`` store per catalog folder via :func:`set_store`.
    Folders created *after* boot must inherit that construction mode instead
    of silently falling back to in-RAM seed stores (audit trail lost on
    restart, divergence across workers). The registered default-folder store
    is the construction template; if the default slot was evicted, any other
    registered memgraph-mode store serves as the template.
    """
    default_id = load_folder_catalog().default_folder_id
    candidate = _stores.get(default_id)
    if candidate is not None and candidate.mode == "memgraph":
        return candidate
    for candidate in _stores.values():
        if candidate.mode == "memgraph":
            return candidate
    return None


def deployment_store_mode() -> str:
    """``"memgraph"`` when the boot wiring registered Memgraph-backed stores
    for this deployment, ``"seed"`` otherwise (demo/dev/standalone)."""
    return "memgraph" if _memgraph_template_store() is not None else "seed"


def _build_store_for_deployment(folder_id: str) -> WebuiStore:
    """Build a folder store matching the deployment's construction mode.

    Seed deployments keep the historical behavior (:meth:`WebuiStore.for_folder`
    seed mode). Memgraph deployments mirror the template store: same
    ``mode="memgraph"`` shell, and a Memgraph backend for every resource the
    template has one for (``server/app.py`` may wire only a per-setting
    subset). Folder-owned resources use the new folder id exactly as boot
    wiring would; document-owned source links reuse the global backend so
    provenance follows documents shared into several folders.
    """
    template = _memgraph_template_store()
    if template is None:
        return WebuiStore.for_folder(folder_id)
    store = WebuiStore.for_folder(folder_id, mode="memgraph")
    if isinstance(template._tag_backend, MemgraphTagStore):
        store._tag_backend = MemgraphTagStore(workspace=folder_id)
    if isinstance(template._activity_backend, MemgraphActivityStore):
        store._activity_backend = MemgraphActivityStore(workspace=folder_id)
    if isinstance(template._notification_backend, MemgraphNotificationStore):
        store._notification_backend = MemgraphNotificationStore(workspace=folder_id)
    if isinstance(template._source_link_backend, MemgraphSourceLinkStore):
        # Source links belong to documents, which can be projected into more
        # than one folder.  Share the single global backend: constructing a
        # folder-scoped copy would invite accidental provenance partitioning
        # and would replay the same index DDL for every runtime folder.
        store._source_link_backend = template._source_link_backend
    return store


async def initialize_store_backends(store: WebuiStore) -> None:
    """Ensure indexes (and the tag-category taxonomy) for Memgraph backends.

    Idempotent — mirrors what the boot wiring runs per catalog folder while
    skipping the already-initialized global source-link backend. For a
    brand-new folder the activity legacy-scalar backfill is a no-op and
    ``bootstrap_categories_if_empty`` only seeds an empty label. Deployments
    that mirror categories from a config file (``webui_categories_config``)
    re-assert the taxonomy at next boot (replace-not-merge), so a
    seed-bootstrapped runtime folder self-heals to the config-as-code state.
    """
    tag_backend = store._tag_backend
    if isinstance(tag_backend, MemgraphTagStore):
        await tag_backend.initialize()
        await tag_backend.bootstrap_categories_if_empty()
    activity_backend = store._activity_backend
    if isinstance(activity_backend, MemgraphActivityStore):
        await activity_backend.initialize()
    notification_backend = store._notification_backend
    if isinstance(notification_backend, MemgraphNotificationStore):
        await notification_backend.initialize()
    source_link_backend = store._source_link_backend
    template = _memgraph_template_store()
    template_source_links = (
        template._source_link_backend if template is not None else None
    )
    if (
        isinstance(source_link_backend, MemgraphSourceLinkStore)
        and source_link_backend is not template_source_links
    ):
        await source_link_backend.initialize()


def _schedule_backend_initialize(store: WebuiStore, folder_id: str) -> None:
    """Best-effort async backend init from the sync :func:`get_store` path.

    Index creation is a performance concern, not a correctness one (a
    Memgraph label index created later picks up pre-existing nodes), so a
    deferred or failed init never blocks the request. The deterministic path
    is :func:`ensure_folder_store` (awaited by the folder-create route), and
    the next boot re-initializes every catalog folder anyway.
    """

    async def _run() -> None:
        try:
            await initialize_store_backends(store)
        except Exception:
            logger.exception(
                "WebUI store backends for runtime folder %r: initialize "
                "failed (indexes/taxonomy will be ensured at next boot)",
                folder_id,
            )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(
            "WebUI store backends for runtime folder %r built outside an "
            "event loop; index/taxonomy init deferred to next boot",
            folder_id,
        )
        return
    task = loop.create_task(_run())
    _pending_backend_inits.add(task)
    task.add_done_callback(_pending_backend_inits.discard)


def get_store(folder: str | None = None) -> WebuiStore:
    folder_id = folder or current_folder_id()
    store = _stores.get(folder_id)
    if store is None:
        store = _build_store_for_deployment(folder_id)
        _stores[folder_id] = store
        if store.mode == "memgraph":
            _schedule_backend_initialize(store, folder_id)
    return store


async def ensure_folder_store(folder: str) -> WebuiStore:
    """Deterministic async counterpart of :func:`get_store`.

    Builds the folder's store with the deployment's construction mode and
    *awaits* Memgraph backend initialization instead of scheduling it —
    used by the folder-create route so a runtime-created folder is
    production-grade (indexed, taxonomy bootstrapped) before the 201 returns.
    """
    store = _stores.get(folder)
    if store is None:
        store = _build_store_for_deployment(folder)
        _stores[folder] = store
    if store.mode == "memgraph":
        await initialize_store_backends(store)
    return store


def set_store(store: WebuiStore, folder: str | None = None) -> None:
    folder_id = folder or load_folder_catalog().default_folder_id
    _stores[folder_id] = store


def reset_store() -> None:
    _stores.clear()
    folder_store.reset_runtime_store()
    set_store(WebuiStore.from_seed())
