"""WebUI phase-1 router — exposes the endpoints the Twin operator console
expects.

Wire contract = the TypeScript fixtures in ``lightrag_webui_twin/src/fixtures/``.
The router is mounted at the FastAPI app root by ``create_app()`` (toggleable
via the ``enable_webui_routes`` setting, default True).

Phase-1 storage model: an in-process ``WebuiStore`` keyed off the seed data,
exposing read accessors with filter semantics and a couple of mutation
helpers for notifications. Subsequent slices will swap this out for real
persistence (Memgraph for docs+graph, dedicated stores for tag governance,
an events table for activity).

Why a separate store class instead of mutating module-level lists:
- A single ``WebuiStore`` instance can be replaced wholesale in tests
  (``set_store(WebuiStore.from_seed())``) so test isolation is trivial.
- Future slices can drop in a Memgraph-backed implementation by writing a
  ``MemgraphWebuiStore`` with the same surface.
"""

from __future__ import annotations

import copy
import threading
from typing import Any

from fastapi import APIRouter, Query

from . import webui_seed
from .webui_models import (
    AckResponse,
    ActivityEnvelope,
    ActivityEvent,
    Document,
    GraphEntity,
    GraphRelation,
    ListEnvelope,
    Notification,
    OpenApiEnvelope,
    OpenApiGroup,
    TagCategory,
    TagEntry,
    ThesaurusEntry,
    Workspace,
)


# ---------------------------------------------------------------------------
# In-memory store (phase-1 backing for the WebUI surface)
# ---------------------------------------------------------------------------


class WebuiStore:
    """Mutable in-process state for the WebUI endpoints.

    Each accessor returns deep copies so callers can't mutate the seed by
    side-effect. Mutations (e.g. ``mark_all_notifications_read``) hold a
    lock to keep the state coherent under concurrent FastAPI requests.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        workspaces: list[dict[str, Any]],
        notifications: list[dict[str, Any]],
        thesaurus: list[dict[str, Any]],
        tags: list[dict[str, Any]],
        tag_categories: list[dict[str, Any]],
        activity: list[dict[str, Any]],
        activity_now_ms: int,
        openapi_groups: list[dict[str, Any]],
        openapi_version: str,
        graph_entities: list[dict[str, Any]],
        graph_relations: list[dict[str, Any]],
    ) -> None:
        self._documents = documents
        self._workspaces = workspaces
        self._notifications = notifications
        self._thesaurus = thesaurus
        self._tags = tags
        self._tag_categories = tag_categories
        self._activity = activity
        self._activity_now_ms = activity_now_ms
        self._openapi_groups = openapi_groups
        self._openapi_version = openapi_version
        self._graph_entities = graph_entities
        self._graph_relations = graph_relations
        self._lock = threading.Lock()

    # -- Construction ---------------------------------------------------

    @classmethod
    def from_seed(cls) -> WebuiStore:
        """Build a fresh store from the module-level seed (deep-copied)."""
        return cls(
            documents=copy.deepcopy(webui_seed.DOCUMENTS),
            workspaces=copy.deepcopy(webui_seed.WORKSPACES),
            notifications=copy.deepcopy(webui_seed.NOTIFICATIONS),
            thesaurus=copy.deepcopy(webui_seed.THESAURUS),
            tags=copy.deepcopy(webui_seed.TAGS),
            tag_categories=copy.deepcopy(webui_seed.TAG_CATEGORIES),
            activity=copy.deepcopy(webui_seed.ACTIVITY),
            activity_now_ms=webui_seed.ACTIVITY_NOW_MS,
            openapi_groups=copy.deepcopy(webui_seed.OPENAPI_GROUPS),
            openapi_version=webui_seed.OPENAPI_VERSION,
            graph_entities=copy.deepcopy(webui_seed.GRAPH_ENTITIES),
            graph_relations=copy.deepcopy(webui_seed.GRAPH_RELATIONS),
        )

    # -- Documents ------------------------------------------------------

    def list_documents(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        items = self._documents
        if status and status != "all":
            items = [d for d in items if d["status"] == status]
        if q:
            needle = q.lower()
            items = [d for d in items if needle in str(d.get("source", "")).lower()]
        if tag:
            items = [d for d in items if tag in d.get("tags", [])]
        return copy.deepcopy(items)

    # -- Workspaces / notifications ------------------------------------

    def list_workspaces(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._workspaces)

    def list_notifications(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._notifications)

    def mark_all_notifications_read(self) -> None:
        with self._lock:
            for n in self._notifications:
                n["read"] = True

    def clear_notifications(self) -> None:
        with self._lock:
            self._notifications.clear()

    # -- Thesaurus + tags ----------------------------------------------

    def list_thesaurus(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._thesaurus)

    def list_tags(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._tags)

    def list_tag_categories(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._tag_categories)

    # -- Activity ------------------------------------------------------

    def list_activity(
        self,
        *,
        kind: str | None = None,
        sev: str | None = None,
        actor: str | None = None,
        q: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        items = self._activity
        if kind:
            wanted = {k for k in kind.split(",") if k}
            if wanted:
                items = [e for e in items if e["kind"] in wanted]
        if sev and sev != "any":
            items = [e for e in items if e["sev"] == sev]
        if actor and actor != "any":
            items = [e for e in items if e["actor"]["user"] == actor]
        if q:
            needle = q.lower()
            items = [
                e
                for e in items
                if needle
                in (
                    str(e.get("summary", ""))
                    + " "
                    + str(e.get("target", {}).get("label", ""))
                    + " "
                    + str(e.get("actor", {}).get("user", ""))
                ).lower()
            ]
        return copy.deepcopy(items), self._activity_now_ms

    # -- OpenAPI -------------------------------------------------------

    def openapi(self) -> tuple[list[dict[str, Any]], str]:
        return copy.deepcopy(self._openapi_groups), self._openapi_version

    # -- Graph ---------------------------------------------------------

    def list_graph_entities(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._graph_entities)

    def list_graph_relations(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._graph_relations)


# ---------------------------------------------------------------------------
# Module-level store + accessors (replaceable in tests)
# ---------------------------------------------------------------------------


_store: WebuiStore = WebuiStore.from_seed()


def get_store() -> WebuiStore:
    return _store


def set_store(store: WebuiStore) -> None:
    """Replace the module-level store. Intended for tests."""
    global _store
    _store = store


def reset_store() -> None:
    """Rebuild a fresh store from the seed (drops mutations)."""
    set_store(WebuiStore.from_seed())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["webui"])


@router.get("/documents", response_model=ListEnvelope[Document])
async def list_documents(
    status: str | None = Query(default=None),
    q: str | None = Query(default=None),
    tag: str | None = Query(default=None),
) -> dict[str, Any]:
    items = get_store().list_documents(status=status, q=q, tag=tag)
    return {"items": items, "total": len(items)}


@router.get("/workspaces", response_model=list[Workspace])
async def list_workspaces() -> list[dict[str, Any]]:
    return get_store().list_workspaces()


@router.get("/notifications", response_model=list[Notification])
async def list_notifications() -> list[dict[str, Any]]:
    return get_store().list_notifications()


@router.post("/notifications/read-all", response_model=AckResponse)
async def mark_all_notifications_read() -> dict[str, bool]:
    get_store().mark_all_notifications_read()
    return {"ok": True}


@router.delete("/notifications", response_model=AckResponse)
async def clear_notifications() -> dict[str, bool]:
    get_store().clear_notifications()
    return {"ok": True}


@router.get("/thesaurus", response_model=list[ThesaurusEntry])
async def list_thesaurus() -> list[dict[str, Any]]:
    return get_store().list_thesaurus()


@router.get("/tags", response_model=list[TagEntry])
async def list_tags() -> list[dict[str, Any]]:
    return get_store().list_tags()


@router.get("/tags/categories", response_model=list[TagCategory])
async def list_tag_categories() -> list[dict[str, Any]]:
    return get_store().list_tag_categories()


@router.get("/activity", response_model=ActivityEnvelope)
async def list_activity(
    kind: str | None = Query(default=None),
    sev: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    q: str | None = Query(default=None),
) -> dict[str, Any]:
    items, now_ms = get_store().list_activity(kind=kind, sev=sev, actor=actor, q=q)
    return {"items": items, "total": len(items), "nowMs": now_ms}


# Note: this endpoint exposes a *curated* OpenAPI surface for the WebUI's API
# tab. It does NOT shadow FastAPI's own ``/openapi.json`` (the auto-generated
# spec) — they coexist.
@router.get("/openapi", response_model=OpenApiEnvelope)
async def get_openapi_groups() -> dict[str, Any]:
    groups, version = get_store().openapi()
    return {"groups": groups, "version": version}


@router.get("/graph/entities", response_model=list[GraphEntity])
async def list_graph_entities() -> list[dict[str, Any]]:
    return get_store().list_graph_entities()


@router.get("/graph/relations", response_model=list[GraphRelation])
async def list_graph_relations() -> list[dict[str, Any]]:
    return get_store().list_graph_relations()


# Convenience re-export for ``app.py`` import.
__all__ = [
    "router",
    "WebuiStore",
    "get_store",
    "set_store",
    "reset_store",
    "OpenApiGroup",  # surfaced for test type assertions
]
