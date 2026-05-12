"""WebUI phase-1 router — exposes the endpoints the Twin operator console
expects.

Wire contract = the TypeScript fixtures in ``lightrag_webui_twin/src/fixtures/``.
The router is mounted at the FastAPI app root by ``create_app()`` (toggleable
via the ``enable_webui_routes`` setting, default True).

Storage model (S4c):
  - Tags / categories       — `TagStore` (InMemory or MemgraphTagStore).
  - Activity audit feed     — `ActivityStore` (InMemory or MemgraphActivityStore).
  - Notifications           — `NotificationStore` (InMemory or MemgraphNotificationStore).
  - Everything else         — stays on the in-memory WebuiStore seed.

Each backend is selected at app startup via a setting and wired into the
single module-level WebuiStore through ``set_store()``. Tests start each
case with ``reset_store()`` to drop mutations.

Tag mutations emit a synthesized activity event AND push a notification —
the WebUI ``/activity`` and ``/notifications`` queries refresh on each
mutation so the operator sees an audit trail without any extra plumbing.
"""

from __future__ import annotations

import copy
import datetime
import secrets
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from . import webui_seed
from .webui_activitystore import InMemoryActivityStore, MemgraphActivityStore
from .webui_models import (
    AckResponse,
    ActivityEnvelope,
    Document,
    GraphEntity,
    GraphRelation,
    ListEnvelope,
    Notification,
    OpenApiEnvelope,
    OpenApiGroup,
    TagApproveBody,
    TagCategory,
    TagDeleteBody,
    TagDeprecateBody,
    TagEditBody,
    TagEntry,
    TagRejectBody,
    TagRequestBody,
    TagSynonymsBody,
    ThesaurusEntry,
    Workspace,
)
from .webui_notificationstore import (
    InMemoryNotificationStore,
    MemgraphNotificationStore,
)
from .webui_tagstore import InMemoryTagStore, MemgraphTagStore


# ---------------------------------------------------------------------------
# Helpers — synthesized event + notification payloads
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def _make_event(
    *,
    kind: str,
    sev: str,
    actor: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
    target_type: str = "tag",
) -> dict[str, Any]:
    return {
        "id": _new_id("evt"),
        "ts": _utcnow_iso(),
        "rel": "now",
        "day": "Today",
        "kind": kind,
        "sev": sev,
        "actor": {"user": actor, "role": "operator"},
        "target": {"type": target_type, "label": target_label},
        "summary": summary,
        "meta": meta,
    }


def _make_notification(
    *,
    title: str,
    tagname: str | None,
    suffix: str | None,
    sub: str,
    kind: str = "tag-mutation",
) -> dict[str, Any]:
    return {
        "id": _new_id("n"),
        "kind": kind,
        "title": title,
        "tagname": tagname,
        "suffix": suffix,
        "sub": sub,
        "rel": "now",
        "read": False,
    }


# ---------------------------------------------------------------------------
# WebuiStore — pluggable backends for the mutation-heavy resources
# ---------------------------------------------------------------------------


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
        workspaces: list[dict[str, Any]],
        thesaurus: list[dict[str, Any]],
        tag_categories_seed: list[dict[str, Any]],
        tags_seed: list[dict[str, Any]],
        openapi_groups: list[dict[str, Any]],
        openapi_version: str,
        graph_entities: list[dict[str, Any]],
        graph_relations: list[dict[str, Any]],
        tag_backend: InMemoryTagStore | MemgraphTagStore | None = None,
        activity_backend: InMemoryActivityStore | MemgraphActivityStore | None = None,
        notification_backend: InMemoryNotificationStore
        | MemgraphNotificationStore
        | None = None,
    ) -> None:
        self._documents = documents
        self._workspaces = workspaces
        self._thesaurus = thesaurus
        self._openapi_groups = openapi_groups
        self._openapi_version = openapi_version
        self._graph_entities = graph_entities
        self._graph_relations = graph_relations
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
            workspaces=copy.deepcopy(webui_seed.WORKSPACES),
            thesaurus=copy.deepcopy(webui_seed.THESAURUS),
            tag_categories_seed=copy.deepcopy(webui_seed.TAG_CATEGORIES),
            tags_seed=copy.deepcopy(webui_seed.TAGS),
            openapi_groups=copy.deepcopy(webui_seed.OPENAPI_GROUPS),
            openapi_version=webui_seed.OPENAPI_VERSION,
            graph_entities=copy.deepcopy(webui_seed.GRAPH_ENTITIES),
            graph_relations=copy.deepcopy(webui_seed.GRAPH_RELATIONS),
        )

    # -- Backend accessors --------------------------------------------

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
        items = self._documents
        if status and status != "all":
            items = [d for d in items if d["status"] == status]
        if q:
            needle = q.lower()
            items = [d for d in items if needle in str(d.get("source", "")).lower()]
        if tag:
            items = [d for d in items if tag in d.get("tags", [])]
        return copy.deepcopy(items)

    # -- Workspaces ----------------------------------------------------

    def list_workspaces(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._workspaces)

    # -- Notifications -------------------------------------------------

    async def list_notifications(self) -> list[dict[str, Any]]:
        return await self._notification_backend.list()

    async def mark_all_notifications_read(self) -> None:
        await self._notification_backend.mark_all_read()

    async def clear_notifications(self) -> None:
        await self._notification_backend.clear()

    async def push_notification(
        self, notification: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._notification_backend.push(notification)

    # -- Thesaurus + tags ---------------------------------------------

    def list_thesaurus(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._thesaurus)

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
    ) -> tuple[list[dict[str, Any]], int]:
        return await self._activity_backend.list(
            kind=kind, sev=sev, actor=actor, q=q
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


# ---------------------------------------------------------------------------
# Module-level store + accessors (replaceable in tests)
# ---------------------------------------------------------------------------


_store: WebuiStore = WebuiStore.from_seed()


def get_store() -> WebuiStore:
    return _store


def set_store(store: WebuiStore) -> None:
    global _store
    _store = store


def reset_store() -> None:
    set_store(WebuiStore.from_seed())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["webui"])


# -- Read endpoints ----------------------------------------------------------


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
    return await get_store().list_notifications()


@router.post("/notifications/read-all", response_model=AckResponse)
async def mark_all_notifications_read() -> dict[str, bool]:
    await get_store().mark_all_notifications_read()
    return {"ok": True}


@router.delete("/notifications", response_model=AckResponse)
async def clear_notifications() -> dict[str, bool]:
    await get_store().clear_notifications()
    return {"ok": True}


@router.get("/thesaurus", response_model=list[ThesaurusEntry])
async def list_thesaurus() -> list[dict[str, Any]]:
    return get_store().list_thesaurus()


@router.get("/tags", response_model=list[TagEntry])
async def list_tags() -> list[dict[str, Any]]:
    return await get_store().list_tags()


@router.get("/tags/categories", response_model=list[TagCategory])
async def list_tag_categories() -> list[dict[str, Any]]:
    return await get_store().list_tag_categories()


@router.get("/activity", response_model=ActivityEnvelope)
async def list_activity(
    kind: str | None = Query(default=None),
    sev: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    q: str | None = Query(default=None),
) -> dict[str, Any]:
    items, now_ms = await get_store().list_activity(
        kind=kind, sev=sev, actor=actor, q=q
    )
    return {"items": items, "total": len(items), "nowMs": now_ms}


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


# -- Tag mutation endpoints (S4c slice 2) -----------------------------------


async def _emit_tag_audit(
    *,
    store: WebuiStore,
    actor: str,
    kind: str,
    sev: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
    notification: dict[str, Any] | None,
) -> None:
    """Record an activity event and (optionally) push a notification."""
    event = _make_event(
        kind=kind,
        sev=sev,
        actor=actor,
        target_label=target_label,
        summary=summary,
        meta=meta,
    )
    await store.record_activity(event)
    if notification is not None:
        await store.push_notification(notification)


@router.post("/tags", response_model=TagEntry, status_code=201)
async def request_tag(body: TagRequestBody) -> dict[str, Any]:
    """Propose a new tag (tier='requested', status='pending-review')."""
    store = get_store()
    existing = await store.tags.get_tag(body.tag)
    if existing is not None:
        raise HTTPException(409, f"Tag '{body.tag}' already exists")
    actor = body.actor or "system"
    now = _utcnow_iso()[:10]
    entry: dict[str, Any] = {
        "tag": body.tag,
        "tier": "requested",
        "category": body.category,
        "status": "pending-review",
        "def": body.def_,
        "aliases": list(body.aliases),
        "deprecates": [],
        "sources_count": 0,
        "chunks_count": 0,
        "query_freq_30d": 0,
        "created": {"by": actor, "at": now},
        "last_edit": {"by": actor, "at": now, "action": "requested"},
        "related": [],
        "examples": [],
        "requested_by": actor,
        "requested_at": now,
        "justification": body.justification or "",
    }
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=body.tag,
        summary=f"Tag {body.tag} requested for palier-3 review",
        meta={"category": body.category, "justification": body.justification},
        notification=_make_notification(
            title="Tag",
            tagname=body.tag,
            suffix="requested",
            sub=body.justification or f"category {body.category}",
        ),
    )
    return stored


@router.post("/tags/{name}/approve", response_model=TagEntry)
async def approve_tag(name: str, body: TagApproveBody) -> dict[str, Any]:
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = body.actor or "system"
    now = _utcnow_iso()[:10]
    entry["tier"] = 3
    entry["status"] = "active"
    entry["last_edit"] = {"by": actor, "at": now, "action": "approved"}
    entry.pop("requested_by", None)
    entry.pop("requested_at", None)
    entry.pop("justification", None)
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} approved (now Tier 3)",
        meta={"tier": 3, "status": "active"},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="approved",
            sub="Added to thesaurus · Tier 3",
        ),
    )
    return stored


@router.post("/tags/{name}/reject", response_model=TagEntry)
async def reject_tag(name: str, body: TagRejectBody) -> dict[str, Any]:
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = body.actor or "system"
    now = _utcnow_iso()[:10]
    entry["status"] = "rejected"
    entry["last_edit"] = {"by": actor, "at": now, "action": "rejected"}
    entry["reject_reason"] = body.reason
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="warning",
        target_label=name,
        summary=f"Tag {name} rejected — {body.reason}",
        meta={"reason": body.reason},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="rejected",
            sub=body.reason,
        ),
    )
    return stored


@router.patch("/tags/{name}", response_model=TagEntry)
async def edit_tag(name: str, body: TagEditBody) -> dict[str, Any]:
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = body.actor or "system"
    now = _utcnow_iso()[:10]
    changed: list[str] = []
    if body.def_ is not None and body.def_ != entry.get("def"):
        entry["def"] = body.def_
        changed.append("def")
    if body.category is not None and body.category != entry.get("category"):
        entry["category"] = body.category
        changed.append("category")
    if body.aliases is not None and body.aliases != entry.get("aliases"):
        entry["aliases"] = list(body.aliases)
        changed.append("aliases")
    if body.deprecates is not None:
        entry["deprecates"] = list(body.deprecates)
        changed.append("deprecates")
    entry["last_edit"] = {"by": actor, "at": now, "action": "edited"}
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} edited ({', '.join(changed) or 'no-op'})",
        meta={"fields": changed},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="updated",
            sub=", ".join(changed) or "no field change",
        ),
    )
    return stored


@router.post("/tags/{name}/deprecate", response_model=TagEntry)
async def deprecate_tag(name: str, body: TagDeprecateBody) -> dict[str, Any]:
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = body.actor or "system"
    now = _utcnow_iso()[:10]
    entry["status"] = "deprecated"
    entry["last_edit"] = {"by": actor, "at": now, "action": "deprecated"}
    if body.reason:
        entry["deprecate_reason"] = body.reason
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="warning",
        target_label=name,
        summary=f"Tag {name} deprecated"
        + (f" — {body.reason}" if body.reason else ""),
        meta={"reason": body.reason},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="deprecated",
            sub=body.reason or "Excluded from default retrieval",
        ),
    )
    return stored


@router.post("/tags/{name}/synonyms", response_model=TagEntry)
async def update_synonyms(name: str, body: TagSynonymsBody) -> dict[str, Any]:
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = body.actor or "system"
    now = _utcnow_iso()[:10]
    entry["aliases"] = list(body.aliases)
    entry["last_edit"] = {"by": actor, "at": now, "action": "synonyms updated"}
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} synonyms updated ({len(body.aliases)} alias)",
        meta={"aliases": list(body.aliases)},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="synonyms updated",
            sub=", ".join(body.aliases) if body.aliases else "no aliases",
        ),
    )
    return stored


@router.delete("/tags/{name}", response_model=AckResponse)
async def delete_tag(
    name: str, body: TagDeleteBody | None = None
) -> dict[str, bool]:
    """Delete a tag. With strategy='migrate' the request must carry `to`; the
    endpoint surfaces the migration intent in audit + notification but
    document re-tagging (which lives elsewhere) is out of scope for slice 2."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    payload = body or TagDeleteBody()
    if payload.strategy == "migrate" and not payload.to:
        raise HTTPException(422, "strategy=migrate requires 'to'")
    actor = payload.actor or "system"
    deleted = await store.tags.delete_tag(name)
    suffix = (
        f"migrated to {payload.to}"
        if payload.strategy == "migrate"
        else "deleted (docs untagged)"
    )
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="warning",
        target_label=name,
        summary=f"Tag {name} {suffix}",
        meta={
            "strategy": payload.strategy,
            "to": payload.to,
            "sources_count_at_delete": entry.get("sources_count", 0),
        },
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix=suffix,
            sub=f"{entry.get('sources_count', 0)} docs affected",
        ),
    )
    return {"ok": deleted}


__all__ = [
    "router",
    "WebuiStore",
    "get_store",
    "set_store",
    "reset_store",
    "OpenApiGroup",
]
