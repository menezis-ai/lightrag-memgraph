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
import json
import secrets
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from . import webui_seed
from .idp_jwt import require_admin_user
from .folder import (
    bind_request_folder,
    current_folder_id,
    is_env_seeded_folder,
    load_folder_catalog,
)
from . import folder_store
from .webui_activitystore import InMemoryActivityStore, MemgraphActivityStore
from .webui_models import (
    AckResponse,
    ActivityEnvelope,
    Document,
    GraphEntity,
    GraphEntityCreate,
    GraphEntityPatch,
    GraphRelation,
    GraphRelationCreate,
    GraphRelationPatch,
    ListEnvelope,
    Notification,
    OpenApiEnvelope,
    OpenApiGroup,
    FolderCreate,
    FolderPatch,
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
    Folder,
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
        notification_backend: InMemoryNotificationStore
        | MemgraphNotificationStore
        | None = None,
    ) -> None:
        self._documents = documents
        self._folders = folders
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
            folders=copy.deepcopy(webui_seed.FOLDERS),
            thesaurus=copy.deepcopy(webui_seed.THESAURUS),
            tag_categories_seed=copy.deepcopy(webui_seed.TAG_CATEGORIES),
            tags_seed=copy.deepcopy(webui_seed.TAGS),
            openapi_groups=copy.deepcopy(webui_seed.OPENAPI_GROUPS),
            openapi_version=webui_seed.OPENAPI_VERSION,
            graph_entities=copy.deepcopy(webui_seed.GRAPH_ENTITIES),
            graph_relations=copy.deepcopy(webui_seed.GRAPH_RELATIONS),
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
        default_folder = load_folder_catalog().default_folder_id
        active_folder = current_folder_id()
        items = [
            d
            for d in self._documents
            if (d.get("folder") or d.get("metadata", {}).get("folder") or default_folder)
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

    async def push_notification(
        self, notification: dict[str, Any]
    ) -> dict[str, Any]:
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
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        return await self._activity_backend.list(
            kind=kind, sev=sev, actor=actor, q=q, limit=limit
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


# ---------------------------------------------------------------------------
# Document overlay helpers
# ---------------------------------------------------------------------------


def _status_to_dict(doc: Any) -> dict[str, Any]:
    """Normalize LightRAG DocStatus rows returned as dicts or dataclasses."""
    if isinstance(doc, dict):
        payload = dict(doc)
    else:
        import dataclasses

        payload = dataclasses.asdict(doc) if dataclasses.is_dataclass(doc) else {}
    status = payload.get("status")
    if hasattr(status, "value"):
        payload["status"] = status.value
    payload["metadata"] = _coerce_doc_metadata(payload.get("metadata"))
    return payload


def _coerce_doc_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _doc_matches_active_folder(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata") or {}
    default_folder = load_folder_catalog().default_folder_id
    return metadata.get("folder", default_folder) == current_folder_id()


async def _get_doc_for_active_folder(doc_id: str) -> dict[str, Any]:
    rag = _get_rag()
    raw = await rag.doc_status.get_by_id(doc_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    doc = _status_to_dict(raw)
    if not _doc_matches_active_folder(doc):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


async def _graph_tags_for_doc(doc_id: str) -> list[str]:
    """Best-effort doc tag lookup through [:TAGGED_WITH] relations."""
    try:
        from .. import _pool
        from .._constants import resolve_workspace

        workspace = resolve_workspace()
        folder = current_folder_id()
        doc_label = f"DocStatus_{workspace}"
        tag_label = f"WebuiTag_{folder}"
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                MATCH (d:`{doc_label}` {{id: $doc_id}})
                OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:`{tag_label}`)
                RETURN collect(t.id) AS tags
                """,
                doc_id=doc_id,
            )
            record = await result.single()
            await result.consume()
        return sorted(tid for tid in ((record or {}).get("tags") or []) if tid)
    except Exception:
        return []


async def _attach_graph_tags_for_documents(docs: list[dict[str, Any]]) -> None:
    """Attach graph-backed tag ids to WebUI document list rows."""
    if not docs:
        return
    try:
        from .. import _pool
        from .._constants import resolve_workspace

        workspace = resolve_workspace()
        folder = current_folder_id()
        doc_label = f"DocStatus_{workspace}"
        tag_label = f"WebuiTag_{folder}"
        doc_ids = [doc["doc_id"] for doc in docs if doc.get("doc_id")]
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS docId
                MATCH (d:`{doc_label}` {{id: docId}})
                OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:`{tag_label}`)
                RETURN docId, collect(t.id) AS tags
                """,
                ids=doc_ids,
            )
            tags_by_id: dict[str, list[str]] = {}
            async for record in result:
                tags_by_id[record["docId"]] = sorted(
                    tag for tag in (record["tags"] or []) if tag
                )
            await result.consume()
        for doc in docs:
            doc["tags"] = tags_by_id.get(doc.get("doc_id") or "", [])
    except Exception:
        for doc in docs:
            doc.setdefault("tags", [])


def _webui_doc_status(raw: Any) -> str:
    status = raw.value if hasattr(raw, "value") else str(raw or "")
    return status.upper()


def _status_filter_for_doc_status(status: str | None) -> str | None:
    if not status or status.lower() == "all":
        return None
    normalized = status.lower()
    if normalized in ("completed", "processed"):
        return "processed"
    if normalized in ("pending", "processing", "failed"):
        return normalized
    upper_map = {
        "processed": "processed",
        "pending": "pending",
        "processing": "processing",
        "failed": "failed",
    }
    return upper_map.get(status.upper().removeprefix("DOCSTATUS.").lower())


def _infer_document_type(file_path: str, metadata: dict[str, Any]) -> str:
    raw_type = str(metadata.get("type") or metadata.get("source_type") or "").lower()
    if raw_type in {"file", "confluence", "sharepoint", "url"}:
        return raw_type
    lowered = file_path.lower()
    if lowered.startswith(("http://", "https://")):
        return "url"
    if "confluence" in lowered:
        return "confluence"
    if "sharepoint" in lowered:
        return "sharepoint"
    return "file"


def _project_doc_status_for_webui(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = _coerce_doc_metadata(doc.get("metadata"))
    doc_id = str(doc.get("id") or doc.get("doc_id") or "")
    file_path = str(doc.get("file_path") or doc.get("source") or doc_id)
    summary = str(doc.get("content_summary") or doc.get("summary") or "")
    folder = str(metadata.get("folder") or current_folder_id())
    updated_at = str(
        doc.get("updated_at")
        or doc.get("created_at")
        or metadata.get("updated_at")
        or metadata.get("processing_end_time")
        or _utcnow_iso()
    )
    chunks_count = doc.get("chunks_count")
    if chunks_count is None:
        chunks = doc.get("chunks")
        chunks_count = chunks if isinstance(chunks, int) else 0
    content_length = doc.get("content_length")
    if content_length is None:
        content_length = len(summary)
    return {
        "id": doc_id,
        "doc_id": doc_id,
        "track_id": doc.get("track_id"),
        "type": _infer_document_type(file_path, metadata),
        "source": file_path,
        "file_path": file_path,
        "summary": summary,
        "content_summary": summary,
        "content_length": content_length,
        "tags": list(doc.get("tags") or metadata.get("tags") or []),
        "status": _webui_doc_status(doc.get("status")),
        "chunks": chunks_count,
        "chunks_count": chunks_count,
        "updated": updated_at,
        "updated_at": updated_at,
        "created_at": str(doc.get("created_at") or updated_at),
        "error_msg": doc.get("error_msg"),
        "visibility": str(metadata.get("visibility") or "internal"),
        "folder": folder,
        "review": metadata.get("review"),
        "metadata": metadata,
    }


def _filter_doc_status_rows(
    items: list[dict[str, Any]],
    *,
    q: str | None,
    tag: str | None,
) -> list[dict[str, Any]]:
    folder = current_folder_id()
    default_folder = load_folder_catalog().default_folder_id
    filtered = [
        doc
        for doc in items
        if (doc.get("metadata") or {}).get("folder", default_folder) == folder
    ]
    if q:
        needle = q.lower()
        filtered = [
            doc
            for doc in filtered
            if needle in str(doc.get("file_path") or doc.get("source") or "").lower()
            or needle in str(doc.get("content_summary") or doc.get("summary") or "").lower()
        ]
    if tag:
        filtered = [doc for doc in filtered if tag in (doc.get("tags") or [])]
    return filtered


async def _list_documents_from_doc_status(
    *,
    status: str | None,
    q: str | None,
    tag: str | None,
) -> list[dict[str, Any]]:
    rag = _get_rag()
    status_value = _status_filter_for_doc_status(status)
    status_filter = None
    if status_value:
        from lightrag.base import DocStatus

        try:
            status_filter = DocStatus(status_value)
        except ValueError:
            return []

    docs_tuples, _total = await rag.doc_status.get_docs_paginated(
        page=1,
        page_size=500,
        status_filter=status_filter,
    )
    docs: list[dict[str, Any]] = []
    for doc_id, raw in docs_tuples:
        payload = _status_to_dict(raw)
        payload["id"] = doc_id
        docs.append(_project_doc_status_for_webui(payload))

    await _attach_graph_tags_for_documents(docs)
    return _filter_doc_status_rows(docs, q=q, tag=tag)


def _cascade_seed_document_tags(
    store: WebuiStore,
    *,
    name: str,
    strategy: str,
    to: str | None,
) -> int:
    """Apply tag delete/migrate semantics to the in-memory document seed."""
    default_folder = load_folder_catalog().default_folder_id
    active_folder = current_folder_id()
    affected = 0

    def _rewrite(tags: Any) -> list[str] | None:
        if not isinstance(tags, list) or name not in tags:
            return None
        rewritten = [tag for tag in tags if tag != name]
        if strategy == "migrate" and to and to not in rewritten:
            rewritten.append(to)
        return rewritten

    with store._lock:  # noqa: SLF001 - same-module store maintenance
        for doc in store._documents:  # noqa: SLF001 - same-module store maintenance
            metadata = doc.get("metadata") or {}
            folder = doc.get("folder") or metadata.get("folder") or default_folder
            if folder != active_folder:
                continue
            rewritten = _rewrite(doc.get("tags"))
            if rewritten is None:
                continue
            doc["tags"] = rewritten
            if isinstance(metadata, dict):
                metadata_tags = _rewrite(metadata.get("tags"))
                if metadata_tags is not None:
                    metadata["tags"] = metadata_tags
            affected += 1
    return affected


async def _cascade_graph_tag_edges(
    *,
    name: str,
    strategy: str,
    to: str | None,
    actor: str,
    strict: bool,
) -> int | None:
    """Retag or untag DocStatus->WebuiTag edges for the active folder.

    Returns ``None`` when the graph pool is unavailable in non-strict seed/dev
    mode. In strict Memgraph-backed mode, failures surface as 500 so the API
    does not report a successful migration while documents were left stale.
    """
    try:
        from .. import _pool
        from .._constants import resolve_workspace
        workspace = resolve_workspace()
        folder = current_folder_id()
        doc_label = f"DocStatus_{workspace}"
        tag_label = f"WebuiTag_{folder}"
        now = _utcnow_iso()

        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                if strategy == "migrate":
                    result = await session.run(
                        f"""
                        MATCH (from:`{tag_label}` {{id: $from_tag}})
                        MATCH (to:`{tag_label}` {{id: $to_tag}})
                        MATCH (d:`{doc_label}`)-[old:TAGGED_WITH]->(from)
                        MERGE (d)-[new_rel:TAGGED_WITH]->(to)
                          ON CREATE SET
                            new_rel.at = $now,
                            new_rel.actor = $actor,
                            new_rel.migrated_from = $from_tag
                        DELETE old
                        RETURN count(DISTINCT d) AS affected
                        """,
                        from_tag=name,
                        to_tag=to,
                        now=now,
                        actor=actor,
                    )
                else:
                    result = await session.run(
                        f"""
                        MATCH (d:`{doc_label}`)-[old:TAGGED_WITH]->(:`{tag_label}` {{id: $tag}})
                        DELETE old
                        RETURN count(DISTINCT d) AS affected
                        """,
                        tag=name,
                    )
                record = await result.single()
                await result.consume()
        return int(record["affected"]) if record else 0
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise HTTPException(
                status_code=500,
                detail="Tag delete migration cascade failed.",
            ) from exc
        return None


async def _delete_doc_from_rag(rag: Any, doc_id: str) -> None:
    if hasattr(rag, "adelete_by_doc_id"):
        await rag.adelete_by_doc_id(doc_id)
        return
    await rag.doc_status.delete([doc_id])


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["webui"], dependencies=[Depends(bind_request_folder)])


# -- Read endpoints ----------------------------------------------------------


@router.get("/documents", response_model=ListEnvelope[Document])
async def list_documents(
    status: str | None = Query(default=None),
    q: str | None = Query(default=None),
    tag: str | None = Query(default=None),
) -> dict[str, Any]:
    store = get_store()
    with store._lock:  # noqa: SLF001 - same-module route/store coordination
        has_seed_documents = bool(store._documents)  # noqa: SLF001
    if has_seed_documents:
        items = store.list_documents(status=status, q=q, tag=tag)
        return {"items": items, "total": len(items)}
    try:
        items = await _list_documents_from_doc_status(status=status, q=q, tag=tag)
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        items = store.list_documents(status=status, q=q, tag=tag)
    return {"items": items, "total": len(items)}


@router.get("/documents/{doc_id}/metadata")
async def get_document_metadata(doc_id: str) -> dict[str, Any]:
    doc = await _get_doc_for_active_folder(doc_id)
    metadata = doc.get("metadata") or {}
    graph_tags = await _graph_tags_for_doc(doc_id)
    tags = graph_tags or list(metadata.get("tags") or doc.get("tags") or [])
    folder = metadata.get("folder") or current_folder_id()
    return {
        "tags": tags,
        "folder": folder,
        "review": metadata.get("review"),
        "classification": metadata.get("classification"),
        "metadata": metadata,
    }


@router.post("/documents/bulk-delete")
async def bulk_delete_documents(body: dict[str, Any]) -> dict[str, Any]:
    doc_ids = body.get("doc_ids")
    actor = body.get("actor") or "system"
    if not isinstance(doc_ids, list) or not doc_ids:
        raise HTTPException(
            status_code=400,
            detail="doc_ids must be a non-empty list of document ids.",
        )
    if len(doc_ids) > 500:
        raise HTTPException(
            status_code=413,
            detail="bulk-delete accepts at most 500 target documents.",
        )

    rag = _get_rag()
    deleted = 0
    failed: list[str] = []
    for doc_id in doc_ids:
        if not isinstance(doc_id, str) or not doc_id:
            failed.append(str(doc_id))
            continue
        try:
            doc = await _get_doc_for_active_folder(doc_id)
            await _delete_doc_from_rag(rag, doc_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                failed.append(doc_id)
                continue
            raise
        except Exception:
            failed.append(doc_id)
            continue

        deleted += 1
        event = _make_event(
            kind="doc-deleted",
            sev="info",
            actor=actor,
            target_label=doc.get("file_path") or doc_id,
            summary=f"deleted by {actor}",
            meta={"doc_id": doc_id, "operation": "bulk-delete"},
            target_type="document",
        )
        await get_store().record_activity(event)

    return {"deleted": deleted, "failed": failed}


@router.get("/health")
async def twin_health() -> dict[str, Any]:
    try:
        _get_rag()
        rag_captured = True
    except HTTPException:
        rag_captured = False

    store = get_store()
    stores = {
        "tags": store.tags.__class__.__name__,
        "activity": store.activity.__class__.__name__,
        "notifications": store.notifications.__class__.__name__,
    }
    return {
        "status": "ok" if rag_captured else "degraded",
        "folder": current_folder_id(),
        "ragCaptured": rag_captured,
        "stores": stores,
    }


@router.get("/folders", response_model=list[Folder])
async def list_folders() -> list[dict[str, Any]]:
    active = current_folder_id()
    return [
        folder.as_api(current=folder.id == active)
        for folder in load_folder_catalog().folders
    ]


@router.post(
    "/folders",
    response_model=Folder,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
)
async def create_folder(body: FolderCreate) -> dict[str, Any]:
    """Admin: provision a new Twin folder at runtime.

    Returns 201 + the new folder. Errors:
    - 409 if an env-seeded folder already owns this id (operator cannot
      shadow an SRE-provisioned default).
    - 409 if a runtime folder with this id already exists.
    - 422 if the id fails the safe-identifier rule.
    - 422 if adding this folder would exceed `maxFolders` from the env
      configuration.
    """
    catalog = load_folder_catalog()
    if len(catalog.folders) >= catalog.max_folders:
        raise HTTPException(
            422,
            f"Cannot create folder: catalog already at max ({catalog.max_folders}). "
            "Remove an existing folder first.",
        )
    if is_env_seeded_folder(body.id):
        raise HTTPException(
            409,
            f"Folder id '{body.id}' is provisioned by the deploy env "
            "and cannot be re-created via the API.",
        )
    try:
        folder = folder_store.add_runtime_folder(
            folder_id=body.id,
            label=body.label,
            kind=body.kind,
            description=body.description,
        )
    except KeyError as exc:
        raise HTTPException(
            409, f"Folder '{exc.args[0]}' already exists"
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store = get_store()
    event = _make_event(
        kind="settings",
        sev="info",
        actor="operator",
        target_label=folder.label,
        summary=f"Folder '{folder.id}' created ({folder.kind})",
        meta={"folder_id": folder.id, "operation": "create"},
        target_type="folder",
    )
    await store.record_activity(event)
    return folder.as_api(current=False)


@router.patch(
    "/folders/{folder_id}",
    response_model=Folder,
    dependencies=[Depends(require_admin_user)],
)
async def update_folder(folder_id: str, body: FolderPatch) -> dict[str, Any]:
    """Admin: edit label / kind / description of a runtime folder.

    Env-seeded folders are 403 — those changes must go through the
    deploy env (`TWIN_FOLDERS_JSON`).
    """
    if is_env_seeded_folder(folder_id):
        raise HTTPException(
            403,
            f"Folder '{folder_id}' is env-seeded and cannot be edited via the API.",
    )
    patch = body.model_dump(exclude_unset=True)
    folder = folder_store.update_runtime_folder(
        folder_id,
        label=patch.get("label"),
        kind=patch.get("kind"),
        description=patch.get("description"),
    )
    if folder is None:
        raise HTTPException(404, f"Folder '{folder_id}' not found")
    store = get_store()
    event = _make_event(
        kind="settings",
        sev="info",
        actor="operator",
        target_label=folder.label,
        summary=f"Folder '{folder.id}' updated",
        meta={
            "folder_id": folder.id,
            "operation": "update",
            "patch_keys": list(patch.keys()),
        },
        target_type="folder",
    )
    await store.record_activity(event)
    active = current_folder_id()
    return folder.as_api(current=folder.id == active)


@router.delete(
    "/folders/{folder_id}",
    status_code=204,
    dependencies=[Depends(require_admin_user)],
)
async def delete_folder(folder_id: str) -> None:
    """Admin: remove a runtime folder.

    - 403 if env-seeded (only the deploy env can remove those).
    - 404 if no runtime folder with this id exists.
    - 409 if the folder still has WebUI data (tags, activity events,
      docs scoped to it). Refusing to delete avoids orphaning state.
    """
    if is_env_seeded_folder(folder_id):
        raise HTTPException(
            403,
            f"Folder '{folder_id}' is env-seeded and cannot be deleted via the API.",
        )
    # Probe the in-memory store for residual data before deleting.
    bound_store = _stores.get(folder_id)
    if bound_store is not None:
        has_docs = len(bound_store._documents) > 0  # noqa: SLF001
        # `list_tags` is sync on the in-memory backend, async on the
        # Memgraph one — normalise via inspect.iscoroutine.
        tags_result = bound_store.tags.list_tags()
        if hasattr(tags_result, "__await__"):
            tags_result = await tags_result  # type: ignore[assignment]
        has_tags = len(tags_result) > 0
        if has_docs or has_tags:
            raise HTTPException(
                409,
                f"Folder '{folder_id}' still has data (docs and/or tags). "
                "Remove the contents before deleting the folder.",
            )
    if not folder_store.delete_runtime_folder(folder_id):
        raise HTTPException(404, f"Folder '{folder_id}' not found")
    # Evict the per-folder WebUI store so future GETs don't resurrect it.
    _stores.pop(folder_id, None)
    store = get_store()
    event = _make_event(
        kind="settings",
        sev="info",
        actor="operator",
        target_label=folder_id,
        summary=f"Folder '{folder_id}' deleted",
        meta={"folder_id": folder_id, "operation": "delete"},
        target_type="folder",
    )
    await store.record_activity(event)
    return None


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
    return await get_store().list_thesaurus()


@router.get("/tags", response_model=list[TagEntry])
async def list_tags() -> list[dict[str, Any]]:
    return await get_store().list_tags()


@router.get("/tags/categories", response_model=list[TagCategory])
async def list_tag_categories() -> list[dict[str, Any]]:
    return await get_store().list_tag_categories()


# ------------------------------------------------------------------
# Categories template + import — admin convenience for non-shell ops
# ------------------------------------------------------------------
#
# Doctrine recap: categories are governance taxonomy, not user-generated.
# The default flow is Config-as-Code (mount a JSON via ConfigMap, restart
# Twin). These two endpoints exist for admins who own the JSON but lack
# shell access on the host — the upload mirrors Memgraph in-place using
# the exact same validator as `webui_categories_config`.

_CATEGORIES_TEMPLATE: list[dict[str, Any]] = [
    {"id": "network", "label": "Network", "color": "#1F8A7A"},
    {"id": "infra", "label": "Infrastructure", "color": "#5A7FB4"},
    {"id": "compliance", "label": "Compliance", "color": "#9C2D8E"},
    {"id": "operations", "label": "Operations", "color": "#C24A24"},
    {"id": "governance", "label": "Governance", "color": "#2C3E50"},
    {"id": "lifecycle", "label": "Lifecycle", "color": "#8A5C0E"},
]


@router.get("/tags/categories/template")
async def get_categories_template():
    """Return the canonical template JSON that operators can save + edit.

    Served as application/json with a Content-Disposition so a browser
    "Download template" button receives a file save dialog rather than
    rendering inline. The 6 entries here mirror
    ``docs/templates/twin-categories.template.json`` and the schema
    lives at ``docs/templates/twin-categories.schema.json``.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        content=_CATEGORIES_TEMPLATE,
        headers={
            "Content-Disposition": (
                'attachment; filename="twin-categories.template.json"'
            ),
        },
    )


# ------------------------------------------------------------------
# Bulk-retag — persistent doc-level tagging (doctrine: tag is a
# Memgraph node attribute, not a separate lookup table).
# ------------------------------------------------------------------


def _get_rag():
    """Resolve the host LightRAG instance captured at register() time."""
    from .. import _twindb_state

    rag = _twindb_state.get("rag")
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Twin overlay: LightRAG instance not captured. The host must "
                "boot via register(shim_native_routes=True) so the rag "
                "captures the create_document_routes call."
            ),
        )
    return rag


@router.post("/documents/_bulk-retag")
async def bulk_retag_documents(
    body: dict[str, Any],
) -> dict[str, Any]:
    """Apply tag adds/removes to a list of documents as graph edges.

    Doctrine (refined): a tag is a **Memgraph node**
    (``WebuiTag_{workspace}``), not a string attribute. The fact that
    a document carries a tag is a **graph relation**
    ``(:DocStatus_{workspace})-[:TAGGED_WITH]->(:WebuiTag_{workspace})``.

    Why this shape:
      * Native Cypher queries on tags work: "all docs with rman AND
        production" = ``MATCH (d)-[:TAGGED_WITH]->({id:'rman'}),
        (d)-[:TAGGED_WITH]->({id:'production'}) RETURN d``.
      * Cascade on doc delete is automatic via ``DETACH DELETE``.
      * Foundation for V2: propagate the [:TAGGED_WITH] edge to chunks
        / entities / vectors so retrieval can be filtered server-side.
      * No string-in-JSON-array heresy — we use the graph engine as
        intended.

    Governance loop: a tag id that doesn't yet exist in the catalog
    gets ``MERGE``d on the fly with ``tier="requested"`` /
    ``status="pending-review"``. The Steward sees it in the Tags
    tab's "Pending requests" section and approves or rejects.
    Nothing leaks past the catalog by surprise.

    Body shape::

        {
          "targets": ["doc-abc", "doc-def", ...],
          "adds":    ["rman", "oracle"],
          "removes": ["deprecated"],
          "actor":   "claire.benoit"  # optional, falls back to "system"
        }

    Returns ``{updated: N, failed: [doc_id, ...]}`` — ``failed`` lists
    doc ids that didn't exist in DocStatus_{workspace} (the catalog
    auto-creation makes tag-not-found impossible by construction).
    """
    import json as _json

    from .. import _pool
    from .._constants import resolve_workspace

    targets = body.get("targets") or []
    adds = list(body.get("adds") or [])
    removes = list(body.get("removes") or [])
    actor = body.get("actor") or "system"

    if not isinstance(targets, list) or not targets:
        raise HTTPException(
            status_code=400,
            detail="targets must be a non-empty list of doc_id strings.",
        )
    if len(targets) > 500:
        raise HTTPException(
            status_code=413,
            detail="bulk-retag accepts at most 500 target documents.",
        )
    if len(adds) + len(removes) > 50:
        raise HTTPException(
            status_code=413,
            detail="bulk-retag accepts at most 50 tag mutations.",
        )
    for tag in (*adds, *removes):
        if not isinstance(tag, str) or not tag.strip():
            raise HTTPException(
                status_code=400,
                detail="tags must be non-empty strings.",
            )

    workspace = resolve_workspace()
    folder = current_folder_id()
    doc_label = f"DocStatus_{workspace}"
    tag_label = f"WebuiTag_{folder}"
    now = _utcnow_iso()

    placeholder = _json.dumps(
        {
            "tag": "<PLACEHOLDER>",  # overwritten per-tag via Cypher SET n.data
            "tier": "requested",
            "category": "uncategorized",
            "status": "pending-review",
            "def": "Auto-created via retag — needs Steward review.",
            "aliases": [],
            "deprecates": [],
            "sources_count": 0,
            "chunks_count": 0,
            "query_freq_30d": 0,
            "related": [],
            "examples": [],
            "requested_by": actor,
            "requested_at": now,
            "justification": "auto-created via retag",
            "created": {
                "by": actor,
                "at": now[:10],
                "action": "auto-requested-via-retag",
            },
            "last_edit": {
                "by": actor,
                "at": now[:10],
                "action": "auto-requested-via-retag",
            },
        },
        sort_keys=True,
    )

    # 1) Probe which target docs actually exist + capture file_path
    # for the audit event labels.
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"""
            UNWIND $ids AS id
            MATCH (n:`{doc_label}` {{id: id}})
            RETURN n.id AS id, n.file_path AS file_path
            """,
            ids=targets,
        )
        existing: dict[str, str | None] = {}
        async for record in result:
            existing[record["id"]] = record.get("file_path")
        await result.consume()

    failed = [t for t in targets if t not in existing]
    if not existing:
        return {"updated": 0, "failed": failed}

    doc_ids = list(existing.keys())

    # 2) Apply adds/removes in one bounded write transaction. Adds run before
    # removes so a payload containing the same tag in both lists settles on
    # "removed" deterministically.
    if adds or removes:
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                tx = await session.begin_transaction()
                try:
                    if adds:
                        tag_rows = [
                            {
                                "id": tag_id,
                                "data": placeholder.replace(
                                    '"<PLACEHOLDER>"', _json.dumps(tag_id)
                                ),
                            }
                            for tag_id in adds
                        ]
                        result = await tx.run(
                            f"""
                            UNWIND $tags AS tag
                            MERGE (t:`{tag_label}` {{id: tag.id}})
                              ON CREATE SET
                                t.data = tag.data,
                                t.`__created_at` = timestamp(),
                                t.`__auto_via_retag` = true
                              SET t.`__updated_at` = timestamp()
                            WITH t
                            UNWIND $docs AS docId
                            MATCH (d:`{doc_label}` {{id: docId}})
                            MERGE (d)-[r:TAGGED_WITH]->(t)
                              ON CREATE SET r.at = $now, r.actor = $actor
                            """,
                            tags=tag_rows,
                            docs=doc_ids,
                            now=now,
                            actor=actor,
                        )
                        await result.consume()

                    if removes:
                        result = await tx.run(
                            f"""
                            UNWIND $docs AS docId
                            UNWIND $tags AS tagId
                            MATCH (d:`{doc_label}` {{id: docId}})-[r:TAGGED_WITH]->(t:`{tag_label}` {{id: tagId}})
                            DELETE r
                            """,
                            docs=doc_ids,
                            tags=removes,
                        )
                        await result.consume()

                    await tx.commit()
                except Exception:
                    await tx.rollback()
                    raise

    # 3) Emit one audit event per doc with the resulting tag list
    # fetched via the relation (single Cypher batch round-trip).
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"""
            UNWIND $docs AS docId
            MATCH (d:`{doc_label}` {{id: docId}})
            OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:`{tag_label}`)
            RETURN docId, collect(t.id) AS tags
            """,
            docs=doc_ids,
        )
        resulting_by_doc: dict[str, list[str]] = {}
        async for record in result:
            tags = sorted(tid for tid in (record["tags"] or []) if tid)
            resulting_by_doc[record["docId"]] = tags
        await result.consume()

    for doc_id in doc_ids:
        new_tags = resulting_by_doc.get(doc_id, [])
        event = _make_event(
            kind="doc-retagged",
            sev="info",
            actor=actor,
            target_label=existing[doc_id] or doc_id,
            summary=(
                f"tags: +{','.join(adds) or '∅'} -{','.join(removes) or '∅'}"
            ),
            meta={
                "doc_id": doc_id,
                "adds": adds,
                "removes": removes,
                "resulting_tags": new_tags,
            },
            target_type="document",
        )
        await get_store().record_activity(event)

    return {"updated": len(doc_ids), "failed": failed}


@router.post("/auth/logout", response_model=AckResponse)
async def logout() -> dict[str, Any]:
    """Sign out the current operator.

    Under the current Traefik Basic Auth gate, sign-out is mostly a
    client-side concern (clear React Query cache + reload to retrigger
    the browser's auth prompt). The endpoint exists so the frontend
    can confirm round-trip before clearing local state — when JWT/IdP
    arrives (Couche 3 §3.3), this also clears the HttpOnly cookie
    via Set-Cookie: Max-Age=0.

    Returns {ok: true} always — sign-out cannot fail server-side
    under the current model.
    """
    from fastapi.responses import JSONResponse

    response = JSONResponse(content={"ok": True})
    # Pre-emptive cookie clear for the future JWT flow. Currently a
    # no-op because Basic Auth uses HTTP headers, not cookies.
    response.delete_cookie("twin_session", path="/")
    response.delete_cookie("twin_id_token", path="/")
    return response


@router.post("/documents/uploads/activity", response_model=AckResponse)
async def record_source_uploaded(
    body: dict[str, Any],
) -> dict[str, bool]:
    """Record Activity for LightRAG-native upload accepts.

    The actual upload endpoint is the native LightRAG
    ``/documents/upload`` route, outside this Twin router. The WebUI
    calls this route only after that native endpoint accepts a file so
    the audit feed still has a durable ``source-uploaded`` event.
    """
    source = str(body.get("source") or "").strip()
    if not source:
        raise HTTPException(
            status_code=400,
            detail="record_source_uploaded requires a non-empty source.",
        )
    actor = str(body.get("actor") or "system").strip() or "system"
    track_id = str(body.get("track_id") or "").strip()
    status = str(body.get("status") or "accepted").strip() or "accepted"
    event = _make_event(
        kind="source-uploaded",
        sev="info",
        actor=actor,
        target_label=source,
        summary=f"uploaded by {actor}",
        meta={"source": source, "track_id": track_id, "status": status},
        target_type="source",
    )
    await get_store().record_activity(event)
    return {"ok": True}


@router.post("/documents/{doc_id}/approve")
async def approve_document(
    doc_id: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark a document as reviewer-approved.

    Persists ``DocStatus.metadata.review = {state: 'approved',
    actor, at, edits?}`` on the Memgraph node and emits a
    ``doc-approved`` activity event. The ``edits`` (optional) carries
    operator-supplied corrections that were applied at the same time
    as the approval — the front-end's EditApproveModal sends these
    alongside the approve when the reviewer needed to fix something
    before signing off.
    """
    rag = _get_rag()
    body = body or {}
    actor = body.get("actor") or "system"
    edits = body.get("edits") or {}

    doc = await rag.doc_status.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    metadata = doc.get("metadata") or {}
    review = metadata.get("review") or {}
    review.update(
        {
            "state": "approved",
            "actor": actor,
            "at": _utcnow_iso(),
        }
    )
    if edits:
        review["edits"] = edits
    metadata["review"] = review
    doc["metadata"] = metadata
    await rag.doc_status.upsert({doc_id: doc})

    event = _make_event(
        kind="doc-approved",
        sev="info",
        actor=actor,
        target_label=doc.get("file_path") or doc_id,
        summary=(
            f"approved by {actor}" + (f" with edits" if edits else "")
        ),
        meta={"doc_id": doc_id, "edits": edits},
        target_type="document",
    )
    await get_store().record_activity(event)

    return {"doc_id": doc_id, "review": review}


@router.post("/documents/{doc_id}/reject")
async def reject_document(
    doc_id: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Mark a document as reviewer-rejected.

    Persists ``DocStatus.metadata.review = {state: 'rejected',
    actor, at, justification}`` on the Memgraph node and emits a
    ``doc-rejected`` activity event with the rejection reason in the
    summary (visible in the audit feed). The doc itself is NOT
    deleted — it stays in DocStatus with its rejected review so the
    operator can still see it in the table with the right badge.
    """
    rag = _get_rag()
    actor = body.get("actor") or "system"
    reason = body.get("reason") or ""
    if not reason:
        raise HTTPException(
            status_code=400,
            detail="reject_document requires a non-empty `reason` field.",
        )

    doc = await rag.doc_status.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    metadata = doc.get("metadata") or {}
    review = metadata.get("review") or {}
    review.update(
        {
            "state": "rejected",
            "actor": actor,
            "at": _utcnow_iso(),
            "justification": reason,
        }
    )
    metadata["review"] = review
    doc["metadata"] = metadata
    await rag.doc_status.upsert({doc_id: doc})

    event = _make_event(
        kind="doc-rejected",
        sev="warning",
        actor=actor,
        target_label=doc.get("file_path") or doc_id,
        summary=f"rejected: {reason}",
        meta={"doc_id": doc_id, "reason": reason},
        target_type="document",
    )
    await get_store().record_activity(event)

    return {"doc_id": doc_id, "review": review}


@router.post("/tags/categories/_import", response_model=AckResponse)
async def import_categories(body: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror the uploaded JSON into the active folder's categories store.

    Same validator as ``webui_categories_config`` — only JSON matching
    the schema is accepted. On success, returns ``{ok: True}`` and the
    operator's next page refresh sees the new taxonomy. Rejects with
    400 if the JSON shape is wrong, 503 if the store backend is not
    Memgraph (the seed/in-memory backend has no concept of "import").
    """
    backend = get_store().tags
    if not isinstance(backend, MemgraphTagStore):
        raise HTTPException(
            status_code=503,
            detail=(
                "Categories import requires the Memgraph store backend "
                "(register with webui_stores='memgraph'). The current "
                "backend does not support taxonomy mutation."
            ),
        )

    try:
        await backend.replace_categories_from_list(body, source="import")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True}


@router.get("/activity", response_model=ActivityEnvelope)
async def list_activity(
    kind: str | None = Query(default=None),
    sev: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    items, now_ms = await get_store().list_activity(
        kind=kind, sev=sev, actor=actor, q=q, limit=limit
    )
    return {"items": items, "total": len(items), "nowMs": now_ms}


@router.get("/openapi", response_model=OpenApiEnvelope)
async def get_openapi_groups() -> dict[str, Any]:
    groups, version = get_store().openapi()
    return {"groups": groups, "version": version}


def _graph_memgraph_label() -> str:
    """Resolve the Cypher label LightRAG uses for entity nodes.

    Until per-folder isolation lands at the LightRAG layer (one
    workspace per Twin folder), the KG view is globally scoped to the
    single LightRAG workspace configured by the deploy (env var
    `MEMGRAPH_WORKSPACE` / `WORKSPACE`). The Twin folder catalog still
    drives UX (default vs sandbox) but the underlying graph is shared.
    """
    from .._constants import resolve_workspace

    return resolve_workspace()


async def _validate_graph_entity_tags(
    tags: list[str] | None,
) -> None:
    """Enforce that node tags belong to the active tag catalog
    (TR-KG-03 / Alberto recette 2026-06-12).

    Both ``PATCH /twin/api/graph/entities/{id}`` and
    ``POST /twin/api/graph/entities`` accept a ``tags`` field on the
    request body. Before this gate, the helpers in
    ``graph_reader.py`` serialised whatever list was sent into
    ``twin_tags_json`` with no check — a curl/API caller could write
    arbitrary strings onto a node, contradicting the canonical tag
    catalog the WebUI surfaces.

    Allowed = ``{entry["tag"] for entry in await store.list_tags()
                  if entry.get("status") == "active"}``. Other statuses
    (``pending-promotion``, ``pending-review``, ``deprecated``,
    ``rejected``) are intentionally rejected: a non-active tag is not
    part of the operational vocabulary, and writing it on a graph
    node would silently re-promote / re-introduce it.

    On unknown tags, raises ``HTTPException(422)`` with a detail that
    lists the rejected values plus a bounded sample of allowed ones
    so the caller has actionable feedback without leaking the entire
    catalog when it is large.
    """
    if not tags:
        return
    store = get_store()
    catalog_entries = await store.list_tags()
    allowed = {
        entry["tag"]
        for entry in catalog_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("tag"), str)
        and entry.get("status") == "active"
    }
    unknown = sorted(
        {t for t in tags if isinstance(t, str)} - allowed
    )
    if not unknown:
        return
    # Bounded sample of allowed tags so the error stays readable on
    # large catalogs but still gives the caller a starting point.
    sample = sorted(allowed)[:10]
    suffix = "" if len(allowed) <= 10 else f" (+{len(allowed) - 10} more)"
    raise HTTPException(
        422,
        (
            f"Unknown node tag(s): {', '.join(unknown)}. "
            f"Allowed (active catalog): {', '.join(sample)}{suffix}."
        ),
    )


@router.get("/graph/entities", response_model=list[GraphEntity])
async def list_graph_entities() -> list[dict[str, Any]]:
    """Return the live Memgraph-backed entities for the deployed KB.

    Falls back to the in-memory seed when Memgraph is unreachable or
    contains no nodes yet (typical pre-ingestion). The fallback keeps
    demo / dev / standalone paths working without a backend.
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    entities = await graph_reader.read_graph_entities(label)
    if entities:
        return entities
    return get_store().list_graph_entities()


@router.get("/graph/relations", response_model=list[GraphRelation])
async def list_graph_relations() -> list[dict[str, Any]]:
    """Return the live Memgraph-backed relations for the deployed KB.

    Same fallback policy as `/graph/entities`. Relations are filtered to
    endpoints that survived the entity read so a truncated node set
    doesn't show dangling edges.
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    entities = await graph_reader.read_graph_entities(label)
    if entities:
        valid_ids = [e["id"] for e in entities]
        return await graph_reader.read_graph_relations(
            label, valid_node_ids=valid_ids
        )
    return get_store().list_graph_relations()


@router.patch("/graph/entities/{entity_id}", response_model=GraphEntity)
async def update_graph_entity_endpoint(
    entity_id: str, body: GraphEntityPatch
) -> dict[str, Any]:
    """Persist an edit to a graph entity in Memgraph.

    Returns the updated canonical projection on success, 404 if no
    node matches. The Twin overlay store also receives a
    ``graph-entity-edited`` activity event so the audit feed picks
    the action up.
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    patch_dict = body.model_dump(exclude_unset=True)
    # TR-KG-03: reject node tags outside the active catalog before
    # we let graph_reader serialize them onto ``twin_tags_json``.
    await _validate_graph_entity_tags(patch_dict.get("tags"))
    updated = await graph_reader.update_graph_entity(
        label, entity_id, patch_dict
    )
    if updated is None:
        raise HTTPException(
            404, f"Graph entity '{entity_id}' not found in workspace '{label}'"
        )
    store = get_store()
    event = _make_event(
        kind="graph-entity-edited",
        sev="info",
        actor="operator",
        target_label=updated.get("name") or entity_id,
        summary=f"Graph entity '{updated.get('name') or entity_id}' updated",
        meta={
            "entity_id": entity_id,
            "patch_keys": list(patch_dict.keys()),
        },
        target_type="entity",
    )
    await store.record_activity(event)
    return updated


@router.post(
    "/graph/entities", response_model=GraphEntity, status_code=201
)
async def create_graph_entity_endpoint(
    body: GraphEntityCreate,
) -> dict[str, Any]:
    """Manually add a new entity to the KB.

    Status contract (TR-KG-01):

    - 201 + projected entity on success.
    - 422 if the payload is malformed (Pydantic — empty/whitespace
      name, missing/invalid type, name longer than 255 chars).
    - 409 if an entity with the same canonical name already exists
      in the workspace. Manual creation never silently overwrites an
      LLM-extracted entry.
    - 503 if the Memgraph ``CREATE`` itself fails (driver down,
      session unavailable, lock contention). Body carries no driver
      detail; the full trace lands in server logs.
    - 500 if the write succeeded but the post-CREATE projection
      failed. The entity exists server-side — a fresh
      ``GET /twin/api/graph/entities`` will surface it.
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    payload = body.model_dump(exclude_unset=True)
    # TR-KG-03: same catalog-binding gate as the PATCH endpoint.
    await _validate_graph_entity_tags(payload.get("tags"))
    try:
        entity = await graph_reader.create_graph_entity(label, payload)
    except graph_reader.EntityExistsError:
        raise HTTPException(
            409,
            f"Graph entity '{body.name}' already exists in workspace '{label}'",
        )
    except graph_reader.EntityProjectionError:
        # The node was written but we can't read it back to project
        # it. Surface the half-success honestly instead of pretending
        # the write failed.
        raise HTTPException(
            500,
            (
                f"Graph entity '{body.name}' was created in workspace "
                f"'{label}' but the projection failed. Refresh "
                "/twin/api/graph/entities to surface it."
            ),
        )
    except graph_reader.EntityCreateBackendError:
        raise HTTPException(
            503,
            (
                f"Graph entity '{body.name}' could not be created: the "
                "Memgraph backend rejected the write. Check server logs "
                "for the underlying error."
            ),
        )
    store = get_store()
    event = _make_event(
        kind="graph-entity-edited",
        sev="info",
        actor="operator",
        target_label=entity.get("name") or body.name,
        summary=f"Graph entity '{entity.get('name') or body.name}' created",
        meta={
            "entity_id": entity["id"],
            "patch_keys": list(payload.keys()),
            "operation": "create",
        },
        target_type="entity",
    )
    await store.record_activity(event)
    return entity


@router.delete("/graph/entities/{entity_id}", status_code=204)
async def delete_graph_entity_endpoint(entity_id: str) -> None:
    """Remove an entity from the KB (cascade-deletes its edges).

    Returns 204 on success, 404 if the entity wasn't found. Stale
    relation ids referencing the deleted node are evicted from the
    endpoint cache so subsequent PATCH/DELETE on those edges fail
    cleanly with 404.
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    ok = await graph_reader.delete_graph_entity(label, entity_id)
    if not ok:
        raise HTTPException(
            404, f"Graph entity '{entity_id}' not found in workspace '{label}'"
        )
    store = get_store()
    event = _make_event(
        kind="graph-entity-edited",
        sev="info",
        actor="operator",
        target_label=entity_id,
        summary=f"Graph entity '{entity_id}' deleted",
        meta={"entity_id": entity_id, "operation": "delete"},
        target_type="entity",
    )
    await store.record_activity(event)
    return None


@router.post(
    "/graph/relations", response_model=GraphRelation, status_code=201
)
async def create_graph_relation_endpoint(
    body: GraphRelationCreate,
) -> dict[str, Any]:
    """Manually add a new relation between two entities.

    Returns 201 + projected relation. 422 if either endpoint doesn't
    exist in the workspace. The route is idempotent: re-issuing the
    same source/target pair MERGEs onto the existing edge instead of
    erroring.
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    payload = body.model_dump(exclude_unset=True)
    relation = await graph_reader.create_graph_relation(label, payload)
    if relation is None:
        raise HTTPException(
            422,
            "Cannot create relation — one or both endpoints are missing, "
            "or the label is empty.",
        )
    store = get_store()
    event = _make_event(
        kind="graph-relation-edited",
        sev="info",
        actor="operator",
        target_label=relation.get("label") or body.label,
        summary=f"Graph relation '{relation.get('label') or body.label}' created",
        meta={
            "rel_id": relation["id"],
            "source": body.source,
            "target": body.target,
            "operation": "create",
        },
        target_type="relation",
    )
    await store.record_activity(event)
    return relation


@router.delete("/graph/relations/{rel_id}", status_code=204)
async def delete_graph_relation_endpoint(rel_id: str) -> None:
    """Remove a relation from the KB."""
    from . import graph_reader

    label = _graph_memgraph_label()
    ok = await graph_reader.delete_graph_relation(label, rel_id)
    if not ok:
        raise HTTPException(
            404,
            f"Graph relation '{rel_id}' not found. The relation may have "
            "been removed, or the server restarted since the last read — "
            "refresh the Graph tab to repopulate the endpoint cache.",
        )
    store = get_store()
    event = _make_event(
        kind="graph-relation-edited",
        sev="info",
        actor="operator",
        target_label=rel_id,
        summary=f"Graph relation '{rel_id}' deleted",
        meta={"rel_id": rel_id, "operation": "delete"},
        target_type="relation",
    )
    await store.record_activity(event)
    return None


@router.patch("/graph/relations/{rel_id}", response_model=GraphRelation)
async def update_graph_relation_endpoint(
    rel_id: str, body: GraphRelationPatch
) -> dict[str, Any]:
    """Persist an edit to a graph relation in Memgraph.

    The relation id is opaque to the client; resolution back to the
    Cypher MATCH happens via an in-process cache primed by the last
    `/graph/relations` read. A 404 with a hint guides the client to
    refresh when the cache is cold (process restart, etc.).
    """
    from . import graph_reader

    label = _graph_memgraph_label()
    patch_dict = body.model_dump(exclude_unset=True)
    updated = await graph_reader.update_graph_relation(
        label, rel_id, patch_dict
    )
    if updated is None:
        raise HTTPException(
            404,
            f"Graph relation '{rel_id}' not found. The relation may have "
            "been removed, or the server restarted since the last read — "
            "refresh the Graph tab to repopulate the endpoint cache.",
        )
    store = get_store()
    event = _make_event(
        kind="graph-relation-edited",
        sev="info",
        actor="operator",
        target_label=updated.get("label") or rel_id,
        summary=f"Graph relation '{updated.get('label') or rel_id}' updated",
        meta={
            "rel_id": rel_id,
            "patch_keys": list(patch_dict.keys()),
        },
        target_type="relation",
    )
    await store.record_activity(event)
    return updated


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
            sub="Added to tag catalog · Tier 3",
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
    renamed_from: str | None = None
    if body.tag is not None:
        new_name = body.tag.strip()
        if not new_name:
            raise HTTPException(400, "Tag name cannot be empty")
        if new_name != entry.get("tag"):
            existing = await store.tags.get_tag(new_name)
            if existing is not None:
                raise HTTPException(409, f"Tag '{new_name}' already exists")
            old_name = entry["tag"]
            entry["tag"] = new_name
            renamed_from = old_name
            changed.append("tag")
    if body.def_ is not None and body.def_ != entry.get("def"):
        entry["def"] = body.def_
        changed.append("def")
    if (
        body.long_description is not None
        and body.long_description != entry.get("long_description", "")
    ):
        entry["long_description"] = body.long_description
        changed.append("long_description")
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
    cascade_affected: int | None = None
    if renamed_from is not None:
        new_name = entry["tag"]
        seed_affected = _cascade_seed_document_tags(
            store,
            name=renamed_from,
            strategy="migrate",
            to=new_name,
        )
        graph_affected = await _cascade_graph_tag_edges(
            name=renamed_from,
            strategy="migrate",
            to=new_name,
            actor=actor,
            strict=isinstance(store.tags, MemgraphTagStore),
        )
        cascade_affected = (
            graph_affected if graph_affected is not None else seed_affected
        )
        await store.tags.delete_tag(renamed_from)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=entry.get("tag") or name,
        summary=(
            f"Tag {entry.get('tag') or name} edited "
            f"({', '.join(changed) or 'no-op'})"
        ),
        meta={
            "fields": changed,
            "renamed_from": renamed_from,
            "cascade_affected": cascade_affected,
        },
        notification=_make_notification(
            title="Tag",
            tagname=entry.get("tag") or name,
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
    """Delete a tag and cascade the selected migration strategy to documents."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    payload = body or TagDeleteBody()
    if payload.strategy == "migrate" and not payload.to:
        raise HTTPException(422, "strategy=migrate requires 'to'")
    if payload.strategy == "migrate" and payload.to == name:
        raise HTTPException(422, "strategy=migrate requires a different target tag")
    if payload.strategy == "migrate" and payload.to:
        target = await store.tags.get_tag(payload.to)
        if target is None:
            raise HTTPException(404, f"Migration target tag '{payload.to}' not found")
    actor = payload.actor or "system"

    seed_affected = _cascade_seed_document_tags(
        store,
        name=name,
        strategy=payload.strategy,
        to=payload.to,
    )
    graph_affected = await _cascade_graph_tag_edges(
        name=name,
        strategy=payload.strategy,
        to=payload.to,
        actor=actor,
        strict=isinstance(store.tags, MemgraphTagStore),
    )
    # The two cascades target disjoint stores:
    #   - `_cascade_seed_document_tags` mutates `WebuiStore._documents` (the
    #     in-memory list seeded from `webui_seed.DOCUMENTS` in dev / CI
    #     without `webui_tag_backend="memgraph"`).
    #   - `_cascade_graph_tag_edges` rewrites Memgraph
    #     `DocStatus_{ws} -[:TAGGED_WITH]-> WebuiTag_{folder}` edges in
    #     production. With mock-kill F6 in memgraph mode, `_documents` is
    #     empty so the seed count contributes 0 — no double-count risk.
    # The previous expression (`graph_affected if graph_affected is not None
    # else seed_affected`) silently dropped the seed count whenever the
    # pool was reachable, even against a fresh Memgraph DB returning 0.
    # That made `test_migrate_with_to_succeeds` fail in the CI integration
    # job (Memgraph reachable but empty + default memory tag backend that
    # still seeds `_documents`).
    affected_docs = (graph_affected or 0) + seed_affected

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
            "affected_docs": affected_docs,
            "sources_count_at_delete": entry.get("sources_count", 0),
        },
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix=suffix,
            sub=f"{affected_docs} docs affected",
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
