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

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth import require_auth
from .folder import (
    bind_request_folder,
    current_folder_id,
    load_folder_catalog,
)
from .webui_models import (
    AckResponse,
    GraphEntity,
    GraphEntityCreate,
    GraphEntityPatch,
    GraphRelation,
    GraphRelationCreate,
    GraphRelationPatch,
    OpenApiEnvelope,
    OpenApiGroup,
)
from .webui.events import _make_event, _utcnow_iso
from .webui.store import WebuiStore, _stores, get_store, reset_store, set_store
from .webui.routes_activity import router as activity_router
from .webui.routes_documents import router as documents_router
from .webui.routes_folders import router as folders_router
from .webui.routes_notifications import router as notifications_router
from .webui.routes_tags import router as tags_router
from .document_hash import enrich_metadata_with_document_hash

_security = HTTPBearer(auto_error=False)


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
    return (doc.get("folder") or metadata.get("folder") or default_folder) == current_folder_id()


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
    doc_id = str(doc.get("id") or doc.get("doc_id") or "")
    metadata = enrich_metadata_with_document_hash(
        _coerce_doc_metadata(doc.get("metadata")),
        doc_id,
    )
    file_path = str(doc.get("file_path") or doc.get("source") or doc_id)
    summary = str(doc.get("content_summary") or doc.get("summary") or "")
    folder = str(doc.get("folder") or metadata.get("folder") or current_folder_id())
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
        if (
            doc.get("folder")
            or (doc.get("metadata") or {}).get("folder")
            or default_folder
        )
        == folder
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

    folder = current_folder_id()
    try:
        docs_tuples, _total = await rag.doc_status.get_docs_paginated(
            page=1,
            page_size=500,
            status_filter=status_filter,
            folder=folder,
        )
    except TypeError:
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


async def _require_auth_except_health(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> str | None:
    if request.url.path.endswith("/health"):
        return None
    return await require_auth(request=request, credentials=credentials)


router = APIRouter(
    tags=["webui"],
    dependencies=[Depends(_require_auth_except_health), Depends(bind_request_folder)],
)


router.include_router(documents_router)


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


router.include_router(folders_router)
router.include_router(notifications_router)
router.include_router(activity_router)
router.include_router(tags_router)


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
    (TR-KG-03 / QA report 2026-06-12).

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


def _graph_seed_fallback_allowed() -> bool:
    """Audit C5: serve the in-memory demo graph as a fallback ONLY when
    we are explicitly in demo mode AND no IdP is configured.

    Allowed iff:
      * IdP is dormant (``get_active_config() is None``), AND
      * the active ``WebuiStore`` was built with ``webui_stores="seed"``.

    Rejected when:
      * IdP is configured (production deploy, regardless of store mode),
      * the store was built with ``webui_stores="memgraph"``,
      * the store reports an unknown mode (defensive default — refuse
        rather than guess).

    The check is **explicit on configuration**, not inferred from
    store contents. A seed store with no graph fixtures is still
    "demo mode" by construction; conversely, a memgraph store that
    happens to be empty is still production. Confusing data with
    config is exactly the bug this gate closes.
    """
    from . import idp_jwt

    if idp_jwt.get_active_config() is not None:
        return False
    return get_store().mode == "seed"


async def _native_graph(label: str, max_nodes: int, max_depth: int):
    """Run LightRAG's native focus+context selection once; returns
    (entities, relations) or ([], []) when rag is absent/failed."""
    from . import graph_reader

    ws = _graph_memgraph_label()
    try:
        rag = _get_rag()
    except Exception:
        return [], []
    return await graph_reader.read_graph_native(
        rag, ws, node_label=label, max_nodes=max_nodes, max_depth=max_depth
    )


@router.get("/graph/entities", response_model=list[GraphEntity])
async def list_graph_entities(
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Live Memgraph entities via LightRAG's native focus+context selection
    (``get_knowledge_graph``): ``label="*"`` = top hubs by degree (capped at
    ``max_nodes``); ``label=<entity>`` = BFS neighbourhood (``max_depth``).
    Replaces the old flat ``LIMIT 200`` scan that surfaced an arbitrary 1%
    slice of a 17k+ entity KB. Seed fallback only under the demo gate (C5).
    """
    entities, _ = await _native_graph(label, max_nodes, max_depth)
    if entities:
        return entities
    if not _graph_seed_fallback_allowed():
        return []
    return get_store().list_graph_entities()


@router.get("/graph/relations", response_model=list[GraphRelation])
async def list_graph_relations(
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Relations for the same native subgraph as ``/graph/entities`` (same
    ``label``/``max_nodes``/``max_depth`` → consistent node set, no dangling
    edges). Seed fallback under the audit-C5 demo gate only."""
    entities, relations = await _native_graph(label, max_nodes, max_depth)
    if entities:
        return relations
    if not _graph_seed_fallback_allowed():
        return []
    return get_store().list_graph_relations()


@router.get("/graph/search")
async def search_graph_entities(q: str, limit: int = 50) -> list[str]:
    """Server-side fuzzy entity-label search (native ``search_labels``) so the
    Graph search box reaches the whole KB, not just the loaded subgraph. The
    selected label is passed back as ``?label=`` to focus the subgraph."""
    from . import graph_reader

    try:
        rag = _get_rag()
    except Exception:
        return []
    return await graph_reader.search_graph_labels(rag, q, limit=limit)


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


__all__ = [
    "router",
    "WebuiStore",
    "get_store",
    "set_store",
    "reset_store",
    "OpenApiGroup",
]
