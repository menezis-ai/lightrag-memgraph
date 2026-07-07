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

import asyncio
import inspect
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth import require_auth
from ..folder import (
    bind_request_folder,
    current_folder_id,
    load_folder_catalog,
)
from ..status_vocab import storage_status_filter, to_twin_uppercase
from ..webui_models import (
    AckResponse,
    OpenApiEnvelope,
    OpenApiGroup,
)
from .events import _make_event, _request_actor, _utcnow_iso
from .store import WebuiStore, _stores, get_store, reset_store, set_store
from .routes_activity import router as activity_router
from .routes_documents import router as documents_router
from .routes_folders import router as folders_router
from .routes_graph import (
    _graph_memgraph_label,
    _graph_seed_fallback_allowed,
    _native_graph,
    _validate_graph_entity_tags,
    router as graph_router,
)
from .routes_notifications import router as notifications_router
from .routes_tags import router as tags_router
from ..document_hash import enrich_metadata_with_document_hash

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
    return (
        doc.get("folder") or metadata.get("folder") or default_folder
    ) == current_folder_id()


async def _get_doc_for_active_folder(doc_id: str) -> dict[str, Any]:
    rag = _get_rag()
    raw = await rag.doc_status.get_by_id(doc_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    doc = _status_to_dict(raw)
    # Membership is authoritative: a doc shared into the active folder is
    # accessible there even if its legacy single `folder` property names another
    # one. The legacy property is consulted ONLY when the backend has no
    # membership method at all — never for an existing doc that returns an empty
    # membership list, because that orphan state is exactly what the refactor
    # makes meaningful (it must NOT be silently visible via the stale property).
    in_folder: bool
    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    if get_folders is None:
        in_folder = _doc_matches_active_folder(doc)
    else:
        folders = await get_folders(doc_id)  # doc exists → a list, never None
        in_folder = current_folder_id() in (folders or [])
    if not in_folder:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


async def _graph_tags_for_doc_or_none(doc_id: str) -> list[str] | None:
    """Return graph-backed tag ids, or ``None`` when the graph lookup failed.

    An empty list is authoritative: the document has no tags in the active
    folder. Callers must not treat it like "unknown" and fall back to stale
    ``DocStatus.metadata.tags``.
    """
    try:
        from ... import _pool
        from ..._constants import resolve_workspace

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
        return None


async def _graph_tags_for_doc(doc_id: str) -> list[str]:
    """Best-effort doc tag lookup through [:TAGGED_WITH] relations."""
    return await _graph_tags_for_doc_or_none(doc_id) or []


async def _attach_graph_tags_for_documents(docs: list[dict[str, Any]]) -> None:
    """Attach graph-backed tag ids to WebUI document list rows."""
    if not docs:
        return
    try:
        from ... import _pool
        from ..._constants import resolve_workspace

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
    # Twin overlay emits the UPPERCASE spelling; the projection is owned by
    # server.status_vocab (audit 2026-07-02, DUP-1).
    return to_twin_uppercase(raw)


def _status_filter_for_doc_status(status: str | None) -> str | None:
    # Filter-string → DocStatus-value mapping is owned by server.status_vocab
    # (audit 2026-07-02, DUP-1). Kept as a module-level function because the
    # webui_router compat shim and tests import it by this name.
    return storage_status_filter(status)


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


def _project_doc_status_for_webui(
    doc: dict[str, Any],
    *,
    visible_folder: str | None = None,
) -> dict[str, Any]:
    doc_id = str(doc.get("id") or doc.get("doc_id") or "")
    metadata = enrich_metadata_with_document_hash(
        _coerce_doc_metadata(doc.get("metadata")),
        doc_id,
    )
    file_path = str(doc.get("file_path") or doc.get("source") or doc_id)
    summary = str(doc.get("content_summary") or doc.get("summary") or "")
    folder = str(
        visible_folder
        or doc.get("folder")
        or metadata.get("folder")
        or current_folder_id()
    )
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
    folder: str | None = None,
) -> list[dict[str, Any]]:
    default_folder = load_folder_catalog().default_folder_id
    filtered = list(items)
    if folder is not None:
        filtered = [
            doc
            for doc in filtered
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
            or needle
            in str(doc.get("content_summary") or doc.get("summary") or "").lower()
        ]
    if tag:
        filtered = [doc for doc in filtered if tag in (doc.get("tags") or [])]
    return filtered


def _doc_status_get_docs_paginated_supports_folder(storage: Any) -> bool:
    """Return True when ``storage.get_docs_paginated()`` accepts a ``folder`` kwarg."""
    get_docs_paginated = getattr(storage, "get_docs_paginated", None)
    if not callable(get_docs_paginated):
        return False
    try:
        params = inspect.signature(get_docs_paginated).parameters
    except (TypeError, ValueError):
        return False
    if "folder" in params:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())


def _doc_row_has_active_folder_hint(doc: dict[str, Any], folder: str) -> bool:
    default_folder = load_folder_catalog().default_folder_id
    row_folder = doc.get("folder") or (doc.get("metadata") or {}).get("folder")
    if row_folder is None:
        return folder == default_folder
    return str(row_folder) == folder


def _doc_matches_query(doc: dict[str, Any], q: str | None) -> bool:
    if not q:
        return True
    needle = q.lower()
    return (
        needle in str(doc.get("file_path") or doc.get("source") or "").lower()
        or needle in str(doc.get("content_summary") or doc.get("summary") or "").lower()
    )


async def _filter_docs_to_active_folder(
    items: list[dict[str, Any]],
    *,
    folder: str,
    rag: Any,
) -> list[dict[str, Any]]:
    """In-memory fallback filter used when DocStatus listing is not folder-scoped."""
    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    if not callable(get_folders):
        return [
            doc for doc in items if _doc_row_has_active_folder_hint(doc, folder=folder)
        ]

    if not items:
        return []

    doc_ids = [str(doc.get("doc_id") or doc.get("id") or "") for doc in items]
    memberships_by_doc = await asyncio.gather(
        *(get_folders(doc_id) for doc_id in doc_ids)
    )
    filtered: list[dict[str, Any]] = []
    for doc, memberships in zip(items, memberships_by_doc):
        if memberships is None:
            if _doc_row_has_active_folder_hint(doc, folder=folder):
                filtered.append(doc)
            continue
        if folder in memberships:
            filtered.append(doc)
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
    scoped_by_storage = _doc_status_get_docs_paginated_supports_folder(rag.doc_status)
    kwargs = {
        "page": 1,
        "page_size": 500,
        "status_filter": status_filter,
    }
    if scoped_by_storage and folder is not None:
        kwargs["folder"] = folder

    docs_tuples, _total = await rag.doc_status.get_docs_paginated(**kwargs)

    doc_rows: list[dict[str, Any]] = []
    for doc_id, raw in docs_tuples:
        payload = _status_to_dict(raw)
        payload["id"] = doc_id
        doc_rows.append(payload)

    if folder is not None:
        doc_rows = await _filter_docs_to_active_folder(doc_rows, folder=folder, rag=rag)

    doc_rows = [
        _project_doc_status_for_webui(doc, visible_folder=folder) for doc in doc_rows
    ]

    await _attach_graph_tags_for_documents(doc_rows)
    return _filter_doc_status_rows(
        doc_rows,
        q=q,
        tag=tag,
        folder=None,
    )


def _rewrite_doc_tags(
    tags: Any, name: str, strategy: str, to: str | None
) -> list[str] | None:
    """Return the tag list with ``name`` removed (and ``to`` added on migrate),
    or None when ``tags`` doesn't contain ``name``."""
    if not isinstance(tags, list) or name not in tags:
        return None
    rewritten = [tag for tag in tags if tag != name]
    if strategy == "migrate" and to and to not in rewritten:
        rewritten.append(to)
    return rewritten


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

    with store._lock:  # noqa: SLF001 - same-module store maintenance
        for doc in store._documents:  # noqa: SLF001 - same-module store maintenance
            metadata = doc.get("metadata") or {}
            folder = doc.get("folder") or metadata.get("folder") or default_folder
            if folder != active_folder:
                continue
            rewritten = _rewrite_doc_tags(doc.get("tags"), name, strategy, to)
            if rewritten is None:
                continue
            doc["tags"] = rewritten
            if isinstance(metadata, dict):
                metadata_tags = _rewrite_doc_tags(
                    metadata.get("tags"), name, strategy, to
                )
                if metadata_tags is not None:
                    metadata["tags"] = metadata_tags
            affected += 1
    return affected


def _metadata_from_raw(raw: Any) -> dict[str, Any]:
    try:
        metadata = json.loads(raw) if isinstance(raw, str) and raw else {}
    except json.JSONDecodeError:
        return {}
    return metadata if isinstance(metadata, dict) else {}


async def _rewrite_legacy_tag_metadata(
    session: Any,
    *,
    doc_label: str,
    tag_label: str,
    name: str,
    strategy: str,
    to: str | None,
) -> None:
    result = await session.run(
        f"""
        MATCH (d:`{doc_label}`)-[:TAGGED_WITH]->(:`{tag_label}` {{id: $tag}})
        RETURN d.id AS id, d.metadata AS metadata
        """,
        tag=name,
    )
    rows: list[dict[str, str]] = []
    async for record in result:
        metadata = _metadata_from_raw(record.get("metadata"))
        rewritten = _rewrite_doc_tags(metadata.get("tags"), name, strategy, to)
        if rewritten is None:
            continue
        metadata["tags"] = rewritten
        rows.append(
            {
                "id": record["id"],
                "metadata": json.dumps(metadata, sort_keys=True),
            }
        )
    await result.consume()
    if not rows:
        return
    update = await session.run(
        f"""
        UNWIND $rows AS row
        MATCH (d:`{doc_label}` {{id: row.id}})
        SET d.metadata = row.metadata
        """,
        rows=rows,
    )
    await update.consume()


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
        from ... import _pool
        from ..._constants import resolve_workspace

        workspace = resolve_workspace()
        folder = current_folder_id()
        doc_label = f"DocStatus_{workspace}"
        tag_label = f"WebuiTag_{folder}"
        now = _utcnow_iso()

        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                await _rewrite_legacy_tag_metadata(
                    session,
                    doc_label=doc_label,
                    tag_label=tag_label,
                    name=name,
                    strategy=strategy,
                    to=to,
                )
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


@router.get(
    "/health",
    responses={503: {"description": "LightRAG instance unavailable"}},
)
def twin_health() -> dict[str, Any]:
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
router.include_router(graph_router)


def _get_rag():
    """Resolve the host LightRAG instance captured at register() time."""
    from ... import _twindb_state

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


@router.post("/auth/logout")
async def logout(request: Request) -> dict[str, Any]:
    """Sign out the current operator.

    This endpoint clears Twin-owned local cookies and records the audit
    event. It does not claim to perform global SSO logout: IdP sessions
    owned by the upstream SSO remain authoritative until the browser is
    redirected through that provider's own logout flow.
    """
    from fastapi.responses import JSONResponse
    from ..auth import logout as auth_logout
    from ..idp_jwt import get_active_config

    idp_config = get_active_config()
    idp_cookie_name = idp_config.cookie_name if idp_config is not None else None
    cleared_cookies = ["twin_local_token", "twin_session", "twin_id_token"]
    if idp_cookie_name:
        cleared_cookies.append(idp_cookie_name)
    content = {
        "ok": True,
        "local_session_cleared": True,
        "twin_cookie_cleared": True,
        "idp_cookie_cleared": idp_cookie_name is not None,
        "cleared_cookies": cleared_cookies,
        "sso_logout": False,
        "detail": (
            "Twin local cookies cleared; upstream SSO session is not "
            "terminated by this endpoint."
        ),
    }
    response = JSONResponse(content=content)
    await auth_logout(response, request)
    response.delete_cookie("twin_session", path="/")
    response.delete_cookie("twin_id_token", path="/")
    if idp_cookie_name:
        response.delete_cookie(idp_cookie_name, path="/")
    return response


@router.post(
    "/documents/{doc_id}/approve",
    responses={
        404: {"description": "Document not found"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def approve_document(
    doc_id: str,
    request: Request,
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
    actor = _request_actor(request)
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
        summary=(f"approved by {actor}" + (" with edits" if edits else "")),
        meta={"doc_id": doc_id, "edits": edits},
        target_type="document",
        target_id=doc_id,
    )
    await get_store().record_activity(event)

    return {"doc_id": doc_id, "review": review}


@router.post(
    "/documents/{doc_id}/reject",
    responses={
        400: {"description": "Missing rejection reason"},
        404: {"description": "Document not found"},
        503: {"description": "LightRAG instance unavailable"},
    },
)
async def reject_document(
    doc_id: str,
    body: dict[str, Any],
    request: Request,
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
    actor = _request_actor(request)
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
        summary=f"Document {doc.get('file_path') or doc_id} rejected: {reason}",
        meta={"doc_id": doc_id, "reason": reason},
        target_type="document",
        target_id=doc_id,
    )
    await get_store().record_activity(event)

    return {"doc_id": doc_id, "review": review}


@router.get("/openapi", response_model=OpenApiEnvelope)
def get_openapi_groups() -> dict[str, Any]:
    groups, version = get_store().openapi()
    return {"groups": groups, "version": version}


__all__ = [
    "router",
    "WebuiStore",
    "get_store",
    "set_store",
    "reset_store",
    "OpenApiGroup",
]
