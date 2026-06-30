"""
Native-route shims — re-shape LightRAG's native FastAPI surface to match
the React port's contract (``lightrag_webui_twin/src/api/resources.ts``).

Doctrine: Twin = AI-Readable Surface. The WebUI contract is the source
of truth; LightRAG's native routes get translated, not vice-versa.

Mounted by :func:`twindb_lightrag_memgraph.register` when called with
``shim_native_routes=True``. The router is inserted at the HEAD of
``app.router.routes`` so shadow routes win the match against LightRAG's
later ``include_router(create_document_routes(...))`` registration.

Coverage map (cf. ``/tmp/webui-lightrag-compat-*.md``):

=========================  ========================================  ===========
WebUI emits                LightRAG native                           Shim action
=========================  ========================================  ===========
``GET /documents``         ``POST /documents/paginated``             reshape envelope
``GET /documents/{id}/     (none)                                    build from KV
   chunks``                                                          text_chunks
``POST /documents/{id}/    ``POST /documents/scan`` (global)         reject targeted
   scan``                                                           scans clearly
``DELETE /documents/{id}`` ``DELETE /documents/delete_document``     translate
                           (body=id)                                  path → body
``GET /health``            ``GET /health`` (rich)                    project
``GET /pipeline_status``   ``GET /documents/pipeline_status``        alias + project
``GET /openapi``           ``GET /openapi.json`` (full FastAPI spec) static groups
=========================  ========================================  ===========

The shims never touch LightRAG source code; they only sit *before* the
native routes in the FastAPI router list.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .auth import LoginRequest, LoginResponse
from .document_hash import enrich_metadata_with_document_hash

logger = logging.getLogger(__name__)
_security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Response models — mirror lightrag_webui_twin/src/types/document.ts etc.
# ---------------------------------------------------------------------------


class _DocumentEnvelope(BaseModel):
    """Mirror of TS ``Document`` (partial, only what the React port reads)."""

    doc_id: str
    file_path: str
    status: str
    chunks_count: int
    content_summary: str | None = None
    content_length: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    track_id: str | None = None
    error_msg: str | None = None
    # Twin overlay fields (populated by the overlay store, none here yet):
    tags: list[str] = []
    folder: str | None = None
    review: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class _ListEnvelope(BaseModel):
    """Mirror of TS ``ListEnvelope<T>`` from resources.ts."""

    items: list[_DocumentEnvelope]
    total: int
    page: int
    page_size: int
    # Opaque cursor for the next page (page number as string), or None when
    # this is the last page.
    next_cursor: str | None = None
    status_counts: dict[str, int] | None = None


class _DocumentChunk(BaseModel):
    chunk_id: str
    order: int
    text: str
    redacted: bool | None = None


class _SimpleHealth(BaseModel):
    """Simplified health for the React port (vs LightRAG's rich payload)."""

    status: str  # 'ok' | 'degraded' | 'down'
    version: str | None = None


class _SimplePipelineStatus(BaseModel):
    busy: bool
    job_count: int
    job_name: str | None = None
    latest_message: str | None = None
    history_messages: list[str] = Field(default_factory=list)


class _OkResponse(BaseModel):
    ok: bool = True


class _AuthStatusResponse(BaseModel):
    auth_enabled: bool
    authenticated: bool
    user: str | None = None
    expires_at: str | None = None
    login_required: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_docs(
    items: list[dict[str, Any]],
    q: str | None,
    tag: str | None,
    folder: str,
    source: str | None = None,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Apply WebUI-side filters that LightRAG's paginated endpoint doesn't.

    ``q`` matches a substring of the source-ish fields (case-insensitive);
    ``source`` is the explicit alias for file_path/source filtering; ``doc_id``
    matches the document id. ``tag`` matches the graph-injected ``tags`` field.
    """
    from .folder import load_folder_catalog

    default_folder = load_folder_catalog().default_folder_id
    out = [
        d
        for d in items
        if (
            d.get("folder")
            or (d.get("metadata") or {}).get("folder")
            or default_folder
        )
        == folder
    ]
    if q:
        needle = q.lower()
        out = [
            d
            for d in out
            if needle
            in " ".join(
                str(d.get(key) or "")
                for key in (
                    "doc_id",
                    "id",
                    "file_path",
                    "source",
                    "content_summary",
                    "summary",
                )
            ).lower()
        ]
    if source:
        source_needle = source.lower()
        out = [
            d
            for d in out
            if source_needle
            in str(d.get("file_path") or d.get("source") or "").lower()
        ]
    if doc_id:
        out = [
            d
            for d in out
            if doc_id == str(d.get("doc_id") or d.get("id") or "")
        ]
    if tag:
        # Tags now come from the [:TAGGED_WITH] graph relation and are
        # injected on top of each doc dict by the list endpoint
        # (see :func:`_attach_tags_via_graph`). The filter respects
        # whichever representation is on the dict.
        out = [d for d in out if tag in (d.get("tags") or [])]
    return out


def _has_local_document_filter(
    q: str | None,
    tag: str | None,
    source: str | None,
    doc_id: str | None,
) -> bool:
    """Filters not handled by LightRAG's paginated DocStatus storage call."""
    return any((q, tag, source, doc_id))


def _project_doc_tuples(docs_tuples: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Flatten storage ``(doc_id, DocProcessingStatus)`` rows for the wire model."""
    import dataclasses

    projected: list[dict[str, Any]] = []
    for doc_id, dps in docs_tuples:
        if dataclasses.is_dataclass(dps):
            payload = dataclasses.asdict(dps)
        elif isinstance(dps, dict):
            payload = dict(dps)
        elif hasattr(dps, "model_dump"):
            payload = dps.model_dump()
        else:
            payload = dict(getattr(dps, "__dict__", {}))
        # asdict() leaves enums as enums — coerce to their string value.
        if hasattr(payload.get("status"), "value"):
            payload["status"] = payload["status"].value
        payload["id"] = doc_id
        projected.append(_project_doc(payload))
    return projected


async def _attach_tags_via_graph(docs: list[dict[str, Any]], folder: str) -> None:
    """Mutate ``docs`` in place to add a ``tags`` field via graph join.

    Single Cypher batch round-trip joining the doc nodes to their
    [:TAGGED_WITH] tag nodes. Doctrine: tags are graph relations, not
    a JSON array nested in metadata. Docs with zero edges receive
    ``tags=[]`` (the OPTIONAL MATCH guarantees a row per input id).
    """
    if not docs:
        return

    from .. import _pool
    from .._constants import resolve_workspace

    workspace = resolve_workspace()
    doc_label = f"DocStatus_{workspace}"
    tag_label = f"WebuiTag_{folder}"
    doc_ids = [d["doc_id"] for d in docs if d.get("doc_id")]

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
                tid for tid in (record["tags"] or []) if tid
            )
        await result.consume()

    for d in docs:
        d["tags"] = tags_by_id.get(d["doc_id"], [])


def _project_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Coerce a LightRAG DocStatusResponse into the Twin _DocumentEnvelope shape.

    LightRAG's DocStatusResponse uses ``id`` for the doc identifier whereas
    the WebUI uses ``doc_id``. Pulls structured Twin fields out of
    ``metadata`` when present (folder id + review state). The
    ``tags`` field is left at the empty list here — it is populated
    in a separate graph-join pass by :func:`_attach_tags_via_graph`
    because tags now live as [:TAGGED_WITH] edges, not as a JSON array
    in ``metadata.tags`` (doctrine: a graph engine deserves graph
    queries, not string-array-in-property heresy).
    """
    doc_id = str(doc.get("id") or doc.get("doc_id") or "")
    metadata = enrich_metadata_with_document_hash(doc.get("metadata") or {}, doc_id)
    # TR-ING-01: ``chunks_count`` must use an explicit ``is not None``
    # check rather than ``or 0`` — the latter collapses ``None`` (never
    # started chunking) and ``0`` (started, indexed zero) into the same
    # rendered value. The operator-facing contract distinguishes them.
    raw_chunks_count = doc.get("chunks_count")
    return {
        "doc_id": doc_id,
        "file_path": doc.get("file_path") or "",
        "status": doc.get("status") or "",
        "chunks_count": raw_chunks_count if raw_chunks_count is not None else 0,
        "content_summary": doc.get("content_summary"),
        "content_length": doc.get("content_length"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "track_id": doc.get("track_id"),
        "error_msg": doc.get("error_msg"),
        # Tags are populated by _attach_tags_via_graph after this
        # projection (graph-join via [:TAGGED_WITH] edges).
        "tags": [],
        "folder": doc.get("folder") or metadata.get("folder"),
        "review": metadata.get("review"),
        "metadata": metadata,
    }


def _doc_matches_folder(doc_status: Any, folder: str) -> bool:
    from .folder import load_folder_catalog

    if isinstance(doc_status, dict):
        metadata = doc_status.get("metadata") or {}
    else:
        metadata = getattr(doc_status, "metadata", None) or {}
    default_folder = load_folder_catalog().default_folder_id
    current = (
        doc_status.get("folder")
        if isinstance(doc_status, dict)
        else getattr(doc_status, "folder", None)
    )
    return (current or metadata.get("folder") or default_folder) == folder


# ---------------------------------------------------------------------------
# Route implementations (module-level so the factory stays simple)
# ---------------------------------------------------------------------------


async def _get_docs_paginated_for_shim(
    rag: Any,
    *,
    page: int,
    page_size: int,
    status_enum: Any,
    folder: str,
) -> tuple[list[tuple[str, Any]], int]:
    """Call DocStatus pagination with folder support when the backend has it."""
    try:
        return await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=page_size,
            status_filter=status_enum,
            folder=folder,
        )
    except TypeError:
        return await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=page_size,
            status_filter=status_enum,
        )


async def _list_documents_impl(
    get_rag, request, status, q, tag, cursor, source=None, doc_id=None
) -> _ListEnvelope:
    """Body of the ``GET /documents`` shim (flat paginated envelope)."""
    from lightrag.base import DocStatus
    from twindb_lightrag_memgraph._constants import DEFAULT_PAGE_SIZE
    from .folder import resolve_folder_for_request

    rag = get_rag()
    folder = resolve_folder_for_request(request)
    page = int(cursor) if (cursor and cursor.isdigit()) else 1
    page_size = DEFAULT_PAGE_SIZE

    # Translate UI string status → DocStatus enum (the storage method
    # only accepts the enum; the WebUI sends uppercase strings).
    status_enum: DocStatus | None = None
    if status and status not in ("all", ""):
        try:
            status_enum = DocStatus(status.lower())
        except ValueError:
            logger.warning("twindb shim: unknown status filter %r", status)

    local_filter = _has_local_document_filter(q, tag, source, doc_id)
    if local_filter:
        all_tuples: list[tuple[str, Any]] = []
        fetch_page = 1
        total = 0
        while True:
            docs_tuples, total = await _get_docs_paginated_for_shim(
                rag,
                page=fetch_page,
                page_size=page_size,
                status_enum=status_enum,
                folder=folder,
            )
            all_tuples.extend(docs_tuples)
            if not docs_tuples or len(all_tuples) >= total:
                break
            fetch_page += 1
        projected = _project_doc_tuples(all_tuples)
        if tag:
            await _attach_tags_via_graph(projected, folder=folder)
        filtered = _filter_docs(
            projected,
            q=q,
            tag=tag,
            folder=folder,
            source=source,
            doc_id=doc_id,
        )
        filtered_total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = filtered[start:end]
        if not tag:
            await _attach_tags_via_graph(page_items, folder=folder)
        next_cursor = str(page + 1) if end < filtered_total else None
    else:
        docs_tuples, total = await _get_docs_paginated_for_shim(
            rag,
            page=page,
            page_size=page_size,
            status_enum=status_enum,
            folder=folder,
        )
        # More pages exist when this DB page came back full. Cursor = next page
        # number (opaque to the client). The storage call is folder-scoped when
        # supported, so this cursor no longer skips over non-folder rows.
        next_cursor = str(page + 1) if page * page_size < total else None
        page_items = _project_doc_tuples(docs_tuples)
        # Tags via graph join — single batch Cypher round-trip.
        await _attach_tags_via_graph(page_items, folder=folder)
        page_items = _filter_docs(page_items, q=None, tag=None, folder=folder)
        filtered_total = total

    status_counts = None
    try:
        status_counts = await rag.doc_status.get_status_counts(folder=folder)
    except TypeError:
        status_counts = await rag.doc_status.get_status_counts()
    except AttributeError:
        status_counts = None

    return _ListEnvelope(
        items=[_DocumentEnvelope(**d) for d in page_items],
        total=filtered_total,
        page=page,
        page_size=page_size,
        next_cursor=next_cursor,
        status_counts=status_counts,
    )


async def _list_document_chunks_impl(
    get_rag, request, doc_id: str
) -> list[_DocumentChunk]:
    """Body of the ``GET /documents/{doc_id}/chunks`` shim."""
    from .folder import resolve_folder_for_request

    rag = get_rag()
    folder = resolve_folder_for_request(request)
    doc_status = await rag.doc_status.get_by_id(doc_id)
    if doc_status is None or not _doc_matches_folder(doc_status, folder):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    # DocStatus may come back as dict (Memgraph backend) or dataclass
    # depending on storage impl. Read chunks_list defensively.
    if isinstance(doc_status, dict):
        chunks_list = doc_status.get("chunks_list")
    else:
        chunks_list = getattr(doc_status, "chunks_list", None)

    if not chunks_list:
        return []

    raw_chunks = await rag.text_chunks.get_by_ids(chunks_list)
    items: list[_DocumentChunk] = []
    for chunk_id, raw in zip(chunks_list, raw_chunks):
        if not raw:
            continue
        items.append(
            _DocumentChunk(
                chunk_id=chunk_id,
                order=int(raw.get("chunk_order_index") or 0),
                text=raw.get("content") or "",
            )
        )
    items.sort(key=lambda c: c.order)
    return items


async def _delete_or_unshare(rag, doc_id: str, folder: str) -> None:
    """Ref-counted delete, parity with the bulk-delete surface.

    A document can be MEMBER_OF several folders (one physical record). Deleting
    it from the active folder must only **un-share** it there; the physical
    cascade (chunks/vectors/KG via ``adelete_by_doc_id``) runs ONLY when this was
    its LAST membership — otherwise a single-delete from folder A would destroy a
    doc still shared into folder B (data loss).

    Backends without ``get_folders_for_doc`` fall back to the legacy hard delete,
    preserving LightRAG-native behaviour when the membership model is absent.
    Mirrors ``routes_documents._apply_membership_delete`` (same lock + physical
    helper) so single- and bulk-delete cannot diverge.
    """
    from .webui.router import _delete_doc_from_rag

    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    if get_folders is None:
        await _delete_doc_from_rag(rag, doc_id)
        return

    from .webui.routes_documents import _membership_lock

    async with _membership_lock(doc_id):
        folders = await get_folders(doc_id)
        # Physical delete only on the last (or unknown) membership; otherwise
        # unshare from the active folder, keeping the doc alive for the others.
        if folders is None or folders == [folder]:
            await _delete_doc_from_rag(rag, doc_id)
        else:
            await rag.doc_status.remove_from_folder(doc_id, folder)


async def _doc_visible_in_folder(rag, doc_id: str, doc_status: Any, folder: str) -> bool:
    """Membership-first folder visibility for the native shim gate.

    Authority is the MEMBER_OF graph (``get_folders_for_doc``) when the backend
    exposes it; the legacy ``folder``/metadata property is only a fallback when
    the membership API is absent. Without this the shim could 404 a doc that is
    visible in ``folder`` via MEMBER_OF but whose legacy ``folder`` property
    points elsewhere (or accept one where the two diverge)."""
    get_folders = getattr(rag.doc_status, "get_folders_for_doc", None)
    if get_folders is not None:
        folders = await get_folders(doc_id)
        if folders is not None:
            return folder in folders
    return _doc_matches_folder(doc_status, folder)


async def _delete_document_impl(get_rag, request, doc_id: str) -> _OkResponse:
    """Body of the ``DELETE /documents/{doc_id}`` shim."""
    from .folder import resolve_folder_for_request

    rag = get_rag()
    folder = resolve_folder_for_request(request)
    doc_status = await rag.doc_status.get_by_id(doc_id)
    if doc_status is None or not await _doc_visible_in_folder(
        rag, doc_id, doc_status, folder
    ):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    try:
        await _delete_or_unshare(rag, doc_id, folder)
    except Exception as exc:
        logger.exception("twindb shim: delete_document(%s) failed", doc_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return _OkResponse()


async def _pipeline_status_impl(get_rag) -> _SimplePipelineStatus:
    """Body of the ``GET /pipeline_status`` shim (projected namespace data)."""
    rag = get_rag()
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        data = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        data = dict(data)
    except Exception as exc:
        logger.warning("twindb shim: pipeline_status fallback (%s)", exc)
        data = {}
    history = data.get("history_messages") or []
    if not isinstance(history, list):
        history = []

    return _SimplePipelineStatus(
        busy=bool(data.get("busy", False)),
        # job_count = total docs being processed; LightRAG calls it ``docs``
        job_count=int(data.get("docs", 0)),
        job_name=data.get("job_name") or None,
        latest_message=data.get("latest_message") or None,
        history_messages=[str(message) for message in history],
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def build_native_shims_router(
    get_rag,
    *,
    auth_dependency: Callable | None = None,
) -> APIRouter:
    """Build the shim APIRouter.

    Args:
        get_rag: zero-arg callable returning the host ``LightRAG`` instance.
            Late binding lets us register the router before the host's
            lifespan has finished instantiating the RAG.
        auth_dependency: FastAPI dependency callable applied to every
            shim route EXCEPT the public auth handshake
            (``/auth-status``, ``/login``, ``/logout``) which must stay
            reachable so unauthenticated callers can log in. Pass
            ``server.auth.require_auth`` in production. ``None`` leaves
            the routes public — only acceptable in test setups that
            explicitly assert the unprotected shape.
    """
    protected_deps: list = []
    if auth_dependency is not None:
        protected_deps = [Depends(auth_dependency)]
    router = APIRouter(tags=["twin-shim"])

    @router.get("/auth-status")
    async def auth_status_shim(
        request: Request,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_security),
        ] = None,
    ) -> _AuthStatusResponse:
        """Shadow LightRAG's native auth status without minting guest JWTs.

        LightRAG 1.4.9.11 emits guest tokens from ``/auth-status`` and
        ``/login`` when local accounts are disabled. With ``TOKEN_SECRET=""``
        this raises ``InvalidKeyError: HMAC key must not be empty``. Twin does
        not use guest tokens, so the shims delegate to our auth module, whose
        disabled-auth responses are explicit and side-effect free.
        """
        from .auth import auth_status

        return await auth_status(request, credentials)

    @router.post("/login")
    async def login_shim(
        body: LoginRequest,
        response: Response,
    ) -> LoginResponse:
        """Shadow LightRAG's native login without minting guest JWTs.

        The Twin WebUI posts JSON credentials here. The native LightRAG route
        expects OAuth2 form data and also mints a guest token when accounts are
        disabled, so it must not handle this path in Twin deployments.
        """
        from .auth import login

        return await login(body, response)

    @router.post("/logout", response_model=_OkResponse)
    async def logout_shim(response: Response, request: Request) -> dict[str, bool]:
        from .auth import logout

        return await logout(response, request)

    @router.get(
        "/documents",
        dependencies=protected_deps,
    )
    async def list_documents(
        request: Request,
        status: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        tag: Annotated[str | None, Query()] = None,
        source: Annotated[str | None, Query()] = None,
        doc_id: Annotated[str | None, Query()] = None,
        cursor: Annotated[str | None, Query()] = None,
    ) -> _ListEnvelope:
        """Shadow the native ``GET /documents`` to expose a flat envelope.

        Pagination model: opaque cursor = page number. The response echoes
        ``page``/``page_size`` and emits ``next_cursor`` so the React port can
        render pagination without offset arithmetic.

        Calls ``MemgraphDocStatusStorage.get_docs_paginated`` which returns
        ``(list[tuple[doc_id, DocProcessingStatus]], total)`` — a tuple, not
        a dict envelope (the native LightRAG paginated HTTP route assembles
        the dict in the route handler).
        """
        return await _list_documents_impl(
            get_rag,
            request,
            status,
            q,
            tag,
            cursor,
            source=source,
            doc_id=doc_id,
        )

    @router.get(
        "/documents/{doc_id}/chunks",
        dependencies=protected_deps,
        responses={404: {"description": "Document not found"}},
    )
    async def list_document_chunks(
        request: Request,
        doc_id: str,
    ) -> list[_DocumentChunk]:
        """Return text chunks for a doc.

        Resolution path: ``DocProcessingStatus.chunks_list`` carries the
        ordered chunk IDs at indexation time; we look up their content
        via ``text_chunks.get_by_ids()``. Avoids depending on a
        non-standard ``get_all()`` and keeps the query O(chunks per doc)
        instead of O(total chunks in the workspace).
        """
        return await _list_document_chunks_impl(get_rag, request, doc_id)

    @router.post(
        "/documents/{doc_id}/scan",
        dependencies=protected_deps,
        responses={409: {"description": "Per-document scan unsupported"}},
    )
    def scan_document(doc_id: str) -> None:
        """Reject unsupported per-doc re-scan requests.

        LightRAG only has a global ``POST /documents/scan`` (scans the input
        directory for new files). A targeted re-scan of a single doc would
        require ``adelete_by_doc_id`` + re-ingest, which is destructive.
        """
        logger.info(
            "twindb shim: rejected unsupported per-doc scan for doc_id=%s",
            doc_id,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "Per-document scan is not supported by LightRAG. "
                "Use /documents/reprocess_failed for failed-doc retries, or "
                "delete and re-upload the source."
            ),
        )

    @router.delete(
        "/documents/{doc_id}",
        dependencies=protected_deps,
        responses={
            404: {"description": "Document not found"},
            500: {"description": "Document deletion failed"},
        },
    )
    async def delete_document(
        request: Request,
        doc_id: str,
    ) -> _OkResponse:
        """Translate REST per-id deletion → LightRAG's body-based delete.

        LightRAG offers ``adelete_by_doc_id`` on the LightRAG instance
        directly (the HTTP route ``DELETE /documents/delete_document``
        wraps it). We bypass the HTTP and call the method.
        """
        return await _delete_document_impl(get_rag, request, doc_id)

    @router.get(
        "/pipeline_status",
        dependencies=protected_deps,
    )
    async def pipeline_status() -> _SimplePipelineStatus:
        """Root-level alias of ``/documents/pipeline_status`` with projection.

        The Twin UI's Pipeline popover renders only these values. They come
        straight from LightRAG's shared ``pipeline_status`` namespace; no
        extra runtime details are fabricated in the frontend.
        """
        return await _pipeline_status_impl(get_rag)

    @router.get("/openapi", dependencies=protected_deps)
    def webui_openapi() -> dict[str, Any]:
        """Twin-specific OpenAPI tour, not the full FastAPI spec.

        Distinct from ``/openapi.json`` (FastAPI default). Returns a static
        grouping designed for the WebUI's "API" tab — a curated reading
        of the surface, not the auto-generated firehose.
        """
        # Static fixture; full spec available at /openapi.json
        return {
            "version": "v1",
            "groups": [
                {
                    "name": "Documents",
                    "endpoints": [
                        {"method": "GET", "path": "/documents"},
                        {"method": "GET", "path": "/documents/{id}/chunks"},
                        {"method": "POST", "path": "/documents/{id}/scan"},
                        {"method": "DELETE", "path": "/documents/{id}"},
                    ],
                },
                {
                    "name": "Pipeline",
                    "endpoints": [
                        {"method": "GET", "path": "/pipeline_status"},
                        {"method": "GET", "path": "/health"},
                    ],
                },
                {
                    "name": "Twin overlay",
                    "endpoints": [
                        {"method": "GET", "path": "/twin/api/folders"},
                        {"method": "GET", "path": "/twin/api/tags"},
                        {"method": "GET", "path": "/twin/api/activity"},
                        {"method": "GET", "path": "/twin/api/notifications"},
                    ],
                },
            ],
        }

    return router


def build_health_shim(get_rag) -> APIRouter:
    """Build a separate router for ``/health`` shadow.

    Kept distinct so we can omit it (``shim_health=False``) when an
    operator wants to keep LightRAG's rich health payload — e.g. for
    Prometheus scraping that already parses the native shape.
    """
    router = APIRouter(tags=["twin-shim"])

    @router.get("/health")
    def health() -> _SimpleHealth:
        try:
            get_rag()
        except Exception:
            return _SimpleHealth(status="degraded")

        import lightrag

        return _SimpleHealth(
            status="ok",
            version=getattr(lightrag, "__version__", None),
        )

    return router
