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
``POST /documents/{id}/    ``POST /documents/scan`` (global)         no-op + 202
   scan``
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
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .auth import LoginRequest, LoginResponse

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
    latest_message: str | None = None


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
) -> list[dict[str, Any]]:
    """Apply WebUI-side filters that LightRAG's paginated endpoint doesn't.

    ``q`` matches a substring of file_path (case-insensitive); ``tag``
    matches against ``metadata.tags`` if present (the overlay tagstore
    is the authoritative source — but documents already carrying tags
    on their DocStatus.metadata get filtered here too).
    """
    from .folder import load_folder_catalog

    default_folder = load_folder_catalog().default_folder_id
    out = [
        d
        for d in items
        if (d.get("metadata") or {}).get("folder", default_folder) == folder
    ]
    if q:
        needle = q.lower()
        out = [d for d in out if needle in (d.get("file_path") or "").lower()]
    if tag:
        # Tags now come from the [:TAGGED_WITH] graph relation and are
        # injected on top of each doc dict by the list endpoint
        # (see :func:`_attach_tags_via_graph`). The filter respects
        # whichever representation is on the dict.
        out = [d for d in out if tag in (d.get("tags") or [])]
    return out


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
    metadata = doc.get("metadata") or {}
    return {
        "doc_id": doc.get("id") or doc.get("doc_id") or "",
        "file_path": doc.get("file_path") or "",
        "status": doc.get("status") or "",
        "chunks_count": doc.get("chunks_count") or 0,
        "content_summary": doc.get("content_summary"),
        "content_length": doc.get("content_length"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "track_id": doc.get("track_id"),
        "error_msg": doc.get("error_msg"),
        # Tags are populated by _attach_tags_via_graph after this
        # projection (graph-join via [:TAGGED_WITH] edges).
        "tags": [],
        "folder": metadata.get("folder"),
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
    return metadata.get("folder", default_folder) == folder


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

    @router.get("/auth-status", response_model=_AuthStatusResponse)
    async def auth_status_shim(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_security),
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

    @router.post("/login", response_model=LoginResponse)
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
    async def logout_shim(response: Response) -> dict[str, bool]:
        from .auth import logout

        return await logout(response)

    @router.get(
        "/documents",
        response_model=_ListEnvelope,
        dependencies=protected_deps,
    )
    async def list_documents(
        request: Request,
        status: str | None = Query(default=None),
        q: str | None = Query(default=None),
        tag: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
    ) -> _ListEnvelope:
        """Shadow the native ``GET /documents`` to expose a flat envelope.

        Pagination model: opaque cursor = page number. The React port doesn't
        need offset arithmetic; it just forwards whatever ``cursor`` it got
        from the previous response (TODO: emit a ``next_cursor`` field, see
        ``_ListEnvelope`` extension).

        Calls ``MemgraphDocStatusStorage.get_docs_paginated`` which returns
        ``(list[tuple[doc_id, DocProcessingStatus]], total)`` — a tuple, not
        a dict envelope (the native LightRAG paginated HTTP route assembles
        the dict in the route handler).
        """
        import dataclasses

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

        docs_tuples, _total = await rag.doc_status.get_docs_paginated(
            page=page,
            page_size=page_size,
            status_filter=status_enum,
        )

        # Flatten (doc_id, DocProcessingStatus dataclass) → dict for projection
        projected: list[dict[str, Any]] = []
        for doc_id, dps in docs_tuples:
            payload = dataclasses.asdict(dps)
            # asdict() leaves enums as enums — coerce to their string value
            if hasattr(payload.get("status"), "value"):
                payload["status"] = payload["status"].value
            payload["id"] = doc_id
            projected.append(_project_doc(payload))

        # Tags via graph join — single batch Cypher round-trip.
        await _attach_tags_via_graph(projected, folder=folder)

        filtered = _filter_docs(projected, q=q, tag=tag, folder=folder)

        return _ListEnvelope(
            items=[_DocumentEnvelope(**d) for d in filtered],
            total=len(filtered),
        )

    @router.get(
        "/documents/{doc_id}/chunks",
        response_model=list[_DocumentChunk],
        dependencies=protected_deps,
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
        from .folder import resolve_folder_for_request

        rag = get_rag()
        folder = resolve_folder_for_request(request)
        doc_status = await rag.doc_status.get_by_id(doc_id)
        if doc_status is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        if not _doc_matches_folder(doc_status, folder):
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # DocStatus may come back as dict (Memgraph backend) or dataclass
        # depending on storage impl. Read chunks_list defensively.
        chunks_list: list[str] | None
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

    @router.post(
        "/documents/{doc_id}/scan",
        response_model=_OkResponse,
        status_code=202,
        dependencies=protected_deps,
    )
    async def scan_document(doc_id: str) -> _OkResponse:
        """Per-doc re-scan stub.

        LightRAG only has a global ``POST /documents/scan`` (scans the input
        directory for new files). A targeted re-scan of a single doc would
        require ``adelete_by_doc_id`` + re-ingest, which is destructive.
        For now we ack the request and emit an audit event (TODO).
        """
        logger.info("twindb shim: per-doc scan ack for doc_id=%s (no-op)", doc_id)
        return _OkResponse()

    @router.delete(
        "/documents/{doc_id}",
        response_model=_OkResponse,
        dependencies=protected_deps,
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
        from .folder import resolve_folder_for_request

        rag = get_rag()
        folder = resolve_folder_for_request(request)
        doc_status = await rag.doc_status.get_by_id(doc_id)
        if doc_status is None or not _doc_matches_folder(doc_status, folder):
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        try:
            await rag.adelete_by_doc_id(doc_id)
        except Exception as exc:
            logger.exception("twindb shim: delete_document(%s) failed", doc_id)
            raise HTTPException(status_code=500, detail=str(exc))
        return _OkResponse()

    @router.get(
        "/pipeline_status",
        response_model=_SimplePipelineStatus,
        dependencies=protected_deps,
    )
    async def pipeline_status() -> _SimplePipelineStatus:
        """Root-level alias of ``/documents/pipeline_status`` with projection.

        LightRAG's payload is ~10 fields; the WebUI consumes only 3. We
        keep the shim contract narrow to surface accidental over-coupling.
        """
        rag = get_rag()
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            data = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            data = dict(data)
        except Exception as exc:
            logger.warning("twindb shim: pipeline_status fallback (%s)", exc)
            data = {}

        return _SimplePipelineStatus(
            busy=bool(data.get("busy", False)),
            # job_count = total docs being processed; LightRAG calls it ``docs``
            job_count=int(data.get("docs", 0)),
            latest_message=data.get("latest_message") or None,
        )

    @router.get("/openapi", dependencies=protected_deps)
    async def webui_openapi() -> dict[str, Any]:
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

    @router.get("/health", response_model=_SimpleHealth)
    async def health() -> _SimpleHealth:
        try:
            rag = get_rag()
        except Exception:
            return _SimpleHealth(status="degraded")

        import lightrag

        return _SimpleHealth(
            status="ok",
            version=getattr(lightrag, "__version__", None),
        )

    return router
