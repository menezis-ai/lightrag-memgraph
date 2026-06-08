"""LightRAG Server -- FastAPI app factory.

Creates a complete HTTP server on top of LightRAG with:
- Memgraph storage backends (via ``register()`` patch -- L1)
- Chunk/document routes (P3 -- context expansion)
- LangSmith tracing (P1 -- distributed tracing)
- Dual auth: static API key + JWT (compatible with CFT agent)
- Health endpoint

Architecture
------------
::

    register()           <- L1: storage backends (existing)
    create_app()         <- L2: server + tracing + chunk routes (this file)
    |-- auth middleware
    |-- /query, /insert, /health
    |-- /chunks/*, /documents/*
    +-- tracing (LangSmith spans)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from lightrag import LightRAG
from pydantic import BaseModel

from .auth import auth_router, configure_auth, require_auth
from .chunk_routes import create_chunk_routes, router as chunk_router
from .settings import LightRAGServerSettings, get_settings
from .tracing import apply_lang_with_tracing, extract_trace_parent

logger = logging.getLogger(__name__)

# Module-level RAG instance (set during lifespan)
_rag: LightRAG | None = None


def _get_rag() -> LightRAG:
    if _rag is None:
        raise RuntimeError("LightRAG not initialized")
    return _rag


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False
    workspace: str | None = None


class QueryResponse(BaseModel):
    response: str
    source_doc_ids: list[str] = []


class InsertRequest(BaseModel):
    text: str
    file_path: str | None = None
    metadata: dict[str, Any] | None = None


class InsertResponse(BaseModel):
    status: str
    doc_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    workspace: str
    storage_backends: dict[str, str]
    tracing_enabled: bool


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: LightRAGServerSettings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _rag

        # -- L1 Patch: storage backends --
        from twindb_lightrag_memgraph import register

        register()
        logger.info("L1 patch applied (Memgraph storage backends)")

        # -- Resolve API keys --
        llm_api_key = settings.llm_binding_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        embed_api_key = settings.embedding_binding_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )

        # -- Build functions --
        embedding_func = _build_embedding_func(settings, embed_api_key)
        llm_func = _build_llm_func(settings, llm_api_key)

        # -- Instantiate LightRAG --
        rag_kwargs: dict[str, Any] = {
            "working_dir": settings.working_dir,
            "kv_storage": settings.kv_storage,
            "vector_storage": settings.vector_storage,
            "graph_storage": settings.graph_storage,
            "doc_status_storage": settings.doc_status_storage,
            "chunk_token_size": settings.chunk_token_size,
            "chunk_overlap_token_size": settings.chunk_overlap_token_size,
            "embedding_func": embedding_func,
            "llm_model_func": llm_func,
            "embedding_batch_num": 32,
            "embedding_func_max_async": 16,
        }
        if settings.workspace:
            rag_kwargs["workspace"] = settings.workspace

        _rag = LightRAG(**rag_kwargs)
        await _rag.initialize()
        logger.info(
            "LightRAG initialized (workspace=%s, kv=%s, vec=%s, graph=%s)",
            settings.workspace or "(default)",
            settings.kv_storage,
            settings.vector_storage,
            settings.graph_storage,
        )

        # -- L2 Patch: tracing --
        if settings.enable_langsmith_tracing:
            apply_lang_with_tracing(_rag)
            logger.info("L2 patch applied (LangSmith tracing)")

        # -- L2 Patch: chunk routes --
        create_chunk_routes(_rag)
        logger.info("L2 patch applied (chunk/document routes)")

        # -- L2 Patch: WebUI store backends (S4c) --
        if settings.enable_webui_routes:
            from .webui_router import WebuiStore, set_store
            from .webui_activitystore import MemgraphActivityStore
            from .webui_notificationstore import MemgraphNotificationStore
            from .webui_tagstore import MemgraphTagStore
            from .space import load_space_catalog

            backends_applied: list[str] = []
            if settings.webui_tag_backend == "memgraph":
                backends_applied.append("tags")
            if settings.webui_activity_backend == "memgraph":
                backends_applied.append("activity")
            if settings.webui_notifications_backend == "memgraph":
                backends_applied.append("notifications")
            if backends_applied:
                for space in load_space_catalog().spaces:
                    # `mode="memgraph"` so the default space doesn't
                    # silently expose the demo documents/graph from
                    # `webui_seed` through /twin/api/documents and
                    # /twin/api/graph/* on a real deploy (mock-kill F6).
                    store = WebuiStore.for_space(space.id, mode="memgraph")
                    if settings.webui_tag_backend == "memgraph":
                        tag_store = MemgraphTagStore(workspace=space.id)
                        await tag_store.initialize()
                        await tag_store.bootstrap_categories_if_empty()
                        store._tag_backend = tag_store  # noqa: SLF001
                    if settings.webui_activity_backend == "memgraph":
                        activity_store = MemgraphActivityStore(workspace=space.id)
                        await activity_store.initialize()
                        store._activity_backend = activity_store  # noqa: SLF001
                    if settings.webui_notifications_backend == "memgraph":
                        notification_store = MemgraphNotificationStore(
                            workspace=space.id
                        )
                        await notification_store.initialize()
                        store._notification_backend = (  # noqa: SLF001
                            notification_store
                        )
                    set_store(store, space=space.id)
                logger.info(
                    "L2 patch applied (WebUI Memgraph backends: %s, spaces=%s)",
                    ", ".join(backends_applied),
                    ",".join(space.id for space in load_space_catalog().spaces),
                )

        yield

        # -- Shutdown --
        _rag = None
        if settings.enable_webui_routes and (
            settings.webui_tag_backend == "memgraph"
            or settings.webui_activity_backend == "memgraph"
            or settings.webui_notifications_backend == "memgraph"
        ):
            from .webui_router import reset_store

            reset_store()
        logger.info("LightRAG server shut down")

    app = FastAPI(
        title="LightRAG Server (Memgraph)",
        description="LightRAG HTTP API with Memgraph backends, chunk routes, and distributed tracing",
        version="0.3.0",
        lifespan=lifespan,
    )

    # -- CORS --
    if settings.cors_allow_credentials and "*" in settings.cors_allowed_origins:
        raise ValueError(
            "LIGHTRAG_CORS_ALLOWED_ORIGINS='*' cannot be combined with "
            "LIGHTRAG_CORS_ALLOW_CREDENTIALS=true"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Auth --
    configure_auth(
        api_key=settings.api_key,
        jwt_secret=settings.jwt_secret,
        jwt_algorithm=settings.jwt_algorithm,
        jwt_expiration_hours=settings.jwt_expiration_hours,
        jwt_username=settings.jwt_username,
        jwt_password=settings.jwt_password,
    )
    app.include_router(auth_router)

    # -- Core routes (auth-protected) --

    @app.get("/health", response_model=HealthResponse)
    async def health():
        from twindb_lightrag_memgraph import __version__ as plugin_version
        from .tracing import is_tracing_enabled

        return HealthResponse(
            status="ok",
            version=plugin_version,
            workspace=settings.workspace or "(default)",
            storage_backends={
                "kv": settings.kv_storage,
                "vector": settings.vector_storage,
                "graph": settings.graph_storage,
                "doc_status": settings.doc_status_storage,
            },
            tracing_enabled=is_tracing_enabled(),
        )

    @app.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(require_auth)],
    )
    async def query(body: QueryRequest, request: Request):
        rag = _get_rag()

        # P1: Extract distributed trace context from agent headers
        trace_ctx = extract_trace_parent(dict(request.headers))
        if trace_ctx:
            logger.debug("Distributed trace context: %s", trace_ctx)

        result = await rag.aquery(
            body.query,
            param={
                "mode": body.mode,
                "only_need_context": body.only_need_context,
            },
        )

        # P2: Extract full_doc_ids from the query result
        source_doc_ids = _extract_doc_ids(result)

        return QueryResponse(
            response=result if isinstance(result, str) else str(result),
            source_doc_ids=source_doc_ids,
        )

    @app.post(
        "/insert",
        response_model=InsertResponse,
        dependencies=[Depends(require_auth)],
    )
    async def insert(body: InsertRequest):
        rag = _get_rag()
        await rag.ainsert(body.text)
        return InsertResponse(status="ok")

    # -- Chunk routes (auth-protected) --
    app.include_router(chunk_router, dependencies=[Depends(require_auth)])

    # -- WebUI phase-1 surface (auth-protected) --
    #
    # Doctrine: the React port assumes every Twin endpoint sits under
    # `/twin/api/...`. The standalone server therefore mounts the same
    # router twice:
    #   * un-prefixed → backwards compat for the existing pytest suite
    #     and any pre-React-port caller still on `/documents`, /spaces,
    #     etc.
    #   * `/twin/api`-prefixed → mirrors the plugin topology
    #     (`register(mount_server=True)`), so the React port works
    #     against the standalone server without proxy rewrites.
    if settings.enable_webui_routes:
        from .webui_router import router as webui_router

        app.include_router(webui_router, dependencies=[Depends(require_auth)])
        app.include_router(
            webui_router,
            prefix="/twin/api",
            dependencies=[Depends(require_auth)],
        )
        logger.info("L2 patch applied (WebUI phase-1 router; mounted at / and /twin/api)")

    # -- Twin overlay query routes (`/twin/api/query` + `/twin/api/query/stream`)
    # The native `POST /query` declared above is the legacy single-shot
    # contract (no sources, plain string answer). The Twin overlay
    # routes added here return `{response, sources}` + an NDJSON
    # streaming variant that the React Retrieval tab consumes.
    try:
        from .twin_query_routes import build_twin_query_router

        def _get_rag_for_twin_query():
            rag = _get_rag()
            if rag is None:
                raise RuntimeError(
                    "twindb twin_query: RAG instance not initialised yet."
                )
            return rag

        app.include_router(
            build_twin_query_router(_get_rag_for_twin_query),
            prefix="/twin/api",
            dependencies=[Depends(require_auth)],
        )
        logger.info("Twin overlay query routes mounted at /twin/api/query{,/stream}")
    except ImportError:
        pass

    return app


# ---------------------------------------------------------------------------
# Helper: extract doc IDs from query result (P2)
# ---------------------------------------------------------------------------


def _extract_doc_ids(result: Any) -> list[str]:
    """Best-effort extraction of full_doc_id references from query results."""
    doc_ids: list[str] = []
    if isinstance(result, dict):
        contexts = result.get("contexts", [])
        for ctx in contexts:
            if isinstance(ctx, dict):
                doc_id = ctx.get("full_doc_id") or ctx.get("doc_id")
                if doc_id and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
    return doc_ids


# ---------------------------------------------------------------------------
# Helpers: build LLM and embedding functions from settings
# ---------------------------------------------------------------------------


def _build_embedding_func(settings: LightRAGServerSettings, api_key: str):
    """Build the async embedding function from settings."""
    from lightrag.llm.openai import openai_embedding

    async def embedding_func(texts: list[str]) -> list[list[float]]:
        return await openai_embedding(
            texts,
            model=settings.embedding_model,
            base_url=settings.embedding_binding_host,
            api_key=api_key,
        )

    embedding_func.embedding_dim = settings.embedding_dim
    embedding_func.max_token_size = settings.max_embed_tokens

    return embedding_func


def _build_llm_func(settings: LightRAGServerSettings, api_key: str):
    """Build the async LLM function from settings."""
    from lightrag.llm.openai import openai_complete

    async def llm_func(prompt: str, **kwargs) -> str:
        return await openai_complete(
            prompt,
            model=settings.llm_model,
            base_url=settings.llm_binding_host,
            api_key=api_key,
            **kwargs,
        )

    return llm_func
