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

        yield

        # -- Shutdown --
        _rag = None
        logger.info("LightRAG server shut down")

    app = FastAPI(
        title="LightRAG Server (Memgraph)",
        description="LightRAG HTTP API with Memgraph backends, chunk routes, and distributed tracing",
        version="0.3.0",
        lifespan=lifespan,
    )

    # -- CORS --
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
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
