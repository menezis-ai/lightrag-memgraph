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

import inspect
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from lightrag import LightRAG
from neo4j.exceptions import ClientError as Neo4jClientError
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel

from .auth import auth_router, configure_auth, require_auth
from .api_wiring import log_api_wiring_sanity
from .chunk_routes import create_chunk_routes, router as chunk_router
from .settings import LightRAGServerSettings, get_settings
from .tracing import (
    apply_lang_with_tracing,
    extract_trace_parent,
    increment_metric,
    metrics_snapshot,
)

logger = logging.getLogger(__name__)

DOCUMENTS_UPLOAD_PATH = "/documents/upload"
TWIN_API_PREFIX = "/twin/api"

# Module-level RAG instance (set during lifespan)
_rag: LightRAG | None = None


def _get_rag() -> LightRAG:
    if _rag is None:
        raise RuntimeError("LightRAG not initialized")
    return _rag


def _production_auth_required(env: dict[str, str] | None = None) -> bool:
    """Return whether the deployment explicitly asks auth to fail closed."""
    env = env if env is not None else os.environ
    require_auth_flag = (env.get("TWIN_REQUIRE_AUTH") or "").strip().lower()
    if require_auth_flag in {"1", "true", "yes", "on"}:
        return True
    twin_env = (env.get("TWIN_ENV") or "").strip().lower()
    return twin_env == "production"


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


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, dict[str, Any]]


def _classification_guard_requires_file_path(
    require_production_auth: bool,
    env: dict[str, str] | None = None,
) -> bool:
    """Fail closed on raw-text /insert in production when MIP gating is active."""
    env = env if env is not None else os.environ
    if not require_production_auth:
        return False
    if (env.get("TWIN_MIP_LABEL_MAP") or "").strip():
        return True
    return getattr(LightRAG, "_twin_classification_hook", None) is not None


def _usable_file_path(file_path: str | None) -> str | None:
    if file_path is None:
        return None
    cleaned = file_path.strip()
    return cleaned or None


def _ainsert_accepts_file_paths(ainsert: Any) -> bool:
    try:
        signature = inspect.signature(ainsert)
    except (TypeError, ValueError):
        return True
    return any(
        name == "file_paths" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for name, parameter in signature.parameters.items()
    )


async def _ainsert_with_optional_file_path(
    rag: Any,
    text: str,
    *,
    file_path: str | None,
    classification_guard: bool,
) -> None:
    ainsert = rag.ainsert
    if file_path is None:
        await ainsert(text)
        return
    if _ainsert_accepts_file_paths(ainsert):
        await ainsert(text, file_paths=file_path)
        return
    if classification_guard:
        raise HTTPException(
            status_code=501,
            detail=(
                "This LightRAG version cannot receive file_paths on ainsert; "
                "classification-gated /insert would bypass source-file checks."
            ),
        )
    logger.warning(
        "Ignoring /insert file_path because LightRAG.ainsert does not support "
        "file_paths in this LightRAG version"
    )
    await ainsert(text)


def _route_group(path: str) -> str:
    """Classify paths into low-cardinality observability groups."""
    if path in {"/health", "/ready"} or path.endswith("/health"):
        return "health"
    if path in {"/query", "/query/data", "/query/stream"}:
        return "query"
    if path.startswith(f"{TWIN_API_PREFIX}/query"):
        return "query"
    if path in {"/insert", DOCUMENTS_UPLOAD_PATH, "/documents/reprocess_failed"}:
        return "ingestion"
    if path.endswith("/scan") and path.startswith("/documents/"):
        return "ingestion"
    if path.startswith(f"{TWIN_API_PREFIX}/documents"):
        return "documents"
    if path.startswith(f"{TWIN_API_PREFIX}/graph"):
        return "graph"
    if path.startswith(f"{TWIN_API_PREFIX}/settings/api-keys"):
        return "admin"
    if path.startswith(TWIN_API_PREFIX):
        return "twin"
    return "other"


def _is_upload_path(path: str) -> bool:
    return path in {DOCUMENTS_UPLOAD_PATH, f"{TWIN_API_PREFIX}/documents/upload"}


def _body_limit_for_path(path: str, settings: LightRAGServerSettings) -> int:
    if _is_upload_path(path):
        return settings.max_upload_body_bytes
    return settings.max_request_body_bytes


def _content_length(headers: Any) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _record_status_metrics(path: str, status_code: int) -> None:
    route_group = _route_group(path)
    if status_code in {401, 403}:
        increment_metric("auth_rejects_total")
    if status_code == 507:
        increment_metric("quota_rejects_total")
    if route_group == "query" and status_code >= 500:
        increment_metric("query_failures_total")
    if route_group == "ingestion" and status_code >= 500:
        increment_metric("ingestion_failures_total")


def _access_denied_reason(status_code: int) -> str | None:
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    return None


def _vector_index_check(rag: Any) -> dict[str, Any]:
    """Best-effort vector handle readiness without running a vector search."""
    if rag is None:
        return {"status": "failed", "detail": "LightRAG is not initialized"}
    for attr in (
        "chunks_vdb",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb_storage",
        "entities_vdb_storage",
    ):
        store = getattr(rag, attr, None)
        if store is None:
            continue
        has_callable = any(
            callable(getattr(store, method, None))
            for method in ("query", "upsert", "search")
        )
        if has_callable:
            return {"status": "ok", "detail": f"{attr} handle is callable"}
        return {"status": "failed", "detail": f"{attr} handle is not callable"}
    return {
        "status": "skipped",
        "detail": "no public vector handle exposed by this LightRAG instance",
    }


async def _memgraph_readiness_check() -> dict[str, Any]:
    """Verify that the shared Memgraph pool can execute a trivial query."""
    try:
        from .. import _pool

        async with _pool.get_session() as session:
            result = await session.run("RETURN 1 AS ok")
            await result.consume()
    except Exception as exc:  # noqa: BLE001 - readiness must report dependency state
        return {
            "status": "failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "ok"}


def _memgraph_routing_check() -> dict[str, Any]:
    """Report whether the configured driver URI can route writes in HA."""
    from .._constants import DEFAULT_MEMGRAPH_URI

    uri = os.environ.get("MEMGRAPH_URI", DEFAULT_MEMGRAPH_URI)
    parsed = urlparse(uri)
    scheme = parsed.scheme
    hostname = parsed.hostname or ""
    if scheme.startswith("neo4j"):
        return {
            "status": "ok",
            "detail": "routing protocol enabled; driver can target MAIN for writes",
            "scheme": scheme,
        }
    if hostname in {"", "localhost", "127.0.0.1", "::1", "memgraph"}:
        return {
            "status": "ok",
            "detail": "direct Bolt endpoint; acceptable for standalone/main service",
            "scheme": scheme,
        }
    return {
        "status": "degraded",
        "detail": (
            "direct Bolt endpoint configured. In Memgraph HA, prefer neo4j:// "
            "routing or ensure this service only points at MAIN."
        ),
        "scheme": scheme,
    }


async def _memgraph_role_check() -> dict[str, Any]:
    """Best-effort check that the app is not connected to a read-only replica."""
    result = await _run_optional_memgraph_command("SHOW REPLICATION ROLE;")
    if result["status"] != "ok":
        return result
    rows = result.get("rows") or []
    role = _first_memgraph_scalar(rows)
    if role is None:
        return {"status": "skipped", "detail": "replication role not reported"}
    role_text = str(role).lower()
    if any(token in role_text for token in ("replica", "secondary", "read_only")):
        return {
            "status": "failed",
            "detail": (
                f"connected Memgraph role is {role!r}; writes must target MAIN "
                "or use neo4j:// routing"
            ),
            "role": role,
        }
    return {"status": "ok", "detail": f"connected Memgraph role is {role!r}"}


async def _memgraph_replication_check() -> dict[str, Any]:
    """Best-effort replication visibility for Memgraph MAIN nodes."""
    result = await _run_optional_memgraph_command("SHOW REPLICAS;")
    if result["status"] != "ok":
        return result
    rows = result.get("rows") or []
    if not rows:
        return {
            "status": "ok",
            "detail": "no replicas reported; standalone or replication not configured",
            "replicas": [],
        }
    degraded = []
    for row in rows:
        status = _row_value(row, "status", "state", "health")
        mode = _row_value(row, "sync_mode", "mode", "replication_mode")
        status_text = str(status or "").lower()
        if any(
            marker in status_text
            for marker in ("down", "failed", "error", "invalid", "not ready")
        ):
            degraded.append(
                {
                    "name": _row_value(row, "name", "replica_name", "server") or "?",
                    "mode": mode,
                    "status": status,
                }
            )
    if degraded:
        return {
            "status": "degraded",
            "detail": "one or more Memgraph replicas are not healthy",
            "replicas": degraded,
        }
    return {
        "status": "ok",
        "detail": f"{len(rows)} replica(s) reported",
        "replica_count": len(rows),
    }


async def _run_optional_memgraph_command(query: str) -> dict[str, Any]:
    try:
        from .. import _pool

        async with _pool.get_read_session() as session:
            result = await session.run(query)
            rows = [dict(record) async for record in result]
            await result.consume()
    except Neo4jClientError as exc:
        if _is_optional_memgraph_command_unsupported(exc):
            return {
                "status": "skipped",
                "detail": f"Memgraph command not supported here: {query.rstrip(';')}",
            }
        return {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - readiness must report dependency state
        return {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "rows": rows}


def _is_optional_memgraph_command_unsupported(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "syntax",
            "unrecognized",
            "unknown command",
            "not supported",
            "does not exist",
            "unsupported",
        )
    )


def _first_memgraph_scalar(rows: list[dict[str, Any]]) -> Any:
    if not rows:
        return None
    row = rows[0]
    for key in ("role", "replication_role", "status"):
        if key in row:
            return row[key]
    if len(row) == 1:
        return next(iter(row.values()))
    return None


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row:
            return row[key]
        lowered = key.lower()
        if lowered in lower:
            return lower[lowered]
    return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


async def _init_webui_backends(settings: LightRAGServerSettings) -> None:
    """L2: wire the Memgraph-backed WebUI stores per folder (S4c).

    No-op unless ``enable_webui_routes`` and at least one ``memgraph`` store
    backend is configured.
    """
    if not settings.enable_webui_routes:
        return

    from .webui_router import WebuiStore, set_store
    from .webui_activitystore import MemgraphActivityStore
    from .webui_notificationstore import MemgraphNotificationStore
    from .webui_tagstore import MemgraphTagStore
    from .folder import load_folder_catalog

    backends_applied: list[str] = []
    if settings.webui_tag_backend == "memgraph":
        backends_applied.append("tags")
    if settings.webui_activity_backend == "memgraph":
        backends_applied.append("activity")
    if settings.webui_notifications_backend == "memgraph":
        backends_applied.append("notifications")
    if not backends_applied:
        return

    for folder in load_folder_catalog().folders:
        # `mode="memgraph"` so the default folder doesn't silently expose
        # the demo documents/graph from `webui_seed` through
        # /twin/api/documents and /twin/api/graph/* on a real deploy
        # (mock-kill F6).
        store = WebuiStore.for_folder(folder.id, mode="memgraph")
        if settings.webui_tag_backend == "memgraph":
            tag_store = MemgraphTagStore(workspace=folder.id)
            await tag_store.initialize()
            await tag_store.bootstrap_categories_if_empty()
            store._tag_backend = tag_store  # noqa: SLF001
        if settings.webui_activity_backend == "memgraph":
            activity_store = MemgraphActivityStore(workspace=folder.id)
            await activity_store.initialize()
            store._activity_backend = activity_store  # noqa: SLF001
        if settings.webui_notifications_backend == "memgraph":
            notification_store = MemgraphNotificationStore(workspace=folder.id)
            await notification_store.initialize()
            store._notification_backend = notification_store  # noqa: SLF001
        set_store(store, folder=folder.id)
    logger.info(
        "L2 patch applied (WebUI Memgraph backends: %s, folders=%s)",
        ", ".join(backends_applied),
        ",".join(folder.id for folder in load_folder_catalog().folders),
    )


def _effective_graph_workspace(settings: LightRAGServerSettings, rag: Any) -> str:
    raw = getattr(rag, "workspace", None)
    if isinstance(raw, str) and raw.strip():
        candidate = raw.strip()
    elif settings.workspace:
        candidate = settings.workspace
    else:
        from .._constants import resolve_workspace

        candidate = resolve_workspace()
    from .._constants import validate_identifier

    return validate_identifier(candidate, "workspace")


async def _backfill_graph_relation_ids(
    settings: LightRAGServerSettings, rag: Any
) -> int:
    if not settings.graph_relation_id_backfill_on_startup:
        return 0
    if settings.graph_storage != "MemgraphStorage":
        return 0

    from .graph_reader import backfill_relation_ids

    workspace = _effective_graph_workspace(settings, rag)
    updated = await backfill_relation_ids(
        workspace,
        batch_size=settings.graph_relation_id_backfill_batch_size,
    )
    logger.info(
        "Graph relation id backfill complete (workspace=%s, updated=%s)",
        workspace,
        updated,
    )
    return updated


def _webui_uses_memgraph(settings: LightRAGServerSettings) -> bool:
    return settings.enable_webui_routes and (
        settings.webui_tag_backend == "memgraph"
        or settings.webui_activity_backend == "memgraph"
        or settings.webui_notifications_backend == "memgraph"
    )


def _build_rag_kwargs(
    settings: LightRAGServerSettings,
    embedding_func: Any,
    llm_func: Any,
) -> dict[str, Any]:
    """Assemble the LightRAG constructor kwargs from settings.

    Extracted from the lifespan so the wiring (notably the query-cache flag)
    is unit-testable without standing up Memgraph or the LLM bindings.
    """
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
        # Query cache off by default: upstream compute_args_hash keys on
        # query+keywords only -- not the retrieved context, conversation
        # history, active folder, or doc/tag/min_score filters. Folders share
        # one physical workspace (MEMBER_OF), so an enabled cache would return
        # folder A's generated answer for the same question asked in folder B
        # (false-grounded + cross-folder leak). See settings.enable_llm_cache.
        "enable_llm_cache": settings.enable_llm_cache,
    }
    if settings.workspace:
        rag_kwargs["workspace"] = settings.workspace
    return rag_kwargs


def _build_lifespan(settings: LightRAGServerSettings):
    """Build the FastAPI lifespan context manager for ``create_app``."""

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
        rag_kwargs = _build_rag_kwargs(settings, embedding_func, llm_func)
        _rag = LightRAG(**rag_kwargs)
        # Mirror the upstream server boot (lightrag.api.lightrag_server
        # lifespan): initialize_storages() is the real API — LightRAG has no
        # initialize() — and it auto-initializes pipeline_status for
        # rag.workspace on every supported version (1.4.9.11 wheel
        # lightrag.py:684-687, 1.5.4 lightrag.py:1287-1289), so no separate
        # initialize_pipeline_status() call is needed.
        await _rag.initialize_storages()
        logger.info(
            "LightRAG initialized (workspace=%s, kv=%s, vec=%s, graph=%s)",
            settings.workspace or "(default)",
            settings.kv_storage,
            settings.vector_storage,
            settings.graph_storage,
        )
        await _backfill_graph_relation_ids(settings, _rag)

        # -- L2 Patch: tracing --
        if settings.enable_langsmith_tracing:
            apply_lang_with_tracing(_rag)
            logger.info("L2 patch applied (LangSmith tracing)")

        # -- L2 Patch: WebUI store backends (S4c) --
        await _init_webui_backends(settings)

        yield

        # -- Shutdown --
        _rag = None
        if _webui_uses_memgraph(settings):
            from .webui_router import reset_store

            reset_store()
        logger.info("LightRAG server shut down")

    return lifespan


_QUOTA_GATED_PATHS = {
    DOCUMENTS_UPLOAD_PATH,
    "/documents/reprocess_failed",
}
_SCAN_PREFIX = "/documents/"
_SCAN_SUFFIX = "/scan"


async def _instance_quota_middleware(request, call_next):
    """Refuse 507 on ingestion endpoints when Memgraph is at its memory limit.

    Path-matched so the snapshot endpoint and every read path stay fast.
    """
    if request.method == "POST":
        path = request.url.path
        if path in _QUOTA_GATED_PATHS or (
            path.startswith(_SCAN_PREFIX) and path.endswith(_SCAN_SUFFIX)
        ):
            from fastapi import HTTPException

            from .quota import enforce_instance_quota

            try:
                await enforce_instance_quota()
            except HTTPException as exc:
                return JSONResponse(
                    {"detail": exc.detail},
                    status_code=exc.status_code,
                )
    return await call_next(request)


def _request_folder(request) -> str:
    return getattr(request.state, "folder", None) or request.headers.get(
        "x-twin-folder", "-"
    )


def _request_trace_id(request) -> str:
    return getattr(request.state, "trace_id", None) or "-"


def _oversized_response(
    request, request_id, route_group, auth_mode_label, started, limit, body_bytes
):
    """Return a 413 response (and log it) when the body exceeds the limit, else None.

    Runs before ``call_next`` so ``request.state.folder`` is not set yet — the
    folder is read from the header directly here (unlike the post-response logs).
    """
    if not (limit and body_bytes is not None and body_bytes > limit):
        return None
    response = JSONResponse(
        {
            "detail": "Request body too large",
            "limit_bytes": limit,
            "request_id": request_id,
        },
        status_code=413,
    )
    response.headers["x-request-id"] = request_id
    latency_ms = (time.perf_counter() - started) * 1000
    logger.warning(
        "http_request method=%s path=%s status=%s request_id=%s "
        "folder=%s auth_mode=%s route_group=%s latency_ms=%.2f "
        "trace_id=%s content_length=%s limit_bytes=%s",
        request.method,
        request.url.path,
        413,
        request_id,
        request.headers.get("x-twin-folder", "-"),
        auth_mode_label,
        route_group,
        latency_ms,
        _request_trace_id(request),
        body_bytes,
        limit,
    )
    return response


def _log_request_failed(request, request_id, route_group, auth_mode_label, started):
    latency_ms = (time.perf_counter() - started) * 1000
    logger.exception(
        "http_request_failed method=%s path=%s status=%s request_id=%s "
        "folder=%s auth_mode=%s route_group=%s latency_ms=%.2f trace_id=%s",
        request.method,
        request.url.path,
        500,
        request_id,
        _request_folder(request),
        auth_mode_label,
        route_group,
        latency_ms,
        _request_trace_id(request),
    )


def _log_request_completed(
    request, request_id, route_group, auth_mode_label, started, status_code
):
    latency_ms = (time.perf_counter() - started) * 1000
    log_level = logging.WARNING if status_code >= 500 else logging.INFO
    logger.log(
        log_level,
        "http_request method=%s path=%s status=%s request_id=%s folder=%s "
        "auth_mode=%s route_group=%s latency_ms=%.2f trace_id=%s",
        request.method,
        request.url.path,
        status_code,
        request_id,
        _request_folder(request),
        auth_mode_label,
        route_group,
        latency_ms,
        _request_trace_id(request),
    )


def _make_operational_middleware(
    settings: LightRAGServerSettings, auth_mode_label: str
):
    """Build the request-logging / body-limit / metrics middleware dispatch."""

    async def _operational_middleware(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        path = request.url.path
        route_group = _route_group(path)
        request.state.request_id = request_id
        request.state.route_group = route_group

        trace_ctx = extract_trace_parent(dict(request.headers))
        if trace_ctx and "traceparent" in trace_ctx:
            request.state.traceparent = trace_ctx["traceparent"]
            request.state.trace_id = trace_ctx.get("trace_id")

        limit = _body_limit_for_path(path, settings)
        body_bytes = _content_length(request.headers)
        oversized = _oversized_response(
            request,
            request_id,
            route_group,
            auth_mode_label,
            started,
            limit,
            body_bytes,
        )
        if oversized is not None:
            return oversized

        try:
            response = await call_next(request)
        except Exception:
            _record_status_metrics(path, 500)
            _log_request_failed(
                request, request_id, route_group, auth_mode_label, started
            )
            raise

        response.headers["x-request-id"] = request_id
        _record_status_metrics(path, response.status_code)
        if response.status_code in {401, 403}:
            from .activity_events import emit_access_denied_event_background

            emit_access_denied_event_background(
                request,
                status_code=response.status_code,
                reason=_access_denied_reason(response.status_code),
            )
        _log_request_completed(
            request,
            request_id,
            route_group,
            auth_mode_label,
            started,
            response.status_code,
        )
        return response

    return _operational_middleware


def _auth_policy_readiness_check(
    require_production_auth: bool,
    auth_backend_configured: bool,
    idp_strict_claims_configured: bool,
) -> dict[str, Any]:
    if not require_production_auth:
        return {
            "status": "ok",
            "detail": "production auth policy not required",
            "production_required": False,
        }
    if not auth_backend_configured:
        return {
            "status": "failed",
            "detail": "production auth policy missing",
            "production_required": True,
            "strict_claims": False,
        }
    if not idp_strict_claims_configured:
        return {
            "status": "failed",
            "detail": (
                "production IdP auth requires TWIN_IDP_ISSUER and TWIN_IDP_AUDIENCE"
            ),
            "production_required": True,
            "strict_claims": False,
        }
    return {
        "status": "ok",
        "detail": "production auth policy loaded",
        "production_required": True,
        "strict_claims": True,
    }


async def _readiness_response(
    require_production_auth: bool,
    auth_backend_configured: bool,
    idp_strict_claims_configured: bool,
) -> JSONResponse:
    """Build the ``GET /ready`` payload + status code from live subsystem checks."""
    rag = _rag
    checks: dict[str, dict[str, Any]] = {
        "lightrag": {
            "status": "ok" if rag is not None else "failed",
            "detail": "initialized" if rag is not None else "not initialized",
        },
        "memgraph": await _memgraph_readiness_check(),
        "memgraph_routing": _memgraph_routing_check(),
        "memgraph_role": await _memgraph_role_check(),
        "memgraph_replication": await _memgraph_replication_check(),
        "vector_index": _vector_index_check(rag),
        "auth_policy": _auth_policy_readiness_check(
            require_production_auth,
            auth_backend_configured,
            idp_strict_claims_configured,
        ),
    }
    failed = [name for name, check in checks.items() if check["status"] == "failed"]
    body = {"status": "ready" if not failed else "not_ready", "checks": checks}
    return JSONResponse(body, status_code=200 if not failed else 503)


def _memgraph_exception_response(exc: BaseException) -> tuple[int, dict[str, Any]]:
    from .. import _pool

    payload = _pool.memgraph_exception_payload(exc)
    return 503, {"error": payload["message"], **payload}


def _handle_memgraph_exception(request: Request, exc: Neo4jError) -> JSONResponse:
    status_code, payload = _memgraph_exception_response(exc)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        payload["request_id"] = request_id
    logger.warning(
        "memgraph_dependency_error path=%s type=%s request_id=%s detail=%s",
        request.url.path,
        payload.get("type"),
        request_id or "-",
        payload.get("detail"),
    )
    return JSONResponse(payload, status_code=status_code)


def _register_core_routes(
    app: FastAPI,
    settings: LightRAGServerSettings,
    require_production_auth: bool,
    auth_backend_configured: bool,
    idp_strict_claims_configured: bool,
) -> None:
    """Register the auth-protected core routes (/health, /ready, /query, /insert)."""

    @app.get("/health", response_model=HealthResponse)
    def health():
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

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready():
        return await _readiness_response(
            require_production_auth,
            auth_backend_configured,
            idp_strict_claims_configured,
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
        responses={
            400: {
                "description": "file_path is required when production MIP classification is active"
            },
            501: {
                "description": "The installed LightRAG ainsert API cannot receive file_paths"
            },
        },
    )
    async def insert(body: InsertRequest):
        rag = _get_rag()
        file_path = _usable_file_path(body.file_path)
        classification_guard = _classification_guard_requires_file_path(
            require_production_auth
        )
        if classification_guard and file_path is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "file_path is required for /insert when MIP classification "
                    "is active in production."
                ),
            )
        await _ainsert_with_optional_file_path(
            rag,
            body.text,
            file_path=file_path,
            classification_guard=classification_guard,
        )
        return InsertResponse(status="ok")


def create_app(settings: LightRAGServerSettings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    if settings is None:
        settings = get_settings()

    lifespan = _build_lifespan(settings)

    app = FastAPI(
        title="LightRAG Server (Memgraph)",
        description="LightRAG HTTP API with Memgraph backends, chunk routes, and distributed tracing",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(Neo4jError, _handle_memgraph_exception)

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
    from .idp_jwt import IdpConfig as _IdpConfig, configure_idp

    _idp_cfg = _IdpConfig.from_env()
    _resolved_jwt_secret = settings.jwt_secret or os.environ.get("TOKEN_SECRET")
    _auth_accounts = os.environ.get("AUTH_ACCOUNTS")
    _require_production_auth = _production_auth_required()
    configure_auth(
        api_key=settings.api_key,
        jwt_secret=_resolved_jwt_secret,
        jwt_algorithm=settings.jwt_algorithm,
        jwt_expiration_hours=int(
            os.environ.get("TOKEN_EXPIRE_HOURS") or settings.jwt_expiration_hours
        ),
        jwt_username=settings.jwt_username,
        jwt_password=settings.jwt_password,
        auth_accounts=_auth_accounts,
        production_mode=_require_production_auth,
        idp_enabled=_idp_cfg is not None,
    )
    configure_idp(_idp_cfg)
    _auth_backend_configured = bool(
        settings.api_key or _resolved_jwt_secret or _idp_cfg is not None
    )
    if _idp_cfg is not None:
        _auth_mode_label = "idp"
    elif _auth_backend_configured:
        _auth_mode_label = "legacy"
    else:
        _auth_mode_label = "open"
    app.include_router(auth_router)

    # -- Core routes (auth-protected) --
    _register_core_routes(
        app,
        settings,
        _require_production_auth,
        _auth_backend_configured,
        _idp_cfg.strict_claims_configured if _idp_cfg is not None else True,
    )

    # -- Chunk routes (auth-protected) --
    create_chunk_routes(_get_rag)
    app.include_router(chunk_router, dependencies=[Depends(require_auth)])

    # -- WebUI phase-1 surface (auth-protected) --
    #
    # Doctrine: the React port assumes every Twin endpoint sits under
    # `/twin/api/...`. The standalone server therefore mounts the same
    # router twice:
    #   * un-prefixed → backwards compat for the existing pytest suite
    #     and any pre-React-port caller still on `/documents`,
    #     etc.
    #   * `/twin/api`-prefixed → mirrors the plugin topology
    #     (`register(mount_server=True)`), so the React port works
    #     against the standalone server without proxy rewrites.
    if settings.enable_webui_routes:
        from .webui_router import router as webui_router

        app.include_router(webui_router, dependencies=[Depends(require_auth)])
        app.include_router(
            webui_router,
            prefix=TWIN_API_PREFIX,
            dependencies=[Depends(require_auth)],
        )
        logger.info(
            "L2 patch applied (WebUI phase-1 router; mounted at / and %s)",
            TWIN_API_PREFIX,
        )

    # -- Twin overlay query routes (`/twin/api/query` + `/twin/api/query/stream`)
    # The native `POST /query` declared above is the legacy single-shot
    # contract (no sources, plain string answer). The Twin overlay
    # routes added here return `{response, sources}` + an NDJSON
    # streaming variant that the React Retrieval tab consumes.
    from .twin_query_routes import build_twin_query_router

    def _get_rag_for_twin_query():
        rag = _get_rag()
        if rag is None:
            raise RuntimeError("twindb twin_query: RAG instance not initialised yet.")
        return rag

    app.include_router(
        build_twin_query_router(_get_rag_for_twin_query),
        prefix=TWIN_API_PREFIX,
        dependencies=[Depends(require_auth)],
    )
    logger.info(
        "Twin overlay query routes mounted at %s/query{,/stream}",
        TWIN_API_PREFIX,
    )

    # -- API key management routes (Settings → API keys, admin only).
    # ``require_admin_user`` is applied at the sub-router level; the
    # outer ``require_auth`` is still needed so anonymous requests are
    # rejected before the admin check sees them.
    from .api_key_routes import router as api_key_router

    app.include_router(
        api_key_router,
        prefix=TWIN_API_PREFIX,
        dependencies=[Depends(require_auth)],
    )
    logger.info(
        "API key management routes mounted at %s/settings/api-keys",
        TWIN_API_PREFIX,
    )

    # -- Quota snapshot endpoint (auth-protected operational read).
    from .quota_routes import router as quota_router

    app.include_router(quota_router, prefix=TWIN_API_PREFIX)
    logger.info("Quota snapshot route mounted at %s/quota", TWIN_API_PREFIX)

    @app.get("/twin/api/ops/metrics", dependencies=[Depends(require_auth)])
    def operational_metrics():
        return metrics_snapshot()

    # -- Instance quota + operational middleware. Quota is registered first
    # so the operational logger wraps it as the outer layer (FastAPI applies
    # middleware in reverse registration order). --
    app.middleware("http")(_instance_quota_middleware)
    app.middleware("http")(_make_operational_middleware(settings, _auth_mode_label))

    log_api_wiring_sanity(app, surface="standalone")

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
    """Build the async embedding function from settings.

    Two upstream contracts this must honor (audit 2026-07-02 COMPAT-1, plus a
    third break found while fixing it):

    * the upstream symbol is ``openai_embed`` on every supported LightRAG
      (1.4.9.11 wheel ``llm/openai.py:717``, 1.5.4 ``llm/openai.py:896``);
      ``openai_embedding`` never existed there;
    * ``LightRAG.__post_init__`` requires an ``EmbeddingFunc`` dataclass
      instance — it reads ``.func`` and calls ``dataclasses.replace`` on it
      (1.4.9.11 wheel ``lightrag.py:549-551``, 1.5.4 ``lightrag.py:1089-1091``)
      — so a bare function with attributes crashes LightRAG construction.
    """
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    async def embedding_func(texts: list[str]) -> list[list[float]]:
        return await openai_embed(
            texts,
            model=settings.embedding_model,
            base_url=settings.embedding_binding_host,
            api_key=api_key,
        )

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.max_embed_tokens,
        func=embedding_func,
    )


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
