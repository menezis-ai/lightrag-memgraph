"""Tests for the app factory and helper functions."""

import logging

import pytest

from twindb_lightrag_memgraph.server.app import _extract_doc_ids


class TestExtractDocIds:
    def test_dict_with_contexts(self):
        result = {
            "response": "answer",
            "contexts": [
                {"full_doc_id": "doc-1", "content": "..."},
                {"full_doc_id": "doc-2", "content": "..."},
                {"full_doc_id": "doc-1", "content": "..."},  # duplicate
            ],
        }
        doc_ids = _extract_doc_ids(result)
        assert doc_ids == ["doc-1", "doc-2"]

    def test_dict_with_doc_id_fallback(self):
        result = {
            "contexts": [
                {"doc_id": "doc-3"},
            ],
        }
        doc_ids = _extract_doc_ids(result)
        assert doc_ids == ["doc-3"]

    def test_string_result(self):
        doc_ids = _extract_doc_ids("plain text response")
        assert doc_ids == []

    def test_empty_contexts(self):
        doc_ids = _extract_doc_ids({"contexts": []})
        assert doc_ids == []

    def test_none_result(self):
        doc_ids = _extract_doc_ids(None)
        assert doc_ids == []


# ---------------------------------------------------------------------------
# HTTP-level tests using httpx.AsyncClient + ASGITransport
# ---------------------------------------------------------------------------
#
# httpx.ASGITransport does NOT trigger ASGI lifespan events, so the
# module-level ``_rag`` is never set by the lifespan during tests.
# Strategy:
#   - Endpoint tests: inject ``_rag`` directly into the module.
#   - Lifespan tests: invoke ``app.router.lifespan_context`` directly.
# ---------------------------------------------------------------------------

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

import twindb_lightrag_memgraph.server.app as app_module
from twindb_lightrag_memgraph.server.app import (
    _build_embedding_func,
    _build_llm_func,
    _effective_graph_workspace,
    _get_rag,
    _production_auth_required,
    create_app,
)
from twindb_lightrag_memgraph.server.auth import AuthConfigurationError
from twindb_lightrag_memgraph.server import webui_seed
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings
from twindb_lightrag_memgraph.server.tracing import metrics_snapshot, reset_metrics


def _make_settings(*, api_key="test-key", jwt_secret=None, **overrides):
    """Build a LightRAGServerSettings with sensible test defaults."""
    defaults = dict(
        working_dir="/tmp/lightrag_test",
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        graph_storage="MemgraphStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        workspace="test_ws",
        enable_langsmith_tracing=False,
        graph_relation_id_backfill_on_startup=False,
        webui_tag_backend="memory",
        webui_activity_backend="memory",
        webui_notifications_backend="memory",
        api_key=api_key,
        jwt_secret=jwt_secret,
    )
    defaults.update(overrides)
    return LightRAGServerSettings(**defaults)


def _make_mock_rag():
    """Build a mock LightRAG instance with async methods."""
    mock_rag = MagicMock()
    mock_rag.initialize = AsyncMock()
    mock_rag.aquery = AsyncMock(return_value="mocked answer")
    mock_rag.ainsert = AsyncMock()
    mock_rag.text_chunks = MagicMock()
    mock_rag.doc_status = MagicMock()
    return mock_rag


def _stub_embedding_func(settings, api_key):
    """Stub replacement for _build_embedding_func (avoids openai import)."""

    async def _embed(texts):
        return [[0.0] * settings.embedding_dim for _ in texts]

    _embed.embedding_dim = settings.embedding_dim
    _embed.max_token_size = settings.max_embed_tokens
    return _embed


def _stub_llm_func(settings, api_key):
    """Stub replacement for _build_llm_func (avoids openai import)."""

    async def _llm(prompt, **kwargs):
        return "stub"

    return _llm


class TestCorsConfiguration:
    def test_rejects_wildcard_with_credentials(self):
        settings = _make_settings(
            cors_allowed_origins=["*"],
            cors_allow_credentials=True,
        )

        with pytest.raises(ValueError, match="CORS_ALLOWED_ORIGINS"):
            create_app(settings)

    async def test_allows_only_configured_credentialed_origin(self):
        settings = _make_settings(
            cors_allowed_origins=["https://spa.example"],
            cors_allow_credentials=True,
        )
        app = create_app(settings)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://api.example",
        ) as client:
            allowed = await client.options(
                "/health",
                headers={
                    "Origin": "https://spa.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            denied = await client.options(
                "/health",
                headers={
                    "Origin": "https://evil.example",
                    "Access-Control-Request-Method": "GET",
                },
            )

        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "https://spa.example"
        assert allowed.headers["access-control-allow-credentials"] == "true"
        assert denied.status_code == 400
        assert "access-control-allow-origin" not in denied.headers


class TestProductionAuthMode:
    def test_flag_parser_accepts_explicit_require_auth(self):
        assert _production_auth_required({"TWIN_REQUIRE_AUTH": "true"}) is True
        assert _production_auth_required({"TWIN_REQUIRE_AUTH": "1"}) is True
        assert _production_auth_required({"TWIN_REQUIRE_AUTH": "false"}) is False

    def test_flag_parser_accepts_twin_env_production(self):
        assert _production_auth_required({"TWIN_ENV": "production"}) is True
        assert _production_auth_required({"TWIN_ENV": "dev"}) is False

    def test_production_without_auth_backend_fails_fast(self, monkeypatch):
        monkeypatch.setenv("TWIN_REQUIRE_AUTH", "true")
        monkeypatch.delenv("TOKEN_SECRET", raising=False)
        monkeypatch.delenv("AUTH_ACCOUNTS", raising=False)
        monkeypatch.delenv("TWIN_IDP_JWKS_URL", raising=False)

        with pytest.raises(AuthConfigurationError, match="Production auth requires"):
            create_app(_make_settings(api_key=None, jwt_secret=None))

    def test_production_with_static_api_key_boots(self, monkeypatch):
        monkeypatch.setenv("TWIN_REQUIRE_AUTH", "true")
        app = create_app(_make_settings(api_key="test-key", jwt_secret=None))
        assert app.title.startswith("LightRAG Server")

    def test_production_with_strong_local_jwt_boots(self, monkeypatch):
        monkeypatch.setenv("TWIN_ENV", "production")
        app = create_app(
            _make_settings(
                api_key=None,
                jwt_secret="x" * 32,
                jwt_password="not-the-default",
            )
        )
        assert app.title.startswith("LightRAG Server")

    def test_production_with_idp_backend_boots(self, monkeypatch):
        monkeypatch.setenv("TWIN_REQUIRE_AUTH", "true")
        monkeypatch.setenv("TWIN_IDP_JWKS_URL", "https://idp.example/jwks")
        app = create_app(_make_settings(api_key=None, jwt_secret=None))
        assert app.title.startswith("LightRAG Server")

    def test_production_rejects_default_local_jwt_password(self, monkeypatch):
        monkeypatch.setenv("TWIN_REQUIRE_AUTH", "true")

        with pytest.raises(AuthConfigurationError, match="default"):
            create_app(_make_settings(api_key=None, jwt_secret="x" * 32))

    def test_production_rejects_default_auth_accounts_password(self, monkeypatch):
        monkeypatch.setenv("TWIN_REQUIRE_AUTH", "true")
        monkeypatch.setenv("AUTH_ACCOUNTS", "alice:changeme,bob:ok")

        with pytest.raises(AuthConfigurationError, match="alice"):
            create_app(
                _make_settings(
                    api_key=None,
                    jwt_secret="x" * 32,
                    jwt_password="not-the-default",
                )
            )

    def test_production_rejects_weak_hs256_secret(self, monkeypatch):
        monkeypatch.setenv("TWIN_REQUIRE_AUTH", "true")

        with pytest.raises(AuthConfigurationError, match="at least 32 bytes"):
            create_app(
                _make_settings(
                    api_key=None,
                    jwt_secret="short",
                    jwt_password="not-the-default",
                )
            )


def _apply_lifespan_patches(stack, mock_rag, register_mock=None):
    """Enter patches needed for lifespan tests on an ExitStack."""
    stack.enter_context(
        patch(
            "twindb_lightrag_memgraph.server.app.LightRAG",
            return_value=mock_rag,
        )
    )
    stack.enter_context(
        patch(
            "twindb_lightrag_memgraph.register",
            register_mock or MagicMock(),
        )
    )
    stack.enter_context(
        patch(
            "twindb_lightrag_memgraph.server.app._build_embedding_func",
            side_effect=_stub_embedding_func,
        )
    )
    stack.enter_context(
        patch(
            "twindb_lightrag_memgraph.server.app._build_llm_func",
            side_effect=_stub_llm_func,
        )
    )
    return stack


@pytest.fixture()
def _mock_rag():
    """Fixture providing a fresh mock LightRAG for each test."""
    return _make_mock_rag()


@pytest.fixture()
async def _client_with_auth(_mock_rag):
    """Yield an AsyncClient against a FastAPI app with auth enabled.

    Manually injects ``_rag`` into the app module since ASGITransport
    does not trigger ASGI lifespan events.
    The mock_rag is accessible via ``client._test_mock_rag``.
    """
    settings = _make_settings(api_key="test-key")
    app = create_app(settings)
    original_rag = app_module._rag
    app_module._rag = _mock_rag
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            client._test_mock_rag = _mock_rag
            yield client
    finally:
        app_module._rag = original_rag


@pytest.fixture()
async def _client_no_auth(_mock_rag):
    """Yield an AsyncClient against a FastAPI app with auth disabled."""
    settings = _make_settings(api_key=None, jwt_secret=None)
    app = create_app(settings)
    original_rag = app_module._rag
    app_module._rag = _mock_rag
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            client._test_mock_rag = _mock_rag
            yield client
    finally:
        app_module._rag = original_rag


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    async def test_health_returns_200(self, _client_with_auth):
        """GET /health returns 200 with expected fields (no auth required)."""
        resp = await _client_with_auth.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["workspace"] == "test_ws"
        assert "version" in body
        assert "storage_backends" in body
        backends = body["storage_backends"]
        assert backends["kv"] == "MemgraphKVStorage"
        assert backends["vector"] == "MemgraphVectorDBStorage"
        assert backends["graph"] == "MemgraphStorage"
        assert backends["doc_status"] == "MemgraphDocStatusStorage"
        assert body["tracing_enabled"] is False

    async def test_health_no_auth_required(self, _client_with_auth):
        """GET /health succeeds even when auth is enabled and no token is sent."""
        resp = await _client_with_auth.get("/health")
        assert resp.status_code == 200

    async def test_health_does_not_probe_memgraph(self, monkeypatch, _client_no_auth):
        async def broken_readiness_probe():
            raise AssertionError("/health must stay lightweight")

        monkeypatch.setattr(
            app_module,
            "_memgraph_readiness_check",
            broken_readiness_probe,
        )
        resp = await _client_no_auth.get("/health")
        assert resp.status_code == 200


class _ReadyVector:
    async def query(self, *args, **kwargs):
        return []


class TestReadinessEndpoint:
    async def test_ready_returns_200_when_dependencies_are_available(
        self, monkeypatch, _mock_rag
    ):
        async def memgraph_ok():
            return {"status": "ok"}

        monkeypatch.setattr(app_module, "_memgraph_readiness_check", memgraph_ok)
        _mock_rag.chunks_vdb = _ReadyVector()
        app = create_app(_make_settings(api_key=None, jwt_secret=None))
        original_rag = app_module._rag
        app_module._rag = _mock_rag
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/ready")
        finally:
            app_module._rag = original_rag

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["checks"]["memgraph"]["status"] == "ok"
        assert body["checks"]["lightrag"]["status"] == "ok"
        assert body["checks"]["vector_index"]["status"] == "ok"

    async def test_ready_returns_503_when_memgraph_is_unreachable(
        self, monkeypatch, _mock_rag
    ):
        async def memgraph_failed():
            return {"status": "failed", "detail": "ServiceUnavailable"}

        monkeypatch.setattr(app_module, "_memgraph_readiness_check", memgraph_failed)
        _mock_rag.chunks_vdb = _ReadyVector()
        app = create_app(_make_settings(api_key=None, jwt_secret=None))
        original_rag = app_module._rag
        app_module._rag = _mock_rag
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/ready")
        finally:
            app_module._rag = original_rag

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["memgraph"]["status"] == "failed"


class TestOperationalMiddleware:
    async def test_request_id_header_and_access_log_do_not_include_body(
        self, caplog
    ):
        app = create_app(_make_settings(api_key=None, jwt_secret=None))
        caplog.set_level(logging.INFO, logger=app_module.__name__)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get(
                "/health",
                headers={
                    "x-request-id": "rid-123",
                    "authorization": "Bearer secret",
                    "traceparent": (
                        "00-4bf92f3577b34da6a3ce929d0e0e4736-"
                        "00f067aa0ba902b7-01"
                    ),
                },
            )

        assert resp.status_code == 200
        assert resp.headers["x-request-id"] == "rid-123"
        assert "request_id=rid-123" in caplog.text
        assert "route_group=health" in caplog.text
        assert "trace_id=4bf92f3577b34da6a3ce929d0e0e4736" in caplog.text
        assert "secret" not in caplog.text

    async def test_regular_body_limit_returns_413_before_auth(self):
        app = create_app(
            _make_settings(
                api_key="test-key",
                max_request_body_bytes=8,
                max_upload_body_bytes=100,
            )
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/query", content=b"123456789")

        assert resp.status_code == 413
        assert resp.json()["limit_bytes"] == 8
        assert "x-request-id" in resp.headers

    async def test_upload_path_uses_upload_limit(self):
        app = create_app(
            _make_settings(
                api_key=None,
                jwt_secret=None,
                max_request_body_bytes=1,
                max_upload_body_bytes=100,
            )
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/documents/upload", content=b"123456789")

        assert resp.status_code != 413

    async def test_auth_reject_counter_increments(self):
        reset_metrics()
        app = create_app(_make_settings(api_key="test-key"))

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/query", json={"query": "hello"})

        assert resp.status_code in {401, 403}
        assert metrics_snapshot()["auth_rejects_total"] == 1

    async def test_query_failure_counter_increments_without_logging_prompt(
        self, caplog, _mock_rag
    ):
        reset_metrics()
        _mock_rag.aquery.side_effect = RuntimeError("backend offline")
        app = create_app(_make_settings(api_key=None, jwt_secret=None))
        original_rag = app_module._rag
        app_module._rag = _mock_rag
        caplog.set_level(logging.WARNING, logger=app_module.__name__)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/query",
                    json={"query": "secret prompt body"},
                )
        finally:
            app_module._rag = original_rag

        assert resp.status_code == 500
        assert metrics_snapshot()["query_failures_total"] == 1
        assert "route_group=query" in caplog.text
        assert "secret prompt body" not in caplog.text


# ---------------------------------------------------------------------------
# Query endpoint -- auth
# ---------------------------------------------------------------------------


class TestQueryAuth:
    async def test_query_returns_200_with_auth(self, _client_with_auth):
        """POST /query with valid Bearer token returns 200."""
        _client_with_auth._test_mock_rag.aquery.return_value = "answer"

        resp = await _client_with_auth.post(
            "/query",
            json={"query": "What is LightRAG?"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "answer"

    async def test_query_returns_401_without_auth(self, _client_with_auth):
        """POST /query without Authorization header returns 401 when auth enabled."""
        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test"},
        )

        assert resp.status_code == 401 or resp.status_code == 403

    async def test_query_returns_401_with_wrong_key(self, _client_with_auth):
        """POST /query with wrong API key returns 401."""
        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Query endpoint -- mode forwarding
# ---------------------------------------------------------------------------


class TestQueryMode:
    async def test_query_passes_mode_param(self, _client_with_auth):
        """The mode parameter is forwarded to rag.aquery."""
        mock_rag = _client_with_auth._test_mock_rag
        mock_rag.aquery.return_value = "result"

        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test", "mode": "local"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        mock_rag.aquery.assert_awaited_once()
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["param"]["mode"] == "local"

    async def test_query_default_mode_hybrid(self, _client_with_auth):
        """When mode is not specified, it defaults to 'hybrid'."""
        mock_rag = _client_with_auth._test_mock_rag
        mock_rag.aquery.return_value = "result"

        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["param"]["mode"] == "hybrid"

    async def test_query_passes_only_need_context(self, _client_with_auth):
        """The only_need_context parameter is forwarded to rag.aquery."""
        mock_rag = _client_with_auth._test_mock_rag
        mock_rag.aquery.return_value = "ctx only"

        resp = await _client_with_auth.post(
            "/query",
            json={
                "query": "test",
                "only_need_context": True,
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["param"]["only_need_context"] is True


# ---------------------------------------------------------------------------
# Query endpoint -- response parsing
# ---------------------------------------------------------------------------


class TestQueryResponse:
    async def test_query_extracts_doc_ids_from_dict(self, _client_with_auth):
        """When rag.aquery returns a dict with contexts, source_doc_ids is populated."""
        mock_rag = _client_with_auth._test_mock_rag
        mock_rag.aquery.return_value = {
            "response": "answer with sources",
            "contexts": [
                {"full_doc_id": "doc-A", "content": "..."},
                {"full_doc_id": "doc-B", "content": "..."},
            ],
        }

        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_doc_ids"] == ["doc-A", "doc-B"]

    async def test_query_string_result(self, _client_with_auth):
        """When rag.aquery returns a plain string, source_doc_ids is empty."""
        mock_rag = _client_with_auth._test_mock_rag
        mock_rag.aquery.return_value = "just a string"

        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "just a string"
        assert body["source_doc_ids"] == []

    async def test_query_dict_result_is_stringified(self, _client_with_auth):
        """When rag.aquery returns a dict, response is the string representation."""
        mock_rag = _client_with_auth._test_mock_rag
        result_dict = {"response": "some answer", "contexts": []}
        mock_rag.aquery.return_value = result_dict

        resp = await _client_with_auth.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        body = resp.json()
        # Dict results are stringified via str()
        assert body["response"] == str(result_dict)


# ---------------------------------------------------------------------------
# Insert endpoint
# ---------------------------------------------------------------------------


class TestInsertEndpoint:
    async def test_insert_returns_200(self, _client_with_auth):
        """POST /insert with valid auth and text body returns 200."""
        resp = await _client_with_auth.post(
            "/insert",
            json={"text": "Hello world document."},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    async def test_insert_calls_ainsert(self, _client_with_auth):
        """POST /insert delegates to rag.ainsert with the provided text."""
        mock_rag = _client_with_auth._test_mock_rag

        resp = await _client_with_auth.post(
            "/insert",
            json={"text": "Some document content."},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 200
        mock_rag.ainsert.assert_awaited_once_with("Some document content.")

    async def test_insert_returns_401_without_auth(self, _client_with_auth):
        """POST /insert without auth header returns 401 when auth enabled."""
        resp = await _client_with_auth.post(
            "/insert",
            json={"text": "test"},
        )

        assert resp.status_code == 401 or resp.status_code == 403

    async def test_insert_missing_text_field(self, _client_with_auth):
        """POST /insert without required 'text' field returns 422."""
        resp = await _client_with_auth.post(
            "/insert",
            json={"metadata": {"key": "value"}},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Auth disabled
# ---------------------------------------------------------------------------


class TestAuthDisabled:
    async def test_query_accessible_when_auth_disabled(self, _client_no_auth):
        """When api_key and jwt_secret are both None, /query works without auth."""
        _client_no_auth._test_mock_rag.aquery.return_value = "open access answer"

        resp = await _client_no_auth.post(
            "/query",
            json={"query": "test"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "open access answer"

    async def test_insert_accessible_when_auth_disabled(self, _client_no_auth):
        """When auth disabled, /insert works without auth header."""
        resp = await _client_no_auth.post(
            "/insert",
            json={"text": "no auth needed"},
        )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _get_rag guard
# ---------------------------------------------------------------------------


class TestGetRag:
    def test_get_rag_raises_when_none(self):
        """_get_rag raises RuntimeError when _rag is None."""
        original = app_module._rag
        try:
            app_module._rag = None
            with pytest.raises(RuntimeError, match="LightRAG not initialized"):
                _get_rag()
        finally:
            app_module._rag = original

    def test_get_rag_returns_instance_when_set(self):
        """_get_rag returns the module-level _rag when it is set."""
        original = app_module._rag
        sentinel = object()
        try:
            app_module._rag = sentinel
            assert _get_rag() is sentinel
        finally:
            app_module._rag = original


# ---------------------------------------------------------------------------
# _build_embedding_func / _build_llm_func
# ---------------------------------------------------------------------------


class TestBuildEmbeddingFunc:
    def test_build_embedding_func_attributes(self):
        """_build_embedding_func sets .embedding_dim and .max_token_size."""
        settings = _make_settings(embedding_dim=768, max_embed_tokens=512)

        # openai_embedding may not exist as a top-level attr in the installed
        # lightrag version; use create=True so the patch adds it.
        with patch(
            "lightrag.llm.openai.openai_embedding",
            new_callable=AsyncMock,
            create=True,
        ):
            func = _build_embedding_func(settings, api_key="fake")

        assert func.embedding_dim == 768
        assert func.max_token_size == 512

    async def test_build_embedding_func_delegates(self):
        """The returned embedding function calls openai_embedding."""
        settings = _make_settings()
        mock_openai_embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        with patch(
            "lightrag.llm.openai.openai_embedding",
            mock_openai_embed,
            create=True,
        ):
            func = _build_embedding_func(settings, api_key="embed-key")
            result = await func(["hello"])

        mock_openai_embed.assert_awaited_once_with(
            ["hello"],
            model=settings.embedding_model,
            base_url=settings.embedding_binding_host,
            api_key="embed-key",
        )
        assert result == [[0.1, 0.2, 0.3]]


class TestBuildLlmFunc:
    async def test_build_llm_func_delegates(self):
        """The returned LLM function calls openai_complete with the right params."""
        settings = _make_settings()
        mock_openai_complete = AsyncMock(return_value="LLM response")

        with patch(
            "lightrag.llm.openai.openai_complete",
            mock_openai_complete,
        ):
            func = _build_llm_func(settings, api_key="llm-key")
            result = await func("What is AI?", temperature=0.5)

        mock_openai_complete.assert_awaited_once_with(
            "What is AI?",
            model=settings.llm_model,
            base_url=settings.llm_binding_host,
            api_key="llm-key",
            temperature=0.5,
        )
        assert result == "LLM response"

    async def test_build_llm_func_no_extra_kwargs(self):
        """The LLM function works without extra kwargs."""
        settings = _make_settings()
        mock_openai_complete = AsyncMock(return_value="ok")

        with patch(
            "lightrag.llm.openai.openai_complete",
            mock_openai_complete,
        ):
            func = _build_llm_func(settings, api_key="key")
            await func("prompt")

        mock_openai_complete.assert_awaited_once_with(
            "prompt",
            model=settings.llm_model,
            base_url=settings.llm_binding_host,
            api_key="key",
        )


# ---------------------------------------------------------------------------
# Lifespan: register() is called, RAG initialized, teardown cleans up
#
# We invoke ``app.router.lifespan_context(app)`` directly as an async
# context manager, which is how Starlette/FastAPI runs the lifespan
# internally.  This avoids the ASGITransport limitation.
# ---------------------------------------------------------------------------


class TestLifespan:
    def test_effective_graph_workspace_prefers_rag_workspace(self, _mock_rag):
        settings = _make_settings(workspace="settings_ws")
        _mock_rag.workspace = "runtime_ws"

        assert _effective_graph_workspace(settings, _mock_rag) == "runtime_ws"

    async def test_register_called_during_startup(self, _mock_rag):
        """register() is called exactly once when the app starts."""
        settings = _make_settings()
        mock_register = MagicMock()

        with ExitStack() as stack:
            _apply_lifespan_patches(stack, _mock_rag, register_mock=mock_register)
            app = create_app(settings)

            async with app.router.lifespan_context(app):
                mock_register.assert_called_once()

    async def test_rag_initialize_called(self, _mock_rag):
        """LightRAG.initialize() is awaited during lifespan startup."""
        settings = _make_settings()

        with ExitStack() as stack:
            _apply_lifespan_patches(stack, _mock_rag)
            app = create_app(settings)

            async with app.router.lifespan_context(app):
                _mock_rag.initialize.assert_awaited_once()

    async def test_relation_id_backfill_runs_during_startup(self, _mock_rag):
        settings = _make_settings(
            graph_relation_id_backfill_on_startup=True,
            graph_relation_id_backfill_batch_size=77,
        )
        _mock_rag.workspace = "runtime_ws"
        backfill = AsyncMock(return_value=3)

        with ExitStack() as stack:
            _apply_lifespan_patches(stack, _mock_rag)
            stack.enter_context(
                patch(
                    "twindb_lightrag_memgraph.server.graph_reader.backfill_relation_ids",
                    backfill,
                )
            )
            app = create_app(settings)

            async with app.router.lifespan_context(app):
                pass

        backfill.assert_awaited_once_with("runtime_ws", batch_size=77)

    async def test_memgraph_webui_stores_boot_fresh_without_seed(
        self, monkeypatch, _mock_rag
    ):
        """Memgraph-backed WebUI stores must not seed demo tags/feed/notifs."""
        import json

        from twindb_lightrag_memgraph.server import webui_activitystore
        from twindb_lightrag_memgraph.server import webui_notificationstore
        from twindb_lightrag_memgraph.server import webui_router
        from twindb_lightrag_memgraph.server import webui_tagstore

        calls: list[str] = []

        class FakeTagStore:
            def __init__(self, workspace: str = "default") -> None:
                self.workspace = workspace
                self._tags: list[dict] = []
                self._categories: list[dict] = []

            async def initialize(self) -> None:
                calls.append(f"tag:init:{self.workspace}")

            async def bootstrap_categories_if_empty(self) -> bool:
                calls.append(f"tag:categories:{self.workspace}")
                self._categories = [
                    {"id": "governance", "label": "Governance", "color": "#000000"}
                ]
                return True

            async def bootstrap_if_empty(self) -> bool:
                calls.append(f"tag:seed:{self.workspace}")
                self._tags = [{"tag": "rman"}]
                return True

            def list_tags(self) -> list[dict]:
                return list(self._tags)

            def list_categories(self) -> list[dict]:
                return list(self._categories)

        class FakeActivityStore:
            def __init__(self, workspace: str = "default") -> None:
                self.workspace = workspace
                self._events: list[dict] = []

            async def initialize(self) -> None:
                calls.append(f"activity:init:{self.workspace}")

            async def bootstrap_if_empty(self) -> bool:
                calls.append(f"activity:seed:{self.workspace}")
                self._events = [{"id": "evt_seed"}]
                return True

            async def list(self, **_filters):
                return list(self._events), len(self._events), webui_seed.ACTIVITY_NOW_MS

            async def append(self, event: dict) -> dict:
                self._events.insert(0, dict(event))
                return dict(event)

        class FakeNotificationStore:
            def __init__(self, workspace: str = "default") -> None:
                self.workspace = workspace
                self._items: list[dict] = []

            async def initialize(self) -> None:
                calls.append(f"notification:init:{self.workspace}")

            async def bootstrap_if_empty(self) -> bool:
                calls.append(f"notification:seed:{self.workspace}")
                self._items = [{"id": "n_seed"}]
                return True

            async def list(self) -> list[dict]:
                return list(self._items)

            async def mark_all_read(self) -> None:
                pass

            async def clear(self) -> None:
                self._items.clear()

            async def push(self, notification: dict) -> dict:
                self._items.insert(0, dict(notification))
                return dict(notification)

        monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
        monkeypatch.setenv(
            "TWIN_FOLDERS_JSON",
            json.dumps(
                [
                    {"id": "default", "label": "Default", "kind": "primary"},
                    {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
                ]
            ),
        )
        monkeypatch.setattr(webui_tagstore, "MemgraphTagStore", FakeTagStore)
        monkeypatch.setattr(
            webui_activitystore, "MemgraphActivityStore", FakeActivityStore
        )
        monkeypatch.setattr(
            webui_notificationstore,
            "MemgraphNotificationStore",
            FakeNotificationStore,
        )

        webui_router.reset_store()
        settings = _make_settings(
            webui_tag_backend="memgraph",
            webui_activity_backend="memgraph",
            webui_notifications_backend="memgraph",
        )

        try:
            with ExitStack() as stack:
                _apply_lifespan_patches(stack, _mock_rag)
                app = create_app(settings)

                async with app.router.lifespan_context(app):
                    default_store = webui_router.get_store("default")
                    sandbox_store = webui_router.get_store("sandbox")

                    assert await default_store.list_tags() == []
                    assert await sandbox_store.list_tags() == []
                    assert await default_store.list_notifications() == []
                    assert await sandbox_store.list_notifications() == []
                    default_activity, default_total, _ = await default_store.list_activity()
                    sandbox_activity, sandbox_total, _ = await sandbox_store.list_activity()
                    assert default_total == 0
                    assert sandbox_total == 0
                    assert default_activity == []
                    assert sandbox_activity == []
                    assert await default_store.list_tag_categories() == [
                        {
                            "id": "governance",
                            "label": "Governance",
                            "color": "#000000",
                        }
                    ]
        finally:
            webui_router.reset_store()

        assert "tag:seed:default" not in calls
        assert "tag:seed:sandbox" not in calls
        assert "activity:seed:default" not in calls
        assert "activity:seed:sandbox" not in calls
        assert "notification:seed:default" not in calls
        assert "notification:seed:sandbox" not in calls

    async def test_rag_set_during_startup(self, _mock_rag):
        """_rag is set to the LightRAG instance during lifespan startup."""
        settings = _make_settings()
        original_rag = app_module._rag

        try:
            with ExitStack() as stack:
                _apply_lifespan_patches(stack, _mock_rag)
                app = create_app(settings)

                async with app.router.lifespan_context(app):
                    assert app_module._rag is _mock_rag
        finally:
            app_module._rag = original_rag

    async def test_rag_set_to_none_after_shutdown(self, _mock_rag):
        """After the lifespan exits, _rag is set back to None."""
        settings = _make_settings()
        original_rag = app_module._rag

        try:
            with ExitStack() as stack:
                _apply_lifespan_patches(stack, _mock_rag)
                app = create_app(settings)

                async with app.router.lifespan_context(app):
                    pass  # startup done

            # After exiting, _rag should be None
            assert app_module._rag is None
        finally:
            app_module._rag = original_rag


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    async def test_query_missing_query_field(self, _client_with_auth):
        """POST /query without 'query' field returns 422."""
        resp = await _client_with_auth.post(
            "/query",
            json={"mode": "local"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert resp.status_code == 422

    async def test_query_empty_body(self, _client_with_auth):
        """POST /query with empty body returns 422."""
        resp = await _client_with_auth.post(
            "/query",
            content="",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )

        assert resp.status_code == 422

    async def test_insert_empty_body(self, _client_with_auth):
        """POST /insert with empty body returns 422."""
        resp = await _client_with_auth.post(
            "/insert",
            content="",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )

        assert resp.status_code == 422
