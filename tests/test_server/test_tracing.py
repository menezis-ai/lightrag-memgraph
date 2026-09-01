"""Tests for tracing module (LangSmith span wrappers + trace propagation)."""

import functools

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server.observability import (
    make_request_observability_middleware,
)
from twindb_lightrag_memgraph.server.tracing import (
    bind_trace_context,
    current_trace_context,
    extract_trace_parent,
    make_trace_headers,
    resolve_trace_context,
    is_tracing_enabled,
    _check_langsmith_config,
    _wrap_llm_func,
    _wrap_embedding_func,
    _wrap_rerank_func,
    apply_lang_with_tracing,
)

_LANGSMITH_PARENT = "20260901T004915204525Z01a05a71-13c4-7582-bcf9-e358e56dc683"
_W3C_PARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class TestExtractTraceParent:
    def test_langsmith_trace(self):
        headers = {"langsmith-trace": _LANGSMITH_PARENT}
        result = extract_trace_parent(headers)
        assert result["source"] == "langsmith"
        assert result["trace_id"] == "01a05a7113c47582bcf9e358e56dc683"
        assert result["langsmith_trace_id"] == _LANGSMITH_PARENT

    def test_x_langsmith_trace(self):
        headers = {"x-langsmith-trace": _LANGSMITH_PARENT}
        result = extract_trace_parent(headers)
        assert result["source"] == "langsmith"
        assert result["langsmith_trace_id"] == _LANGSMITH_PARENT

    def test_w3c_traceparent(self):
        headers = {"traceparent": _W3C_PARENT}
        result = extract_trace_parent(headers)
        assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert result["parent_span_id"] == "b7ad6b7169203331"
        assert "traceparent" in result

    def test_custom_x_trace_id(self):
        headers = {"x-trace-id": "my-custom-trace"}
        result = extract_trace_parent(headers)
        assert result["source"] == "custom"
        assert result["legacy_trace_id"] == "my-custom-trace"
        assert len(result["trace_id"]) == 32

    def test_no_trace_context(self):
        headers = {"content-type": "application/json"}
        result = extract_trace_parent(headers)
        assert result is None

    def test_w3c_is_canonical_while_langsmith_parent_is_preserved(self):
        headers = {
            "langsmith-trace": _LANGSMITH_PARENT,
            "traceparent": _W3C_PARENT,
        }
        result = extract_trace_parent(headers)
        assert result["source"] == "w3c"
        assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert result["langsmith_trace_id"] == _LANGSMITH_PARENT


class TestResolveTraceContext:
    def test_valid_parent_creates_a_new_server_span(self):
        context = resolve_trace_context({"traceparent": _W3C_PARENT})

        assert context.source == "w3c"
        assert context.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert context.parent_span_id == "b7ad6b7169203331"
        assert context.span_id != context.parent_span_id
        assert len(context.span_id) == 16
        assert context.traceparent == (f"00-{context.trace_id}-{context.span_id}-01")

    @staticmethod
    def _invalid_traceparents():
        return (
            "",
            "00-abc",
            "ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "00-00000000000000000000000000000000-b7ad6b7169203331-01",
            "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-xyz",
            "00-0AF7651916CD43DD8448EB211C80319C-b7ad6b7169203331-01",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra",
        )

    def test_absent_or_invalid_parent_generates_a_valid_context(self):
        for invalid in self._invalid_traceparents():
            context = resolve_trace_context({"traceparent": invalid} if invalid else {})
            assert context.source == "generated"
            assert len(context.trace_id) == 32
            assert context.trace_id != "0" * 32
            assert len(context.span_id) == 16
            assert context.span_id != "0" * 16
            assert context.traceparent == (
                f"00-{context.trace_id}-{context.span_id}-01"
            )

    def test_valid_custom_hex_id_precedes_generation(self):
        context = resolve_trace_context({"x-trace-id": "A" * 32})
        assert context.source == "custom"
        assert context.trace_id == "a" * 32

    def test_context_binding_restores_previous_value(self):
        context = resolve_trace_context({"traceparent": _W3C_PARENT})
        assert current_trace_context() is None
        with bind_trace_context(context):
            assert current_trace_context() is context
        assert current_trace_context() is None


class TestMakeTraceHeaders:
    def test_none_context(self):
        assert make_trace_headers(None) == {}

    def test_traceparent(self):
        ctx = {"traceparent": "00-abc-def-01"}
        headers = make_trace_headers(ctx)
        assert headers["traceparent"] == "00-abc-def-01"

    def test_langsmith_trace(self):
        ctx = {"langsmith_trace_id": "ls-123"}
        headers = make_trace_headers(ctx)
        assert headers["langsmith-trace"] == "ls-123"

    def test_custom_trace_id(self):
        ctx = {"trace_id": "custom-123"}
        headers = make_trace_headers(ctx)
        assert headers["x-trace-id"] == "custom-123"

    def test_active_context_propagates_w3c_and_langsmith(self):
        context = resolve_trace_context(
            {
                "traceparent": _W3C_PARENT,
                "langsmith-trace": _LANGSMITH_PARENT,
                "baggage": "langsmith-project=twin-test",
            }
        )
        with bind_trace_context(context):
            headers = make_trace_headers()

        assert headers == {
            "traceparent": context.traceparent,
            "x-trace-id": context.trace_id,
            "langsmith-trace": _LANGSMITH_PARENT,
            "baggage": "langsmith-project=twin-test",
        }


class TestWrapFunctions:
    """Test that wrapping preserves function behavior when langsmith is unavailable."""

    async def test_wrap_llm_no_langsmith(self):
        original = AsyncMock(return_value="response")
        with patch(
            "twindb_lightrag_memgraph.server.tracing._langsmith_available", False
        ):
            wrapped = _wrap_llm_func(original)
        assert wrapped is original

    async def test_wrap_embedding_no_langsmith(self):
        original = AsyncMock(return_value=[[0.1, 0.2]])
        with patch(
            "twindb_lightrag_memgraph.server.tracing._langsmith_available", False
        ):
            wrapped = _wrap_embedding_func(original)
        assert wrapped is original

    async def test_wrap_rerank_no_langsmith(self):
        original = AsyncMock(return_value=[1, 0])
        with patch(
            "twindb_lightrag_memgraph.server.tracing._langsmith_available", False
        ):
            wrapped = _wrap_rerank_func(original)
        assert wrapped is original


class TestApplyLangWithTracing:
    def test_disabled_without_api_key(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        rag = MagicMock()
        apply_lang_with_tracing(rag)
        from twindb_lightrag_memgraph.server import tracing

        assert tracing._TRACING_ENABLED is False

    def test_disabled_when_optional_package_is_missing(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "configured")
        rag = MagicMock()
        original = rag.llm_model_func
        with patch(
            "twindb_lightrag_memgraph.server.tracing._langsmith_available", False
        ):
            apply_lang_with_tracing(rag)
        from twindb_lightrag_memgraph.server import tracing

        assert tracing._TRACING_ENABLED is False
        assert rag.llm_model_func is original


# ---------------------------------------------------------------------------
# New test classes
# ---------------------------------------------------------------------------


def _mock_traceable(**kwargs):
    """Simulate langsmith.traceable as a passthrough async decorator."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kw):
            kw.pop("langsmith_extra", None)
            return await fn(*args, **kw)

        wrapper._traced = True
        return wrapper

    return decorator


class TestCheckLangsmithConfig:
    def test_returns_true_with_api_key(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test")
        assert _check_langsmith_config() is True

    def test_returns_false_without_api_key(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert _check_langsmith_config() is False

    def test_logs_project_name(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-proj")
        # When LANGSMITH_PROJECT is set the function still returns True
        # (the project name is logged internally).
        assert _check_langsmith_config() is True


class TestIsTracingEnabled:
    def test_true_when_both_flags(self):
        with (
            patch("twindb_lightrag_memgraph.server.tracing._TRACING_ENABLED", True),
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
        ):
            assert is_tracing_enabled() is True

    def test_false_when_langsmith_unavailable(self):
        with (
            patch("twindb_lightrag_memgraph.server.tracing._TRACING_ENABLED", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing._langsmith_available", False
            ),
        ):
            assert is_tracing_enabled() is False

    def test_false_when_not_enabled(self):
        with (
            patch("twindb_lightrag_memgraph.server.tracing._TRACING_ENABLED", False),
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
        ):
            assert is_tracing_enabled() is False


class TestWrapWithLangsmith:
    """Test wrapping behaviour when langsmith.traceable is available (mocked)."""

    async def test_wrap_llm_returns_different_func(self):
        original = AsyncMock(return_value="llm-response")
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            wrapped = _wrap_llm_func(original)
        assert wrapped is not original
        assert getattr(wrapped, "_twin_langsmith_wrapped", False) is True

    async def test_wrap_llm_preserves_behavior(self):
        original = AsyncMock(return_value="llm-response")
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            wrapped = _wrap_llm_func(original)
        result = await wrapped("prompt")
        assert result == "llm-response"

    async def test_wrap_embedding_preserves_behavior(self):
        original = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            wrapped = _wrap_embedding_func(original)
        result = await wrapped(["text"])
        assert result == [[0.1, 0.2, 0.3]]

    async def test_wrap_rerank_preserves_behavior(self):
        original = AsyncMock(return_value=[2, 0, 1])
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            wrapped = _wrap_rerank_func(original)
        result = await wrapped("query", ["a", "b", "c"])
        assert result == [2, 0, 1]

    async def test_all_spans_receive_the_native_parent_and_w3c_metadata(self):
        captured: list[tuple[str, dict | None]] = []

        def capturing_traceable(*, name, run_type):
            del run_type

            def decorator(fn):
                @functools.wraps(fn)
                async def wrapper(*args, **kwargs):
                    captured.append((name, kwargs.pop("langsmith_extra", None)))
                    return await fn(*args, **kwargs)

                return wrapper

            return decorator

        context = resolve_trace_context(
            {
                "traceparent": _W3C_PARENT,
                "langsmith-trace": _LANGSMITH_PARENT,
                "baggage": "langsmith-project=twin-test",
            }
        )
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                capturing_traceable,
                create=True,
            ),
        ):
            wrapped = (
                _wrap_llm_func(AsyncMock(return_value="answer")),
                _wrap_embedding_func(AsyncMock(return_value=[[0.1]])),
                _wrap_rerank_func(AsyncMock(return_value=[0])),
            )

        with bind_trace_context(context):
            await wrapped[0]("question")
            await wrapped[1](["text"])
            await wrapped[2]("question", ["text"])

        assert [name for name, _ in captured] == [
            "Lightrag:llm",
            "Lightrag:embedding",
            "Lightrag:rerank",
        ]
        for _, extra in captured:
            assert extra["parent"] == {
                "langsmith-trace": _LANGSMITH_PARENT,
                "baggage": "langsmith-project=twin-test",
            }
            assert extra["metadata"] == {
                "w3c.trace_id": context.trace_id,
                "w3c.span_id": context.span_id,
            }

    async def test_http_parent_reaches_the_llm_span_end_to_end(self):
        captured: list[dict] = []

        def capturing_traceable(**_decorator_kwargs):
            def decorator(fn):
                @functools.wraps(fn)
                async def wrapper(*args, **kwargs):
                    captured.append(kwargs.pop("langsmith_extra"))
                    return await fn(*args, **kwargs)

                return wrapper

            return decorator

        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                capturing_traceable,
                create=True,
            ),
        ):
            wrapped_llm = _wrap_llm_func(AsyncMock(return_value="answer"))

        app = FastAPI()
        app.middleware("http")(make_request_observability_middleware())

        @app.get("/query-probe")
        async def query_probe():
            return {"answer": await wrapped_llm("question")}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/query-probe",
                headers={
                    "traceparent": _W3C_PARENT,
                    "langsmith-trace": _LANGSMITH_PARENT,
                },
            )

        assert response.status_code == 200
        assert len(captured) == 1
        assert captured[0]["parent"] == {"langsmith-trace": _LANGSMITH_PARENT}
        assert captured[0]["metadata"]["w3c.trace_id"] == (
            "0af7651916cd43dd8448eb211c80319c"
        )
        assert response.headers["traceparent"].split("-")[1] == (
            captured[0]["metadata"]["w3c.trace_id"]
        )


class TestApplyTracingWithMocks:
    """Test apply_lang_with_tracing with mocked _check_langsmith_config and traceable."""

    def _make_rag(self, **overrides):
        """Build a minimal mock RAG instance with standard attributes."""
        rag = MagicMock()
        rag.llm_model_func = AsyncMock(return_value="answer")
        rag.embedding_func = AsyncMock(return_value=[[0.1]])
        # Storage instances that should receive the traced embedding.
        for attr in ("text_chunks", "entities_vdb", "relationships_vdb", "chunks_vdb"):
            storage = MagicMock()
            storage.embedding_func = AsyncMock()
            setattr(rag, attr, storage)
        # By default, no rerank attributes -- tests add them explicitly.
        del rag.rerank_func
        del rag.reranking_func
        del rag._rerank_func
        # Apply overrides.
        for key, value in overrides.items():
            setattr(rag, key, value)
        return rag

    def test_wraps_llm_func(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rag = self._make_rag()
        original_llm = rag.llm_model_func
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            apply_lang_with_tracing(rag)
        assert rag.llm_model_func is not original_llm

    def test_wraps_embedding_and_propagates(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rag = self._make_rag()
        original_embed = rag.embedding_func
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            apply_lang_with_tracing(rag)
        # embedding_func on rag itself is replaced.
        assert rag.embedding_func is not original_embed
        # Propagated to storage instances.
        traced_embed = rag.embedding_func
        for attr in ("text_chunks", "entities_vdb", "relationships_vdb", "chunks_vdb"):
            storage = getattr(rag, attr)
            assert storage.embedding_func is traced_embed

    def test_wraps_rerank_func(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        original_rerank = AsyncMock(return_value=[1, 0])
        rag = self._make_rag(rerank_func=original_rerank)
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            apply_lang_with_tracing(rag)
        assert rag.rerank_func is not original_rerank

    def test_tries_reranking_func_variant(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        original_reranking = AsyncMock(return_value=[0, 1])
        rag = self._make_rag(reranking_func=original_reranking)
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            apply_lang_with_tracing(rag)
        # reranking_func should have been wrapped.
        assert rag.reranking_func is not original_reranking

    def test_no_rerank_attribute_no_error(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rag = self._make_rag()
        # No rerank_func / reranking_func / _rerank_func at all.
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            # Should not raise.
            apply_lang_with_tracing(rag)

    def test_propagate_skips_none_storage(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rag = self._make_rag()
        rag.entities_vdb = None
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            # Should not raise even though entities_vdb is None.
            apply_lang_with_tracing(rag)
        # Other storages still got the propagated embedding.
        assert rag.text_chunks.embedding_func is rag.embedding_func

    def test_propagate_skips_storage_without_embedding_func(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rag = self._make_rag()
        # Replace one storage with an object that has no embedding_func attr.
        plain_storage = MagicMock(spec=[])  # spec=[] means no attributes
        rag.chunks_vdb = plain_storage
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            # Should not raise.
            apply_lang_with_tracing(rag)
        # The plain_storage should NOT have gained an embedding_func attribute.
        assert not hasattr(plain_storage, "embedding_func")

    def test_apply_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rag = self._make_rag(rerank_func=AsyncMock(return_value=[0]))
        with (
            patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
            patch(
                "twindb_lightrag_memgraph.server.tracing.traceable",
                _mock_traceable,
                create=True,
            ),
        ):
            apply_lang_with_tracing(rag)
            first = (rag.llm_model_func, rag.embedding_func, rag.rerank_func)
            apply_lang_with_tracing(rag)

        assert (rag.llm_model_func, rag.embedding_func, rag.rerank_func) == first


class TestExtractTraceParentEdgeCases:
    def test_traceparent_fewer_than_3_parts(self):
        """A traceparent with fewer than 3 dash-separated parts returns None."""
        headers = {"traceparent": "00-abc"}
        result = extract_trace_parent(headers)
        assert result is None

    def test_make_trace_headers_empty_dict(self):
        """An empty dict has no recognised keys, so output is empty."""
        assert make_trace_headers({}) == {}
