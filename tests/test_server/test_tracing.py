"""Tests for tracing module (LangSmith span wrappers + trace propagation)."""

import functools

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from twindb_lightrag_memgraph.server.tracing import (
    extract_trace_parent,
    make_trace_headers,
    is_tracing_enabled,
    _check_langsmith_config,
    _wrap_llm_func,
    _wrap_embedding_func,
    _wrap_rerank_func,
    apply_lang_with_tracing,
)


class TestExtractTraceParent:
    def test_langsmith_trace(self):
        headers = {"langsmith-trace": "trace-abc-123"}
        result = extract_trace_parent(headers)
        assert result == {"langsmith_trace_id": "trace-abc-123"}

    def test_x_langsmith_trace(self):
        headers = {"x-langsmith-trace": "trace-xyz"}
        result = extract_trace_parent(headers)
        assert result == {"langsmith_trace_id": "trace-xyz"}

    def test_w3c_traceparent(self):
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
        result = extract_trace_parent(headers)
        assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert result["parent_span_id"] == "b7ad6b7169203331"
        assert "traceparent" in result

    def test_custom_x_trace_id(self):
        headers = {"x-trace-id": "my-custom-trace"}
        result = extract_trace_parent(headers)
        assert result == {"trace_id": "my-custom-trace"}

    def test_no_trace_context(self):
        headers = {"content-type": "application/json"}
        result = extract_trace_parent(headers)
        assert result is None

    def test_priority_langsmith_over_traceparent(self):
        headers = {
            "langsmith-trace": "ls-123",
            "traceparent": "00-abc-def-01",
        }
        result = extract_trace_parent(headers)
        assert result == {"langsmith_trace_id": "ls-123"}


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


# ---------------------------------------------------------------------------
# New test classes
# ---------------------------------------------------------------------------


def _mock_traceable(**kwargs):
    """Simulate langsmith.traceable as a passthrough async decorator."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kw):
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
        assert getattr(wrapped, "_traced", False) is True

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


class TestExtractTraceParentEdgeCases:
    def test_traceparent_fewer_than_3_parts(self):
        """A traceparent with fewer than 3 dash-separated parts returns None."""
        headers = {"traceparent": "00-abc"}
        result = extract_trace_parent(headers)
        assert result is None

    def test_make_trace_headers_empty_dict(self):
        """An empty dict has no recognised keys, so output is empty."""
        assert make_trace_headers({}) == {}
