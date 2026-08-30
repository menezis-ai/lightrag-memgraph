"""LangSmith tracing integration for LightRAG server functions.

Wraps LLM, embedding, and reranking calls with LangSmith spans so that
the full RAG pipeline is visible in the LangSmith dashboard.

Spans
-----
=================  ==========  ====================================
Span               Run type    Notes
=================  ==========  ====================================
Lightrag:llm       llm         Keyword extraction / answer synthesis
Lightrag:embedding embedding   Query vectorisation
Lightrag:rerank    chain       Result reranking (cross-encoder)
=================  ==========  ====================================

Configuration
-------------
- Server-side spans: ``LIGHTRAG_ENABLE_LANGSMITH_TRACING=true``
- Distributed tracing: ``RETRIEVER_LIGHTRAG_DISTRIBUTED_TRACING=true``
  plus ``LANGSMITH_API_KEY`` (and optionally ``LANGSMITH_PROJECT``).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRACING_ENABLED = False
_langsmith_available = False
_METRIC_COUNTERS: dict[str, int] = {
    "auth_rejects_total": 0,
    "ingestion_failures_total": 0,
    "query_failures_total": 0,
    "quota_rejects_total": 0,
}

try:
    from langsmith import traceable

    _langsmith_available = True
except ImportError:
    pass


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is active."""
    return _TRACING_ENABLED and _langsmith_available


def increment_metric(name: str, amount: int = 1) -> None:
    """Increment an in-process operational counter."""
    if amount < 1:
        return
    _METRIC_COUNTERS[name] = _METRIC_COUNTERS.get(name, 0) + amount


def metrics_snapshot() -> dict[str, int]:
    """Return a copy of operational counters for tests or exporters."""
    return dict(_METRIC_COUNTERS)


def reset_metrics() -> None:
    """Reset operational counters. Intended for tests."""
    for name in _METRIC_COUNTERS:
        _METRIC_COUNTERS[name] = 0


def _check_langsmith_config() -> bool:
    """Verify that LangSmith env vars are properly configured."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning(
            "LANGSMITH_API_KEY not set -- LangSmith tracing will be disabled"
        )
        return False
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    logger.info("LangSmith tracing enabled (project=%s)", project)
    return True


# ---------------------------------------------------------------------------
# Span wrappers
# ---------------------------------------------------------------------------


def _wrap_llm_func(original_func):
    """Wrap the LLM completion function with a Lightrag:llm span."""
    if not _langsmith_available:
        return original_func

    @traceable(name="Lightrag:llm", run_type="llm")
    @functools.wraps(original_func)
    async def traced_llm(*args, **kwargs):
        return await original_func(*args, **kwargs)

    return traced_llm


def _wrap_embedding_func(original_func):
    """Wrap the embedding function with a Lightrag:embedding span."""
    if not _langsmith_available:
        return original_func

    @traceable(name="Lightrag:embedding", run_type="embedding")
    @functools.wraps(original_func)
    async def traced_embedding(*args, **kwargs):
        return await original_func(*args, **kwargs)

    return traced_embedding


def _wrap_rerank_func(original_func):
    """Wrap the reranking function with a Lightrag:rerank span."""
    if not _langsmith_available:
        return original_func

    @traceable(name="Lightrag:rerank", run_type="chain")
    @functools.wraps(original_func)
    async def traced_rerank(*args, **kwargs):
        return await original_func(*args, **kwargs)

    return traced_rerank


# ---------------------------------------------------------------------------
# Public API: apply tracing to a LightRAG instance
# ---------------------------------------------------------------------------


def apply_lang_with_tracing(rag) -> None:
    """Wrap LLM, embedding, and rerank functions on a LightRAG instance.

    Call this AFTER ``LightRAG()`` is constructed and ``await rag.initialize()``
    has completed, so that the internal ``wait_func`` wrappers are in place.

    The traced embedding function is also propagated to all internal storage
    instances that hold a reference to ``embedding_func`` so that sub-spans
    nest correctly.
    """
    global _TRACING_ENABLED

    if not _check_langsmith_config():
        _TRACING_ENABLED = False
        return

    _TRACING_ENABLED = True

    # --- Wrap LLM ---
    if hasattr(rag, "llm_model_func") and rag.llm_model_func is not None:
        rag.llm_model_func = _wrap_llm_func(rag.llm_model_func)
        logger.info("Traced LLM function: Lightrag:llm")

    # --- Wrap Embedding ---
    if hasattr(rag, "embedding_func") and rag.embedding_func is not None:
        traced_embed = _wrap_embedding_func(rag.embedding_func)
        rag.embedding_func = traced_embed
        _propagate_embedding(rag, traced_embed)
        logger.info("Traced embedding function: Lightrag:embedding")

    # --- Wrap Reranking (if present) ---
    for attr_name in ("rerank_func", "reranking_func", "_rerank_func"):
        if hasattr(rag, attr_name):
            original_rerank = getattr(rag, attr_name)
            if original_rerank is not None:
                setattr(rag, attr_name, _wrap_rerank_func(original_rerank))
                logger.info("Traced rerank function: Lightrag:rerank")
                break

    logger.info(
        "LangSmith tracing applied -- 3 span types active "
        "(Lightrag:llm, Lightrag:embedding, Lightrag:rerank)"
    )


def _propagate_embedding(rag, traced_embed) -> None:
    """Push the traced embedding_func into all storage instances."""
    storage_attrs = [
        "text_chunks",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
    ]
    for attr_name in storage_attrs:
        storage = getattr(rag, attr_name, None)
        if storage is not None and hasattr(storage, "embedding_func"):
            storage.embedding_func = traced_embed
            logger.debug("Propagated traced embedding to %s", attr_name)


# ---------------------------------------------------------------------------
# Distributed tracing: extract/inject trace context from HTTP headers
# ---------------------------------------------------------------------------


def extract_trace_parent(headers: dict[str, str]) -> dict[str, Any] | None:
    """Extract LangSmith / OpenTelemetry trace context from request headers.

    Supports:
    - ``langsmith-trace`` header (LangSmith native)
    - ``traceparent`` header (W3C Trace Context / OpenTelemetry)
    - ``x-trace-id`` header (custom fallback)
    """
    # LangSmith native header
    langsmith_trace = headers.get("langsmith-trace") or headers.get("x-langsmith-trace")
    if langsmith_trace:
        return {"langsmith_trace_id": langsmith_trace}

    # W3C traceparent
    traceparent = headers.get("traceparent")
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 3:
            return {
                "trace_id": parts[1],
                "parent_span_id": parts[2],
                "traceparent": traceparent,
            }

    # Custom fallback
    trace_id = headers.get("x-trace-id")
    if trace_id:
        return {"trace_id": trace_id}

    return None


def make_trace_headers(trace_context: dict[str, Any] | None) -> dict[str, str]:
    """Build outbound headers for propagating trace context downstream."""
    if not trace_context:
        return {}
    headers: dict[str, str] = {}
    if "traceparent" in trace_context:
        headers["traceparent"] = trace_context["traceparent"]
    if "langsmith_trace_id" in trace_context:
        headers["langsmith-trace"] = trace_context["langsmith_trace_id"]
    if "trace_id" in trace_context:
        headers["x-trace-id"] = trace_context["trace_id"]
    return headers
