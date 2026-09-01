"""W3C request propagation and LangSmith integration.

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
TwinRAG:llm        llm         L3 intent / reason / rerank / synthesis
=================  ==========  ====================================

Incoming context precedence is deliberately one contract for both server
topologies:

1. a valid W3C ``traceparent`` supplies the canonical trace id and flags;
2. otherwise a valid native ``langsmith-trace`` supplies its trace UUID;
3. otherwise a bounded ``x-trace-id`` supplies a hexadecimal id directly, or
   a stable SHA-256 projection for legacy non-hexadecimal ids;
4. otherwise a new W3C context is generated.

The native LangSmith header remains orthogonal when W3C is present: it is used
as the exact parent for LangSmith spans, while W3C remains authoritative for
HTTP propagation and technical-log correlation.

Configuration: ``LIGHTRAG_ENABLE_LANGSMITH_TRACING=true`` plus
``LANGSMITH_API_KEY`` (and optionally ``LANGSMITH_PROJECT``).
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import logging
import os
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Awaitable, Callable
from typing import Any, Iterator, Mapping, TypeVar

logger = logging.getLogger(__name__)

_TRACING_ENABLED = False
_langsmith_available = False
try:
    from langsmith import traceable

    _langsmith_available = True
except ImportError:
    pass


# Keep the historical metric re-exports without making Prometheus a runtime
# dependency of the dependency-light tracing context.  In particular, the
# intelligence extra imports this module for L3 correlation on every provider
# attempt; it must not load ``server.metrics`` unless a caller actually uses a
# metric helper.
def increment_metric(name: str, amount: int = 1) -> None:
    from .metrics import increment_metric as _increment_metric

    _increment_metric(name, amount)


def metrics_snapshot() -> dict[str, int]:
    from .metrics import metrics_snapshot as _metrics_snapshot

    return _metrics_snapshot()


def reset_metrics() -> None:
    from .metrics import reset_metrics as _reset_metrics

    _reset_metrics()


_W3C_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_W3C_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_W3C_FLAGS_RE = re.compile(r"^[0-9a-f]{2}$")
_LEGACY_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LANGSMITH_PART_RE = re.compile(
    r"^(?P<timestamp>[0-9]{8}T[0-9]{12}Z)"
    r"(?P<run_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_MAX_LANGSMITH_HEADER_CHARS = 4096
_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16
_T = TypeVar("_T")


@dataclass(frozen=True)
class TraceContext:
    """One request's canonical W3C context and optional LangSmith parent."""

    trace_id: str
    span_id: str
    trace_flags: str
    source: str
    parent_span_id: str | None = None
    incoming_traceparent: str | None = None
    langsmith_trace: str | None = None
    langsmith_baggage: str | None = None
    legacy_trace_id: str | None = None

    @property
    def traceparent(self) -> str:
        """Context returned to the caller and injected into child requests."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"

    @property
    def langsmith_parent_headers(self) -> dict[str, str] | None:
        if self.langsmith_trace is None:
            return None
        headers = {"langsmith-trace": self.langsmith_trace}
        if self.langsmith_baggage is not None:
            headers["baggage"] = self.langsmith_baggage
        return headers


trace_context_var: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "twin_trace_context", default=None
)


def _header(headers: Mapping[str, Any], name: str) -> str:
    value = headers.get(name)
    if value is None:
        # Plain dicts in unit tests are not case-insensitive like Starlette's
        # Headers. Keep the public helper correct for either mapping shape.
        lowered = name.casefold()
        value = next(
            (item for key, item in headers.items() if str(key).casefold() == lowered),
            None,
        )
    return str(value).strip() if value is not None else ""


def _parse_w3c_traceparent(value: str) -> tuple[str, str, str] | None:
    """Parse the W3C Trace Context core format without accepting ambiguity."""
    if not value or value != value.lower():
        return None
    parts = value.split("-")
    if len(parts) < 4:
        return None
    version, trace_id, parent_id, trace_flags = parts[:4]
    if not re.fullmatch(r"[0-9a-f]{2}", version) or version == "ff":
        return None
    if version == "00" and len(parts) != 4:
        return None
    if version != "00" and len(parts) > 4 and any(not part for part in parts[4:]):
        return None
    if not _W3C_TRACE_ID_RE.fullmatch(trace_id) or trace_id == _ZERO_TRACE_ID:
        return None
    if not _W3C_SPAN_ID_RE.fullmatch(parent_id) or parent_id == _ZERO_SPAN_ID:
        return None
    if not _W3C_FLAGS_RE.fullmatch(trace_flags):
        return None
    return trace_id, parent_id, trace_flags


def _parse_langsmith_trace(value: str) -> str | None:
    """Return the root LangSmith UUID as 32 lowercase hex characters."""
    if not value or len(value) > _MAX_LANGSMITH_HEADER_CHARS:
        return None
    root_trace_id: str | None = None
    for index, part in enumerate(value.split(".")):
        match = _LANGSMITH_PART_RE.fullmatch(part)
        if match is None:
            return None
        try:
            datetime.strptime(match.group("timestamp"), "%Y%m%dT%H%M%S%fZ")
        except ValueError:
            return None
        run_id = match.group("run_id").replace("-", "").lower()
        if index == 0:
            root_trace_id = run_id
    return root_trace_id


def _new_nonzero_hex(bytes_count: int) -> str:
    while True:
        value = secrets.token_hex(bytes_count)
        if int(value, 16) != 0:
            return value


def resolve_trace_context(headers: Mapping[str, Any]) -> TraceContext:
    """Resolve headers into a fresh server span under one canonical trace."""
    incoming_traceparent = _header(headers, "traceparent")
    w3c = _parse_w3c_traceparent(incoming_traceparent)

    langsmith_trace = _header(headers, "langsmith-trace") or _header(
        headers, "x-langsmith-trace"
    )
    langsmith_trace_id = _parse_langsmith_trace(langsmith_trace)
    if langsmith_trace_id is None:
        langsmith_trace = ""
    baggage = _header(headers, "baggage")
    if len(baggage) > _MAX_LANGSMITH_HEADER_CHARS:
        baggage = ""

    legacy_trace_id = _header(headers, "x-trace-id")
    if not _LEGACY_TRACE_ID_RE.fullmatch(legacy_trace_id):
        legacy_trace_id = ""

    if w3c is not None:
        trace_id, parent_span_id, trace_flags = w3c
        source = "w3c"
    elif langsmith_trace_id is not None:
        trace_id = langsmith_trace_id
        parent_span_id = None
        trace_flags = "01"
        source = "langsmith"
    elif legacy_trace_id:
        candidate = legacy_trace_id.lower()
        if _W3C_TRACE_ID_RE.fullmatch(candidate) and candidate != _ZERO_TRACE_ID:
            trace_id = candidate
        else:
            trace_id = hashlib.sha256(legacy_trace_id.encode("utf-8")).hexdigest()[:32]
        parent_span_id = None
        trace_flags = "01"
        source = "custom"
    else:
        trace_id = _new_nonzero_hex(16)
        parent_span_id = None
        trace_flags = "01"
        source = "generated"

    return TraceContext(
        trace_id=trace_id,
        span_id=_new_nonzero_hex(8),
        trace_flags=trace_flags,
        source=source,
        parent_span_id=parent_span_id,
        incoming_traceparent=incoming_traceparent if w3c is not None else None,
        langsmith_trace=langsmith_trace or None,
        langsmith_baggage=baggage or None,
        legacy_trace_id=legacy_trace_id or None,
    )


@contextmanager
def bind_trace_context(context: TraceContext) -> Iterator[None]:
    """Bind a request trace and always restore the previous task context."""
    token = trace_context_var.set(context)
    try:
        yield
    finally:
        trace_context_var.reset(token)


def current_trace_context() -> TraceContext | None:
    return trace_context_var.get()


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is active."""
    return _TRACING_ENABLED and _langsmith_available


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


def _langsmith_extra() -> dict[str, Any] | None:
    context = current_trace_context()
    if context is None:
        return None
    extra: dict[str, Any] = {
        "metadata": {
            "w3c.trace_id": context.trace_id,
            "w3c.span_id": context.span_id,
        }
    }
    parent = context.langsmith_parent_headers
    if parent is not None:
        extra["parent"] = parent
    return extra


def _wrap_traceable_func(original_func, *, name: str, run_type: str):
    if not _langsmith_available:
        return original_func

    @traceable(name=name, run_type=run_type)
    @functools.wraps(original_func)
    async def traced_call(*args, **kwargs):
        return await original_func(*args, **kwargs)

    @functools.wraps(original_func)
    async def with_distributed_parent(*args, **kwargs):
        langsmith_extra = kwargs.pop("langsmith_extra", None) or _langsmith_extra()
        if langsmith_extra is None:
            return await traced_call(*args, **kwargs)
        return await traced_call(
            *args,
            **kwargs,
            langsmith_extra=langsmith_extra,
        )

    with_distributed_parent._twin_langsmith_wrapped = True
    return with_distributed_parent


def _wrap_llm_func(original_func):
    """Wrap the LLM completion function with a Lightrag:llm span."""
    return _wrap_traceable_func(original_func, name="Lightrag:llm", run_type="llm")


def _wrap_embedding_func(original_func):
    """Wrap the embedding function with a Lightrag:embedding span."""
    return _wrap_traceable_func(
        original_func,
        name="Lightrag:embedding",
        run_type="embedding",
    )


def _wrap_rerank_func(original_func):
    """Wrap the reranking function with a Lightrag:rerank span."""
    return _wrap_traceable_func(
        original_func,
        name="Lightrag:rerank",
        run_type="chain",
    )


async def trace_l3_llm_call(operation: Callable[[], Awaitable[_T]]) -> _T:
    """Run one L3 provider attempt in a request-parented LangSmith span.

    The operation deliberately accepts no prompt arguments: LangSmith receives
    correlation metadata and timing without duplicating sensitive prompt data.
    Wrapping happens per attempt so retries are counted as real provider calls.
    """
    if not is_tracing_enabled():
        return await operation()
    traced_operation = _wrap_traceable_func(
        operation,
        name="TwinRAG:llm",
        run_type="llm",
    )
    return await traced_operation()


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

    if not _langsmith_available:
        _TRACING_ENABLED = False
        logger.warning("LangSmith package not installed -- tracing will be disabled")
        return
    if not _check_langsmith_config():
        _TRACING_ENABLED = False
        return

    _TRACING_ENABLED = True

    # --- Wrap LLM ---
    if (
        hasattr(rag, "llm_model_func")
        and rag.llm_model_func is not None
        and getattr(rag.llm_model_func, "_twin_langsmith_wrapped", False) is not True
    ):
        rag.llm_model_func = _wrap_llm_func(rag.llm_model_func)
        logger.info("Traced LLM function: Lightrag:llm")

    # --- Wrap Embedding ---
    if (
        hasattr(rag, "embedding_func")
        and rag.embedding_func is not None
        and getattr(rag.embedding_func, "_twin_langsmith_wrapped", False) is not True
    ):
        traced_embed = _wrap_embedding_func(rag.embedding_func)
        rag.embedding_func = traced_embed
        _propagate_embedding(rag, traced_embed)
        logger.info("Traced embedding function: Lightrag:embedding")

    # --- Wrap Reranking (if present) ---
    for attr_name in ("rerank_func", "reranking_func", "_rerank_func"):
        if hasattr(rag, attr_name):
            original_rerank = getattr(rag, attr_name)
            if original_rerank is None:
                continue
            if getattr(original_rerank, "_twin_langsmith_wrapped", False) is not True:
                setattr(rag, attr_name, _wrap_rerank_func(original_rerank))
                logger.info("Traced rerank function: Lightrag:rerank")
            break

    logger.info(
        "LangSmith tracing applied -- 4 span types active "
        "(Lightrag:llm, Lightrag:embedding, Lightrag:rerank, TwinRAG:llm)"
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


def extract_trace_parent(headers: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compatibility projection of a valid incoming parent.

    New middleware should use :func:`resolve_trace_context`, which also creates
    the server span required when the parent is absent or malformed.
    """
    incoming_traceparent = _header(headers, "traceparent")
    langsmith_trace = _header(headers, "langsmith-trace") or _header(
        headers, "x-langsmith-trace"
    )
    legacy_trace_id = _header(headers, "x-trace-id")
    if (
        _parse_w3c_traceparent(incoming_traceparent) is None
        and _parse_langsmith_trace(langsmith_trace) is None
        and not _LEGACY_TRACE_ID_RE.fullmatch(legacy_trace_id)
    ):
        return None

    context = resolve_trace_context(headers)
    result: dict[str, Any] = {
        "trace_id": context.trace_id,
        "span_id": context.span_id,
        "traceparent": context.traceparent,
        "source": context.source,
    }
    if context.parent_span_id is not None:
        result["parent_span_id"] = context.parent_span_id
    if context.langsmith_trace is not None:
        result["langsmith_trace_id"] = context.langsmith_trace
    if context.legacy_trace_id is not None:
        result["legacy_trace_id"] = context.legacy_trace_id
    return result


def make_trace_headers(
    trace_context: TraceContext | Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build outbound W3C, LangSmith, and legacy correlation headers."""
    context = trace_context if trace_context is not None else current_trace_context()
    if context is None:
        return {}
    if isinstance(context, TraceContext):
        headers = {
            "traceparent": context.traceparent,
            "x-trace-id": context.trace_id,
        }
        if context.langsmith_trace is not None:
            headers["langsmith-trace"] = context.langsmith_trace
        if context.langsmith_baggage is not None:
            headers["baggage"] = context.langsmith_baggage
        return headers

    headers: dict[str, str] = {}
    if "traceparent" in context:
        headers["traceparent"] = str(context["traceparent"])
    if "langsmith_trace_id" in context:
        headers["langsmith-trace"] = str(context["langsmith_trace_id"])
    if "trace_id" in context:
        headers["x-trace-id"] = str(context["trace_id"])
    return headers


__all__ = [
    "TraceContext",
    "apply_lang_with_tracing",
    "bind_trace_context",
    "current_trace_context",
    "extract_trace_parent",
    "increment_metric",
    "is_tracing_enabled",
    "make_trace_headers",
    "metrics_snapshot",
    "reset_metrics",
    "resolve_trace_context",
    "trace_l3_llm_call",
    "trace_context_var",
]
