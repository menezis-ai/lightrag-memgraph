"""Request context and structured technical logging for Twin runtimes.

This module deliberately covers technical logs only.  The versioned business
audit contract lives in :mod:`.audit`; its durable export remains issue #122.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import socket
import time
import traceback
import uuid
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .metrics import record_http_request
from .tracing import bind_trace_context, resolve_trace_context

logger = logging.getLogger(__name__)

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_request_id", default="-"
)
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_trace_id", default="-"
)
span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_span_id", default="-"
)
route_group_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_route_group", default="other"
)
http_method_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_http_method", default="-"
)
auth_method_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_auth_method", default="unknown"
)
auth_actor_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twin_auth_actor", default="-"
)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?P<key_quote>[\"']?)"
    r"(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})(?P=key_quote)"
    r"(?![A-Za-z0-9_])"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;}]+)"
)
_URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:access_token|api_key|token|secret|password)=)[^&#\s]+"
)
_SENSITIVE_NAME_SEGMENTS = frozenset(
    {
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
        "apikey",
        "credential",
        "documentbody",
        "requestbody",
        "prompt",
        "llmresponse",
    }
)
_SENSITIVE_NAME_PAIRS = frozenset(
    {
        ("api", "key"),
        ("document", "body"),
        ("request", "body"),
        ("llm", "response"),
    }
)
_CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_JSON_REDACTION_LIMIT = 65_536
_ORIGINAL_FORMATTERS: weakref.WeakKeyDictionary[
    logging.Handler, logging.Formatter | None
] = weakref.WeakKeyDictionary()
_FACTORY_INSTALLED = False


def _normalized_name_segments(value: Any) -> tuple[str, ...]:
    camel_split = _CAMEL_CASE_BOUNDARY_RE.sub("_", str(value))
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.casefold()).strip("_")
    return tuple(segment for segment in normalized.split("_") if segment)


def _is_sensitive_name(value: Any) -> bool:
    segments = _normalized_name_segments(value)
    if any(segment in _SENSITIVE_NAME_SEGMENTS for segment in segments):
        return True
    return any(pair in _SENSITIVE_NAME_PAIRS for pair in zip(segments, segments[1:]))


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if _is_sensitive_name(key) else _redact_json_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    return value


def _redact_json_text(text: str) -> str | None:
    if len(text) > _JSON_REDACTION_LIMIT:
        return None
    try:
        value = json.loads(text)
        if not isinstance(value, (dict, list)):
            return None
        redacted = _redact_json_value(value)
    except (json.JSONDecodeError, RecursionError):
        return None
    return json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))


def _redact_assignment(match: re.Match[str]) -> str:
    if not _is_sensitive_name(match.group("key")):
        return match.group(0)
    value = match.group("value")
    if len(value) >= 2 and value[0] in {'"', "'"} and value[-1] == value[0]:
        replacement = f"{value[0]}[REDACTED]{value[0]}"
    else:
        replacement = "[REDACTED]"
    quote = match.group("key_quote")
    return (
        f"{quote}{match.group('key')}{quote}" f"{match.group('separator')}{replacement}"
    )


def _redact(value: Any, *, limit: int = 4096) -> str:
    text = str(value)
    text = _redact_json_text(text) or text
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(_redact_assignment, text)
    text = _URL_SECRET_RE.sub(r"\1[REDACTED]", text)
    if len(text) > limit:
        return f"{text[:limit]}…[truncated:{len(text) - limit}]"
    return text


def _package_version() -> str:
    try:
        return importlib.metadata.version("twindb-lightrag-memgraph")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@contextmanager
def bind_request_context(
    *,
    request_id: str,
    trace_id: str,
    span_id: str = "-",
    route_group: str,
    http_method: str = "-",
    auth_method: str = "unknown",
    auth_actor: str = "-",
) -> Iterator[None]:
    """Bind and reliably reset one request's correlation context."""
    request_token = request_id_var.set(request_id)
    trace_token = trace_id_var.set(trace_id)
    span_token = span_id_var.set(span_id)
    route_token = route_group_var.set(route_group)
    http_method_token = http_method_var.set(http_method)
    auth_method_token = auth_method_var.set(auth_method)
    auth_actor_token = auth_actor_var.set(auth_actor)
    try:
        yield
    finally:
        auth_actor_var.reset(auth_actor_token)
        auth_method_var.reset(auth_method_token)
        http_method_var.reset(http_method_token)
        route_group_var.reset(route_token)
        span_id_var.reset(span_token)
        trace_id_var.reset(trace_token)
        request_id_var.reset(request_token)


def current_request_context() -> dict[str, str]:
    return {
        "request_id": request_id_var.get(),
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "route_group": route_group_var.get(),
        "http_method": http_method_var.get(),
        "auth_method": auth_method_var.get(),
        "auth_actor": auth_actor_var.get(),
    }


def set_request_auth_method(method: str, actor: str | None = None) -> None:
    """Bind the verified credential path and actor to this request."""
    auth_method_var.set(method)
    if actor:
        auth_actor_var.set(actor)


def route_group(path: str) -> str:
    """Map request paths to a fixed, low-cardinality route vocabulary."""
    if path.endswith("/ops/metrics"):
        return "metrics"
    if path in {"/health", "/ready"} or path.endswith("/health"):
        return "health"
    if path in {"/query", "/query/data", "/query/stream"}:
        return "query"
    if path.startswith("/twin/api/query"):
        return "query"
    if path in {"/insert", "/documents/upload", "/documents/reprocess_failed"}:
        return "ingestion"
    if path.endswith("/scan") and path.startswith("/documents/"):
        return "ingestion"
    if path.startswith("/twin/api/documents"):
        return "documents"
    if path.startswith("/twin/api/graph"):
        return "graph"
    if path.startswith("/twin/api/settings/api-keys"):
        return "admin"
    if path.startswith("/twin/api"):
        return "twin"
    return "other"


def _request_id(headers: Any) -> str:
    candidate = (headers.get("x-request-id") or "").strip()
    if _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _install_record_factory() -> None:
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return
    previous = logging.getLogRecordFactory()

    def twin_record_factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.twin_request_id = request_id_var.get()
        record.twin_trace_id = trace_id_var.get()
        record.twin_span_id = span_id_var.get()
        record.twin_route_group = route_group_var.get()
        return record

    twin_record_factory._twin_context_factory = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(twin_record_factory)
    _FACTORY_INSTALLED = True


class TwinJsonFormatter(logging.Formatter):
    """One-line ECS-compatible JSON formatter with bounded exceptions."""

    def __init__(
        self,
        *,
        service_name: str = "twin-kms",
        service_version: str | None = None,
        environment: str = "development",
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.service_version = service_version or _package_version()
        self.environment = environment
        self.hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:
            message = "[unformattable log message]"
        payload: dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "log.level": record.levelname.lower(),
            "log.logger": record.name,
            "service.name": self.service_name,
            "service.version": self.service_version,
            "service.environment": self.environment,
            "host.name": self.hostname,
            "process.pid": record.process,
            "message": _redact(message),
            "twin.request.id": getattr(record, "twin_request_id", request_id_var.get()),
            "trace.id": getattr(record, "twin_trace_id", trace_id_var.get()),
            "span.id": getattr(record, "twin_span_id", span_id_var.get()),
            "twin.route.group": getattr(
                record, "twin_route_group", route_group_var.get()
            ),
        }

        extra_fields = {
            "event_action": "event.action",
            "http_method": "http.request.method",
            "http_path": "url.path",
            "http_status_code": "http.response.status_code",
            "auth_mode": "twin.auth.mode",
        }
        for source, target in extra_fields.items():
            value = getattr(record, source, None)
            if value is not None:
                payload[target] = (
                    value if isinstance(value, int) else _redact(value, limit=512)
                )

        duration_seconds = getattr(record, "duration_seconds", None)
        if isinstance(duration_seconds, (float, int)):
            payload["event.duration"] = max(0, int(duration_seconds * 1_000_000_000))

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            raw_stack = "".join(traceback.format_exception(*record.exc_info))
            payload.update(
                {
                    "error.type": getattr(exc_type, "__name__", "Exception"),
                    "error.message": _redact(exc_value, limit=512),
                    "error.stack_hash": hashlib.sha256(
                        raw_stack.encode("utf-8", errors="replace")
                    ).hexdigest(),
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_runtime_logging(format_name: str | None = None) -> bool:
    """Activate JSON logging on existing runtime handlers.

    ``text`` is deliberately the default and restores formatters changed by an
    earlier test/app construction.  Third-party records need provide no Twin
    fields; the record factory supplies only the common correlation envelope.
    """
    selected = (format_name or os.environ.get("TWIN_LOG_FORMAT", "text")).lower()
    if selected not in {"json", "text"}:
        raise ValueError("TWIN_LOG_FORMAT must be 'text' or 'json'")
    if selected == "text":
        for handler, formatter in list(_ORIGINAL_FORMATTERS.items()):
            handler.setFormatter(formatter)
        _ORIGINAL_FORMATTERS.clear()
        return False

    _install_record_factory()
    formatter = TwinJsonFormatter(
        service_name=os.environ.get("TWIN_SERVICE_NAME", "twin-kms"),
        service_version=os.environ.get("TWIN_SERVICE_VERSION") or None,
        environment=os.environ.get("TWIN_ENV", "development"),
    )
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    loggers = [
        root,
        logging.getLogger("twindb_lightrag_memgraph"),
        logging.getLogger("lightrag"),
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("uvicorn.error"),
    ]
    seen: set[int] = set()
    for configured_logger in loggers:
        for handler in configured_logger.handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            if handler not in _ORIGINAL_FORMATTERS:
                _ORIGINAL_FORMATTERS[handler] = handler.formatter
            handler.setFormatter(formatter)
    return True


def make_request_observability_middleware():
    """Build middleware for the production native+Twin overlay surface."""

    async def request_observability(request, call_next):
        started = time.perf_counter()
        request_id = _request_id(request.headers)
        trace_context = resolve_trace_context(request.headers)
        group = route_group(request.url.path)
        request.state.request_id = request_id
        request.state.trace_id = trace_context.trace_id
        request.state.span_id = trace_context.span_id
        request.state.traceparent = trace_context.traceparent
        request.state.route_group = group

        with (
            bind_trace_context(trace_context),
            bind_request_context(
                request_id=request_id,
                trace_id=trace_context.trace_id,
                span_id=trace_context.span_id,
                route_group=group,
                http_method=request.method,
            ),
        ):
            try:
                response = await call_next(request)
            except Exception:
                duration = time.perf_counter() - started
                record_http_request(
                    route_group=group,
                    method=request.method,
                    status_code=500,
                    duration_seconds=duration,
                )
                logger.exception(
                    "http_request_failed",
                    extra={
                        "event_action": "http_request_failed",
                        "http_method": request.method,
                        "http_path": request.url.path,
                        "http_status_code": 500,
                        "duration_seconds": duration,
                    },
                )
                raise

            duration = time.perf_counter() - started
            response.headers["x-request-id"] = request_id
            response.headers["traceparent"] = trace_context.traceparent
            record_http_request(
                route_group=group,
                method=request.method,
                status_code=response.status_code,
                duration_seconds=duration,
            )
            logger.log(
                logging.WARNING if response.status_code >= 500 else logging.INFO,
                "http_request",
                extra={
                    "event_action": "http_request",
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status_code": response.status_code,
                    "duration_seconds": duration,
                },
            )
            return response

    return request_observability


def install_request_observability(app: Any) -> None:
    """Idempotently configure logs and middleware on a native host app."""
    configure_runtime_logging()
    if getattr(app.state, "twin_observability_installed", False):
        return
    app.middleware("http")(make_request_observability_middleware())
    app.state.twin_observability_installed = True


__all__ = [
    "TwinJsonFormatter",
    "auth_actor_var",
    "auth_method_var",
    "bind_request_context",
    "configure_runtime_logging",
    "current_request_context",
    "http_method_var",
    "install_request_observability",
    "make_request_observability_middleware",
    "request_id_var",
    "route_group",
    "route_group_var",
    "set_request_auth_method",
    "span_id_var",
    "trace_id_var",
]
