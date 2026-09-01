"""Versioned regulatory audit contract projected from the Activity ledger.

Activity remains the operator-facing product ledger.  This module is the
strict, secret-safe boundary between those business events and the future
durable exporter tracked by issue #122.  The exporter receives only validated
``AuditEvent`` models; it never receives the free-form Activity ``meta`` map.

Contract v1 follows ECS 9.4.0 where ECS has a matching concept and keeps Twin
domain fields under ``twin.*``.  Additive fields are compatible within v1;
renaming, removing or changing the meaning/type of a field requires v2.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from .activity_events import _FOLDER_INVALID, _FOLDER_NOT_APPLICABLE
from .metrics import record_audit_event
from .webui_models import ActivityEvent

logger = logging.getLogger(__name__)

AUDIT_SCHEMA_VERSION = "1.0.0"
ECS_VERSION = "9.4.0"


class AuditAction(StrEnum):
    """Activity kinds actually accepted and emitted by the runtime."""

    RETRIEVAL = "retrieval"
    TAG_MUTATION = "tag-mutation"
    DOC_RETAGGED = "doc-retagged"
    DOC_APPROVED = "doc-approved"
    DOC_REJECTED = "doc-rejected"
    DOC_DELETED = "doc-deleted"
    DOC_FOLDER_ADDED = "doc-folder-added"
    DOC_FOLDER_REMOVED = "doc-folder-removed"
    CLASSIFICATION_REJECTED = "classification-rejected"
    SOURCE_UPLOADED = "source-uploaded"
    SOURCE_READY = "source-ready"
    SOURCE_FAILED = "source-failed"
    PIPELINE_WARNING = "pipeline-warning"
    GRAPH_ENTITY_EDITED = "graph-entity-edited"
    GRAPH_RELATION_EDITED = "graph-relation-edited"
    AUTH = "auth"
    SETTINGS = "settings"
    API_KEY_CREATED = "api-key-created"
    API_KEY_REVOKED = "api-key-revoked"
    VISION_SETTINGS_UPDATED = "vision-settings-updated"
    PROCEDURE_PARKED = "procedure-parked"
    PROCEDURE_FAILED = "procedure-failed"
    PROCEDURE_APPROVED = "procedure-approved"
    PROCEDURE_REJECTED = "procedure-rejected"
    PROCEDURE_RETRIED = "procedure-retried"
    PROCEDURE_REROUTED = "procedure-rerouted"
    PROCEDURE_STORE_RECOVERED = "procedure-store-recovered"
    LINKED_SOURCE_DECLARED = "linked-source-declared"
    LINKED_SOURCE_UPDATED = "linked-source-updated"
    LINKED_SOURCE_DISABLED = "linked-source-disabled"
    KB_EXPORTED = "kb-exported"
    KB_IMPORTED = "kb-imported"


AuditOutcome = Literal["failure", "success", "unknown"]
AuthMethod = Literal[
    "idp",
    "local_jwt",
    "open",
    "operator_api_key",
    "static_api_key",
    "system",
    "unknown",
]
SeverityName = Literal["info", "warning", "error", "critical"]
EventCategory = Literal[
    "api", "authentication", "configuration", "database", "file", "iam"
]
EventType = Literal[
    "access",
    "admin",
    "change",
    "creation",
    "deletion",
    "denied",
    "end",
    "error",
    "info",
    "start",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EcsFields(_StrictModel):
    version: Literal["9.4.0"]


class EventFields(_StrictModel):
    id: str = Field(min_length=1, max_length=256)
    kind: Literal["event"]
    category: list[EventCategory] = Field(min_length=1)
    type: list[EventType] = Field(min_length=1)
    action: AuditAction
    outcome: AuditOutcome
    severity: int = Field(ge=1, le=4)
    provider: Literal["twin-kms"]
    module: Literal["activity"]
    dataset: Literal["twin.audit"]


class ServiceFields(_StrictModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    environment: str = Field(min_length=1, max_length=64)


class TraceFields(_StrictModel):
    id: str | None = Field(max_length=128)


class HttpRequestFields(_StrictModel):
    id: str | None = Field(max_length=128)
    method: str | None = Field(pattern=r"^[A-Z]+$", max_length=16)


class HttpResponseFields(_StrictModel):
    status_code: int | None = Field(ge=100, le=599)


class HttpFields(_StrictModel):
    request: HttpRequestFields
    response: HttpResponseFields


class UserFields(_StrictModel):
    name: str = Field(min_length=1, max_length=256)
    roles: list[str] = Field(max_length=16)


class TwinAuditFields(_StrictModel):
    schema_version: Literal["1.0.0"]
    source: Literal["activity"]
    source_event_kind: AuditAction
    severity_name: SeverityName
    operation: str | None = Field(max_length=128)
    emitted_by: str | None = Field(max_length=64)
    route_group: str | None = Field(max_length=64)


class TwinAuthFields(_StrictModel):
    method: AuthMethod


class TwinResourceFields(_StrictModel):
    type: str = Field(min_length=1, max_length=128)
    id: str | None = Field(max_length=512)
    name: str | None = Field(max_length=512)


class TwinFolderFields(_StrictModel):
    id: str | None = Field(max_length=128)


class TwinWorkspaceFields(_StrictModel):
    id: str | None = Field(max_length=128)


class TwinDataFields(_StrictModel):
    classification: str | None = Field(max_length=128)
    retention_policy: str | None = Field(max_length=128)


class TwinFields(_StrictModel):
    audit: TwinAuditFields
    auth: TwinAuthFields
    resource: TwinResourceFields
    folder: TwinFolderFields
    workspace: TwinWorkspaceFields
    data: TwinDataFields


class AuditEvent(_StrictModel):
    """Wire model consumed by a future durable audit exporter."""

    timestamp: datetime = Field(alias="@timestamp")
    message: str = Field(min_length=1, max_length=1024)
    ecs: EcsFields
    event: EventFields
    service: ServiceFields
    trace: TraceFields
    http: HttpFields
    user: UserFields
    twin: TwinFields

    @field_validator("timestamp", mode="before")
    @classmethod
    def _require_canonical_wire_timestamp(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.endswith("Z"):
            raise ValueError("@timestamp string must use canonical UTC Z form")
        return value

    @field_validator("timestamp")
    @classmethod
    def _require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("@timestamp must carry the UTC offset")
        return value

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )


AuditEventSink = Callable[[AuditEvent], Awaitable[None] | None]
_audit_event_sink: AuditEventSink | None = None

_SEVERITY_NUMBER: dict[str, int] = {
    "info": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}
_AUTH_METHODS = {
    "idp",
    "local_jwt",
    "open",
    "operator_api_key",
    "static_api_key",
    "system",
    "unknown",
}
_FAILURE_ACTIONS = {
    AuditAction.CLASSIFICATION_REJECTED,
    AuditAction.SOURCE_FAILED,
    AuditAction.PROCEDURE_FAILED,
}
_NOT_APPLICABLE_FOLDERS = {
    "",
    "-",
    _FOLDER_INVALID,
    _FOLDER_NOT_APPLICABLE,
}
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_COMMON_API_KEY_RE = re.compile(r"\b(?:sk|twk|tcp)_[A-Za-z0-9_-]{8,}\b")
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|cookie|set-cookie|password|secret|token|"
    r"api[_-]?key|credential(?:_path)?|secret_path|document_body|request_body|"
    r"response_body|prompt|llm_response)"
    r"\s*([:=])\s*(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:access_token|api_key|token|secret|password)=)[^&#\s]+"
)


def _package_version() -> str:
    try:
        return importlib.metadata.version("twindb-lightrag-memgraph")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _redact_text(value: Any, *, limit: int) -> str:
    """Redact credential-shaped text and enforce a hard output bound."""
    text = str(value)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _JWT_RE.sub("[REDACTED]", text)
    text = _COMMON_API_KEY_RE.sub("[REDACTED]", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", text)
    text = _URL_SECRET_RE.sub(r"\1[REDACTED]", text)
    if len(text) > limit:
        removed = len(text) - limit
        while True:
            suffix = f"…[truncated:{removed}]"
            kept = max(0, limit - len(suffix))
            actual_removed = len(text) - kept
            if actual_removed == removed:
                break
            removed = actual_removed
        text = f"{text[:kept]}{suffix[:limit]}"
    return text or "-"


def _optional_text(value: Any, *, limit: int) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return _redact_text(text, limit=limit) if text else None


def _actor_name(value: str, auth_method: str) -> str:
    """Preserve a non-secret operator-key id only on its verified auth path."""
    if auth_method == "operator_api_key" and value.startswith("api_key:"):
        key_id = value.split(":", 1)[1]
        if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", key_id):
            return f"api_key:{key_id}"
    return _redact_text(value, limit=256)


def _event_id(value: str) -> str:
    text = value.strip()
    if not _EVENT_ID_RE.fullmatch(text) or _redact_text(text, limit=256) != text:
        raise ValueError("Activity event id is not a safe stable identifier")
    return text


def _activity_timestamp(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Activity timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("Activity timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _event_category(action: AuditAction) -> list[EventCategory]:
    if action is AuditAction.AUTH:
        return ["authentication"]
    if action in {
        AuditAction.API_KEY_CREATED,
        AuditAction.API_KEY_REVOKED,
    }:
        return ["iam", "configuration"]
    if action is AuditAction.RETRIEVAL:
        return ["api"]
    if action in {
        AuditAction.GRAPH_ENTITY_EDITED,
        AuditAction.GRAPH_RELATION_EDITED,
        AuditAction.KB_EXPORTED,
        AuditAction.KB_IMPORTED,
    }:
        return ["database"]
    if action.value.startswith(("doc-", "source-", "procedure-")) or action in {
        AuditAction.CLASSIFICATION_REJECTED,
        AuditAction.PIPELINE_WARNING,
    }:
        return ["file"]
    return ["configuration"]


def _auth_operation(meta: Mapping[str, Any]) -> str:
    return str(meta.get("operation") or "").strip().lower().replace("-", "_")


def _event_type(action: AuditAction, meta: Mapping[str, Any]) -> list[EventType]:
    if action is AuditAction.AUTH:
        operation = _auth_operation(meta)
        if operation in {"access_denied", "login_failed", "login_failure"}:
            return ["denied"]
        if operation == "logout":
            return ["end"]
        if operation in {"login", "login_success"}:
            return ["start"]
        return ["info"]
    if action in {AuditAction.DOC_DELETED, AuditAction.API_KEY_REVOKED}:
        return ["deletion"]
    if action in {AuditAction.SOURCE_UPLOADED, AuditAction.API_KEY_CREATED}:
        return ["creation"]
    if action in {AuditAction.SOURCE_READY, AuditAction.KB_EXPORTED}:
        return ["end"]
    if action in _FAILURE_ACTIONS:
        return ["error"]
    if action is AuditAction.RETRIEVAL:
        return ["access"]
    if action is AuditAction.PIPELINE_WARNING:
        return ["info"]
    return ["change"]


def _event_outcome(action: AuditAction, meta: Mapping[str, Any]) -> AuditOutcome:
    if action is AuditAction.AUTH:
        operation = _auth_operation(meta)
        if operation in {"access_denied", "login_failed", "login_failure"}:
            return "failure"
        success = meta.get("success")
        if isinstance(success, bool):
            return "success" if success else "failure"
    if action in _FAILURE_ACTIONS:
        return "failure"
    if action is AuditAction.PIPELINE_WARNING:
        return "unknown"
    return "success"


def _safe_status(meta: Mapping[str, Any]) -> int | None:
    value = meta.get("status_code")
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    return None


def _safe_method(meta: Mapping[str, Any], context: Mapping[str, str]) -> str | None:
    value = context.get("http_method") or meta.get("method")
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    return upper if upper.isascii() and upper.isalpha() and len(upper) <= 16 else None


def _meta_value(meta: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _optional_text(meta.get(key), limit=128)
        if value is not None:
            return value
    return None


def _runtime_context() -> dict[str, str]:
    from .observability import current_request_context

    return current_request_context()


def _runtime_workspace() -> str | None:
    try:
        from .._constants import resolve_workspace

        return resolve_workspace()
    except Exception:  # noqa: BLE001 - audit projection stays best effort.
        return None


def _runtime_folder() -> str | None:
    try:
        from .folder import current_folder_id

        return current_folder_id()
    except Exception:  # noqa: BLE001 - audit projection stays best effort.
        return None


def activity_to_audit_event(
    activity: Mapping[str, Any],
    *,
    request_context: Mapping[str, str] | None = None,
    folder_id: str | None = None,
    workspace_id: str | None = None,
    service_name: str | None = None,
    service_version: str | None = None,
    service_environment: str | None = None,
) -> AuditEvent:
    """Validate and project one Activity event into the v1 audit contract.

    Projection is deliberately allow-list based.  Free-form ``meta`` values,
    document bodies, prompts, model responses, cookies and tokens have no path
    into the returned model.
    """
    source = ActivityEvent.model_validate(activity)
    action = AuditAction(source.kind)
    meta = source.meta if isinstance(source.meta, Mapping) else {}
    context = dict(request_context or _runtime_context())

    explicit_folder = _meta_value(meta, "folder_id", "folder")
    resolved_folder = explicit_folder or folder_id or _runtime_folder()
    if resolved_folder in _NOT_APPLICABLE_FOLDERS:
        resolved_folder = None
    resolved_workspace = (
        _meta_value(meta, "workspace_id", "workspace")
        or workspace_id
        or _runtime_workspace()
    )

    # Credential attribution is a verified runtime capability.  Activity meta
    # is business data and must never be allowed to claim or override it.
    method = str(context.get("auth_method") or "unknown")
    if method not in _AUTH_METHODS:
        method = "unknown"
    if source.actor.user in {"system", "scheduler"} and method in {
        "open",
        "unknown",
    }:
        method = "system"

    request_id = _optional_text(context.get("request_id"), limit=128)
    trace_id = _optional_text(context.get("trace_id"), limit=128)
    if request_id == "-":
        request_id = None
    if trace_id == "-":
        trace_id = None

    target_id = _optional_text(source.target.id, limit=512)
    target_name = _optional_text(source.target.label, limit=512)
    if action in {AuditAction.RETRIEVAL, AuditAction.AUTH}:
        # Retrieval labels can be the query itself; auth labels can be a raw
        # route.  Neither is needed once action/resource type are present.
        target_name = None

    operation = _meta_value(meta, "operation")
    emitted_by = _meta_value(meta, "emitted_by")
    route_group = _optional_text(context.get("route_group"), limit=64)
    if route_group in {None, "other"}:
        route_group = None

    context_actor = _optional_text(context.get("auth_actor"), limit=256)
    if context_actor == "-":
        context_actor = None

    return AuditEvent(
        **{
            "@timestamp": _activity_timestamp(source.ts),
            "message": _redact_text(source.summary, limit=1024),
            "ecs": EcsFields(version=ECS_VERSION),
            "event": EventFields(
                id=_event_id(source.id),
                kind="event",
                category=_event_category(action),
                type=_event_type(action, meta),
                action=action,
                outcome=_event_outcome(action, meta),
                severity=_SEVERITY_NUMBER[source.sev],
                provider="twin-kms",
                module="activity",
                dataset="twin.audit",
            ),
            "service": ServiceFields(
                name=service_name or os.environ.get("TWIN_SERVICE_NAME", "twin-kms"),
                version=(
                    service_version
                    or os.environ.get("TWIN_SERVICE_VERSION")
                    or _package_version()
                ),
                environment=(
                    service_environment or os.environ.get("TWIN_ENV", "development")
                ),
            ),
            "trace": TraceFields(id=trace_id),
            "http": HttpFields(
                request=HttpRequestFields(
                    id=request_id,
                    method=_safe_method(meta, context),
                ),
                response=HttpResponseFields(status_code=_safe_status(meta)),
            ),
            "user": UserFields(
                name=_actor_name(context_actor or source.actor.user, method),
                roles=[_redact_text(source.actor.role, limit=128)],
            ),
            "twin": TwinFields(
                audit=TwinAuditFields(
                    schema_version=AUDIT_SCHEMA_VERSION,
                    source="activity",
                    source_event_kind=action,
                    severity_name=source.sev,
                    operation=operation,
                    emitted_by=emitted_by,
                    route_group=route_group,
                ),
                auth=TwinAuthFields(method=method),
                resource=TwinResourceFields(
                    type=_redact_text(source.target.type, limit=128),
                    id=target_id,
                    name=target_name,
                ),
                folder=TwinFolderFields(id=_optional_text(resolved_folder, limit=128)),
                workspace=TwinWorkspaceFields(
                    id=_optional_text(resolved_workspace, limit=128)
                ),
                data=TwinDataFields(
                    classification=_meta_value(
                        meta,
                        "classification",
                        "classification_label",
                        "sensitivity",
                    ),
                    retention_policy=_meta_value(meta, "retention_policy", "retention"),
                ),
            ),
        }
    )


def set_audit_event_sink(sink: AuditEventSink | None) -> None:
    """Install or clear the future #122 enqueue boundary."""
    global _audit_event_sink
    _audit_event_sink = sink


def _source_fingerprint(activity: Mapping[str, Any]) -> str:
    raw = f"{activity.get('id', '-')!s}:{activity.get('kind', '-')!s}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


async def submit_activity_audit_event(
    activity: Mapping[str, Any],
) -> AuditEvent | None:
    """Validate one Activity event and hand it to the installed sink.

    Invalid events and sink failures never break the product ledger.  Both are
    counted and logged using only a one-way source fingerprint and exception
    class; payloads and exception messages are intentionally excluded.
    """
    fingerprint = _source_fingerprint(activity)
    try:
        event = activity_to_audit_event(activity)
    except Exception as exc:  # noqa: BLE001 - audit never breaks the product ledger.
        record_audit_event("invalid")
        logger.warning(
            "audit_event_invalid source_fingerprint=%s error_type=%s",
            fingerprint,
            type(exc).__name__,
        )
        return None

    sink = _audit_event_sink
    if sink is None:
        return event
    try:
        result = sink(event)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - audit cannot break business writes.
        record_audit_event("dropped")
        logger.error(
            "audit_event_dropped source_fingerprint=%s error_type=%s",
            fingerprint,
            type(exc).__name__,
        )
        return None
    return event


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "ECS_VERSION",
    "AuditAction",
    "AuditEvent",
    "activity_to_audit_event",
    "set_audit_event_sink",
    "submit_activity_audit_event",
]
