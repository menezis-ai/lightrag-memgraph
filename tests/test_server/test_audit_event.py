"""Contract tests for issue #89's versioned regulatory AuditEvent."""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import resources
from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from twindb_lightrag_memgraph.server import activity_events, audit
from twindb_lightrag_memgraph.server.audit import (
    AuditAction,
    AuditEvent,
    activity_to_audit_event,
    submit_activity_audit_event,
)
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.metrics import metrics_snapshot, reset_metrics
from twindb_lightrag_memgraph.server.observability import bind_request_context
from twindb_lightrag_memgraph.server.webui.store import WebuiStore, reset_store
from twindb_lightrag_memgraph.server.webui_models import ActivityEvent
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings

FIXTURES = Path(__file__).parents[1] / "fixtures" / "audit" / "v1"


def _activity(
    *,
    event_id: str,
    timestamp: str,
    kind: str,
    severity: str,
    actor: str,
    target_type: str,
    target_label: str,
    summary: str,
    meta: dict,
    target_id: str | None = None,
) -> dict:
    target = {"type": target_type, "label": target_label}
    if target_id is not None:
        target["id"] = target_id
    return {
        "id": event_id,
        "ts": timestamp,
        "rel": "now",
        "day": "Today",
        "kind": kind,
        "sev": severity,
        "actor": {"user": actor, "role": "operator"},
        "target": target,
        "summary": summary,
        "meta": meta,
    }


GOLDEN_CASES = (
    (
        "auth-denied.json",
        _activity(
            event_id="evt_auth_denied",
            timestamp="2026-09-01T08:00:00Z",
            kind="auth",
            severity="warning",
            actor="alice",
            target_type="route",
            target_label="/twin/api/folders",
            summary="access denied on DELETE /twin/api/folders",
            meta={
                "operation": "access_denied",
                "method": "DELETE",
                "status_code": 403,
                "folder": "not-applicable",
                "reason": "forbidden",
            },
        ),
        {
            "request_id": "req-auth",
            "trace_id": "a" * 32,
            "route_group": "admin",
            "http_method": "DELETE",
            "auth_method": "idp",
        },
        None,
    ),
    (
        "query.json",
        _activity(
            event_id="evt_query",
            timestamp="2026-09-01T08:01:00Z",
            kind="retrieval",
            severity="info",
            actor="bob",
            target_type="query",
            target_label="How do I rotate password=never-export-this?",
            summary="retrieval completed (hybrid)",
            meta={
                "query": "How do I rotate password=never-export-this?",
                "mode": "hybrid",
                "sources": [{"content": "document body"}],
            },
        ),
        {
            "request_id": "req-query",
            "trace_id": "b" * 32,
            "route_group": "query",
            "http_method": "POST",
            "auth_method": "local_jwt",
        },
        "ops",
    ),
    (
        "ingestion-terminal.json",
        _activity(
            event_id="evt_source_failed",
            timestamp="2026-09-01T08:02:00Z",
            kind="source-failed",
            severity="error",
            actor="system",
            target_type="document",
            target_label="employee-handbook.pdf",
            target_id="doc-17",
            summary="password=hunter2 ingestion failed",
            meta={
                "doc_id": "doc-17",
                "folder": "hr",
                "classification": "C2",
                "retention_policy": "7y",
                "emitted_by": "server",
                "error_msg": "Bearer terminal-secret",
            },
        ),
        {
            "request_id": "-",
            "trace_id": "-",
            "route_group": "other",
            "http_method": "-",
            "auth_method": "unknown",
        },
        None,
    ),
    (
        "deletion.json",
        _activity(
            event_id="evt_delete",
            timestamp="2026-09-01T08:03:00Z",
            kind="doc-deleted",
            severity="info",
            actor="api_key:key-4",
            target_type="document",
            target_label="doc-42",
            target_id="doc-42",
            summary="removed from folder ops by key operator",
            meta={
                "doc_id": "doc-42",
                "folder_id": "ops",
                "operation": "remove-membership",
                "physically_deleted": True,
            },
        ),
        {
            "request_id": "req-delete",
            "trace_id": "c" * 32,
            "route_group": "documents",
            "http_method": "DELETE",
            "auth_method": "operator_api_key",
        },
        None,
    ),
    (
        "governance-mutation.json",
        _activity(
            event_id="evt_governance",
            timestamp="2026-09-01T08:04:00Z",
            kind="tag-mutation",
            severity="info",
            actor="steward",
            target_type="tag",
            target_label="oracle",
            target_id="oracle",
            summary="tag definition updated",
            meta={"operation": "update", "changed_fields": ["definition"]},
        ),
        {
            "request_id": "req-governance",
            "trace_id": "d" * 32,
            "route_group": "twin",
            "http_method": "PATCH",
            "auth_method": "idp",
        },
        "ops",
    ),
)


def _schema() -> dict:
    path = resources.files("twindb_lightrag_memgraph.server").joinpath(
        "schemas/audit-event-v1.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("fixture_name", "activity", "context", "folder_id"), GOLDEN_CASES
)
def test_golden_activity_projection_validates_model_and_json_schema(
    fixture_name: str,
    activity: dict,
    context: dict[str, str],
    folder_id: str | None,
):
    expected = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    projected = activity_to_audit_event(
        activity,
        request_context=context,
        folder_id=folder_id,
        workspace_id="default",
        service_name="twin-test",
        service_version="1.2.0",
        service_environment="test",
    )
    actual = projected.model_dump(mode="json", by_alias=True)

    assert actual == expected
    assert AuditEvent.model_validate(expected) == projected
    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(expected)


def test_audit_action_enum_is_exactly_the_activity_inventory():
    activity_kinds = set(get_args(ActivityEvent.model_fields["kind"].annotation))
    audit_actions = {action.value for action in AuditAction}
    schema = _schema()

    assert audit_actions == activity_kinds
    assert set(schema["properties"]["event"]["properties"]["action"]["enum"]) == (
        activity_kinds
    )
    assert set(schema["$defs"]["action"]["enum"]) == activity_kinds


@pytest.mark.parametrize("action", list(AuditAction))
def test_every_emitted_action_projects_to_the_shipped_schema(action: AuditAction):
    activity = _activity(
        event_id=f"evt_{action.value}",
        timestamp="2026-09-01T08:30:00Z",
        kind=action.value,
        severity="info",
        actor="system",
        target_type="resource",
        target_label="resource",
        target_id="resource-1",
        summary=f"{action.value} completed",
        meta={"operation": "complete"},
    )

    payload = activity_to_audit_event(
        activity,
        request_context={
            "request_id": "req-inventory",
            "trace_id": "7" * 32,
            "route_group": "twin",
            "http_method": "POST",
            "auth_method": "system",
        },
        folder_id="ops",
        workspace_id="default",
        service_name="twin-test",
        service_version="1.2.0",
        service_environment="test",
    ).model_dump(mode="json", by_alias=True)

    Draft202012Validator(_schema(), format_checker=FormatChecker()).validate(payload)


def test_projection_is_allow_listed_and_redacts_every_credential_surface():
    jwt = "eyJhbGciOiJIUzI1NiJ9.c2VjcmV0.c2lnbmF0dXJl"
    activity = _activity(
        event_id="evt_redaction",
        timestamp="2026-09-01T09:00:00Z",
        kind="tag-mutation",
        severity="critical",
        actor="sk_operator_secret_123456",
        target_type="tag",
        target_label="Bearer label-token",
        target_id="twk_resource_secret_123456",
        summary=(
            "cookie=session-secret prompt='ignore policy' "
            f"llm_response=model-secret {jwt}"
        ),
        meta={
            "operation": "update",
            "classification": "secret=classified-token",
            "document_body": "private document body",
            "request_body": {"password": "nested-secret"},
            "prompt": "private prompt",
            "response": "private model response",
            "authorization": "Bearer meta-secret",
            "cookie": "session=meta-cookie",
        },
    )

    serialized = activity_to_audit_event(
        activity,
        request_context={
            "request_id": "req-redact",
            "trace_id": "e" * 32,
            "route_group": "twin",
            "http_method": "PATCH",
            "auth_method": "idp",
        },
        folder_id="ops",
        workspace_id="default",
        service_name="twin-test",
        service_version="1.2.0",
        service_environment="test",
    ).model_dump_json(by_alias=True)

    for secret in (
        "operator_secret",
        "label-token",
        "resource_secret",
        "session-secret",
        "ignore policy",
        "model-secret",
        jwt,
        "classified-token",
        "private document body",
        "nested-secret",
        "private prompt",
        "private model response",
        "meta-secret",
        "meta-cookie",
    ):
        assert secret not in serialized
    assert "[REDACTED]" in serialized
    assert "document_body" not in serialized
    assert '"meta"' not in serialized


@pytest.mark.parametrize(
    ("context_method", "claimed_method", "expected"),
    (("open", "idp", "open"), ("invented", "static_api_key", "unknown")),
)
def test_activity_meta_cannot_claim_or_override_verified_auth_method(
    context_method: str,
    claimed_method: str,
    expected: str,
):
    event = _activity(
        event_id="evt_auth_trust",
        timestamp="2026-09-01T09:10:00Z",
        kind="auth",
        severity="warning",
        actor="client-claim",
        target_type="route",
        target_label="/folders",
        summary="access denied",
        meta={"operation": "access_denied", "auth_method": claimed_method},
    )

    projected = activity_to_audit_event(
        event,
        request_context={
            "auth_actor": "anonymous",
            "auth_method": context_method,
            "http_method": "GET",
            "request_id": "req-auth-trust",
            "route_group": "admin",
            "trace_id": "1" * 32,
        },
        workspace_id="default",
    )

    assert projected.twin.auth.method == expected
    assert projected.user.name == "anonymous"


def test_non_ascii_http_method_is_omitted_without_invalidating_event():
    event = _activity(
        event_id="evt_unicode_method",
        timestamp="2026-09-01T09:11:00Z",
        kind="settings",
        severity="info",
        actor="admin",
        target_type="settings",
        target_label="general",
        summary="settings inspected",
        meta={},
    )

    projected = activity_to_audit_event(
        event,
        request_context={"auth_method": "idp", "http_method": "DÉLETE"},
        workspace_id="default",
    )

    assert projected.http.request.method is None


@pytest.mark.parametrize(
    "folder_sentinel",
    (
        activity_events._FOLDER_INVALID,
        activity_events._FOLDER_NOT_APPLICABLE,
    ),
)
def test_activity_folder_sentinels_are_never_exported_as_folder_ids(
    folder_sentinel: str,
):
    event = _activity(
        event_id="evt_folder_sentinel",
        timestamp="2026-09-01T09:12:00Z",
        kind="auth",
        severity="warning",
        actor="anonymous",
        target_type="route",
        target_label="/settings",
        summary="access denied",
        meta={"operation": "access_denied", "folder": folder_sentinel},
    )

    projected = activity_to_audit_event(
        event,
        request_context={"auth_method": "unknown"},
        workspace_id="default",
    )

    assert projected.twin.folder.id is None


@pytest.mark.parametrize(
    ("event_id", "timestamp"),
    (
        ("unsafe event id", "2026-09-01T09:13:00Z"),
        ("evt_naive_timestamp", "2026-09-01T09:13:00"),
    ),
)
def test_unsafe_event_id_and_naive_timestamp_are_rejected(
    event_id: str, timestamp: str
):
    event = _activity(
        event_id=event_id,
        timestamp=timestamp,
        kind="settings",
        severity="info",
        actor="admin",
        target_type="settings",
        target_label="general",
        summary="settings inspected",
        meta={},
    )

    with pytest.raises(ValueError):
        activity_to_audit_event(
            event,
            request_context={"auth_method": "idp"},
            workspace_id="default",
        )


def test_logout_projects_as_successful_session_end():
    event = _activity(
        event_id="evt_logout",
        timestamp="2026-09-01T09:14:00Z",
        kind="auth",
        severity="info",
        actor="alice",
        target_type="auth",
        target_label="logout",
        summary="local session closed",
        meta={"operation": "logout", "success": True},
    )

    projected = activity_to_audit_event(
        event,
        request_context={"auth_actor": "alice", "auth_method": "local_jwt"},
        workspace_id="default",
    )

    assert projected.event.type == ["end"]
    assert projected.event.outcome == "success"


def test_truncation_annotation_reports_the_actual_removed_character_count():
    source = "x" * 200

    result = audit._redact_text(source, limit=64)
    prefix, annotation = result.split("…", 1)

    assert len(result) == 64
    assert annotation == f"[truncated:{len(source) - len(prefix)}]"


async def test_invalid_event_is_counted_and_logged_without_payload(
    caplog: pytest.LogCaptureFixture,
):
    reset_metrics()
    secret = "invalid-event-secret"
    invalid = _activity(
        event_id="evt_invalid",
        timestamp="not-a-timestamp",
        kind="retrieval",
        severity="info",
        actor="operator",
        target_type="query",
        target_label=secret,
        summary=f"token={secret}",
        meta={"document_body": secret},
    )

    with caplog.at_level(logging.WARNING, logger=audit.__name__):
        result = await submit_activity_audit_event(invalid)

    assert result is None
    assert metrics_snapshot()["audit_invalid_total"] == 1
    assert "audit_event_invalid" in caplog.text
    assert "source_fingerprint=" in caplog.text
    assert secret not in caplog.text


async def test_unexpected_projection_error_never_escapes_after_ledger_append(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    reset_metrics()
    secret = "runtime-import-secret"

    def broken_runtime_context() -> dict[str, str]:
        raise ImportError(secret)

    monkeypatch.setattr(audit, "_runtime_context", broken_runtime_context)
    store = WebuiStore.from_seed()
    event = _activity(
        event_id="evt_projection_import_error",
        timestamp="2026-09-01T09:20:00Z",
        kind="settings",
        severity="info",
        actor="admin",
        target_type="settings",
        target_label="general",
        target_id="general",
        summary="settings updated",
        meta={"operation": "update"},
    )

    with caplog.at_level(logging.WARNING, logger=audit.__name__):
        stored = await store.record_activity(event)

    rows, _, _ = await store.list_activity(resource_id="general")
    assert stored == event
    assert rows == [event]
    assert metrics_snapshot()["audit_invalid_total"] == 1
    assert "ImportError" in caplog.text
    assert secret not in caplog.text


async def test_validated_model_reaches_sink_and_ledger_contract_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    received: list[AuditEvent] = []

    async def sink(event: AuditEvent) -> None:
        received.append(event)

    monkeypatch.setattr(audit, "_audit_event_sink", sink)
    store = WebuiStore.from_seed()
    event = _activity(
        event_id="evt_sink",
        timestamp="2026-09-01T10:00:00Z",
        kind="settings",
        severity="info",
        actor="client-claim",
        target_type="folder",
        target_label="Operations",
        target_id="ops",
        summary="folder created",
        meta={"operation": "create", "folder_id": "ops"},
    )

    with bind_request_context(
        request_id="req-sink",
        trace_id="f" * 32,
        route_group="admin",
        http_method="POST",
        auth_method="idp",
        auth_actor="verified-admin",
    ):
        stored = await store.record_activity(event)

    rows, _, _ = await store.list_activity(resource_id="ops")
    assert stored == event
    assert rows[0] == event
    assert len(received) == 1
    assert isinstance(received[0], AuditEvent)
    assert received[0].event.action is AuditAction.SETTINGS
    assert received[0].http.request.id == "req-sink"
    assert received[0].twin.auth.method == "idp"
    assert received[0].user.name == "verified-admin"


async def test_sink_failure_is_counted_and_logged_without_exception_secret(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    reset_metrics()
    secret = "queue-credential-must-not-leak"

    def broken_sink(_event: AuditEvent) -> None:
        raise RuntimeError(secret)

    monkeypatch.setattr(audit, "_audit_event_sink", broken_sink)
    event = _activity(
        event_id="evt_drop",
        timestamp="2026-09-01T10:01:00Z",
        kind="settings",
        severity="info",
        actor="admin",
        target_type="settings",
        target_label="vision",
        target_id="vision",
        summary="settings updated",
        meta={"operation": "update"},
    )

    with caplog.at_level(logging.ERROR, logger=audit.__name__):
        result = await submit_activity_audit_event(event)

    assert result is None
    assert metrics_snapshot()["audit_dropped_total"] == 1
    assert "audit_event_dropped" in caplog.text
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text


def test_runtime_login_projects_verified_actor_auth_method_and_correlation(
    monkeypatch: pytest.MonkeyPatch,
):
    received: list[AuditEvent] = []
    monkeypatch.setattr(audit, "_audit_event_sink", received.append)
    monkeypatch.setenv("TWIN_LOGIN_RATE_LIMIT_PER_MINUTE", "0")
    app = create_app(
        LightRAGServerSettings(
            api_key=None,
            jwt_secret="audit-test-jwt-secret-at-least-32-bytes",
            jwt_username="alice",
            jwt_password="correct-password",
            llm_binding_api_key="test",
            embedding_binding_api_key="test",
        )
    )

    try:
        response = TestClient(app).post(
            "/login",
            headers={
                "x-request-id": "req-runtime-login",
                "traceparent": f"00-{'9' * 32}-{'8' * 16}-01",
            },
            json={"username": "alice", "password": "correct-password"},
        )

        assert response.status_code == 200
        login_event = next(
            event
            for event in received
            if event.event.action is AuditAction.AUTH
            and event.twin.audit.operation == "login_success"
        )
        assert login_event.user.name == "alice"
        assert login_event.twin.auth.method == "local_jwt"
        assert login_event.http.request.id == "req-runtime-login"
        assert login_event.trace.id == "9" * 32
        assert login_event.http.request.method == "POST"
    finally:
        configure_auth(api_key=None, jwt_secret=None)
        reset_store()


async def test_runtime_missing_credential_records_unknown_auth_method(
    monkeypatch: pytest.MonkeyPatch,
):
    received: list[AuditEvent] = []
    monkeypatch.setattr(audit, "_audit_event_sink", received.append)
    app = create_app(
        LightRAGServerSettings(
            api_key="audit-static-api-key",
            jwt_secret=None,
            llm_binding_api_key="test",
            embedding_binding_api_key="test",
        )
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/folders",
                json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            )
            for _ in range(50):
                if any(
                    event.twin.audit.operation == "access_denied" for event in received
                ):
                    break
                await asyncio.sleep(0.01)

        assert response.status_code == 401
        denied = next(
            event for event in received if event.twin.audit.operation == "access_denied"
        )
        assert denied.twin.auth.method == "unknown"
        assert denied.user.name == "anonymous"
    finally:
        configure_auth(api_key=None, jwt_secret=None)
        reset_store()


def test_schema_is_shipped_from_the_installed_package():
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith("audit-event-v1.schema.json")
    assert schema["properties"]["ecs"]["properties"]["version"]["const"] == ("9.4.0")


def test_model_and_schema_reject_missing_fields_and_noncanonical_timestamp():
    payload = json.loads((FIXTURES / "deletion.json").read_text(encoding="utf-8"))
    payload["@timestamp"] = "2026-09-01T10:03:00+02:00"
    del payload["twin"]["resource"]["id"]

    with pytest.raises(ValidationError):
        AuditEvent.model_validate(payload)
    errors = list(
        Draft202012Validator(_schema(), format_checker=FormatChecker()).iter_errors(
            payload
        )
    )
    assert any(error.validator == "required" for error in errors)
    assert any(error.validator == "pattern" for error in errors)
