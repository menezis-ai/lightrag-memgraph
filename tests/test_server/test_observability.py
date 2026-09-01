"""Contracts for issues #119 (JSON/ECS logs) and #121 (Prometheus)."""

from __future__ import annotations

import asyncio
import io
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.metrics import (
    metrics_snapshot,
    record_audit_event,
    reset_metrics,
)
from twindb_lightrag_memgraph.server.observability import (
    TwinJsonFormatter,
    bind_request_context,
    configure_runtime_logging,
    current_request_context,
    make_request_observability_middleware,
)
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _app(*, api_key: str | None = None):
    return create_app(
        LightRAGServerSettings(
            api_key=api_key,
            jwt_secret=None,
            llm_binding_api_key="test",
            embedding_binding_api_key="test",
        )
    )


def _format_record(record: logging.LogRecord) -> dict:
    return json.loads(
        TwinJsonFormatter(
            service_name="twin-test",
            service_version="1.2.0",
            environment="test",
        ).format(record)
    )


_SENSITIVE_LOG_KEY_VARIANTS = (
    "OPENAI_API_KEY",
    "TOKEN_SECRET",
    "client_secret",
    "refresh_token",
    "openaiApiKey",
    "tokenSecret",
    "clientSecret",
    "refreshToken",
)


def test_json_formatter_emits_stable_ecs_fields_and_redacts_secrets():
    record = logging.LogRecord(
        name="third.party",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=(
            "authorization=Bearer top-secret cookie=session-id "
            "prompt='ignore policy' request_body=document-text"
        ),
        args=(),
        exc_info=None,
    )
    with bind_request_context(
        request_id="request-a", trace_id="a" * 32, route_group="query"
    ):
        payload = _format_record(record)

    assert payload["@timestamp"].endswith("Z")
    assert payload["log.level"] == "warning"
    assert payload["log.logger"] == "third.party"
    assert payload["service.name"] == "twin-test"
    assert payload["service.version"] == "1.2.0"
    assert payload["service.environment"] == "test"
    assert payload["process.pid"] > 0
    assert payload["twin.request.id"] == "request-a"
    assert payload["trace.id"] == "a" * 32
    assert payload["span.id"] == "-"
    assert payload["twin.route.group"] == "query"
    assert "top-secret" not in payload["message"]
    assert "session-id" not in payload["message"]
    assert "ignore policy" not in payload["message"]
    assert "document-text" not in payload["message"]


@pytest.mark.parametrize("sensitive_key", _SENSITIVE_LOG_KEY_VARIANTS)
def test_sensitive_name_variants_are_redacted_in_valid_json(sensitive_key):
    record = logging.LogRecord(
        name="twin.json",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=json.dumps({sensitive_key: "valid-json-secret", "safe_value": "visible"}),
        args=(),
        exc_info=None,
    )

    message = json.loads(_format_record(record)["message"])

    assert message[sensitive_key] == "[REDACTED]"
    assert message["safe_value"] == "visible"


@pytest.mark.parametrize("sensitive_key", _SENSITIVE_LOG_KEY_VARIANTS)
def test_sensitive_name_variants_are_redacted_in_embedded_malformed_json(
    sensitive_key,
):
    record = logging.LogRecord(
        name="twin.text",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=f'embedded payload {{"{sensitive_key}": "malformed-json-secret", trailing',
        args=(),
        exc_info=None,
    )

    message = _format_record(record)["message"]

    assert "malformed-json-secret" not in message
    assert f'"{sensitive_key}": "[REDACTED]"' in message


@pytest.mark.parametrize("sensitive_key", _SENSITIVE_LOG_KEY_VARIANTS)
def test_sensitive_name_variants_are_redacted_in_exception_messages(sensitive_key):
    try:
        raise RuntimeError(f'{{"{sensitive_key}":"exception-json-secret", trailing')
    except RuntimeError:
        record = logging.LogRecord(
            name="twin.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="operation failed",
            args=(),
            exc_info=__import__("sys").exc_info(),
        )

    payload = _format_record(record)

    assert "exception-json-secret" not in payload["error.message"]
    assert f'"{sensitive_key}":"[REDACTED]"' in payload["error.message"]


def test_sensitive_name_segments_do_not_redact_unrelated_prefixes():
    safe_values = {
        "tokenizer_count": 12,
        "secretary": "visible",
        "api_version": "v1",
        "request_id": "request-visible",
    }
    record = logging.LogRecord(
        name="twin.json",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=json.dumps(safe_values),
        args=(),
        exc_info=None,
    )

    assert json.loads(_format_record(record)["message"]) == safe_values


@pytest.mark.parametrize(
    "message",
    [
        '{"password":"s3cr3t","api_key":"abc123",'
        '"request_body":"classified text","nested":{"prompt":"override"}}',
        '{ "password" : "s3cr3t", "api_key" : "abc123", '
        '"request_body" : "classified text", "nested" : { "prompt" : "override" } }',
    ],
)
def test_json_formatter_recursively_redacts_compact_and_spaced_json(message):
    record = logging.LogRecord(
        name="twin.json",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )

    payload = _format_record(record)
    redacted_message = json.loads(payload["message"])

    assert redacted_message == {
        "password": "[REDACTED]",
        "api_key": "[REDACTED]",
        "request_body": "[REDACTED]",
        "nested": {"prompt": "[REDACTED]"},
    }
    assert not any(
        secret in payload["message"]
        for secret in ("s3cr3t", "abc123", "classified text", "override")
    )


def test_json_key_redaction_also_covers_non_json_text_messages():
    record = logging.LogRecord(
        name="twin.text",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='failed payload "password": "text-secret", "request_body":"body"',
        args=(),
        exc_info=None,
    )

    message = _format_record(record)["message"]

    assert "text-secret" not in message
    assert '"password": "[REDACTED]"' in message
    assert '"request_body":"[REDACTED]"' in message


def test_json_exception_is_structured_bounded_and_contains_no_stack_path():
    secret = "never-emit-this-credential"
    try:
        raise RuntimeError(
            f"password={secret} secret_path=/Users/operator/private/key.pem "
            + ("x" * 2000)
        )
    except RuntimeError:
        record = logging.LogRecord(
            name="twin.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="operation failed",
            args=(),
            exc_info=__import__("sys").exc_info(),
        )

    payload = _format_record(record)
    serialized = json.dumps(payload)
    assert payload["error.type"] == "RuntimeError"
    assert len(payload["error.message"]) <= 550
    assert len(payload["error.stack_hash"]) == 64
    assert secret not in serialized
    assert "/Users/operator" not in serialized
    assert "error.stack_trace" not in payload


def test_json_exception_message_is_recursively_redacted():
    secret = "exception-secret-value"
    try:
        raise RuntimeError(
            json.dumps(
                {
                    "password": secret,
                    "metadata": {"api_key": "nested-api-key"},
                    "prompt": "classified prompt",
                }
            )
        )
    except RuntimeError:
        record = logging.LogRecord(
            name="twin.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="operation failed",
            args=(),
            exc_info=__import__("sys").exc_info(),
        )

    payload = _format_record(record)
    serialized = json.dumps(payload)
    assert payload["error.type"] == "RuntimeError"
    assert secret not in serialized
    assert "nested-api-key" not in serialized
    assert "classified prompt" not in serialized


async def test_request_context_isolated_between_concurrent_tasks_and_reset():
    release = asyncio.Event()

    async def worker(request_id: str, trace_id: str):
        with bind_request_context(
            request_id=request_id, trace_id=trace_id, route_group="query"
        ):
            await release.wait()
            await asyncio.sleep(0)
            return current_request_context()

    first = asyncio.create_task(worker("request-1", "1" * 32))
    second = asyncio.create_task(worker("request-2", "2" * 32))
    await asyncio.sleep(0)
    release.set()
    contexts = await asyncio.gather(first, second)

    assert contexts == [
        {
            "request_id": "request-1",
            "trace_id": "1" * 32,
            "span_id": "-",
            "route_group": "query",
            "http_method": "-",
            "auth_method": "unknown",
            "auth_actor": "-",
        },
        {
            "request_id": "request-2",
            "trace_id": "2" * 32,
            "span_id": "-",
            "route_group": "query",
            "http_method": "-",
            "auth_method": "unknown",
            "auth_actor": "-",
        },
    ]
    assert current_request_context() == {
        "request_id": "-",
        "trace_id": "-",
        "span_id": "-",
        "route_group": "other",
        "http_method": "-",
        "auth_method": "unknown",
        "auth_actor": "-",
    }


async def test_two_concurrent_http_requests_keep_distinct_w3c_contexts():
    app = FastAPI()
    app.middleware("http")(make_request_observability_middleware())
    release = asyncio.Event()
    entered = 0

    @app.get("/context/{name}")
    async def context_probe(name: str):
        nonlocal entered
        entered += 1
        if entered == 2:
            release.set()
        await release.wait()
        await asyncio.sleep(0)
        return {"name": name, **current_request_context()}

    first_parent = f"00-{'1' * 32}-{'a' * 16}-01"
    second_parent = f"00-{'2' * 32}-{'b' * 16}-00"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first, second = await asyncio.gather(
            client.get("/context/first", headers={"traceparent": first_parent}),
            client.get("/context/second", headers={"traceparent": second_parent}),
        )

    assert first.json()["trace_id"] == "1" * 32
    assert second.json()["trace_id"] == "2" * 32
    assert first.json()["span_id"] != second.json()["span_id"]
    assert first.headers["traceparent"] == (
        f"00-{'1' * 32}-{first.json()['span_id']}-01"
    )
    assert second.headers["traceparent"] == (
        f"00-{'2' * 32}-{second.json()['span_id']}-00"
    )
    assert current_request_context() == {
        "request_id": "-",
        "trace_id": "-",
        "span_id": "-",
        "route_group": "other",
        "http_method": "-",
        "auth_method": "unknown",
        "auth_actor": "-",
    }


async def test_request_context_resets_after_unhandled_exception():
    app = FastAPI()
    app.middleware("http")(make_request_observability_middleware())

    @app.get("/boom")
    async def boom():
        assert current_request_context()["span_id"] != "-"
        raise RuntimeError("expected test failure")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert current_request_context() == {
        "request_id": "-",
        "trace_id": "-",
        "span_id": "-",
        "route_group": "other",
        "http_method": "-",
        "auth_method": "unknown",
        "auth_actor": "-",
    }


def test_json_logging_is_configuration_gated(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    runtime_logger = logging.getLogger("twindb_lightrag_memgraph")
    original_level = runtime_logger.level
    runtime_logger.addHandler(handler)
    runtime_logger.setLevel(logging.INFO)
    monkeypatch.setenv("TWIN_LOG_FORMAT", "json")
    monkeypatch.setenv("TWIN_SERVICE_NAME", "configured-service")
    monkeypatch.setenv("TWIN_ENV", "test")
    try:
        assert configure_runtime_logging() is True
        runtime_logger.info("configured event", extra={"event_action": "test"})
        payload = json.loads(stream.getvalue().strip())
        assert payload["service.name"] == "configured-service"
        assert payload["event.action"] == "test"
    finally:
        monkeypatch.setenv("TWIN_LOG_FORMAT", "text")
        configure_runtime_logging()
        runtime_logger.setLevel(original_level)
        runtime_logger.removeHandler(handler)


def test_prometheus_endpoint_is_authenticated_parseable_and_low_cardinality():
    reset_metrics()
    client = TestClient(_app(api_key="metrics-secret"))

    assert client.get("/twin/api/ops/metrics/prometheus").status_code == 401
    headers = {"Authorization": "Bearer metrics-secret"}
    assert client.get("/health", headers=headers).status_code == 200
    assert client.get("/twin/api/documents", headers=headers).status_code == 200
    response = client.get("/twin/api/ops/metrics/prometheus", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    families = list(text_string_to_metric_families(response.text))
    samples = [sample for family in families for sample in family.samples]
    request_samples = [s for s in samples if s.name == "twin_http_requests_total"]
    assert request_samples
    assert {s.labels["route_group"] for s in request_samples} >= {
        "health",
        "documents",
    }
    assert all(
        set(s.labels) <= {"route_group", "method", "status_class"}
        for s in request_samples
    )
    assert all("path" not in s.labels for s in samples)


def test_metrics_reset_isolates_http_audit_and_storage_counters():
    reset_metrics()
    record_audit_event("invalid")
    record_audit_event("dropped")
    assert metrics_snapshot()["audit_invalid_total"] == 1
    assert metrics_snapshot()["audit_dropped_total"] == 1

    reset_metrics()
    snapshot = metrics_snapshot()
    assert snapshot["audit_invalid_total"] == 0
    assert snapshot["audit_dropped_total"] == 0
    assert snapshot["storage_writes_total"] == 0


async def test_write_slot_records_success_and_error_without_changing_semantics():
    reset_metrics()
    async with _pool.acquire_write_slot():
        pass
    with pytest.raises(RuntimeError, match="write failed"):
        async with _pool.acquire_write_slot():
            raise RuntimeError("write failed")

    snapshot = metrics_snapshot()
    assert snapshot["storage_writes_total"] == 2
    assert snapshot["storage_errors_total"] == 1
