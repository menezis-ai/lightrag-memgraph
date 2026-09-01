"""Contracts for L3 LLM profiles, shared clients, and typed retry."""

import asyncio
import functools
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from twindb_lightrag_memgraph.intelligence import llm as llm_module
from twindb_lightrag_memgraph.intelligence.config import (
    LLMProfileKind,
    TwinRAGConfig,
    resolve_llm_profile,
)
from twindb_lightrag_memgraph.intelligence.engine import TwinRAGEngine
from twindb_lightrag_memgraph.intelligence.llm import (
    get_llm_client,
    inject_llm_client_for_testing,
    reset_llm_clients_for_testing,
    with_llm_retry,
)
from twindb_lightrag_memgraph.intelligence.ontology.config import (
    OntologyConfig,
    WorkspaceOntologyConfig,
)
from twindb_lightrag_memgraph.intelligence.ontology.pipeline import OntologyPipeline
from twindb_lightrag_memgraph.server.tracing import (
    TraceContext,
    bind_trace_context,
)
from twindb_lightrag_memgraph.server.observability import (
    make_request_observability_middleware,
)

_LANGSMITH_PARENT = "20260901T004915204525Z01a05a71-13c4-7582-bcf9-e358e56dc683"
_W3C_PARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture(autouse=True)
def _clean_llm_client_registry():
    reset_llm_clients_for_testing()
    yield
    reset_llm_clients_for_testing()


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
    return httpx.Response(status_code, request=request)


def _rate_limit(secret: str = "provider-secret") -> RateLimitError:
    return RateLimitError(
        f"rate limited; echoed {secret}",
        response=_response(429),
        body=None,
    )


def _server_error(secret: str = "provider-secret") -> InternalServerError:
    return InternalServerError(
        f"server failed; echoed {secret}",
        response=_response(503),
        body=None,
    )


def _bad_request() -> BadRequestError:
    return BadRequestError("bad request", response=_response(400), body=None)


def _authentication_error() -> AuthenticationError:
    return AuthenticationError("bad credential", response=_response(401), body=None)


def _completion_response(content: str, total_tokens: int = 100):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock(total_tokens=total_tokens)
    return response


def test_profiles_resolve_distinct_chat_and_indexing_connections():
    config = TwinRAGConfig(
        llm_api_key="chat-key",
        llm_api_base="https://chat.invalid/v1",
        llm_model="chat-model",
        indexing_api_key="index-key",
        indexing_api_base="https://index.invalid/v1",
        indexing_model="index-model",
    )

    chat = resolve_llm_profile(config, LLMProfileKind.CHAT)
    indexing = resolve_llm_profile(config, LLMProfileKind.INDEXING)

    assert (chat.api_key, chat.api_base, chat.model) == (
        "chat-key",
        "https://chat.invalid/v1",
        "chat-model",
    )
    assert (indexing.api_key, indexing.api_base, indexing.model) == (
        "index-key",
        "https://index.invalid/v1",
        "index-model",
    )


@pytest.mark.parametrize(
    ("indexing_overrides", "expected"),
    [
        (
            {"indexing_api_key": None},
            ("chat-key", "https://index.invalid/v1", "index-model"),
        ),
        (
            {"indexing_api_base": None},
            ("index-key", "https://chat.invalid/v1", "index-model"),
        ),
        (
            {"indexing_model": None},
            ("index-key", "https://index.invalid/v1", "chat-model"),
        ),
    ],
)
def test_each_missing_indexing_field_falls_back_independently(
    indexing_overrides, expected
):
    values = {
        "llm_api_key": "chat-key",
        "llm_api_base": "https://chat.invalid/v1",
        "llm_model": "chat-model",
        "indexing_api_key": "index-key",
        "indexing_api_base": "https://index.invalid/v1",
        "indexing_model": "index-model",
        **indexing_overrides,
    }

    profile = resolve_llm_profile(
        TwinRAGConfig(**values),
        LLMProfileKind.INDEXING,
    )

    assert (profile.api_key, profile.api_base, profile.model) == expected


def test_clients_are_reused_and_sdk_retries_are_disabled():
    config = TwinRAGConfig(
        llm_api_key="chat-key",
        llm_api_base="https://chat.invalid/v1",
    )
    client = MagicMock(name="shared-chat-client")
    factory = MagicMock(return_value=client)

    first = get_llm_client(config, LLMProfileKind.CHAT, client_factory=factory)
    second = get_llm_client(config, LLMProfileKind.CHAT, client_factory=factory)

    assert first is client
    assert second is client
    factory.assert_called_once_with(
        api_key="chat-key",
        base_url="https://chat.invalid/v1",
        max_retries=0,
    )


def test_distinct_effective_connections_receive_distinct_clients():
    config = TwinRAGConfig(
        llm_api_key="chat-key",
        llm_api_base="https://chat.invalid/v1",
        indexing_api_key="index-key",
        indexing_api_base="https://index.invalid/v1",
    )
    chat_client = MagicMock(name="chat-client")
    indexing_client = MagicMock(name="indexing-client")
    factory = MagicMock(side_effect=[chat_client, indexing_client])

    assert (
        get_llm_client(config, LLMProfileKind.CHAT, client_factory=factory)
        is chat_client
    )
    assert (
        get_llm_client(config, LLMProfileKind.INDEXING, client_factory=factory)
        is indexing_client
    )
    assert (
        get_llm_client(config, LLMProfileKind.INDEXING, client_factory=factory)
        is indexing_client
    )

    assert factory.call_count == 2
    assert factory.call_args_list[0].kwargs["base_url"] == "https://chat.invalid/v1"
    assert factory.call_args_list[1].kwargs["base_url"] == "https://index.invalid/v1"


async def test_user_facing_pipeline_uses_only_the_chat_model():
    config = TwinRAGConfig(
        llm_api_key="chat-key",
        llm_api_base="https://chat.invalid/v1",
        llm_model="chat-model",
        indexing_api_key="index-key",
        indexing_api_base="https://index.invalid/v1",
        indexing_model="index-model",
        enable_query_expansion=False,
        enable_folder_routing=False,
    )
    engine = TwinRAGEngine(config)
    chat_client = AsyncMock()
    chat_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion_response(
                json.dumps({"i": "IN_SCOPE", "c": 0.99, "r": "technical"})
            ),
            _completion_response(
                json.dumps(
                    {
                        "t": "reason",
                        "q": "oracle memory",
                        "d": "oracle",
                        "cr": False,
                    }
                )
            ),
            _completion_response(json.dumps({"s": [{"p": 0, "v": 9}]})),
            _completion_response("Grounded answer [Passage 0]"),
        ]
    )
    indexing_client = AsyncMock()
    inject_llm_client_for_testing(config, LLMProfileKind.CHAT, chat_client)
    inject_llm_client_for_testing(config, LLMProfileKind.INDEXING, indexing_client)

    rag = MagicMock()
    rag.aquery_data = AsyncMock(
        return_value={
            "data": {
                "chunks": [
                    {"chunk_id": "c-1", "content": "Oracle memory", "score": 0.9}
                ]
            }
        }
    )
    engine._get_rag = MagicMock(return_value=rag)

    captured_spans: list[tuple[str, str, dict]] = []

    def capturing_traceable(*, name, run_type):
        def decorator(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                captured_spans.append((name, run_type, kwargs.pop("langsmith_extra")))
                return await fn(*args, **kwargs)

            return wrapper

        return decorator

    app = FastAPI()
    app.middleware("http")(make_request_observability_middleware())

    @app.get("/l3-query-probe")
    async def l3_query_probe():
        await engine.aquery("Pourquoi ?", authorized_folders={"commons"})
        return {"ok": True}

    with (
        patch("twindb_lightrag_memgraph.server.tracing._TRACING_ENABLED", True),
        patch("twindb_lightrag_memgraph.server.tracing._langsmith_available", True),
        patch(
            "twindb_lightrag_memgraph.server.tracing.traceable",
            capturing_traceable,
            create=True,
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as http_client:
            response = await http_client.get(
                "/l3-query-probe",
                headers={
                    "traceparent": _W3C_PARENT,
                    "langsmith-trace": _LANGSMITH_PARENT,
                    "baggage": "langsmith-project=twin-test",
                },
            )

    assert response.status_code == 200
    response_traceparent = response.headers["traceparent"]
    _, response_trace_id, response_span_id, _ = response_traceparent.split("-")

    assert chat_client.chat.completions.create.await_count == 4
    assert {
        call.kwargs["model"]
        for call in chat_client.chat.completions.create.await_args_list
    } == {"chat-model"}
    for call in chat_client.chat.completions.create.await_args_list:
        assert call.kwargs["extra_headers"] == {
            "traceparent": response_traceparent,
            "x-trace-id": response_trace_id,
            "langsmith-trace": _LANGSMITH_PARENT,
            "baggage": "langsmith-project=twin-test",
        }
    assert [(name, run_type) for name, run_type, _ in captured_spans] == [
        ("TwinRAG:llm", "llm"),
    ] * 4
    for _, _, extra in captured_spans:
        assert extra == {
            "metadata": {
                "w3c.trace_id": response_trace_id,
                "w3c.span_id": response_span_id,
            },
            "parent": {
                "langsmith-trace": _LANGSMITH_PARENT,
                "baggage": "langsmith-project=twin-test",
            },
        }
    indexing_client.chat.completions.create.assert_not_awaited()


async def test_shared_l3_client_merges_headers_without_cross_request_leakage():
    config = TwinRAGConfig(
        llm_api_key="chat-key",
        llm_api_base="https://chat.invalid/v1",
        llm_model="chat-model",
    )
    calls: list[dict] = []
    both_started = asyncio.Event()

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            both_started.set()
        await both_started.wait()
        return _completion_response("ok")

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=create)
    inject_llm_client_for_testing(config, LLMProfileKind.CHAT, client)

    async def one_call(label: str, trace_id: str, span_id: str):
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags="01",
            source="w3c",
        )
        with bind_trace_context(context):
            return await llm_module.create_chat_completion(
                config,
                LLMProfileKind.CHAT,
                messages=[{"role": "user", "content": label}],
                extra_headers={
                    "x-provider-option": label,
                    "TraceParent": "stale-parent",
                },
            )

    await asyncio.gather(
        one_call("first", "a" * 32, "1" * 16),
        one_call("second", "b" * 32, "2" * 16),
    )

    by_label = {call["messages"][0]["content"]: call for call in calls}
    assert by_label["first"]["extra_headers"] == {
        "x-provider-option": "first",
        "traceparent": f"00-{'a' * 32}-{'1' * 16}-01",
        "x-trace-id": "a" * 32,
    }
    assert by_label["second"]["extra_headers"] == {
        "x-provider-option": "second",
        "traceparent": f"00-{'b' * 32}-{'2' * 16}-01",
        "x-trace-id": "b" * 32,
    }


async def test_ontology_pipeline_uses_only_the_indexing_model():
    config = TwinRAGConfig(
        llm_api_key="chat-key",
        llm_api_base="https://chat.invalid/v1",
        llm_model="chat-model",
        indexing_api_key="index-key",
        indexing_api_base="https://index.invalid/v1",
        indexing_model="index-model",
    )
    pipeline = OntologyPipeline(
        config,
        OntologyConfig(
            enabled=True,
            require_review=True,
            workspaces={"ws": WorkspaceOntologyConfig(mode="emergence")},
        ),
    )
    chat_client = AsyncMock()
    indexing_client = AsyncMock()
    indexing_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion_response(
                json.dumps(
                    {
                        "e": [{"n": "Oracle", "t": "Tool", "c": 0.9}],
                        "r": [],
                    }
                )
            ),
            _completion_response(json.dumps({"domains": []})),
            _completion_response(json.dumps({"new_relations": []})),
        ]
    )
    inject_llm_client_for_testing(config, LLMProfileKind.CHAT, chat_client)
    inject_llm_client_for_testing(config, LLMProfileKind.INDEXING, indexing_client)

    result = await pipeline.run(["Oracle operational document"], "ws")

    assert result.nodes
    assert indexing_client.chat.completions.create.await_count == 3
    assert {
        call.kwargs["model"]
        for call in indexing_client.chat.completions.create.await_args_list
    } == {"index-model"}
    chat_client.chat.completions.create.assert_not_awaited()


def test_cache_keys_and_representations_do_not_contain_credentials():
    secret = "SECRET_TOKEN=cache-key-must-not-leak"
    secret_base = "https://user:password@llm.invalid/v1?token=base-secret"
    config = TwinRAGConfig(
        llm_api_key=secret,
        llm_api_base=secret_base,
    )
    factory = MagicMock(return_value=MagicMock(name="client"))

    profile = resolve_llm_profile(config, LLMProfileKind.CHAT)
    get_llm_client(config, LLMProfileKind.CHAT, client_factory=factory)

    assert secret not in repr(config)
    assert secret not in str(config)
    assert secret_base not in repr(config)
    assert secret_base not in str(config)
    assert secret not in repr(profile)
    assert secret_base not in repr(profile)
    assert secret not in repr(llm_module._CLIENT_POOLS)
    assert secret_base not in repr(llm_module._CLIENT_POOLS)
    assert all(secret not in repr(key) for key in llm_module._CLIENT_POOLS)


def test_injection_and_reset_are_deterministic():
    config = TwinRAGConfig(llm_api_key="chat-key")
    injected = MagicMock(name="injected")
    replacement = MagicMock(name="replacement")
    factory = MagicMock(return_value=replacement)

    inject_llm_client_for_testing(config, LLMProfileKind.CHAT, injected)
    assert (
        get_llm_client(config, LLMProfileKind.CHAT, client_factory=factory) is injected
    )
    factory.assert_not_called()

    reset_llm_clients_for_testing(config)

    assert (
        get_llm_client(config, LLMProfileKind.CHAT, client_factory=factory)
        is replacement
    )
    factory.assert_called_once()


@pytest.mark.parametrize(
    "transient_error",
    [
        _rate_limit(),
        APITimeoutError(httpx.Request("POST", "https://llm.invalid")),
        APIConnectionError(
            message="connection reset",
            request=httpx.Request("POST", "https://llm.invalid"),
        ),
        _server_error(),
    ],
)
async def test_transient_error_categories_are_retried(transient_error):
    attempts = 0
    sleep = AsyncMock()

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise transient_error
        return "ok"

    result = await with_llm_retry(
        operation,
        profile=LLMProfileKind.CHAT,
        max_attempts=3,
        base_seconds=0.25,
        max_seconds=1.0,
        jitter_ratio=0.2,
        sleep=sleep,
        random_value=lambda: 0.5,
    )

    assert result == "ok"
    assert attempts == 2
    sleep.assert_awaited_once_with(0.275)


@pytest.mark.parametrize(
    "permanent_error",
    [_bad_request(), _authentication_error(), ValueError("invalid config")],
)
async def test_request_config_and_auth_errors_fail_fast(permanent_error):
    operation = AsyncMock(side_effect=permanent_error)
    sleep = AsyncMock()

    with pytest.raises(type(permanent_error)):
        await with_llm_retry(
            operation,
            profile=LLMProfileKind.CHAT,
            max_attempts=3,
            base_seconds=0.25,
            max_seconds=1.0,
            jitter_ratio=0.2,
            sleep=sleep,
        )

    assert operation.await_count == 1
    sleep.assert_not_awaited()


async def test_retry_count_backoff_jitter_and_secret_safe_log(caplog):
    secret = "SECRET_TOKEN=retry-log-must-not-leak"
    outcomes = iter([_rate_limit(secret), _server_error(secret), "ok"])
    sleep = AsyncMock()

    async def operation():
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    with caplog.at_level("WARNING", logger="twin_rag_intelligence.llm"):
        result = await with_llm_retry(
            operation,
            profile=LLMProfileKind.INDEXING,
            max_attempts=3,
            base_seconds=0.5,
            max_seconds=1.0,
            jitter_ratio=0.2,
            sleep=sleep,
            random_value=lambda: 0.5,
        )

    assert result == "ok"
    assert [call.args[0] for call in sleep.await_args_list] == [0.55, 1.0]
    assert "RateLimitError" in caplog.text
    assert "InternalServerError" in caplog.text
    assert secret not in caplog.text


async def test_transient_failure_stops_at_the_configured_attempt_bound():
    operation = AsyncMock(side_effect=_rate_limit())
    sleep = AsyncMock()

    with pytest.raises(RateLimitError):
        await with_llm_retry(
            operation,
            profile=LLMProfileKind.CHAT,
            max_attempts=3,
            base_seconds=0.25,
            max_seconds=1.0,
            jitter_ratio=0.0,
            sleep=sleep,
        )

    assert operation.await_count == 3
    assert sleep.await_count == 2


async def test_asyncio_cancellation_is_never_swallowed():
    operation = AsyncMock(side_effect=asyncio.CancelledError())
    sleep = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await with_llm_retry(
            operation,
            profile=LLMProfileKind.CHAT,
            max_attempts=3,
            base_seconds=0.25,
            max_seconds=1.0,
            jitter_ratio=0.2,
            sleep=sleep,
        )

    assert operation.await_count == 1
    sleep.assert_not_awaited()
