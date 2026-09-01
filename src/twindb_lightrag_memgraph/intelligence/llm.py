"""Shared OpenAI-compatible clients and typed retry for the L3 layer.

The registry is scoped by the identity of a ``TwinRAGConfig`` object, then by
the safe ``chat``/``indexing`` profile name. Credentials are values inside a
redacted entry, never part of a cache key, log message, or representation.

The OpenAI SDK's implicit retries are disabled (``max_retries=0``). This module
is therefore the single owner of retry semantics: rate limits, SDK timeouts,
connection failures, and HTTP 5xx responses retry with bounded exponential
backoff plus jitter. Request/config/auth failures pass through immediately so
the existing call-site business fallbacks remain in control.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from .config import (
    EffectiveLLMProfile,
    LLMProfileKind,
    TwinRAGConfig,
    resolve_llm_profile,
)

logger = logging.getLogger("twin_rag_intelligence.llm")

_T = TypeVar("_T")
_ClientFactory = Callable[..., Any]
_MAX_LLM_RESPONSE_CHARS = 64_000


class LLMResponseTooLargeError(ValueError):
    """Raised when a provider ignores output limits by an unsafe margin."""


@dataclass(slots=True, repr=False)
class _ClientEntry:
    profile: EffectiveLLMProfile
    factory: object
    client: Any
    injected: bool = False

    def __repr__(self) -> str:
        api_base = "<configured>" if self.profile.api_base is not None else None
        return (
            "_ClientEntry("
            f"kind={self.profile.kind.value!r}, api_key=<redacted>, "
            f"api_base={api_base!r}, "
            f"model={self.profile.model!r}, injected={self.injected!r})"
        )


@dataclass(slots=True, repr=False)
class _ConfigClients:
    config_ref: weakref.ReferenceType[TwinRAGConfig]
    clients: dict[LLMProfileKind, _ClientEntry] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"_ConfigClients(profiles={[kind.value for kind in self.clients]!r})"


_CLIENT_POOLS: dict[int, _ConfigClients] = {}
_CLIENT_POOLS_LOCK = threading.RLock()


def _purge_dead_config(
    config_id: int,
    config_ref: weakref.ReferenceType[TwinRAGConfig],
) -> None:
    with _CLIENT_POOLS_LOCK:
        current = _CLIENT_POOLS.get(config_id)
        if current is not None and current.config_ref is config_ref:
            _CLIENT_POOLS.pop(config_id, None)


def _pool_for(config: TwinRAGConfig) -> _ConfigClients:
    config_id = id(config)
    with _CLIENT_POOLS_LOCK:
        current = _CLIENT_POOLS.get(config_id)
        if current is not None and current.config_ref() is config:
            return current

        config_ref = weakref.ref(
            config,
            lambda ref, key=config_id: _purge_dead_config(key, ref),
        )
        current = _ConfigClients(config_ref=config_ref)
        _CLIENT_POOLS[config_id] = current
        return current


def _same_connection(
    left: EffectiveLLMProfile,
    right: EffectiveLLMProfile,
) -> bool:
    return left.api_key == right.api_key and left.api_base == right.api_base


def get_llm_client(
    config: TwinRAGConfig,
    kind: LLMProfileKind | str,
    *,
    client_factory: _ClientFactory = AsyncOpenAI,
) -> Any:
    """Return a reusable client for an effective chat or indexing profile."""
    profile = resolve_llm_profile(config, kind)
    pool = _pool_for(config)

    with _CLIENT_POOLS_LOCK:
        existing = pool.clients.get(profile.kind)
        if existing is not None and existing.injected:
            return existing.client
        if (
            existing is not None
            and existing.factory is client_factory
            and _same_connection(existing.profile, profile)
        ):
            existing.profile = profile
            return existing.client

        # When indexing fully falls back to the chat connection (or vice versa),
        # reuse that transport too. Model selection happens per request.
        for candidate in pool.clients.values():
            if (
                not candidate.injected
                and candidate.factory is client_factory
                and _same_connection(candidate.profile, profile)
            ):
                pool.clients[profile.kind] = _ClientEntry(
                    profile=profile,
                    factory=client_factory,
                    client=candidate.client,
                )
                return candidate.client

        client = client_factory(
            api_key=profile.api_key,
            base_url=profile.api_base,
            max_retries=0,
        )
        pool.clients[profile.kind] = _ClientEntry(
            profile=profile,
            factory=client_factory,
            client=client,
        )
        return client


def is_transient_llm_error(exc: BaseException) -> bool:
    """Return whether an OpenAI-compatible failure is safe to retry."""
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


async def with_llm_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    profile: LLMProfileKind | str,
    max_attempts: int,
    base_seconds: float,
    max_seconds: float,
    jitter_ratio: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_value: Callable[[], float] = random.random,
) -> _T:
    """Run an LLM operation under bounded, typed retry semantics."""
    profile_kind = LLMProfileKind(profile)
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_llm_error(exc):
                raise

            backoff = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
            delay = min(
                max_seconds,
                backoff + (backoff * jitter_ratio * random_value()),
            )
            logger.warning(
                "Transient LLM failure; retrying profile=%s error_type=%s "
                "attempt=%d/%d delay_ms=%d",
                profile_kind.value,
                type(exc).__name__,
                attempt,
                max_attempts,
                round(delay * 1000),
                extra={
                    "llm_profile": profile_kind.value,
                    "exception_type": type(exc).__name__,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )
            await sleep(delay)

    raise AssertionError("LLM retry loop exited without a result")


async def create_chat_completion(
    config: TwinRAGConfig,
    kind: LLMProfileKind | str,
    *,
    client_factory: _ClientFactory = AsyncOpenAI,
    **request: Any,
) -> Any:
    """Create one chat completion with the resolved model and common retry."""
    if "model" in request:
        raise TypeError("model is resolved by the LLM profile")

    profile = resolve_llm_profile(config, kind)
    client = get_llm_client(config, profile.kind, client_factory=client_factory)

    async def _request() -> Any:
        # Resolve correlation inside every provider attempt. The shared client
        # remains context-free, so concurrent requests cannot inherit headers
        # captured by whichever request created the cached transport first.
        from ..server.tracing import make_trace_headers, trace_l3_llm_call

        provider_request = dict(request)
        trace_headers = make_trace_headers()
        if trace_headers:
            trace_header_names = {name.casefold() for name in trace_headers}
            merged_headers = {
                name: value
                for name, value in dict(
                    provider_request.get("extra_headers") or {}
                ).items()
                if str(name).casefold() not in trace_header_names
            }
            merged_headers.update(trace_headers)
            provider_request["extra_headers"] = merged_headers

        async def _provider_call() -> Any:
            return await client.chat.completions.create(
                model=profile.model,
                **provider_request,
            )

        return await trace_l3_llm_call(_provider_call)

    response = await with_llm_retry(
        _request,
        profile=profile.kind,
        max_attempts=config.llm_retry_max_attempts,
        base_seconds=config.llm_retry_base_seconds,
        max_seconds=config.llm_retry_max_seconds,
        jitter_ratio=config.llm_retry_jitter_ratio,
    )
    # Streaming responses expose ``delta`` chunks rather than a materialised
    # ``message``.  Size/citation validation is owned incrementally by the
    # OBSERVE phase, which also enforces the same bounded character ceiling.
    if request.get("stream"):
        return response
    content = response.choices[0].message.content
    if isinstance(content, str) and len(content) > _MAX_LLM_RESPONSE_CHARS:
        raise LLMResponseTooLargeError(
            "LLM response exceeded the bounded character limit"
        )
    return response


def log_llm_fallback(
    target_logger: logging.Logger,
    operation: str,
    exc: BaseException,
    **extra: object,
) -> None:
    """Log a business fallback without rendering exception or credential text."""
    error_type = type(exc).__name__
    target_logger.error(
        "%s failed; using business fallback (error_type=%s)",
        operation,
        error_type,
        extra={"exception_type": error_type, **extra},
    )


def inject_llm_client_for_testing(
    config: TwinRAGConfig,
    kind: LLMProfileKind | str,
    client: Any,
) -> None:
    """Install a deterministic client until the matching reset call."""
    profile = resolve_llm_profile(config, kind)
    pool = _pool_for(config)
    with _CLIENT_POOLS_LOCK:
        pool.clients[profile.kind] = _ClientEntry(
            profile=profile,
            factory=None,
            client=client,
            injected=True,
        )


def reset_llm_clients_for_testing(config: TwinRAGConfig | None = None) -> None:
    """Forget injected/cached clients globally or for one configuration."""
    with _CLIENT_POOLS_LOCK:
        if config is None:
            _CLIENT_POOLS.clear()
            return
        current = _CLIENT_POOLS.get(id(config))
        if current is not None and current.config_ref() is config:
            _CLIENT_POOLS.pop(id(config), None)
