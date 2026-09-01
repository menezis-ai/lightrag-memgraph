"""LangSmith activation parity for the production LightRAG host overlay."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import Mock

from fastapi import FastAPI

from twindb_lightrag_memgraph.patches import registry
from twindb_lightrag_memgraph.server import tracing


async def test_overlay_applies_tracing_after_the_host_lifespan_starts(monkeypatch):
    events: list[str] = []
    rag = object()

    @asynccontextmanager
    async def host_lifespan(_: FastAPI) -> AsyncIterator[None]:
        events.append("host-started")
        try:
            yield
        finally:
            events.append("host-stopped")

    apply_tracing = Mock(side_effect=lambda value: events.append("tracing-applied"))
    monkeypatch.setenv("LIGHTRAG_ENABLE_LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracing, "apply_lang_with_tracing", apply_tracing)
    monkeypatch.setitem(registry._twindb_state, "rag", rag)
    app = FastAPI(lifespan=host_lifespan)

    registry._mount_twin_subapp(app, "/twin/api", webui_stores="seed")
    async with app.router.lifespan_context(app):
        assert events == ["host-started", "tracing-applied"]

    apply_tracing.assert_called_once_with(rag)
    assert events == ["host-started", "tracing-applied", "host-stopped"]


async def test_overlay_flag_off_preserves_the_native_lifespan(monkeypatch):
    events: list[str] = []

    @asynccontextmanager
    async def host_lifespan(_: FastAPI) -> AsyncIterator[None]:
        events.append("host-started")
        try:
            yield
        finally:
            events.append("host-stopped")

    apply_tracing = Mock()
    monkeypatch.delenv("LIGHTRAG_ENABLE_LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr(tracing, "apply_lang_with_tracing", apply_tracing)
    app = FastAPI(lifespan=host_lifespan)

    registry._mount_twin_subapp(app, "/twin/api", webui_stores="seed")
    async with app.router.lifespan_context(app):
        assert events == ["host-started"]

    apply_tracing.assert_not_called()
    assert events == ["host-started", "host-stopped"]
