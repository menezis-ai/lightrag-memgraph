"""Activity-event emission behavior for auth-denial paths.

The goal is to ensure auth denials stay on the hot path for response time
while activity writes are best-effort and single-sourced.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.activity_events import (
    emit_access_denied_event_background,
)
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        working_dir="/tmp/twindb-lightrag-activity-events-test",
        workspace="activity-events",
        enable_langsmith_tracing=False,
        enable_webui_routes=True,
        api_key="test-key",
        jwt_secret=None,
        webui_activity_backend="memory",
        webui_tag_backend="memory",
        webui_notifications_backend="memory",
    )


@pytest.fixture()
async def client():
    webui_router.reset_store()
    app = create_app(_settings())
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    webui_router.reset_store()


async def test_access_denied_response_is_not_blocked_by_slow_activity_store(
    client,
    monkeypatch,
):
    store = webui_router.get_store()
    started = asyncio.Event()
    unblock = asyncio.Event()

    async def slow_record_activity(_event):
        started.set()
        await unblock.wait()

    monkeypatch.setattr(store, "record_activity", slow_record_activity)

    response = await asyncio.wait_for(
        client.get(
            "/documents",
            headers={"X-Twin-Folder": "default"},
        ),
        timeout=0.5,
    )
    assert response.status_code == 401

    # The request must return before background work can unblock, while still
    # scheduling the denied-event write.
    await asyncio.sleep(0)
    assert started.is_set()

    # Cleanup pending background task.
    unblock.set()
    await asyncio.sleep(0)


async def test_failing_activity_write_does_not_change_auth_status(client, monkeypatch):
    store = webui_router.get_store()

    async def broken_record_activity(_event):
        raise RuntimeError("activity store unavailable")

    monkeypatch.setattr(store, "record_activity", broken_record_activity)

    response = await client.get(
        "/documents",
        headers={"X-Twin-Folder": "default"},
    )
    assert response.status_code == 401


async def test_access_denied_event_emission_is_deduplicated_on_same_request(
    monkeypatch,
):
    store = webui_router.get_store()
    calls: list[dict] = []

    async def delayed_record_activity(event):
        calls.append(event)

    monkeypatch.setattr(store, "record_activity", delayed_record_activity)

    async def receive():
        return {"type": "http.request"}

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/documents",
            "headers": [(b"x-twin-folder", b"default")],
        },
        receive=receive,
    )

    emit_access_denied_event_background(
        request,
        status_code=401,
        reason="unauthorized",
    )
    emit_access_denied_event_background(
        request,
        status_code=401,
        reason="unauthorized",
    )

    for _ in range(20):
        if len(calls) >= 1:
            break
        await asyncio.sleep(0.01)

    assert len(calls) == 1
