"""Contract tests for GET /health on the Twin overlay router."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import webui_router


@pytest.fixture(autouse=True)
def _folder_env(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": "default", "label": "Default", "kind": "primary"}]),
    )
    webui_router.reset_store()
    yield
    _twindb_state.pop("rag", None)
    webui_router.reset_store()


async def _client():
    app = FastAPI()
    app.include_router(webui_router.router)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestTwinHealthEndpoint:
    async def test_reports_ok_when_rag_is_captured(self):
        _twindb_state["rag"] = object()
        async with await _client() as client:
            r = await client.get("/health")

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["ragCaptured"] is True
        assert body["folder"] == "default"
        assert body["stores"]["tags"] == "InMemoryTagStore"

    async def test_reports_degraded_when_rag_is_missing(self):
        async with await _client() as client:
            r = await client.get("/health")

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["ragCaptured"] is False
