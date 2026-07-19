"""HTTP + provider tests for /twin/api/settings/vision.

Covers: env-default GET on a fresh workspace, PUT → runtime persistence,
provider precedence inside ``_vision._effective_settings`` (a PUT applies
to the pipeline without restart), validation (422), admin/auth gating, and
activity emission. HTTP flows need Memgraph (integration marker); the
provider-precedence unit tests run offline.
"""

from __future__ import annotations

import secrets

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _vision
from twindb_lightrag_memgraph.server import vision_settings_store, webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp
from twindb_lightrag_memgraph.server.vision_settings_routes import (
    router as vision_settings_router,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    for var in ("TWIN_VISION_MIN_OCR_CHARS", "TWIN_VISION_DROP_CLASSES"):
        monkeypatch.delenv(var, raising=False)
    configure_idp(None)
    _vision.reset_caches()
    yield
    configure_idp(None)
    configure_auth(api_key=None, jwt_secret=None)
    _vision.reset_caches()


# ---------------------------------------------------------------------------
# Provider precedence (offline)
# ---------------------------------------------------------------------------


async def test_effective_settings_env_defaults_without_provider(monkeypatch):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "33")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", "logo,stamp")
    threshold, classes = await _vision._effective_settings()
    assert threshold == 33
    assert classes == frozenset({"logo", "stamp"})


async def test_effective_settings_provider_wins_over_env(monkeypatch):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "33")

    async def provider():
        return {"min_ocr_chars": 0, "drop_classes": ["Logo", " signature "]}

    _vision.set_settings_provider(provider)
    threshold, classes = await _vision._effective_settings()
    assert threshold == 0
    assert classes == frozenset({"logo", "signature"})


async def test_effective_settings_partial_or_broken_provider_falls_back(
    monkeypatch,
):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "42")

    async def partial():
        return {"drop_classes": ["logo"]}  # no min_ocr_chars

    _vision.set_settings_provider(partial)
    threshold, classes = await _vision._effective_settings()
    assert threshold == 42  # env fallback per-field
    assert classes == frozenset({"logo"})

    async def boom():
        raise RuntimeError("store down")

    _vision.set_settings_provider(boom)
    threshold, classes = await _vision._effective_settings()
    assert threshold == 42
    assert classes == _vision.drop_classes()


# ---------------------------------------------------------------------------
# Router gating (offline)
# ---------------------------------------------------------------------------


def test_vision_settings_router_rejects_anonymous_when_mounted_directly():
    configure_auth(api_key="root-secret")
    app = FastAPI()
    app.include_router(vision_settings_router, prefix="/twin/api")

    assert TestClient(app).get("/twin/api/settings/vision").status_code == 401
    assert (
        TestClient(app)
        .put(
            "/twin/api/settings/vision",
            json={"min_ocr_chars": 0, "drop_classes": []},
        )
        .status_code
        == 401
    )


# ---------------------------------------------------------------------------
# HTTP flow (Memgraph required)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(monkeypatch):
    monkeypatch.setenv("WORKSPACE", f"vision_settings_{secrets.token_hex(4)}")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-infra-root")
    webui_router.reset_store()
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    ) as c:
        yield c
    from twindb_lightrag_memgraph._constants import resolve_workspace

    try:
        await vision_settings_store.reset_workspace(resolve_workspace())
    except Exception:
        pass
    webui_router.reset_store()


@pytest.mark.integration
class TestVisionSettingsRoutes:
    async def test_fresh_workspace_returns_env_defaults(self, client):
        r = await client.get("/twin/api/settings/vision")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "env-default"
        assert body["min_ocr_chars"] == _vision.DEFAULT_MIN_OCR_CHARS
        assert set(body["drop_classes"]) == _vision.DEFAULT_DROP_CLASSES

    async def test_put_persists_and_get_reflects_runtime(self, client):
        r = await client.put(
            "/twin/api/settings/vision",
            json={"min_ocr_chars": 5, "drop_classes": ["Logo", "watermark"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "runtime"
        assert body["min_ocr_chars"] == 5
        assert body["drop_classes"] == ["logo", "watermark"]
        assert body["updated_by"]

        r = await client.get("/twin/api/settings/vision")
        assert r.json()["source"] == "runtime"
        assert r.json()["min_ocr_chars"] == 5

    async def test_put_applies_to_pipeline_without_restart(self, client):
        """create_app installed the provider: a PUT must be visible to
        ``_vision._effective_settings`` immediately (per-image re-read)."""
        await client.put(
            "/twin/api/settings/vision",
            json={"min_ocr_chars": 0, "drop_classes": ["diagram"]},
        )
        threshold, classes = await _vision._effective_settings()
        assert threshold == 0
        assert classes == frozenset({"diagram"})

    async def test_put_rejects_invalid_values(self, client):
        r = await client.put(
            "/twin/api/settings/vision",
            json={"min_ocr_chars": -1, "drop_classes": []},
        )
        assert r.status_code == 422

        r = await client.put(
            "/twin/api/settings/vision",
            json={"min_ocr_chars": 0, "drop_classes": ["../etc"]},
        )
        assert r.status_code == 422

    async def test_put_emits_activity_event(self, client):
        await client.put(
            "/twin/api/settings/vision",
            json={"min_ocr_chars": 10, "drop_classes": ["logo"]},
        )
        raw = (await client.get("/twin/api/activity")).json()
        items = raw.get("items", raw if isinstance(raw, list) else [])
        evt = items[0]
        assert evt["kind"] == "vision-settings-updated"
        assert evt["meta"]["min_ocr_chars"] == 10
        assert evt["meta"]["drop_classes"] == ["logo"]
