"""HTTP + provider tests for /twin/api/settings/vision.

Covers: env-default GET on a fresh workspace, PUT → runtime persistence,
provider precedence inside the vision and procedure pipelines (a PUT
applies without restart), validation (422/409), admin/auth gating, and
activity emission. HTTP flows need Memgraph (integration marker); the
provider-precedence unit tests run offline.
"""

from __future__ import annotations

import secrets

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _procedure, _vision
from twindb_lightrag_memgraph.server import (
    vision_settings_routes,
    vision_settings_store,
    webui_router,
)
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp
from twindb_lightrag_memgraph.server.vision_settings_routes import (
    router as vision_settings_router,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    for var in (
        "TWIN_PROCEDURE",
        "TWIN_VISION_MIN_OCR_CHARS",
        "TWIN_VISION_DROP_CLASSES",
    ):
        monkeypatch.delenv(var, raising=False)
    configure_idp(None)
    _procedure.reset_caches()
    _vision.reset_caches()
    yield
    configure_idp(None)
    configure_auth(api_key=None, jwt_secret=None)
    _procedure.reset_caches()
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


async def test_procedure_runtime_activation_overrides_deployment_default(
    monkeypatch,
):
    monkeypatch.setattr(_procedure, "is_available", lambda: True)
    monkeypatch.setenv("TWIN_PROCEDURE", "off")

    async def enabled():
        return {"procedure_enabled": True}

    _procedure.set_settings_provider(enabled)
    assert await _procedure.is_effectively_enabled() is True

    async def disabled():
        return {"procedure_enabled": False}

    _procedure.set_settings_provider(disabled)
    assert await _procedure.is_effectively_enabled() is False

    monkeypatch.delenv("TWIN_PROCEDURE")

    async def boom():
        raise RuntimeError("store down")

    _procedure.set_settings_provider(boom)
    assert await _procedure.is_effectively_enabled() is False


async def test_procedure_admin_surface_defaults_off_without_explicit_opt_in(
    monkeypatch,
):
    monkeypatch.setattr(_procedure, "is_available", lambda: True)

    async def no_runtime_choice():
        return None

    _procedure.set_settings_provider(no_runtime_choice)
    assert await _procedure.is_effectively_enabled() is False

    monkeypatch.setenv("TWIN_PROCEDURE", "on")
    assert await _procedure.is_effectively_enabled() is True


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


def test_vision_settings_openapi_documents_toggle_and_failures():
    app = FastAPI()
    app.include_router(vision_settings_router, prefix="/twin/api")
    schema = app.openapi()
    path = schema["paths"]["/twin/api/settings/vision"]

    assert path["get"]["summary"] == "Read the image and procedure ingestion settings"
    assert {"200", "401"} <= set(path["get"]["responses"])
    assert {"200", "401", "403", "409", "422"} <= set(path["put"]["responses"])

    public_schema = schema["components"]["schemas"]["VisionSettingsPublic"]
    assert public_schema["properties"]["procedure_enabled"]["description"]
    assert public_schema["properties"]["procedure_available"]["description"]
    assert public_schema["properties"]["source"]["enum"] == [
        "runtime",
        "env-default",
    ]

    input_schema = schema["components"]["schemas"]["VisionSettings"]
    assert "procedure_enabled" in input_schema["properties"]
    assert "procedure_enabled" not in input_schema.get("required", [])


def test_admin_can_persist_procedure_toggle_without_restart(monkeypatch):
    configure_auth(api_key="root-secret")
    captured = {}

    async def get_settings(_workspace):
        return None

    async def initialize(_workspace):
        return None

    async def update_settings(_workspace, **settings):
        captured.update(settings)
        return {
            **settings,
            "updated_at": 1_722_000_000_000,
        }

    async def emit_event(**_kwargs):
        return None

    monkeypatch.setattr(vision_settings_store, "get_settings", get_settings)
    monkeypatch.setattr(vision_settings_store, "initialize", initialize)
    monkeypatch.setattr(vision_settings_store, "update_settings", update_settings)
    monkeypatch.setattr(vision_settings_routes, "_emit_event", emit_event)
    monkeypatch.setattr(_procedure, "is_available", lambda: True)
    monkeypatch.setattr(_procedure, "is_enabled", lambda: False)

    app = FastAPI()
    app.include_router(vision_settings_router, prefix="/twin/api")
    response = TestClient(app).put(
        "/twin/api/settings/vision",
        headers={"Authorization": "Bearer root-secret"},
        json={
            "min_ocr_chars": 12,
            "drop_classes": ["logo"],
            "procedure_enabled": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["procedure_enabled"] is True
    assert captured["procedure_enabled"] is True


def test_legacy_client_put_preserves_existing_procedure_choice(monkeypatch):
    configure_auth(api_key="root-secret")
    captured = {}

    async def get_settings(_workspace):
        return {
            "min_ocr_chars": 10,
            "drop_classes": ["logo"],
            "procedure_enabled": True,
        }

    async def initialize(_workspace):
        return None

    async def update_settings(_workspace, **settings):
        captured.update(settings)
        return {
            **settings,
            "updated_at": 1_722_000_000_000,
        }

    async def emit_event(**_kwargs):
        return None

    monkeypatch.setattr(vision_settings_store, "get_settings", get_settings)
    monkeypatch.setattr(vision_settings_store, "initialize", initialize)
    monkeypatch.setattr(vision_settings_store, "update_settings", update_settings)
    monkeypatch.setattr(vision_settings_routes, "_emit_event", emit_event)
    monkeypatch.setattr(_procedure, "is_available", lambda: True)

    app = FastAPI()
    app.include_router(vision_settings_router, prefix="/twin/api")
    response = TestClient(app).put(
        "/twin/api/settings/vision",
        headers={"Authorization": "Bearer root-secret"},
        # Older clients know only these two fields.
        json={"min_ocr_chars": 20, "drop_classes": ["signature"]},
    )

    assert response.status_code == 200
    assert captured["procedure_enabled"] is True


def test_admin_cannot_enable_procedures_without_prerequisites(monkeypatch):
    configure_auth(api_key="root-secret")

    async def get_settings(_workspace):
        return None

    monkeypatch.setattr(vision_settings_store, "get_settings", get_settings)
    monkeypatch.setattr(_procedure, "is_available", lambda: False)
    monkeypatch.setattr(_procedure, "is_enabled", lambda: False)

    app = FastAPI()
    app.include_router(vision_settings_router, prefix="/twin/api")
    response = TestClient(app).put(
        "/twin/api/settings/vision",
        headers={"Authorization": "Bearer root-secret"},
        json={
            "min_ocr_chars": 12,
            "drop_classes": ["logo"],
            "procedure_enabled": True,
        },
    )

    assert response.status_code == 409
    assert "prerequisites" in response.json()["detail"]


# ---------------------------------------------------------------------------
# HTTP flow (Memgraph required)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(monkeypatch):
    monkeypatch.setenv("WORKSPACE", f"vision_settings_{secrets.token_hex(4)}")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-infra-root")
    monkeypatch.setattr(_procedure, "is_available", lambda: True)
    monkeypatch.setattr(_procedure, "is_enabled", lambda: False)
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
        assert body["procedure_enabled"] is False
        assert body["procedure_available"] is True

    async def test_put_persists_and_get_reflects_runtime(self, client):
        r = await client.put(
            "/twin/api/settings/vision",
            json={
                "min_ocr_chars": 5,
                "drop_classes": ["Logo", "watermark"],
                "procedure_enabled": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "runtime"
        assert body["min_ocr_chars"] == 5
        assert body["drop_classes"] == ["logo", "watermark"]
        assert body["procedure_enabled"] is True
        assert body["procedure_available"] is True
        assert body["updated_by"]

        r = await client.get("/twin/api/settings/vision")
        assert r.json()["source"] == "runtime"
        assert r.json()["min_ocr_chars"] == 5

    async def test_put_applies_to_pipeline_without_restart(self, client):
        """create_app installed the provider: a PUT must be visible to
        ``_vision._effective_settings`` immediately (per-image re-read)."""
        await client.put(
            "/twin/api/settings/vision",
            json={
                "min_ocr_chars": 0,
                "drop_classes": ["diagram"],
                "procedure_enabled": True,
            },
        )
        threshold, classes = await _vision._effective_settings()
        assert threshold == 0
        assert classes == frozenset({"diagram"})
        assert await _procedure.is_effectively_enabled() is True

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
            json={
                "min_ocr_chars": 10,
                "drop_classes": ["logo"],
                "procedure_enabled": False,
            },
        )
        raw = (await client.get("/twin/api/activity")).json()
        items = raw.get("items", raw if isinstance(raw, list) else [])
        evt = items[0]
        assert evt["kind"] == "vision-settings-updated"
        assert evt["meta"]["min_ocr_chars"] == 10
        assert evt["meta"]["drop_classes"] == ["logo"]
        assert evt["meta"]["procedure_enabled"] is False

    async def test_put_rejects_enabling_when_prerequisites_are_missing(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(_procedure, "is_available", lambda: False)
        r = await client.put(
            "/twin/api/settings/vision",
            json={
                "min_ocr_chars": 10,
                "drop_classes": ["logo"],
                "procedure_enabled": True,
            },
        )
        assert r.status_code == 409
        assert "prerequisites" in r.json()["detail"]
