"""Startup API wiring sanity logs.

Operators only have Elastic in BNP, so critical route wiring must be visible
at boot instead of requiring shell access to inspect ``app.routes``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from twindb_lightrag_memgraph.server.api_wiring import (
    ApiWiringProbe,
    api_wiring_probes,
    log_api_wiring_sanity,
)
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        memgraph_uri="bolt://localhost:7687",
        host="127.0.0.1",
        port=9622,
        api_key="t",
        llm_binding_api_key="t",
        embedding_binding_api_key="t",
    )


def test_static_twin_mount_does_not_pass_api_wiring_sanity(tmp_path, caplog):
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    app = FastAPI()
    app.mount("/twin", StaticFiles(directory=str(tmp_path)), name="twin-ui")

    caplog.set_level(
        logging.INFO,
        logger="twindb_lightrag_memgraph.server.api_wiring",
    )
    missing = log_api_wiring_sanity(app, surface="static-only")

    assert {probe.label for probe in missing} == {
        probe.label for probe in api_wiring_probes()
    }
    assert "🚨 API CHECK FAILED ❌ surface=static-only" in caplog.text
    assert "action=route_wiring_broken" in caplog.text
    assert "POST /twin/api/settings/api-keys" in caplog.text


def test_hidden_included_route_passes_api_wiring_sanity(caplog):
    router = APIRouter()

    @router.get("/catalog-profile", include_in_schema=False)
    async def catalog_profile():
        return {}

    parent = APIRouter()
    parent.include_router(router, prefix="/api")
    app = FastAPI()
    app.include_router(parent, prefix="/twin")
    probe = ApiWiringProbe("GET", "/twin/api/catalog-profile", "catalog-profile:read")
    caplog.set_level(
        logging.INFO,
        logger="twindb_lightrag_memgraph.server.api_wiring",
    )

    missing = log_api_wiring_sanity(
        app, probes=(probe,), surface="hidden-internal-route"
    )

    assert missing == []
    assert "/twin/api/catalog-profile" not in app.openapi()["paths"]
    assert "All API Check passes ✅☀️ surface=hidden-internal-route" in caplog.text


def test_route_introspection_failure_logs_the_real_cause(caplog):
    class BrokenIncludedRouter:
        def effective_candidates(self):
            raise RuntimeError("unsupported FastAPI route wrapper")

    app = SimpleNamespace(router=SimpleNamespace(routes=[BrokenIncludedRouter()]))
    probe = ApiWiringProbe("GET", "/twin/api/health", "health")
    caplog.set_level(
        logging.INFO,
        logger="twindb_lightrag_memgraph.server.api_wiring",
    )

    missing = log_api_wiring_sanity(
        app, probes=(probe,), surface="broken-introspection"
    )

    assert missing == [probe]
    assert "action=route_introspection_failed" in caplog.text
    assert "unsupported FastAPI route wrapper" in caplog.text
    assert "action=route_wiring_broken" in caplog.text


def test_standalone_app_logs_api_wiring_ok(caplog):
    caplog.set_level(
        logging.INFO,
        logger="twindb_lightrag_memgraph.server.api_wiring",
    )

    create_app(_settings())

    assert "All API Check passes ✅☀️ surface=standalone" in caplog.text
    assert "POST /twin/api/query" in caplog.text
    assert "GET /twin/api/folders" in caplog.text
    assert "POST /twin/api/settings/api-keys" in caplog.text
    assert "GET /twin/api/quota" in caplog.text
    assert "GET /twin/api/system/about" in caplog.text


def test_overlay_logs_api_wiring_ok(caplog):
    import twindb_lightrag_memgraph as t

    app = FastAPI()
    caplog.set_level(
        logging.INFO,
        logger="twindb_lightrag_memgraph.server.api_wiring",
    )

    t._mount_twin_subapp(app, "/twin/api", webui_stores="seed")

    assert "All API Check passes ✅☀️ surface=overlay:/twin/api" in caplog.text
    assert "POST /twin/api/query" in caplog.text
    assert "GET /twin/api/folders" in caplog.text
    assert "POST /twin/api/settings/api-keys" in caplog.text
    assert "GET /twin/api/quota" in caplog.text
    # The probe list is maintained by hand and has drifted in production;
    # router-level parity does not cover this mount point.
    assert "GET /twin/api/system/about" in caplog.text
