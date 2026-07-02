"""Mock-kill safeguards — regression for findings F5 + F6.

Covers the safeguards landed during the 2026-06-04 mock-kill hardening:

- **F5**: ``_mount_twin_subapp`` emits a loud WARNING when the operator
  enables the IdP middleware (``TWIN_IDP_JWKS_URL`` set) but leaves
  ``webui_stores="seed"`` — a strong signal of a real deployment that
  would otherwise serve in-memory demo fixtures.

- **F6**: the WebuiStore ``for_folder`` factory's ``mode="memgraph"``
  branch is wired into both the in-process Twin sub-app path
  (``register()`` mounting) and the standalone ``server/app.py`` lifespan,
  so the default folder never silently exposes the demo
  ``DOCUMENTS`` / ``GRAPH_ENTITIES`` via ``/twin/api/documents`` and
  ``/twin/api/graph/*``.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import (
    _mount_twin_subapp,
    _mount_twin_ui,
    _replace_webui_mount,
)
from twindb_lightrag_memgraph.server import idp_jwt, webui_router


@pytest.fixture(autouse=True)
def _reset_idp_and_store():
    yield
    idp_jwt.configure_idp(None)
    webui_router.reset_store()


# ---------------------------------------------------------------------------
# F5 — boot WARN when IdP active + webui_stores="seed"
# ---------------------------------------------------------------------------


class TestSeedStoresWarningUnderActiveIdp:
    """The combination "real IdP + demo stores" is a trap: visible
    fixtures look like real data until the next restart erases them.
    Emit a loud WARN so the operator can fix the runbook before going
    live.
    """

    def test_warn_emitted_when_idp_active_and_seed_mode(self, monkeypatch, caplog):
        monkeypatch.setenv("TWIN_IDP_JWKS_URL", "https://idp.example/jwks")
        app = FastAPI()
        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            _mount_twin_subapp(app, prefix="/twin/api", webui_stores="seed")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        relevant = [r for r in warnings if "DEMO STORES IN PROD" in r.getMessage()]
        assert relevant, (
            "Expected DEMO STORES IN PROD warning when IdP is active and "
            "webui_stores='seed'; saw: "
            + repr([r.getMessage() for r in caplog.records])
        )

    def test_no_warn_when_idp_dormant(self, monkeypatch, caplog):
        # Dev / standalone demo path — no IdP, seed mode is
        # the legitimate demo backend.
        monkeypatch.delenv("TWIN_IDP_JWKS_URL", raising=False)
        app = FastAPI()
        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            _mount_twin_subapp(app, prefix="/twin/api", webui_stores="seed")
        assert not any("DEMO STORES IN PROD" in r.getMessage() for r in caplog.records)

    def test_no_warn_when_idp_active_but_memgraph_mode(self, monkeypatch, caplog):
        monkeypatch.setenv("TWIN_IDP_JWKS_URL", "https://idp.example/jwks")
        app = FastAPI()
        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            # `memgraph` mode in this context tries to call the async
            # store factories during lifespan setup, which we don't
            # exercise here — we only assert no WARN about DEMO STORES.
            # An exception or error log unrelated to F5 is acceptable.
            try:
                _mount_twin_subapp(app, prefix="/twin/api", webui_stores="memgraph")
            except Exception:
                pass
        assert not any("DEMO STORES IN PROD" in r.getMessage() for r in caplog.records)

    def test_register_default_webui_stores_is_memgraph(self):
        import inspect

        import twindb_lightrag_memgraph as pkg

        signature = inspect.signature(pkg.register)
        assert signature.parameters["webui_stores"].default == "memgraph"


class TestStandaloneDefaults:
    def test_webui_backends_default_to_memgraph(self):
        from twindb_lightrag_memgraph.server.settings import (
            LightRAGServerSettings,
        )

        settings = LightRAGServerSettings()
        assert settings.webui_tag_backend == "memgraph"
        assert settings.webui_activity_backend == "memgraph"
        assert settings.webui_notifications_backend == "memgraph"


# ---------------------------------------------------------------------------
# F7 — runtime never exposes MSW from a polluted Twin dist
# ---------------------------------------------------------------------------


def _write_twin_dist_with_msw(tmp_path):
    dist = tmp_path / "twin-dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<html><head></head><body>__TWIN_CONFIG_JSON__</body></html>",
        encoding="utf-8",
    )
    (dist / "mockServiceWorker.js").write_text(
        "self.__MSW_PROBE__ = true;",
        encoding="utf-8",
    )
    return dist


class TestMockServiceWorkerRuntimeExposure:
    async def test_twin_ui_mount_does_not_serve_mock_service_worker(self, tmp_path):
        dist = _write_twin_dist_with_msw(tmp_path)
        app = FastAPI()
        _mount_twin_ui(app, str(dist), prefix="/twin")

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            root_response = await client.get("/mockServiceWorker.js")
            mounted_response = await client.get("/twin/mockServiceWorker.js")

        assert root_response.status_code == 404
        assert mounted_response.status_code == 404
        assert "MSW_PROBE" not in root_response.text
        assert "MSW_PROBE" not in mounted_response.text

    async def test_replaced_webui_mount_does_not_serve_mock_service_worker(
        self, tmp_path
    ):
        native_dist = tmp_path / "native-dist"
        native_dist.mkdir()
        (native_dist / "index.html").write_text(
            "<html>native</html>",
            encoding="utf-8",
        )
        dist = _write_twin_dist_with_msw(tmp_path)
        app = FastAPI()
        app.mount(
            "/webui",
            StaticFiles(directory=str(native_dist), html=True),
            name="webui",
        )

        _replace_webui_mount(app, str(dist))

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            root_response = await client.get("/mockServiceWorker.js")
            mounted_response = await client.get("/webui/mockServiceWorker.js")

        assert root_response.status_code == 404
        assert mounted_response.status_code == 404
        assert "MSW_PROBE" not in root_response.text
        assert "MSW_PROBE" not in mounted_response.text


# ---------------------------------------------------------------------------
# F6 — `for_folder(default, mode="memgraph")` integration through
# `_mount_twin_subapp(webui_stores="seed")` doesn't apply, so we exercise
# the contract directly here.
# ---------------------------------------------------------------------------


class TestForFolderMemgraphModeIntegration:
    """Module-level integration alongside the per-class unit tests in
    ``test_webui_router.TestForFolderMode`` — confirms the seed leak is
    closed from both the in-process Twin sub-app and the
    ``server/app.py`` standalone lifespan call sites."""

    def test_register_path_uses_memgraph_mode(self):
        """The overlay memgraph branch calls `for_folder(..., mode="memgraph")`.

        We assert the wiring by reading the source — a guard against a future
        refactor accidentally dropping the keyword arg. The per-folder store
        init lives in `_init_overlay_memgraph_stores` (extracted from
        `_mount_twin_subapp` by the cognitive-complexity refactor).
        """
        import inspect

        from twindb_lightrag_memgraph.patches.registry import (
            _init_overlay_memgraph_stores as fn,
        )

        source = inspect.getsource(fn)
        assert '.for_folder(folder.id, mode="memgraph")' in source, (
            "register() memgraph branch must pass mode='memgraph' to "
            "for_folder — see mock-kill F6."
        )

    def test_app_lifespan_uses_memgraph_mode(self):
        """`server/app.py` WebUI backend init must pass `mode="memgraph"`
        when any *_backend setting is 'memgraph'. Same guard rationale.

        The call lives in `_init_webui_backends`, invoked from the lifespan
        built by `_build_lifespan` (extracted from `create_app` by the
        cognitive-complexity refactor)."""
        import inspect

        from twindb_lightrag_memgraph.server.app import _init_webui_backends

        source = inspect.getsource(_init_webui_backends)
        assert 'WebuiStore.for_folder(folder.id, mode="memgraph")' in source, (
            "server/app.py _init_webui_backends must pass mode='memgraph' to "
            "for_folder when memgraph backends are configured — see "
            "mock-kill F6."
        )

    def test_app_route_imports_fail_fast(self):
        """Internal route modules are mandatory, not optional plugins.

        Swallowing ImportError here lets the server boot while dropping
        /twin/api/query, API-key management, or quota routes from the live
        surface. That is worse than a startup failure because operators see a
        partially healthy app.
        """
        import inspect

        from twindb_lightrag_memgraph.server.app import create_app

        source = inspect.getsource(create_app)
        assert "except ImportError" not in source
