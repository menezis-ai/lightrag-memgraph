"""Regression: the production overlay mounts the vision-settings routes.

Same failure class as ``test_overlay_api_keys_quota.py``: the BNP overlay
entrypoint (``register(mount_server=True)``) builds its Twin surface in
``_mount_twin_subapp`` — a hand-maintained router list that can diverge
from ``server.app.create_app()``. The vision-settings router was wired
into the standalone factory but not the overlay, so ``/twin/api/settings/
vision`` was absent from the live surface — caught in CI by the e2e
api-coverage battery ("admin operation missing from live surface:
PUT /twin/api/settings/vision"), fixed by mounting it in the overlay.

Behavioural guard (TestClient), no Memgraph: the GET degrades to env
defaults when the store is unreachable (200), and an empty-body PUT is
rejected at validation (422) before any store call.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from twindb_lightrag_memgraph import _vision


def _overlay_client() -> TestClient:
    import twindb_lightrag_memgraph as t
    from twindb_lightrag_memgraph.server.auth import configure_auth

    app = FastAPI()
    t._mount_twin_subapp(app, "/twin/api", webui_stores="seed")
    configure_auth(api_key="test-infra-root")
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={"Authorization": "Bearer test-infra-root"},
    )


class TestOverlayMountsVisionSettings:
    def test_get_is_mounted_and_degrades_to_env_defaults(self):
        resp = _overlay_client().get("/twin/api/settings/vision")
        assert resp.status_code != 404, "vision GET not mounted in overlay"
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "env-default"
        assert body["min_ocr_chars"] == _vision.DEFAULT_MIN_OCR_CHARS

    def test_put_is_mounted_not_404(self):
        # Empty body fails validation (422) before any store call — proves
        # the route is mounted and wired (404/405 = static-mount fallthrough).
        resp = _overlay_client().put("/twin/api/settings/vision", json={})
        assert resp.status_code != 404, "vision PUT not mounted in overlay"
        assert resp.status_code != 405, "vision PUT shadowed by static mount"
        assert resp.status_code == 422

    def test_mount_installs_runtime_settings_provider(self):
        _vision.reset_caches()
        try:
            _overlay_client()
            assert _vision._settings_provider is not None, (
                "overlay mount must wire the _vision runtime-settings "
                "provider (install_settings_provider)"
            )
        finally:
            _vision.reset_caches()
