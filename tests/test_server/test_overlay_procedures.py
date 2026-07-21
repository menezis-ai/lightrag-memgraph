"""Regression: the production overlay mounts the procedure approval routes.

Same failure class as ``test_overlay_vision_settings.py`` /
``test_overlay_api_keys_quota.py``: ``_mount_twin_subapp`` is a
hand-maintained router list that can diverge from ``server.app.create_app``.
A router wired only into the standalone factory is silently absent from the
BNP overlay surface (``register(mount_server=True)``).

Behavioural guard (TestClient), no Memgraph: the list responds through the
folder binding, a decision route is reachable (404 on an unknown bundle
proves routing + auth, not a static-mount fallthrough), and the overlay
mount installs the seam event sink.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from twindb_lightrag_memgraph import _procedure


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "TWIN_PROCEDURE_STORE_FILE", str(tmp_path / "bundles" / "store.json")
    )
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "f1")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": "f1", "label": "Folder 1", "kind": "kb"}]),
    )
    _procedure.reset_caches()
    yield
    _procedure.reset_caches()


def _overlay_client() -> TestClient:
    import twindb_lightrag_memgraph as t
    from twindb_lightrag_memgraph.server.auth import configure_auth

    app = FastAPI()
    t._mount_twin_subapp(app, "/twin/api", webui_stores="seed")
    configure_auth(api_key="test-infra-root")
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={
            "Authorization": "Bearer test-infra-root",
            "X-Twin-Folder": "f1",
        },
    )


class TestOverlayMountsProcedures:
    def test_list_is_mounted(self):
        resp = _overlay_client().get("/twin/api/procedures")
        assert resp.status_code != 404, "procedures list not mounted in overlay"
        assert resp.status_code == 200
        assert resp.json() == []

    def test_decision_routes_are_mounted(self):
        client = _overlay_client()
        # 404 on an UNKNOWN bundle proves the route matched and executed
        # (a static-mount fallthrough would 404 on the prefix itself, but
        # then the list above would 404 too; approve on ghost must be the
        # handler's own 404, with its detail).
        resp = client.post("/twin/api/procedures/ghost/approve")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "unknown bundle"
        resp = client.post("/twin/api/procedures/ghost/reject")
        assert resp.status_code == 409  # transition: unknown or wrong state
        resp = client.get("/twin/api/procedures/store/health")
        assert resp.status_code == 200
        assert resp.json()["degraded"] is False

    def test_mount_installs_seam_event_sink(self):
        _procedure.reset_caches()
        try:
            _overlay_client()
            assert _procedure._event_sink is not None, (
                "overlay mount must wire the procedure seam event sink "
                "(install_procedure_event_sink)"
            )
        finally:
            _procedure.reset_caches()
