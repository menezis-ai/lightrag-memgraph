"""Regression: the production overlay mounts api-keys + quota.

The BNP overlay entrypoint (`register(mount_server=True)`) builds its
Twin surface in `_mount_twin_subapp` — a hand-maintained router list
that diverged from `server.app.create_app()`: `api_key_router` and
`quota_router` were wired into the standalone factory but never the
overlay, so `POST /twin/api/settings/api-keys` fell through to the
`/twin` static mount and returned 404/405 in production (verified on
lightrag 1.4.9.11, the BNP target).

This guard is **behavioural** (TestClient), not introspective. Reading
`route.path` off `app.routes` is NOT reliable across FastAPI versions
(0.137 wraps included routers so the attribute is absent) — the prior
introspection-based guard passed on a local FastAPI yet was meaningless
on the prod one. Asserting the route *responds* (≠ 404) is version-proof.

No Memgraph required: POST with an empty body is rejected at the
validation layer (422) before any store call, which already proves the
route exists and is wired. Quota's snapshot answers 200 with no limit
configured.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _overlay_client() -> TestClient:
    """Mount the production overlay on a bare app (seed stores → no
    Memgraph lifespan) exactly as `_mount_twin_subapp` does in prod."""
    import twindb_lightrag_memgraph as t

    app = FastAPI()
    t._mount_twin_subapp(app, "/twin/api", webui_stores="seed")
    return TestClient(app, raise_server_exceptions=False)


class TestOverlayMountsApiKeys:
    def test_create_api_key_post_is_mounted_not_404(self):
        # The exact request the WebUI "Create API key" button sends.
        # Before the fix this hit the /twin static mount → 404/405.
        # 422 (empty body fails validation) proves the route is mounted
        # and wired, without needing Memgraph.
        resp = _overlay_client().post("/twin/api/settings/api-keys", json={})
        assert resp.status_code != 404, "api-keys POST not mounted in overlay"
        assert resp.status_code != 405, "api-keys POST shadowed by static mount"
        assert resp.status_code == 422

    def test_list_route_is_mounted(self):
        # GET list: a mounted route never 404s (200 with a reachable store,
        # 500 without — both prove it exists; 404 would mean it fell through
        # to the static mount). DELETE/{id} ships in the SAME atomic
        # include_router, so GET + POST mounting implies it is mounted too —
        # and we must NOT assert DELETE != 404, because a DELETE on an unknown
        # key is a *legitimate* 404 ("not found") once the store is reachable.
        assert _overlay_client().get("/twin/api/settings/api-keys").status_code != 404


class TestOverlayMountsQuota:
    def test_quota_snapshot_is_public_and_mounted(self):
        # Public route the QuotaBanner polls; 200 with no limit configured.
        resp = _overlay_client().get("/twin/api/quota")
        assert resp.status_code == 200


class TestOverlayNonRegression:
    def test_existing_overlay_routes_unaffected(self):
        c = _overlay_client()
        assert c.get("/twin/api/folders").status_code == 200
        assert c.get("/twin/api/tags").status_code == 200
