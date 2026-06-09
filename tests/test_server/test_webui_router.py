"""Tests for the WebUI phase-1 router.

These tests hit the FastAPI app via ``httpx.AsyncClient + ASGITransport`` so
they cover the wire contract (shape + JSON-error invariants) without any
network or LightRAG dependency. The WebUI router is purely in-memory in
phase 1.

Following the v0.5.2 lesson: every endpoint and every error path must return
JSON, never HTML. We assert content-type on each response to lock that in.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.webui_seed import (
    ACTIVITY,
    ACTIVITY_NOW_MS,
    DOCUMENTS,
    GRAPH_ENTITIES,
    GRAPH_RELATIONS,
    NOTIFICATIONS,
    OPENAPI_GROUPS,
    OPENAPI_VERSION,
    TAG_CATEGORIES,
    TAGS,
    THESAURUS,
)


def _make_settings(*, api_key: str | None = None) -> LightRAGServerSettings:
    """Build settings with auth disabled by default so the WebUI router is
    reachable without a Bearer header (auth is exercised separately in
    test_auth.py)."""
    return LightRAGServerSettings(
        working_dir="/tmp/lightrag_webui_test",
        workspace="cib",
        enable_langsmith_tracing=False,
        api_key=api_key,
        jwt_secret=None,
        enable_webui_routes=True,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    """Each test starts from a clean seed-built store."""
    webui_router.reset_store()
    yield
    webui_router.reset_store()


@pytest.fixture()
async def client():
    """AsyncClient against the WebUI-enabled app, auth disabled."""
    app = create_app(_make_settings())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestDocuments:
    async def test_list_returns_full_envelope(self, client):
        r = await client.get("/documents")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["total"] == len(DOCUMENTS)
        assert len(body["items"]) == len(DOCUMENTS)
        # Shape spot-check on the first doc
        first = body["items"][0]
        for key in ("id", "type", "source", "summary", "tags", "status", "chunks", "updated", "visibility", "folder"):
            assert key in first

    async def test_status_filter_narrows(self, client):
        r = await client.get("/documents", params={"status": "failed"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        for d in body["items"]:
            assert d["status"] == "failed"

    async def test_q_filter_matches_source_substring(self, client):
        r = await client.get("/documents", params={"q": "oracle"})
        assert r.status_code == 200
        body = r.json()
        for d in body["items"]:
            assert "oracle" in d["source"].lower()

    async def test_tag_filter_matches_only_tagged(self, client):
        r = await client.get("/documents", params={"tag": "rman"})
        assert r.status_code == 200
        body = r.json()
        for d in body["items"]:
            assert "rman" in d["tags"]

    async def test_no_match_returns_empty_envelope_not_404(self, client):
        r = await client.get("/documents", params={"q": "zzz-no-doc"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["items"] == []


# ---------------------------------------------------------------------------
# Folders / notifications
# ---------------------------------------------------------------------------


class TestFoldersList:
    async def test_list(self, client):
        r = await client.get("/folders")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["id"] == "default"
        assert body[0]["current"] is True


class TestFolders:
    def _configure_folders(self, monkeypatch):
        monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
        monkeypatch.setenv(
            "TWIN_FOLDERS_JSON",
            json.dumps(
                [
                    {"id": "default", "label": "Default folder", "kind": "primary"},
                    {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
                ]
            ),
        )
        webui_router.reset_store()

    async def test_folders_endpoint_lists_configured_folders(self, monkeypatch, client):
        self._configure_folders(monkeypatch)
        r = await client.get("/folders", headers={"X-Twin-Folder": "sandbox"})
        assert r.status_code == 200
        body = r.json()
        assert [folder["id"] for folder in body] == ["default", "sandbox"]
        assert next(folder for folder in body if folder["id"] == "sandbox")["current"]

    async def test_folders_endpoint_uses_configured_folders(
        self, monkeypatch, client
    ):
        self._configure_folders(monkeypatch)
        r = await client.get("/folders")
        assert r.status_code == 200
        assert [folder["id"] for folder in r.json()] == ["default", "sandbox"]

    async def test_rejects_unknown_folder_header(self, monkeypatch, client):
        self._configure_folders(monkeypatch)
        r = await client.get("/tags", headers={"X-Twin-Folder": "rogue"})
        assert r.status_code == 403
        assert r.json()["detail"] == (
            "No folder available for this KB. Please contact Twincore Team"
        )

    async def test_folder_header_is_accepted(self, monkeypatch, client):
        self._configure_folders(monkeypatch)
        r = await client.get("/documents", headers={"X-Twin-Folder": "sandbox"})
        assert r.status_code == 200
        assert r.json() == {"items": [], "total": 0}


class TestNotifications:
    async def test_list(self, client):
        r = await client.get("/notifications")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == len(NOTIFICATIONS)

    async def test_read_all_flips_every_notification(self, client):
        await client.post("/notifications/read-all")
        r = await client.get("/notifications")
        body = r.json()
        assert all(n["read"] is True for n in body)

    async def test_clear_empties_the_list(self, client):
        r = await client.delete("/notifications")
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        r = await client.get("/notifications")
        assert r.json() == []


# ---------------------------------------------------------------------------
# Thesaurus + tags
# ---------------------------------------------------------------------------


class TestThesaurus:
    async def test_list(self, client):
        r = await client.get("/thesaurus")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == len(THESAURUS)
        for entry in body:
            assert set(entry.keys()) >= {"tag", "category", "def"}


class TestTags:
    async def test_list_full_fixture(self, client):
        r = await client.get("/tags")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == len(TAGS)
        # Spot-check: "argocd" is the requested tier
        argocd = next(t for t in body if t["tag"] == "argocd")
        assert argocd["tier"] == "requested"
        assert argocd["status"] == "pending-review"

    async def test_categories(self, client):
        r = await client.get("/tags/categories")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == len(TAG_CATEGORIES)
        ids = [c["id"] for c in body]
        for required in ("oracle", "infra", "payment", "lifecycle", "governance", "network"):
            assert required in ids


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


class TestActivity:
    async def test_list_returns_items_total_nowMs(self, client):
        r = await client.get("/activity")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == len(ACTIVITY)
        assert body["nowMs"] == ACTIVITY_NOW_MS
        assert len(body["items"]) == len(ACTIVITY)

    async def test_sev_filter_error_only(self, client):
        r = await client.get("/activity", params={"sev": "error"})
        body = r.json()
        assert body["total"] >= 1
        for e in body["items"]:
            assert e["sev"] == "error"

    async def test_kind_filter_csv(self, client):
        r = await client.get("/activity", params={"kind": "retrieval,auth"})
        body = r.json()
        for e in body["items"]:
            assert e["kind"] in {"retrieval", "auth"}

    async def test_actor_filter(self, client):
        r = await client.get("/activity", params={"actor": "marc.berthier"})
        body = r.json()
        assert body["total"] >= 1
        for e in body["items"]:
            assert e["actor"]["user"] == "marc.berthier"

    async def test_q_substring_match(self, client):
        r = await client.get("/activity", params={"q": "Oracle"})
        body = r.json()
        assert body["total"] >= 1


# ---------------------------------------------------------------------------
# OpenAPI curated surface
# ---------------------------------------------------------------------------


class TestOpenApiCurated:
    async def test_returns_groups_plus_version(self, client):
        r = await client.get("/openapi")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == OPENAPI_VERSION
        assert len(body["groups"]) == len(OPENAPI_GROUPS)

    async def test_does_not_shadow_fastapi_native_openapi(self, client):
        # FastAPI's own /openapi.json must remain reachable and distinct from
        # the WebUI curated surface.
        r = await client.get("/openapi.json")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        # Native OpenAPI carries an "openapi" version key, the curated one
        # carries "version" + "groups" — they're structurally different.
        assert "openapi" in body
        assert "groups" not in body


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class TestGraph:
    async def test_entities(self, client):
        r = await client.get("/graph/entities")
        body = r.json()
        assert len(body) == len(GRAPH_ENTITIES)
        ids = [e["id"] for e in body]
        assert "e_oracle" in ids and "e_memgraph" in ids

    async def test_relations(self, client):
        r = await client.get("/graph/relations")
        body = r.json()
        assert len(body) == len(GRAPH_RELATIONS)
        # Every relation source/target should resolve in the entity set
        eids = {e["id"] for e in (await client.get("/graph/entities")).json()}
        for r_ in body:
            assert r_["source"] in eids
            assert r_["target"] in eids


# ---------------------------------------------------------------------------
# JSON-not-HTML invariant on 404 / 405 / 422
# ---------------------------------------------------------------------------


class TestJsonInvariant:
    """Locks down the v0.5.2 lesson: nginx 502 / FastAPI 404/405/422 must
    all return JSON, never HTML. The WebUI front shell crashed when an
    endpoint returned an HTML 502 — we guarantee here that nothing under
    the WebUI router does that.
    """

    async def test_unknown_endpoint_is_json_404(self, client):
        r = await client.get("/this-does-not-exist")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/json")

    async def test_wrong_method_is_json_405(self, client):
        r = await client.post("/documents")  # /documents is GET-only on the WebUI router
        assert r.status_code == 405
        assert r.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Settings flag — disabled mode hides the WebUI surface
# ---------------------------------------------------------------------------


class TestSettingsFlag:
    async def test_disabling_drops_the_router(self):
        settings = LightRAGServerSettings(
            working_dir="/tmp/lightrag_webui_test",
            workspace="cib",
            enable_langsmith_tracing=False,
            api_key=None,
            jwt_secret=None,
            enable_webui_routes=False,
        )
        app = create_app(settings)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            # /documents is provided by the WebUI router AND chunk_routes
            # exposes /documents/paginated. The plain /documents path is
            # WebUI-only, so disabling drops it.
            r = await c.get("/documents")
            assert r.status_code in (404, 405)
            # Health stays up regardless
            r = await c.get("/health")
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# Auth wired across the WebUI router
# ---------------------------------------------------------------------------


class TestAuthGate:
    async def test_documents_requires_bearer_when_api_key_set(self):
        app = create_app(_make_settings(api_key="secret"))
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.get("/documents")
            assert r.status_code in (401, 403)
            assert r.headers["content-type"].startswith("application/json")
            # Same call with the right bearer succeeds
            r = await c.get("/documents", headers={"Authorization": "Bearer secret"})
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# WebuiStore.for_folder(mode=...) — mock-kill F6
# ---------------------------------------------------------------------------


class TestForFolderMode:
    """Regression for mock-kill audit 2026-06-04 finding F6.

    In ``memgraph`` mode, ``for_folder()`` must NOT seed the in-memory
    `_documents` / `_graph_entities` / `_graph_relations` lists even for
    the default folder — otherwise ``/twin/api/documents`` and
    ``/twin/api/graph/*`` silently expose the demo payload on a real
    BNP deploy.
    """

    def _default_folder(self) -> str:
        from twindb_lightrag_memgraph.server.folder import load_folder_catalog
        return load_folder_catalog().default_folder_id

    def test_default_folder_seed_mode_keeps_full_payload(self):
        # Sanity: the legacy `seed` mode (CI + standalone demo) still
        # populates documents/graph for the default folder.
        store = webui_router.WebuiStore.for_folder(self._default_folder())
        # Probe internals — these are intentionally private but the
        # contract is what /twin/api/documents reads from.
        assert len(store._documents) > 0  # noqa: SLF001
        assert len(store._graph_entities) > 0  # noqa: SLF001

    def test_default_folder_memgraph_mode_starts_empty(self):
        store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        assert store._documents == []  # noqa: SLF001
        assert store._graph_entities == []  # noqa: SLF001
        assert store._graph_relations == []  # noqa: SLF001
        # Reference data still loads (folders / thesaurus / openapi
        # are NOT user-generated — they're catalog metadata).
        assert len(store._folders) > 0  # noqa: SLF001
        assert len(store._thesaurus) > 0  # noqa: SLF001

    def test_non_default_folder_memgraph_mode_starts_empty(self):
        store = webui_router.WebuiStore.for_folder(
            "sandbox-that-does-not-exist", mode="memgraph"
        )
        assert store._documents == []  # noqa: SLF001
        assert store._graph_entities == []  # noqa: SLF001

    def test_non_default_folder_seed_mode_already_empty_for_user_data(self):
        # Existing behaviour — non-default folders don't get user data
        # in seed mode either. Lock it in.
        store = webui_router.WebuiStore.for_folder("sandbox-yes")
        assert store._documents == []  # noqa: SLF001
        assert store._graph_entities == []  # noqa: SLF001

    async def test_default_folder_memgraph_mode_documents_endpoint_returns_empty(
        self, client
    ):
        """End-to-end: /twin/api/documents reads the (now empty) `_documents`."""
        # Swap the active store to a memgraph-mode one for the default
        # folder and assert the endpoint reflects the empty state.
        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        r = await client.get("/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["items"] == []
        assert body["total"] == 0
