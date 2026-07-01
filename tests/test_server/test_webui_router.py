"""Tests for the WebUI phase-1 router.

These tests hit the FastAPI app via ``httpx.AsyncClient + ASGITransport`` so
they cover the wire contract (shape + JSON-error invariants) without any
network or LightRAG dependency. The WebUI router is purely in-memory in
phase 1.

Following the v0.5.2 lesson: every endpoint and every error path must return
JSON, never HTML. We assert content-type on each response to lock that in.
"""

from __future__ import annotations

import datetime
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import configure_auth
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

    async def test_folders_endpoint_uses_live_docstatus_counts(
        self, monkeypatch, client
    ):
        self._configure_folders(monkeypatch)

        class FakeDocStatus:
            async def get_status_counts(self, folder=None):
                return {
                    "default": {"processed": 3, "failed": 1},
                    "sandbox": {"processed": 2},
                }.get(folder, {})

        class FakeRag:
            doc_status = FakeDocStatus()

        monkeypatch.setattr(webui_router, "_get_rag", lambda: FakeRag())

        r = await client.get("/folders")
        assert r.status_code == 200
        by_id = {folder["id"]: folder for folder in r.json()}
        assert by_id["default"]["sources"] == 4
        assert by_id["sandbox"]["sources"] == 2

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
    async def test_list_is_legacy_projection_of_tags(self, client):
        r = await client.get("/thesaurus")
        assert r.status_code == 200
        body = r.json()
        expected = [
            t
            for t in TAGS
            if t["tier"] != "requested"
            and t["status"] not in {"deprecated", "rejected"}
        ]
        assert len(body) == len(expected)
        for entry in body:
            assert set(entry.keys()) >= {"tag", "category", "def"}
        assert {entry["tag"] for entry in body} == {entry["tag"] for entry in expected}


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
        for required in ("oracle", "infra", "messaging", "lifecycle", "governance", "network"):
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

    async def test_q_matches_event_id(self, client):
        r = await client.get("/activity", params={"q": ACTIVITY[0]["id"]})
        body = r.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == ACTIVITY[0]["id"]

    async def test_record_source_uploaded_persists_activity_without_trusting_client_actor(
        self, client
    ):
        r = await client.post(
            "/documents/uploads/activity",
            json={
                "source": "runbook.pdf",
                "track_id": "upload-track-1",
                "status": "success",
                "actor": "claire.benoit",
            },
        )
        assert r.status_code == 200

        feed = await client.get(
            "/activity", params={"kind": "source-uploaded", "q": "runbook.pdf"}
        )
        body = feed.json()
        assert body["total"] == 1
        event = body["items"][0]
        assert event["actor"]["user"] == "operator"
        assert event["target"]["type"] == "source"
        assert event["target"]["label"] == "runbook.pdf"
        assert event["meta"]["track_id"] == "upload-track-1"

    async def test_twin_auth_logout_emits_activity(self, client):
        r = await client.post("/twin/api/auth/logout")
        assert r.status_code == 200
        set_cookie = "\n".join(r.headers.get_list("set-cookie"))
        assert "twin_local_token=" in set_cookie
        assert "Max-Age=0" in set_cookie

        feed = await client.get("/activity", params={"kind": "auth", "limit": 1})
        body = feed.json()
        assert body["total"] >= 1
        event = body["items"][0]
        assert event["kind"] == "auth"
        assert event["sev"] == "info"
        assert event["actor"]["user"] == "unknown"
        assert event["target"]["type"] == "auth"
        assert event["target"]["label"] == "logout"
        assert event["meta"]["operation"] == "logout"

    async def test_total_is_pre_limit(self, client):
        response = await client.get("/activity", params={"kind": "retrieval", "limit": 1})
        body = response.json()
        assert body["total"] >= len(body["items"])
        assert body["total"] >= 2
        assert len(body["items"]) == 1

    async def test_range_filter(self, client):
        now = datetime.datetime.fromtimestamp(
            ACTIVITY_NOW_MS / 1000, tz=datetime.timezone.utc
        ).replace(microsecond=0)
        recent = (now - datetime.timedelta(hours=1)).isoformat().replace(
            "+00:00", "Z"
        )
        old = (now - datetime.timedelta(days=2)).isoformat().replace("+00:00", "Z")
        store = webui_router.get_store()
        await store.record_activity(
            {
                "id": "evt_range_recent",
                "ts": recent,
                "rel": "now",
                "day": "Today",
                "kind": "retrieval",
                "sev": "info",
                "actor": {"user": "range-tester", "role": "operator"},
                "target": {"type": "query", "label": "quick lookup"},
                "summary": "within 24h",
                "meta": {},
            }
        )
        await store.record_activity(
            {
                "id": "evt_range_old",
                "ts": old,
                "rel": "today",
                "day": "Yesterday",
                "kind": "retrieval",
                "sev": "info",
                "actor": {"user": "range-tester", "role": "operator"},
                "target": {"type": "query", "label": "older lookup"},
                "summary": "older than 24h",
                "meta": {},
            }
        )

        response = await client.get(
            "/activity",
            params={"actor": "range-tester", "range": "24h"},
        )
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == "evt_range_recent"

    async def test_resource_id_filter(self, client):
        store = webui_router.get_store()
        await store.record_activity(
            {
                "id": "evt_resource_target",
                "ts": "2026-05-13T00:00:00Z",
                "rel": "today",
                "day": "Today",
                "kind": "doc-approved",
                "sev": "info",
                "actor": {"user": "system", "role": "operator"},
                "target": {"type": "document", "label": "doc-one", "id": "doc-123"},
                "summary": "resource id in target",
                "meta": {},
            }
        )
        await store.record_activity(
            {
                "id": "evt_resource_meta",
                "ts": "2026-05-13T00:01:00Z",
                "rel": "today",
                "day": "Today",
                "kind": "doc-approved",
                "sev": "info",
                "actor": {"user": "system", "role": "operator"},
                "target": {"type": "document", "label": "doc-two"},
                "summary": "resource id in meta.doc_id",
                "meta": {"doc_id": "doc-456"},
            }
        )
        await store.record_activity(
            {
                "id": "evt_resource_historic",
                "ts": "2026-05-13T00:02:00Z",
                "rel": "today",
                "day": "Today",
                "kind": "doc-approved",
                "sev": "info",
                "actor": {"user": "system", "role": "operator"},
                "target": {"type": "document", "label": "legacy"},
                "summary": "resource id missing",
                "meta": {},
            }
        )

        response = await client.get("/activity", params={"resource.id": "doc-456"})
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == "evt_resource_meta"


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

    async def test_router_rejects_anonymous_when_mounted_directly(self):
        configure_auth(api_key="secret")
        app = FastAPI()
        app.include_router(webui_router.router)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as c:
                r = await c.get("/documents")
                assert r.status_code == 401
                r = await c.get(
                    "/documents",
                    headers={"Authorization": "Bearer secret"},
                )
                assert r.status_code == 200
        finally:
            configure_auth(api_key=None, jwt_secret=None)

    async def test_health_stays_public_when_router_mounted_directly(self):
        configure_auth(api_key="secret")
        app = FastAPI()
        app.include_router(webui_router.router)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as c:
                r = await c.get("/health")
                assert r.status_code == 200
        finally:
            configure_auth(api_key=None, jwt_secret=None)


# ---------------------------------------------------------------------------
# WebuiStore.for_folder(mode=...) — mock-kill F6
# ---------------------------------------------------------------------------


class TestForFolderMode:
    """Regression for mock-kill audit 2026-06-04 finding F6.

    In ``memgraph`` mode, ``for_folder()`` must NOT seed the in-memory
    `_documents` / `_graph_entities` / `_graph_relations` lists even for
    the default folder — otherwise ``/twin/api/documents`` and
    ``/twin/api/graph/*`` silently expose the demo payload on a real
    production deploy.
    """

    def _default_folder(self) -> str:
        from twindb_lightrag_memgraph.server.folder import load_folder_catalog
        return load_folder_catalog().default_folder_id

    def _configure_default_plus_sandbox_folders(self, monkeypatch):
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
        assert store._thesaurus == []  # noqa: SLF001
        assert len(store._folders) > 0  # noqa: SLF001

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
        """Without a host RAG, memgraph mode falls back to empty WebUI storage."""
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

    async def test_default_folder_memgraph_mode_reads_doc_status_when_rag_exists(
        self, client, monkeypatch
    ):
        """Production path: /twin/api/documents reads Memgraph DocStatus rows."""

        class FakeDocStatus:
            def __init__(self) -> None:
                self.docs = {
                    "doc-abcdef0123456789abcdef0123456789": {
                        "status": "processed",
                        "file_path": "data-science-handbook.pdf",
                        "content_summary": "Data science and machine learning",
                        "chunks_count": 142,
                        "metadata": json.dumps(
                            {
                                "tags": ["data-science"],
                                "processing_end_time": "2026-06-11T00:30:00Z",
                            }
                        ),
                    },
                    "doc-mlops": {
                        "status": "processing",
                        "file_path": "practical-mlops.pdf",
                        "content_summary": "Production model operations",
                        "chunks_count": 14,
                        "metadata": {},
                    },
                    "doc-sandbox": {
                        "status": "processed",
                        "file_path": "sandbox.pdf",
                        "content_summary": "Wrong folder",
                        "chunks_count": 1,
                        "metadata": {"folder": "sandbox"},
                    },
                }
                self.folders = {
                    "doc-abcdef0123456789abcdef0123456789": {"default"},
                    "doc-mlops": {"default"},
                    "doc-sandbox": {"sandbox"},
                }

            async def get_by_id(self, doc_id: str):
                return self.docs.get(doc_id)

            async def get_folders_for_doc(self, doc_id: str):
                folders = self.folders.get(doc_id)
                if folders is None:
                    return None
                return sorted(folders)

            async def add_to_folder(self, doc_id: str, folder: str) -> bool:
                if doc_id not in self.docs:
                    return False
                self.folders.setdefault(doc_id, set()).add(folder)
                return True

            async def remove_from_folder(self, doc_id: str, folder: str) -> int:
                folders = self.folders.get(doc_id, set())
                folders.discard(folder)
                self.folders[doc_id] = folders
                return len(folders)

            async def get_docs_paginated(self, **kwargs: Any):
                status_filter = kwargs.get("status_filter")
                folder = kwargs.get("folder")
                wanted = getattr(status_filter, "value", None)
                rows = [
                    (doc_id, doc)
                    for doc_id, doc in self.docs.items()
                    if (wanted is None or doc["status"] == wanted)
                    and (not folder or folder in self.folders.get(doc_id, set()))
                ]
                return rows, len(rows)

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        async def no_graph_tags(_docs):
            return None

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        _twindb_state["rag"] = FakeRag()
        self._configure_default_plus_sandbox_folders(monkeypatch)
        monkeypatch.setattr(
            webui_router,
            "_attach_graph_tags_for_documents",
            no_graph_tags,
        )
        try:
            r = await client.get("/documents")
            assert r.status_code == 200
            body = r.json()
            assert body["total"] == 2
            assert [doc["doc_id"] for doc in body["items"]] == [
                "doc-abcdef0123456789abcdef0123456789",
                "doc-mlops",
            ]
            assert body["items"][0]["status"] == "PROCESSED"
            assert body["items"][0]["tags"] == ["data-science"]
            assert body["items"][0]["chunks_count"] == 142
            assert (
                body["items"][0]["metadata"]["content_hash"]
                == "abcdef0123456789abcdef0123456789"
            )
            assert (
                body["items"][0]["metadata"]["content_hash_source"]
                == "lightrag_doc_id"
            )

            filtered = await client.get("/documents", params={"status": "PROCESSED"})
            assert filtered.status_code == 200
            assert [doc["doc_id"] for doc in filtered.json()["items"]] == [
                "doc-abcdef0123456789abcdef0123456789"
            ]

            tagged = await client.get("/documents", params={"tag": "data-science"})
            assert tagged.status_code == 200
            assert [doc["doc_id"] for doc in tagged.json()["items"]] == [
                "doc-abcdef0123456789abcdef0123456789"
            ]

            copied = await client.post(
                "/documents/doc-abcdef0123456789abcdef0123456789/folders",
                json={"folder_id": "sandbox"},
            )
            assert copied.status_code == 200

            default_docs = await client.get("/documents")
            sandbox_docs = await client.get(
                "/documents",
                headers={"X-Twin-Folder": "sandbox"},
            )
            assert default_docs.status_code == 200
            assert sandbox_docs.status_code == 200

            assert default_docs.json()["total"] == 2
            assert [
                doc["doc_id"]
                for doc in default_docs.json()["items"]
            ] == [
                "doc-abcdef0123456789abcdef0123456789",
                "doc-mlops",
            ]
            assert sandbox_docs.json()["total"] == 2
            assert {
                doc["doc_id"]
                for doc in sandbox_docs.json()["items"]
            } == {
                "doc-abcdef0123456789abcdef0123456789",
                "doc-sandbox",
            }
        finally:
            _twindb_state.pop("rag", None)

    async def test_documents_query_uses_folder_arg_when_supported(
        self, client, monkeypatch
    ):
        """DocStatus listing must pass folder explicitly when the storage supports it."""

        class FakeDocStatus:
            def __init__(self) -> None:
                self.docs = {
                    "doc-default": {
                        "status": "processed",
                        "file_path": "default.pdf",
                        "content_summary": "default",
                        "chunks_count": 1,
                        "metadata": {"folder": "default"},
                    },
                    "doc-sandbox": {
                        "status": "processed",
                        "file_path": "sandbox.pdf",
                        "content_summary": "sandbox",
                        "chunks_count": 1,
                        "metadata": {"folder": "sandbox"},
                    },
                }
                self.folders = {"doc-default": {"default"}, "doc-sandbox": {"sandbox"}}
                self.last_kwargs: dict[str, Any] = {}

            async def get_by_id(self, doc_id: str):
                return self.docs.get(doc_id)

            async def get_folders_for_doc(self, doc_id: str):
                return sorted(self.folders.get(doc_id, []))

            async def get_docs_paginated(
                self,
                page: int = 1,
                page_size: int = 500,
                status_filter: Any | None = None,
                folder: str | None = None,
            ):
                self.last_kwargs["folder"] = folder
                wanted = getattr(status_filter, "value", None)
                rows = [
                    (doc_id, doc)
                    for doc_id, doc in self.docs.items()
                    if wanted is None or doc["status"] == wanted
                ]
                if folder:
                    rows = [
                        row
                        for row in rows
                        if row[1]["metadata"]["folder"] == folder
                    ]
                return rows, len(rows)

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        async def no_graph_tags(_docs):
            return None

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        rag = FakeRag()
        _twindb_state["rag"] = rag
        self._configure_default_plus_sandbox_folders(monkeypatch)
        monkeypatch.setattr(webui_router, "_attach_graph_tags_for_documents", no_graph_tags)
        try:
            sandbox_docs = await client.get(
                "/documents",
                headers={"X-Twin-Folder": "sandbox"},
            )
            assert sandbox_docs.status_code == 200
            assert sandbox_docs.json()["total"] == 1
            assert sandbox_docs.json()["items"][0]["doc_id"] == "doc-sandbox"
            assert rag.doc_status.last_kwargs.get("folder") == "sandbox"
        finally:
            _twindb_state.pop("rag", None)

    async def test_folder_aware_documents_query_is_safety_filtered_for_active_folder(
        self, client, monkeypatch
    ):
        """Even folder-aware storages must not leak documents from other folders."""

        class FakeDocStatus:
            def __init__(self) -> None:
                self.docs = {
                    "doc-default": {
                        "status": "processed",
                        "file_path": "default.pdf",
                        "content_summary": "default",
                        "chunks_count": 1,
                        "metadata": {"folder": "default"},
                    },
                    "doc-sandbox": {
                        "status": "processed",
                        "file_path": "sandbox.pdf",
                        "content_summary": "sandbox",
                        "chunks_count": 1,
                        "metadata": {"folder": "sandbox"},
                    },
                }
                self.folders = {"doc-default": {"default"}, "doc-sandbox": {"sandbox"}}
                self.calls: list[str] = []

            async def get_by_id(self, doc_id: str):
                return self.docs.get(doc_id)

            async def get_folders_for_doc(self, doc_id: str):
                self.calls.append(doc_id)
                return sorted(self.folders.get(doc_id, []))

            async def get_docs_paginated(
                self,
                page: int = 1,
                page_size: int = 500,
                status_filter: Any | None = None,
                folder: str | None = None,
            ):
                self.calls.append(f"folder={folder}")
                wanted = getattr(status_filter, "value", None)
                rows = [
                    (doc_id, doc)
                    for doc_id, doc in self.docs.items()
                    if wanted is None or doc["status"] == wanted
                ]
                # Simulate a folder-aware storage bug: returns all folders even when filtered.
                return rows, len(rows)

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        async def no_graph_tags(_docs):
            return None

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        rag = FakeRag()
        _twindb_state["rag"] = rag
        self._configure_default_plus_sandbox_folders(monkeypatch)
        monkeypatch.setattr(webui_router, "_attach_graph_tags_for_documents", no_graph_tags)
        try:
            sandbox_docs = await client.get(
                "/documents",
                headers={"X-Twin-Folder": "sandbox"},
            )
            assert sandbox_docs.status_code == 200
            assert sandbox_docs.json()["total"] == 1
            assert sandbox_docs.json()["items"][0]["doc_id"] == "doc-sandbox"
            assert any(call == "folder=sandbox" for call in rag.doc_status.calls)
            assert "doc-default" not in {
                doc["doc_id"] for doc in sandbox_docs.json()["items"]
            }
            assert "doc-default" in {
                value for value in rag.doc_status.calls if value.startswith("doc-")
            }
        finally:
            _twindb_state.pop("rag", None)

    async def test_legacy_doc_status_listing_falls_back_without_folder_scoping(
        self, client, monkeypatch
    ):
        """Legacy storage fallback keeps list memory-filtered by active folder."""

        class FakeDocStatus:
            def __init__(self) -> None:
                self.docs = {
                    "doc-default": {
                        "status": "processed",
                        "file_path": "default.pdf",
                        "content_summary": "default",
                        "chunks_count": 1,
                        "metadata": {"folder": "default"},
                    },
                    "doc-sandbox": {
                        "status": "processed",
                        "file_path": "sandbox.pdf",
                        "content_summary": "sandbox",
                        "chunks_count": 1,
                        "metadata": {"folder": "sandbox"},
                    },
                    "doc-other": {
                        "status": "processed",
                        "file_path": "other.pdf",
                        "content_summary": "other",
                        "chunks_count": 1,
                        "metadata": {},
                    },
                }

            async def get_by_id(self, doc_id: str):
                return self.docs.get(doc_id)

            async def get_docs_paginated(
                self,
                page: int = 1,
                page_size: int = 500,
                status_filter: Any | None = None,
            ):
                wanted = getattr(status_filter, "value", None)
                rows = [
                    (doc_id, doc)
                    for doc_id, doc in self.docs.items()
                    if wanted is None or doc["status"] == wanted
                ]
                return rows, len(rows)

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        async def no_graph_tags(_docs):
            return None

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        self._configure_default_plus_sandbox_folders(monkeypatch)
        rag = FakeRag()
        _twindb_state["rag"] = rag
        try:
            sandbox_docs = await client.get(
                "/documents",
                headers={"X-Twin-Folder": "sandbox"},
            )
            assert sandbox_docs.status_code == 200
            assert {doc["doc_id"] for doc in sandbox_docs.json()["items"]} == {
                "doc-sandbox"
            }
        finally:
            _twindb_state.pop("rag", None)

    async def test_internal_folder_scoped_storage_typeerror_is_not_suppressed(
        self, client
    ):
        """A TypeError from a folder-aware storage path must bubble up."""

        class FakeDocStatus:
            async def get_by_id(self, doc_id: str):
                return {"status": "processed"}

            async def get_docs_paginated(
                self,
                page: int = 1,
                page_size: int = 500,
                status_filter: Any | None = None,
                folder: str | None = None,
            ):
                if folder == "default":
                    raise TypeError("simulated type mismatch in folder-aware path")
                return [("doc-default", {"status": "processed"})], 1

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        _twindb_state["rag"] = FakeRag()
        try:
            with pytest.raises(TypeError, match="simulated type mismatch"):
                await client.get("/documents")
        finally:
            _twindb_state.pop("rag", None)

    async def test_destination_folder_visible_after_copy_to_folder(self, client, monkeypatch):
        """Copying a doc into another folder makes it appear there in list queries."""

        class FakeDocStatus:
            def __init__(self) -> None:
                self.docs = {
                    "doc-a": {
                        "status": "processed",
                        "file_path": "doc-a.pdf",
                        "content_summary": "Source file",
                        "chunks_count": 2,
                        "metadata": {"folder": "default"},
                    },
                }
                self.folders = {"doc-a": {"default"}}

            async def get_by_id(self, doc_id: str):
                return self.docs.get(doc_id)

            async def get_folders_for_doc(self, doc_id: str):
                folders = self.folders.get(doc_id)
                return sorted(folders) if folders is not None else None

            async def add_to_folder(self, doc_id: str, folder: str) -> bool:
                if doc_id not in self.docs:
                    return False
                self.folders.setdefault(doc_id, set()).add(folder)
                return True

            async def remove_from_folder(self, doc_id: str, folder: str) -> int:
                folders = self.folders.get(doc_id, set())
                folders.discard(folder)
                self.folders[doc_id] = folders
                return len(folders)

            async def get_docs_paginated(self, **kwargs: Any):
                folder = kwargs.get("folder")
                status_filter = kwargs.get("status_filter")
                wanted = getattr(status_filter, "value", None)
                rows = [
                    (doc_id, doc)
                    for doc_id, doc in self.docs.items()
                    if (wanted is None or doc["status"] == wanted)
                    and (not folder or folder in self.folders.get(doc_id, set()))
                ]
                return rows, len(rows)

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        async def no_graph_tags(_docs):
            return None

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        _twindb_state["rag"] = FakeRag()
        self._configure_default_plus_sandbox_folders(monkeypatch)
        monkeypatch.setattr(webui_router, "_attach_graph_tags_for_documents", no_graph_tags)
        try:
            copy_response = await client.post(
                "/documents/doc-a/folders",
                json={"folder_id": "sandbox"},
            )
            assert copy_response.status_code == 200
            copied_docs = await client.get(
                "/documents",
                headers={"X-Twin-Folder": "sandbox"},
            )
            assert copied_docs.status_code == 200
            assert copied_docs.json()["total"] == 1
            assert copied_docs.json()["items"][0]["doc_id"] == "doc-a"
            assert copied_docs.json()["items"][0]["folder"] == "sandbox"
        finally:
            _twindb_state.pop("rag", None)

    async def test_destination_folder_visible_after_move_to_folder(self, client, monkeypatch):
        """Moving a doc removes it from source and keeps it visible in destination."""

        class FakeDocStatus:
            def __init__(self) -> None:
                self.docs = {
                    "doc-a": {
                        "status": "processed",
                        "file_path": "doc-a.pdf",
                        "content_summary": "Source file",
                        "chunks_count": 2,
                        "metadata": {"folder": "default"},
                    },
                }
                self.folders = {"doc-a": {"default"}}

            async def get_by_id(self, doc_id: str):
                return self.docs.get(doc_id)

            async def get_folders_for_doc(self, doc_id: str):
                folders = self.folders.get(doc_id)
                return sorted(folders) if folders is not None else None

            async def add_to_folder(self, doc_id: str, folder: str) -> bool:
                if doc_id not in self.docs:
                    return False
                self.folders.setdefault(doc_id, set()).add(folder)
                return True

            async def remove_from_folder(self, doc_id: str, folder: str) -> int:
                folders = self.folders.get(doc_id, set())
                folders.discard(folder)
                self.folders[doc_id] = folders
                return len(folders)

            async def get_docs_paginated(self, **kwargs: Any):
                folder = kwargs.get("folder")
                status_filter = kwargs.get("status_filter")
                wanted = getattr(status_filter, "value", None)
                rows = [
                    (doc_id, doc)
                    for doc_id, doc in self.docs.items()
                    if (wanted is None or doc["status"] == wanted)
                    and (not folder or folder in self.folders.get(doc_id, set()))
                ]
                return rows, len(rows)

            async def adelete_by_doc_id(self, doc_id: str) -> None:
                self.docs.pop(doc_id, None)
                self.folders.pop(doc_id, None)

        class FakeRag:
            def __init__(self) -> None:
                self.doc_status = FakeDocStatus()

        async def no_graph_tags(_docs):
            return None

        empty_store = webui_router.WebuiStore.for_folder(
            self._default_folder(), mode="memgraph"
        )
        webui_router.set_store(empty_store)
        _twindb_state["rag"] = FakeRag()
        self._configure_default_plus_sandbox_folders(monkeypatch)
        monkeypatch.setattr(webui_router, "_attach_graph_tags_for_documents", no_graph_tags)
        try:
            move_to_b = await client.post(
                "/documents/doc-a/folders",
                json={"folder_id": "sandbox"},
            )
            assert move_to_b.status_code == 200
            remove_from_default = await client.delete("/documents/doc-a/folders/default")
            assert remove_from_default.status_code == 200

            default_docs = await client.get("/documents")
            sandbox_docs = await client.get(
                "/documents",
                headers={"X-Twin-Folder": "sandbox"},
            )

            assert default_docs.status_code == 200
            assert sandbox_docs.status_code == 200
            assert default_docs.json()["total"] == 0
            assert sandbox_docs.json()["total"] == 1
            assert sandbox_docs.json()["items"][0]["doc_id"] == "doc-a"
            assert sandbox_docs.json()["items"][0]["folder"] == "sandbox"
        finally:
            _twindb_state.pop("rag", None)
