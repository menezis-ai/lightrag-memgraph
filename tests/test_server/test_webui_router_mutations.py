"""Tests for the WebUI tag-mutation endpoints (S4c slice 2).

Each mutation must:
- update the tag store (verified via subsequent GET /tags/{name} look-up
  through GET /tags),
- emit an activity event (verified via GET /activity),
- push a notification (verified via GET /notifications).
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _make_settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        working_dir="/tmp/lightrag_webui_mutation_test",
        workspace="cib",
        enable_langsmith_tracing=False,
        api_key=None,
        jwt_secret=None,
        enable_webui_routes=True,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    webui_router.reset_store()
    yield
    webui_router.reset_store()


@pytest.fixture()
async def client():
    app = create_app(_make_settings())
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def _get_tag(client: AsyncClient, name: str) -> dict | None:
    r = await client.get("/tags")
    for t in r.json():
        if t["tag"] == name:
            return t
    return None


async def _get_activity(client: AsyncClient) -> list[dict]:
    r = await client.get("/activity")
    return r.json()["items"]


async def _get_notifications(client: AsyncClient) -> list[dict]:
    r = await client.get("/notifications")
    return r.json()


# ---------------------------------------------------------------------------
# POST /tags — request new tag
# ---------------------------------------------------------------------------


class TestRequestTag:
    async def test_creates_pending_tag(self, client):
        r = await client.post(
            "/tags",
            json={
                "tag": "newtag",
                "def": "A brand new tag",
                "category": "infra",
                "justification": "Needed for upcoming sprint",
                "actor": "marc.berthier",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["tag"] == "newtag"
        assert body["tier"] == "requested"
        assert body["status"] == "pending-review"
        # GET round-trip
        tag = await _get_tag(client, "newtag")
        assert tag is not None
        assert tag["tier"] == "requested"

    async def test_duplicate_returns_409(self, client):
        r = await client.post(
            "/tags",
            json={"tag": "rman", "def": "dup", "category": "oracle"},
        )
        assert r.status_code == 409

    async def test_emits_activity_event(self, client):
        await client.post(
            "/tags",
            json={
                "tag": "newtag",
                "def": "test",
                "category": "infra",
                "actor": "marc.berthier",
            },
        )
        events = await _get_activity(client)
        assert events
        assert events[0]["kind"] == "tag-mutation"
        assert "newtag" in events[0]["summary"]
        assert events[0]["actor"]["user"] == "marc.berthier"

    async def test_pushes_notification(self, client):
        await client.post(
            "/tags",
            json={"tag": "newtag", "def": "test", "category": "infra"},
        )
        notifs = await _get_notifications(client)
        assert notifs
        assert notifs[0]["tagname"] == "newtag"
        assert notifs[0]["suffix"] == "requested"
        assert notifs[0]["read"] is False

    async def test_tag_mutation_is_isolated_by_space(self, monkeypatch, client):
        monkeypatch.setenv("TWIN_DEFAULT_SPACE", "default")
        monkeypatch.setenv(
            "TWIN_SPACES_JSON",
            json.dumps(
                [
                    {"id": "default", "label": "Default space", "kind": "primary"},
                    {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
                ]
            ),
        )
        webui_router.reset_store()

        r = await client.post(
            "/tags",
            headers={"X-Twin-Space": "default"},
            json={
                "tag": "spaceonly",
                "def": "Only in the default space",
                "category": "infra",
            },
        )
        assert r.status_code == 201

        default_tags = (await client.get(
            "/tags",
            headers={"X-Twin-Space": "default"},
        )).json()
        sandbox_tags = (await client.get(
            "/tags",
            headers={"X-Twin-Space": "sandbox"},
        )).json()
        assert any(tag["tag"] == "spaceonly" for tag in default_tags)
        assert all(tag["tag"] != "spaceonly" for tag in sandbox_tags)


# ---------------------------------------------------------------------------
# POST /tags/{name}/approve
# ---------------------------------------------------------------------------


class TestApproveTag:
    async def test_flips_tier_and_status(self, client):
        # argocd is seeded as a requested tag
        r = await client.post(
            "/tags/argocd/approve",
            json={"actor": "claire.benoit"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["tier"] == 3
        assert body["status"] == "active"
        # Requested-only fields are dropped
        assert "requested_by" not in body or body.get("requested_by") is None
        assert "justification" not in body or body.get("justification") is None

    async def test_404_when_unknown(self, client):
        r = await client.post(
            "/tags/zzz-no-tag/approve", json={"actor": "claire.benoit"}
        )
        assert r.status_code == 404

    async def test_emits_event_and_notification(self, client):
        await client.post(
            "/tags/argocd/approve", json={"actor": "claire.benoit"}
        )
        events = await _get_activity(client)
        notifs = await _get_notifications(client)
        assert events[0]["summary"].startswith("Tag argocd approved")
        assert notifs[0]["suffix"] == "approved"


# ---------------------------------------------------------------------------
# POST /tags/{name}/reject
# ---------------------------------------------------------------------------


class TestRejectTag:
    async def test_sets_status_rejected_with_reason(self, client):
        r = await client.post(
            "/tags/argocd/reject",
            json={"reason": "duplicate of k8s", "actor": "claire.benoit"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "rejected"
        assert body.get("reject_reason") == "duplicate of k8s"

    async def test_emits_warning_event(self, client):
        await client.post(
            "/tags/argocd/reject",
            json={"reason": "scope creep", "actor": "claire.benoit"},
        )
        events = await _get_activity(client)
        assert events[0]["sev"] == "warning"
        assert "scope creep" in events[0]["summary"]

    async def test_missing_reason_is_422(self, client):
        r = await client.post(
            "/tags/argocd/reject", json={"actor": "claire.benoit"}
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /tags/{name}
# ---------------------------------------------------------------------------


class TestEditTag:
    async def test_updates_def_and_aliases(self, client):
        r = await client.patch(
            "/tags/rman",
            json={
                "def": "Updated definition",
                "aliases": ["recovery-manager", "rmgr"],
                "actor": "claire.benoit",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["def"] == "Updated definition"
        assert body["aliases"] == ["recovery-manager", "rmgr"]

    async def test_no_op_is_still_successful(self, client):
        r = await client.patch(
            "/tags/rman", json={"actor": "claire.benoit"}
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /tags/{name}/deprecate
# ---------------------------------------------------------------------------


class TestDeprecateTag:
    async def test_sets_status_deprecated(self, client):
        r = await client.post(
            "/tags/rman/deprecate",
            json={"reason": "superseded by rman-v2"},
        )
        body = r.json()
        assert body["status"] == "deprecated"
        assert body.get("deprecate_reason") == "superseded by rman-v2"

    async def test_emits_warning_event(self, client):
        await client.post("/tags/rman/deprecate", json={})
        events = await _get_activity(client)
        assert events[0]["sev"] == "warning"


# ---------------------------------------------------------------------------
# POST /tags/{name}/synonyms
# ---------------------------------------------------------------------------


class TestSynonyms:
    async def test_replaces_alias_list(self, client):
        r = await client.post(
            "/tags/rman/synonyms",
            json={"aliases": ["recovery-mgr", "rmgr"]},
        )
        body = r.json()
        assert body["aliases"] == ["recovery-mgr", "rmgr"]


# ---------------------------------------------------------------------------
# DELETE /tags/{name}
# ---------------------------------------------------------------------------


class TestDeleteTag:
    async def test_migrate_requires_to(self, client):
        r = await client.request(
            "DELETE",
            "/tags/rman",
            json={"strategy": "migrate"},
        )
        assert r.status_code == 422

    async def test_migrate_with_to_succeeds(self, client):
        r = await client.request(
            "DELETE",
            "/tags/rman",
            json={"strategy": "migrate", "to": "oracle"},
        )
        assert r.status_code == 200
        # GET confirms it's gone
        tag = await _get_tag(client, "rman")
        assert tag is None
        rman_docs = await client.get("/documents", params={"tag": "rman"})
        assert rman_docs.json()["items"] == []
        oracle_docs = await client.get("/documents", params={"tag": "oracle"})
        assert oracle_docs.json()["total"] >= 1
        assert all("rman" not in doc["tags"] for doc in oracle_docs.json()["items"])
        events = await _get_activity(client)
        assert "migrated to oracle" in events[0]["summary"]
        assert events[0]["meta"]["affected_docs"] >= 1

    async def test_untag_strategy_default(self, client):
        r = await client.request("DELETE", "/tags/vault")
        assert r.status_code == 200
        vault_docs = await client.get("/documents", params={"tag": "vault"})
        assert vault_docs.json()["items"] == []
        events = await _get_activity(client)
        assert "deleted (docs untagged)" in events[0]["summary"]

    async def test_migrate_requires_existing_target(self, client):
        r = await client.request(
            "DELETE",
            "/tags/rman",
            json={"strategy": "migrate", "to": "zzz-no-tag"},
        )
        assert r.status_code == 404

    async def test_unknown_returns_404(self, client):
        r = await client.request("DELETE", "/tags/zzz-no-tag")
        assert r.status_code == 404
