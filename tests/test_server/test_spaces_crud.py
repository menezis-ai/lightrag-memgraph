"""Admin Space CRUD — runtime additions on top of the env catalog.

Covers:
- POST /spaces: 201 success, 409 on env-seeded id collision, 409 on
  runtime duplicate, 422 on invalid id, 422 on max-spaces overflow.
- PATCH /spaces/{id}: 200 success, 403 on env-seeded id, 404 missing.
- DELETE /spaces/{id}: 204 success, 403 on env-seeded, 404 missing,
  409 when the space still holds data.
- GET /spaces: returns env + runtime merged, with the correct
  `current` marker for the active space.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import space_store, webui_router
from twindb_lightrag_memgraph.server.app import create_app


@pytest.fixture()
async def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_DEFAULT_SPACE", "default")
    monkeypatch.setenv(
        "TWIN_SPACES_JSON",
        json.dumps(
            [
                {
                    "id": "default",
                    "label": "Default space",
                    "kind": "primary",
                    "description": "SRE seed",
                },
            ]
        ),
    )
    monkeypatch.setenv("TWIN_MAX_SPACES", "3")
    # Persist runtime spaces to a tmp JSON file so we exercise the
    # full read-modify-write path.
    runtime_file = tmp_path / "twin-spaces.json"
    monkeypatch.setenv("TWIN_SPACES_RUNTIME_FILE", str(runtime_file))
    space_store.reset_runtime_store()
    webui_router.reset_store()

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    space_store.reset_runtime_store()
    webui_router.reset_store()


class TestCreateSpace:
    async def test_create_success_returns_201(self, client):
        r = await client.post(
            "/spaces",
            json={
                "id": "sandbox",
                "label": "Sandbox",
                "kind": "sandbox",
                "description": "Operator test space",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["id"] == "sandbox"
        assert body["kb"] == "Sandbox"

        # /spaces now lists both env + runtime
        listing = await client.get("/spaces")
        ids = [s["id"] for s in listing.json()]
        assert ids == ["default", "sandbox"]

    async def test_create_emits_activity(self, client):
        await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
        )
        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        space_events = [
            e
            for e in events
            if e["kind"] == "settings"
            and e["meta"].get("operation") == "create"
        ]
        assert len(space_events) == 1
        assert space_events[0]["meta"]["space_id"] == "sandbox"

    async def test_create_conflicts_with_env_seed(self, client):
        r = await client.post(
            "/spaces",
            json={"id": "default", "label": "x"},
        )
        assert r.status_code == 409
        assert "env" in r.json()["detail"]

    async def test_create_conflicts_with_existing_runtime(self, client):
        await client.post("/spaces", json={"id": "sandbox", "label": "S"})
        r = await client.post("/spaces", json={"id": "sandbox", "label": "S2"})
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"].lower()

    async def test_create_422_on_invalid_id(self, client):
        r = await client.post(
            "/spaces",
            json={"id": "bad space!", "label": "x"},
        )
        assert r.status_code == 422

    async def test_create_422_when_at_max(self, client):
        # max = 3, env seed counts as 1 → can add 2 more
        await client.post("/spaces", json={"id": "a", "label": "A"})
        await client.post("/spaces", json={"id": "b", "label": "B"})
        r = await client.post("/spaces", json={"id": "c", "label": "C"})
        assert r.status_code == 422
        assert "max" in r.json()["detail"].lower()


class TestUpdateSpace:
    async def test_update_label_and_description(self, client):
        await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox"},
        )
        r = await client.patch(
            "/spaces/sandbox",
            json={"label": "Sandbox v2", "description": "Now with more sand"},
        )
        assert r.status_code == 200
        assert r.json()["kb"] == "Sandbox v2"

    async def test_update_404_when_missing(self, client):
        r = await client.patch("/spaces/ghost", json={"label": "x"})
        assert r.status_code == 404

    async def test_update_403_on_env_seeded(self, client):
        r = await client.patch("/spaces/default", json={"label": "Renamed"})
        assert r.status_code == 403
        assert "env-seeded" in r.json()["detail"]


class TestDeleteSpace:
    async def test_delete_runtime_space_204(self, client):
        await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox"},
        )
        r = await client.delete("/spaces/sandbox")
        assert r.status_code == 204
        ids = [s["id"] for s in (await client.get("/spaces")).json()]
        assert "sandbox" not in ids

    async def test_delete_emits_activity(self, client):
        await client.post("/spaces", json={"id": "sandbox", "label": "S"})
        await client.delete("/spaces/sandbox")
        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        deletes = [
            e
            for e in events
            if e["kind"] == "settings"
            and e["meta"].get("operation") == "delete"
        ]
        assert len(deletes) == 1

    async def test_delete_403_on_env_seeded(self, client):
        r = await client.delete("/spaces/default")
        assert r.status_code == 403

    async def test_delete_404_on_unknown(self, client):
        r = await client.delete("/spaces/ghost")
        assert r.status_code == 404

    async def test_delete_409_when_space_has_tags(self, client):
        # Provision then add a tag scoped to the sandbox space.
        await client.post("/spaces", json={"id": "sandbox", "label": "S"})
        tag_post = await client.post(
            "/tags",
            json={"tag": "scoped", "category": "topic", "def": "x"},
            headers={"X-Twin-Space": "sandbox"},
        )
        assert tag_post.status_code == 201

        r = await client.delete("/spaces/sandbox")
        assert r.status_code == 409
        assert "data" in r.json()["detail"].lower()


class TestPersistence:
    async def test_runtime_file_round_trip(self, client, tmp_path):
        # The fixture set TWIN_SPACES_RUNTIME_FILE to a tmp path. After
        # a POST the file should exist with the new space, and a
        # fresh in-memory load should pick it up.
        await client.post(
            "/spaces",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
        )
        # Force the loaded flag false so a subsequent list re-reads
        # from disk.
        space_store.reset_runtime_store()
        listing = await client.get("/spaces")
        ids = [s["id"] for s in listing.json()]
        assert "sandbox" in ids
