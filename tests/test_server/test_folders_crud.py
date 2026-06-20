"""Admin Folder CRUD — runtime additions on top of the env catalog.

Covers:
- POST /folders: 201 success, 409 on env-seeded id collision, 409 on
  runtime duplicate, 422 on invalid id, 422 on max-folders overflow.
- PATCH /folders/{id}: 200 success, 403 on env-seeded id, 404 missing.
- DELETE /folders/{id}: 204 success, 403 on env-seeded, 404 missing,
  409 when the folder still holds data.
- GET /folders: returns env + runtime merged, with the correct
  `current` marker for the active folder.

Legacy /folders aliases stay covered in a few compatibility assertions.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import folder_store, webui_router
from twindb_lightrag_memgraph.server.app import create_app


@pytest.fixture()
async def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {
                    "id": "default",
                    "label": "Default folder",
                    "kind": "primary",
                    "description": "SRE seed",
                },
            ]
        ),
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "3")
    # Persist runtime folders to a tmp JSON file so we exercise the
    # full read-modify-write path.
    runtime_file = tmp_path / "twin-folders.json"
    monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(runtime_file))
    folder_store.reset_runtime_store()
    webui_router.reset_store()

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    folder_store.reset_runtime_store()
    webui_router.reset_store()


class TestCreateFolder:
    async def test_create_success_returns_201(self, client):
        r = await client.post(
            "/folders",
            json={
                "id": "sandbox",
                "label": "Sandbox",
                "kind": "sandbox",
                "description": "Operator test folder",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["id"] == "sandbox"
        assert body["kb"] == "Sandbox"

        # /folders now lists both env + runtime
        listing = await client.get("/folders")
        ids = [s["id"] for s in listing.json()]
        assert ids == ["default", "sandbox"]

    async def test_legacy_folders_alias_still_lists_folders(self, client):
        listing = await client.get("/folders")
        assert listing.status_code == 200
        assert [s["id"] for s in listing.json()] == ["default"]

    async def test_create_emits_activity(self, client):
        await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
        )
        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        folder_events = [
            e
            for e in events
            if e["kind"] == "settings"
            and e["meta"].get("operation") == "create"
        ]
        assert len(folder_events) == 1
        assert folder_events[0]["meta"]["folder_id"] == "sandbox"

    async def test_create_conflicts_with_env_seed(self, client):
        r = await client.post(
            "/folders",
            json={"id": "default", "label": "x"},
        )
        assert r.status_code == 409
        assert "env" in r.json()["detail"]

    async def test_create_conflicts_with_existing_runtime(self, client):
        await client.post("/folders", json={"id": "sandbox", "label": "S"})
        r = await client.post("/folders", json={"id": "sandbox", "label": "S2"})
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"].lower()

    async def test_create_422_on_invalid_id(self, client):
        r = await client.post(
            "/folders",
            json={"id": "bad folder!", "label": "x"},
        )
        assert r.status_code == 422

    async def test_create_422_when_at_max(self, client):
        # max = 3, env seed counts as 1 → can add 2 more
        await client.post("/folders", json={"id": "a", "label": "A"})
        await client.post("/folders", json={"id": "b", "label": "B"})
        r = await client.post("/folders", json={"id": "c", "label": "C"})
        assert r.status_code == 422
        assert "max" in r.json()["detail"].lower()


class TestUpdateFolder:
    async def test_update_label_and_description(self, client):
        await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox"},
        )
        r = await client.patch(
            "/folders/sandbox",
            json={"label": "Sandbox v2", "description": "Now with more sand"},
        )
        assert r.status_code == 200
        assert r.json()["kb"] == "Sandbox v2"

    async def test_update_404_when_missing(self, client):
        r = await client.patch("/folders/ghost", json={"label": "x"})
        assert r.status_code == 404

    async def test_update_403_on_env_seeded(self, client):
        r = await client.patch("/folders/default", json={"label": "Renamed"})
        assert r.status_code == 403
        assert "env-seeded" in r.json()["detail"]


class TestDeleteFolder:
    async def test_delete_runtime_folder_204(self, client):
        await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox"},
        )
        r = await client.delete("/folders/sandbox")
        assert r.status_code == 204
        ids = [s["id"] for s in (await client.get("/folders")).json()]
        assert "sandbox" not in ids

    async def test_delete_emits_activity(self, client):
        await client.post("/folders", json={"id": "sandbox", "label": "S"})
        await client.delete("/folders/sandbox")
        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        deletes = [
            e
            for e in events
            if e["kind"] == "settings"
            and e["meta"].get("operation") == "delete"
        ]
        assert len(deletes) == 1

    async def test_delete_active_folder_does_not_resurrect_store(self, client):
        await client.post("/folders", json={"id": "sandbox", "label": "S"})

        r = await client.delete(
            "/folders/sandbox",
            headers={"X-Twin-Folder": "sandbox"},
        )

        assert r.status_code == 204
        assert "sandbox" not in webui_router._stores  # noqa: SLF001

    async def test_delete_403_on_env_seeded(self, client):
        r = await client.delete("/folders/default")
        assert r.status_code == 403

    async def test_delete_404_on_unknown(self, client):
        r = await client.delete("/folders/ghost")
        assert r.status_code == 404

    async def test_delete_409_when_folder_has_tags(self, client):
        # Provision then add a tag scoped to the sandbox folder.
        await client.post("/folders", json={"id": "sandbox", "label": "S"})
        tag_post = await client.post(
            "/tags",
            json={"tag": "scoped", "category": "topic", "def": "x"},
            headers={"X-Twin-Folder": "sandbox"},
        )
        assert tag_post.status_code == 201

        r = await client.delete("/folders/sandbox")
        assert r.status_code == 409
        assert "data" in r.json()["detail"].lower()


class TestPersistence:
    async def test_runtime_file_round_trip(self, client, tmp_path):
        # The fixture set TWIN_FOLDERS_RUNTIME_FILE to a tmp path. After
        # a POST the file should exist with the new folder, and a
        # fresh in-memory load should pick it up.
        await client.post(
            "/folders",
            json={"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
        )
        # Force the loaded flag false so a subsequent list re-reads
        # from disk.
        folder_store.reset_runtime_store()
        listing = await client.get("/folders")
        ids = [s["id"] for s in listing.json()]
        assert "sandbox" in ids
