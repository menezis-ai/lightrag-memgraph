"""Router-level tests for the Memgraph-backed Graph endpoints.

Verifies the seed fallback policy:
- when ``graph_reader.read_graph_entities`` returns rows, the route
  serves them (no fallback)
- when it returns ``[]`` (Memgraph empty or down), the route falls
  back to the in-memory seed so dev / standalone demos keep working
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server import graph_reader as gr
from twindb_lightrag_memgraph.server.app import create_app


@pytest.fixture()
async def client():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    webui_router.reset_store()


class TestGraphRoutesMemgraphFirst:
    async def test_entities_serves_memgraph_rows_when_present(
        self, monkeypatch, client
    ):
        async def fake_entities(workspace, *, max_nodes=200):
            return [
                {
                    "id": "kg_Oracle Database",
                    "name": "Oracle Database",
                    "type": "PRODUCT",
                    "x": 480,
                    "y": 310,
                    "mentions": 5,
                    "sources": 5,
                    "summary": "Live from Memgraph",
                }
            ]

        async def fake_relations(workspace, *, valid_node_ids=None, max_edges=600):
            return []

        monkeypatch.setattr(gr, "read_graph_entities", fake_entities)
        monkeypatch.setattr(gr, "read_graph_relations", fake_relations)

        r = await client.get("/graph/entities")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["name"] == "Oracle Database"
        assert body[0]["summary"] == "Live from Memgraph"

    async def test_relations_uses_memgraph_when_entities_present(
        self, monkeypatch, client
    ):
        async def fake_entities(workspace, *, max_nodes=200):
            return [
                {
                    "id": "kg_A",
                    "name": "A",
                    "type": "PRODUCT",
                    "x": 100,
                    "y": 100,
                    "mentions": 1,
                    "sources": 1,
                    "summary": "",
                },
                {
                    "id": "kg_B",
                    "name": "B",
                    "type": "PRODUCT",
                    "x": 200,
                    "y": 200,
                    "mentions": 1,
                    "sources": 1,
                    "summary": "",
                },
            ]

        async def fake_relations(workspace, *, valid_node_ids=None, max_edges=600):
            # The router must pass the node ids it just read so dangling
            # edges don't surface.
            assert valid_node_ids is not None
            assert "kg_A" in valid_node_ids
            return [
                {
                    "id": "kr_000000",
                    "source": "kg_A",
                    "target": "kg_B",
                    "label": "USES",
                    "strength": 0.7,
                }
            ]

        monkeypatch.setattr(gr, "read_graph_entities", fake_entities)
        monkeypatch.setattr(gr, "read_graph_relations", fake_relations)

        r = await client.get("/graph/relations")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["source"] == "kg_A"
        assert body[0]["target"] == "kg_B"

    async def test_entities_falls_back_to_seed_when_memgraph_empty(
        self, monkeypatch, client
    ):
        async def fake_empty(workspace, *, max_nodes=200):
            return []

        monkeypatch.setattr(gr, "read_graph_entities", fake_empty)

        r = await client.get("/graph/entities")
        assert r.status_code == 200
        body = r.json()
        # The seed has 19 entities and their ids start with ``e_``
        assert len(body) > 0
        assert any(e["id"] == "e_oracle" for e in body)

    async def test_relations_falls_back_to_seed_when_memgraph_empty(
        self, monkeypatch, client
    ):
        async def fake_empty_entities(workspace, *, max_nodes=200):
            return []

        # Relations should never be queried when entities are empty —
        # set up a tripwire so we'd notice if the route changed shape.
        async def must_not_be_called(*args, **kwargs):
            raise AssertionError("read_graph_relations called despite empty entities")

        monkeypatch.setattr(gr, "read_graph_entities", fake_empty_entities)
        monkeypatch.setattr(gr, "read_graph_relations", must_not_be_called)

        r = await client.get("/graph/relations")
        assert r.status_code == 200
        body = r.json()
        # Seed has 21 relations
        assert len(body) > 0
        assert any(rel["id"] == "r_01" for rel in body)


class TestGraphPatchPersistence:
    async def test_patch_entity_returns_updated_shape_and_emits_activity(
        self, monkeypatch, client
    ):
        async def fake_update(workspace, entity_id, patch):
            return {
                "id": entity_id,
                "name": "Renamed",
                "type": "PRODUCT",
                "x": 100,
                "y": 200,
                "mentions": 3,
                "sources": 3,
                "summary": "After edit",
                "tags": ["critical", "db"],
                "properties": {"owner": "dba"},
            }

        monkeypatch.setattr(gr, "update_graph_entity", fake_update)

        r = await client.patch(
            "/graph/entities/kg_oracle",
            json={"name": "Renamed", "summary": "After edit"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Renamed"
        assert body["summary"] == "After edit"
        assert body["tags"] == ["critical", "db"]
        assert body["properties"] == {"owner": "dba"}

        # Activity event should be appended
        activity = await client.get("/activity")
        assert activity.status_code == 200
        events = activity.json().get("items", [])
        graph_events = [e for e in events if e["kind"] == "graph-entity-edited"]
        assert len(graph_events) == 1
        assert graph_events[0]["target"]["label"] == "Renamed"
        assert "name" in graph_events[0]["meta"]["patch_keys"]
        assert "summary" in graph_events[0]["meta"]["patch_keys"]

    async def test_patch_entity_404_when_not_found(self, monkeypatch, client):
        async def fake_update_missing(workspace, entity_id, patch):
            return None

        monkeypatch.setattr(gr, "update_graph_entity", fake_update_missing)

        r = await client.patch(
            "/graph/entities/kg_nope",
            json={"summary": "x"},
        )
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    async def test_patch_relation_returns_updated_shape_and_emits_activity(
        self, monkeypatch, client
    ):
        async def fake_update(workspace, rel_id, patch):
            return {
                "id": rel_id,
                "source": "kg_A",
                "target": "kg_B",
                "label": "USES",
                "strength": 0.95,
                "properties": {"since": "2024"},
            }

        monkeypatch.setattr(gr, "update_graph_relation", fake_update)

        r = await client.patch(
            "/graph/relations/kr_abc123",
            json={"label": "USES", "strength": 0.95},
        )
        assert r.status_code == 200
        assert r.json()["label"] == "USES"
        assert r.json()["properties"] == {"since": "2024"}

        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        rel_events = [e for e in events if e["kind"] == "graph-relation-edited"]
        assert len(rel_events) == 1
        assert rel_events[0]["target"]["label"] == "USES"

    async def test_patch_relation_404_when_cache_cold(self, monkeypatch, client):
        async def fake_update_missing(workspace, rel_id, patch):
            return None

        monkeypatch.setattr(gr, "update_graph_relation", fake_update_missing)

        r = await client.patch(
            "/graph/relations/kr_unknown",
            json={"strength": 0.5},
        )
        assert r.status_code == 404
        # The 404 message should include a hint about refreshing
        assert "refresh" in r.json()["detail"].lower()


class TestGraphLifecycle:
    async def test_post_entity_created_201_with_projection(
        self, monkeypatch, client
    ):
        async def fake_create(workspace, payload, *, actor="operator"):
            return {
                "id": "kg_NewEntity",
                "name": "NewEntity",
                "type": "PRODUCT",
                "x": 400,
                "y": 300,
                "mentions": 0,
                "sources": 0,
                "summary": "Operator-added.",
            }

        monkeypatch.setattr(gr, "create_graph_entity", fake_create)

        r = await client.post(
            "/graph/entities",
            json={"name": "NewEntity", "type": "PRODUCT", "summary": "Operator-added."},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["id"] == "kg_NewEntity"
        assert body["type"] == "PRODUCT"

        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        creates = [
            e
            for e in events
            if e["kind"] == "graph-entity-edited"
            and e["meta"].get("operation") == "create"
        ]
        assert len(creates) == 1

    async def test_post_entity_409_on_duplicate(self, monkeypatch, client):
        async def fake_create_dup(workspace, payload, *, actor="operator"):
            return None  # signals existing

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_dup)

        r = await client.post(
            "/graph/entities",
            json={"name": "Existing", "type": "PRODUCT"},
        )
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"]

    async def test_delete_entity_204_and_audit(self, monkeypatch, client):
        async def fake_delete(workspace, webui_id):
            return True

        monkeypatch.setattr(gr, "delete_graph_entity", fake_delete)

        r = await client.delete("/graph/entities/kg_to-remove")
        assert r.status_code == 204

        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        deletes = [
            e
            for e in events
            if e["kind"] == "graph-entity-edited"
            and e["meta"].get("operation") == "delete"
        ]
        assert len(deletes) == 1

    async def test_delete_entity_404_when_missing(self, monkeypatch, client):
        async def fake_delete_missing(workspace, webui_id):
            return False

        monkeypatch.setattr(gr, "delete_graph_entity", fake_delete_missing)

        r = await client.delete("/graph/entities/kg_nope")
        assert r.status_code == 404

    async def test_post_relation_created_201(self, monkeypatch, client):
        async def fake_create_rel(workspace, payload):
            return {
                "id": "kr_xyz",
                "source": payload["source"],
                "target": payload["target"],
                "label": payload["label"].upper().replace(" ", "_"),
                "strength": payload.get("strength", 0.5),
            }

        monkeypatch.setattr(gr, "create_graph_relation", fake_create_rel)

        r = await client.post(
            "/graph/relations",
            json={
                "source": "kg_A",
                "target": "kg_B",
                "label": "uses",
                "strength": 0.9,
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["source"] == "kg_A"
        assert body["target"] == "kg_B"
        assert body["label"] == "USES"

        activity = await client.get("/activity")
        events = activity.json().get("items", [])
        creates = [
            e
            for e in events
            if e["kind"] == "graph-relation-edited"
            and e["meta"].get("operation") == "create"
        ]
        assert len(creates) == 1

    async def test_post_relation_422_when_endpoint_missing(
        self, monkeypatch, client
    ):
        async def fake_create_rel_missing(workspace, payload):
            return None

        monkeypatch.setattr(gr, "create_graph_relation", fake_create_rel_missing)

        r = await client.post(
            "/graph/relations",
            json={"source": "kg_A", "target": "kg_phantom", "label": "uses"},
        )
        assert r.status_code == 422

    async def test_delete_relation_204(self, monkeypatch, client):
        async def fake_delete_rel(workspace, rel_id):
            return True

        monkeypatch.setattr(gr, "delete_graph_relation", fake_delete_rel)

        r = await client.delete("/graph/relations/kr_xyz")
        assert r.status_code == 204

    async def test_delete_relation_404(self, monkeypatch, client):
        async def fake_delete_rel_missing(workspace, rel_id):
            return False

        monkeypatch.setattr(
            gr, "delete_graph_relation", fake_delete_rel_missing
        )

        r = await client.delete("/graph/relations/kr_phantom")
        assert r.status_code == 404
        assert "refresh" in r.json()["detail"].lower()
