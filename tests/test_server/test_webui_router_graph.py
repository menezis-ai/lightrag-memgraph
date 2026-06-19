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

        async def fake_native(
            rag, workspace, *, node_label="*", max_depth=3, max_nodes=1000
        ):
            return (await fake_entities(workspace), [])

        monkeypatch.setattr(gr, "read_graph_native", fake_native)
        monkeypatch.setattr(webui_router, "_get_rag", lambda: object())

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

        async def fake_native(
            rag, workspace, *, node_label="*", max_depth=3, max_nodes=1000
        ):
            # read_graph_native returns a CONSISTENT (entities, relations)
            # pair from one native subgraph selection — no dangling edges by
            # construction, so the route no longer threads valid_node_ids.
            return (
                await fake_entities(workspace),
                [
                    {
                        "id": "kr_000000",
                        "source": "kg_A",
                        "target": "kg_B",
                        "label": "USES",
                        "strength": 0.7,
                    }
                ],
            )

        monkeypatch.setattr(gr, "read_graph_native", fake_native)
        monkeypatch.setattr(webui_router, "_get_rag", lambda: object())

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


class TestGraphSeedFallbackGate:
    """Audit C5: the in-memory demo graph must only be served as a
    fallback when ``webui_stores='seed'`` AND no IdP is configured.

    The two tests above (``test_entities_falls_back_to_seed_when_
    memgraph_empty`` / ``test_relations_…``) pin the demo-mode path:
    seed-mode store + IdP dormant → seed surfaces, regression guard
    for back-compat with dev / standalone.

    The three tests below pin the production-mode path: any of
    {IdP active, memgraph store, unknown mode} → empty list,
    never the demo seed.
    """

    def test_helper_blocks_seed_when_idp_is_active(self, monkeypatch):
        """Unit-level: exercising the IdP branch through HTTP would
        require an end-to-end auth setup (the middleware rejects the
        request with 401 before the route runs). Test the helper
        directly — that's the unit under contract."""
        from twindb_lightrag_memgraph.server import idp_jwt

        # Force a seed-mode store so only the IdP gate could block.
        webui_router.set_store(webui_router.WebuiStore.from_seed())
        monkeypatch.setattr(
            idp_jwt,
            "get_active_config",
            lambda: idp_jwt.IdpConfig(jwks_url="https://test/jwks"),
        )
        assert webui_router._graph_seed_fallback_allowed() is False

    def test_helper_allows_seed_when_idp_dormant_and_store_seed(self, monkeypatch):
        from twindb_lightrag_memgraph.server import idp_jwt

        webui_router.set_store(webui_router.WebuiStore.from_seed())
        # IdP dormant is the test default, but pin it explicitly so a
        # future env change can't flip this assertion silently.
        monkeypatch.setattr(idp_jwt, "get_active_config", lambda: None)
        assert webui_router._graph_seed_fallback_allowed() is True

    def test_helper_blocks_seed_when_store_mode_is_memgraph(self, monkeypatch):
        from twindb_lightrag_memgraph.server import idp_jwt

        store = webui_router.WebuiStore.from_seed()
        store._mode = "memgraph"  # type: ignore[attr-defined]
        webui_router.set_store(store)
        monkeypatch.setattr(idp_jwt, "get_active_config", lambda: None)
        assert webui_router._graph_seed_fallback_allowed() is False

    def test_helper_blocks_seed_when_store_mode_is_unknown(self, monkeypatch):
        from twindb_lightrag_memgraph.server import idp_jwt

        store = webui_router.WebuiStore.from_seed()
        store._mode = "something-weird"  # type: ignore[attr-defined]
        webui_router.set_store(store)
        monkeypatch.setattr(idp_jwt, "get_active_config", lambda: None)
        assert webui_router._graph_seed_fallback_allowed() is False

    async def test_route_entities_returns_empty_when_store_mode_is_memgraph(
        self, monkeypatch, client
    ):
        """End-to-end: a memgraph-mode store reflects a production
        deploy. Even without IdP, the seed MUST NEVER leak via
        ``/graph/entities``. Seed fixtures are intentionally left
        populated on the store to prove the mode gate alone (not the
        data) blocks the leak — confirming the audit C5 doctrine
        that data and config must not be conflated."""

        async def fake_empty(workspace, *, max_nodes=200):
            return []

        monkeypatch.setattr(gr, "read_graph_entities", fake_empty)
        seed_like_store = webui_router.WebuiStore.from_seed()
        seed_like_store._mode = "memgraph"  # type: ignore[attr-defined]
        webui_router.set_store(seed_like_store)

        r = await client.get("/graph/entities")
        assert r.status_code == 200
        assert r.json() == []

    async def test_route_relations_returns_empty_when_store_mode_is_memgraph(
        self, monkeypatch, client
    ):
        async def fake_empty(workspace, *, max_nodes=200):
            return []

        monkeypatch.setattr(gr, "read_graph_entities", fake_empty)
        seed_like_store = webui_router.WebuiStore.from_seed()
        seed_like_store._mode = "memgraph"  # type: ignore[attr-defined]
        webui_router.set_store(seed_like_store)

        r = await client.get("/graph/relations")
        assert r.status_code == 200
        assert r.json() == []

    async def test_for_folder_memgraph_mode_sets_mode_attribute(self):
        """``WebuiStore.for_folder(..., mode='memgraph')`` must carry
        the explicit mode forward so the route-level gate can read it.
        Regression guard against a future refactor that loses the
        propagation between constructor argument and instance state."""

        store = webui_router.WebuiStore.for_folder(
            "default", mode="memgraph"
        )
        assert store.mode == "memgraph"


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
        """Honest 409: the function raises EntityExistsError, the
        route surfaces a truthful message naming the entity. Guards
        against a regression to the ``return None`` sentinel that used
        to conflate "duplicate" with "backend failed" (TR-KG-01)."""

        async def fake_create_dup(workspace, payload, *, actor="operator"):
            raise gr.EntityExistsError(payload["name"])

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_dup)

        r = await client.post(
            "/graph/entities",
            json={"name": "Existing", "type": "PRODUCT"},
        )
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert "already exists" in detail
        assert "Existing" in detail

    async def test_post_entity_422_on_empty_name(self, monkeypatch, client):
        """Pydantic validator gate: empty name is rejected before the
        function runs, so we never reach Memgraph (TR-KG-01)."""

        async def fake_create_never_called(workspace, payload, *, actor="operator"):
            raise AssertionError("handler should not invoke create on 422")

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_never_called)

        r = await client.post(
            "/graph/entities",
            json={"name": "", "type": "PRODUCT"},
        )
        assert r.status_code == 422

    async def test_post_entity_422_on_whitespace_name(self, monkeypatch, client):
        """Whitespace-only name is stripped to empty by the
        field_validator and rejected at 422. Order matters: if
        ``min_length`` ran before strip, this would slip through to the
        backend (TR-KG-01)."""

        async def fake_create_never_called(workspace, payload, *, actor="operator"):
            raise AssertionError("handler should not invoke create on 422")

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_never_called)

        r = await client.post(
            "/graph/entities",
            json={"name": "   ", "type": "PRODUCT"},
        )
        assert r.status_code == 422

    async def test_post_entity_503_on_backend_create_failure(
        self, monkeypatch, client
    ):
        """Memgraph CREATE fails → honest 503, not a misleading 409
        (TR-KG-01). The detail tells the operator to check server
        logs; the underlying driver message is not leaked."""

        async def fake_create_backend_fail(workspace, payload, *, actor="operator"):
            raise gr.EntityCreateBackendError(
                "Bolt driver: session closed"
            )

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_backend_fail)

        r = await client.post(
            "/graph/entities",
            json={"name": "FreshEntity", "type": "PRODUCT"},
        )
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "could not be created" in detail
        assert "Bolt driver" not in detail  # raw driver detail stays in logs

    async def test_post_entity_500_on_projection_failure(
        self, monkeypatch, client
    ):
        """Half-success: write committed, post-CREATE projection
        failed. We tell the operator to refresh
        ``/twin/api/graph/entities`` rather than pretending the create
        failed (TR-KG-01)."""

        async def fake_create_projection_fail(
            workspace, payload, *, actor="operator"
        ):
            raise gr.EntityProjectionError(payload["name"])

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_projection_fail)

        r = await client.post(
            "/graph/entities",
            json={"name": "WroteButCantProject", "type": "PRODUCT"},
        )
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert "was created" in detail
        assert "Refresh" in detail

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


class TestGraphEntityTagThesaurusBinding:
    """TR-KG-03 / QA audit 2026-06-12: node tags must come from
    the active tag catalog. Both PATCH and POST entry points enforce
    this; an unknown tag is a 422 with an explicit message naming
    the rejected values, not a silent acceptance into
    ``twin_tags_json``.

    The default test seed exposes the active tags ``rman`` and
    ``oracle`` (see ``server/webui_seed.TAGS``); we use those as
    "known" values and a deliberately fake string as the "unknown".
    """

    async def test_patch_entity_with_known_tag_passes(
        self, monkeypatch, client
    ):
        captured: dict[str, object] = {}

        async def fake_update(workspace, entity_id, patch):
            captured["patch"] = patch
            return {
                "id": entity_id,
                "name": "Oracle",
                "type": "PRODUCT",
                "x": 100,
                "y": 100,
                "mentions": 0,
                "sources": 0,
                "summary": "",
                "tags": list(patch.get("tags") or []),
                "properties": {},
            }

        monkeypatch.setattr(gr, "update_graph_entity", fake_update)

        r = await client.patch(
            "/graph/entities/kg_oracle",
            json={"tags": ["rman", "oracle"]},
        )
        assert r.status_code == 200
        assert captured["patch"]["tags"] == ["rman", "oracle"]

    async def test_patch_entity_with_unknown_tag_returns_422(
        self, monkeypatch, client
    ):
        async def fake_update_never_called(workspace, entity_id, patch):
            raise AssertionError(
                "graph_reader.update_graph_entity must not be reached"
            )

        monkeypatch.setattr(gr, "update_graph_entity", fake_update_never_called)

        r = await client.patch(
            "/graph/entities/kg_oracle",
            json={"tags": ["rman", "random-bullshit-bingo"]},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        # The rejected value appears verbatim so the operator can
        # correct it; the known one does NOT (it's not the problem).
        assert "random-bullshit-bingo" in detail
        assert "rman" not in detail.split("Allowed")[0]
        # The "Allowed" hint lists a bounded sample of the active
        # catalog so the caller knows what to type instead.
        assert "Allowed (active catalog):" in detail

    async def test_post_entity_with_known_tag_passes(
        self, monkeypatch, client
    ):
        captured: dict[str, object] = {}

        async def fake_create(workspace, payload, *, actor="operator"):
            captured["payload"] = payload
            return {
                "id": "kg_NewEntity",
                "name": payload["name"],
                "type": payload["type"],
                "x": 100,
                "y": 100,
                "mentions": 0,
                "sources": 0,
                "summary": payload.get("summary") or "",
                "tags": list(payload.get("tags") or []),
            }

        monkeypatch.setattr(gr, "create_graph_entity", fake_create)

        r = await client.post(
            "/graph/entities",
            json={"name": "NewEntity", "type": "PRODUCT", "tags": ["oracle"]},
        )
        assert r.status_code == 201
        assert captured["payload"]["tags"] == ["oracle"]

    async def test_post_entity_with_unknown_tag_returns_422(
        self, monkeypatch, client
    ):
        async def fake_create_never_called(workspace, payload, *, actor="operator"):
            raise AssertionError(
                "graph_reader.create_graph_entity must not be reached"
            )

        monkeypatch.setattr(gr, "create_graph_entity", fake_create_never_called)

        r = await client.post(
            "/graph/entities",
            json={
                "name": "NewEntity",
                "type": "PRODUCT",
                "tags": ["does-not-exist-in-catalog"],
            },
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "does-not-exist-in-catalog" in detail
        assert "Allowed (active catalog):" in detail

    async def test_patch_entity_with_empty_tags_list_skips_validation(
        self, monkeypatch, client
    ):
        """``tags: []`` means "clear all node tags"; that's a
        legitimate intent and must not be treated as an unknown tag."""
        async def fake_update(workspace, entity_id, patch):
            return {
                "id": entity_id,
                "name": "Oracle",
                "type": "PRODUCT",
                "x": 0,
                "y": 0,
                "mentions": 0,
                "sources": 0,
                "summary": "",
                "tags": [],
                "properties": {},
            }

        monkeypatch.setattr(gr, "update_graph_entity", fake_update)

        r = await client.patch(
            "/graph/entities/kg_oracle",
            json={"tags": []},
        )
        assert r.status_code == 200
        assert r.json()["tags"] == []


class TestGraphNativeDelegation:
    """The graph routes delegate node/edge selection + label search to
    LightRAG's native focus+context API (get_knowledge_graph / search_labels)
    instead of a flat LIMIT scan. Regression for the 'graph dénutri' recette
    finding (200 arbitrary nodes of a 17k-entity KB; searched entities absent).
    """

    async def test_search_delegates_to_native_search_labels(
        self, monkeypatch, client
    ):
        async def fake_search(rag, q, *, limit=50):
            assert q == "schizo"
            return ["Schizophrenia", "Schizoaffective Disorder"]

        monkeypatch.setattr(gr, "search_graph_labels", fake_search)
        monkeypatch.setattr(webui_router, "_get_rag", lambda: object())

        r = await client.get("/graph/search", params={"q": "schizo"})
        assert r.status_code == 200
        assert r.json() == ["Schizophrenia", "Schizoaffective Disorder"]

    async def test_entities_forwards_focus_label_to_native(
        self, monkeypatch, client
    ):
        seen: dict[str, object] = {}

        async def fake_native(
            rag, workspace, *, node_label="*", max_depth=3, max_nodes=1000
        ):
            seen["label"] = node_label
            seen["max_nodes"] = max_nodes
            return (
                [
                    {
                        "id": "kg_Schizophrenia",
                        "name": "Schizophrenia",
                        "type": "CONCEPT",
                        "x": 0,
                        "y": 0,
                        "mentions": 1,
                        "sources": 1,
                        "summary": "",
                    }
                ],
                [],
            )

        monkeypatch.setattr(gr, "read_graph_native", fake_native)
        monkeypatch.setattr(webui_router, "_get_rag", lambda: object())

        r = await client.get(
            "/graph/entities", params={"label": "Schizophrenia", "max_nodes": 500}
        )
        assert r.status_code == 200
        assert seen["label"] == "Schizophrenia"
        assert seen["max_nodes"] == 500
        assert r.json()[0]["name"] == "Schizophrenia"
