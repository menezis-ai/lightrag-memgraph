"""Folder cloisonnement of the WebUI Knowledge Graph — batch 3.

The graph tab (`/twin/api/graph/entities` + `/relations`) must not show
entities, relations or `source_docs` from another folder. Acceptance:

- entity visible in folder X iff ≥1 of its `source_id` chunks belongs to a doc
  MEMBER_OF X;
- relation visible iff ≥1 of its OWN `source_id` chunks (not its endpoints')
  belongs to a member doc;
- `source_docs` / `sources` / `mentions` scoped to the active folder;
- no active folder (off the Twin routes) → legacy global behaviour.

Layers: pure projector unit tests, `read_graph_native` with mocked index
loaders (the route's real path), route contract tests with an `X-Twin-Folder`
header, and a live integration pass over `read_graph_entities` / `relations`.
"""

from __future__ import annotations

import json

import pytest

from twindb_lightrag_memgraph.server import graph_reader


# ── pure projectors ───────────────────────────────────────────────────────


class _Node:
    def __init__(self, props):
        self.properties = props
        self.id = props.get("entity_id")


class _Edge:
    def __init__(self, source, target, props):
        self.source = source
        self.target = target
        self.properties = props


_CTD = {"chunk-a": "doc-a", "chunk-b": "doc-b"}  # chunk → doc


class TestEntityProjectorScoping:
    def test_visible_when_a_source_chunk_is_member(self):
        row = {"entity_id": "e1", "source_id": "chunk-a<SEP>chunk-b"}
        ent = graph_reader._node_record_to_entity(row, _CTD, {"doc-a"})
        assert ent is not None
        # source_docs scoped to the member doc only (NOT doc-b)
        assert ent["source_docs"] == ["doc-a"]
        assert ent["sources"] == 1
        assert ent["mentions"] == 1  # only the member chunk counts

    def test_dropped_when_no_source_chunk_is_member(self):
        row = {"entity_id": "e1", "source_id": "chunk-b"}
        assert graph_reader._node_record_to_entity(row, _CTD, {"doc-a"}) is None

    def test_global_when_member_docs_none(self):
        row = {"entity_id": "e1", "source_id": "chunk-a<SEP>chunk-b"}
        ent = graph_reader._node_record_to_entity(row, _CTD, None)
        assert ent is not None
        assert ent["source_docs"] == ["doc-a", "doc-b"]  # both, unscoped

    def test_mixed_entity_visible_but_summary_masked(self):
        # chunk-a (doc-a, member) + chunk-b (doc-b, NOT member) → mixed.
        # The node + scoped source_docs stay, but the blended description is
        # masked (mixed record is structurally visible, text payload hidden
        # because LightRAG stores a single blended description).
        row = {
            "entity_id": "e1",
            "source_id": "chunk-a<SEP>chunk-b",
            "description": "secret B content",
        }
        ent = graph_reader._node_record_to_entity(row, _CTD, {"doc-a"})
        assert ent is not None
        assert ent["source_docs"] == ["doc-a"]
        assert ent["summary"] == graph_reader._MASKED_ENTITY_SUMMARY
        assert "secret B" not in ent["summary"]

    def test_pure_member_entity_keeps_summary(self):
        # all source chunks in-folder → not mixed → real description kept.
        row = {"entity_id": "e1", "source_id": "chunk-a", "description": "A content"}
        ent = graph_reader._node_record_to_entity(row, _CTD, {"doc-a"})
        assert ent is not None
        assert ent["summary"] == "A content"


class TestRelationProjectorScoping:
    def test_relation_scoped_by_its_own_source_chunk(self):
        # endpoints are entity ids; chunk_source_id is the relation provenance.
        row = {
            "source_id": "e1",
            "target_id": "e2",
            "chunk_source_id": "chunk-b",
            "keywords": "rel",
        }
        # relation extracted from chunk-b (doc-b) → hidden in folder of doc-a
        assert graph_reader._edge_record_to_relation(row, 0, _CTD, {"doc-a"}) is None
        # visible in doc-b's folder
        rel = graph_reader._edge_record_to_relation(row, 0, _CTD, {"doc-b"})
        assert rel is not None
        assert rel["source"] == graph_reader._entity_id_to_node_id("e1")

    def test_relation_global_when_member_docs_none(self):
        row = {"source_id": "e1", "target_id": "e2", "chunk_source_id": "chunk-b"}
        assert graph_reader._edge_record_to_relation(row, 0) is not None

    def test_mixed_relation_visible_but_text_masked(self):
        # chunk-a (member) + chunk-b (non-member) → mixed: edge stays, but the
        # keyword-derived label and operator props are neutralised.
        row = {
            "source_id": "e1",
            "target_id": "e2",
            "chunk_source_id": "chunk-a<SEP>chunk-b",
            "keywords": "leaks B",
            "twin_props_json": json.dumps({"note": "from B"}),
        }
        rel = graph_reader._edge_record_to_relation(row, 0, _CTD, {"doc-a"})
        assert rel is not None
        assert rel["label"] == graph_reader._MIXED_RELATION_LABEL
        assert rel["properties"] == {}

    def test_pure_member_relation_keeps_text(self):
        row = {
            "source_id": "e1",
            "target_id": "e2",
            "chunk_source_id": "chunk-a",
            "keywords": "uses",
        }
        rel = graph_reader._edge_record_to_relation(row, 0, _CTD, {"doc-a"})
        assert rel is not None
        assert rel["label"] == "USES"


# ── read_graph_native (the route's path) with mocked index loaders ────────


class _KG:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


class _FakeRag:
    def __init__(self, kg):
        self._kg = kg

    async def get_knowledge_graph(self, *, node_label, max_depth, max_nodes):
        return self._kg


@pytest.fixture
def patched_loaders(monkeypatch):
    async def _ctd(_ws):
        return _CTD

    monkeypatch.setattr(graph_reader, "_load_chunk_to_doc_index", _ctd)

    def _set_members(members):
        async def _md(_ws, _folder):
            return set(members)

        monkeypatch.setattr(graph_reader, "_load_member_docs", _md)

    return _set_members


def _kg_two_entities():
    # e1 sourced from chunk-a (doc-a), e2 from chunk-b (doc-b); one edge e1-e2
    # whose own provenance is chunk-a (doc-a).
    nodes = [
        _Node({"entity_id": "e1", "source_id": "chunk-a", "description": "A"}),
        _Node({"entity_id": "e2", "source_id": "chunk-b", "description": "B"}),
    ]
    edges = [_Edge("e1", "e2", {"source_id": "chunk-a", "keywords": "k"})]
    return _KG(nodes, edges)


async def test_native_scopes_entities_and_relations(patched_loaders, monkeypatch):
    patched_loaders({"doc-a"})  # folder A: only doc-a is a member
    # active_folder_id is imported lazily inside _active_member_docs.
    import twindb_lightrag_memgraph.server.folder as folder_mod

    monkeypatch.setattr(folder_mod, "active_folder_id", lambda: "A")

    rag = _FakeRag(_kg_two_entities())
    entities, relations = await graph_reader.read_graph_native(rag, "ws")
    ids = {e["name"] for e in entities}
    assert ids == {"e1"}  # e2 (doc-b) hidden
    # the edge e1-e2 drops: e2 is not a visible node (endpoint filter), and its
    # own provenance chunk-a IS member but the target node is gone.
    assert relations == []


async def test_native_global_when_no_folder(patched_loaders, monkeypatch):
    patched_loaders({"doc-a"})  # ignored: no folder bound
    import twindb_lightrag_memgraph.server.folder as folder_mod

    monkeypatch.setattr(folder_mod, "active_folder_id", lambda: None)

    rag = _FakeRag(_kg_two_entities())
    entities, relations = await graph_reader.read_graph_native(rag, "ws")
    assert {e["name"] for e in entities} == {"e1", "e2"}
    assert len(relations) == 1  # global: edge kept


# ── route wiring: X-Twin-Folder header drives the scoping ─────────────────


@pytest.fixture
async def graph_client(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default", "kind": "primary"},
                {"id": "A", "label": "Folder A", "kind": "sandbox"},
            ]
        ),
    )
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from twindb_lightrag_memgraph import _twindb_state
    from twindb_lightrag_memgraph.server import webui_router

    # No DB: stub the two index loaders the scoping path needs.
    async def _ctd(_ws):
        return _CTD

    async def _md(_ws, folder):
        return {"doc-a"} if folder == "A" else set()

    monkeypatch.setattr(graph_reader, "_load_chunk_to_doc_index", _ctd)
    monkeypatch.setattr(graph_reader, "_load_member_docs", _md)

    webui_router.reset_store()
    _twindb_state["rag"] = _FakeRag(_kg_two_entities())
    app = FastAPI()
    app.include_router(webui_router.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    _twindb_state.pop("rag", None)
    webui_router.reset_store()


async def test_graph_entities_route_scopes_by_folder_header(graph_client):
    r = await graph_client.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    assert r.status_code == 200
    names = {e["name"] for e in r.json()}
    assert names == {"e1"}  # e2 (doc-b) hidden in folder A


async def test_graph_relations_route_scopes_by_folder_header(graph_client):
    # edge provenance is chunk-a (doc-a ∈ A), but its target e2 is hidden in A
    # → endpoint filter drops the relation. Net: no cross-folder relation leaks.
    r = await graph_client.get("/graph/relations", headers={"X-Twin-Folder": "A"})
    assert r.status_code == 200
    assert r.json() == []


async def test_empty_scoped_folder_never_falls_back_to_seed(monkeypatch):
    """A folder that scopes to zero entities must return [] — NOT the unscoped
    seed graph — even when the seed fallback would otherwise be allowed
    (seed-mode store + IdP dormant). Regression guard for the seed-leak (P1)."""
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": "A", "label": "A", "kind": "sandbox"}]),
    )
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from twindb_lightrag_memgraph import _twindb_state
    from twindb_lightrag_memgraph.server import webui_router

    async def _ctd(_ws):
        return _CTD

    async def _md(_ws, _folder):
        return {"doc-a"}  # folder A members

    monkeypatch.setattr(graph_reader, "_load_chunk_to_doc_index", _ctd)
    monkeypatch.setattr(graph_reader, "_load_member_docs", _md)

    # seed-mode store + no IdP → seed fallback WOULD be allowed, if not gated.
    webui_router.set_store(webui_router.WebuiStore.from_seed())
    # KG holds only e2 (doc-b) — entirely outside folder A → scoped-empty.
    only_non_member = _KG(
        [_Node({"entity_id": "e2", "source_id": "chunk-b", "description": "B"})], []
    )
    _twindb_state["rag"] = _FakeRag(only_non_member)
    app = FastAPI()
    app.include_router(webui_router.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/graph/entities", headers={"X-Twin-Folder": "A"})
        assert r.status_code == 200
        assert r.json() == []  # NOT the 19-entity seed graph
    finally:
        _twindb_state.pop("rag", None)
        webui_router.reset_store()


# ── live integration (real Memgraph) ──────────────────────────────────────

pytestmark_integration = pytest.mark.integration

_WS = "graph_webui_scope_test"


def _doc(doc_id: str) -> dict:
    return {
        "id": doc_id,
        "status": "processed",
        "content_summary": "x",
        "content_length": 1,
        "file_path": f"{doc_id}.md",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "content_hash": doc_id,
        "chunks_list": json.dumps([f"chunk-{doc_id[-1]}"]),
    }


@pytest.fixture
async def seeded(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", _WS)
    from twindb_lightrag_memgraph import _pool, register
    from twindb_lightrag_memgraph._constants import storage_folder_context
    from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

    register()
    ds = MemgraphDocStatusStorage(
        namespace="doc_status", global_config={}, embedding_func=None
    )
    await ds.initialize()

    async def _wipe():
        async with _pool.get_session() as s:
            for lbl in (f"DocStatus_{_WS}", f"Folder_{_WS}", _WS):
                await (await s.run(f"MATCH (n:`{lbl}`) DETACH DELETE n")).consume()

    await _wipe()
    # doc-a ∈ A (chunk-a), doc-b ∈ B (chunk-b)
    with storage_folder_context("A"):
        await ds.upsert({"doc-a": _doc("doc-a")})
    with storage_folder_context("B"):
        await ds.upsert({"doc-b": _doc("doc-b")})

    async with _pool.get_session() as s:
        await (
            await s.run(
                f"CREATE (:`{_WS}` {{entity_id:'e1', entity_type:'CONCEPT', "
                f"source_id:'chunk-a', description:'A'}}) "
                f"CREATE (:`{_WS}` {{entity_id:'e2', entity_type:'CONCEPT', "
                f"source_id:'chunk-b', description:'B'}})"
            )
        ).consume()
        await (
            await s.run(
                f"MATCH (a:`{_WS}` {{entity_id:'e1'}}), (b:`{_WS}` {{entity_id:'e2'}}) "
                f"CREATE (a)-[:DIRECTED {{keywords:'k', weight:1.0, "
                f"source_id:'chunk-b'}}]->(b)"
            )
        ).consume()
    yield
    await _wipe()


def _bind_folder(folder):
    import twindb_lightrag_memgraph.server.folder as folder_mod

    return folder_mod._active_folder_id.set(folder)


@pytestmark_integration
async def test_read_entities_live_scoped(seeded):
    import twindb_lightrag_memgraph.server.folder as folder_mod

    # global (no folder bound) → both entities
    ents = await graph_reader.read_graph_entities(_WS)
    assert {e["name"] for e in ents} == {"e1", "e2"}

    tok = _bind_folder("A")
    try:
        ents = await graph_reader.read_graph_entities(_WS)
        names = {e["name"] for e in ents}
        assert names == {"e1"}  # e2 (doc-b) hidden in folder A
        assert next(e for e in ents if e["name"] == "e1")["source_docs"] == ["doc-a"]
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_read_relations_live_scoped(seeded):
    import twindb_lightrag_memgraph.server.folder as folder_mod

    # the edge's own provenance is chunk-b (doc-b) → visible in B, not A
    tok = _bind_folder("A")
    try:
        rels = await graph_reader.read_graph_relations(_WS)
        assert rels == []
    finally:
        folder_mod._active_folder_id.reset(tok)

    tok = _bind_folder("B")
    try:
        rels = await graph_reader.read_graph_relations(_WS)
        assert len(rels) == 1
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_search_labels_scoped_to_folder(seeded):
    """P1: the /graph/search label search must not reveal out-of-folder labels.
    In folder A only e1 (doc-a) is reachable; e2 (doc-b) must never surface."""
    import twindb_lightrag_memgraph.server.folder as folder_mod

    tok = _bind_folder("A")
    try:
        # rag is unused on the scoped path (pure Cypher) — pass None.
        labels = await graph_reader.search_graph_labels(None, "e", workspace=_WS)
        assert set(labels) == {"e1"}  # e2 (folder B) NOT revealed
    finally:
        folder_mod._active_folder_id.reset(tok)

    tok = _bind_folder("C")  # empty folder → reveals nothing
    try:
        assert await graph_reader.search_graph_labels(None, "e", workspace=_WS) == []
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_read_entities_filters_before_limit(seeded):
    """P2: the membership predicate is pushed BEFORE the LIMIT, so a member
    entity isn't starved by a global cap that lands on non-member nodes."""
    import twindb_lightrag_memgraph.server.folder as folder_mod
    from twindb_lightrag_memgraph import _pool

    # Add several folder-B-only entities so a post-LIMIT filter with max_nodes=1
    # would very likely return a non-member (→ []). Pre-LIMIT makes e1 the only
    # candidate → deterministically returned.
    async with _pool.get_session() as s:
        for i in range(5):
            await (
                await s.run(
                    f"CREATE (:`{_WS}` {{entity_id:'eb{i}', entity_type:'CONCEPT', "
                    f"source_id:'chunk-b', description:'B{i}'}})"
                )
            ).consume()

    tok = _bind_folder("A")
    try:
        ents = await graph_reader.read_graph_entities(_WS, max_nodes=1)
        assert {e["name"] for e in ents} == {"e1"}
    finally:
        folder_mod._active_folder_id.reset(tok)
