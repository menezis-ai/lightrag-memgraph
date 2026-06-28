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


# ── write-side folder cloisonnement (audit Graph/KG #1 + #2) ───────────────
#
# The GET path drops out-of-folder entities/relations and masks mixed
# provenance. Before this fix, the PATCH/DELETE helpers matched globally by id
# (workspace label only) and re-projected without scoping — so a caller in
# folder A could mutate a B-only object, and a mixed edit leaked B text in the
# response. These contract tests pin the gate (no write when out-of-folder) and
# the scoped re-projection (masked payload on the response). They run at the
# graph_reader function layer because the routes monkeypatch these helpers.

from contextlib import asynccontextmanager  # noqa: E402


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        async def _gen():
            for r in self._rows:
                yield r

        return _gen()

    async def consume(self):
        return None


class _RecordingSession:
    """Yields preset rows for every ``run`` and records the query strings so a
    test can assert whether a mutating statement actually executed."""

    def __init__(self, rows=None):
        self.rows = rows or []
        self.queries: list[str] = []

    async def run(self, query, **_params):
        self.queries.append(query)
        return _RowsResult(self.rows)


def _cm(session):
    @asynccontextmanager
    async def _factory():
        yield session

    return _factory


def _ent_row(source_id, description="x"):
    """A full entity row as `_read_one_entity`'s query returns it."""
    return {
        "entity_id": "e1",
        "entity_type": "CONCEPT",
        "description": description,
        "source_id": source_id,
        "display_name": "e1",
        "twin_tags_json": None,
        "twin_props_json": None,
    }


def _rel_row(chunk_source_id, keywords="uses"):
    return {
        "source_id": "e1",
        "target_id": "e2",
        "keywords": keywords,
        "weight": 1.0,
        "chunk_source_id": chunk_source_id,
        "twin_props_json": None,
    }


@pytest.fixture
def folder_a(monkeypatch):
    """Bind active folder A: member docs = {doc-a}, chunk→doc = _CTD."""
    import twindb_lightrag_memgraph.server.folder as folder_mod

    monkeypatch.setattr(folder_mod, "active_folder_id", lambda: "A")

    async def _ctd(_ws):
        return _CTD

    async def _md(_ws, _folder):
        return {"doc-a"}

    monkeypatch.setattr(graph_reader, "_load_chunk_to_doc_index", _ctd)
    monkeypatch.setattr(graph_reader, "_load_member_docs", _md)
    monkeypatch.setattr(graph_reader, "acquire_write_slot", _cm(None))


class TestUpdateEntityFolderGate:
    async def test_patch_b_only_entity_from_folder_a_is_refused(
        self, folder_a, monkeypatch
    ):
        # e1's only source chunk is chunk-b (doc-b) → invisible in folder A.
        read = _RecordingSession([_ent_row("chunk-b")])
        write = _RecordingSession([{"entity_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        out = await graph_reader.update_graph_entity(
            "ws", "kg_e1", {"summary": "tampered"}
        )
        assert out is None  # 404 to the caller
        # The gate must short-circuit BEFORE the write — no SET ran.
        assert not any("SET n +=" in q for q in write.queries)

    async def test_patch_member_entity_writes_and_returns_scoped(
        self, folder_a, monkeypatch
    ):
        read = _RecordingSession([_ent_row("chunk-a", description="A content")])
        write = _RecordingSession([{"entity_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        out = await graph_reader.update_graph_entity(
            "ws", "kg_e1", {"summary": "A content"}
        )
        assert out is not None
        assert out["source_docs"] == ["doc-a"]
        assert any("SET n +=" in q for q in write.queries)  # write happened

    async def test_patch_mixed_entity_is_refused_no_write(
        self, folder_a, monkeypatch
    ):
        # member chunk-a + non-member chunk-b → MIXED. The node is shared with
        # folder B; a global SET would corrupt B's view. Mutation must be refused
        # (409), NOT merely masked in the response. Visibility ≠ mutability.
        read = _RecordingSession(
            [_ent_row("chunk-a<SEP>chunk-b", description="secret B content")]
        )
        write = _RecordingSession([{"entity_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        with pytest.raises(graph_reader.MixedProvenanceError):
            await graph_reader.update_graph_entity("ws", "kg_e1", {"summary": "x"})
        # The shared node must be untouched.
        assert not any("SET n +=" in q for q in write.queries)


class TestDeleteEntityFolderGate:
    async def test_delete_b_only_entity_from_folder_a_is_refused(
        self, folder_a, monkeypatch
    ):
        read = _RecordingSession([_ent_row("chunk-b")])
        write = _RecordingSession([])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        ok = await graph_reader.delete_graph_entity("ws", "kg_e1")
        assert ok is False
        assert not any("DETACH DELETE" in q for q in write.queries)

    async def test_delete_member_entity_proceeds(self, folder_a, monkeypatch):
        read = _RecordingSession([_ent_row("chunk-a")])
        write = _RecordingSession([])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        ok = await graph_reader.delete_graph_entity("ws", "kg_e1")
        assert ok is True
        assert any("DETACH DELETE" in q for q in write.queries)

    async def test_delete_mixed_entity_is_refused_no_write(
        self, folder_a, monkeypatch
    ):
        # Shared node: a global DETACH DELETE would remove it from folder B too.
        read = _RecordingSession([_ent_row("chunk-a<SEP>chunk-b")])
        write = _RecordingSession([])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        with pytest.raises(graph_reader.MixedProvenanceError):
            await graph_reader.delete_graph_entity("ws", "kg_e1")
        assert not any("DETACH DELETE" in q for q in write.queries)


class TestRelationFolderGate:
    async def test_patch_b_only_relation_is_refused(self, folder_a, monkeypatch):
        # relation provenance chunk-b (doc-b) → invisible in folder A.
        monkeypatch.setattr(
            graph_reader, "lookup_relation_endpoints",
            lambda rid: ("ws", "e1", "e2"),
        )
        read = _RecordingSession([_rel_row("chunk-b")])
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        out = await graph_reader.update_graph_relation(
            "ws", "kr_x", {"label": "tampered"}
        )
        assert out is None
        assert not any("SET r +=" in q for q in write.queries)

    async def test_delete_b_only_relation_is_refused(self, folder_a, monkeypatch):
        monkeypatch.setattr(
            graph_reader, "lookup_relation_endpoints",
            lambda rid: ("ws", "e1", "e2"),
        )
        read = _RecordingSession([_rel_row("chunk-b")])
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        ok = await graph_reader.delete_graph_relation("ws", "kr_x")
        assert ok is False
        assert not any("DELETE r" in q for q in write.queries)

    async def test_patch_member_relation_writes(self, folder_a, monkeypatch):
        monkeypatch.setattr(
            graph_reader, "lookup_relation_endpoints",
            lambda rid: ("ws", "e1", "e2"),
        )
        read = _RecordingSession([_rel_row("chunk-a")])
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        out = await graph_reader.update_graph_relation(
            "ws", "kr_x", {"label": "uses"}
        )
        assert out is not None
        assert any("SET r +=" in q for q in write.queries)

    async def test_patch_mixed_relation_is_refused_no_write(
        self, folder_a, monkeypatch
    ):
        monkeypatch.setattr(
            graph_reader, "lookup_relation_endpoints",
            lambda rid: ("ws", "e1", "e2"),
        )
        # edge provenance spans chunk-a (member) + chunk-b (non-member) → mixed.
        read = _RecordingSession([_rel_row("chunk-a<SEP>chunk-b")])
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        with pytest.raises(graph_reader.MixedProvenanceError):
            await graph_reader.update_graph_relation("ws", "kr_x", {"label": "x"})
        assert not any("SET r +=" in q for q in write.queries)

    async def test_delete_mixed_relation_is_refused_no_write(
        self, folder_a, monkeypatch
    ):
        monkeypatch.setattr(
            graph_reader, "lookup_relation_endpoints",
            lambda rid: ("ws", "e1", "e2"),
        )
        read = _RecordingSession([_rel_row("chunk-a<SEP>chunk-b")])
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        with pytest.raises(graph_reader.MixedProvenanceError):
            await graph_reader.delete_graph_relation("ws", "kr_x")
        assert not any("DELETE r" in q for q in write.queries)


class TestGlobalModeUnchanged:
    """No folder bound (native / legacy caller) → no gate, no scoping."""

    async def test_update_entity_global_has_no_gate(self, monkeypatch):
        import twindb_lightrag_memgraph.server.folder as folder_mod

        monkeypatch.setattr(folder_mod, "active_folder_id", lambda: None)
        # Even a B-only entity is editable globally (no folder → no membership).
        read = _RecordingSession([_ent_row("chunk-b", description="global")])
        write = _RecordingSession([{"entity_id": "e1"}])
        monkeypatch.setattr(graph_reader, "acquire_write_slot", _cm(None))
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        out = await graph_reader.update_graph_entity(
            "ws", "kg_e1", {"summary": "global"}
        )
        assert out is not None
        assert out["summary"] == "global"  # not masked, global projection
        assert any("SET n +=" in q for q in write.queries)


# ── create-relation endpoint cloisonnement (audit Graph/KG #2) ─────────────
#
# A relation created from folder A must not link to a B-only entity known by id.
# Both endpoints are gated pure-member: out-of-folder → 422 (the security gate),
# mixed (shared) endpoint → 409.


class _PerEntityReadSession:
    """Read session that returns a ``source_id`` row keyed by the queried
    entity id (the ``eid`` bind param of `_entity_mutation_gate`)."""

    def __init__(self, source_by_id):
        self.source_by_id = source_by_id

    async def run(self, _query, **params):
        eid = params.get("eid")
        rows = (
            [{"source_id": self.source_by_id[eid]}]
            if eid in self.source_by_id
            else []
        )
        return _RowsResult(rows)


class TestCreateRelationFolderGate:
    async def test_create_to_b_only_entity_is_refused(self, folder_a, monkeypatch):
        # src e1 member (chunk-a); tgt e2 is B-only (chunk-b) → absent → refuse.
        read = _PerEntityReadSession({"e1": "chunk-a", "e2": "chunk-b"})
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        out = await graph_reader.create_graph_relation(
            "ws", {"source": "e1", "target": "e2", "label": "uses"}
        )
        assert out is None  # 422 to the caller — no link to a B-only entity
        assert not any("MERGE" in q for q in write.queries)

    async def test_create_with_mixed_endpoint_is_refused(
        self, folder_a, monkeypatch
    ):
        read = _PerEntityReadSession(
            {"e1": "chunk-a", "em": "chunk-a<SEP>chunk-b"}
        )
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        with pytest.raises(graph_reader.MixedProvenanceError):
            await graph_reader.create_graph_relation(
                "ws", {"source": "e1", "target": "em", "label": "uses"}
            )
        assert not any("MERGE" in q for q in write.queries)

    async def test_create_between_members_proceeds(self, folder_a, monkeypatch):
        read = _PerEntityReadSession({"e1": "chunk-a", "e1b": "chunk-a"})
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_read_session", _cm(read))
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        async def fake_proj(_ws, _src, _tgt, *_a, **_k):
            return {
                "id": "kr_x", "source": "kg_e1", "target": "kg_e1b",
                "label": "USES", "strength": 0.5, "properties": {},
            }

        monkeypatch.setattr(graph_reader, "_read_one_relation", fake_proj)

        out = await graph_reader.create_graph_relation(
            "ws", {"source": "e1", "target": "e1b", "label": "uses"}
        )
        assert out is not None
        assert any("MERGE" in q for q in write.queries)

    async def test_create_global_mode_uses_entity_exists(self, monkeypatch):
        import twindb_lightrag_memgraph.server.folder as folder_mod

        monkeypatch.setattr(folder_mod, "active_folder_id", lambda: None)
        monkeypatch.setattr(graph_reader, "acquire_write_slot", _cm(None))

        async def yes(_ws, _eid):
            return True

        monkeypatch.setattr(graph_reader, "entity_exists", yes)
        write = _RecordingSession([{"source_id": "e1"}])
        monkeypatch.setattr(graph_reader, "get_session", _cm(write))

        async def fake_proj(_ws, _src, _tgt, *_a, **_k):
            return {
                "id": "kr_x", "source": "kg_e1", "target": "kg_e2",
                "label": "USES", "strength": 0.5, "properties": {},
            }

        monkeypatch.setattr(graph_reader, "_read_one_relation", fake_proj)

        out = await graph_reader.create_graph_relation(
            "ws", {"source": "e1", "target": "e2", "label": "uses"}
        )
        assert out is not None  # global path unchanged: existence-only check
        assert any("MERGE" in q for q in write.queries)


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


# ── write-side cloisonnement, live (audit Graph/KG #1 + #2) ────────────────


@pytestmark_integration
async def test_update_entity_live_refused_out_of_folder(seeded):
    """From folder A, a PATCH on e2 (doc-b, B-only) must be refused and write
    nothing — the real MEMBER_OF traversal, not a mocked membership set."""
    import twindb_lightrag_memgraph.server.folder as folder_mod

    tok = _bind_folder("A")
    try:
        out = await graph_reader.update_graph_entity(
            _WS, "kg_e2", {"summary": "tampered from A"}
        )
        assert out is None
    finally:
        folder_mod._active_folder_id.reset(tok)

    # Prove no write landed: e2's description is untouched (read it in folder B).
    tok = _bind_folder("B")
    try:
        ents = await graph_reader.read_graph_entities(_WS)
        e2 = next(e for e in ents if e["name"] == "e2")
        assert e2["summary"] == "B"  # original, not "tampered from A"
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_update_entity_live_allowed_in_folder(seeded):
    """From folder A, a PATCH on e1 (doc-a, member) succeeds and the response is
    folder-scoped (source_docs == [doc-a])."""
    import twindb_lightrag_memgraph.server.folder as folder_mod

    tok = _bind_folder("A")
    try:
        out = await graph_reader.update_graph_entity(
            _WS, "kg_e1", {"summary": "A content edited"}
        )
        assert out is not None
        assert out["summary"] == "A content edited"
        assert out["source_docs"] == ["doc-a"]
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_delete_entity_live_refused_out_of_folder(seeded):
    """From folder A, DELETE on e2 (B-only) must be refused; e2 still exists."""
    import twindb_lightrag_memgraph.server.folder as folder_mod

    tok = _bind_folder("A")
    try:
        ok = await graph_reader.delete_graph_entity(_WS, "kg_e2")
        assert ok is False
    finally:
        folder_mod._active_folder_id.reset(tok)

    tok = _bind_folder("B")
    try:
        ents = await graph_reader.read_graph_entities(_WS)
        assert "e2" in {e["name"] for e in ents}  # survived the refused delete
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_update_relation_live_refused_out_of_folder(seeded):
    """The e1→e2 edge's provenance is chunk-b (doc-b). Primed via a folder-B
    read, a PATCH from folder A must be refused (gate), leaving it unchanged."""
    import twindb_lightrag_memgraph.server.folder as folder_mod

    # Prime the endpoint cache from folder B (where the edge is visible).
    tok = _bind_folder("B")
    try:
        rels = await graph_reader.read_graph_relations(_WS)
        assert len(rels) == 1
        rel_id = rels[0]["id"]
    finally:
        folder_mod._active_folder_id.reset(tok)

    tok = _bind_folder("A")
    try:
        out = await graph_reader.update_graph_relation(
            _WS, rel_id, {"label": "tampered"}
        )
        assert out is None  # gated: edge provenance is doc-b, not in folder A
    finally:
        folder_mod._active_folder_id.reset(tok)


@pytestmark_integration
async def test_update_entity_live_refused_when_mixed(seeded):
    """A node co-owned by A and B (source chunks chunk-a + chunk-b) must NOT be
    mutable from folder A: a global SET would corrupt B's view. Expect a refusal
    (MixedProvenanceError) and the node left unchanged."""
    import twindb_lightrag_memgraph.server.folder as folder_mod
    from twindb_lightrag_memgraph import _pool

    # e3 is sourced from both chunk-a (doc-a ∈ A) and chunk-b (doc-b ∈ B).
    async with _pool.get_session() as s:
        await (
            await s.run(
                f"CREATE (:`{_WS}` {{entity_id:'e3', entity_type:'CONCEPT', "
                f"source_id:'chunk-a<SEP>chunk-b', description:'shared'}})"
            )
        ).consume()

    tok = _bind_folder("A")
    try:
        with pytest.raises(graph_reader.MixedProvenanceError):
            await graph_reader.update_graph_entity(
                _WS, "kg_e3", {"summary": "tampered from A"}
            )
    finally:
        folder_mod._active_folder_id.reset(tok)

    # The shared node's description must be intact (read globally, no folder).
    async with _pool.get_read_session() as s:
        result = await s.run(
            f"MATCH (n:`{_WS}` {{entity_id:'e3'}}) RETURN n.description AS d"
        )
        row = await result.single()
        await result.consume()
    assert row["d"] == "shared"  # unchanged


@pytestmark_integration
async def test_create_relation_live_refused_to_out_of_folder_endpoint(seeded):
    """From folder A, creating a relation e1→e2 (e2 is doc-b, B-only) must be
    refused: no edge linking into another folder's entity by id. No edge lands."""
    import twindb_lightrag_memgraph.server.folder as folder_mod
    from twindb_lightrag_memgraph import _pool

    tok = _bind_folder("A")
    try:
        out = await graph_reader.create_graph_relation(
            _WS, {"source": "e1", "target": "e2", "label": "leaks"}
        )
        assert out is None  # e2 is out-of-folder for A → refused (422)
    finally:
        folder_mod._active_folder_id.reset(tok)

    # No DIRECTED edge e1→e2 should exist (the seeded edge is e1→e2 with
    # provenance chunk-b; assert no edge carries the new 'leaks' keyword).
    async with _pool.get_read_session() as s:
        result = await s.run(
            f"MATCH (:`{_WS}` {{entity_id:'e1'}})-[r:DIRECTED]->"
            f"(:`{_WS}` {{entity_id:'e2'}}) RETURN r.keywords AS k"
        )
        rows = [rec["k"] async for rec in result]
        await result.consume()
    assert "leaks" not in rows  # the refused create wrote nothing
