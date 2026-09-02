"""End-to-end contract test for the native graph read — `docs/test-doctrine-graph.md`.

That doctrine requires every PR touching ``server/graph_reader.py`` to add a
contract test beyond a driver-mocked benchmark. This file covers the real
Cypher → FastAPI half; ``lightrag_webui_twin/src/api/graphCacheScope.test.tsx``
covers the cache/refetch half when the active folder changes.

What is real here: a live Memgraph holds the DocStatus rows, the
``MEMBER_OF`` folder membership, the entity nodes with their chunk provenance,
the ``DIRECTED`` edges, and the folder overlay nodes. Every read
``read_graph_native`` performs — the chunk→doc index, the member docs, the
entity and relation overrides, the direct-member rows, and the stored relation
rows — is a real Cypher round-trip against it, and the assertions run on the
JSON the FastAPI route actually returns.

Node *selection* is delegated to LightRAG's ``get_knowledge_graph`` by design
(see ``read_graph_native``'s docstring), so the RAG stand-in here returns the
selected nodes/edges by reading them from that same live Memgraph — the
selection is the one part of the chain this file does not own, and it is not the
part this change touches.

Covers the doctrine's sensitive axes 3 (folder binding, including the negative
case) and 4 (``sources`` = distinct parent documents, not chunk count), plus the
membership-load contract behind them: a folder-bound render resolves chunk
provenance through ONE member-scoped DocStatus read (``_load_folder_membership``)
and never scans the whole ``DocStatus_{ws}`` label — asserted on the real Cypher
the route issues, so a revert to the full-workspace index is caught here.

Integration only (real Memgraph; auto-skipped when ``MEMGRAPH_URI`` is unset).
"""

from __future__ import annotations

import contextlib
import json
from functools import partial

import pytest

from twindb_lightrag_memgraph.server import graph_reader

pytestmark = pytest.mark.integration

_WS = "graph_native_contract_e2e"

# The TypeScript contract these responses must satisfy — keep in sync with
# lightrag_webui_twin/src/types/graph.ts (GraphEntity / GraphRelation).
_ENTITY_REQUIRED = {"id", "name", "type", "x", "y", "mentions", "sources", "summary"}
_RELATION_REQUIRED = {"id", "source", "target", "label", "strength"}


def _doc(doc_id: str, chunk_ids: list[str]) -> dict:
    return {
        "id": doc_id,
        "status": "processed",
        "content_summary": "x",
        "content_length": 1,
        "file_path": f"{doc_id}.md",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "content_hash": doc_id,
        "chunks_list": json.dumps(chunk_ids),
    }


class _Node:
    """Shape of a LightRAG KnowledgeGraph node."""

    def __init__(self, entity_id: str, props: dict) -> None:
        self.id = entity_id
        self.labels = [entity_id]
        self.properties = props


class _Edge:
    def __init__(self, source: str, target: str, props: dict) -> None:
        self.id = f"{source}->{target}"
        self.type = "DIRECTED"
        self.source = source
        self.target = target
        self.properties = props


class _KnowledgeGraph:
    def __init__(self, nodes, edges) -> None:
        self.nodes = nodes
        self.edges = edges


class _LiveSelectionRag:
    """Stands in for LightRAG's node selection, reading from the SAME Memgraph.

    ``read_graph_native`` delegates *which* nodes to show to LightRAG; every
    folder/override/provenance decision after that is graph_reader's own and is
    what this test pins. Reading the selection back out of Memgraph keeps the
    whole chain on real data rather than a hand-built fixture.
    """

    async def get_knowledge_graph(self, **_kwargs):
        from twindb_lightrag_memgraph import _pool

        nodes, edges = [], []
        async with _pool.get_read_session() as session:
            # Order by the RETURNed aliases: Memgraph resolves ORDER BY against
            # the projection, so `ORDER BY n.entity_id` after `RETURN
            # properties(n)` fails with "Only nodes, edges, maps ... have
            # properties to be looked up" — which read_graph_native would then
            # swallow into a None result and an empty, silently-passing route.
            result = await session.run(
                f"MATCH (n:`{_WS}`) "
                "RETURN n.entity_id AS eid, properties(n) AS p ORDER BY eid"
            )
            async for record in result:
                props = dict(record["p"])
                nodes.append(_Node(str(props.get("entity_id")), props))
            await result.consume()

            result = await session.run(
                f"MATCH (s:`{_WS}`)-[r:DIRECTED]->(t:`{_WS}`) "
                "RETURN s.entity_id AS s, t.entity_id AS t, properties(r) AS p "
                "ORDER BY s, t"
            )
            async for record in result:
                edges.append(_Edge(record["s"], record["t"], dict(record["p"])))
            await result.consume()
        return _KnowledgeGraph(nodes, edges)


@pytest.fixture
async def live_graph(monkeypatch):
    """Seed a real Memgraph: doc-a ∈ folder A (two chunks), doc-b ∈ folder B."""
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
        async with _pool.get_session() as session:
            for label in (
                f"DocStatus_{_WS}",
                f"Folder_{_WS}",
                f"GraphOverride_{_WS}",
                f"GraphRelOverride_{_WS}",
                _WS,
            ):
                await (
                    await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                ).consume()

    await _wipe()
    # doc-a owns TWO chunks — axis 4: `sources` must count the distinct parent
    # document once, not the two chunks that mention the entity.
    with storage_folder_context("A"):
        await ds.upsert({"doc-a": _doc("doc-a", ["chunk-a1", "chunk-a2"])})
    with storage_folder_context("B"):
        await ds.upsert({"doc-b": _doc("doc-b", ["chunk-b1"])})

    async with _pool.get_session() as session:
        await (
            await session.run(
                f"CREATE (:`{_WS}` {{entity_id:'ent-a', entity_type:'CONCEPT', "
                f"display_name:'Entity A', description:'from A', "
                f"source_id:'chunk-a1{graph_reader._GRAPH_FIELD_SEP}chunk-a2'}}) "
                f"CREATE (:`{_WS}` {{entity_id:'ent-b', entity_type:'CONCEPT', "
                f"display_name:'Entity B', description:'from B', "
                f"source_id:'chunk-b1'}})"
            )
        ).consume()
        await (
            await session.run(
                f"MATCH (a:`{_WS}` {{entity_id:'ent-a'}}), "
                f"(b:`{_WS}` {{entity_id:'ent-b'}}) "
                f"CREATE (a)-[:DIRECTED {{keywords:'links', weight:1.0, "
                f"source_id:'chunk-a1'}}]->(b)"
            )
        ).consume()
    yield ds
    await _wipe()


@pytest.fixture
async def graph_api(live_graph, monkeypatch):
    """The real FastAPI graph routes over the seeded live Memgraph."""
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-infra-root")
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "A")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "A", "label": "Folder A", "kind": "primary"},
                {"id": "B", "label": "Folder B", "kind": "sandbox"},
            ]
        ),
    )
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from twindb_lightrag_memgraph import _twindb_state
    from twindb_lightrag_memgraph.server import webui_router
    from twindb_lightrag_memgraph.server.auth import configure_auth

    webui_router.reset_store()
    configure_auth(api_key="test-infra-root")
    _twindb_state["rag"] = _LiveSelectionRag()
    monkeypatch.setattr(
        webui_router, "_graph_memgraph_label", lambda: _WS, raising=False
    )

    app = FastAPI()
    app.include_router(webui_router.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    ) as client:
        yield client
    _twindb_state.pop("rag", None)
    webui_router.reset_store()
    configure_auth()


async def test_entities_route_matches_typescript_contract(graph_api):
    """Cypher → route JSON → the shape `GraphEntity` declares."""
    response = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    assert response.status_code == 200
    entities = response.json()
    assert entities, "folder A must expose its entity"

    for entity in entities:
        missing = _ENTITY_REQUIRED - set(entity)
        assert not missing, f"GraphEntity contract missing {sorted(missing)}"
        assert isinstance(entity["id"], str)
        assert isinstance(entity["name"], str)
        assert isinstance(entity["mentions"], int)
        assert isinstance(entity["sources"], int)
        assert isinstance(entity["summary"], str)
        assert isinstance(entity["x"], (int, float))
        assert isinstance(entity["y"], (int, float))


async def test_entities_route_is_folder_scoped_both_ways(graph_api):
    """Axis 3 — including the negative case: A must not see B's entity."""
    in_a = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    names_a = {e["name"] for e in in_a.json()}
    assert names_a == {"Entity A"}, f"folder A leaked: {names_a}"

    in_b = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "B"})
    names_b = {e["name"] for e in in_b.json()}
    assert names_b == {"Entity B"}, f"folder B leaked: {names_b}"

    # Switching folders in the same process must re-resolve membership, never
    # reuse the previous folder's answer.
    assert names_a.isdisjoint(names_b)


async def test_sources_counts_distinct_parent_documents(graph_api):
    """Axis 4 — `sources` is distinct parent docs, not chunk count.

    ``ent-a`` is mentioned by TWO chunks of the SAME document, so a
    chunk-counting regression shows up here as ``sources == 2``.
    """
    response = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    entity = next(e for e in response.json() if e["name"] == "Entity A")
    assert entity["sources"] == 1, "sources must count doc-a once, not its 2 chunks"
    assert entity["mentions"] == 2, "mentions still counts both in-folder chunks"
    assert entity.get("source_docs") == ["doc-a"]


async def test_relations_route_matches_contract_and_scoping(graph_api):
    """The edge is provenance-A but points at an entity invisible in A."""
    response = await graph_api.get("/graph/relations", headers={"X-Twin-Folder": "A"})
    assert response.status_code == 200
    relations = response.json()
    for relation in relations:
        missing = _RELATION_REQUIRED - set(relation)
        assert not missing, f"GraphRelation contract missing {sorted(missing)}"
    # ent-b is not visible in A, so no relation may dangle to it.
    visible = {
        e["id"]
        for e in (
            await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
        ).json()
    }
    for relation in relations:
        assert relation["source"] in visible
        assert relation["target"] in visible


async def test_folder_overlay_is_applied_through_the_route(live_graph, graph_api):
    """The gathered override loads must still reach the projection.

    This is the read that the fan-out bound could plausibly break: the entity
    overlay is one of the five concurrent loads, and its result has to land on
    the right entity. Written through real Cypher, asserted through the route.
    """
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_session() as session:
        await (
            await session.run(
                f"MATCH (n:`{_WS}` {{entity_id:'ent-a'}}) "
                f"CREATE (n)-[:`{graph_reader._HAS_OVERRIDE_REL}`]->"
                f"(:`GraphOverride_{_WS}` {{folder:'A', "
                f"display_name:'Renamed In A'}})"
            )
        ).consume()

    response = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    names = {e["name"] for e in response.json()}
    assert names == {"Renamed In A"}, f"folder overlay not applied: {names}"

    # Folder B never sees A's overlay.
    in_b = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "B"})
    assert {e["name"] for e in in_b.json()} == {"Entity B"}


async def test_membership_fanout_is_bounded(monkeypatch):
    """The membership reads overlap without exceeding the default cap."""
    import asyncio

    monkeypatch.delenv("TWIN_GRAPH_MEMBERSHIP_FANOUT", raising=False)
    peak = 0
    live = 0

    async def _probe(index: int) -> int:
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return index

    results = await graph_reader._gather_membership_reads(
        *(partial(_probe, i) for i in range(5))
    )
    assert results == [0, 1, 2, 3, 4], "results must keep their argument order"
    assert peak == graph_reader._DEFAULT_MEMBERSHIP_FANOUT


# ── membership load contract (perf/graph-folder-membership-scoped) ──────────


class _RecordingSession:
    """Delegate to a real Bolt session while recording the Cypher it runs."""

    def __init__(self, session, seen: list[str]) -> None:
        self._session = session
        self._seen = seen

    async def run(self, query: str, *args, **kwargs):
        self._seen.append(query)
        return await self._session.run(query, *args, **kwargs)


async def _entity(client, folder: str, name: str) -> dict | None:
    response = await client.get("/graph/entities", headers={"X-Twin-Folder": folder})
    assert response.status_code == 200
    return next((e for e in response.json() if e["name"] == name), None)


async def test_folder_bound_render_loads_membership_in_one_scoped_read(
    graph_api, monkeypatch
):
    """Revert guard on the real chain: no full DocStatus scan on a folder read.

    The chunk→doc index used to be built from EVERY DocStatus row of the
    workspace on each render, then consulted only for member documents. The
    folder-bound path must now touch DocStatus exactly once, through the
    ``MEMBER_OF`` traversal, and still return the same folder-scoped answer.
    """
    from twindb_lightrag_memgraph import _pool

    seen: list[str] = []
    real_session = _pool.get_read_session

    @contextlib.asynccontextmanager
    async def recording_session():
        async with real_session() as session:
            yield _RecordingSession(session, seen)

    monkeypatch.setattr(graph_reader, "get_read_session", recording_session)

    response = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    assert {e["name"] for e in response.json()} == {"Entity A"}

    docstatus_reads = [
        q for q in seen if f"DocStatus_{_WS}" in q and "chunks_list" in q
    ]
    assert len(docstatus_reads) == 1, docstatus_reads
    assert "MEMBER_OF" in docstatus_reads[0], "membership read must be folder-scoped"
    assert not [
        q for q in seen if f"DocStatus_{_WS}" in q and "MEMBER_OF" not in q
    ], "a folder-bound render scanned the whole DocStatus label"


async def test_mixed_provenance_counts_only_the_folder_s_own_chunks(
    live_graph, graph_api
):
    """An entity cited from doc-a (A) AND doc-b (B) is visible in both folders,
    each seeing only its own chunk and parent doc, and masked as mixed. The
    counts come from the member-scoped index; ``mixed`` still compares against
    the entity's full ``source_id`` list."""
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_session() as session:
        await (
            await session.run(
                f"CREATE (:`{_WS}` {{entity_id:'ent-ab', entity_type:'CONCEPT', "
                f"display_name:'Entity AB', description:'blended', "
                f"source_id:'chunk-a2{graph_reader._GRAPH_FIELD_SEP}chunk-b1'}})"
            )
        ).consume()

    in_a = await _entity(graph_api, "A", "Entity AB")
    assert in_a is not None
    assert (in_a["mentions"], in_a["sources"]) == (1, 1)
    assert in_a["source_docs"] == ["doc-a"]
    assert in_a["summary"] == graph_reader._MASKED_ENTITY_SUMMARY

    in_b = await _entity(graph_api, "B", "Entity AB")
    assert in_b is not None
    assert (in_b["mentions"], in_b["sources"]) == (1, 1)
    assert in_b["source_docs"] == ["doc-b"]
    assert in_b["summary"] == graph_reader._MASKED_ENTITY_SUMMARY


async def test_chunk_shared_across_documents_resolves_to_the_member_document(
    live_graph, graph_api
):
    """LightRAG hashes chunk ids from content, so identical text in two documents
    yields ONE chunk id listed by both. doc-c (folder B) lists chunk-a1, which
    doc-a (folder A) also owns. Each folder resolves that chunk to ITS member
    document — deterministically, where the former full-workspace index picked
    whichever row the scan returned last."""
    from twindb_lightrag_memgraph import _pool
    from twindb_lightrag_memgraph._constants import storage_folder_context

    with storage_folder_context("B"):
        await live_graph.upsert({"doc-c": _doc("doc-c", ["chunk-a1", "chunk-c1"])})
    async with _pool.get_session() as session:
        await (
            await session.run(
                f"CREATE (:`{_WS}` {{entity_id:'ent-shared', entity_type:'CONCEPT', "
                f"display_name:'Entity Shared', description:'shared text', "
                f"source_id:'chunk-a1'}})"
            )
        ).consume()

    in_a = await _entity(graph_api, "A", "Entity Shared")
    assert in_a is not None
    assert in_a["source_docs"] == ["doc-a"]
    assert in_a["summary"] == "shared text"  # pure-member in A, not masked

    in_b = await _entity(graph_api, "B", "Entity Shared")
    assert in_b is not None
    assert in_b["source_docs"] == ["doc-c"]
    assert in_b["summary"] == "shared text"  # pure-member in B as well


async def test_member_document_without_chunks_keeps_membership_intact(
    live_graph, graph_api
):
    """A member doc with no ``chunks_list`` (queued, empty, or pre-chunking) is
    still a member: it must neither break the membership read nor change what
    the folder sees."""
    from twindb_lightrag_memgraph._constants import storage_folder_context

    with storage_folder_context("A"):
        await live_graph.upsert({"doc-empty": _doc("doc-empty", [])})

    response = await graph_api.get("/graph/entities", headers={"X-Twin-Folder": "A"})
    entities = response.json()
    assert {e["name"] for e in entities} == {"Entity A"}
    assert entities[0]["sources"] == 1
    assert entities[0]["source_docs"] == ["doc-a"]
