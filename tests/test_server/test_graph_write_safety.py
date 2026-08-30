"""Memgraph write-safety contract tests for ``server.graph_reader``.

Covers the two S2 findings of ``docs/audits/ingestion-reindex/audit-2026-07-02.md``:

- **MG-1** — ``create_graph_entity`` was a check-then-CREATE: two concurrent
  POSTs (or a client retry after a timeout whose write landed) could mint two
  nodes with the same ``entity_id``. The fix is an atomic
  ``MERGE … ON CREATE SET`` keyed on ``entity_id``; the sequential HTTP
  contract (409 on duplicate, ``routes_graph.py``) must stay identical.
- **MG-2** — ``delete_graph_entity`` / ``delete_graph_relation`` were
  graph-only and orphaned the ``Vec_{ws}_entities`` / ``Vec_{ws}_relationships``
  vector rows, so retrieval kept grounding on deleted entities. The fix
  cascades the vdb cleanup with the REMOVE-label-before-DETACH-DELETE
  mitigation (Memgraph 3.10+ stale vector-index landmine,
  ``reference_memgraph_310_vector_delete``).

Per ``docs/test-doctrine-graph.md`` the integration cases here drive the full
chain — Cypher/storage seed → FastAPI route → observable Memgraph state +
vector_search behaviour — against a real Memgraph (``@pytest.mark.integration``,
auto-skipped without ``MEMGRAPH_URI``). Workspaces are uuid-suffixed because
the Memgraph instance is shared.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import graph_reader as gr
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.auth import configure_auth

EMBEDDING_DIM = 4


def _unique_ws() -> str:
    return f"gws_{uuid.uuid4().hex[:10]}"


# ----------------------------------------------------------------------
# Unit level (no Memgraph) — label parity + MERGE race contract
# ----------------------------------------------------------------------


class TestVecLabelParity:
    """The cascade must target the exact labels MemgraphVectorDBStorage
    writes — a silent drift between the two builders would make MG-2
    reappear without any test going red."""

    def test_vec_label_mirrors_vector_impl_label(self):
        from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

        store = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
        store.workspace = "wsx"
        store.namespace = "entities"
        assert gr._vec_label("wsx", "entities") == store._label()
        assert gr._vec_label("wsx", "relationships") == "Vec_wsx_relationships"

    def test_vec_label_rejects_unsafe_identifiers(self):
        with pytest.raises(ValueError):
            gr._vec_label("ws`) DETACH DELETE n //", "entities")
        with pytest.raises(ValueError):
            gr._vec_label("wsx", "enti ties")

    def test_namespace_constants_match_lightrag(self):
        ns_mod = pytest.importorskip("lightrag.namespace")
        assert gr._VDB_ENTITIES_NS == ns_mod.NameSpace.VECTOR_STORE_ENTITIES
        assert gr._VDB_RELATIONSHIPS_NS == ns_mod.NameSpace.VECTOR_STORE_RELATIONSHIPS


class _FakeResult:
    def __init__(self, rows: list[Any]):
        self._rows = rows

    def __aiter__(self):
        async def _gen():
            for row in self._rows:
                yield row

        return _gen()

    async def consume(self) -> None:
        return None


class _FakeSession:
    def __init__(self, rows: list[Any] | None = None):
        self._rows = rows or []

    async def run(self, *_args, **_kwargs):
        return _FakeResult(self._rows)


def _fake_session_cm(session: _FakeSession):
    @asynccontextmanager
    async def _cm():
        yield session

    return _cm


def _fake_write_slot():
    @asynccontextmanager
    async def _cm():
        yield

    return _cm


class TestCreateMergeRaceContract:
    async def test_probe_miss_duplicate_raises_exists_error(self, monkeypatch):
        """The TOCTOU window MG-1 describes: ``entity_exists`` misses (probe
        raced a concurrent write / a retry whose first attempt landed) but
        the MERGE matches the existing node. ``created=false`` must surface
        as ``EntityExistsError`` — the route's 409 — and NOT as a successful
        201 that would have minted a second node with the old CREATE."""

        async def fake_exists(workspace, entity_id):
            return False

        monkeypatch.setattr(gr, "entity_exists", fake_exists)
        monkeypatch.setattr(gr, "acquire_write_slot", _fake_write_slot())
        monkeypatch.setattr(
            gr,
            "get_session",
            _fake_session_cm(
                _FakeSession(rows=[{"entity_id": "Dup", "created": False}])
            ),
        )

        with pytest.raises(gr.EntityExistsError):
            await gr.create_graph_entity("demo", {"name": "Dup", "type": "PRODUCT"})


# ----------------------------------------------------------------------
# Write/write conflict retry — every graph_reader write must absorb
# Memgraph's "Cannot resolve conflicting transactions" abort instead of
# propagating it (observed live: _persist_relation_ids failing 3× on the
# OVH maquette, 2026-07-21..28). The retried thunk re-acquires its own
# slot + session per _retry.py invariant 2.
# ----------------------------------------------------------------------

CONFLICT_MESSAGE = (
    "Cannot resolve conflicting transactions. Retry this transaction when "
    "the conflicting transaction is finished."
)


class _ScriptedSession:
    """Session whose ``run`` follows a script: each entry is either an
    exception to raise or a row list to return. Records every query."""

    def __init__(self, script: list[Any]):
        self._script = list(script)
        self.queries: list[str] = []

    async def run(self, query, *_args, **_kwargs):
        self.queries.append(query)
        step = self._script.pop(0) if self._script else []
        if isinstance(step, BaseException):
            raise step
        return _FakeResult(step)


def _patch_write_path(monkeypatch, session: _ScriptedSession) -> None:
    monkeypatch.setattr(gr, "acquire_write_slot", _fake_write_slot())
    monkeypatch.setattr(gr, "get_session", _fake_session_cm(session))


class TestConflictRetryContract:
    async def test_persist_relation_ids_retries_conflict(self, monkeypatch):
        """The exact failure observed on the maquette: one conflict abort,
        then success — must report the bindings as persisted, not 0."""
        session = _ScriptedSession([RuntimeError(CONFLICT_MESSAGE), []])
        _patch_write_path(monkeypatch, session)
        count = await gr._persist_relation_ids(
            "demo", [{"source_id": "A", "target_id": "B"}]
        )
        assert count == 1
        assert len(session.queries) == 2

    async def test_entity_override_retries_conflict(self, monkeypatch):
        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE), [{"folder": "demo"}]]
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr._upsert_entity_override("demo", "demo", "E1", {}, deleted=False)
        assert ok is True
        assert len(session.queries) == 2

    async def test_entity_props_retries_conflict(self, monkeypatch):
        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE), [{"entity_id": "E1"}]]
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr._write_entity_props(
            workspace="demo", entity_id="E1", props={"description": "d"}
        )
        assert ok is True
        assert len(session.queries) == 2

    async def test_rel_override_retries_conflict(self, monkeypatch):
        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE), [{"folder": "demo"}]]
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr._upsert_rel_override("demo", "demo", "A", "B", {}, deleted=False)
        assert ok is True
        assert len(session.queries) == 2

    async def test_relation_props_retries_conflict(self, monkeypatch):
        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE), [{"source_id": "A"}]]
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr._write_relation_props(
            workspace="demo", src="A", tgt="B", props={"weight": 0.7}
        )
        assert ok is True
        assert len(session.queries) == 2

    async def test_relation_vdb_cascade_retries_conflict(self, monkeypatch):
        session = _ScriptedSession([RuntimeError(CONFLICT_MESSAGE), []])
        _patch_write_path(monkeypatch, session)
        ok = await gr._cascade_relation_vdb_row("demo", "A", "B")
        assert ok is True
        assert len(session.queries) == 2

    async def test_entity_physical_delete_retries_conflict(self, monkeypatch):
        async def fake_member_context(workspace):
            return None, {}

        async def fake_exists(workspace, entity_id):
            return True

        async def fake_cascade(workspace, entity_id):
            return True

        monkeypatch.setattr(gr, "_member_context", fake_member_context)
        monkeypatch.setattr(gr, "entity_exists", fake_exists)
        monkeypatch.setattr(gr, "_cascade_entity_vdb_rows", fake_cascade)
        session = _ScriptedSession([RuntimeError(CONFLICT_MESSAGE), []])
        _patch_write_path(monkeypatch, session)
        ok = await gr.delete_graph_entity("demo", "E1")
        assert ok is True
        assert len(session.queries) == 2

    async def test_relation_physical_delete_retries_conflict(self, monkeypatch):
        async def fake_endpoints(workspace, rel_id):
            return ("demo", "A", "B")

        async def fake_member_context(workspace):
            return None, {}

        async def fake_cascade(workspace, src, tgt):
            return True

        monkeypatch.setattr(gr, "_resolve_relation_endpoints", fake_endpoints)
        monkeypatch.setattr(gr, "_member_context", fake_member_context)
        monkeypatch.setattr(gr, "_cascade_relation_vdb_row", fake_cascade)
        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE), [{"source_id": "A"}]]
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr.delete_graph_relation("demo", "r1")
        assert ok is True
        assert len(session.queries) == 2

    async def test_merge_relation_retries_conflict(self, monkeypatch):
        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE), [{"source_id": "A"}]]
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr._merge_relation(workspace="demo", src="A", tgt="B", props={})
        assert ok is True
        assert len(session.queries) == 2

    async def test_entity_vdb_cascade_retries_conflict(self, monkeypatch):
        # First statement conflicts; the replay re-runs BOTH statements
        # (idempotent deletes), so the script sees 3 runs total.
        session = _ScriptedSession([RuntimeError(CONFLICT_MESSAGE), [], []])
        _patch_write_path(monkeypatch, session)
        ok = await gr._cascade_entity_vdb_rows("demo", "E1")
        assert ok is True
        assert len(session.queries) == 3

    async def test_non_conflict_error_does_not_retry(self, monkeypatch):
        session = _ScriptedSession([RuntimeError("syntax error"), [{"x": 1}]])
        _patch_write_path(monkeypatch, session)
        ok = await gr._write_entity_props(
            workspace="demo", entity_id="E1", props={"description": "d"}
        )
        assert ok is False
        assert len(session.queries) == 1

    async def test_conflict_exhaustion_maps_to_failure_contract(self, monkeypatch):
        from twindb_lightrag_memgraph._retry import MAX_WRITE_ATTEMPTS

        session = _ScriptedSession(
            [RuntimeError(CONFLICT_MESSAGE)] * MAX_WRITE_ATTEMPTS
        )
        _patch_write_path(monkeypatch, session)
        ok = await gr._write_entity_props(
            workspace="demo", entity_id="E1", props={"description": "d"}
        )
        assert ok is False
        assert len(session.queries) == MAX_WRITE_ATTEMPTS

    async def test_create_stamp_conflict_does_not_replay_merge(self, monkeypatch):
        """A conflict on the folder-membership stamp must retry the stamp
        ALONE. Replaying the whole create block would re-run the MERGE
        against the node the first attempt committed and misreport the
        create as a 409 duplicate."""

        async def fake_exists(workspace, entity_id):
            return False

        async def fake_read_one(workspace, entity_id, *args, **kwargs):
            return {"id": f"node-{entity_id}"}

        from twindb_lightrag_memgraph.server import folder as folder_mod

        monkeypatch.setattr(gr, "entity_exists", fake_exists)
        monkeypatch.setattr(gr, "_read_one_entity", fake_read_one)
        monkeypatch.setattr(folder_mod, "active_folder_id", lambda: "demo")
        session = _ScriptedSession(
            [
                [{"entity_id": "E1", "created": True}],  # MERGE — once only
                RuntimeError(CONFLICT_MESSAGE),  # stamp, attempt 1
                [],  # stamp, attempt 2
            ]
        )
        _patch_write_path(monkeypatch, session)

        projected = await gr.create_graph_entity(
            "demo", {"name": "E1", "type": "PRODUCT"}
        )
        assert projected == {"id": "node-E1"}
        assert len(session.queries) == 3
        merge_queries = [q for q in session.queries if "MERGE (n:" in q]
        assert len(merge_queries) == 1


# ----------------------------------------------------------------------
# Integration (real Memgraph) — Cypher → API route → observable effect
# ----------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration

_CHUNK = "chunk-wsafe"
_DOC = "doc-wsafe"


async def _mock_embed(texts: list[str]) -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return rng.random((len(texts), EMBEDDING_DIM)).astype(np.float32)


@pytest.fixture
async def graph_env(monkeypatch):
    """Isolated workspace + bare app mounting the Twin webui router.

    The router-level ``bind_request_folder`` dependency binds the catalog
    default folder on every request (there is no unscoped route path), so a
    ``DocStatus`` doc MEMBER_OF the default folder is seeded with the chunk
    the test entities cite — making them *pure-member* and eligible for the
    physical-delete path that MG-1/MG-2 exercise.
    """
    ws = _unique_ws()
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", ws)
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": "default", "label": "Default", "kind": "primary"}]),
    )
    from twindb_lightrag_memgraph import _pool, register
    from twindb_lightrag_memgraph._constants import storage_folder_context
    from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

    register()
    ds = MemgraphDocStatusStorage(
        namespace="doc_status", global_config={}, embedding_func=None
    )
    await ds.initialize()
    with storage_folder_context("default"):
        await ds.upsert(
            {
                _DOC: {
                    "id": _DOC,
                    "status": "processed",
                    "content_summary": "x",
                    "content_length": 1,
                    "file_path": f"{_DOC}.md",
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                    "content_hash": _DOC,
                    "chunks_list": json.dumps([_CHUNK]),
                }
            }
        )

    app = FastAPI()
    configure_auth(api_key="test-infra-root")
    app.include_router(webui_router.router)
    # Deterministic in-memory store for the routes' activity recording —
    # the graph mutations under test persist through graph_reader Cypher,
    # not through the WebuiStore.
    webui_router.set_store(webui_router.WebuiStore.from_seed())

    yield ws, app

    async with _pool.get_session() as s:
        for lbl in (ws, f"Folder_{ws}", f"DocStatus_{ws}"):
            await (await s.run(f"MATCH (n:`{lbl}`) DETACH DELETE n")).consume()
    webui_router.reset_store()
    configure_auth()


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    )


async def _count_entities(ws: str, entity_id: str) -> int:
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_read_session() as s:
        result = await s.run(
            f"MATCH (n:`{ws}` {{entity_id: $eid}}) RETURN count(n) AS c",
            eid=entity_id,
        )
        record = await result.single()
        await result.consume()
        return int(record["c"])


async def _node_props(ws: str, entity_id: str) -> dict[str, Any] | None:
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_read_session() as s:
        result = await s.run(
            f"MATCH (n:`{ws}` {{entity_id: $eid}}) RETURN properties(n) AS p LIMIT 1",
            eid=entity_id,
        )
        record = await result.single()
        await result.consume()
        return dict(record["p"]) if record else None


@pytestmark_integration
class TestMg1EntityCreateAtomicity:
    async def test_sequential_double_create_is_409_and_single_node(self, graph_env):
        """API contract unchanged: second POST of the same name → 409
        (routes_graph.py duplicate mapping), and exactly one node exists."""
        ws, app = graph_env
        payload = {"name": "Mg1Seq", "type": "CONCEPT", "summary": "first write"}
        async with _client(app) as client:
            r1 = await client.post("/graph/entities", json=payload)
            assert r1.status_code == 201, r1.text
            r2 = await client.post("/graph/entities", json=payload)
            assert r2.status_code == 409, r2.text
        assert await _count_entities(ws, "Mg1Seq") == 1

    async def test_probe_miss_retry_cannot_mint_second_node(
        self, graph_env, monkeypatch
    ):
        """The concurrent/retry case MG-1 confirmed: force the ``entity_exists``
        probe to miss (as it does when the duplicate landed inside the
        check-then-write window) and re-POST. With the old CREATE this
        minted a second node behind a 201; the MERGE must answer 409, keep
        a single node, and leave the first write's properties untouched
        (ON CREATE-only SET) with no marker residue."""
        ws, app = graph_env
        payload = {"name": "Mg1Race", "type": "CONCEPT", "summary": "original"}
        async with _client(app) as client:
            r1 = await client.post("/graph/entities", json=payload)
            assert r1.status_code == 201, r1.text

            async def probe_always_misses(workspace, entity_id):
                return False

            monkeypatch.setattr(gr, "entity_exists", probe_always_misses)
            retry = {"name": "Mg1Race", "type": "PRODUCT", "summary": "clobber"}
            r2 = await client.post("/graph/entities", json=retry)
            assert r2.status_code == 409, r2.text

        assert await _count_entities(ws, "Mg1Race") == 1
        props = await _node_props(ws, "Mg1Race")
        assert props is not None
        assert props["description"] == "original"  # retry did not clobber
        assert props["entity_type"] == "CONCEPT"
        assert "__twin_create_marker" not in props  # no marker residue


@pytest.fixture
async def vdb_stores(graph_env):
    """Real entity/relationship vector stores in the test workspace, with
    their vector indexes — so post-delete assertions exercise the actual
    ``vector_search`` path, not a property lookup."""
    from lightrag.utils import EmbeddingFunc

    from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=_mock_embed
    )
    config = {"vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.0}}
    entities = MemgraphVectorDBStorage(
        namespace="entities",
        global_config=config,
        embedding_func=embedding_func,
        meta_fields={"entity_name", "content"},
    )
    relationships = MemgraphVectorDBStorage(
        namespace="relationships",
        global_config=config,
        embedding_func=embedding_func,
        meta_fields={"src_id", "tgt_id", "content"},
    )
    await entities.initialize()
    await relationships.initialize()
    yield entities, relationships
    await entities.drop()
    await relationships.drop()


def _rel_pair_id(a: str, b: str) -> str:
    """LightRAG relationship vdb id: sorted pair, ``rel-`` prefix
    (operate.py, identical on 1.4.9.11 and 1.5.x)."""
    from lightrag.utils import compute_mdhash_id

    src, tgt = (a, b) if a <= b else (b, a)
    return compute_mdhash_id(src + tgt, prefix="rel-")


async def _seed_graph(ws: str, *entity_ids: str) -> None:
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_session() as s:
        for eid in entity_ids:
            await (
                await s.run(
                    f"CREATE (:`{ws}` {{entity_id: $eid, entity_type: 'CONCEPT', "
                    "description: 'seeded', source_id: $chunk})",
                    eid=eid,
                    chunk=_CHUNK,
                )
            ).consume()


async def _seed_edge(ws: str, src: str, tgt: str, rel_id: str | None = None) -> None:
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_session() as s:
        await (
            await s.run(
                f"MATCH (a:`{ws}` {{entity_id: $src}}), (b:`{ws}` {{entity_id: $tgt}}) "
                "CREATE (a)-[:DIRECTED {keywords: 'links', weight: 1.0, "
                "source_id: $chunk, twin_relation_id: $rid}]->(b)",
                src=src,
                tgt=tgt,
                chunk=_CHUNK,
                rid=rel_id or "",
            )
        ).consume()


@pytestmark_integration
class TestMg2DeleteCascadesVectorRows:
    async def test_entity_delete_cascades_entity_and_relation_rows(
        self, graph_env, vdb_stores
    ):
        """Seed E1/E2/E3 in graph + vdb (LightRAG ``ent-``/``rel-`` id and
        property conventions), DELETE E1 via the API, then assert: graph
        node gone, its vdb rows gone, control rows (E2, E3, rel E2↔E3)
        untouched, and vector_search stays clean (no stale-index error,
        no grounding on the deleted entity)."""
        from lightrag.utils import compute_mdhash_id

        ws, app = graph_env
        entities_vdb, relationships_vdb = vdb_stores
        await _seed_graph(ws, "E1", "E2", "E3")
        await _seed_edge(ws, "E1", "E2")
        await _seed_edge(ws, "E2", "E3")

        ent_rows = {
            compute_mdhash_id(name, prefix="ent-"): {
                "entity_name": name,
                "content": f"{name} seeded description",
                "source_id": _CHUNK,
                "file_path": f"{_DOC}.md",
            }
            for name in ("E1", "E2", "E3")
        }
        await entities_vdb.upsert(ent_rows)
        rel_rows = {}
        for a, b in (("E1", "E2"), ("E2", "E3")):
            src, tgt = (a, b) if a <= b else (b, a)
            rel_rows[_rel_pair_id(a, b)] = {
                "src_id": src,
                "tgt_id": tgt,
                "content": f"links\t{src}\n{tgt}\nseeded",
                "keywords": "links",
                "source_id": _CHUNK,
            }
        await relationships_vdb.upsert(rel_rows)

        async with _client(app) as client:
            r = await client.delete("/graph/entities/kg_E1")
            assert r.status_code == 204, r.text

        # Graph: E1 gone, controls intact.
        assert await _count_entities(ws, "E1") == 0
        assert await _count_entities(ws, "E2") == 1
        assert await _count_entities(ws, "E3") == 1

        # Vdb: E1's rows gone — including the relationship row it anchored.
        e1_id = compute_mdhash_id("E1", prefix="ent-")
        assert await entities_vdb.get_by_id(e1_id) is None
        assert await relationships_vdb.get_by_id(_rel_pair_id("E1", "E2")) is None

        # Native-path guard: rows not belonging to E1 are untouched.
        assert await entities_vdb.get_by_id(compute_mdhash_id("E2", prefix="ent-"))
        assert await entities_vdb.get_by_id(compute_mdhash_id("E3", prefix="ent-"))
        assert await relationships_vdb.get_by_id(_rel_pair_id("E2", "E3"))

        # vector_search: no grounding on E1, and no 50N42-style stale-index
        # error on the post-delete queries (the Memgraph 3.10+ landmine the
        # REMOVE-label mitigation exists for; CI runs 3.12.0).
        ent_hits = await entities_vdb.query("E1 seeded description", top_k=10)
        assert all(h.get("entity_name") != "E1" for h in ent_hits)
        rel_hits = await relationships_vdb.query("links seeded", top_k=10)
        assert all("E1" not in (h.get("src_id"), h.get("tgt_id")) for h in rel_hits)

    async def test_relation_delete_cascades_pair_row_only(self, graph_env, vdb_stores):
        """DELETE of one relation removes exactly its vdb pair row; the
        endpoints' entity rows and unrelated relation rows survive."""
        from lightrag.utils import compute_mdhash_id

        ws, app = graph_env
        entities_vdb, relationships_vdb = vdb_stores
        await _seed_graph(ws, "E2", "E3", "E4")
        rel_id = gr._relation_id_from_endpoints("E2", "E3")
        await _seed_edge(ws, "E2", "E3", rel_id)
        await _seed_edge(ws, "E3", "E4", gr._relation_id_from_endpoints("E3", "E4"))

        await entities_vdb.upsert(
            {
                compute_mdhash_id(name, prefix="ent-"): {
                    "entity_name": name,
                    "content": f"{name} seeded",
                    "source_id": _CHUNK,
                }
                for name in ("E2", "E3", "E4")
            }
        )
        rel_rows = {}
        for a, b in (("E2", "E3"), ("E3", "E4")):
            rel_rows[_rel_pair_id(a, b)] = {
                "src_id": a,
                "tgt_id": b,
                "content": f"links {a} {b}",
                "keywords": "links",
                "source_id": _CHUNK,
            }
        await relationships_vdb.upsert(rel_rows)

        async with _client(app) as client:
            r = await client.delete(f"/graph/relations/{rel_id}")
            assert r.status_code == 204, r.text

        # Edge gone from the graph.
        from twindb_lightrag_memgraph import _pool

        async with _pool.get_read_session() as s:
            result = await s.run(
                f"MATCH (:`{ws}` {{entity_id: 'E2'}})-[r:DIRECTED]->"
                f"(:`{ws}` {{entity_id: 'E3'}}) RETURN count(r) AS c"
            )
            record = await result.single()
            await result.consume()
            assert int(record["c"]) == 0

        # Vdb: exactly the pair row gone; everything else intact.
        assert await relationships_vdb.get_by_id(_rel_pair_id("E2", "E3")) is None
        assert await relationships_vdb.get_by_id(_rel_pair_id("E3", "E4"))
        for name in ("E2", "E3", "E4"):
            assert await entities_vdb.get_by_id(compute_mdhash_id(name, prefix="ent-"))
        # Post-delete vector query stays clean (stale-index guard).
        hits = await relationships_vdb.query("links", top_k=10)
        assert all(
            not (h.get("src_id") == "E2" and h.get("tgt_id") == "E3") for h in hits
        )
