"""Folder-scoped retrieval — batch 2 of FOLDER-MEMBERSHIP-REFACTOR.md.

Real cloisonnement of the query path: a retrieval issued *in folder X* may only
ground on chunks / entities / relations whose document is ``MEMBER_OF X``.
LightRAG cannot scope its own retrieval (verified against the BNP-pinned
1.4.9.11: ``QueryParam`` has no id filter, ``BaseVectorStorage.query`` has no
``ids`` param), so the constraint is applied at the Memgraph storage layer in
``MemgraphVectorDBStorage.query`` when a ``storage_folder_context`` is active.

Two layers of test:

1. **Cypher-builder unit tests** (no DB) — lock the strict-compat contract:
   *folder absent → the query Cypher is byte-for-byte the legacy search*, and
   the two membership join shapes (chunks via ``full_doc_id``; entity/relation
   via ``split(source_id)`` → chunks).
2. **Live contract** (``@pytest.mark.integration``, needs Memgraph) — the real
   boundary across all three vdb, including the critical *mixed* ``source_id``
   case (``chunk_A<SEP>chunk_B`` → selected in A and B, never in C).

**Known residual (acceptance, not a gap to fix here):** entity/relation scoping
filters the *selection* by member source chunks, but a kept record's aggregated
``content``/description was merged by LightRAG across *all* its source docs at
extraction — so the payload may still encode non-member text. Un-blending needs
per-folder re-extraction (out of scope).
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph._constants import storage_folder_context
from twindb_lightrag_memgraph.vector_impl import (
    GRAPH_FIELD_SEP,
    MemgraphVectorDBStorage,
    _folder_scope_overfetch,
)


def _store(namespace: str, meta_fields: set[str], workspace: str = "ws"):
    """A MemgraphVectorDBStorage with attributes set, no DB / embedding wiring."""
    st = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
    st.namespace = namespace
    st.workspace = workspace
    st.meta_fields = meta_fields
    st.cosine_better_than_threshold = 0.2
    return st


# ── Cypher builder: strict compat + join shapes (no DB) ───────────────────


class TestSearchCypherBuilder:
    def test_no_folder_is_the_legacy_unscoped_query(self):
        st = _store("chunks", {"full_doc_id", "content", "file_path"})
        cypher, params = st._build_search_cypher(20, None)
        # No membership traversal, no overfetch/folder params — identical to the
        # pre-batch-2 search so the native LightRAG path is unchanged.
        assert "MEMBER_OF" not in cypher
        assert "Folder_" not in cypher
        assert 'vector_search.search("vec_ws_chunks", $top_k, $embedding)' in cypher
        assert "folder" not in params
        assert "overfetch" not in params
        assert params["top_k"] == 20

    def test_chunks_join_on_full_doc_id(self):
        st = _store("chunks", {"full_doc_id", "content", "file_path"})
        cypher, params = st._build_search_cypher(20, "A")
        assert "MATCH (d:`DocStatus_ws` {id: node.full_doc_id})" in cypher
        assert "-[:MEMBER_OF]->(:`Folder_ws` {id: $folder})" in cypher
        assert "$overfetch" in cypher  # over-fetch before the inner-join
        assert "LIMIT $top_k" in cypher
        assert params["folder"] == "A"
        assert params["overfetch"] == _folder_scope_overfetch(20)
        assert "sep" not in params  # chunks don't split source_id

    def test_entities_join_on_source_id_chunks(self):
        st = _store("entities", {"entity_name", "source_id", "content", "file_path"})
        cypher, params = st._build_search_cypher(20, "A")
        assert "split(coalesce(node.source_id, ''), $sep)" in cypher
        assert "MATCH (c:`Vec_ws_chunks` {id: cid})" in cypher
        assert "MATCH (d:`DocStatus_ws` {id: c.full_doc_id})" in cypher
        assert "-[:MEMBER_OF]->(:`Folder_ws` {id: $folder})" in cypher
        assert params["sep"] == GRAPH_FIELD_SEP

    def test_relationships_use_the_source_id_shape(self):
        st = _store(
            "relationships",
            {"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        cypher, _ = st._build_search_cypher(20, "A")
        assert "split(coalesce(node.source_id, ''), $sep)" in cypher
        assert "MATCH (c:`Vec_ws_chunks` {id: cid})" in cypher

    def test_unknown_vdb_category_with_folder_fails_closed(self):
        # A vdb with neither full_doc_id nor source_id has no membership concept.
        # With an active folder we CANNOT prove membership — fail closed (signal
        # the caller to return nothing) rather than leak a global result set.
        st = _store("misc", {"content"})
        cypher, _ = st._build_search_cypher(20, "A")
        assert cypher is None

    def test_unknown_vdb_category_without_folder_is_unscoped(self):
        # No active folder → the unknown category is irrelevant; the legacy
        # unscoped search runs as before (strict compat).
        st = _store("misc", {"content"})
        cypher, params = st._build_search_cypher(20, None)
        assert cypher is not None
        assert "MEMBER_OF" not in cypher
        assert "folder" not in params


class TestFailClosed:
    async def test_query_returns_empty_when_category_unknown_and_folder_active(self):
        # No DB is touched: the fail-closed branch returns before opening a
        # session (query_embedding passed so embedding_func is never called).
        st = _store("misc", {"content"})
        with storage_folder_context("A"):
            out = await st.query("q", top_k=5, query_embedding=[0.0])
        assert out == []


class TestOverfetch:
    def test_default_factor(self):
        assert _folder_scope_overfetch(10) == 40  # 10 * 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TWIN_QUERY_FOLDER_SCOPE_OVERFETCH", "10")
        assert _folder_scope_overfetch(10) == 100

    def test_cap(self, monkeypatch):
        monkeypatch.setenv("TWIN_QUERY_FOLDER_SCOPE_OVERFETCH", "1000")
        assert _folder_scope_overfetch(10) == 500  # capped

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("TWIN_QUERY_FOLDER_SCOPE_OVERFETCH", "not-a-number")
        assert _folder_scope_overfetch(10) == 40

    def test_never_below_top_k(self, monkeypatch):
        monkeypatch.setenv("TWIN_QUERY_FOLDER_SCOPE_OVERFETCH", "0")
        assert _folder_scope_overfetch(10) == 10


# ── Live cloisonnement contract (real Memgraph) ───────────────────────────

pytestmark_integration = pytest.mark.integration

_WS = "folder_query_test"
_DIM = 8


class _FakeEmbed:
    """Minimal embedding_func: fixed unit vector, only embedding_dim is read."""

    embedding_dim = _DIM

    async def func(self, texts):
        return [[1.0] + [0.0] * (_DIM - 1) for _ in texts]


def _qvec():
    return [1.0] + [0.0] * (_DIM - 1)


@pytest.fixture
async def stores(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", _WS)
    from twindb_lightrag_memgraph import _pool, register
    from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

    register()
    embed = _FakeEmbed()

    def _vec(namespace, meta):
        return MemgraphVectorDBStorage(
            namespace=namespace,
            global_config={},
            embedding_func=embed,
            meta_fields=meta,
        )

    ds = MemgraphDocStatusStorage(
        namespace="doc_status", global_config={}, embedding_func=None
    )
    chunks = _vec("chunks", {"full_doc_id", "content", "file_path"})
    entities = _vec("entities", {"entity_name", "source_id", "content", "file_path"})
    rels = _vec(
        "relationships", {"src_id", "tgt_id", "source_id", "content", "file_path"}
    )
    for st in (ds, chunks, entities, rels):
        await st.initialize()

    async def _wipe():
        async with _pool.get_session() as s:
            # Vec_* labels carry a vector index. On Memgraph 3.10+ a plain
            # DETACH DELETE of an indexed vertex leaves a STALE vector-index
            # entry (vector_index_memory_tracked is not freed on delete); a
            # later vector_search then returns the deleted node and strict-errors
            # on property access ("Trying to get a property from a deleted
            # object", 50N42). Mirror the production delete path
            # (vector_impl.delete_entity): REMOVE the indexed label first so the
            # index entry is pruned cleanly, THEN delete the orphan.
            for lbl in (
                f"Vec_{_WS}_chunks",
                f"Vec_{_WS}_entities",
                f"Vec_{_WS}_relationships",
            ):
                await (
                    await s.run(
                        f"MATCH (n:`{lbl}`) REMOVE n:`{lbl}` "
                        f"WITH n DETACH DELETE n"
                    )
                ).consume()
            for lbl in (f"DocStatus_{_WS}", f"Folder_{_WS}"):
                await (await s.run(f"MATCH (n:`{lbl}`) DETACH DELETE n")).consume()

    await _wipe()
    yield ds, chunks, entities, rels
    await _wipe()


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
    }


async def _seed(ds, chunks, entities, rels):
    # docs: a∈A, b∈B, shared∈A∩B
    with storage_folder_context("A"):
        await ds.upsert({"doc-a": _doc("doc-a"), "doc-shared": _doc("doc-shared")})
    with storage_folder_context("B"):
        await ds.upsert({"doc-b": _doc("doc-b")})
    await ds.add_to_folder("doc-shared", "B")

    qv = _qvec()
    await chunks.upsert(
        {
            "chunk-a": {"full_doc_id": "doc-a", "content": "a", "embedding": qv},
            "chunk-b": {"full_doc_id": "doc-b", "content": "b", "embedding": qv},
            "chunk-shared": {
                "full_doc_id": "doc-shared",
                "content": "s",
                "embedding": qv,
            },
        }
    )
    sep = GRAPH_FIELD_SEP
    await entities.upsert(
        {
            "ent-a": {"entity_name": "A", "source_id": "chunk-a", "embedding": qv},
            "ent-b": {"entity_name": "B", "source_id": "chunk-b", "embedding": qv},
            "ent-mixed": {
                "entity_name": "M",
                "source_id": f"chunk-a{sep}chunk-b",
                "embedding": qv,
            },
        }
    )
    await rels.upsert(
        {
            "rel-a": {"src_id": "A", "tgt_id": "X", "source_id": "chunk-a", "embedding": qv},
            "rel-mixed": {
                "src_id": "A",
                "tgt_id": "B",
                "source_id": f"chunk-a{sep}chunk-b",
                "embedding": qv,
            },
        }
    )


async def _ids(store):
    return {r["id"] for r in await store.query("q", top_k=20, query_embedding=_qvec())}


@pytestmark_integration
async def test_chunks_are_scoped_to_folder_membership(stores):
    ds, chunks, entities, rels = stores
    await _seed(ds, chunks, entities, rels)

    # Compat: no active folder → the legacy unscoped search returns everything.
    assert await _ids(chunks) == {"chunk-a", "chunk-b", "chunk-shared"}

    with storage_folder_context("A"):
        assert await _ids(chunks) == {"chunk-a", "chunk-shared"}
    with storage_folder_context("B"):
        assert await _ids(chunks) == {"chunk-b", "chunk-shared"}
    with storage_folder_context("C"):
        assert await _ids(chunks) == set()


@pytestmark_integration
async def test_entities_scoped_by_member_source_chunks_mixed_case(stores):
    ds, chunks, entities, rels = stores
    await _seed(ds, chunks, entities, rels)

    # The critical real case: ent-mixed has source_id chunk-a<SEP>chunk-b, so it
    # is a valid candidate in BOTH A and B (member via one source chunk each),
    # but NOT in C. (Selection is scoped; the aggregated description is a known
    # residual — see module docstring.)
    with storage_folder_context("A"):
        assert await _ids(entities) == {"ent-a", "ent-mixed"}
    with storage_folder_context("B"):
        assert await _ids(entities) == {"ent-b", "ent-mixed"}
    with storage_folder_context("C"):
        assert await _ids(entities) == set()

    # Compat: unscoped → all.
    assert await _ids(entities) == {"ent-a", "ent-b", "ent-mixed"}


@pytestmark_integration
async def test_relationships_scoped_by_member_source_chunks(stores):
    ds, chunks, entities, rels = stores
    await _seed(ds, chunks, entities, rels)

    with storage_folder_context("A"):
        assert await _ids(rels) == {"rel-a", "rel-mixed"}
    with storage_folder_context("B"):
        assert await _ids(rels) == {"rel-mixed"}  # rel-a is A-only
    with storage_folder_context("C"):
        assert await _ids(rels) == set()
