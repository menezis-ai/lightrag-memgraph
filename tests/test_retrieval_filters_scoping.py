"""Retrieval filters enforced at the storage layer — fixes the audit
``tag_filter``/``doc_filter``/``min_score`` "faux grounding" gap.

Until this change ``tag_filter`` / ``doc_filter`` / ``min_score`` were attached
to ``QueryParam`` but read by nothing in the retrieval path: the LLM was
grounded on the *unfiltered* context and the filters only trimmed the Sources
panel *after the fact*. That is a product lie — the UI sells "grounded +
filtered" while the prompt saw chunks the filter was meant to exclude.

These filters are now enforced where folder scoping already lives:
``MemgraphVectorDBStorage._build_search_cypher`` turns an active
``storage_filter_context`` into Cypher predicates, so an excluded chunk never
enters the vector-search result and therefore never reaches the prompt.

Pinned semantics (the sharp one is ``doc_filter`` ``all``):

- ``doc_filter`` on **chunks** (single ``full_doc_id``):
  ``any:S`` → keep iff ``full_doc_id ∈ S``;
  ``all:S`` → keep iff every element of ``S`` equals ``full_doc_id`` (so ``all``
  with ≥2 docs is *impossible on a single chunk* → empty; NOT union-as-any).
- ``tag_filter`` (doc-level via ``TAGGED_WITH`` → ``WebuiTag_{folder}``):
  ``all`` → doc has every required tag; ``any`` → doc has ≥1 optional tag.
- ``min_score``: unfiltered / folder-only retrieval keeps the configured cosine
  floor. Doc/tag filtered retrieval treats the filter as the candidate corpus
  and uses ``min_score`` only when the caller explicitly sends one.

Strict compat (non-negotiable for LightRAG upgrades): a filter fragment is
emitted *only when that filter is non-empty*. Filters empty ⇒ the folder /
legacy Cypher is byte-for-byte identical to the pre-filter build.

Entity/relation graph vectors fail closed whenever a folder is active because
their payloads are globally blended across source documents.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph._constants import (
    RetrievalFilters,
    get_active_retrieval_filters,
    storage_filter_context,
    storage_folder_context,
)
from twindb_lightrag_memgraph.vector_impl import (
    GRAPH_FIELD_SEP,
    MemgraphVectorDBStorage,
)


def _store(namespace: str, meta_fields: set[str], workspace: str = "ws"):
    st = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
    st.namespace = namespace
    st.workspace = workspace
    st.meta_fields = meta_fields
    st.cosine_better_than_threshold = 0.2
    return st


# ── ContextVar plumbing ───────────────────────────────────────────────────


class TestRetrievalFilterContext:
    def test_default_is_none(self):
        assert get_active_retrieval_filters() is None

    def test_context_sets_and_resets(self):
        filt = RetrievalFilters(doc_any=frozenset({"docA"}))
        with storage_filter_context(filt):
            assert get_active_retrieval_filters() is filt
        assert get_active_retrieval_filters() is None

    def test_empty_filters_is_empty(self):
        assert RetrievalFilters().is_empty is True
        assert RetrievalFilters(min_score=0.0).is_empty is True

    def test_non_empty_predicates(self):
        assert RetrievalFilters(min_score=0.3).is_empty is False
        assert RetrievalFilters(doc_any=frozenset({"a"})).has_doc is True
        assert RetrievalFilters(tag_all=frozenset({"oracle"})).has_tag is True


# ── Strict compat: filters absent ⇒ byte-for-byte ─────────────────────────


class TestStrictCompat:
    def test_folder_only_unchanged_by_empty_filters(self):
        st = _store("chunks", {"full_doc_id", "content"})
        base_cypher, base_params = st._build_search_cypher(20, "folderX")
        f_cypher, f_params = st._build_search_cypher(20, "folderX", RetrievalFilters())
        assert f_cypher == base_cypher
        assert f_params == base_params

    def test_no_folder_no_filters_is_legacy(self):
        st = _store("chunks", {"full_doc_id", "content"})
        cypher, _ = st._build_search_cypher(20, None, RetrievalFilters())
        assert "MEMBER_OF" not in cypher
        assert "Folder_" not in cypher
        assert "TAGGED_WITH" not in cypher

    def test_min_score_zero_keeps_legacy_threshold(self):
        st = _store("chunks", {"full_doc_id"})
        _, base = st._build_search_cypher(20, "f")
        _, params = st._build_search_cypher(20, "f", RetrievalFilters(min_score=0.0))
        assert params["threshold"] == base["threshold"] == 0.2


# ── min_score ──────────────────────────────────────────────────────────────


class TestMinScore:
    def test_min_score_raises_floor(self):
        st = _store("chunks", {"full_doc_id"})
        _, params = st._build_search_cypher(20, "f", RetrievalFilters(min_score=0.5))
        assert params["threshold"] == 0.5

    def test_min_score_below_cosine_floor_is_ignored(self):
        st = _store("chunks", {"full_doc_id"})
        _, params = st._build_search_cypher(20, "f", RetrievalFilters(min_score=0.1))
        assert params["threshold"] == 0.2  # max(0.2, 0.1)

    def test_tag_filter_without_min_score_disables_default_floor(self):
        st = _store("chunks", {"full_doc_id"})
        _, params = st._build_search_cypher(
            20, "f", RetrievalFilters(tag_all=frozenset({"oracle"}))
        )
        assert params["threshold"] == 0.0

    def test_tag_filter_with_min_score_respects_explicit_floor(self):
        st = _store("chunks", {"full_doc_id"})
        _, params = st._build_search_cypher(
            20,
            "f",
            RetrievalFilters(tag_all=frozenset({"oracle"}), min_score=0.5),
        )
        assert params["threshold"] == 0.5


# ── doc_filter on chunks ───────────────────────────────────────────────────


class TestDocFilterChunks:
    def test_doc_any_membership(self):
        st = _store("chunks", {"full_doc_id"})
        cypher, params = st._build_search_cypher(
            20, "f", RetrievalFilters(doc_any=frozenset({"A", "B"}))
        )
        assert "CALL vector_search.search" not in cypher
        assert "MATCH (node:`Vec_ws_chunks`)" in cypher
        assert "reduce(__dot" in cypher
        assert "d.id IN $doc_any" in cypher
        assert set(params["doc_any"]) == {"A", "B"}
        assert "overfetch" not in params

    def test_doc_all_is_strict_not_union(self):
        st = _store("chunks", {"full_doc_id"})
        cypher, params = st._build_search_cypher(
            20, "f", RetrievalFilters(doc_all=frozenset({"A", "B"}))
        )
        # ``all(x IN $doc_all WHERE x = d.id)`` ⇒ empty for |all|≥2 on a chunk.
        assert "all(" in cypher
        assert "$doc_all" in cypher
        assert "d.id" in cypher
        assert set(params["doc_all"]) == {"A", "B"}


# ── tag_filter ─────────────────────────────────────────────────────────────


class TestTagFilter:
    def test_tag_join_uses_folder_scoped_label(self):
        st = _store("chunks", {"full_doc_id"})
        cypher, params = st._build_search_cypher(
            20, "folderX", RetrievalFilters(tag_all=frozenset({"oracle"}))
        )
        assert "CALL vector_search.search" not in cypher
        assert "MATCH (node:`Vec_ws_chunks`)" in cypher
        assert "reduce(__dot" in cypher
        assert "TAGGED_WITH" in cypher
        assert "WebuiTag_folderX" in cypher
        assert set(params["tag_all"]) == {"oracle"}
        assert "overfetch" not in params

    def test_tag_all_and_any_both_emitted(self):
        st = _store("chunks", {"full_doc_id"})
        cypher, params = st._build_search_cypher(
            20,
            "f",
            RetrievalFilters(
                tag_all=frozenset({"oracle"}), tag_any=frozenset({"rman", "asm"})
            ),
        )
        assert "$tag_all" in cypher
        assert "$tag_any" in cypher
        assert set(params["tag_all"]) == {"oracle"}
        assert set(params["tag_any"]) == {"rman", "asm"}


# ── entity/relation: globally blended payloads fail closed ────────────────


class TestEntityRelationFilters:
    @pytest.mark.parametrize("namespace", ["entities", "relationships"])
    @pytest.mark.parametrize(
        "filters",
        [
            RetrievalFilters(doc_all=frozenset({"A", "B"})),
            RetrievalFilters(doc_any=frozenset({"A"})),
            RetrievalFilters(tag_all=frozenset({"oracle"})),
            RetrievalFilters(
                doc_any=frozenset({"doc-a"}),
                tag_any=frozenset({"oracle"}),
            ),
        ],
    )
    def test_filters_cannot_make_blended_graph_vectors_safe(self, namespace, filters):
        st = _store(namespace, {"source_id", "content"})
        cypher, params = st._build_search_cypher(20, "folderX", filters)
        assert cypher is None
        assert params == {}


# ── Live contract (real Memgraph) — out-of-filter rows are actually dropped ─
#
# This is the guarantee the audit cared about: an excluded chunk/entity never
# enters the vector-search result, so it can never reach the prompt. The
# builder tests above prove the predicates are *correct*; these prove they
# *exclude* on a real backend across doc / tag / min_score / entity-set cases.

pytestmark_integration = pytest.mark.integration

_WS = "retrieval_filters_test"
_DIM = 8
_FOLDER = "F"


class _FakeEmbed:
    embedding_dim = _DIM

    async def func(self, texts):
        return [[1.0] + [0.0] * (_DIM - 1) for _ in texts]


def _qvec():
    return [1.0] + [0.0] * (_DIM - 1)


def _vec_low():
    # cosine with _qvec() = 0.6 (0.6/1·1) — above the 0.2 floor, below a 0.8
    # min_score, so it isolates the min_score behaviour from the cosine floor.
    return [0.6, 0.8] + [0.0] * (_DIM - 2)


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
    for st in (ds, chunks, entities):
        await st.initialize()

    async def _wipe():
        async with _pool.get_session() as s:
            # REMOVE the indexed Vec_* label before delete (Memgraph 3.10+ stale
            # vector-index guard — see test_folder_query_scoping for the why).
            for lbl in (f"Vec_{_WS}_chunks", f"Vec_{_WS}_entities"):
                await (
                    await s.run(
                        f"MATCH (n:`{lbl}`) REMOVE n:`{lbl}` WITH n DETACH DELETE n"
                    )
                ).consume()
            for lbl in (
                f"DocStatus_{_WS}",
                f"Folder_{_WS}",
                f"WebuiTag_{_FOLDER}",
            ):
                await (await s.run(f"MATCH (n:`{lbl}`) DETACH DELETE n")).consume()

    await _wipe()
    yield ds, chunks, entities
    await _wipe()


async def _seed(ds, chunks, entities):
    from twindb_lightrag_memgraph import _pool

    with storage_folder_context(_FOLDER):
        await ds.upsert({"doc-a": _doc("doc-a"), "doc-b": _doc("doc-b")})
    # Tag doc-a only, via the canonical TAGGED_WITH → WebuiTag_{folder} edge.
    async with _pool.get_session() as s:
        await (
            await s.run(
                f"MATCH (d:`DocStatus_{_WS}` {{id: 'doc-a'}}) "
                f"MERGE (t:`WebuiTag_{_FOLDER}` {{id: 'oracle'}}) "
                f"MERGE (d)-[:TAGGED_WITH]->(t)"
            )
        ).consume()
    await chunks.upsert(
        {
            "chunk-a": {"full_doc_id": "doc-a", "content": "a", "embedding": _qvec()},
            "chunk-b": {
                "full_doc_id": "doc-b",
                "content": "b",
                "embedding": _vec_low(),
            },
        }
    )
    sep = GRAPH_FIELD_SEP
    await entities.upsert(
        {
            "ent-a": {"entity_name": "A", "source_id": "chunk-a", "embedding": _qvec()},
            "ent-mixed": {
                "entity_name": "M",
                "source_id": f"chunk-a{sep}chunk-b",
                "embedding": _qvec(),
            },
        }
    )


async def _ids(store, filters: RetrievalFilters | None):
    with storage_folder_context(_FOLDER), storage_filter_context(filters):
        rows = await store.query("q", top_k=20, query_embedding=_qvec())
    return {r["id"] for r in rows}


@pytestmark_integration
async def test_live_doc_filter_any_drops_other_doc(stores):
    ds, chunks, entities = stores
    await _seed(ds, chunks, entities)
    assert await _ids(chunks, RetrievalFilters(doc_any=frozenset({"doc-a"}))) == {
        "chunk-a"
    }


@pytestmark_integration
async def test_live_doc_filter_all_two_docs_is_empty_for_chunks(stores):
    ds, chunks, entities = stores
    await _seed(ds, chunks, entities)
    # Strict ``all``: a chunk has one doc, so requiring two docs drops every
    # chunk (the pinned semantic; NOT union-as-any).
    assert (
        await _ids(chunks, RetrievalFilters(doc_all=frozenset({"doc-a", "doc-b"})))
        == set()
    )


@pytestmark_integration
async def test_live_tag_filter_drops_untagged_doc(stores):
    ds, chunks, entities = stores
    await _seed(ds, chunks, entities)
    # Only doc-a carries the ``oracle`` tag → chunk-b is excluded from grounding.
    assert await _ids(chunks, RetrievalFilters(tag_all=frozenset({"oracle"}))) == {
        "chunk-a"
    }


@pytestmark_integration
async def test_live_min_score_drops_low_similarity(stores):
    ds, chunks, entities = stores
    await _seed(ds, chunks, entities)
    # sim(chunk-a)=1.0, sim(chunk-b)=0.6. min_score 0.8 drops chunk-b…
    assert await _ids(chunks, RetrievalFilters(min_score=0.8)) == {"chunk-a"}
    # …while a floor below both keeps both (folder-scoped, unfiltered otherwise).
    assert await _ids(chunks, RetrievalFilters(min_score=0.3)) == {
        "chunk-a",
        "chunk-b",
    }


@pytestmark_integration
async def test_live_entity_doc_filter_set_semantics(stores):
    ds, chunks, entities = stores
    await _seed(ds, chunks, entities)
    # Filters cannot unblend graph-vector payloads, so they remain unavailable.
    assert (
        await _ids(entities, RetrievalFilters(doc_all=frozenset({"doc-a", "doc-b"})))
        == set()
    )
    assert await _ids(entities, RetrievalFilters(doc_any=frozenset({"doc-b"}))) == set()


@pytestmark_integration
async def test_live_entity_tag_filter_set_semantics(stores):
    ds, chunks, entities = stores
    await _seed(ds, chunks, entities)
    assert (
        await _ids(entities, RetrievalFilters(tag_all=frozenset({"oracle"}))) == set()
    )
    assert (
        await _ids(entities, RetrievalFilters(tag_all=frozenset({"vmware"}))) == set()
    )
    assert (
        await _ids(entities, RetrievalFilters(tag_any=frozenset({"oracle"}))) == set()
    )
