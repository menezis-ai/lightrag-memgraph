"""Version negotiation and real-server regressions for Memgraph 3.2–3.14."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc
from neo4j.exceptions import ClientError

from lightrag_memgraph import _pool
from lightrag_memgraph.capabilities import MemgraphCapabilities, inspect_memgraph
from lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


async def embed(texts):
    return np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts])


@pytest.fixture
def make_store(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "compat")

    def make(namespace="vectors", dim=4, **options):
        return MemgraphVectorDBStorage(
            namespace,
            {"vector_db_storage_cls_kwargs": options},
            EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=embed),
            meta_fields={"content", "entity_name"},
        )

    return make


def capabilities(version="3.14.0", config=True, scores=True):
    arguments = ", config = {} :: MAP" if config else ", limit = 1000 :: INTEGER"
    output = "node :: NODE, score :: FLOAT" if scores else "node :: NODE"
    return MemgraphCapabilities(
        version,
        (
            ("vector_search.search", ""),
            (
                "text_search.search_all",
                f"text_search.search_all(index :: STRING{arguments}) :: ({output})",
            ),
        ),
    )


@pytest.mark.parametrize(
    "version,quantization,text",
    [
        ("3.2.1", False, False),
        ("3.4.0", True, False),
        ("3.6.0", True, True),
        ("3.8.1", True, True),
        ("3.13.2", True, True),
        ("3.14.0", True, True),
    ],
)
def test_capabilities_version_gates(version, quantization, text):
    caps = capabilities(version)
    assert caps.vector_quantization is quantization
    assert caps.text_search is text


def test_procedure_signatures_are_authoritative():
    old = capabilities("3.8.1", config=False, scores=False)
    new = capabilities()
    assert not old.accepts_config("text_search.search_all")
    assert old.accepts_limit("text_search.search_all")
    assert not old.returns_score("text_search.search_all")
    assert new.accepts_config("text_search.search_all")
    assert not new.accepts_limit("text_search.search_all")
    assert new.returns_score("text_search.search_all")
    assert not new.has_procedure("text_search.fuzzy_phrase_search")
    assert not MemgraphCapabilities("3.14.0", ()).text_search


@pytest.mark.parametrize(
    "options",
    [
        {"capacity": 0},
        {"capacity": True},
        {"capacity": "100"},
        {"resize_coefficient": float("inf")},
        {"resize_coefficient": 1},
        {"scalar_kind": 'f16"} RETURN 1'},
        {"dimension": 128},
        {"metric": "l2sq"},
    ],
)
def test_invalid_index_options_fail_before_database_access(make_store, options):
    with pytest.raises(ValueError):
        make_store(memgraph_vector_index_config=options)


async def test_empty_query_does_not_embed_or_connect(make_store):
    store = make_store()
    store.embedding_func.func = AsyncMock(side_effect=AssertionError("must not embed"))
    assert await store.query("hello", 0) == []
    assert await store.query_hybrid("hello", -1) == []
    assert await store.query_text("hello", 0) == []


async def test_ddl_errors_are_not_swallowed():
    session = AsyncMock()
    session.run.side_effect = ClientError("Permission denied")
    with pytest.raises(ClientError, match="Permission denied"):
        await MemgraphVectorDBStorage._create_index(session, "CREATE INDEX")
    session.run.side_effect = None
    session.run.return_value.consume.side_effect = ClientError("deferred DDL failure")
    with pytest.raises(ClientError, match="deferred DDL failure"):
        await MemgraphVectorDBStorage._create_index(session, "CREATE INDEX")


async def test_discovery_errors_propagate():
    session = AsyncMock()
    session.run.side_effect = ClientError("Inspection denied")
    with pytest.raises(ClientError, match="Inspection denied"):
        await inspect_memgraph(session)


@pytest.mark.parametrize("known_community", [False, True])
async def test_custom_database_never_falls_back_to_default(
    monkeypatch, known_community
):
    monkeypatch.setattr(
        _pool, "_enterprise_supported", False if known_community else None
    )
    session = AsyncMock()
    session.run.side_effect = ClientError(
        "Trying to use enterprise feature without a valid license"
    )
    with pytest.raises(RuntimeError, match="requires Memgraph Enterprise"):
        await _pool._try_use_database(session, "private_tenant")


async def test_deferred_database_selection_error_is_checked(monkeypatch):
    monkeypatch.setattr(_pool, "_enterprise_supported", None)
    session = AsyncMock()
    session.run.return_value.consume.side_effect = ClientError("database is suspended")
    with pytest.raises(ClientError, match="database is suspended"):
        await _pool._try_use_database(session, "custom_db")


async def test_old_text_signature_uses_no_config_or_score(make_store, monkeypatch):
    store = make_store()
    store.capabilities = capabilities("3.8.1", config=False, scores=False)
    store._text_index_ready = True
    result = AsyncMock()
    result.__aiter__.return_value = [
        {
            "id": "doc",
            "score": 1.0,
            "props": {"content": "hello"},
            "embedding": [1, 0, 0, 0],
        }
    ]
    session = AsyncMock()
    session.run.return_value = result

    @asynccontextmanager
    async def read_session():
        yield session

    monkeypatch.setattr(_pool, "get_read_session", read_session)
    hits = await store.query_text("hello", 3)
    assert hits[0]["content"] == "hello"
    (query,) = session.run.call_args.args
    assert "YIELD node WITH" in query
    assert "1.0 AS score" in query
    assert "config" not in session.run.call_args.kwargs
    with pytest.raises(RuntimeError, match="does not support"):
        await store.query_text("hello", 3, config={"fuzzy_distance": 1})


async def test_regex_rejects_fuzzy_configuration_before_database_access(
    make_store, monkeypatch
):
    store = make_store()
    store._text_index_ready = True
    store.capabilities = MemgraphCapabilities(
        "3.14.0",
        (("text_search.regex_search", "regex_search(config = {} :: MAP)"),),
    )

    def forbidden_session():
        raise AssertionError("Invalid regex configuration must not reach the database")

    monkeypatch.setattr(_pool, "get_read_session", forbidden_session)
    with pytest.raises(ValueError, match="Regex.*fuzzy"):
        await store.query_text(
            "need.*", 5, mode="regex_search", config={"fuzzy_distance": 1}
        )


@pytest.mark.integration
class TestRealMemgraphCompatibility:
    @pytest.mark.parametrize(
        "assignment,props",
        [
            (
                "= $props",
                {"id": "doc", "content": "uncommitted", "embedding": [0, 1, 0, 0]},
            ),
            ("= $props", {}),
            ("+= $props", {"content": "uncommitted", "embedding": [0, 1, 0, 0]}),
        ],
        ids=["replace", "clear", "update"],
    )
    async def test_rolled_back_map_write_preserves_embedding(
        self, make_store, assignment, props
    ):
        store = make_store(namespace="rollback")
        await store.initialize()
        try:
            if store.capabilities.version_tuple < (3, 14, 0):
                pytest.skip("Vector map-write rollback fix requires Memgraph 3.14")
            embedding = [1, 0, 0, 0]
            await store.upsert({"doc": {"content": "original", "embedding": embedding}})
            async with _pool.get_session() as session:
                async with await session.begin_transaction() as transaction:
                    result = await transaction.run(
                        f"MATCH (n:`{store._label()}` {{id: $id}}) SET n {assignment}",
                        id="doc",
                        props=props,
                    )
                    await result.consume()
                    # A different session must still see the committed vector.
                    assert (await store.get_vectors_by_ids(["doc"]))["doc"] == embedding
                    await transaction.rollback()
            assert (await store.get_by_id("doc"))["content"] == "original"
            assert (await store.get_vectors_by_ids(["doc"]))["doc"] == embedding
            assert [hit["id"] for hit in await store.query("", 1, embedding)] == ["doc"]
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_failed_vector_index_creation_preserves_embedding(self, make_store):
        store = make_store(namespace="failed_index")
        failed_index = store._index_name() + "_invalid"
        await store.initialize()
        try:
            if store.capabilities.version_tuple < (3, 14, 0):
                pytest.skip("Failed vector-index creation fix requires Memgraph 3.14")
            embedding = [1, 0, 0, 0]
            await store.upsert({"doc": {"content": "original", "embedding": embedding}})
            async with _pool.get_session() as session:
                await (
                    await session.run(f"DROP VECTOR INDEX `{store._index_name()}`")
                ).consume()
                with pytest.raises(ClientError, match="(?i)dimension"):
                    await MemgraphVectorDBStorage._create_index(
                        session,
                        f"CREATE VECTOR INDEX `{failed_index}` "
                        f"ON :`{store._label()}`(embedding) "
                        'WITH CONFIG {dimension: 8, capacity: 100, metric: "cos"}',
                    )
                rows = await (await session.run("SHOW VECTOR INDEX INFO")).data()
                assert all(row["index_name"] != failed_index for row in rows)
            assert (await store.get_by_id("doc"))["content"] == "original"
            assert (await store.get_vectors_by_ids(["doc"]))["doc"] == embedding
            # The adapter recreates its missing index from the original config.
            assert [hit["id"] for hit in await store.query("", 1, embedding)] == ["doc"]
        finally:
            async with _pool.get_session() as session:
                rows = await (await session.run("SHOW VECTOR INDEX INFO")).data()
                if any(row["index_name"] == failed_index for row in rows):
                    await (
                        await session.run(f"DROP VECTOR INDEX `{failed_index}`")
                    ).consume()
            assert (await store.drop())["status"] == "success"

    @pytest.mark.parametrize(
        "mode,query",
        [
            ("search_all", "needle"),
            ("search", "data.content:needle"),
            ("regex_search", "need.*"),
        ],
    )
    async def test_text_top_k_can_exceed_the_native_default(
        self, make_store, mode, query
    ):
        store = make_store(namespace="text_limit", memgraph_text_search=True)
        await store.initialize()
        try:
            if not store.capabilities.text_search:
                pytest.skip("Stable text search is unavailable")
            count = 1005
            await store.upsert(
                {
                    f"doc_{i}": {"content": "needle", "embedding": [1, 0, 0, 0]}
                    for i in range(count)
                }
            )
            hits = await store.query_text(query, count, mode=mode)
            assert len(hits) == count
            assert len({hit["id"] for hit in hits}) == count
        finally:
            assert (await store.drop())["status"] == "success"

    @pytest.mark.parametrize(
        "mode,query", [("search", "data.content:needle"), ("regex_search", "need.*")]
    )
    async def test_property_qualified_and_regex_text_search(
        self, make_store, mode, query
    ):
        store = make_store(namespace="text_modes", memgraph_text_search=True)
        await store.initialize()
        try:
            if not store.capabilities.text_search:
                pytest.skip("Stable text search is unavailable")
            await store.upsert(
                {
                    "match": {"content": "needle in the graph"},
                    "other": {"content": "unrelated words"},
                }
            )
            hits = await store.query_text(query, 1, mode=mode)
            assert [hit["id"] for hit in hits] == ["match"]
            assert hits[0]["score"] > 0
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_fuzzy_terms_prefix_and_transposition_configuration(self, make_store):
        store = make_store(namespace="fuzzy_terms", memgraph_text_search=True)
        await store.initialize()
        try:
            if not store.capabilities.accepts_config("text_search.search_all"):
                pytest.skip(
                    "Fuzzy term configuration requires the newer text signature"
                )
            await store.upsert({"doc": {"content": "needle big bad wolf"}})
            typo = await store.query_text("nedle", 1, config={"fuzzy_distance": 1})
            prefix = await store.query_text(
                "nee", 1, config={"fuzzy_distance": 1, "fuzzy_prefix": True}
            )
            swapped = await store.query_text(
                "needel", 1, config={"fuzzy_distance": 1, "fuzzy_transpositions": True}
            )
            strict = await store.query_text(
                "needel", 1, config={"fuzzy_distance": 1, "fuzzy_transpositions": False}
            )
            assert [hit["id"] for hit in typo] == ["doc"]
            assert [hit["id"] for hit in prefix] == ["doc"]
            assert [hit["id"] for hit in swapped] == ["doc"]
            assert strict == []
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_missing_vector_index_is_recreated_with_its_options(self, make_store):
        store = make_store(
            namespace="recovery",
            memgraph_vector_index_config={"capacity": 100, "scalar_kind": "f16"},
        )
        await store.initialize()
        try:
            await store.upsert({"doc": {"embedding": [1, 0, 0, 0]}})
            async with _pool.get_session() as session:
                await (
                    await session.run(f"DROP VECTOR INDEX `{store._index_name()}`")
                ).consume()
            assert (await store.query("", 1, [1, 0, 0, 0]))[0]["id"] == "doc"
            async with _pool.get_read_session() as session:
                rows = await (await session.run("SHOW VECTOR INDEX INFO")).data()
            index = next(
                row for row in rows if row["index_name"] == store._index_name()
            )
            assert index["dimension"] == 4
            if store.capabilities.vector_quantization:
                assert index["scalar_kind"] == "f16"
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_deleted_index_hits_do_not_hide_live_neighbours(self, make_store):
        store = make_store()
        await store.initialize()
        try:
            if store.capabilities.version_tuple < (3, 4, 0):
                pytest.xfail(
                    "Legacy native ANN can lose live results or SIGSEGV after "
                    "repeated deletes; also reproduced with direct Bolt queries"
                )
            for i in range(12):
                doc_id = f"doc_{i}"
                await store.upsert({doc_id: {"embedding": [1, 0, 0, 0]}})
                hits = await store.query("", 1, [1, 0, 0, 0])
                assert [hit["id"] for hit in hits] == [doc_id]
                await store.delete([doc_id])
            assert await store.query("", 1, [1, 0, 0, 0]) == []
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_existing_namespace_and_embeddings_survive_reinitialize(
        self, make_store
    ):
        store = make_store()
        await store.initialize()
        try:
            await store.upsert(
                {"doc": {"content": "old data", "embedding": [1, 0, 0, 0]}}
            )
            reopened = make_store()
            await reopened.initialize()
            assert (await reopened.get_by_id("doc"))["content"] == "old data"
            assert (await reopened.get_vectors_by_ids(["doc"]))["doc"] == [1, 0, 0, 0]
            assert (await reopened.query("", 2, [1, 0, 0, 0]))[0]["id"] == "doc"
            with pytest.raises(ValueError, match="dimension"):
                await make_store(dim=8).initialize()
            assert await store.get_by_id("doc") is not None
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_advanced_options_fallback_or_strict(self, make_store):
        store = make_store(
            memgraph_query_mode="hybrid",
            memgraph_vector_index_config={
                "scalar_kind": "f16",
                "capacity": 100,
                "resize_coefficient": 2,
            },
        )
        await store.initialize()
        try:
            await store.upsert(
                {"doc": {"content": "needle", "embedding": [1, 0, 0, 0]}}
            )
            assert (await store.query("needle", 1, [1, 0, 0, 0]))[0]["id"] == "doc"
            assert store._text_index_ready is store.capabilities.text_search
            if not store.capabilities.vector_quantization:
                with pytest.raises(RuntimeError, match="quantization"):
                    await make_store(
                        memgraph_feature_policy="strict",
                        memgraph_vector_index_config={"scalar_kind": "f16"},
                    ).initialize()
            else:
                with pytest.raises(ValueError, match="scalar_kind"):
                    await make_store(
                        memgraph_vector_index_config={"scalar_kind": "f32"}
                    ).initialize()
            if store.capabilities.text_search:
                with pytest.raises(ValueError, match="Text index"):
                    await make_store(
                        memgraph_text_search=True,
                        memgraph_text_properties=["title"],
                    ).initialize()
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_text_fuzzy_and_hybrid_union(self, make_store):
        store = make_store(
            memgraph_query_mode="hybrid", cosine_better_than_threshold=0.7
        )
        await store.initialize()
        try:
            if not store.capabilities.text_search:
                pytest.skip("Stable text search is not available on this server")
            await store.upsert(
                {
                    "lexical": {
                        "content": "needle big bad wolf",
                        "embedding": [0, 1, 0, 0],
                    },
                    "semantic": {"content": "unrelated", "embedding": [1, 0, 0, 0]},
                    "shared": {
                        "content": "needle big bad wolf",
                        "embedding": [0.8, 0.6, 0, 0],
                    },
                }
            )
            text_hits = await store.query_text("needle", 5)
            assert {hit["id"] for hit in text_hits} == {"lexical", "shared"}
            assert all(hit["score"] > 0 for hit in text_hits)
            hits = await store.query("needle", 3, [1, 0, 0, 0])
            assert hits[0]["id"] == "shared"
            assert {hit["id"] for hit in hits} == {"lexical", "semantic", "shared"}
            lexical = next(hit for hit in hits if hit["id"] == "lexical")
            assert lexical["similarity"] == pytest.approx(0)
            assert lexical["distance"] == pytest.approx(1)
            assert "embedding" not in lexical
            if store.capabilities.has_procedure("text_search.fuzzy_phrase_search"):
                fuzzy = await store.query_text(
                    "data.content:big bd wo",
                    5,
                    mode="fuzzy_phrase_search",
                    config={"fuzzy_distance": 1},
                )
                assert {hit["id"] for hit in fuzzy} == {"lexical", "shared"}
        finally:
            assert (await store.drop())["status"] == "success"

    async def test_text_indexes_are_scoped_to_workspace(self, make_store, monkeypatch):
        first = make_store(memgraph_text_search=True)
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "compat_other")
        second = make_store(memgraph_text_search=True)
        await first.initialize()
        await second.initialize()
        try:
            if not first.capabilities.text_search:
                pytest.skip("Stable text search is not available on this server")
            for store, doc_id in ((first, "first"), (second, "second")):
                await store.upsert(
                    {doc_id: {"content": "needle", "embedding": [1, 0, 0, 0]}}
                )
            assert [hit["id"] for hit in await first.query_text("needle", 5)] == [
                "first"
            ]
            assert [hit["id"] for hit in await second.query_text("needle", 5)] == [
                "second"
            ]
        finally:
            assert (await first.drop())["status"] == "success"
            assert (await second.drop())["status"] == "success"
