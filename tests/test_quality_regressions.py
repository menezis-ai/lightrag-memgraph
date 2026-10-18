"""Regression checks for lifecycle, configuration and installed resources."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from lightrag.utils import EmbeddingFunc
from neo4j import AsyncGraphDatabase

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph._constants import (
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
    validate_identifier,
)
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


@pytest.mark.parametrize("read", [False, True])
async def test_pool_replacement_does_not_block_other_coroutines(monkeypatch, read):
    prefix = "_read" if read else ""
    old = MagicMock()
    new = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def close():
        entered.set()
        await release.wait()

    old.close = AsyncMock(side_effect=close)
    monkeypatch.setattr(_pool, f"{prefix}_driver", old)
    monkeypatch.setattr(_pool, f"{prefix}_database", "memgraph")
    monkeypatch.setattr(_pool, f"{prefix}_bound_loop_id", -1)
    monkeypatch.setattr(_pool, "_enterprise_supported", None)
    factory = MagicMock(return_value=new)
    monkeypatch.setattr(_pool.AsyncGraphDatabase, "driver", factory)
    getter = _pool._get_read_driver if read else _pool.get_driver
    task = asyncio.create_task(getter())
    await asyncio.wait_for(entered.wait(), 2)
    try:
        # Check before calling getter again: a synchronous lock held across
        # close() would freeze the entire event loop rather than time out.
        assert not _pool._thread_lock.locked()
        assert (await getter())[0] is new
    finally:
        release.set()
        await task
    factory.assert_called_once()


@pytest.mark.parametrize("scheme", ["bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"])
async def test_tls_uri_can_be_combined_with_encrypted_true(monkeypatch, scheme):
    monkeypatch.setenv("MEMGRAPH_URI", f"{scheme}://localhost:7687")
    monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "true")
    monkeypatch.delenv("MEMGRAPH_TRUST", raising=False)
    uri, _, kwargs = _pool._read_connection_config()
    driver = AsyncGraphDatabase.driver(uri, **kwargs)
    await driver.close()


@pytest.mark.parametrize(
    "scheme,trust",
    [("bolt+s", "TRUST_ALL"), ("bolt+ssc", "TRUST_SYSTEM_CA")],
)
def test_tls_uri_rejects_conflicting_trust(monkeypatch, scheme, trust):
    monkeypatch.setenv("MEMGRAPH_URI", f"{scheme}://localhost:7687")
    monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "true")
    monkeypatch.setenv("MEMGRAPH_TRUST", trust)
    with pytest.raises(ValueError, match="MEMGRAPH_TRUST conflicts"):
        _pool._read_connection_config()


def test_tls_uri_rejects_explicit_plaintext(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_URI", "bolt+s://localhost:7687")
    monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "false")
    monkeypatch.delenv("MEMGRAPH_TRUST", raising=False)
    with pytest.raises(ValueError, match="MEMGRAPH_ENCRYPTED=false conflicts"):
        _pool._read_connection_config()


@pytest.mark.parametrize("setting", ["nan", "inf", "-inf"])
def test_connection_timeout_is_finite(monkeypatch, setting):
    monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", setting)
    assert (
        _pool._read_connection_acquire_timeout() == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
    )


@pytest.mark.parametrize("identifier", ["trailing\n", "name\r\n", 123, ["name"]])
def test_identifier_validation_rejects_non_identifiers(identifier):
    with pytest.raises(ValueError):
        validate_identifier(identifier)


@pytest.fixture
def store_factory(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "quality")

    async def embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def make(namespace="regression", meta_fields=None, **options):
        return MemgraphVectorDBStorage(
            namespace,
            {"vector_db_storage_cls_kwargs": options},
            EmbeddingFunc(embedding_dim=4, max_token_size=8192, func=embed),
            meta_fields=meta_fields or {"content"},
        )

    return make


def test_metadata_cannot_replace_vector_scores(store_factory):
    store = store_factory(meta_fields={"similarity", "distance", "content"})
    entry = store._record_to_entry(
        {
            "id": "doc",
            "similarity": 0.8,
            "props": {"similarity": -100, "distance": 999, "content": "hello"},
        }
    )
    assert entry["similarity"] == pytest.approx(0.8)
    assert entry["distance"] == pytest.approx(0.2)
    assert entry["content"] == "hello"


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), "0.2", 2.0])
def test_invalid_threshold_fails_before_search(store_factory, threshold):
    with pytest.raises(ValueError):
        store_factory(cosine_better_than_threshold=threshold)


@pytest.mark.integration
async def test_drop_removes_text_index_after_vector_only_reopen(store_factory):
    first = store_factory(memgraph_text_search=True)
    await first.initialize()
    try:
        if not first.capabilities.text_search:
            pytest.skip("Requires stable text indexes")
        reopened = store_factory()
        await reopened.initialize()
        assert (await reopened.drop())["status"] == "success"
        async with _pool.get_read_session() as session:
            rows = await (await session.run("SHOW INDEX INFO")).data()
        assert not any(first._text_index_name() in str(row) for row in rows)
    finally:
        await first.drop()


@pytest.mark.integration
async def test_metadata_update_preserves_existing_vector(store_factory):
    store = store_factory()
    await store.initialize()
    try:
        await store.upsert(
            {"doc": {"embedding": [1, 0, 0, 0], "entity_name": "before"}}
        )
        await store.upsert({"doc": {"entity_name": "after"}})
        assert (await store.get_vectors_by_ids(["doc"]))["doc"] == [1, 0, 0, 0]
    finally:
        await store.drop()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["VECTOR", "TEXT"])
async def test_drop_preserves_data_when_index_belongs_to_another_label(
    store_factory, kind
):
    store = store_factory()
    await store.initialize()
    if kind == "TEXT" and not store.capabilities.text_search:
        await store.drop()
        pytest.skip("Requires stable text indexes")
    index_name = store._index_name() if kind == "VECTOR" else store._text_index_name()
    label = store._label()
    async with _pool.get_session() as session:
        if kind == "VECTOR":
            await (await session.run(f"DROP VECTOR INDEX `{index_name}`")).consume()
        await (await session.run(f"CREATE (n:`{label}` {{id: 'keep'}})")).consume()
        query = (
            f"CREATE VECTOR INDEX `{index_name}` ON :QualityOther(embedding) "
            'WITH CONFIG {"dimension": 4, "capacity": 100, "metric": "cos"}'
            if kind == "VECTOR"
            else f"CREATE TEXT INDEX `{index_name}` ON :QualityOther(content)"
        )
        await (await session.run(query)).consume()
    try:
        with pytest.raises(ValueError, match="different namespace"):
            await store.drop()
        assert (await store.get_by_id("keep"))["id"] == "keep"
        async with _pool.get_session() as session:
            if kind == "VECTOR":
                indexes = await (await session.run("SHOW VECTOR INDEX INFO")).data()
                assert any(row["index_name"] == index_name for row in indexes)
            else:
                assert await store._get_text_index(session) is not None
    finally:
        async with _pool.get_session() as session:
            await (await session.run(f"DROP {kind} INDEX `{index_name}`")).consume()
        await store.drop()
