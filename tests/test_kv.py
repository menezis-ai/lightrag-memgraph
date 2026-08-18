"""
Integration tests for MemgraphKVStorage.

Requires a running Memgraph instance (set MEMGRAPH_URI).
"""

import pytest

from twindb_lightrag_memgraph import register
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

register()


@pytest.fixture
async def kv_store():
    store = MemgraphKVStorage(
        namespace="test_kv",
        global_config={},
        embedding_func=None,
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.mark.integration
class TestMemgraphKVStorage:
    async def test_upsert_and_get(self, kv_store):
        await kv_store.upsert({"key1": {"hello": "world", "num": 42}})
        result = await kv_store.get_by_id("key1")
        assert result is not None
        assert result["hello"] == "world"
        assert result["num"] == 42

    async def test_get_by_ids(self, kv_store):
        await kv_store.upsert(
            {
                "a": {"val": 1},
                "b": {"val": 2},
                "c": {"val": 3},
            }
        )
        results = await kv_store.get_by_ids(["a", "c", "missing"])
        assert len(results) == 3
        assert results[0]["val"] == 1
        assert results[1]["val"] == 3
        assert results[2] is None

    async def test_get_missing_key(self, kv_store):
        result = await kv_store.get_by_id("nonexistent")
        assert result is None

    async def test_filter_keys(self, kv_store):
        await kv_store.upsert({"existing": {"val": 1}})
        missing = await kv_store.filter_keys({"existing", "absent"})
        assert "absent" in missing
        assert "existing" not in missing

    async def test_delete(self, kv_store):
        await kv_store.upsert({"to_delete": {"val": 1}})
        await kv_store.delete(["to_delete"])
        result = await kv_store.get_by_id("to_delete")
        assert result is None

    async def test_is_empty(self, kv_store):
        assert await kv_store.is_empty() is True
        await kv_store.upsert({"key": {"val": 1}})
        assert await kv_store.is_empty() is False

    async def test_upsert_overwrites(self, kv_store):
        await kv_store.upsert({"key": {"version": 1}})
        await kv_store.upsert({"key": {"version": 2}})
        result = await kv_store.get_by_id("key")
        assert result["version"] == 2

    async def test_drop(self, kv_store):
        await kv_store.upsert({"key": {"val": 1}})
        result = await kv_store.drop()
        assert result["status"] == "success"
        assert await kv_store.is_empty() is True

    # ── LightRAG 1.5.5 strict point reads ──────────────────────────────
    # The manual FAILED-retry protocol trusts None as "content really
    # absent"; the strict path must distinguish that from an existing node
    # whose payload is unusable (review blocker on kv_impl:get_by_id_strict).

    async def test_get_by_id_strict_returns_value_and_confirmed_absence(self, kv_store):
        await kv_store.upsert({"key1": {"hello": "world"}})
        assert await kv_store.get_by_id_strict("key1") == {"hello": "world"}
        assert await kv_store.get_by_id_strict("missing") is None

    async def test_get_by_id_strict_raises_on_unusable_payload(self, kv_store):
        from twindb_lightrag_memgraph import _pool
        from twindb_lightrag_memgraph.kv_impl import StorageControlPlaneError

        label = kv_store._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                for node_id, data in (
                    ("empty-node", ""),
                    ("null-node", "null"),
                    ("broken-node", "{not json"),
                ):
                    result = await session.run(
                        f"MERGE (n:`{label}` {{id: $id}}) SET n.data = $data",
                        id=node_id,
                        data=data,
                    )
                    await result.consume()

        for node_id in ("empty-node", "null-node", "broken-node"):
            with pytest.raises(StorageControlPlaneError):
                await kv_store.get_by_id_strict(node_id)
        # The lenient reader keeps its best-effort miss semantics where it
        # already had them (empty payload / JSON null → None).
        assert await kv_store.get_by_id("empty-node") is None
        assert await kv_store.get_by_id("null-node") is None
