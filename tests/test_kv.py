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
