"""Unit tests for the write-throttle semaphore in _pool.py.

No Memgraph required — all DB interactions are mocked.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import twindb_lightrag_memgraph._pool as pool
from twindb_lightrag_memgraph._constants import DEFAULT_WRITE_CONCURRENCY


@pytest.fixture(autouse=True)
def reset_semaphore():
    """Reset semaphore state between tests."""
    pool._write_semaphore = None
    pool._semaphore_loop_id = None
    yield
    pool._write_semaphore = None
    pool._semaphore_loop_id = None


# ── _read_write_concurrency ────────────────────────────────────────────


class TestReadWriteConcurrency:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_WRITE_CONCURRENCY", raising=False)
        assert pool._read_write_concurrency() == DEFAULT_WRITE_CONCURRENCY

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WRITE_CONCURRENCY", "3")
        assert pool._read_write_concurrency() == 3

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WRITE_CONCURRENCY", "abc")
        assert pool._read_write_concurrency() == DEFAULT_WRITE_CONCURRENCY

    def test_zero_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WRITE_CONCURRENCY", "0")
        assert pool._read_write_concurrency() == DEFAULT_WRITE_CONCURRENCY

    def test_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WRITE_CONCURRENCY", "-5")
        assert pool._read_write_concurrency() == DEFAULT_WRITE_CONCURRENCY


# ── acquire_write_slot ──────────────────────────────────────────────────


class TestAcquireWriteSlot:
    async def test_basic_enter_exit(self):
        async with pool.acquire_write_slot():
            pass  # Should not raise

    async def test_concurrency_limit_respected(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WRITE_CONCURRENCY", "2")

        high_water = 0
        current = 0

        async def worker():
            nonlocal high_water, current
            async with pool.acquire_write_slot():
                current += 1
                if current > high_water:
                    high_water = current
                await asyncio.sleep(0.01)
                current -= 1

        await asyncio.gather(*[worker() for _ in range(5)])
        assert high_water <= 2

    async def test_slot_released_on_exception(self):
        with pytest.raises(RuntimeError, match="boom"):
            async with pool.acquire_write_slot():
                raise RuntimeError("boom")

        # Slot should be released — subsequent acquire must succeed
        async with pool.acquire_write_slot():
            pass

    async def test_semaphore_recreated_on_loop_change(self):
        # Acquire once to create the semaphore
        async with pool.acquire_write_slot():
            pass

        old_sem = pool._write_semaphore
        assert old_sem is not None

        # Simulate a loop change
        pool._semaphore_loop_id = -1

        async with pool.acquire_write_slot():
            pass

        assert pool._write_semaphore is not old_sem


# ── Backend integration (mock-based) ───────────────────────────────────


def _mock_session():
    """Return an async context manager yielding a mock session."""
    session = AsyncMock()
    result = AsyncMock()
    result.consume = AsyncMock()
    result.single = AsyncMock(return_value=None)
    session.run = AsyncMock(return_value=result)

    @asynccontextmanager
    async def ctx():
        yield session

    return ctx


def _mock_driver():
    """Return a coroutine that yields (mock_driver, 'memgraph')."""
    driver = MagicMock()

    async def get_driver():
        return driver, "memgraph"

    return get_driver


class TestKVWriteSlotIntegration:
    """Verify KV backend acquires write slot for writes but not reads."""

    @pytest.fixture
    def kv_store(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

        store = MemgraphKVStorage.__new__(MemgraphKVStorage)
        store.namespace = "test_kv"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        return store

    async def test_upsert_acquires_write_slot(self, kv_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_session", _mock_session()),
        ):
            await kv_store.upsert({"k1": {"v": 1}})

        assert len(entered) == 1

    async def test_get_by_id_does_not_acquire_write_slot(self, kv_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_read_session", _mock_session()),
        ):
            await kv_store.get_by_id("k1")

        assert len(entered) == 0

    async def test_delete_acquires_write_slot(self, kv_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_session", _mock_session()),
        ):
            await kv_store.delete(["k1"])

        assert len(entered) == 1

    async def test_drop_acquires_write_slot(self, kv_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_session", _mock_session()),
        ):
            await kv_store.drop()

        assert len(entered) == 1


class TestDocStatusWriteSlotIntegration:
    """Verify DocStatus backend acquires write slot for writes but not reads."""

    @pytest.fixture
    def doc_store(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

        store = MemgraphDocStatusStorage.__new__(MemgraphDocStatusStorage)
        store.namespace = "test_doc"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        return store

    async def test_upsert_acquires_write_slot(self, doc_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_session", _mock_session()),
        ):
            await doc_store.upsert({"doc1": {"status": "pending"}})

        assert len(entered) == 1

    async def test_get_by_id_does_not_acquire_write_slot(self, doc_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_read_session", _mock_session()),
        ):
            await doc_store.get_by_id("doc1")

        assert len(entered) == 0


class TestVectorWriteSlotIntegration:
    """Verify Vector backend acquires write slot for writes but not reads."""

    @pytest.fixture
    def vec_store(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

        store = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
        store.namespace = "test_vec"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        store.meta_fields = set()
        store.cosine_better_than_threshold = 0.2
        return store

    async def test_upsert_acquires_write_slot(self, vec_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_session", _mock_session()),
        ):
            await vec_store.upsert({"v1": {"embedding": [0.1, 0.2]}})

        assert len(entered) == 1

    async def test_delete_entity_acquires_write_slot(self, vec_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_session", _mock_session()),
        ):
            await vec_store.delete_entity("some_entity")

        assert len(entered) == 1

    async def test_get_by_id_does_not_acquire_write_slot(self, vec_store):
        entered = []

        @asynccontextmanager
        async def mock_slot():
            entered.append(1)
            yield

        with (
            patch.object(pool, "acquire_write_slot", mock_slot),
            patch.object(pool, "get_read_session", _mock_session()),
        ):
            await vec_store.get_by_id("v1")

        assert len(entered) == 0
