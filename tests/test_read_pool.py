"""Unit tests for read-path fast lane (dual pool) in _pool.py.

No Memgraph required — all driver creation is mocked.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import twindb_lightrag_memgraph._pool as pool


class _AsyncEmpty:
    """Async iterator that yields nothing — for mocking `async for record in result`."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.fixture(autouse=True)
def reset_all_pool_state():
    """Reset ALL pool state between tests (both write and read)."""
    pool._driver = None
    pool._database = None
    pool._bound_loop_id = None
    pool._read_driver = None
    pool._read_database = None
    pool._read_bound_loop_id = None
    pool._write_semaphore = None
    pool._semaphore_loop_id = None
    yield
    pool._driver = None
    pool._database = None
    pool._bound_loop_id = None
    pool._read_driver = None
    pool._read_database = None
    pool._read_bound_loop_id = None
    pool._write_semaphore = None
    pool._semaphore_loop_id = None


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


# ── Dual pool isolation ───────────────────────────────────────────────


class TestDualPool:
    async def test_read_driver_created_separately(self):
        """_get_read_driver creates a separate driver from get_driver."""
        drivers_created = []

        with patch("twindb_lightrag_memgraph._pool.AsyncGraphDatabase") as mock_agd:

            def make_driver(*args, **kwargs):
                d = MagicMock()
                drivers_created.append(d)
                return d

            mock_agd.driver = make_driver

            write_driver, _ = await pool.get_driver()
            read_driver, _ = await pool._get_read_driver()

        assert len(drivers_created) == 2
        assert write_driver is not read_driver

    async def test_read_pool_uses_configured_size(self, monkeypatch):
        """Read pool passes its own pool size to the driver."""
        monkeypatch.setenv("MEMGRAPH_READ_POOL_SIZE", "25")
        captured_kwargs = {}

        with patch("twindb_lightrag_memgraph._pool.AsyncGraphDatabase") as mock_agd:

            def make_driver(uri, **kwargs):
                captured_kwargs.update(kwargs)
                return MagicMock()

            mock_agd.driver = make_driver
            await pool._get_read_driver()

        assert captured_kwargs["max_connection_pool_size"] == 25

    async def test_read_driver_cached_same_loop(self):
        """Read driver is reused within the same event loop."""
        call_count = 0

        with patch("twindb_lightrag_memgraph._pool.AsyncGraphDatabase") as mock_agd:

            def make_driver(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return MagicMock()

            mock_agd.driver = make_driver

            d1, _ = await pool._get_read_driver()
            d2, _ = await pool._get_read_driver()

        assert d1 is d2
        assert call_count == 1

    async def test_read_driver_recreated_on_loop_change(self):
        """Read driver is recreated when event loop changes."""
        mock_driver_1 = MagicMock()
        mock_driver_1.close = AsyncMock()
        mock_driver_2 = MagicMock()
        call_count = 0

        with patch("twindb_lightrag_memgraph._pool.AsyncGraphDatabase") as mock_agd:

            def make_driver(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_driver_1 if call_count == 1 else mock_driver_2

            mock_agd.driver = make_driver

            d1, _ = await pool._get_read_driver()
            assert d1 is mock_driver_1

            # Simulate loop change
            pool._read_bound_loop_id = -1

            d2, _ = await pool._get_read_driver()
            assert d2 is mock_driver_2
            mock_driver_1.close.assert_awaited_once()


# ── close_driver ──────────────────────────────────────────────────────


class TestCloseDriver:
    async def test_close_both_drivers(self):
        """close_driver() closes both write and read drivers."""
        write_drv = AsyncMock()
        read_drv = AsyncMock()

        pool._driver = write_drv
        pool._bound_loop_id = id(asyncio.get_running_loop())
        pool._read_driver = read_drv
        pool._read_bound_loop_id = id(asyncio.get_running_loop())

        await pool.close_driver()

        write_drv.close.assert_awaited_once()
        read_drv.close.assert_awaited_once()
        assert pool._driver is None
        assert pool._read_driver is None
        assert pool._read_database is None
        assert pool._read_bound_loop_id is None

    async def test_close_only_write_when_no_read(self):
        """close_driver() handles case where read pool was never created."""
        write_drv = AsyncMock()
        pool._driver = write_drv
        pool._bound_loop_id = id(asyncio.get_running_loop())

        await pool.close_driver()

        write_drv.close.assert_awaited_once()
        assert pool._driver is None
        assert pool._read_driver is None


# ── get_read_session ──────────────────────────────────────────────────


class TestGetReadSession:
    async def test_get_read_session_yields_session(self):
        """get_read_session() yields a usable session."""
        mock_session = AsyncMock()

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(
            pool, "_get_read_driver", return_value=(mock_driver, "memgraph")
        ):
            async with pool.get_read_session() as session:
                assert session is mock_session


# ── Backend integration ───────────────────────────────────────────────


class TestBackendReadPoolIntegration:
    """Verify backends use get_read_session for reads, get_session for writes."""

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

    async def test_kv_get_by_id_uses_read_pool(self, kv_store):
        read_calls = []
        write_calls = []

        @asynccontextmanager
        async def mock_read():
            read_calls.append(1)
            session = AsyncMock()
            result = AsyncMock()
            result.single = AsyncMock(return_value=None)
            result.consume = AsyncMock()
            session.run = AsyncMock(return_value=result)
            yield session

        @asynccontextmanager
        async def mock_write():
            write_calls.append(1)
            yield AsyncMock()

        with (
            patch.object(pool, "get_read_session", mock_read),
            patch.object(pool, "get_session", mock_write),
        ):
            await kv_store.get_by_id("k1")

        assert len(read_calls) == 1
        assert len(write_calls) == 0

    async def test_kv_upsert_uses_write_pool(self, kv_store):
        read_calls = []
        write_calls = []

        @asynccontextmanager
        async def mock_read():
            read_calls.append(1)
            yield AsyncMock()

        @asynccontextmanager
        async def mock_slot():
            yield

        with (
            patch.object(pool, "get_read_session", mock_read),
            patch.object(pool, "get_session", _mock_session()),
            patch.object(pool, "acquire_write_slot", mock_slot),
        ):
            await kv_store.upsert({"k1": {"v": 1}})

        assert len(read_calls) == 0

    async def test_docstatus_get_docs_paginated_uses_read_pool(self, doc_store):
        """The 502-causing endpoint must use the read pool (parallel count + fetch)."""
        read_calls = []

        @asynccontextmanager
        async def mock_read():
            read_calls.append(1)
            session = AsyncMock()
            # Session handles either count (single()) or page (iter) — support both
            count_result = AsyncMock()
            count_result.single = AsyncMock(return_value={"total": 0})
            count_result.consume = AsyncMock()
            page_result = _AsyncEmpty()
            page_result.consume = AsyncMock()
            # First run() call returns count, subsequent returns page.
            # Either session gets exactly one run() since count and fetch are parallel.
            session.run = AsyncMock(side_effect=[count_result, page_result])
            yield session

        with patch.object(pool, "get_read_session", mock_read):
            await doc_store.get_docs_paginated()

        # Count and fetch run in parallel → 2 separate read sessions.
        assert len(read_calls) == 2

    async def test_vector_query_uses_read_pool(self, vec_store):
        """Vector search (hot path) must use the read pool."""
        read_calls = []

        @asynccontextmanager
        async def mock_read():
            read_calls.append(1)
            session = AsyncMock()
            result = _AsyncEmpty()
            result.consume = AsyncMock()
            session.run = AsyncMock(return_value=result)
            yield session

        mock_embed = AsyncMock()
        mock_embed.func = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        vec_store.embedding_func = mock_embed

        with patch.object(pool, "get_read_session", mock_read):
            await vec_store.query("test query", top_k=5)

        assert len(read_calls) == 1
