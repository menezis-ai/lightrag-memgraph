"""Tests for _retry.py — TransientError retry with exponential backoff."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from neo4j.exceptions import TransientError

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph._retry import (
    _read_base_delay_ms,
    _read_max_attempts,
    retry_transient,
)


# ── Helper: build a callable that fails N times then succeeds ──────────


def _flaky(fail_count: int, return_value=None, exc_cls=TransientError):
    """Return an async callable that raises *exc_cls* for the first
    *fail_count* invocations, then returns *return_value*.
    """
    calls = 0

    async def _fn():
        nonlocal calls
        calls += 1
        if calls <= fail_count:
            raise exc_cls("Cannot resolve conflicting transactions")
        return return_value

    _fn.calls = lambda: calls  # type: ignore[attr-defined]
    return _fn


# ── Mock helpers for pool session ──────────────────────────────────────


@asynccontextmanager
async def _noop_write_slot():
    yield


def _make_session_factory(run_side_effect=None):
    """Return an async context manager yielding a mock session.

    *run_side_effect* is passed to ``session.run.side_effect``.
    Each ``session.run()`` returns an ``AsyncMock`` with a ``.consume()``
    and ``.single()`` that also return ``AsyncMock``.
    """

    @asynccontextmanager
    async def _factory():
        session = AsyncMock()
        if run_side_effect is not None:
            session.run.side_effect = run_side_effect
        else:
            result_mock = AsyncMock()
            result_mock.consume = AsyncMock()
            result_mock.single = AsyncMock(return_value=None)
            session.run.return_value = result_mock
        yield session

    return _factory


def _transient_then_ok():
    """Side-effect list: TransientError on 1st call, success on 2nd."""
    ok_result = AsyncMock()
    ok_result.consume = AsyncMock()
    ok_result.single = AsyncMock(return_value=None)
    return [
        TransientError("Cannot resolve conflicting transactions"),
        ok_result,
    ]


# ═══════════════════════════════════════════════════════════════════════
# Core retry_transient tests
# ═══════════════════════════════════════════════════════════════════════


class TestRetryTransient:
    async def test_succeeds_first_attempt(self):
        fn = _flaky(0, return_value="ok")
        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_transient(fn)
        assert result == "ok"
        assert fn.calls() == 1

    async def test_retries_then_succeeds(self):
        fn = _flaky(2, return_value=42)
        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await retry_transient(fn)
        assert result == 42
        assert fn.calls() == 3
        assert mock_sleep.await_count == 2

    async def test_exhausts_all_attempts(self):
        fn = _flaky(10, return_value="never")
        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TransientError, match="conflicting"):
                await retry_transient(fn, max_attempts=3)
        assert fn.calls() == 3

    async def test_non_transient_error_propagates_immediately(self):
        async def _boom():
            raise RuntimeError("not transient")

        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RuntimeError, match="not transient"):
                await retry_transient(_boom)
        mock_sleep.assert_not_awaited()

    async def test_return_value_forwarded(self):
        async def _fn():
            return {"count": 7}

        result = await retry_transient(_fn)
        assert result == {"count": 7}

    async def test_override_max_attempts(self):
        fn = _flaky(1, return_value="ok")
        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_transient(fn, max_attempts=2)
        assert result == "ok"
        assert fn.calls() == 2

    async def test_override_max_attempts_too_low(self):
        fn = _flaky(2, return_value="never")
        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TransientError):
                await retry_transient(fn, max_attempts=2)
        assert fn.calls() == 2

    async def test_backoff_sleep_called_with_increasing_values(self):
        fn = _flaky(3, return_value="ok")
        sleeps = []

        async def _capture_sleep(seconds):
            sleeps.append(seconds)

        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", side_effect=_capture_sleep):
            with patch("twindb_lightrag_memgraph._retry.random.uniform", side_effect=lambda a, b: b):
                await retry_transient(fn, base_delay_ms=100)

        # With jitter = max (b), delays are 100ms, 200ms, 400ms
        assert len(sleeps) == 3
        assert sleeps[0] == pytest.approx(0.1)
        assert sleeps[1] == pytest.approx(0.2)
        assert sleeps[2] == pytest.approx(0.4)

    async def test_max_delay_cap(self):
        fn = _flaky(5, return_value="ok")
        sleeps = []

        async def _capture_sleep(seconds):
            sleeps.append(seconds)

        with patch("twindb_lightrag_memgraph._retry.asyncio.sleep", side_effect=_capture_sleep):
            with patch("twindb_lightrag_memgraph._retry.random.uniform", side_effect=lambda a, b: b):
                await retry_transient(fn, base_delay_ms=1000)

        # 1000, 2000, 2000(cap), 2000(cap), 2000(cap)
        assert all(s <= 2.0 for s in sleeps)


# ═══════════════════════════════════════════════════════════════════════
# Config reader tests
# ═══════════════════════════════════════════════════════════════════════


class TestConfigReaders:
    def test_max_attempts_default(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_RETRY_MAX_ATTEMPTS", raising=False)
        assert _read_max_attempts() == 6

    def test_max_attempts_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_RETRY_MAX_ATTEMPTS", "10")
        assert _read_max_attempts() == 10

    def test_max_attempts_invalid(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_RETRY_MAX_ATTEMPTS", "abc")
        assert _read_max_attempts() == 6

    def test_max_attempts_zero_clamps(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_RETRY_MAX_ATTEMPTS", "0")
        assert _read_max_attempts() == 1

    def test_base_delay_default(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_RETRY_BASE_DELAY_MS", raising=False)
        assert _read_base_delay_ms() == 50

    def test_base_delay_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_RETRY_BASE_DELAY_MS", "200")
        assert _read_base_delay_ms() == 200

    def test_base_delay_invalid(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_RETRY_BASE_DELAY_MS", "nope")
        assert _read_base_delay_ms() == 50

    def test_base_delay_zero_clamps(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_RETRY_BASE_DELAY_MS", "0")
        assert _read_base_delay_ms() == 1


# ═══════════════════════════════════════════════════════════════════════
# Wiring tests — verify retry is plugged into each backend
# ═══════════════════════════════════════════════════════════════════════


class TestKVRetryWiring:
    @pytest.fixture
    def kv_store(self, monkeypatch):
        from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        store = MemgraphKVStorage.__new__(MemgraphKVStorage)
        store.namespace = "chunks"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        return store

    async def test_upsert_retries_on_transient(self, kv_store):
        with (
            patch.object(_pool, "acquire_write_slot", _noop_write_slot),
            patch.object(_pool, "get_session", _make_session_factory(_transient_then_ok())),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            await kv_store.upsert({"doc1": {"text": "hello"}})

    async def test_delete_retries_on_transient(self, kv_store):
        with (
            patch.object(_pool, "acquire_write_slot", _noop_write_slot),
            patch.object(_pool, "get_session", _make_session_factory(_transient_then_ok())),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            await kv_store.delete(["doc1"])


class TestVectorRetryWiring:
    @pytest.fixture
    def vec_store(self, monkeypatch):
        from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        store = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
        store.namespace = "entities"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        store.meta_fields = set()
        store.cosine_better_than_threshold = 0.2
        return store

    async def test_upsert_retries_on_transient(self, vec_store):
        with (
            patch.object(_pool, "acquire_write_slot", _noop_write_slot),
            patch.object(_pool, "get_session", _make_session_factory(_transient_then_ok())),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            await vec_store.upsert(
                {"e1": {"content": "test", "embedding": [0.1, 0.2]}}
            )


class TestDocStatusRetryWiring:
    @pytest.fixture
    def ds_store(self, monkeypatch):
        from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        store = MemgraphDocStatusStorage.__new__(MemgraphDocStatusStorage)
        store.namespace = "doc_status"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        return store

    async def test_upsert_retries_on_transient(self, ds_store):
        with (
            patch.object(_pool, "acquire_write_slot", _noop_write_slot),
            patch.object(_pool, "get_session", _make_session_factory(_transient_then_ok())),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            await ds_store.upsert({"doc1": {"status": "pending"}})


class TestBufferedGraphRetryWiring:
    async def test_flush_nodes_retries_on_transient(self):
        from twindb_lightrag_memgraph._buffered_graph import _BufferedGraphProxy

        real = AsyncMock()
        real.workspace = "test"
        proxy = _BufferedGraphProxy(real)
        await proxy.upsert_node("entity1", {"name": "Entity 1"})

        with (
            patch(
                "twindb_lightrag_memgraph._buffered_graph.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._buffered_graph.get_session",
                _make_session_factory(_transient_then_ok()),
            ),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            await proxy._flush_nodes()

    async def test_flush_edges_retries_on_transient(self):
        from twindb_lightrag_memgraph._buffered_graph import _BufferedGraphProxy

        real = AsyncMock()
        real.workspace = "test"
        proxy = _BufferedGraphProxy(real)
        proxy._node_buffer = {"A": {"name": "A"}, "B": {"name": "B"}}
        await proxy.upsert_edge("A", "B", {"weight": "1"})

        with (
            patch(
                "twindb_lightrag_memgraph._buffered_graph.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._buffered_graph.get_session",
                _make_session_factory(_transient_then_ok()),
            ),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            await proxy._flush_edges()


class TestBatchedDeleteRetryWiring:
    async def test_batched_delete_retries_on_transient(self):
        from twindb_lightrag_memgraph._batched_ops import batched_delete

        # First call: TransientError, second call: returns record with deleted=0
        ok_result = AsyncMock()
        ok_result.single = AsyncMock(return_value={"deleted": 0})
        ok_result.consume = AsyncMock()

        with (
            patch.object(_pool, "acquire_write_slot", _noop_write_slot),
            patch.object(
                _pool,
                "get_session",
                _make_session_factory([
                    TransientError("Cannot resolve conflicting transactions"),
                    ok_result,
                ]),
            ),
            patch("twindb_lightrag_memgraph._retry.asyncio.sleep", new_callable=AsyncMock),
        ):
            total = await batched_delete("TestLabel", batch_size=100)
        assert total == 0
