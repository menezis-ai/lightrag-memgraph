"""Unit tests for batched delete operations in _batched_ops.py.

No Memgraph required — all DB interactions are mocked.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph._batched_ops import (
    _resolve_batch_size,
    batched_delete,
    batched_delete_by_ids,
)
from twindb_lightrag_memgraph._constants import DEFAULT_DELETE_BATCH_SIZE

# ── Helpers ────────────────────────────────────────────────────────────


@asynccontextmanager
async def _noop_write_slot():
    yield


def _mock_session_factory(deleted_sequence: list[int]):
    """Return a session factory where successive session.run() calls
    return the given sequence of ``deleted`` counts.

    Each call to the returned context manager creates a fresh session
    whose ``run()`` returns the next value in *deleted_sequence*.
    """
    call_idx = {"i": 0}

    @asynccontextmanager
    async def mock_session():
        session = AsyncMock()

        async def mock_run(query, **params):
            result = AsyncMock()
            idx = call_idx["i"]
            call_idx["i"] += 1
            count = deleted_sequence[idx] if idx < len(deleted_sequence) else 0
            record = MagicMock()
            record.__getitem__ = lambda self, key: count if key == "deleted" else None
            result.single = AsyncMock(return_value=record)
            result.consume = AsyncMock()
            return result

        session.run = mock_run
        yield session

    return mock_session


# ── _resolve_batch_size ────────────────────────────────────────────────


class TestResolveBatchSize:
    def test_override_takes_precedence(self):
        assert _resolve_batch_size(override=42) == 42

    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_DELETE_BATCH_SIZE", raising=False)
        assert _resolve_batch_size() == DEFAULT_DELETE_BATCH_SIZE

    def test_env_var_batch_size(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_DELETE_BATCH_SIZE", "500")
        assert _resolve_batch_size() == 500

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_DELETE_BATCH_SIZE", "not_a_number")
        assert _resolve_batch_size() == DEFAULT_DELETE_BATCH_SIZE

    def test_zero_env_clamps_to_one(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_DELETE_BATCH_SIZE", "0")
        assert _resolve_batch_size() == 1


# ── batched_delete ─────────────────────────────────────────────────────


class TestBatchedDelete:
    async def test_batched_delete_single_batch(self):
        """When fewer nodes than batch_size, one iteration suffices."""
        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.get_session",
                _mock_session_factory([5]),
            ),
        ):
            total = await batched_delete("TestLabel", batch_size=100)
        assert total == 5

    async def test_batched_delete_multiple_batches(self):
        """Loop continues until a batch returns fewer than batch_size."""
        bs = 10
        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.get_session",
                _mock_session_factory([bs, bs, 3]),
            ),
        ):
            total = await batched_delete("TestLabel", batch_size=bs)
        assert total == bs + bs + 3

    async def test_batched_delete_empty_label(self):
        """Zero nodes deleted — total is 0 and no log emitted."""
        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.get_session",
                _mock_session_factory([0]),
            ),
            patch("twindb_lightrag_memgraph._batched_ops.logger") as mock_logger,
        ):
            total = await batched_delete("EmptyLabel", batch_size=100)
        assert total == 0
        mock_logger.info.assert_not_called()

    async def test_batched_delete_returns_total(self):
        """Verify count accumulation across batches."""
        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.get_session",
                _mock_session_factory([50, 50, 25]),
            ),
        ):
            total = await batched_delete("CountLabel", batch_size=50)
        assert total == 125


# ── batched_delete_by_ids ──────────────────────────────────────────────


class TestBatchedDeleteByIds:
    async def test_batched_delete_by_ids_splits(self):
        """12 IDs with batch_size=5 should produce 3 session.run calls."""
        ids = [f"id_{i}" for i in range(12)]
        run_calls = []

        @asynccontextmanager
        async def mock_session():
            session = AsyncMock()

            async def mock_run(query, **params):
                run_calls.append(params.get("ids", []))
                result = AsyncMock()
                result.consume = AsyncMock()
                return result

            session.run = mock_run
            yield session

        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.get_session",
                mock_session,
            ),
        ):
            total = await batched_delete_by_ids("TestLabel", ids, batch_size=5)

        assert total == 12
        assert len(run_calls) == 3
        assert len(run_calls[0]) == 5
        assert len(run_calls[1]) == 5
        assert len(run_calls[2]) == 2

    async def test_batched_delete_by_ids_empty(self):
        """Empty ID list produces zero calls."""
        run_calls = []

        @asynccontextmanager
        async def mock_session():
            session = AsyncMock()

            async def mock_run(query, **params):
                run_calls.append(1)
                result = AsyncMock()
                result.consume = AsyncMock()
                return result

            session.run = mock_run
            yield session

        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph._batched_ops._pool.get_session",
                mock_session,
            ),
        ):
            total = await batched_delete_by_ids("TestLabel", [])

        assert total == 0
        assert len(run_calls) == 0


# ── Backend integration ────────────────────────────────────────────────


class TestKVDropUsesBatched:
    """Verify kv_impl.drop() delegates to batched_delete."""

    @pytest.fixture
    def kv_store(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

        store = MemgraphKVStorage.__new__(MemgraphKVStorage)
        store.namespace = "chunks"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        return store

    async def test_kv_drop_uses_batched(self, kv_store):
        with patch(
            "twindb_lightrag_memgraph._batched_ops.batched_delete",
            new_callable=AsyncMock,
            return_value=42,
        ) as mock_bd:
            result = await kv_store.drop()

        mock_bd.assert_awaited_once_with("KV_test_chunks")
        assert result["status"] == "success"
        assert "42 nodes" in result["message"]


class TestVectorDropUsesBatched:
    """Verify vector_impl.drop() delegates to batched_delete."""

    @pytest.fixture
    def vec_store(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

        store = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
        store.namespace = "entities"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = MagicMock()
        store.embedding_func.embedding_dim = 384
        store.meta_fields = set()
        store.cosine_better_than_threshold = 0.2
        return store

    async def test_vector_drop_uses_batched(self, vec_store):
        with (
            patch(
                "twindb_lightrag_memgraph._batched_ops.batched_delete",
                new_callable=AsyncMock,
                return_value=10,
            ) as mock_bd,
            patch(
                "twindb_lightrag_memgraph.vector_impl._pool.acquire_write_slot",
                _noop_write_slot,
            ),
            patch(
                "twindb_lightrag_memgraph.vector_impl._pool.get_session",
                _mock_session_factory([0]),
            ),
        ):
            result = await vec_store.drop()

        mock_bd.assert_awaited_once_with("Vec_test_entities")
        assert result["status"] == "success"
        assert "10 nodes" in result["message"]


class TestDocStatusDropUsesBatched:
    """Verify docstatus_impl.drop() delegates to batched_delete."""

    @pytest.fixture
    def ds_store(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
        from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

        store = MemgraphDocStatusStorage.__new__(MemgraphDocStatusStorage)
        store.namespace = "doc_status"
        store.workspace = "test"
        store.global_config = {}
        store.embedding_func = None
        return store

    async def test_docstatus_drop_uses_batched(self, ds_store):
        with patch(
            "twindb_lightrag_memgraph._batched_ops.batched_delete",
            new_callable=AsyncMock,
            return_value=100,
        ) as mock_bd:
            result = await ds_store.drop()

        mock_bd.assert_awaited_once_with("DocStatus_test")
        assert result["status"] == "success"
        assert "100 nodes" in result["message"]
