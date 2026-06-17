"""Tests that verify result.consume() is called on every write path
and that drop() propagates exceptions instead of swallowing them.

These are regression tests for the bulk indexation silent failure bugs.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph import _pool as pool

# ── Shared helpers ────────────────────────────────────────────────────


class ConsumeTracker:
    """Tracks how many times result.consume() is called."""

    def __init__(self):
        self.count = 0

    def make_session_factory(self):
        tracker = self

        @asynccontextmanager
        async def mock_session():
            session = AsyncMock()

            async def tracking_run(query, **params):
                result = AsyncMock()
                original = result.consume

                async def counting_consume():
                    tracker.count += 1
                    return await original()

                result.consume = counting_consume
                result.single = AsyncMock(return_value=None)
                return result

            session.run = tracking_run
            yield session

        return mock_session


def _failing_session_factory(error_cls=RuntimeError, msg="Bolt connection lost"):
    """Session factory where session.run() raises on call."""

    @asynccontextmanager
    async def failing_session():
        session = AsyncMock()
        session.run.side_effect = error_cls(msg)
        yield session

    return failing_session


def _fail_on_query_session_factory(needle, error_cls=RuntimeError, msg="boom"):
    """Session factory whose run() succeeds until a query contains *needle*.

    Lets us simulate a drop where the DETACH DELETE succeeds but the
    follow-up DROP VECTOR INDEX fails — the path a blanket
    ``except Exception: pass`` used to bury under a false success.
    """

    @asynccontextmanager
    async def session_cm():
        session = AsyncMock()

        async def run(query, **params):
            if needle in query:
                raise error_cls(msg)
            result = AsyncMock()
            result.consume = AsyncMock()
            result.single = AsyncMock(return_value=None)
            return result

        session.run = run
        yield session

    return session_cm


@asynccontextmanager
async def _noop_write_slot():
    yield


def _mock_driver():
    async def get_driver():
        return MagicMock(), "memgraph"

    return get_driver


# ── KV Storage ────────────────────────────────────────────────────────


class TestKVConsumeOnWrite:
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

    async def test_upsert_consumes_result(self, kv_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            await kv_store.upsert({"doc1": {"text": "hello"}})
        assert tracker.count == 1, f"Expected 1 consume(), got {tracker.count}"

    async def test_initialize_consumes_result(self, kv_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "get_driver", _mock_driver()),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            await kv_store.initialize()
        assert tracker.count == 1, f"Expected 1 consume() for CREATE INDEX, got {tracker.count}"

    async def test_drop_propagates_exception(self, kv_store):
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", _failing_session_factory()),
        ):
            with pytest.raises(RuntimeError, match="Bolt connection lost"):
                await kv_store.drop()

    async def test_drop_consumes_result(self, kv_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            result = await kv_store.drop()
        assert result["status"] == "success"
        assert tracker.count == 1


# ── Vector Storage ────────────────────────────────────────────────────


class TestVectorConsumeOnWrite:
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

    async def test_initialize_consumes_both_indexes(self, vec_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "get_driver", _mock_driver()),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            await vec_store.initialize()
        # 1 label index + 1 vector index = 2 consume() calls
        assert tracker.count == 2, f"Expected 2 consume() calls, got {tracker.count}"

    async def test_drop_propagates_exception(self, vec_store):
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", _failing_session_factory()),
        ):
            with pytest.raises(RuntimeError, match="Bolt connection lost"):
                await vec_store.drop()

    async def test_drop_consumes_result(self, vec_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            result = await vec_store.drop()
        assert result["status"] == "success"
        # 1 DETACH DELETE + 1 DROP VECTOR INDEX = 2 consume() calls
        assert tracker.count == 2

    async def test_drop_propagates_drop_index_failure(self, vec_store):
        """A non-trivial DROP VECTOR INDEX failure must NOT be reported as success.

        Regression for the silent ``except Exception: pass`` that swallowed
        every error from DROP VECTOR INDEX. The DETACH DELETE succeeds; only the
        index drop fails (e.g. connection reset). The old code returned
        ``{"status": "success"}`` anyway — burying a half-completed drop. The
        error must now propagate.
        """
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(
                pool,
                "get_session",
                _fail_on_query_session_factory(
                    "DROP VECTOR INDEX", msg="Connection reset by peer"
                ),
            ),
        ):
            with pytest.raises(RuntimeError, match="Connection reset by peer"):
                await vec_store.drop()

    async def test_drop_swallows_missing_index(self, vec_store):
        """Idempotency preserved: a 'does not exist' index is not an error.

        The DETACH DELETE succeeds; DROP VECTOR INDEX reports the index is
        already gone. drop() must still return success — the legitimate
        idempotent case the narrowed handler is allowed to swallow.
        """
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(
                pool,
                "get_session",
                _fail_on_query_session_factory(
                    "DROP VECTOR INDEX",
                    msg="Index vec_test_entities does not exist",
                ),
            ),
        ):
            result = await vec_store.drop()
        assert result["status"] == "success"


# ── DocStatus Storage ─────────────────────────────────────────────────


class TestDocStatusConsumeOnWrite:
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

    async def test_initialize_consumes_all_index_results(self, ds_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "get_driver", _mock_driver()),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            await ds_store.initialize()
        # 7 CREATE INDEX calls: id, status, file_path, track_id, updated_at,
        # created_at, content_hash.
        assert tracker.count == 7, f"Expected 7 consume() calls, got {tracker.count}"

    async def test_drop_propagates_exception(self, ds_store):
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", _failing_session_factory()),
        ):
            with pytest.raises(RuntimeError, match="Bolt connection lost"):
                await ds_store.drop()

    async def test_drop_consumes_result(self, ds_store):
        tracker = ConsumeTracker()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", tracker.make_session_factory()),
        ):
            result = await ds_store.drop()
        assert result["status"] == "success"
        assert tracker.count == 1


# ── USE DATABASE consume ──────────────────────────────────────────────


class TestUseDatabaseConsume:
    """Verify that USE DATABASE results are consumed in _pool.py."""

    async def test_try_use_database_consumes_result(self):
        """_try_use_database must consume the USE DATABASE result."""
        from twindb_lightrag_memgraph._pool import _try_use_database

        session = AsyncMock()
        consume_called = False

        async def tracking_run(query):
            result = AsyncMock()

            async def mark_consume():
                nonlocal consume_called
                consume_called = True

            result.consume = mark_consume
            return result

        session.run = tracking_run

        # Reset enterprise detection so it actually runs
        import twindb_lightrag_memgraph._pool as pool_mod

        old_val = pool_mod._enterprise_supported
        pool_mod._enterprise_supported = None
        try:
            await _try_use_database(session, "custom_db")
        finally:
            pool_mod._enterprise_supported = old_val

        assert consume_called, "result.consume() was not called in _try_use_database"
