"""Unit tests for MemgraphVectorDBStorage.query() with missing vector index.

Verifies that query() auto-creates the vector index when it is missing
(e.g. after a Memgraph restart or before any data has been indexed) and
retries the query. Recovery failures propagate to the caller.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from tests.conftest import initialized_mock
from lightrag.utils import EmbeddingFunc
from neo4j.exceptions import AuthError, ClientError, Forbidden, ServiceUnavailable

from lightrag_memgraph import _pool
from lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


def _make_store():
    """Create a MemgraphVectorDBStorage with patched __init__."""
    with patch.object(MemgraphVectorDBStorage, "__init__", lambda self, **kw: None):
        store = MemgraphVectorDBStorage()
    store.workspace = "base"
    store.namespace = "relationships"
    store.meta_fields = {"src_id", "tgt_id"}
    store.cosine_better_than_threshold = 0.2
    store.embedding_func = EmbeddingFunc(embedding_dim=4, func=AsyncMock())
    store._vector_config = {"capacity": 100_000}
    store.capabilities = None
    return initialized_mock(store)


class TestQueryMissingIndex:
    async def test_missing_index_retry_reuses_query_embedding(self):
        store = _make_store()
        embedding = [0.1, 0.2, 0.3, 0.4]
        store.embedding_func.func = AsyncMock(return_value=np.array([embedding]))
        store._query_vector_once = AsyncMock(
            side_effect=[Exception("Vector index does not exist"), []]
        )
        store._create_vector_index = AsyncMock()

        assert await store.query("test", 5) == []
        store.embedding_func.func.assert_awaited_once_with(["test"], _priority=5)
        assert store._query_vector_once.await_count == 2
        assert all(
            call.args[2] == embedding
            for call in store._query_vector_once.await_args_list
        )

    async def test_auto_creates_and_retries(self):
        """query() should auto-create the index and retry the search."""
        store = _make_store()

        # Retry yields an empty iterator (no results but successful query)
        ok_result = AsyncMock()
        ok_result.__aiter__ = MagicMock(
            return_value=AsyncMock(
                __anext__=AsyncMock(side_effect=StopAsyncIteration),
            ),
        )
        ok_result.consume = AsyncMock()

        read_session = AsyncMock()
        read_session.run = AsyncMock(
            side_effect=[
                ClientError(
                    "vector_search.search: Vector index vec_base_relationships "
                    "does not exist."
                ),
                ok_result,
            ]
        )
        read_ctx = AsyncMock()
        read_ctx.__aenter__ = AsyncMock(return_value=read_session)
        read_ctx.__aexit__ = AsyncMock(return_value=False)

        # Write session for index creation
        create_session = AsyncMock()
        create_result = AsyncMock()
        create_result.consume = AsyncMock()
        create_session.run = AsyncMock(return_value=create_result)

        @asynccontextmanager
        async def _create_ctx():
            yield create_session

        with (
            patch.object(_pool, "get_read_session", return_value=read_ctx),
            patch.object(_pool, "get_session", _create_ctx),
        ):
            results = await store.query(
                query="test",
                top_k=5,
                query_embedding=[0.1, 0.2, 0.3, 0.4],
            )

        assert results == []
        assert read_session.run.call_count == 2  # initial + retry
        assert create_session.run.call_count == 1  # CREATE VECTOR INDEX

    @pytest.mark.parametrize(
        "failure",
        [
            Forbidden("CREATE INDEX denied"),
            AuthError("authentication failed"),
            ServiceUnavailable("connection lost"),
            ClientError("MAGE not loaded"),
        ],
    )
    @pytest.mark.parametrize("deferred", [False, True])
    async def test_recovery_failure_propagates(self, failure, deferred, caplog):
        """Missing data and a failed recovery must stay distinguishable."""
        store = _make_store()

        read_session = AsyncMock()
        read_session.run = AsyncMock(
            side_effect=ClientError(
                "vector_search.search: Vector index vec_base_relationships "
                "does not exist."
            )
        )
        read_ctx = AsyncMock()
        read_ctx.__aenter__ = AsyncMock(return_value=read_session)
        read_ctx.__aexit__ = AsyncMock(return_value=False)

        create_session = AsyncMock()
        if deferred:
            create_session.run.return_value.consume.side_effect = failure
        else:
            create_session.run.side_effect = failure

        @asynccontextmanager
        async def _create_ctx():
            yield create_session

        with (
            patch.object(_pool, "get_read_session", return_value=read_ctx),
            patch.object(_pool, "get_session", _create_ctx),
        ):
            with pytest.raises(
                RuntimeError,
                match="Vector index LRMG2_VECTOR_INDEX_w4_base_n13_relationships recovery failed",
            ) as caught:
                await store.query(
                    query="test",
                    top_k=5,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                )
        assert caught.value.__cause__ is failure
        assert not any(record.exc_info for record in caplog.records)
        assert read_session.run.await_count == 1
        assert create_session.run.await_count == 1

    @pytest.mark.parametrize("mode", ["vector", "hybrid"])
    @pytest.mark.parametrize("failure_stage", ["create", "retry"])
    async def test_recovery_errors_reach_vector_and_hybrid_callers(
        self, mode, failure_stage
    ):
        store = _make_store()
        store._query_mode = mode
        store._text_index_ready = True
        failure = RuntimeError("database unavailable during recovery")
        store._query_vector_once = AsyncMock(
            side_effect=[Exception("Vector index does not exist"), failure]
        )
        store._create_vector_index = AsyncMock(
            side_effect=failure if failure_stage == "create" else None
        )
        # A lexical hit must not turn a failed vector search into apparent success.
        store.query_text = AsyncMock(return_value=[{"id": "lexical-only"}])

        with pytest.raises(
            RuntimeError,
            match="Vector index LRMG2_VECTOR_INDEX_w4_base_n13_relationships recovery failed",
        ) as caught:
            await store.query("test", 5, query_embedding=[1, 0, 0, 0])
        assert caught.value.__cause__ is failure
        store._create_vector_index.assert_awaited_once()
        assert store._query_vector_once.await_count == (
            1 if failure_stage == "create" else 2
        )

    async def test_reraises_other_errors(self):
        """query() should re-raise errors that are NOT about missing index."""
        store = _make_store()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("Connection refused"))

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_pool, "get_read_session", return_value=mock_ctx):
            with pytest.raises(Exception, match="Connection refused"):
                await store.query(
                    query="test",
                    top_k=5,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                )


@pytest.mark.parametrize("failure_side", ["vector", "text"])
async def test_hybrid_failure_cancels_and_drains_the_other_search(failure_side):
    store = _make_store()
    store._query_mode = "hybrid"
    store._text_index_ready = True
    started, closed = asyncio.Event(), asyncio.Event()
    pending = []
    failure = ServiceUnavailable("backend unavailable")

    async def blocked(*args):
        pending.append(asyncio.current_task())
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def fail(*args):
        await started.wait()
        raise failure

    store._query_vector = fail if failure_side == "vector" else blocked
    store.query_text = fail if failure_side == "text" else blocked
    try:
        with pytest.raises(ServiceUnavailable) as caught:
            await store.query("query", 5, query_embedding=[1, 0, 0, 0])
        assert caught.value is failure
        assert closed.is_set(), "The sibling search still holds its resources"
        assert pending[0].done()
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def test_hybrid_caller_cancellation_drains_both_searches():
    store = _make_store()
    store._query_mode = "hybrid"
    store._text_index_ready = True
    started = [asyncio.Event(), asyncio.Event()]
    closed = [asyncio.Event(), asyncio.Event()]

    async def blocked(index, *args):
        try:
            started[index].set()
            await asyncio.Event().wait()
        finally:
            closed[index].set()

    store._query_vector = lambda *args: blocked(0, *args)
    store.query_text = lambda *args: blocked(1, *args)
    task = asyncio.create_task(store.query("query", 5, query_embedding=[1, 0, 0, 0]))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert all(event.is_set() for event in closed)
