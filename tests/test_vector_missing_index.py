"""Unit tests for MemgraphVectorDBStorage.query() with missing vector index.

Verifies that query() auto-creates the vector index when it is missing
(e.g. after a Memgraph restart or before any data has been indexed) and
retries the query. Recovery failures propagate to the caller.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np
from lightrag.utils import EmbeddingFunc

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
    return store


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
        store.embedding_func.func.assert_awaited_once_with(["test"])
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
                Exception(
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

        with patch("lightrag_memgraph.vector_impl._pool") as mock_pool:
            mock_pool.get_read_session = MagicMock(return_value=read_ctx)
            mock_pool.get_session = MagicMock(return_value=_create_ctx())

            results = await store.query(
                query="test",
                top_k=5,
                query_embedding=[0.1, 0.2, 0.3, 0.4],
            )

        assert results == []
        assert read_session.run.call_count == 2  # initial + retry
        assert create_session.run.call_count == 1  # CREATE VECTOR INDEX

    async def test_recovery_failure_propagates(self):
        """Missing data and a failed recovery must stay distinguishable."""
        store = _make_store()

        read_session = AsyncMock()
        read_session.run = AsyncMock(
            side_effect=Exception(
                "vector_search.search: Vector index vec_base_relationships "
                "does not exist."
            )
        )
        read_ctx = AsyncMock()
        read_ctx.__aenter__ = AsyncMock(return_value=read_session)
        read_ctx.__aexit__ = AsyncMock(return_value=False)

        create_session = AsyncMock()
        failure = PermissionError("CREATE INDEX denied")
        create_session.run = AsyncMock(side_effect=failure)

        @asynccontextmanager
        async def _create_ctx():
            yield create_session

        with patch("lightrag_memgraph.vector_impl._pool") as mock_pool:
            mock_pool.get_read_session = MagicMock(return_value=read_ctx)
            mock_pool.get_session = MagicMock(return_value=_create_ctx())

            with pytest.raises(PermissionError, match="CREATE INDEX denied") as caught:
                await store.query(
                    query="test",
                    top_k=5,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                )
        assert caught.value is failure
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

        with pytest.raises(RuntimeError) as caught:
            await store.query("test", 5, query_embedding=[1, 0, 0, 0])
        assert caught.value is failure
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

        with patch("lightrag_memgraph.vector_impl._pool") as mock_pool:
            mock_pool.get_read_session = MagicMock(return_value=mock_ctx)

            with pytest.raises(Exception, match="Connection refused"):
                await store.query(
                    query="test",
                    top_k=5,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                )
