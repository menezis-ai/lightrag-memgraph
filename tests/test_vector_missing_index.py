"""Unit tests for MemgraphVectorDBStorage.query() with missing vector index.

Verifies that query() auto-creates the vector index when it is missing
(e.g. after a Memgraph restart or before any data has been indexed) and
retries the query. Falls back to empty results if auto-create also fails.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


def _make_store():
    """Create a MemgraphVectorDBStorage with patched __init__."""
    with patch.object(MemgraphVectorDBStorage, "__init__", lambda self, **kw: None):
        store = MemgraphVectorDBStorage()
    store.workspace = "base"
    store.namespace = "relationships"
    store.meta_fields = {"src_id", "tgt_id"}
    store.cosine_better_than_threshold = 0.2
    store.embedding_func = MagicMock()
    store.embedding_func.embedding_dim = 1024
    return store


class TestQueryMissingIndex:
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

        with patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool:
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

    async def test_returns_empty_when_auto_create_fails(self):
        """query() returns [] when auto-create itself fails."""
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
        create_session.run = AsyncMock(side_effect=Exception("MAGE not loaded"))

        @asynccontextmanager
        async def _create_ctx():
            yield create_session

        with patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool:
            mock_pool.get_read_session = MagicMock(return_value=read_ctx)
            mock_pool.get_session = MagicMock(return_value=_create_ctx())

            results = await store.query(
                query="test",
                top_k=5,
                query_embedding=[0.1, 0.2, 0.3, 0.4],
            )

        assert results == []

    async def test_reraises_other_errors(self):
        """query() should re-raise errors that are NOT about missing index."""
        store = _make_store()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("Connection refused"))

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool:
            mock_pool.get_read_session = MagicMock(return_value=mock_ctx)

            with pytest.raises(Exception, match="Connection refused"):
                await store.query(
                    query="test",
                    top_k=5,
                    query_embedding=[0.1, 0.2, 0.3, 0.4],
                )
