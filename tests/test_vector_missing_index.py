"""Unit tests for MemgraphVectorDBStorage.query() with missing vector index.

Verifies that query() auto-creates the vector index when it is missing
(e.g. after a Memgraph restart) and retries the query. If auto-create
also fails, returns an empty list gracefully.
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


@asynccontextmanager
async def _noop_session():
    session = AsyncMock()
    result_mock = AsyncMock()
    result_mock.consume = AsyncMock()
    session.run.return_value = result_mock
    yield session


class TestQueryMissingIndex:
    async def test_auto_creates_index_and_retries(self):
        """query() should auto-create the vector index and retry when missing."""
        store = _make_store()

        # First call (read session): raises "does not exist"
        # After auto-create, second call (read session): returns results
        ok_result = AsyncMock()
        ok_result.__aiter__ = MagicMock(return_value=AsyncMock(__anext__=AsyncMock(side_effect=StopAsyncIteration)))
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

        with (
            patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool,
            patch(
                "twindb_lightrag_memgraph._retry.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_pool.get_read_session = MagicMock(return_value=read_ctx)
            mock_pool.get_session = MagicMock(return_value=_noop_session())

            results = await store.query(
                query="test",
                top_k=5,
                query_embedding=[0.1, 0.2, 0.3, 0.4],
            )

        assert results == []
        # read session.run called twice: first fails, second succeeds
        assert read_session.run.call_count == 2

    async def test_returns_empty_when_auto_create_fails(self):
        """query() should return [] when auto-create also fails."""
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

        # Auto-create also fails
        create_session = AsyncMock()
        create_session.run = AsyncMock(side_effect=Exception("MAGE not loaded"))

        @asynccontextmanager
        async def _failing_create():
            yield create_session

        with (
            patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool,
            patch(
                "twindb_lightrag_memgraph._retry.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_pool.get_read_session = MagicMock(return_value=read_ctx)
            mock_pool.get_session = MagicMock(return_value=_failing_create())

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


class TestEnsureVectorIndex:
    async def test_initialize_creates_vector_index(self):
        """initialize() should call _ensure_vector_index with retry."""
        store = _make_store()

        with (
            patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool,
            patch(
                "twindb_lightrag_memgraph._retry.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_pool.get_driver = AsyncMock(return_value=(MagicMock(), "memgraph"))
            mock_pool.get_session = MagicMock(return_value=_noop_session())

            await store.initialize()

    async def test_initialize_tolerates_already_exists(self):
        """initialize() should not raise if vector index already exists."""
        store = _make_store()

        @asynccontextmanager
        async def _exists_session():
            session = AsyncMock()
            session.run = AsyncMock(
                side_effect=Exception("Index already exists")
            )
            yield session

        with (
            patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool,
            patch(
                "twindb_lightrag_memgraph._retry.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_pool.get_driver = AsyncMock(return_value=(MagicMock(), "memgraph"))
            mock_pool.get_session = MagicMock(return_value=_exists_session())

            await store.initialize()
