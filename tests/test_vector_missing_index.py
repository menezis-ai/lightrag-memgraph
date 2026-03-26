"""Unit tests for MemgraphVectorDBStorage.query() with missing vector index.

Verifies that query() returns an empty list (instead of crashing) when the
vector index does not exist yet — e.g. after a Memgraph restart or before
any data has been indexed for that namespace.
"""

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
    return store


class TestQueryMissingIndex:
    async def test_returns_empty_on_missing_index(self):
        """query() should return [] when vector index does not exist."""
        store = _make_store()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(
            side_effect=Exception(
                "vector_search.search: Vector index vec_base_relationships "
                "does not exist."
            )
        )

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("twindb_lightrag_memgraph.vector_impl._pool") as mock_pool:
            mock_pool.get_read_session = MagicMock(return_value=mock_ctx)

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
