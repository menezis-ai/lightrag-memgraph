"""
Tests that delete() methods convert set inputs to list before passing
to the Bolt driver, which does not support Python sets as query parameters.

OFFLINE — no Memgraph needed. All driver calls are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_session():
    """Build a mock async session whose run() records call kwargs."""
    mock_result = AsyncMock()
    mock_result.consume = AsyncMock()

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def _patch_pool(mock_session):
    """Return a context manager that patches _pool.get_session."""
    return patch(
        "twindb_lightrag_memgraph._pool.get_session",
        return_value=mock_session,
    )


def _make_kv_storage():
    from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

    return MemgraphKVStorage(
        namespace="chunks",
        global_config={},
        embedding_func=MagicMock(),
    )


def _make_vector_storage():
    from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

    mock_embed = MagicMock()
    mock_embed.embedding_dim = 128
    return MemgraphVectorDBStorage(
        namespace="entities",
        global_config={},
        embedding_func=mock_embed,
    )


def _make_docstatus_storage():
    from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

    return MemgraphDocStatusStorage(
        namespace="docstatus",
        global_config={},
        embedding_func=MagicMock(),
    )


class TestKVDeleteSetCoercion:
    async def test_delete_accepts_set(self):
        """KV delete() must not raise when called with a set."""
        session = _make_mock_session()
        storage = _make_kv_storage()

        with _patch_pool(session):
            await storage.delete({"id-1", "id-2", "id-3"})

        # Verify the ids parameter is a list (not a set)
        call_kwargs = session.run.call_args
        ids_param = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert isinstance(
            ids_param, list
        ), f"Expected list, got {type(ids_param).__name__}"
        assert sorted(ids_param) == ["id-1", "id-2", "id-3"]

    async def test_delete_still_works_with_list(self):
        """KV delete() must continue to work with a regular list."""
        session = _make_mock_session()
        storage = _make_kv_storage()

        with _patch_pool(session):
            await storage.delete(["id-a", "id-b"])

        call_kwargs = session.run.call_args
        ids_param = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert isinstance(ids_param, list)


class TestVectorDeleteSetCoercion:
    async def test_delete_accepts_set(self):
        """Vector delete() must not raise when called with a set."""
        session = _make_mock_session()
        storage = _make_vector_storage()

        with _patch_pool(session):
            await storage.delete({"vec-1", "vec-2"})

        call_kwargs = session.run.call_args
        ids_param = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert isinstance(
            ids_param, list
        ), f"Expected list, got {type(ids_param).__name__}"
        assert sorted(ids_param) == ["vec-1", "vec-2"]

    async def test_delete_still_works_with_list(self):
        """Vector delete() must continue to work with a regular list."""
        session = _make_mock_session()
        storage = _make_vector_storage()

        with _patch_pool(session):
            await storage.delete(["vec-a"])

        call_kwargs = session.run.call_args
        ids_param = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert isinstance(ids_param, list)


class TestDocStatusDeleteSetCoercion:
    async def test_delete_accepts_set(self):
        """DocStatus delete() must not raise when called with a set."""
        session = _make_mock_session()
        storage = _make_docstatus_storage()

        with _patch_pool(session):
            await storage.delete({"doc-1", "doc-2"})

        call_kwargs = session.run.call_args
        ids_param = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert isinstance(
            ids_param, list
        ), f"Expected list, got {type(ids_param).__name__}"
        assert sorted(ids_param) == ["doc-1", "doc-2"]

    async def test_delete_still_works_with_list(self):
        """DocStatus delete() must continue to work with a regular list."""
        session = _make_mock_session()
        storage = _make_docstatus_storage()

        with _patch_pool(session):
            await storage.delete(["doc-a"])

        call_kwargs = session.run.call_args
        ids_param = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert isinstance(ids_param, list)
