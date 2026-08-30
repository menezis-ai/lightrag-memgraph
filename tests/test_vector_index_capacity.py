"""``TWIN_VECTOR_INDEX_CAPACITY`` reaches the ``CREATE VECTOR INDEX`` DDL.

The capacity used to be the hard-coded ``VECTOR_INDEX_CAPACITY`` constant
(audit 2026-08-25). It is now read from the environment at index-creation
time — so the DDL must carry the configured value, and the default must stay
the historical 100 000 so existing deployments create identical indexes.

OFFLINE — the write session is mocked; only the emitted Cypher is checked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


def _make_store():
    with patch.object(MemgraphVectorDBStorage, "__init__", lambda self, **kw: None):
        store = MemgraphVectorDBStorage()
    store.workspace = "base"
    store.namespace = "chunks"
    store.meta_fields = set()
    store.cosine_better_than_threshold = 0.2
    store.embedding_func = MagicMock()
    store.embedding_func.embedding_dim = 1024
    return store


def _capture_session():
    session = AsyncMock()
    result = AsyncMock()
    result.consume = AsyncMock()
    session.run = AsyncMock(return_value=result)
    return session


def _create_ddl(session) -> str:
    calls = [c.args[0] for c in session.run.call_args_list]
    ddl = [q for q in calls if "CREATE VECTOR INDEX" in q]
    assert len(ddl) == 1, calls
    return ddl[0]


async def test_default_capacity_is_the_historical_100000(monkeypatch):
    monkeypatch.delenv("TWIN_VECTOR_INDEX_CAPACITY", raising=False)
    session = _capture_session()

    await _make_store()._create_vector_index(session=session)

    ddl = _create_ddl(session)
    assert '"capacity": 100000' in ddl
    assert '"dimension": 1024' in ddl


async def test_configured_capacity_reaches_the_ddl(monkeypatch):
    monkeypatch.setenv("TWIN_VECTOR_INDEX_CAPACITY", "250000")
    session = _capture_session()

    await _make_store()._create_vector_index(session=session)

    assert '"capacity": 250000' in _create_ddl(session)


async def test_malformed_capacity_never_reaches_memgraph(monkeypatch):
    """A bad value is a config error (register() refuses to boot on it); if
    it is only set after boot, index creation must fail loudly rather than
    emit a DDL with a wrong or missing capacity."""
    monkeypatch.setenv("TWIN_VECTOR_INDEX_CAPACITY", "lots")
    session = _capture_session()

    with pytest.raises(ValueError, match="TWIN_VECTOR_INDEX_CAPACITY"):
        await _make_store()._create_vector_index(session=session)

    session.run.assert_not_called()
