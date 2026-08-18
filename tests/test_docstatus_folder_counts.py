"""Unit contract for batched folder membership counts."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


class _Result:
    def __init__(self, records):
        self._records = records
        self.consumed = False

    def __aiter__(self):
        async def rows():
            for record in self._records:
                yield record

        return rows()

    async def consume(self):
        self.consumed = True


async def test_get_folder_counts_uses_one_set_query_and_fills_missing(monkeypatch):
    result = _Result([{"folder": "alpha", "cnt": 3}])
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)

    @asynccontextmanager
    async def read_session():
        yield session

    monkeypatch.setattr(_pool, "get_read_session", read_session)
    store = MemgraphDocStatusStorage(
        namespace="doc_status",
        global_config={"workspace": "folder_counts"},
        embedding_func=None,
    )

    counts = await store.get_folder_counts(["alpha", "empty", "alpha"])

    assert counts == {"alpha": 3, "empty": 0}
    session.run.assert_awaited_once()
    query = session.run.await_args.args[0]
    assert "WHERE f.id IN $folders" in query
    assert "count(DISTINCT n) AS cnt" in query
    assert "UNWIND" not in query
    assert session.run.await_args.kwargs == {"folders": ["alpha", "empty"]}
    assert result.consumed is True

    assert await store.get_folder_counts([]) == {}
    session.run.assert_awaited_once()
