"""Contract tests for the persistent source-link backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
import secrets
from typing import Any

import pytest

from twindb_lightrag_memgraph import _pool, _retry
from twindb_lightrag_memgraph.server.source_links_store import (
    MemgraphSourceLinkStore,
    SourceLinkVersionConflict,
)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": "link-1",
        "doc_id": "doc-1",
        "url": "https://example.test/source",
        "label": None,
        "created_by": "alice",
        "created_at": "2026-08-19T10:00:00Z",
        "updated_by": "alice",
        "updated_at": "2026-08-19T10:00:00Z",
        "version": 1,
        "deleted": False,
        "deleted_by": None,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


class _Result:
    def __init__(
        self,
        *,
        record: dict[str, Any] | None = None,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.record = record
        self.records = list(records or [])
        self.index = 0
        self.consumed = False

    async def single(self):
        return self.record

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.records):
            raise StopAsyncIteration
        item = self.records[self.index]
        self.index += 1
        return item

    async def consume(self) -> None:
        self.consumed = True


class _Session:
    def __init__(self, script: list[_Result | Exception]) -> None:
        self.script = script
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, query: str, **params: Any):
        self.calls.append((query, params))
        response = self.script.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _install_pool(monkeypatch, session: _Session) -> list[str]:
    slot_entries: list[str] = []

    @asynccontextmanager
    async def write_slot():
        slot_entries.append("slot")
        yield

    @asynccontextmanager
    async def get_session():
        yield session

    monkeypatch.setattr(_pool, "acquire_write_slot", write_slot)
    monkeypatch.setattr(_pool, "get_session", get_session)
    monkeypatch.setattr(_pool, "get_read_session", get_session)
    return slot_entries


async def test_memgraph_initialize_uses_write_slot_for_each_ddl(monkeypatch):
    results = [_Result(), _Result()]
    session = _Session(results)
    slots = _install_pool(monkeypatch, session)

    await MemgraphSourceLinkStore("source_link_test").initialize()

    assert slots == ["slot"]
    assert len(session.calls) == 2
    assert all("CREATE INDEX" in query for query, _ in session.calls)
    assert all(result.consumed for result in results)


async def test_memgraph_create_retries_conflict_and_normalizes_nullable_shape(
    monkeypatch,
):
    persisted_without_nulls = {
        key: value for key, value in _row().items() if value is not None
    }
    session = _Session(
        [
            RuntimeError("Cannot resolve conflicting transactions"),
            _Result(record={"props": persisted_without_nulls}),
        ]
    )
    slots = _install_pool(monkeypatch, session)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(_retry.asyncio, "sleep", no_sleep)
    created = await MemgraphSourceLinkStore("source_link_test").create(_row())

    assert slots == ["slot", "slot"]
    assert created == _row()
    assert created["label"] is None
    assert created["deleted_by"] is None
    assert created["deleted_at"] is None


async def test_memgraph_update_and_delete_are_versioned_cas(monkeypatch):
    before_update = {key: value for key, value in _row().items() if value is not None}
    after_update = {
        **before_update,
        "updated_by": "bob",
        "updated_at": "2026-08-19T11:00:00Z",
        "version": 2,
    }
    after_delete = {
        **after_update,
        "deleted": True,
        "deleted_by": "bob",
        "deleted_at": "2026-08-19T12:00:00Z",
        "updated_at": "2026-08-19T12:00:00Z",
        "version": 3,
    }
    session = _Session(
        [
            RuntimeError("Cannot resolve conflicting transactions"),
            _Result(record={"before": before_update, "after": after_update}),
            RuntimeError("Cannot resolve conflicting transactions"),
            _Result(record={"before": after_update, "after": after_delete}),
        ]
    )
    slots = _install_pool(monkeypatch, session)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(_retry.asyncio, "sleep", no_sleep)
    store = MemgraphSourceLinkStore("source_link_test")

    before, updated = await store.update(
        "doc-1",
        "link-1",
        expected_version=1,
        url="https://example.test/source",
        label=None,
        actor="bob",
        updated_at="2026-08-19T11:00:00Z",
    )
    _, deleted = await store.delete(
        "doc-1",
        "link-1",
        expected_version=2,
        actor="bob",
        deleted_at="2026-08-19T12:00:00Z",
    )

    assert before == _row()
    assert updated["version"] == 2
    assert updated["label"] is None
    assert deleted["version"] == 3
    assert deleted["deleted"] is True
    assert slots == ["slot", "slot", "slot", "slot"]
    write_queries = [query for query, _ in session.calls]
    assert all("link.version = $expected_version" in query for query in write_queries)


@pytest.fixture
def memgraph_source_link_workspace() -> str:
    return f"source_links_{secrets.token_hex(4)}"


async def _cleanup(workspace: str) -> None:
    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            result = await session.run(
                f"MATCH (n:`TwinSourceLink_{workspace}`) DETACH DELETE n"
            )
            await result.consume()


@pytest.mark.integration
async def test_memgraph_source_link_crud_cas_and_tombstone(
    memgraph_source_link_workspace,
):
    workspace = memgraph_source_link_workspace
    store = MemgraphSourceLinkStore(workspace)
    try:
        await store.initialize()
        created = await store.create(_row())
        assert created == _row()
        assert await store.list_for_document("doc-1") == [_row()]

        before, updated = await store.update(
            "doc-1",
            "link-1",
            expected_version=1,
            url="https://example.test/updated",
            label=None,
            actor="bob",
            updated_at="2026-08-19T11:00:00Z",
        )
        assert before == _row()
        assert updated["version"] == 2
        assert updated["label"] is None

        with pytest.raises(SourceLinkVersionConflict):
            await store.update(
                "doc-1",
                "link-1",
                expected_version=1,
                url="https://example.test/stale",
                label="stale",
                actor="mallory",
                updated_at="2026-08-19T11:30:00Z",
            )

        _, deleted = await store.delete(
            "doc-1",
            "link-1",
            expected_version=2,
            actor="bob",
            deleted_at="2026-08-19T12:00:00Z",
        )
        assert deleted["deleted"] is True
        assert deleted["version"] == 3
        assert await store.list_for_document("doc-1") == []
    finally:
        await _cleanup(workspace)
