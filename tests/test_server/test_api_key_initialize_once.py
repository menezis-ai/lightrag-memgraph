"""Request-path guards for one-time API-key schema initialization."""

from __future__ import annotations

import asyncio
import gc
import weakref
from contextlib import asynccontextmanager

import pytest

from twindb_lightrag_memgraph.server import api_key_store


class _Result:
    async def consume(self) -> None:
        return None


class _FakePool:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.write_slots = 0
        self.queries: list[str] = []
        self.fail_first = fail_first
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    @asynccontextmanager
    async def acquire_write_slot(self):
        self.write_slots += 1
        yield

    @asynccontextmanager
    async def get_session(self):
        yield self

    async def run(self, query: str) -> _Result:
        self.queries.append(query)
        self.started.set()
        await self.release.wait()
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("Memgraph temporarily unavailable")
        return _Result()


@pytest.fixture(autouse=True)
def _isolated_schema_state(monkeypatch):
    monkeypatch.setattr(api_key_store, "_schema_ready", set())
    monkeypatch.setattr(api_key_store, "_schema_locks", {})
    monkeypatch.setattr(api_key_store, "_schema_locks_loop_id", None)


@pytest.fixture
def fake_pool(monkeypatch) -> _FakePool:
    pool = _FakePool()
    monkeypatch.setattr(
        api_key_store._pool, "acquire_write_slot", pool.acquire_write_slot
    )
    monkeypatch.setattr(api_key_store._pool, "get_session", pool.get_session)
    return pool


async def test_successful_initialization_is_reused(fake_pool):
    await api_key_store.initialize("api_key_once")
    await api_key_store.initialize("api_key_once")

    assert fake_pool.write_slots == 1
    assert len(fake_pool.queries) == 2
    assert api_key_store._schema_locks == {}


async def test_concurrent_cold_initialization_is_single_flight(fake_pool):
    fake_pool.release.clear()
    requests = [
        asyncio.create_task(api_key_store.initialize("api_key_burst"))
        for _ in range(20)
    ]
    await fake_pool.started.wait()
    await asyncio.sleep(0)

    assert fake_pool.write_slots == 1
    assert len(fake_pool.queries) == 1

    fake_pool.release.set()
    await asyncio.gather(*requests)
    assert fake_pool.write_slots == 1
    assert len(fake_pool.queries) == 2


async def test_failed_initialization_is_retried(monkeypatch):
    pool = _FakePool(fail_first=True)
    monkeypatch.setattr(
        api_key_store._pool, "acquire_write_slot", pool.acquire_write_slot
    )
    monkeypatch.setattr(api_key_store._pool, "get_session", pool.get_session)

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        await api_key_store.initialize("api_key_retry")
    await api_key_store.initialize("api_key_retry")

    assert pool.write_slots == 2
    assert len(pool.queries) == 3


async def test_connection_target_change_reinitializes(fake_pool, monkeypatch):
    await api_key_store.initialize("api_key_retarget")
    monkeypatch.setenv("MEMGRAPH_DATABASE", "alternate")
    await api_key_store.initialize("api_key_retarget")

    assert fake_pool.write_slots == 2
    assert len(fake_pool.queries) == 4


def test_failed_initialization_does_not_retain_closed_event_loops(monkeypatch):
    async def fail_schema(_label: str) -> None:
        raise RuntimeError("Memgraph temporarily unavailable")

    monkeypatch.setattr(api_key_store, "_initialize_schema", fail_schema)
    loop_refs = []

    for index in range(5):
        loop = asyncio.new_event_loop()
        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            loop.run_until_complete(api_key_store.initialize(f"api_key_fail_{index}"))
        loop_refs.append(weakref.ref(loop))
        loop.close()
        del loop

    gc.collect()

    assert all(loop_ref() is None for loop_ref in loop_refs)
    assert len(api_key_store._schema_locks) == 1


def test_schema_key_uses_pool_connection_identity(monkeypatch):
    monkeypatch.setattr(
        api_key_store._pool,
        "connection_identity",
        lambda: ("neo4j+s://cluster.example", "tenant_a"),
    )

    assert api_key_store._schema_key("workspace") == (
        "workspace",
        "neo4j+s://cluster.example",
        "tenant_a",
    )
