"""Tests for the WebUI notifications store (S4c slice 3)."""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server import webui_seed
from twindb_lightrag_memgraph.server.webui_notificationstore import (
    InMemoryNotificationStore,
    MemgraphNotificationStore,
    make_memgraph_notification_store,
)


class TestInMemoryNotificationStore:
    async def test_seeded_list(self):
        store = InMemoryNotificationStore()
        items = await store.list()
        assert len(items) == len(webui_seed.NOTIFICATIONS)

    async def test_mark_all_read_flips_every_item(self):
        store = InMemoryNotificationStore()
        await store.mark_all_read()
        items = await store.list()
        assert all(n["read"] is True for n in items)

    async def test_clear_empties(self):
        store = InMemoryNotificationStore()
        await store.clear()
        assert await store.list() == []

    async def test_push_prepends(self):
        store = InMemoryNotificationStore()
        new = {
            "id": "n_new",
            "kind": "tag-mutation",
            "title": "Tag",
            "tagname": "rman",
            "suffix": "requested",
            "sub": "",
            "rel": "now",
            "read": False,
        }
        await store.push(new)
        items = await store.list()
        assert items[0]["id"] == "n_new"


# ---------------------------------------------------------------------------
# Integration — Memgraph backend
# ---------------------------------------------------------------------------


@pytest.fixture
def _ws():
    import secrets

    return f"notifstore_{secrets.token_hex(4)}"


async def _cleanup(workspace: str) -> None:
    from twindb_lightrag_memgraph import _pool

    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            result = await session.run(
                f"MATCH (n:`WebuiNotification_{workspace}`) DETACH DELETE n"
            )
            await result.consume()


@pytest.mark.integration
class TestMemgraphNotificationStore:
    async def test_bootstrap_idempotent(self, _ws):
        try:
            store = MemgraphNotificationStore(workspace=_ws)
            await store.initialize()
            assert await store.bootstrap_if_empty() is True
            assert await store.bootstrap_if_empty() is False
        finally:
            await _cleanup(_ws)

    async def test_list_after_bootstrap(self, _ws):
        try:
            store = await make_memgraph_notification_store(workspace=_ws)
            items = await store.list()
            assert len(items) == len(webui_seed.NOTIFICATIONS)
        finally:
            await _cleanup(_ws)

    async def test_mark_all_read_persists(self, _ws):
        try:
            store = await make_memgraph_notification_store(workspace=_ws)
            await store.mark_all_read()
            items = await store.list()
            assert all(n["read"] is True for n in items)
        finally:
            await _cleanup(_ws)

    async def test_clear_then_empty(self, _ws):
        try:
            store = await make_memgraph_notification_store(workspace=_ws)
            await store.clear()
            assert await store.list() == []
        finally:
            await _cleanup(_ws)

    async def test_push_persists_newest_first(self, _ws):
        try:
            store = await make_memgraph_notification_store(workspace=_ws)
            await store.push(
                {
                    "id": "n_new",
                    "kind": "tag-mutation",
                    "title": "Tag",
                    "tagname": "rman",
                    "suffix": "requested",
                    "sub": "test",
                    "rel": "now",
                    "read": False,
                }
            )
            items = await store.list()
            assert items[0]["id"] == "n_new"
        finally:
            await _cleanup(_ws)
