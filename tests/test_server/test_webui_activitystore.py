"""Tests for the WebUI activity store (S4c slice 3)."""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server import webui_seed
from twindb_lightrag_memgraph.server.webui_activitystore import (
    InMemoryActivityStore,
    MemgraphActivityStore,
    make_memgraph_activity_store,
)


class TestInMemoryActivityStore:
    async def test_seeded_list_returns_all_events(self):
        store = InMemoryActivityStore()
        items, now_ms = await store.list()
        assert len(items) == len(webui_seed.ACTIVITY)
        assert now_ms == webui_seed.ACTIVITY_NOW_MS

    async def test_filter_by_sev(self):
        store = InMemoryActivityStore()
        items, _ = await store.list(sev="error")
        assert all(e["sev"] == "error" for e in items)
        assert len(items) >= 1

    async def test_filter_by_kind_csv(self):
        store = InMemoryActivityStore()
        items, _ = await store.list(kind="retrieval,auth")
        assert all(e["kind"] in {"retrieval", "auth"} for e in items)

    async def test_filter_by_actor(self):
        store = InMemoryActivityStore()
        items, _ = await store.list(actor="marc.berthier")
        assert all(e["actor"]["user"] == "marc.berthier" for e in items)

    async def test_filter_by_q_substring(self):
        store = InMemoryActivityStore()
        items, _ = await store.list(q="Oracle")
        assert len(items) >= 1

    async def test_append_prepends_event(self):
        store = InMemoryActivityStore()
        event = {
            "id": "evt_new",
            "ts": "2026-05-13T00:00:00Z",
            "rel": "now",
            "day": "Today",
            "kind": "tag-mutation",
            "sev": "info",
            "actor": {"user": "claire.benoit", "role": "KB Admin"},
            "target": {"type": "tag", "label": "rman"},
            "summary": "test",
            "meta": {},
        }
        await store.append(event)
        items, _ = await store.list()
        assert items[0]["id"] == "evt_new"

    async def test_append_returns_deep_copy(self):
        store = InMemoryActivityStore(events=[])
        event = {
            "id": "evt_1",
            "ts": "2026-05-13T00:00:00Z",
            "rel": "now",
            "day": "Today",
            "kind": "tag-mutation",
            "sev": "info",
            "actor": {"user": "x", "role": "y"},
            "target": {"type": "tag", "label": "z"},
            "summary": "",
            "meta": {},
        }
        stored = await store.append(event)
        stored["meta"]["mutated"] = True
        items, _ = await store.list()
        assert "mutated" not in items[0]["meta"]


# ---------------------------------------------------------------------------
# Integration — Memgraph backend
# ---------------------------------------------------------------------------


@pytest.fixture
def _ws():
    import secrets

    return f"actstore_{secrets.token_hex(4)}"


async def _cleanup(workspace: str) -> None:
    from twindb_lightrag_memgraph import _pool

    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            result = await session.run(
                f"MATCH (n:`WebuiActivity_{workspace}`) DETACH DELETE n"
            )
            await result.consume()


@pytest.mark.integration
class TestMemgraphActivityStore:
    async def test_bootstrap_writes_seed_then_skips(self, _ws):
        try:
            store = MemgraphActivityStore(workspace=_ws)
            await store.initialize()
            first = await store.bootstrap_if_empty()
            second = await store.bootstrap_if_empty()
            assert first is True
            assert second is False
        finally:
            await _cleanup(_ws)

    async def test_list_returns_seed_newest_first(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            items, now_ms = await store.list()
            assert len(items) == len(webui_seed.ACTIVITY)
            assert now_ms == webui_seed.ACTIVITY_NOW_MS
            # Seed insert order = reversed(ACTIVITY), so newest-first reading
            # returns ACTIVITY[0] first.
            assert items[0]["id"] == webui_seed.ACTIVITY[0]["id"]
        finally:
            await _cleanup(_ws)

    async def test_append_then_listed_first(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            new_event = {
                "id": "evt_appended",
                "ts": "2026-05-13T12:00:00Z",
                "rel": "now",
                "day": "Today",
                "kind": "tag-mutation",
                "sev": "info",
                "actor": {"user": "claire.benoit", "role": "operator"},
                "target": {"type": "tag", "label": "argocd"},
                "summary": "appended via test",
                "meta": {"test": True},
            }
            await store.append(new_event)
            items, _ = await store.list()
            assert items[0]["id"] == "evt_appended"
        finally:
            await _cleanup(_ws)

    async def test_filters_round_trip(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            errors, _ = await store.list(sev="error")
            assert all(e["sev"] == "error" for e in errors)
            kinds, _ = await store.list(kind="retrieval")
            assert all(e["kind"] == "retrieval" for e in kinds)
        finally:
            await _cleanup(_ws)
