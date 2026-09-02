"""Tests for the WebUI activity store (S4c slice 3)."""

from __future__ import annotations

import json
import datetime
import pytest

from twindb_lightrag_memgraph.server import webui_seed
from twindb_lightrag_memgraph.server.webui_activitystore import (
    InMemoryActivityStore,
    MemgraphActivityStore,
    SCALARS_VERSION,
    make_memgraph_activity_store,
)


class TestInMemoryActivityStore:
    async def test_seeded_list_returns_all_events(self):
        store = InMemoryActivityStore()
        items, total, now_ms = await store.list()
        assert len(items) == len(webui_seed.ACTIVITY)
        assert total == len(webui_seed.ACTIVITY)
        assert now_ms == webui_seed.ACTIVITY_NOW_MS

    async def test_total_is_pre_limit(self):
        store = InMemoryActivityStore(
            events=[
                {
                    "id": "evt_1",
                    "ts": "2026-05-13T00:00:00Z",
                    "rel": "now",
                    "day": "Today",
                    "kind": "tag-mutation",
                    "sev": "info",
                    "actor": {"user": "demo.steward", "role": "KB Admin"},
                    "target": {"type": "tag", "label": "rman"},
                    "summary": "test1",
                    "meta": {},
                },
                {
                    "id": "evt_2",
                    "ts": "2026-05-13T00:05:00Z",
                    "rel": "now",
                    "day": "Today",
                    "kind": "tag-mutation",
                    "sev": "info",
                    "actor": {"user": "demo.steward", "role": "KB Admin"},
                    "target": {"type": "tag", "label": "swift"},
                    "summary": "test2",
                    "meta": {},
                },
                {
                    "id": "evt_3",
                    "ts": "2026-05-13T00:10:00Z",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "demo.operator", "role": "DBA"},
                    "target": {"type": "query", "label": "Who"},
                    "summary": "other",
                    "meta": {},
                },
            ],
        )
        items, total, _ = await store.list(actor="demo.steward", limit=1)
        assert len(items) == 1
        assert total == 2

    async def test_filter_by_sev(self):
        store = InMemoryActivityStore()
        items, total, _ = await store.list(sev="error")
        assert total >= 1
        assert all(e["sev"] == "error" for e in items)
        assert len(items) >= 1

    async def test_filter_by_kind_csv(self):
        store = InMemoryActivityStore()
        items, total, _ = await store.list(kind="retrieval,auth")
        assert total == len(items)
        assert all(e["kind"] in {"retrieval", "auth"} for e in items)

    async def test_filter_by_actor(self):
        store = InMemoryActivityStore()
        items, total, _ = await store.list(actor="demo.operator")
        assert total == len(items)
        assert all(e["actor"]["user"] == "demo.operator" for e in items)

    async def test_filter_by_q_substring(self):
        store = InMemoryActivityStore()
        items, total, _ = await store.list(q="Oracle")
        assert total >= 1
        assert len(items) >= 1

    async def test_filter_by_q_matches_event_id(self):
        store = InMemoryActivityStore()
        items, total, _ = await store.list(q=webui_seed.ACTIVITY[0]["id"])
        assert total == 1
        assert len(items) == 1
        assert items[0]["id"] == webui_seed.ACTIVITY[0]["id"]

    async def test_filter_by_range_uses_ts_cutoff(self):
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        now_iso = datetime.datetime.now(datetime.timezone.utc)

        def _stamp(minutes: int) -> str:
            return (
                (now_iso - datetime.timedelta(minutes=minutes))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

        store = InMemoryActivityStore(
            now_ms=now_ms,
            events=[
                {
                    "id": "evt_recent",
                    "ts": _stamp(60),
                    "rel": "now",
                    "day": "Today",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "demo.steward", "role": "KB Admin"},
                    "target": {"type": "query", "label": "within 24h"},
                    "summary": "recent",
                    "meta": {},
                },
                {
                    "id": "evt_old",
                    "ts": _stamp(60 * 30),
                    "rel": "today",
                    "day": "Today",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "demo.steward", "role": "KB Admin"},
                    "target": {"type": "query", "label": "older"},
                    "summary": "old",
                    "meta": {},
                },
            ],
        )
        recent, total, _ = await store.list(actor="demo.steward", range="24h")
        assert total == 1
        assert len(recent) == 1
        assert recent[0]["id"] == "evt_recent"

    async def test_range_uses_store_now_ms_not_wall_clock(self):
        now_ms = int(
            datetime.datetime(
                2026, 6, 1, 12, 0, tzinfo=datetime.timezone.utc
            ).timestamp()
            * 1000
        )
        now_iso = datetime.datetime(2026, 6, 1, 12, 0, tzinfo=datetime.timezone.utc)

        store = InMemoryActivityStore(
            now_ms=now_ms,
            events=[
                {
                    "id": "evt_recent",
                    "ts": (now_iso - datetime.timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "rel": "now",
                    "day": "Today",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "demo.steward", "role": "KB Admin"},
                    "target": {"type": "query", "label": "within 24h"},
                    "summary": "deterministic",
                    "meta": {},
                },
                {
                    "id": "evt_old",
                    "ts": (now_iso - datetime.timedelta(days=10))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "rel": "today",
                    "day": "Yesterday",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "demo.steward", "role": "KB Admin"},
                    "target": {"type": "query", "label": "older"},
                    "summary": "historic",
                    "meta": {},
                },
            ],
        )
        recent, total, _ = await store.list(actor="demo.steward", range="24h")
        assert total == 1
        assert len(recent) == 1
        assert recent[0]["id"] == "evt_recent"

    async def test_resource_id_filters_target_id_or_meta_doc_id(self):
        store = InMemoryActivityStore(
            events=[
                {
                    "id": "evt_target",
                    "ts": "2026-05-13T00:00:00Z",
                    "rel": "now",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "document", "label": "doc-one", "id": "doc-123"},
                    "summary": "target id",
                    "meta": {},
                },
                {
                    "id": "evt_meta",
                    "ts": "2026-05-13T00:01:00Z",
                    "rel": "now",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "document", "label": "doc-two"},
                    "summary": "meta doc id",
                    "meta": {"doc_id": "doc-456"},
                },
                {
                    "id": "evt_meta_doc_ids",
                    "ts": "2026-05-13T00:01:30Z",
                    "rel": "now",
                    "day": "Today",
                    "kind": "doc-deleted",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "bulk", "label": "2 documents"},
                    "summary": "bulk delete",
                    "meta": {"doc_ids": ["doc-bulk-a", "doc-bulk-b"]},
                },
                {
                    "id": "evt_historic",
                    "ts": "2026-05-13T00:02:00Z",
                    "rel": "now",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "document", "label": "legacy"},
                    "summary": "no doc id",
                    "meta": {},
                },
            ],
        )
        doc_events, total, _ = await store.list(resource_id="doc-456")
        assert total == 1
        assert len(doc_events) == 1
        assert doc_events[0]["id"] == "evt_meta"
        bulk_events, total, _ = await store.list(resource_id="doc-bulk-a")
        assert total == 1
        assert len(bulk_events) == 1
        assert bulk_events[0]["id"] == "evt_meta_doc_ids"
        historic_events, total, _ = await store.list(resource_id="doc-unknown")
        assert total == 0
        assert len(historic_events) == 0

    async def test_append_prepends_event(self):
        store = InMemoryActivityStore()
        event = {
            "id": "evt_new",
            "ts": "2026-05-13T00:00:00Z",
            "rel": "now",
            "day": "Today",
            "kind": "tag-mutation",
            "sev": "info",
            "actor": {"user": "demo.steward", "role": "KB Admin"},
            "target": {"type": "tag", "label": "rman"},
            "summary": "test",
            "meta": {},
        }
        await store.append(event)
        items, _, _ = await store.list()
        assert items[0]["id"] == "evt_new"

    async def test_append_many_matches_sequential_appends(self):
        events = [
            {
                "id": f"evt_batch_{i}",
                "ts": f"2026-05-13T00:0{i}:00Z",
                "kind": "doc-retagged",
                "sev": "info",
                "actor": {"user": "bolt.batch", "role": "operator"},
                "target": {"type": "document", "id": f"doc-{i}", "label": f"d{i}"},
                "summary": f"batch {i}",
                "meta": {"doc_id": f"doc-{i}"},
            }
            for i in range(3)
        ]
        batched = InMemoryActivityStore(events=[])
        sequential = InMemoryActivityStore(events=[])
        stored = await batched.append_many(events)
        for event in events:
            await sequential.append(event)

        assert [e["id"] for e in stored] == [e["id"] for e in events]
        batched_items, batched_total, _ = await batched.list()
        sequential_items, sequential_total, _ = await sequential.list()
        assert batched_items == sequential_items  # newest first, same order
        assert batched_total == sequential_total == 3
        # Returned copies are detached from the ledger, like ``append``.
        stored[0]["summary"] = "mutated"
        assert (await batched.list())[0][-1]["summary"] == "batch 0"
        assert await batched.append_many([]) == []

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
        items, _, _ = await store.list()
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


async def _seed_legacy_json_event(workspace: str, event: dict[str, object]) -> None:
    from twindb_lightrag_memgraph import _pool

    if not isinstance(event.get("id"), str):
        raise AssertionError("Legacy seed must define id")

    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            result = await session.run(
                f"MERGE (n:`WebuiActivity_{workspace}` {{id: $id}}) "
                "SET n.data = $data, "
                "    n.__created_at = timestamp(), "
                "    n.`__updated_at` = timestamp()",
                id=str(event["id"]),
                data=json.dumps(event, sort_keys=True),
            )
            await result.consume()


def _batch_event(event_id: str, index: int) -> dict[str, object]:
    return {
        "id": event_id,
        "ts": "2026-05-13T12:00:00Z",
        "rel": "now",
        "day": "Today",
        "kind": "doc-retagged",
        "sev": "info",
        "actor": {"user": "bolt.batch", "role": "operator"},
        "target": {"type": "document", "id": f"doc-{index}", "label": f"d{index}"},
        "summary": f"batch {index}",
        "meta": {"doc_id": f"doc-{index}"},
    }


@pytest.mark.integration
class TestMemgraphActivityStore:
    async def test_total_is_pre_limit(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            events, total, _ = await store.list(kind="retrieval", limit=1)
            assert len(events) == 1
            assert total > 1
            assert total >= len(events)
        finally:
            await _cleanup(_ws)

    async def test_range_filter(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            now = datetime.datetime.now(datetime.timezone.utc)
            await store.append(
                {
                    "id": "evt_mg_range_recent",
                    "ts": (now - datetime.timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "rel": "now",
                    "day": "Today",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "mg-range-tester", "role": "operator"},
                    "target": {"type": "query", "label": "mg range recent"},
                    "summary": "within 24h",
                    "meta": {},
                }
            )
            await store.append(
                {
                    "id": "evt_mg_range_old",
                    "ts": (now - datetime.timedelta(days=2))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "rel": "today",
                    "day": "Yesterday",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "mg-range-tester", "role": "operator"},
                    "target": {"type": "query", "label": "mg range old"},
                    "summary": "older than 24h",
                    "meta": {},
                }
            )

            events, total, _ = await store.list(actor="mg-range-tester", range="24h")
            assert total == 1
            assert len(events) == 1
            assert events[0]["id"] == "evt_mg_range_recent"
        finally:
            await _cleanup(_ws)

    async def test_resource_id_filter(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            await store.append(
                {
                    "id": "evt_mg_resource_target",
                    "ts": "2026-05-13T00:00:00Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "document", "label": "doc-one", "id": "doc-123"},
                    "summary": "target id",
                    "meta": {},
                }
            )
            await store.append(
                {
                    "id": "evt_mg_resource_meta",
                    "ts": "2026-05-13T00:01:00Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "document", "label": "doc-two"},
                    "summary": "meta doc id",
                    "meta": {"doc_id": "doc-456"},
                }
            )
            await store.append(
                {
                    "id": "evt_mg_resource_meta_doc_ids",
                    "ts": "2026-05-13T00:02:00Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "doc-deleted",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "bulk", "label": "2 documents"},
                    "summary": "bulk delete",
                    "meta": {"doc_ids": ["doc-bulk-a", "doc-bulk-b"]},
                }
            )

            events, total, _ = await store.list(resource_id="doc-456")
            assert total == 1
            assert len(events) == 1
            assert events[0]["id"] == "evt_mg_resource_meta"
            events, total, _ = await store.list(resource_id="doc-bulk-b")
            assert total == 1
            assert len(events) == 1
            assert events[0]["id"] == "evt_mg_resource_meta_doc_ids"
        finally:
            await _cleanup(_ws)

    async def test_q_matches_event_id(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            await store.append(
                {
                    "id": "evt_mg_query_id_match",
                    "ts": "2026-05-13T00:00:00Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "retrieval",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "query", "label": "unrelated target"},
                    "summary": "unrelated summary",
                    "meta": {},
                }
            )

            events, total, _ = await store.list(q="evt_mg_query_id_match")
            assert total == 1
            assert len(events) == 1
            assert events[0]["id"] == "evt_mg_query_id_match"
        finally:
            await _cleanup(_ws)

    async def test_resource_id_total_does_not_count_events_without_doc_ids(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            await store.append(
                {
                    "id": "evt_mg_resource_meta_only",
                    "ts": "2026-05-13T00:01:00Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "system", "role": "operator"},
                    "target": {"type": "document", "label": "doc-two"},
                    "summary": "meta doc id",
                    "meta": {"doc_id": "doc-456"},
                }
            )
            for idx in range(3):
                await store.append(
                    {
                        "id": f"evt_mg_resource_no_doc_{idx}",
                        "ts": f"2026-05-13T00:0{idx}:00Z",
                        "rel": "today",
                        "day": "Today",
                        "kind": "doc-approved",
                        "sev": "info",
                        "actor": {"user": "system", "role": "operator"},
                        "target": {"type": "document", "label": f"docless-{idx}"},
                        "summary": "valid event without document id",
                        "meta": {},
                    }
                )

            events, total, _ = await store.list(resource_id="doc-456")
            assert total == 1
            assert len(events) == 1
            assert events[0]["id"] == "evt_mg_resource_meta_only"
        finally:
            await _cleanup(_ws)

    async def test_legacy_json_events_are_backfilled_on_initialize(self, _ws):
        try:
            await _seed_legacy_json_event(
                _ws,
                {
                    "id": "legacy-json-event",
                    "ts": "2026-05-13T12:34:56Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "legacy-user", "role": "operator"},
                    "target": {
                        "type": "document",
                        "label": "legacy-doc",
                        "id": "doc-legacy",
                    },
                    "summary": "legacy activity",
                    "meta": {"doc_id": "meta-legacy"},
                },
            )

            store = await make_memgraph_activity_store(workspace=_ws)

            items, total, _ = await store.list(
                actor="legacy-user",
                kind="doc-approved",
                resource_id="doc-legacy",
            )
            assert total == 1
            assert len(items) == 1
            assert items[0]["id"] == "legacy-json-event"

            from twindb_lightrag_memgraph import _pool

            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"MATCH (n:`WebuiActivity_{_ws}` {{id: $id}}) "
                    "RETURN n.kind AS kind, n.sev AS sev, n.actor_user AS actor_user, "
                    "       n.target_id AS target_id, n.meta_doc_id AS meta_doc_id, "
                    "       n.target_label AS target_label, n.summary AS summary, "
                    "       n.ts_ms AS ts_ms, "
                    "       n.`__scalars_version` AS scalars_version",
                    id="legacy-json-event",
                )
                record = await result.single()
                await result.consume()

            assert record is not None
            assert record["kind"] == "doc-approved"
            assert record["sev"] == "info"
            assert record["actor_user"] == "legacy-user"
            assert record["target_id"] == "doc-legacy"
            assert record["meta_doc_id"] == "meta-legacy"
            assert record["target_label"] == "legacy-doc"
            assert record["summary"] == "legacy activity"
            assert record["scalars_version"] == SCALARS_VERSION
            assert record["ts_ms"] == int(
                datetime.datetime(
                    2026, 5, 13, 12, 34, 56, tzinfo=datetime.timezone.utc
                ).timestamp()
                * 1000
            )
        finally:
            await _cleanup(_ws)

    async def test_backfill_is_idempotent_after_scalars_version(self, _ws):
        try:
            await _seed_legacy_json_event(
                _ws,
                {
                    "id": "legacy-idempotent",
                    "ts": "2026-05-13T12:34:56Z",
                    "rel": "today",
                    "day": "Today",
                    "kind": "doc-approved",
                    "sev": "info",
                    "actor": {"user": "legacy-user", "role": "operator"},
                    "target": {"type": "document", "label": "legacy-doc"},
                    "summary": "",
                    "meta": {},
                },
            )

            store = MemgraphActivityStore(workspace=_ws)
            await store.initialize()

            from twindb_lightrag_memgraph import _pool

            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"MATCH (n:`WebuiActivity_{_ws}` {{id: $id}}) "
                        "SET n.`__updated_at` = $marker",
                        id="legacy-idempotent",
                        marker=424242,
                    )
                    await result.consume()

            await store.initialize()

            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"MATCH (n:`WebuiActivity_{_ws}` {{id: $id}}) "
                    "RETURN n.`__updated_at` AS updated_at, "
                    "       n.`__scalars_version` AS scalars_version",
                    id="legacy-idempotent",
                )
                record = await result.single()
                await result.consume()

            assert record is not None
            assert record["scalars_version"] == SCALARS_VERSION
            assert record["updated_at"] == 424242
        finally:
            await _cleanup(_ws)

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
            items, total, now_ms = await store.list()
            assert len(items) == len(webui_seed.ACTIVITY)
            assert isinstance(now_ms, int)
            assert total == len(webui_seed.ACTIVITY)
            assert now_ms >= webui_seed.ACTIVITY_NOW_MS
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
                "actor": {"user": "demo.steward", "role": "operator"},
                "target": {"type": "tag", "label": "argocd"},
                "summary": "appended via test",
                "meta": {"test": True},
            }
            await store.append(new_event)
            items, _, _ = await store.list()
            assert items[0]["id"] == "evt_appended"
        finally:
            await _cleanup(_ws)

    async def test_append_many_orders_and_stores_like_sequential_appends(self, _ws):
        """ONE batched write must read back exactly like five sequential appends.

        Ordering is the trap: Memgraph evaluates ``timestamp()`` once per
        query, so a batch that relied on it would tie every row's
        ``__created_at`` and come back in arbitrary order.
        """
        from twindb_lightrag_memgraph import _pool

        seq_ws = f"{_ws}_seq"
        events = [
            {
                "id": f"evt_batch_{i}",
                "ts": f"2026-05-13T12:0{i}:00Z",
                "rel": "now",
                "day": "Today",
                "kind": "doc-retagged",
                "sev": "info",
                "actor": {"user": "bolt.batch", "role": "operator"},
                "target": {"type": "document", "id": f"doc-{i}", "label": f"d{i}"},
                "summary": f"batch {i}",
                "meta": {"doc_id": f"doc-{i}"},
            }
            for i in range(5)
        ]
        try:
            batched = await make_memgraph_activity_store(workspace=_ws)
            sequential = await make_memgraph_activity_store(workspace=seq_ws)
            stored = await batched.append_many(events)
            for event in events:
                await sequential.append(event)
            assert [e["id"] for e in stored] == [e["id"] for e in events]

            batched_items, batched_total, _ = await batched.list(actor="bolt.batch")
            sequential_items, sequential_total, _ = await sequential.list(
                actor="bolt.batch"
            )
            assert batched_items == sequential_items
            assert batched_total == sequential_total == 5
            assert [e["id"] for e in batched_items] == [
                f"evt_batch_{i}" for i in (4, 3, 2, 1, 0)
            ]

            # The scalars Memgraph filters on are written per row. The batch
            # shares ONE real creation time (one write, one clock reading) and
            # carries its insertion order in __seq — the clock is never faked.
            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"MATCH (n:`WebuiActivity_{_ws}`) WHERE n.actor_user = $actor "
                    "RETURN n.id AS id, n.`__created_at` AS created_at, "
                    "n.`__seq` AS seq, n.kind AS kind, n.target_id AS target_id, "
                    "n.`__scalars_version` AS version "
                    "ORDER BY created_at, seq",
                    actor="bolt.batch",
                )
                rows = await result.data()
                await result.consume()
            assert [r["id"] for r in rows] == [f"evt_batch_{i}" for i in range(5)]
            assert [r["seq"] for r in rows] == list(range(5))
            assert len({r["created_at"] for r in rows}) == 1
            assert all(r["kind"] == "doc-retagged" for r in rows)
            assert [r["target_id"] for r in rows] == [f"doc-{i}" for i in range(5)]
            assert all(r["version"] == SCALARS_VERSION for r in rows)

            # MERGE on id: re-appending the batch neither duplicates nor reorders.
            await batched.append_many(events)
            again, again_total, _ = await batched.list(actor="bolt.batch")
            assert again_total == 5 and again == batched_items
        finally:
            await _cleanup(_ws)
            await _cleanup(seq_ws)

    async def test_append_right_after_a_batch_lists_first(self, _ws):
        """Review finding on PR #486: a synthetic per-row clock offset would let
        a later single append sort behind older batch rows. The clock is real
        and shared by the batch; the later append carries a later clock."""
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            batch = [_batch_event(f"evt_b_{i}", i) for i in range(500)]
            await store.append_many(batch)
            await store.append(_batch_event("evt_after", 500))
            items, total, _ = await store.list(actor="bolt.batch", limit=3)
            assert total == 501
            assert [e["id"] for e in items] == ["evt_after", "evt_b_499", "evt_b_498"]
        finally:
            await _cleanup(_ws)

    async def test_consecutive_batches_do_not_interleave(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            await store.append_many([_batch_event(f"evt_a_{i}", i) for i in range(5)])
            await store.append_many([_batch_event(f"evt_b_{i}", i) for i in range(5)])
            items, total, _ = await store.list(actor="bolt.batch")
            assert total == 10
            assert [e["id"] for e in items] == [
                f"evt_b_{i}" for i in (4, 3, 2, 1, 0)
            ] + [f"evt_a_{i}" for i in (4, 3, 2, 1, 0)]
        finally:
            await _cleanup(_ws)

    async def test_append_many_validates_every_id_before_writing(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            with pytest.raises(ValueError):
                await store.append_many(
                    [{"id": "evt_ok", "kind": "x"}, {"kind": "no-id"}]
                )
            items, _, _ = await store.list(q="evt_ok")
            assert items == []  # nothing from the rejected batch landed
        finally:
            await _cleanup(_ws)

    async def test_append_writes_scalars_for_memgraph_matching(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            appended = {
                "id": "evt_append_scalars",
                "ts": "2026-05-13T12:00:00Z",
                "rel": "now",
                "day": "Today",
                "kind": "tag-mutation",
                "sev": "info",
                "actor": {"user": "demo.steward", "role": "operator"},
                "target": {"type": "document", "label": "doc-target", "id": "doc-001"},
                "summary": "scalar-check",
                "meta": {"doc_id": "doc-meta", "doc_ids": ["doc-a", "doc-b"]},
            }
            await store.append(appended)

            from twindb_lightrag_memgraph import _pool

            async with _pool.get_read_session() as session:
                result = await session.run(
                    f"MATCH (n:`WebuiActivity_{_ws}` {{id: $id}}) "
                    "RETURN n.target_id AS target_id, "
                    "       n.meta_doc_id AS meta_doc_id, "
                    "       n.meta_doc_ids AS meta_doc_ids, "
                    "       n.ts_ms AS ts_ms, "
                    "       n.`__scalars_version` AS scalars_version",
                    id="evt_append_scalars",
                )
                record = await result.single()
                await result.consume()

            assert record is not None
            assert record["target_id"] == "doc-001"
            assert record["meta_doc_id"] == "doc-meta"
            assert record["meta_doc_ids"] == ["doc-a", "doc-b"]
            assert record["scalars_version"] == SCALARS_VERSION
            expected_ts = int(
                datetime.datetime(
                    2026, 5, 13, 12, 0, 0, tzinfo=datetime.timezone.utc
                ).timestamp()
                * 1000
            )
            assert record["ts_ms"] == expected_ts
        finally:
            await _cleanup(_ws)

    async def test_filters_round_trip(self, _ws):
        try:
            store = await make_memgraph_activity_store(workspace=_ws)
            errors, _, _ = await store.list(sev="error")
            assert all(e["sev"] == "error" for e in errors)
            kinds, _, _ = await store.list(kind="retrieval")
            assert all(e["kind"] == "retrieval" for e in kinds)
        finally:
            await _cleanup(_ws)
