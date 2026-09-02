"""``_emit_bulk_retag_events`` records the whole batch in ONE ledger write.

Structural contract behind ``tests/benchmarks/bulk_retag_activity_batch.py``:
the emitter must call ``record_activities`` once with every event in document
order, never the per-event ``record_activity`` — and the events must be the
ones the per-event loop used to build.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server.webui import routes_tags


class _RecordingStore:
    def __init__(self) -> None:
        self.batches: list[list[dict]] = []

    async def record_activities(self, events):
        self.batches.append(list(events))
        return events

    async def record_activity(self, _event):
        raise AssertionError("bulk retag must not write the ledger per event")


@pytest.fixture
def store(monkeypatch):
    recording = _RecordingStore()
    monkeypatch.setattr(routes_tags, "get_store", lambda: recording)
    return recording


async def test_emits_one_batch_with_every_event_in_document_order(store):
    doc_ids = ["doc-b", "doc-a", "doc-c"]
    await routes_tags._emit_bulk_retag_events(
        doc_ids,
        {"doc-a": ["x"], "doc-b": ["x", "y"]},
        {"doc-a": "A.md", "doc-b": "", "doc-c": "C.md"},
        ["x"],
        ["z"],
        "demo.steward",
    )
    assert len(store.batches) == 1
    events = store.batches[0]
    assert [e["target"]["id"] for e in events] == doc_ids
    assert [e["target"]["label"] for e in events] == ["doc-b", "A.md", "C.md"]
    assert {e["kind"] for e in events} == {"doc-retagged"}
    assert {e["actor"]["user"] for e in events} == {"demo.steward"}
    assert [e["meta"]["resulting_tags"] for e in events] == [["x", "y"], ["x"], []]
    assert all(
        e["meta"]["adds"] == ["x"] and e["meta"]["removes"] == ["z"] for e in events
    )
    assert all(e["summary"] == "tags: +x -z" for e in events)
    assert len({e["id"] for e in events}) == 3


async def test_no_documents_means_no_ledger_write(store):
    await routes_tags._emit_bulk_retag_events([], {}, {}, ["x"], [], "demo.steward")
    assert store.batches == []
