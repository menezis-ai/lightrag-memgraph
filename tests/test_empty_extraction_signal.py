"""Operator signal for PROCESSED docs with an empty graph contribution.

Audit 2026-07-02 addendum, finding B: an unparsable extraction LLM output
leaves the document PROCESSED with zero entities / zero relations and no
operator signal. The buffered-merge wrapper now emits a WARNING log (always)
plus a best-effort ``pipeline-warning`` activity event when the whole merge
buffered nothing. The DocStatus transition itself is upstream's contract and
stays untouched.

Compat contract (docs/test-doctrine-lightrag-compat.md): a non-Memgraph
graph backend goes through the native merge untouched — no proxy, no signal.
No Memgraph required — the fake original merges never flush any query.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from twindb_lightrag_memgraph.patches import registry


class _FakeStore:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def record_activity(self, event: dict) -> None:
        self.events.append(event)


def _install_wrapper(monkeypatch, fake_original):
    """Install the buffered wrapper over ``fake_original`` and return it.

    monkeypatch snapshots the pre-test bindings, so the module-global merge
    symbols are restored at teardown even though ``_patch_merge_write_path``
    re-assigns them in between.
    """
    import lightrag.lightrag as lr_mod
    import lightrag.operate as operate

    monkeypatch.setattr(operate, "merge_nodes_and_edges", fake_original)
    monkeypatch.setattr(lr_mod, "merge_nodes_and_edges", fake_original, raising=False)
    registry._patch_merge_write_path()
    return operate.merge_nodes_and_edges


def _memgraph_mock(workspace: str = "sig_ws"):
    from lightrag.kg.memgraph_impl import MemgraphStorage

    graph = MagicMock(spec=MemgraphStorage)
    graph.workspace = workspace
    return graph


async def test_empty_merge_emits_warning_and_activity(monkeypatch, caplog):
    import twindb_lightrag_memgraph.server.webui_router as wr

    async def fake_original(*args, **kwargs):
        return None  # extraction produced nothing → buffers stay empty

    merge = _install_wrapper(monkeypatch, fake_original)
    store = _FakeStore()
    monkeypatch.setattr(wr, "get_store", lambda: store)
    graph = _memgraph_mock()

    with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
        await merge(
            chunk_results=[],
            knowledge_graph_inst=graph,
            doc_id="doc-empty",
            file_path="empty.pdf",
        )

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("EMPTY graph" in m and "doc-empty" in m for m in warnings)

    assert len(store.events) == 1
    event = store.events[0]
    assert event["kind"] == "pipeline-warning"
    assert event["sev"] == "warning"
    assert event["target"]["id"] == "doc-empty"
    assert event["meta"]["entities"] == 0 and event["meta"]["relations"] == 0
    # The emitted payload must honor the WebUI wire contract.
    from twindb_lightrag_memgraph.server.webui_models import ActivityEvent

    ActivityEvent.model_validate(event)


async def test_non_empty_merge_emits_no_signal(monkeypatch, caplog):
    import twindb_lightrag_memgraph.server.webui_router as wr
    from twindb_lightrag_memgraph import _buffered_graph

    async def fake_original(*args, **kwargs):
        proxy = kwargs["knowledge_graph_inst"]
        await proxy.upsert_node("E1", {"entity_id": "E1"})

    merge = _install_wrapper(monkeypatch, fake_original)
    store = _FakeStore()
    monkeypatch.setattr(wr, "get_store", lambda: store)

    # Neutralize the flush's Bolt traffic (no Memgraph in this test).
    async def fake_flush(self):
        self._node_buffer.clear()
        self._node_types.clear()
        self._edge_buffer.clear()

    monkeypatch.setattr(_buffered_graph._BufferedGraphProxy, "flush", fake_flush)

    with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
        await merge(
            chunk_results=[],
            knowledge_graph_inst=_memgraph_mock(),
            doc_id="doc-ok",
            file_path="ok.pdf",
        )

    assert not any("EMPTY graph" in r.getMessage() for r in caplog.records)
    assert store.events == []


async def test_native_graph_backend_stays_untouched(monkeypatch, caplog):
    """LightRAG-compat: a non-Memgraph backend takes the native merge path —
    same instance forwarded, no proxy, no warning, no activity event."""
    import twindb_lightrag_memgraph.server.webui_router as wr

    seen: list = []

    async def fake_original(*args, **kwargs):
        seen.append(kwargs.get("knowledge_graph_inst"))

    merge = _install_wrapper(monkeypatch, fake_original)
    store = _FakeStore()
    monkeypatch.setattr(wr, "get_store", lambda: store)
    native_graph = MagicMock()  # not a MemgraphStorage

    with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
        await merge(
            chunk_results=[],
            knowledge_graph_inst=native_graph,
            doc_id="doc-native",
            file_path="native.pdf",
        )

    assert seen == [native_graph]  # forwarded untouched, not proxied
    assert not any("EMPTY graph" in r.getMessage() for r in caplog.records)
    assert store.events == []


async def test_signal_survives_unavailable_activity_store(monkeypatch, caplog):
    """The activity emit is best-effort: a broken/absent store must degrade
    to the WARNING log alone, never raise into the ingestion pipeline."""
    import twindb_lightrag_memgraph.server.webui_router as wr

    monkeypatch.setattr(
        wr, "get_store", lambda: (_ for _ in ()).throw(RuntimeError("no store"))
    )
    graph = _memgraph_mock()

    with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
        await registry._signal_empty_extraction_merge(
            graph, {"doc_id": "doc-x", "file_path": "x.pdf"}
        )

    assert any("EMPTY graph" in r.getMessage() for r in caplog.records)


async def test_signal_tolerates_legacy_merge_signature(caplog):
    """Old merge signatures carry no doc_id/file_path kwargs — the signal
    must still log without raising."""
    graph = _memgraph_mock()
    with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
        await registry._signal_empty_extraction_merge(graph, {})
    assert any("<unknown>" in r.getMessage() for r in caplog.records)
