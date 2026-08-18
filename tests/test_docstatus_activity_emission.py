"""Unit tests for the ``source-ready`` / ``source-failed`` Activity emission.

QA V8 ACT-V5-001: the Activity ledger exposed Source ready / Source failed
filters with no live emission behind them (audit 2026-06-29 rows "missing").
The seam is ``MemgraphDocStatusStorage.upsert`` — the write path captures the
previous ``status`` per doc and emits only on genuine terminal transitions.

These tests cover the pure transition computation and the best-effort
emission helper without a Memgraph (the Cypher old-status capture itself is
exercised by the integration suite in ``tests/test_docstatus.py``).
"""

from unittest.mock import AsyncMock, patch

import pytest

from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


def _entry(doc_id: str, status: str, **props):
    return {
        "id": doc_id,
        "props": {"id": doc_id, "status": status, **props},
        "folder": props.pop("folder", None) or "twin",
        "membership_updated_at": "2026-08-04T00:00:00+00:00",
    }


@pytest.fixture
def storage():
    return MemgraphDocStatusStorage(
        namespace="doc_status",
        global_config={"workspace": "actv5ws"},
        embedding_func=None,
    )


class TestStatusTransitions:
    def test_first_insert_to_processed_is_a_transition(self):
        transitions = MemgraphDocStatusStorage._status_transitions(
            [_entry("d1", "processed", file_path="a.pdf")], {"d1": None}
        )
        assert [t["doc_id"] for t in transitions] == ["d1"]
        assert transitions[0]["status"] == "processed"
        assert transitions[0]["file_path"] == "a.pdf"

    def test_processing_to_failed_is_a_transition(self):
        transitions = MemgraphDocStatusStorage._status_transitions(
            [_entry("d1", "failed", error_msg="boom")], {"d1": "processing"}
        )
        assert transitions[0]["status"] == "failed"
        assert transitions[0]["error_msg"] == "boom"

    def test_non_terminal_statuses_never_emit(self):
        entries = [
            _entry("d1", "pending"),
            _entry("d2", "processing"),
        ]
        assert (
            MemgraphDocStatusStorage._status_transitions(
                entries, {"d1": None, "d2": "pending"}
            )
            == []
        )

    def test_re_upsert_of_same_terminal_status_is_silent(self):
        # Backfills and metadata rewrites re-upsert processed rows — the
        # ledger must not gain a duplicate source-ready on each one.
        assert (
            MemgraphDocStatusStorage._status_transitions(
                [_entry("d1", "processed")], {"d1": "processed"}
            )
            == []
        )

    def test_status_comparison_is_case_insensitive(self):
        assert (
            MemgraphDocStatusStorage._status_transitions(
                [_entry("d1", "PROCESSED")], {"d1": "processed"}
            )
            == []
        )
        transitions = MemgraphDocStatusStorage._status_transitions(
            [_entry("d1", "PROCESSED")], {"d1": "PENDING"}
        )
        assert transitions[0]["status"] == "processed"

    def test_reprocess_emits_again(self):
        # failed → processed after an operator retry is a real transition.
        transitions = MemgraphDocStatusStorage._status_transitions(
            [_entry("d1", "processed")], {"d1": "failed"}
        )
        assert transitions[0]["status"] == "processed"


class TestEmitSourceStatusActivity:
    async def test_emits_ready_and_failed_events(self, storage):
        store = AsyncMock()
        with patch(
            "twindb_lightrag_memgraph.server.webui_router.get_store",
            return_value=store,
        ) as get_store:
            await storage._emit_source_status_activity(
                [
                    {
                        "doc_id": "d1",
                        "status": "processed",
                        "folder": "twin",
                        "file_path": "guide.pdf",
                        "track_id": "trk-1",
                        "chunks_count": 12,
                        "error_msg": None,
                    },
                    {
                        "doc_id": "d2",
                        "status": "failed",
                        "folder": "sandbox",
                        "file_path": "broken.pdf",
                        "track_id": "trk-2",
                        "chunks_count": None,
                        "error_msg": "vision-timeout: endpoint unreachable",
                    },
                ]
            )
        assert store.record_activity.await_count == 2
        ready = store.record_activity.await_args_list[0].args[0]
        failed = store.record_activity.await_args_list[1].args[0]

        assert ready["kind"] == "source-ready"
        assert ready["sev"] == "info"
        assert ready["actor"]["user"] == "system"
        assert ready["target"] == {
            "type": "source",
            "label": "guide.pdf",
            "id": "d1",
        }
        assert ready["meta"]["doc_id"] == "d1"
        assert ready["meta"]["track_id"] == "trk-1"
        assert ready["meta"]["folder"] == "twin"

        assert failed["kind"] == "source-failed"
        assert failed["sev"] == "error"
        assert failed["target"]["id"] == "d2"
        assert "vision-timeout" in failed["summary"]
        assert failed["meta"]["error_msg"].startswith("vision-timeout")

        # Each event lands in the store of the doc's own folder.
        assert [c.args[0] for c in get_store.call_args_list] == ["twin", "sandbox"]

    async def test_store_failure_never_raises_into_ingestion(self, storage):
        with patch(
            "twindb_lightrag_memgraph.server.webui_router.get_store",
            side_effect=RuntimeError("overlay not mounted"),
        ):
            await storage._emit_source_status_activity(
                [
                    {
                        "doc_id": "d1",
                        "status": "processed",
                        "folder": None,
                        "file_path": None,
                        "track_id": None,
                        "chunks_count": None,
                        "error_msg": None,
                    }
                ]
            )

    async def test_label_falls_back_to_doc_id(self, storage):
        store = AsyncMock()
        with patch(
            "twindb_lightrag_memgraph.server.webui_router.get_store",
            return_value=store,
        ):
            await storage._emit_source_status_activity(
                [
                    {
                        "doc_id": "doc-42",
                        "status": "processed",
                        "folder": None,
                        "file_path": None,
                        "track_id": None,
                        "chunks_count": None,
                        "error_msg": None,
                    }
                ]
            )
        event = store.record_activity.await_args.args[0]
        assert event["target"]["label"] == "doc-42"
        assert event["meta"]["workspace"] == "actv5ws"
