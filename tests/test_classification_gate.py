"""
Negative-path / enforcement tests for the classification ingestion gate.

``classify_for_ingestion`` is already covered by ``test_classification_hook``.
This module covers the parts that actually *enforce* the compliance ceiling at
ingestion time — ``install_lightrag_ingestion_hook`` and its helpers — which
were previously exercised only through (skipped) integration paths:

  * a file above the ceiling must NEVER reach LightRAG's real insert pipeline;
    instead a FAILED DocStatus row carrying the classification is written;
  * mixed batches must split into accepted (indexed) and rejected (failed);
  * helper functions that build the failed-status row, merge metadata, and
    emit the audit event must behave and must swallow their own errors.

All LightRAG/driver calls are faked — no Memgraph and no real index needed.
"""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

from twindb_lightrag_memgraph import _classification_hook as hook_mod
from twindb_lightrag_memgraph._classification_hook import (
    ClassificationRejection,
    _as_list,
    _doc_id_for_insert,
    _failed_status_for_rejection,
    _merge_classification_metadata,
    classify_for_ingestion,
    install_lightrag_ingestion_hook,
)
from twindb_lightrag_memgraph.classification import ClassificationResult

DEMO_GUID = "11111111-2222-3333-4444-555555555555"
C3_GUID = "33333333-3333-3333-3333-333333333333"


def _build_docx_with_label(tmp_path: Path, label_name: str, guid: str) -> Path:
    custom_xml = dedent(f"""
        <?xml version="1.0"?>
        <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
                    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
          <property fmtid="x" pid="2" name="MSIP_Label_{guid}_Name">
            <vt:lpwstr>{label_name}</vt:lpwstr>
          </property>
        </Properties>
    """).strip()
    target = tmp_path / f"{label_name.replace(' ', '_')}_{guid[:4]}.docx"
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("docProps/custom.xml", custom_xml)
    return target


# ── classifier-never-raises contract ────────────────────────────────────────


def test_classifier_exception_becomes_unknown_extraction_failed(monkeypatch, tmp_path):
    """A classifier that throws must not propagate — it yields an UNKNOWN
    'extraction-failed' payload (which the ceiling then fail-closes on)."""

    def boom(*_a, **_k):
        raise RuntimeError("corrupt zip")

    monkeypatch.setattr(hook_mod, "detect_classification", boom)
    path = tmp_path / "x.docx"
    path.write_text("not really a docx")

    # ceiling high enough that UNKNOWN is the only reason it could reject;
    # use is_above semantics: UNKNOWN is fail-closed → rejection.
    with pytest.raises(ClassificationRejection) as exc_info:
        classify_for_ingestion(path, label_map={}, ceiling="C4")
    assert exc_info.value.result.class_id == "UNKNOWN"
    assert exc_info.value.result.reason.startswith("extraction-failed: RuntimeError")


# ── pure helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [(None, []), ("a", ["a"]), (["a", "b"], ["a", "b"]), (3, [3])],
)
def test_as_list(value, expected):
    assert _as_list(value) == expected


def test_doc_id_for_insert_prefers_explicit_id():
    assert _doc_id_for_insert("some content", "doc-explicit") == "doc-explicit"


def test_doc_id_for_insert_computes_deterministic_id_when_absent():
    a = _doc_id_for_insert("identical content", None)
    b = _doc_id_for_insert("identical content", None)
    assert a == b
    assert a.startswith("doc-")


def test_doc_id_for_insert_matches_active_lightrag_source_id_rule():
    """Blank pending parses never collide; non-blank sources follow the
    active LightRAG generation (content on 1.4.x, canonical path on 1.5.x)."""
    import inspect

    from lightrag import LightRAG

    a = _doc_id_for_insert("", None, file_path="report-a.docx")
    b = _doc_id_for_insert("", None, file_path="report-b.docx")
    assert a != b
    assert a.startswith("doc-") and b.startswith("doc-")

    enqueue = getattr(
        LightRAG,
        "_twin_original_enqueue",
        LightRAG.apipeline_enqueue_documents,
    )
    actual = _doc_id_for_insert("body", None, file_path="report-a.docx")
    if "docs_format" in inspect.signature(enqueue).parameters:
        assert actual == a
    else:
        assert actual == _doc_id_for_insert("body", None)


def test_failed_status_for_rejection_shape():
    from lightrag.base import DocStatus

    result = ClassificationResult(class_id="C3", source_format="ooxml")
    exc = ClassificationRejection("/data/secret.docx", result, "C2")
    status = _failed_status_for_rejection(
        content="hello world",
        file_path="/data/secret.docx",
        track_id="track-1",
        exc=exc,
    )
    assert status["status"] == DocStatus.FAILED
    assert status["content_length"] == len("hello world")
    assert status["chunks_count"] == 0
    assert status["track_id"] == "track-1"
    assert status["metadata"]["classification_rejected"] is True
    assert status["metadata"]["classification_ceiling"] == "C2"
    assert status["metadata"]["classification"]["class_id"] == "C3"


def test_failed_status_for_rejection_redacts_content_summary():
    """PIPE-6b: the FAILED row must not persist an excerpt of the very
    content the gate refused — the summary is a fixed redaction placeholder
    naming the rejected class and the ceiling."""
    result = ClassificationResult(class_id="C3", source_format="ooxml")
    exc = ClassificationRejection("/data/secret.docx", result, "C2")
    over_classified_body = "TOP-SECRET payroll data that must never leak"
    status = _failed_status_for_rejection(
        content=over_classified_body,
        file_path="/data/secret.docx",
        track_id="track-1",
        exc=exc,
    )
    assert over_classified_body not in status["content_summary"]
    assert "payroll" not in status["content_summary"]
    assert status["content_summary"] == (
        "[content withheld: classification C3 exceeds ceiling C2]"
    )
    # Length metadata (a size, not content) is preserved for operators.
    assert status["content_length"] == len(over_classified_body)


# ── metadata merge (best-effort, swallows errors) ───────────────────────────


async def test_merge_classification_metadata_updates_existing_doc():
    rag = MagicMock()
    rag.doc_status.get_by_id = AsyncMock(
        return_value={"id": "doc-1", "metadata": {"existing": True}}
    )
    rag.doc_status.upsert = AsyncMock()

    await _merge_classification_metadata(rag, "doc-1", {"class_id": "C2"})

    upserted = rag.doc_status.upsert.call_args.args[0]
    assert upserted["doc-1"]["metadata"]["classification"] == {"class_id": "C2"}
    assert upserted["doc-1"]["metadata"]["existing"] is True  # preserved


async def test_merge_classification_metadata_noop_when_doc_missing():
    rag = MagicMock()
    rag.doc_status.get_by_id = AsyncMock(return_value=None)
    rag.doc_status.upsert = AsyncMock()

    await _merge_classification_metadata(rag, "missing", {"class_id": "C2"})

    rag.doc_status.upsert.assert_not_called()


async def test_merge_classification_metadata_swallows_errors():
    rag = MagicMock()
    rag.doc_status.get_by_id = AsyncMock(side_effect=RuntimeError("db down"))
    # Must not raise into the ingestion path.
    await _merge_classification_metadata(rag, "doc-1", {"class_id": "C2"})


async def test_merge_classification_metadata_batch_reads_and_writes_once():
    rag = MagicMock()
    rag.doc_status.get_by_ids = AsyncMock(
        return_value=[
            {"id": "doc-1", "metadata": {"existing": 1}},
            {"id": "doc-2", "metadata": {"existing": 2}},
        ]
    )
    rag.doc_status.get_by_id = AsyncMock()
    rag.doc_status.upsert = AsyncMock()

    await hook_mod._merge_classification_metadata_batch(
        rag,
        {
            "doc-1": {"class_id": "C1"},
            "doc-2": {"class_id": "C2"},
        },
    )

    rag.doc_status.get_by_ids.assert_awaited_once_with(["doc-1", "doc-2"])
    rag.doc_status.get_by_id.assert_not_awaited()
    rag.doc_status.upsert.assert_awaited_once()
    updates = rag.doc_status.upsert.call_args.args[0]
    assert updates["doc-1"]["metadata"] == {
        "existing": 1,
        "classification": {"class_id": "C1"},
    }
    assert updates["doc-2"]["metadata"] == {
        "existing": 2,
        "classification": {"class_id": "C2"},
    }


async def test_merge_classification_metadata_batch_falls_back_after_read_error():
    rag = MagicMock()
    rag.doc_status.get_by_ids = AsyncMock(side_effect=RuntimeError("batch down"))
    rag.doc_status.get_by_id = AsyncMock(
        side_effect=[
            {"id": "doc-1", "metadata": {}},
            {"id": "doc-2", "metadata": {}},
        ]
    )
    rag.doc_status.upsert = AsyncMock()

    await hook_mod._merge_classification_metadata_batch(
        rag,
        {
            "doc-1": {"class_id": "C1"},
            "doc-2": {"class_id": "C2"},
        },
    )

    assert rag.doc_status.get_by_id.await_count == 2
    assert rag.doc_status.upsert.await_count == 2


async def test_partition_inputs_async_offloads_and_copies_context():
    import contextvars
    import threading

    marker = contextvars.ContextVar("classification_test_marker", default="missing")
    running_loop = asyncio.get_running_loop()
    main_thread = threading.get_ident()
    probe_threads: list[int] = []
    evaluation_threads: list[int] = []
    evaluation_loops: list[asyncio.AbstractEventLoop] = []

    def probe(path: str) -> dict[str, str]:
        assert marker.get() == "copied"
        probe_threads.append(threading.get_ident())
        return {"class_id": "C2", "path": path}

    def evaluate(path: str, result: dict[str, str]) -> dict[str, str]:
        assert result["path"] == path
        evaluation_threads.append(threading.get_ident())
        evaluation_loops.append(asyncio.get_running_loop())
        return result

    def active_hook(path: str) -> dict[str, str]:
        return evaluate(path, probe(path))

    setattr(active_hook, "_twin_classification_probe", probe)
    setattr(active_hook, "_twin_classification_evaluate", evaluate)

    token = marker.set("copied")
    try:
        accepted, rejected = await hook_mod._partition_inputs_async(
            active_hook, ["first.docx", "second.docx"]
        )
    finally:
        marker.reset(token)

    assert [index for index, _ in accepted] == [0, 1]
    assert rejected == []
    assert probe_threads
    assert all(thread_id != main_thread for thread_id in probe_threads)
    assert evaluation_threads == [main_thread, main_thread]
    assert evaluation_loops == [running_loop, running_loop]


async def test_emit_rejection_event_swallows_errors(monkeypatch):
    # Force the activity store lookup to explode; the emitter must absorb it.
    import twindb_lightrag_memgraph.server.webui_router as wr

    monkeypatch.setattr(
        wr, "get_store", lambda: (_ for _ in ()).throw(RuntimeError("no store"))
    )
    result = ClassificationResult(class_id="C3", source_format="ooxml")
    exc = ClassificationRejection("/p.docx", result, "C2")
    # Should complete without raising.
    await hook_mod._emit_rejection_event(
        actor="system", doc_id="doc-1", file_path="/p.docx", exc=exc
    )


# ── the enforcement gate (patched LightRAG.ainsert) ─────────────────────────


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """Install the ingestion gate over a fake LightRAG and yield a driver.

    The fake captures whatever reaches the *real* insert pipeline so tests can
    assert exactly which documents were (or were not) indexed.
    """
    from lightrag import LightRAG

    # Snapshot the class attributes the gate mutates, restore them after.
    saved = {
        name: getattr(LightRAG, name, None)
        for name in (
            "ainsert",
            "apipeline_enqueue_documents",
            "_twin_classification_hook",
            "_twin_classification_patched",
            "_twin_original_ainsert",
            "_twin_original_enqueue",
            "_twin_enqueue_patched",
        )
    }

    indexed: list[dict] = []

    async def fake_original(
        self,
        input,
        split_by_character=None,
        split_by_character_only=False,
        ids=None,
        file_paths=None,
        track_id=None,
    ):
        indexed.append({"input": input, "ids": ids, "file_paths": file_paths})
        return track_id or "track-original"

    LightRAG.ainsert = fake_original
    LightRAG._twin_classification_patched = False

    map_path = tmp_path / "labels.json"
    map_path.write_text(f'{{"{DEMO_GUID}": "C2", "{C3_GUID}": "C3"}}')
    install_lightrag_ingestion_hook(label_map_path=map_path, ceiling="C2")

    class FakeRAG(LightRAG):
        def __init__(self):
            self.doc_status = MagicMock()
            self.doc_status.get_by_id = AsyncMock(return_value=None)
            self.doc_status.upsert = AsyncMock()

    rag = FakeRAG()
    try:
        yield rag, indexed
    finally:
        for name, value in saved.items():
            if value is None:
                if hasattr(LightRAG, name):
                    delattr(LightRAG, name)
            else:
                setattr(LightRAG, name, value)
        # The gate emits classification-rejected events into the
        # module-singleton activity store; reset it so later test modules
        # (e.g. test_server/test_classification_rejection.py) don't inherit
        # this module's events (audit 2026-07-02 addendum, finding A).
        from twindb_lightrag_memgraph.server import webui_router

        webui_router.reset_store()


async def test_gate_blocks_above_ceiling_file(gate, tmp_path):
    rag, indexed = gate
    secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

    track = await rag.ainsert("body text", file_paths=str(secret))

    # The rejected document must NEVER reach the real index.
    assert indexed == []
    # A FAILED DocStatus row carrying the classification is written instead.
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification_rejected"] is True
    assert row["metadata"]["classification"]["class_id"] == "C3"
    assert isinstance(track, str)


async def test_gate_allows_below_ceiling_file(gate, tmp_path):
    rag, indexed = gate
    ok = _build_docx_with_label(tmp_path, "C2 Confidentiel", DEMO_GUID)

    await rag.ainsert("body text", file_paths=str(ok))

    # The accepted document reaches the real index untouched.
    assert len(indexed) == 1
    assert indexed[0]["input"] == "body text"


async def test_gate_splits_mixed_batch(gate, tmp_path):
    rag, indexed = gate
    ok = _build_docx_with_label(tmp_path, "C2 Confidentiel", DEMO_GUID)
    secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

    await rag.ainsert(
        ["clean doc", "secret doc"],
        file_paths=[str(ok), str(secret)],
    )

    # Only the accepted doc is indexed; the rejected one is written as FAILED.
    assert len(indexed) == 1
    assert indexed[0]["input"] == ["clean doc"]
    failed = rag.doc_status.upsert.call_args.args[0]
    assert len(failed) == 1


async def test_gate_rejects_when_no_file_paths(gate):
    rag, indexed = gate
    track = await rag.ainsert("in-memory text only")

    assert indexed == []
    assert isinstance(track, str) and track
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification"]["class_id"] == "UNKNOWN"
    assert row["metadata"]["classification"]["reason"] == "source-file-required"


async def test_gate_rejects_mismatched_path_count(gate):
    rag, _indexed = gate
    with pytest.raises(ValueError, match="file paths must match"):
        await rag.ainsert(["a", "b"], file_paths=["only-one-path"])
