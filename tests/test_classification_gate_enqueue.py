"""
Enqueue-level classification gate (audit 2026-07-02, finding ING-1).

The native HTTP ingestion routes (``POST /documents/upload``, ``/text``,
``/texts``, ``/scan``) call ``rag.apipeline_enqueue_documents`` +
``rag.apipeline_process_enqueue_documents`` and never ``ainsert`` — so the
historical ainsert-only MIP gate never fired on the primary ingestion
surface. This module covers the enqueue-level gate:

  * unit: gate fires when ``apipeline_enqueue_documents`` is called directly
    (over-classified → FAILED row + event + not enqueued; under-ceiling →
    enqueued untouched; missing classification → rejected; detector crash →
    UNKNOWN → fail-closed rejection, parity with the ainsert gate);
  * no-double-event: the ainsert path (which internally calls enqueue on
    every supported LightRAG) classifies and emits exactly once;
  * compat (docs/test-doctrine-lightrag-compat.md): hook absent → the
    patched enqueue forwards verbatim; a LightRAG without the enqueue
    method keeps ainsert-only gating and install never raises;
  * integration (real Memgraph + native routes): HTTP upload of an
    over-classified document → observable rejection (FAILED + event + zero
    chunks), and the FAILED rejection row survives the pipeline's
    consistency pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _classification_hook as hook_mod
from twindb_lightrag_memgraph._classification_hook import (
    _resolve_detection_path,
    install_lightrag_ingestion_hook,
)
from twindb_lightrag_memgraph.classification import ClassificationResult

twindb_lightrag_memgraph.register()

C2_GUID = "11111111-2222-3333-4444-555555555555"
C3_GUID = "33333333-3333-3333-3333-333333333333"

_GATE_ATTRS = (
    "ainsert",
    "apipeline_enqueue_documents",
    "_twin_classification_hook",
    "_twin_classification_patched",
    "_twin_original_ainsert",
    "_twin_original_enqueue",
    "_twin_enqueue_patched",
)


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


def _snapshot_gate_attrs(cls) -> dict:
    return {name: getattr(cls, name, None) for name in _GATE_ATTRS}


def _restore_gate_attrs(cls, saved: dict) -> None:
    for name, value in saved.items():
        if value is None:
            if hasattr(cls, name):
                try:
                    delattr(cls, name)
                except AttributeError:
                    # Attribute lives on a parent/mixin (e.g. 1.5.x
                    # _PipelineMixin methods) — nothing to remove here.
                    pass
        else:
            setattr(cls, name, value)


def _reset_activity_store() -> None:
    from twindb_lightrag_memgraph.server import webui_router

    webui_router.reset_store()


async def _rejected_events() -> list[dict]:
    from twindb_lightrag_memgraph.server import webui_router

    events, _, _ = await webui_router.get_store().list_activity()
    return [e for e in events if e["kind"] == "classification-rejected"]


# ── unit: the enqueue-level gate over a fake original enqueue ────────────────


@pytest.fixture
def enqueue_gate(tmp_path):
    """Install the gate with a fake original enqueue; yield (rag, enqueued)."""
    from lightrag import LightRAG

    saved = _snapshot_gate_attrs(LightRAG)
    _reset_activity_store()

    enqueued: list[dict] = []

    async def fake_original_enqueue(
        self, input, ids=None, file_paths=None, track_id=None, *args, **kwargs
    ):
        enqueued.append(
            {
                "input": input,
                "ids": ids,
                "file_paths": file_paths,
                "track_id": track_id,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return track_id or "track-enqueue-original"

    LightRAG.apipeline_enqueue_documents = fake_original_enqueue
    LightRAG._twin_classification_patched = False
    LightRAG._twin_enqueue_patched = False

    map_path = tmp_path / "labels.json"
    map_path.write_text(f'{{"{C2_GUID}": "C2", "{C3_GUID}": "C3"}}')
    install_lightrag_ingestion_hook(label_map_path=map_path, ceiling="C2")

    class FakeRAG(LightRAG):
        def __init__(self):
            self.doc_status = MagicMock()
            self.doc_status.get_by_id = AsyncMock(return_value=None)
            self.doc_status.upsert = AsyncMock()

    try:
        yield FakeRAG(), enqueued
    finally:
        _restore_gate_attrs(LightRAG, saved)
        _reset_activity_store()


async def test_enqueue_gate_blocks_above_ceiling(enqueue_gate, tmp_path):
    rag, enqueued = enqueue_gate
    secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

    track = await rag.apipeline_enqueue_documents("secret body", file_paths=str(secret))

    # Never handed to the real enqueue → no full_docs, no PENDING row.
    assert enqueued == []
    assert isinstance(track, str) and track
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification_rejected"] is True
    assert row["metadata"]["classification"]["class_id"] == "C3"
    # PIPE-6b: no excerpt of the rejected content in the persisted row.
    assert "secret body" not in row["content_summary"]
    assert "content withheld" in row["content_summary"]
    assert len(await _rejected_events()) == 1


async def test_enqueue_gate_allows_below_ceiling_untouched(enqueue_gate, tmp_path):
    rag, enqueued = enqueue_gate
    ok = _build_docx_with_label(tmp_path, "C2 Confidentiel", C2_GUID)
    rag.doc_status.get_by_id = AsyncMock(
        return_value={"id": "whatever", "metadata": {}}
    )

    await rag.apipeline_enqueue_documents(
        "body text", file_paths=str(ok), track_id="track-ok"
    )

    # Exact original call — input/ids/file_paths/track_id untouched.
    assert enqueued == [
        {
            "input": "body text",
            "ids": None,
            "file_paths": str(ok),
            "track_id": "track-ok",
            "args": (),
            "kwargs": {},
        }
    ]
    # Classification metadata merged onto the (existing) DocStatus row.
    merged = rag.doc_status.upsert.call_args.args[0]
    (row,) = merged.values()
    assert row["metadata"]["classification"]["class_id"] == "C2"
    assert len(await _rejected_events()) == 0


async def test_enqueue_gate_mixed_batch_slices_aligned_extras(enqueue_gate, tmp_path):
    rag, enqueued = enqueue_gate
    ok = _build_docx_with_label(tmp_path, "C2 Confidentiel", C2_GUID)
    secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

    await rag.apipeline_enqueue_documents(
        ["clean doc", "secret doc"],
        None,
        [str(ok), str(secret)],
        "track-mixed",
        process_options=["F", "R"],  # per-doc aligned list (1.5.x shape)
        docs_format="raw",  # scalar broadcast — must pass through
    )

    assert len(enqueued) == 1
    call = enqueued[0]
    assert call["input"] == ["clean doc"]
    assert call["file_paths"] == [str(ok)]
    assert call["track_id"] == "track-mixed"
    assert call["kwargs"]["process_options"] == ["F"]  # sliced with the batch
    assert call["kwargs"]["docs_format"] == "raw"  # scalar untouched
    failed = rag.doc_status.upsert.call_args_list[0].args[0]
    assert len(failed) == 1
    (row,) = failed.values()
    assert row["file_path"] == str(secret)
    assert row["track_id"] == "track-mixed"
    assert len(await _rejected_events()) == 1


async def test_enqueue_gate_all_rejected_returns_track_id(enqueue_gate, tmp_path):
    rag, enqueued = enqueue_gate
    secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

    track = await rag.apipeline_enqueue_documents(
        ["secret one", "secret two"],
        file_paths=[str(secret), str(secret)],
        track_id="track-all-rejected",
    )

    assert track == "track-all-rejected"
    assert enqueued == []


async def test_enqueue_gate_rejects_when_no_file_paths(enqueue_gate, monkeypatch):
    rag, enqueued = enqueue_gate
    calls: list[str] = []

    def counting_detect(path, *, label_map=None):
        calls.append(str(path))
        return ClassificationResult()

    monkeypatch.setattr(hook_mod, "detect_classification", counting_detect)

    track = await rag.apipeline_enqueue_documents("in-memory text only")

    assert enqueued == []
    assert calls == []  # nothing to probe → no classification at all
    assert isinstance(track, str) and track
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification"]["reason"] == "source-file-required"


async def test_enqueue_gate_graceful_nondetection_fails_closed(
    enqueue_gate, tmp_path, monkeypatch
):
    monkeypatch.setenv("TWIN_MIP_UNLABELED_POLICY", "reject")
    rag, enqueued = enqueue_gate
    plain = tmp_path / "notes.txt"
    plain.write_text("unlabeled plain text")

    await rag.apipeline_enqueue_documents("body", file_paths=str(plain))

    assert enqueued == []
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification"]["class_id"] is None
    assert "unsupported-extension" in row["metadata"]["classification"]["reason"]
    assert len(await _rejected_events()) == 1


async def test_enqueue_gate_unlabeled_passes_under_default_policy(
    enqueue_gate, tmp_path
):
    """Permissive default (decision 2026-07-10): an unlabeled format flows
    through the gate; the classification payload traces class None."""
    rag, enqueued = enqueue_gate
    plain = tmp_path / "notes.txt"
    plain.write_text("unlabeled plain text")

    await rag.apipeline_enqueue_documents("body", file_paths=str(plain))

    assert len(enqueued) == 1
    assert await _rejected_events() == []


async def test_enqueue_gate_detector_crash_is_fail_closed(
    enqueue_gate, tmp_path, monkeypatch
):
    """Parity with the ainsert gate: a detector that *raises* yields
    class_id='UNKNOWN', which ``is_above`` fail-closes on."""
    rag, enqueued = enqueue_gate

    def boom(*_a, **_k):
        raise RuntimeError("corrupt container")

    monkeypatch.setattr(hook_mod, "detect_classification", boom)
    path = tmp_path / "x.docx"
    path.write_text("not really a docx")

    await rag.apipeline_enqueue_documents("body", file_paths=str(path))

    assert enqueued == []
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification"]["class_id"] == "UNKNOWN"
    assert row["metadata"]["classification_rejected"] is True


async def test_enqueue_gate_resolves_bare_filename_against_input_dir(
    enqueue_gate, tmp_path, monkeypatch
):
    """Native routes enqueue the bare file name (1.4.9.11
    document_routes.py:1563) while the binary sits in INPUT_DIR."""
    rag, enqueued = enqueue_gate
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    secret = _build_docx_with_label(input_dir, "C3 Strict", C3_GUID)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    await rag.apipeline_enqueue_documents(
        "secret body", file_paths=secret.name, track_id="track-bare"
    )

    assert enqueued == []
    failed = rag.doc_status.upsert.call_args.args[0]
    (row,) = failed.values()
    assert row["metadata"]["classification"]["class_id"] == "C3"
    # The raw (bare) path is what lands in the row — resolution is
    # detection-only, DocStatus keeps LightRAG's own file_path value.
    assert row["file_path"] == secret.name


def test_resolve_detection_path_confined_to_input_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "present.docx").write_bytes(b"x")
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    resolved = _resolve_detection_path("present.docx")
    assert Path(resolved) == (input_dir / "present.docx").resolve()

    # Missing file → raw string back (graceful no-label downstream).
    assert _resolve_detection_path("absent.docx") == "absent.docx"
    # Traversal out of the input tree is refused.
    outside = "../" + Path(tmp_path).name + "/escape.docx"
    assert _resolve_detection_path(outside) == outside
    # Absolute paths are untouched.
    absolute = str(tmp_path / "abs.docx")
    assert _resolve_detection_path(absolute) == absolute


# ── no-double-event: the ainsert path classifies exactly once ────────────────


async def test_no_double_event_or_double_detection_on_ainsert_path(
    tmp_path, monkeypatch
):
    """``ainsert`` internally calls ``apipeline_enqueue_documents`` (1.4.9.11
    lightrag.py:1178, 1.5.4 lightrag.py:1486). With both patch points
    installed, a rejected batch must produce ONE detection pass and ONE
    ``classification-rejected`` event — the enqueue gate passes through
    under the ainsert gate."""
    from lightrag import LightRAG

    saved = _snapshot_gate_attrs(LightRAG)
    _reset_activity_store()

    enqueued: list[dict] = []
    detect_calls: list[str] = []

    async def fake_original_enqueue(
        self, input, ids=None, file_paths=None, track_id=None, *args, **kwargs
    ):
        enqueued.append({"input": input, "file_paths": file_paths})
        return track_id or "track-enqueue-original"

    async def fake_original_ainsert(
        self,
        input,
        split_by_character=None,
        split_by_character_only=False,
        ids=None,
        file_paths=None,
        track_id=None,
    ):
        # Mimic the real ainsert: route everything through self.enqueue —
        # which is the PATCHED enqueue after install.
        await self.apipeline_enqueue_documents(input, ids, file_paths, track_id)
        return track_id or "track-ainsert-original"

    real_detect = hook_mod.detect_classification

    def counting_detect(path, *, label_map=None):
        detect_calls.append(str(path))
        return real_detect(path, label_map=label_map)

    try:
        LightRAG.ainsert = fake_original_ainsert
        LightRAG.apipeline_enqueue_documents = fake_original_enqueue
        LightRAG._twin_classification_patched = False
        LightRAG._twin_enqueue_patched = False
        monkeypatch.setattr(hook_mod, "detect_classification", counting_detect)

        map_path = tmp_path / "labels.json"
        map_path.write_text(f'{{"{C2_GUID}": "C2", "{C3_GUID}": "C3"}}')
        install_lightrag_ingestion_hook(label_map_path=map_path, ceiling="C2")

        class FakeRAG(LightRAG):
            def __init__(self):
                self.doc_status = MagicMock()
                self.doc_status.get_by_id = AsyncMock(return_value=None)
                self.doc_status.upsert = AsyncMock()

        rag = FakeRAG()
        ok = _build_docx_with_label(tmp_path, "C2 Confidentiel", C2_GUID)
        secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

        await rag.ainsert(
            ["clean doc", "secret doc"], file_paths=[str(ok), str(secret)]
        )

        # One detection per document — not per patch layer.
        assert len(detect_calls) == 2
        # Exactly one rejection event for the C3 doc.
        assert len(await _rejected_events()) == 1
        # The accepted subset flowed ainsert → original ainsert → patched
        # enqueue (passthrough) → original enqueue, exactly once.
        assert enqueued == [{"input": ["clean doc"], "file_paths": [str(ok)]}]
        # And exactly one FAILED upsert (the rejection), not two.
        assert rag.doc_status.upsert.call_count == 1
    finally:
        _restore_gate_attrs(LightRAG, saved)
        _reset_activity_store()


# ── LightRAG-native compat (docs/test-doctrine-lightrag-compat.md) ───────────


async def test_stale_patch_without_hook_is_verbatim_passthrough(
    enqueue_gate, tmp_path, monkeypatch
):
    """Hook uninstalled (flag-off equivalent): the patched enqueue forwards
    the exact native call — no detection, no writes, no events."""
    from lightrag import LightRAG

    rag, enqueued = enqueue_gate
    monkeypatch.delattr(LightRAG, "_twin_classification_hook", raising=False)

    calls: list[str] = []

    def counting_detect(path, *, label_map=None):
        calls.append(str(path))
        return ClassificationResult()

    monkeypatch.setattr(hook_mod, "detect_classification", counting_detect)
    secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)

    result = await rag.apipeline_enqueue_documents(
        "secret body",
        file_paths=str(secret),
        track_id="track-native",
        docs_format="raw",
    )

    assert result == "track-native"
    assert enqueued == [
        {
            "input": "secret body",
            "ids": None,
            "file_paths": str(secret),
            "track_id": "track-native",
            "args": (),
            "kwargs": {"docs_format": "raw"},
        }
    ]
    assert calls == []
    rag.doc_status.upsert.assert_not_called()
    assert len(await _rejected_events()) == 0


async def test_install_survives_lightrag_without_enqueue(tmp_path, caplog):
    """Version-skew fallback: no ``apipeline_enqueue_documents`` → install
    never raises, warns, and the ainsert gate still enforces the ceiling."""
    from lightrag import LightRAG

    saved = _snapshot_gate_attrs(LightRAG)
    try:
        indexed: list[dict] = []

        async def fake_original_ainsert(
            self,
            input,
            split_by_character=None,
            split_by_character_only=False,
            ids=None,
            file_paths=None,
            track_id=None,
        ):
            indexed.append({"input": input})
            return track_id or "track-original"

        LightRAG.ainsert = fake_original_ainsert
        LightRAG.apipeline_enqueue_documents = None  # simulate absence
        LightRAG._twin_classification_patched = False
        LightRAG._twin_enqueue_patched = False

        map_path = tmp_path / "labels.json"
        map_path.write_text(f'{{"{C3_GUID}": "C3"}}')
        with caplog.at_level("WARNING", logger="twin.classification.hook"):
            install_lightrag_ingestion_hook(label_map_path=map_path, ceiling="C2")
        assert any(
            "apipeline_enqueue_documents not found" in rec.message
            for rec in caplog.records
        )
        assert not getattr(LightRAG, "_twin_enqueue_patched", False)

        class FakeRAG(LightRAG):
            def __init__(self):
                self.doc_status = MagicMock()
                self.doc_status.get_by_id = AsyncMock(return_value=None)
                self.doc_status.upsert = AsyncMock()

        rag = FakeRAG()
        secret = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)
        await rag.ainsert("secret body", file_paths=str(secret))

        assert indexed == []  # ainsert gate still blocks
        failed = rag.doc_status.upsert.call_args.args[0]
        (row,) = failed.values()
        assert row["metadata"]["classification_rejected"] is True
    finally:
        _restore_gate_attrs(LightRAG, saved)
        _reset_activity_store()


# ── integration: native HTTP surface against real Memgraph ──────────────────

EMBEDDING_DIM = 384


async def _mock_embedding(texts: list[str]):
    import hashlib

    import numpy as np

    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
            :EMBEDDING_DIM
        ].astype(np.float32)
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm else vec)
    return np.array(vectors)


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
    if "entity_types" in prompt_lower or "extract" in prompt_lower:
        return "\n".join(
            [
                "entity<|#|>Atlas<|#|>service<|#|>Atlas is an internal service.",
                "<|COMPLETE|>",
            ]
        )
    return "Internal services summary."


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def gate_runtime_dirs():
    root = tempfile.mkdtemp(prefix="clsgate_runtime_")
    yield {
        "working": os.path.join(root, "work"),
        "input": os.path.join(root, "input"),
    }
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
async def gate_native_runtime(monkeypatch, gate_runtime_dirs, tmp_path):
    """Native document routes + real Memgraph + the classification gate.

    Mirrors the harness of tests/test_native_document_runtime.py (do not
    import from it — it is owned by another workstream); uuid-suffixed
    workspace because the Memgraph instance is shared.
    """
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )
    from lightrag.utils import EmbeddingFunc, Tokenizer

    from twindb_lightrag_memgraph import _install_storage_folder_capture, _pool

    workspace = f"clsgate_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", workspace)
    monkeypatch.setenv("INPUT_DIR", gate_runtime_dirs["input"])
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"primary"}]',
    )

    saved = _snapshot_gate_attrs(LightRAG)
    _reset_activity_store()

    label_map = tmp_path / "labels.json"
    label_map.write_text(f'{{"{C2_GUID}": "C2", "{C3_GUID}": "C3"}}')
    install_lightrag_ingestion_hook(label_map_path=label_map, ceiling="C2")

    finalize_share_data()
    initialize_share_data()
    await _cleanup_workspace(_pool, workspace)

    rag = LightRAG(
        working_dir=gate_runtime_dirs["working"],
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=workspace,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_mock_embedding,
        ),
        llm_model_func=_mock_llm,
        enable_llm_cache=False,
        chunk_token_size=120,
        chunk_overlap_token_size=20,
        tokenizer=Tokenizer("clsgate-char", _CharTokenizer()),
    )
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass

    document_routes = pytest.importorskip(
        "lightrag.api.routers.document_routes",
        reason="native document routes unavailable on this LightRAG",
    )
    from fastapi import FastAPI

    from tests.conftest import ensure_fresh_native_document_router

    doc_manager = document_routes.DocumentManager(
        gate_runtime_dirs["input"], workspace=workspace
    )
    app = FastAPI()
    _install_storage_folder_capture(app)
    ensure_fresh_native_document_router()
    app.include_router(
        document_routes.create_document_routes(rag, doc_manager, api_key=None)
    )

    try:
        yield rag, app, workspace
    finally:
        await _cleanup_workspace(_pool, workspace)
        await rag.finalize_storages()
        _restore_gate_attrs(LightRAG, saved)
        _reset_activity_store()


async def _cleanup_workspace(pool, workspace: str) -> None:
    try:
        async with pool.get_session() as session:
            for prefix in ("KV_", "Vec_", "DocStatus_", "Folder_"):
                result = await session.run(
                    "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $label) "
                    "DETACH DELETE n",
                    label=f"{prefix}{workspace}",
                )
                await result.consume()
            result = await session.run(f"MATCH (n:`{workspace}`) DETACH DELETE n")
            await result.consume()
    except Exception:
        pass


async def _count_vec_nodes(pool, workspace: str) -> int:
    async with pool.get_read_session() as session:
        result = await session.run(
            "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $prefix) "
            "RETURN count(n) AS count",
            prefix=f"Vec_{workspace}",
        )
        record = await result.single()
        await result.consume()
        return record["count"] if record else 0


async def _poll_terminal_track(client, track_id: str) -> dict:
    """Poll track_status until every doc reaches a terminal state."""
    payload: dict = {}
    for _ in range(50):
        response = await client.get(f"/documents/track_status/{track_id}")
        if response.status_code == 404:
            pytest.skip("track_status route unavailable on this LightRAG")
        assert response.status_code == 200, response.text
        payload = response.json()
        documents = payload.get("documents", [])
        if documents and all(
            str(doc["status"]).lower() in ("processed", "failed") for doc in documents
        ):
            return payload
        await asyncio.sleep(0.1)
    return payload


def _row_field(row, field: str):
    return row[field] if isinstance(row, dict) else getattr(row, field)


def _row_metadata(row) -> dict:
    meta = _row_field(row, "metadata")
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta or {}


@pytest.mark.integration
async def test_http_upload_over_classified_operator_header_is_rejected(
    gate_native_runtime,
):
    """Operator upload sensitivity is categorically limited to C1/C2."""
    import httpx

    from twindb_lightrag_memgraph import _pool

    _rag, app, workspace = gate_native_runtime

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/documents/upload",
            files={
                "file": (
                    "over-classified-upload.txt",
                    b"Highly sensitive operator-classified content.",
                    "text/plain",
                )
            },
            headers={
                "X-Twin-Folder": "default",
                "X-Twin-Classification": "C4",
            },
        )
        if response.status_code in (404, 405):
            pytest.skip("native /documents/upload route unavailable")
        assert response.status_code == 400, response.text
        assert "accepts only C1 or C2" in response.json()["detail"]

    assert await _count_vec_nodes(_pool, workspace) == 0
    assert await _rejected_events() == []


@pytest.mark.integration
async def test_direct_enqueue_mixed_batch_and_failed_row_survives_pipeline(
    gate_native_runtime,
):
    """Embedded-label detection through the real enqueue + the preserve
    invariant: a rejected FAILED row (no full_docs content) must survive
    ``apipeline_process_enqueue_documents`` untouched, while the accepted
    doc processes normally."""
    from twindb_lightrag_memgraph import _pool

    rag, _app, workspace = gate_native_runtime

    input_dir = Path(os.environ["INPUT_DIR"])
    input_dir.mkdir(parents=True, exist_ok=True)
    accepted = _build_docx_with_label(input_dir, "C2 Confidentiel", C2_GUID)
    secret = _build_docx_with_label(input_dir, "C3 Strict", C3_GUID)

    clean_content = (
        "Clean enqueue coverage document. Atlas is an internal service with "
        "enough content for the chunker and extractor to persist real state."
    )
    track_id = f"clsgate-mixed-{uuid.uuid4().hex[:6]}"

    await rag.apipeline_enqueue_documents(
        [clean_content, "embedded secret body"],
        file_paths=[accepted.name, secret.name],
        track_id=track_id,
    )

    # The rejected doc: FAILED row present, content never stored.
    from twindb_lightrag_memgraph._classification_hook import _doc_id_for_insert

    rejected_id = _doc_id_for_insert("embedded secret body", None)
    rejected_row = await rag.doc_status.get_by_id(rejected_id)
    assert rejected_row is not None
    assert str(_row_field(rejected_row, "status")).lower().endswith("failed")
    metadata = _row_metadata(rejected_row)
    assert metadata["classification"]["class_id"] == "C3"
    assert await rag.full_docs.get_by_id(rejected_id) is None

    # The accepted doc: enqueued with content persisted. Its doc id is
    # version-dependent (1.4.9.11 keys on content, 1.5.x keys known-source
    # docs on the canonical path) — locate it via the shared track_id.
    if not hasattr(rag.doc_status, "get_docs_by_track_id"):
        pytest.skip("doc_status lacks get_docs_by_track_id on this LightRAG")
    tracked = await rag.doc_status.get_docs_by_track_id(track_id)
    accepted_ids = [d_id for d_id in tracked if d_id != rejected_id]
    assert len(accepted_ids) == 1, f"expected 1 accepted doc, got {tracked.keys()}"
    accepted_id = accepted_ids[0]
    accepted_row = await rag.doc_status.get_by_id(accepted_id)
    assert accepted_row is not None
    assert await rag.full_docs.get_by_id(accepted_id) is not None

    # Run the real pipeline: the rejection row must be PRESERVED (it has no
    # full_docs content, so the consistency pass must not reset/clear it).
    await rag.apipeline_process_enqueue_documents()

    rejected_after = await rag.doc_status.get_by_id(rejected_id)
    assert rejected_after is not None
    assert str(_row_field(rejected_after, "status")).lower().endswith("failed")
    metadata_after = _row_metadata(rejected_after)
    assert metadata_after.get("classification_rejected") is True
    assert metadata_after["classification"]["class_id"] == "C3"

    accepted_after = await rag.doc_status.get_by_id(accepted_id)
    assert str(_row_field(accepted_after, "status")).lower().endswith("processed")
    assert await _count_vec_nodes(_pool, workspace) > 0

    events = await _rejected_events()
    assert len(events) == 1
