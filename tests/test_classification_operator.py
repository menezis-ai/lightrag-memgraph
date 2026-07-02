"""Operator-selected MIP classification at upload (PO decision 2026-06-24).

The upload UI lets an operator pick a MIP class per file. It travels as the
``X-Twin-Classification`` header, is bound into the ingestion context
(``operator_classification_context``), and combined with any auto-detected
embedded label by :func:`classification.apply_operator_classification`.

Policy under test — the embedded label is a FLOOR:
  * operator can RAISE the class, or SET one when nothing was detected
    (e.g. a ``.md`` with no embedded MIP label),
  * operator can NEVER downgrade below a detected label,
  * the ceiling (``TWIN_MIP_MAX_CLASSIFICATION``) applies to the resolved class,
  * with no operator choice the auto-detection path is byte-for-byte unchanged
    (LightRAG-compat).
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph._classification_hook import (
    ClassificationRejection,
    classify_for_ingestion,
)
from twindb_lightrag_memgraph._constants import operator_classification_context
from twindb_lightrag_memgraph.classification import (
    ClassificationResult,
    apply_operator_classification,
)

# --------------------------------------------------------------------------
# Policy helper — apply_operator_classification
# --------------------------------------------------------------------------


def test_no_operator_choice_is_noop():
    det = ClassificationResult(class_id="C3", source_format="ooxml")
    assert apply_operator_classification(det, None) is det
    assert apply_operator_classification(det, "") is det


def test_operator_sets_class_when_nothing_detected():
    det = ClassificationResult(class_id=None, reason="no-msip-label")
    out = apply_operator_classification(det, "C2")
    assert out.class_id == "Internal"
    assert out.source_format == "operator"
    assert out.reason == "operator-set"


def test_operator_raises_above_detected():
    det = ClassificationResult(class_id="C2", source_format="ooxml")
    out = apply_operator_classification(det, "C4")
    assert out.class_id == "Secret"
    assert out.source_format == "operator"
    assert out.reason == "operator-raised"
    # provenance of the underlying detected label preserved for audit
    assert out.meta.get("detected_class_id") == "C2"


def test_operator_cannot_downgrade_below_detected_floor():
    det = ClassificationResult(class_id="C3", source_format="ooxml")
    out = apply_operator_classification(det, "C1")
    assert out.class_id == "C3"  # floor holds
    assert out.source_format == "ooxml"  # detected result kept
    assert out.meta.get("operator_requested") == "Public"  # attempt audited


def test_operator_equal_class_keeps_detected_provenance():
    det = ClassificationResult(class_id="C2", source_format="ooxml", label_guid="abc")
    out = apply_operator_classification(det, "C2")
    assert out.class_id == "C2"
    assert out.source_format == "ooxml"
    assert out.label_guid == "abc"


def test_business_name_operator_value_normalises():
    det = ClassificationResult(class_id=None)
    out = apply_operator_classification(det, "Confidential")
    assert out.class_id == "Confidential"
    assert out.source_format == "operator"


def test_unknown_operator_value_is_ignored():
    det = ClassificationResult(class_id="C1", source_format="ooxml")
    assert apply_operator_classification(det, "NOT_A_CLASS") is det


def test_operator_cannot_downgrade_unmapped_detected_label():
    # A label was present but unmapped (UNKNOWN, fail-closed) — the operator
    # must not be able to launder it down to a lower class.
    det = ClassificationResult(class_id="UNKNOWN", source_format="ooxml")
    out = apply_operator_classification(det, "C1")
    assert out.class_id == "UNKNOWN"
    assert out.meta.get("operator_requested") == "Public"


# --------------------------------------------------------------------------
# Hook integration — classify_for_ingestion reads the ContextVar
# --------------------------------------------------------------------------


def _write(tmp_path, name: str, body: str = "hello") -> str:
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_classify_for_ingestion_applies_operator_for_unlabeled_md(tmp_path):
    # .md carries no embedded MIP label -> operator choice is authoritative.
    path = _write(tmp_path, "faq.md", "# FAQ\nhello")
    with operator_classification_context("C2"):
        payload = classify_for_ingestion(path, label_map={}, ceiling="C4")
    assert payload["class_id"] == "Internal"
    assert payload["source_format"] == "operator"


def test_operator_classification_subject_to_ceiling(tmp_path):
    path = _write(tmp_path, "secret.md")
    with operator_classification_context("C4"):  # Secret
        with pytest.raises(ClassificationRejection):
            classify_for_ingestion(path, label_map={}, ceiling="C2")


def test_no_operator_context_leaves_autodetection_untouched(tmp_path):
    # LightRAG-compat: with no operator choice the resolved payload is exactly
    # what auto-detection alone produces — never tagged 'operator'.
    path = _write(tmp_path, "faq.md")
    baseline = classify_for_ingestion(path, label_map={}, ceiling="C4")
    with operator_classification_context(None):
        with_ctx = classify_for_ingestion(path, label_map={}, ceiling="C4")
    assert with_ctx == baseline
    assert baseline["source_format"] != "operator"


def test_garbage_operator_header_falls_back_to_autodetection(tmp_path):
    # An unsafe/garbage header value is dropped by the context manager, so the
    # ingestion path behaves as if no operator choice was made.
    path = _write(tmp_path, "faq.md")
    with operator_classification_context("../etc/passwd"):
        payload = classify_for_ingestion(path, label_map={}, ceiling="C4")
    assert payload["source_format"] != "operator"


# --------------------------------------------------------------------------
# Wiring — middleware header -> ContextVar -> BackgroundTasks propagation
# --------------------------------------------------------------------------


async def test_background_task_reapplies_operator_classification():
    """The classification context survives the BackgroundTasks boundary where
    LightRAG actually writes DocStatus (same mechanism as the folder context)."""
    from starlette.background import BackgroundTasks

    from twindb_lightrag_memgraph import _patch_background_tasks_folder_context
    from twindb_lightrag_memgraph._constants import (
        get_active_operator_classification,
    )

    seen: list[str | None] = []

    async def task():
        seen.append(get_active_operator_classification())

    _patch_background_tasks_folder_context()
    background = BackgroundTasks()
    with operator_classification_context("C2"):
        background.add_task(task)

    assert get_active_operator_classification() is None
    await background()
    assert seen == ["C2"]


async def test_upload_middleware_binds_classification_header(monkeypatch):
    """The ingestion middleware reads ``X-Twin-Classification`` into the
    context the classification hook later reads."""
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"kb"}]',
    )

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from twindb_lightrag_memgraph import _install_storage_folder_capture
    from twindb_lightrag_memgraph._constants import (
        get_active_operator_classification,
    )

    app = FastAPI()
    _install_storage_folder_capture(app)

    @app.post("/documents/upload")
    async def upload_probe():
        return {"classification": get_active_operator_classification()}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        with_header = await client.post(
            "/documents/upload",
            headers={"X-Twin-Folder": "default", "X-Twin-Classification": "C2"},
        )
        no_header = await client.post(
            "/documents/upload", headers={"X-Twin-Folder": "default"}
        )

    assert with_header.json() == {"classification": "C2"}
    # LightRAG-compat: no operator header -> nothing bound, native path.
    assert no_header.json() == {"classification": None}
