"""Tests for the pre-ingestion classification hook."""

from __future__ import annotations

import zipfile
from pathlib import Path
from textwrap import dedent

import pytest

from twindb_lightrag_memgraph._classification_hook import (
    ClassificationRejection,
    _install_classification_metadata_carry_over,
    classify_for_ingestion,
    install_classification_hook,
)

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
          <property fmtid="x" pid="3" name="MSIP_Label_{guid}_SetDate">
            <vt:lpwstr>2026-03-15T10:42:18Z</vt:lpwstr>
          </property>
          <property fmtid="x" pid="4" name="MSIP_Label_{guid}_Method">
            <vt:lpwstr>Standard</vt:lpwstr>
          </property>
        </Properties>
    """).strip()
    target = tmp_path / f"{label_name.replace(' ', '_')}.docx"
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("docProps/custom.xml", custom_xml)
    return target


class TestClassifyForIngestion:
    def test_below_ceiling_passes_and_returns_payload(self, tmp_path):
        path = _build_docx_with_label(tmp_path, "C2 Confidentiel", DEMO_GUID)
        payload = classify_for_ingestion(
            path,
            label_map={DEMO_GUID: "C2"},
            ceiling="C2",
        )
        assert payload["class_id"] == "C2"
        assert payload["label_guid"] == DEMO_GUID

    def test_above_ceiling_raises_rejection(self, tmp_path):
        path = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)
        with pytest.raises(ClassificationRejection) as exc_info:
            classify_for_ingestion(
                path,
                label_map={C3_GUID: "C3"},
                ceiling="C2",
            )
        assert exc_info.value.result.class_id == "C3"
        assert exc_info.value.ceiling == "C2"
        assert "outranks workspace ceiling" in str(exc_info.value)

    def test_unknown_class_treated_as_above_fail_closed(self, tmp_path):
        path = _build_docx_with_label(tmp_path, "Some Label", DEMO_GUID)
        # No tenant map → class resolves to UNKNOWN → fail-closed
        with pytest.raises(ClassificationRejection):
            classify_for_ingestion(path, label_map={}, ceiling="C2")

    def test_unsupported_extension_fails_closed_in_reject_mode(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TWIN_MIP_UNLABELED_POLICY", "reject")
        path = tmp_path / "notes.txt"
        path.write_text("nothing here")
        with pytest.raises(ClassificationRejection) as exc_info:
            classify_for_ingestion(path, label_map={}, ceiling="C2")
        assert exc_info.value.result.class_id is None
        assert "unsupported-extension" in (exc_info.value.result.reason or "")

    def test_stripped_ooxml_label_fails_closed_in_reject_mode(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TWIN_MIP_UNLABELED_POLICY", "reject")
        path = tmp_path / "unlabelled.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", "<document/>")

        with pytest.raises(ClassificationRejection) as exc_info:
            classify_for_ingestion(path, label_map={}, ceiling="C2")
        assert exc_info.value.result.class_id is None
        assert exc_info.value.result.reason == "no-custom-props"

    def test_unsupported_extension_allowed_by_default(self, tmp_path):
        """Permissive default (decision 2026-07-10): an unlabeled format is
        ingested, classification traced as None in the payload."""
        path = tmp_path / "notes.txt"
        path.write_text("nothing here")
        payload = classify_for_ingestion(path, label_map={}, ceiling="C2")
        assert payload["class_id"] is None
        assert "unsupported-extension" in (payload["reason"] or "")

    def test_unlabeled_ooxml_allowed_by_default_emits_detected(self, tmp_path):
        path = tmp_path / "unlabelled.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", "<document/>")
        events = []
        payload = classify_for_ingestion(
            path,
            label_map={},
            ceiling="C2",
            audit_emit=lambda kind, data: events.append(kind),
        )
        assert payload["class_id"] is None
        assert payload["reason"] == "no-custom-props"
        assert events == ["classification-detected"]

    def test_readable_label_above_ceiling_still_rejected_in_allow_mode(
        self, tmp_path, monkeypatch
    ):
        """The permissive policy only covers UNLABELED docs — a readable C3
        label above a C2 ceiling rejects exactly as in tier-1."""
        monkeypatch.setenv("TWIN_MIP_UNLABELED_POLICY", "allow")
        path = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)
        with pytest.raises(ClassificationRejection):
            classify_for_ingestion(path, label_map={C3_GUID: "C3"}, ceiling="C2")

    def test_audit_emit_called_on_detection_and_rejection(self, tmp_path):
        path = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)
        events = []

        def emit(kind, payload):
            events.append((kind, payload))

        with pytest.raises(ClassificationRejection):
            classify_for_ingestion(
                path,
                label_map={C3_GUID: "C3"},
                ceiling="C2",
                audit_emit=emit,
            )
        kinds = [e[0] for e in events]
        assert "classification-detected" in kinds
        assert "classification-rejected" in kinds

    def test_audit_emit_called_only_for_detection_when_below(self, tmp_path):
        path = _build_docx_with_label(tmp_path, "C2 Confidentiel", DEMO_GUID)
        events = []
        classify_for_ingestion(
            path,
            label_map={DEMO_GUID: "C2"},
            ceiling="C2",
            audit_emit=lambda k, p: events.append((k, p)),
        )
        kinds = [e[0] for e in events]
        assert kinds == ["classification-detected"]


class TestInstallClassificationHook:
    def test_returns_callable_that_classifies(self, tmp_path):
        path = _build_docx_with_label(tmp_path, "C2 Confidentiel", DEMO_GUID)
        # Build a label map file the hook will load.
        map_path = tmp_path / "labels.json"
        map_path.write_text(f'{{"{DEMO_GUID}": "C2"}}')

        hook = install_classification_hook(label_map_path=map_path, ceiling="C2")
        payload = hook(str(path))
        assert payload["class_id"] == "C2"

    def test_env_ceiling_used_when_none_provided(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TWIN_MIP_MAX_CLASSIFICATION", "C1")
        map_path = tmp_path / "labels.json"
        map_path.write_text(f'{{"{DEMO_GUID}": "C2"}}')
        hook = install_classification_hook(label_map_path=map_path)
        path = _build_docx_with_label(tmp_path, "C2 Confidentiel", DEMO_GUID)
        # C2 > C1 ceiling → rejection
        with pytest.raises(ClassificationRejection):
            hook(str(path))

    def test_lightrag_hook_preserves_security_metadata_on_1_5(self):
        _install_classification_metadata_carry_over()

        utils_pipeline = pytest.importorskip("lightrag.utils_pipeline")

        carry_over = getattr(utils_pipeline, "_DOC_STATUS_METADATA_CARRY_OVER_KEYS", ())
        if carry_over:
            assert "classification" in carry_over
            assert "classification_rejected" in carry_over
            assert "classification_ceiling" in carry_over
