"""
Tests for the MSIP classification extractor.

We do NOT depend on real Office documents — we synthesize OOXML packages
in-memory using stdlib (zipfile + xml) so the tests run anywhere with no
fixture files. The format we produce is the minimal valid OOXML container
Office writes when it applies a sensitivity label.

The OLE and PDF detectors are tested via their "missing dependency" code
paths (the package's hard runtime deps don't include olefile/pikepdf, so
those branches return a `*-missing` reason on a vanilla install).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from textwrap import dedent

import pytest

from twindb_lightrag_memgraph.classification import (
    ClassificationResult,
    _LABEL_NAMES,
    _normalize_guid,
    detect_classification,
    is_above,
    load_label_map,
)

# A synthetic GUID we'll reuse in tests. Not a real tenant one (which lives
# in the production label map file, not in test fixtures).
DEMO_GUID = "11111111-2222-3333-4444-555555555555"


def _build_docx_with_label(
    tmp_path: Path,
    label_name: str = "C2 Confidentiel",
    guid: str = DEMO_GUID,
    set_date: str = "2026-03-15T10:42:18Z",
    method: str = "Standard",
    extra_props: dict[str, str] | None = None,
) -> Path:
    """Synthesize a minimal .docx package containing only a custom.xml part
    with the given MSIP_Label_* properties. Sufficient to drive the
    extractor; real Office files contain dozens of other parts we don't
    need to fake here.
    """
    custom_xml = dedent(f"""
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
                    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
          <property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="2" name="MSIP_Label_{guid}_Enabled">
            <vt:lpwstr>true</vt:lpwstr>
          </property>
          <property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="3" name="MSIP_Label_{guid}_SetDate">
            <vt:lpwstr>{set_date}</vt:lpwstr>
          </property>
          <property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="4" name="MSIP_Label_{guid}_Method">
            <vt:lpwstr>{method}</vt:lpwstr>
          </property>
          <property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="5" name="MSIP_Label_{guid}_Name">
            <vt:lpwstr>{label_name}</vt:lpwstr>
          </property>
          <property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" pid="6" name="MSIP_Label_{guid}_SiteId">
            <vt:lpwstr>{{99999999-8888-7777-6666-555555555555}}</vt:lpwstr>
          </property>
          {''.join(_extra_xml(extra_props or {}))}
        </Properties>
    """).strip()
    target = tmp_path / "synth.docx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("docProps/custom.xml", custom_xml)
    return target


def _extra_xml(props: dict[str, str]) -> list[str]:
    lines = []
    for i, (name, value) in enumerate(props.items()):
        lines.append(
            f'<property fmtid="{{D5CDD505-2E9C-101B-9397-08002B2CF9AE}}" '
            f'pid="{100 + i}" name="{name}"><vt:lpwstr>{value}</vt:lpwstr></property>'
        )
    return lines


def _build_docx_without_custom_xml(tmp_path: Path) -> Path:
    """A .docx-like zip with no custom.xml part — simulates a doc that
    has never been labeled."""
    target = tmp_path / "naked.docx"
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("docProps/app.xml", "<?xml version='1.0'?><a/>")
    return target


def _build_docx_with_empty_custom_xml(tmp_path: Path) -> Path:
    """custom.xml exists but contains no MSIP_Label_* properties."""
    target = tmp_path / "no_msip.docx"
    custom_xml = (
        "<?xml version='1.0'?>"
        "<Properties xmlns='http://schemas.openxmlformats.org/officeDocument/2006/custom-properties'>"
        "<property fmtid='{x}' pid='2' name='SomeOtherProperty'>"
        "<vt:lpwstr xmlns:vt='http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'>foo</vt:lpwstr>"
        "</property>"
        "</Properties>"
    )
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("docProps/custom.xml", custom_xml)
    return target


# ---------------------------------------------------------------------------
# OOXML happy path
# ---------------------------------------------------------------------------


class TestOoxmlDetection:
    def test_detects_msip_label_with_default_empty_map(self, tmp_path):
        """No tenant map → label resolves to UNKNOWN, but all the MIP fields
        are still extracted for audit trace.
        """
        path = _build_docx_with_label(tmp_path)
        result = detect_classification(path, label_map={})
        assert isinstance(result, ClassificationResult)
        assert result.class_id == "UNKNOWN"
        assert result.label_guid == DEMO_GUID
        assert result.raw_name == "C2 Confidentiel"
        assert result.set_date == "2026-03-15T10:42:18Z"
        assert result.method == "Standard"
        assert result.source_format == "ooxml"
        assert result.reason == "unknown-label-guid"
        # All MSIP fields preserved in meta
        assert result.meta.get("Enabled") == "true"
        assert result.meta.get("SiteId", "").startswith("{99999999-")

    def test_resolves_class_id_with_tenant_map(self, tmp_path):
        path = _build_docx_with_label(tmp_path)
        result = detect_classification(path, label_map={DEMO_GUID: "C2"})
        assert result.class_id == "C2"
        assert result.reason is None
        # Without explicit name map, class_name falls back to raw_name.
        assert result.class_name == "C2 Confidentiel"

    def test_no_custom_props_returns_no_custom_props_reason(self, tmp_path):
        path = _build_docx_without_custom_xml(tmp_path)
        result = detect_classification(path, label_map={})
        assert result.class_id is None
        assert result.reason == "no-custom-props"
        assert result.source_format == "ooxml"

    def test_custom_xml_without_msip_returns_no_msip_label(self, tmp_path):
        path = _build_docx_with_empty_custom_xml(tmp_path)
        result = detect_classification(path, label_map={})
        assert result.class_id is None
        assert result.reason == "no-msip-label"

    def test_picks_most_recent_label_when_multiple(self, tmp_path):
        """Pathological case: two labels in the same file. The detector
        picks the most recently set one (highest SetDate)."""
        # Build a doc with two MSIP_Label_<GUID>_* blocks.
        old_guid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        new_guid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        custom_xml = dedent(f"""
            <?xml version="1.0"?>
            <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
                        xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
              <property fmtid="x" pid="2" name="MSIP_Label_{old_guid}_Name">
                <vt:lpwstr>C1 Public</vt:lpwstr>
              </property>
              <property fmtid="x" pid="3" name="MSIP_Label_{old_guid}_SetDate">
                <vt:lpwstr>2025-01-01T00:00:00Z</vt:lpwstr>
              </property>
              <property fmtid="x" pid="4" name="MSIP_Label_{new_guid}_Name">
                <vt:lpwstr>C3 Strict Conf</vt:lpwstr>
              </property>
              <property fmtid="x" pid="5" name="MSIP_Label_{new_guid}_SetDate">
                <vt:lpwstr>2026-01-01T00:00:00Z</vt:lpwstr>
              </property>
            </Properties>
        """).strip()
        target = tmp_path / "two_labels.docx"
        with zipfile.ZipFile(target, "w") as z:
            z.writestr("docProps/custom.xml", custom_xml)
        result = detect_classification(target, label_map={
            old_guid: "C1", new_guid: "C3",
        })
        assert result.class_id == "C3"
        assert result.label_guid == new_guid
        assert result.set_date == "2026-01-01T00:00:00Z"

    def test_malformed_zip_returns_parse_error(self, tmp_path):
        path = tmp_path / "fake.docx"
        path.write_bytes(b"this is not a zip file")
        result = detect_classification(path, label_map={})
        assert result.class_id is None
        assert result.reason and result.reason.startswith("parse-error:")
        assert result.source_format == "ooxml"

    def test_macro_enabled_extension_is_supported(self, tmp_path):
        """`.docm` (macro-enabled Word) is structurally identical OOXML."""
        path = _build_docx_with_label(tmp_path)
        # Rename the file extension; OOXML container is unchanged.
        renamed = tmp_path / "synth.docm"
        renamed.write_bytes(path.read_bytes())
        result = detect_classification(renamed, label_map={DEMO_GUID: "C2"})
        assert result.class_id == "C2"
        assert result.source_format == "ooxml"


# ---------------------------------------------------------------------------
# Legacy OLE + PDF: optional-dep graceful degradation
# ---------------------------------------------------------------------------


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


class TestOptionalDeps:
    @pytest.mark.skipif(
        _has_module("olefile"),
        reason="olefile present — this test asserts the missing-dep path",
    )
    def test_ole_without_olefile(self, tmp_path):
        path = tmp_path / "legacy.doc"
        path.write_bytes(b"")  # content doesn't matter; we short-circuit
        result = detect_classification(path, label_map={})
        assert result.class_id is None
        assert result.reason == "olefile-missing"
        assert result.source_format == "ole"

    @pytest.mark.skipif(
        _has_module("pikepdf"),
        reason="pikepdf present — this test asserts the missing-dep path",
    )
    def test_pdf_without_pikepdf(self, tmp_path):
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4\n%fake")
        result = detect_classification(path, label_map={})
        assert result.class_id is None
        assert result.reason == "pikepdf-missing"
        assert result.source_format == "pdf"

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "note.txt"
        path.write_text("nothing to see here")
        result = detect_classification(path, label_map={})
        assert result.class_id is None
        assert result.reason and "unsupported-extension" in result.reason


# ---------------------------------------------------------------------------
# Label-map loading
# ---------------------------------------------------------------------------


class TestLabelMap:
    def test_explicit_path_overrides_env(self, tmp_path, monkeypatch):
        env_map = tmp_path / "env.json"
        env_map.write_text(json.dumps({DEMO_GUID: "C9"}))
        explicit_map = tmp_path / "explicit.json"
        explicit_map.write_text(json.dumps({DEMO_GUID: "C2"}))
        monkeypatch.setenv("TWIN_MIP_LABEL_MAP", str(env_map))
        loaded = load_label_map(explicit_map)
        assert loaded[DEMO_GUID] == "C2"

    def test_env_var_used_when_no_explicit_path(self, tmp_path, monkeypatch):
        path = tmp_path / "from_env.json"
        path.write_text(json.dumps({DEMO_GUID: "C2"}))
        monkeypatch.setenv("TWIN_MIP_LABEL_MAP", str(path))
        loaded = load_label_map()
        assert loaded[DEMO_GUID] == "C2"

    def test_missing_file_returns_empty_map_and_warns(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setenv("TWIN_MIP_LABEL_MAP", str(tmp_path / "nope.json"))
        loaded = load_label_map()
        assert loaded == {}
        assert any("missing" in r.message.lower() for r in caplog.records)

    def test_no_env_no_arg_returns_empty(self, monkeypatch):
        monkeypatch.delenv("TWIN_MIP_LABEL_MAP", raising=False)
        loaded = load_label_map()
        assert loaded == {}

    def test_loads_long_form_with_human_name(self, tmp_path):
        _LABEL_NAMES.clear()
        path = tmp_path / "longform.json"
        path.write_text(json.dumps({
            DEMO_GUID: {"id": "C2", "name": "C2 Confidentiel"},
        }))
        loaded = load_label_map(path)
        assert loaded[DEMO_GUID] == "C2"
        assert _LABEL_NAMES.get(DEMO_GUID) == "C2 Confidentiel"

    def test_normalizes_guids_on_load(self, tmp_path):
        path = tmp_path / "uppercase.json"
        path.write_text(json.dumps({
            "{" + DEMO_GUID.upper() + "}": "C2",
        }))
        loaded = load_label_map(path)
        assert loaded[DEMO_GUID] == "C2"

    def test_human_name_from_map_overrides_raw_name(self, tmp_path):
        """When the tenant map carries a human name, it wins over the
        raw Microsoft label name embedded in the document."""
        _LABEL_NAMES.clear()
        map_path = tmp_path / "named.json"
        map_path.write_text(json.dumps({
            DEMO_GUID: {"id": "C2", "name": "C2 — Confidentiel groupe"},
        }))
        load_label_map(map_path)
        doc = _build_docx_with_label(tmp_path, label_name="Anything")
        result = detect_classification(doc, label_map={DEMO_GUID: "C2"})
        assert result.class_id == "C2"
        assert result.class_name == "C2 — Confidentiel groupe"


# ---------------------------------------------------------------------------
# Policy helper
# ---------------------------------------------------------------------------


class TestIsAbove:
    @pytest.mark.parametrize("class_id,threshold,expected", [
        ("C1", "C2", False),
        ("C2", "C2", False),
        ("C3", "C2", True),
        ("C4", "C2", True),
        ("C4", "C3", True),
        ("C1", "C1", False),
        ("Public", "Internal", False),
        ("Internal", "Internal", False),
        ("Private", "Internal", True),
        ("Confidential", "Internal", True),
        ("Secret", "Confidential", True),
    ])
    def test_known_classes(self, class_id, threshold, expected):
        assert is_above(class_id, threshold) is expected

    @pytest.mark.parametrize("unknown", ["UNKNOWN", "C7", ""])
    def test_unknown_is_treated_as_above_fail_closed(self, unknown):
        assert is_above(unknown, "C2") is True

    def test_missing_class_is_not_a_mip_class(self):
        assert is_above(None, "C2") is False

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            is_above("C1", "NotARealClass")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


class TestNormalizeGuid:
    @pytest.mark.parametrize("raw,expected", [
        (DEMO_GUID, DEMO_GUID),
        (DEMO_GUID.upper(), DEMO_GUID),
        ("{" + DEMO_GUID + "}", DEMO_GUID),
        ("{" + DEMO_GUID.upper() + "}", DEMO_GUID),
        ("  " + DEMO_GUID + "  ", DEMO_GUID),
    ])
    def test_normalize(self, raw, expected):
        assert _normalize_guid(raw) == expected


class TestResultSerialization:
    def test_as_dict_round_trips_through_json(self, tmp_path):
        path = _build_docx_with_label(tmp_path)
        result = detect_classification(path, label_map={DEMO_GUID: "C2"})
        # Must serialize cleanly so the pipeline can stuff it into
        # DocStatus.metadata (which goes through json.dumps).
        as_json = json.dumps(result.as_dict())
        restored = json.loads(as_json)
        assert restored["class_id"] == "C2"
        assert restored["label_guid"] == DEMO_GUID
