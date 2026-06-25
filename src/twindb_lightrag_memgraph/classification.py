"""
Microsoft Information Protection (MIP / AIP) sensitivity-label extraction.

Detects sensitivity labels embedded by Microsoft 365 in Office documents and
maps them to a tenant-specific classification scheme (Public / Internal /
Confidential / Secret / Private). The legacy C1-C4 ladder is still accepted
for older tenant maps.

Why this module exists
----------------------
LightRAG ingests file content but does not inspect the *classification*
attached by the producer. In a regulated environment (banking, healthcare,
defense) the same physical PDF can be **Public** or **Strictly Confidential**
depending solely on a Microsoft 365 sensitivity label applied at authoring
time. Letting an unclassified or wrongly-classified document slip into the
retrieval index is a compliance incident.

This module reads the label *as the document carries it* and emits a small
typed payload that the ingestion pipeline can use to:
  - reject documents above a configured maximum (e.g. refuse C3+)
  - persist `metadata.classification` on the DocStatus for retrieval-time
    gating (the WebUI's `DocDetailPanel` already gates the "View raw" notice
    on `metadata.classification > 'internal'`)
  - emit an audit event so the gate is traceable in `/twin/api/activity`

Supported formats
-----------------
- OOXML: `.docx`, `.xlsx`, `.pptx`, plus their macro-enabled siblings
  (`.docm`, `.xlsm`, `.pptm`). Label lives in `docProps/custom.xml` as
  `MSIP_Label_<GUID>_Name` and related properties. Pure stdlib (no extra
  dependency).
- Legacy OLE binary: `.doc`, `.xls`, `.ppt`, `.msg`. Properties live in OLE streams
  ("\\005DocumentSummaryInformation", "\\005SummaryInformation"). Requires
  `olefile` (optional dependency; the extractor returns `None` gracefully
  if olefile is unavailable, with `reason='olefile-missing'` in `meta`).
- PDF: XMP metadata block. Requires `pikepdf` (optional). Same graceful
  degradation if pikepdf is unavailable.

The label-to-classification mapping is *tenant-specific* — the same GUID
means Internal in one tenant but might mean something else in another tenant.
Mappings are loaded from a JSON file pointed to by `TWIN_MIP_LABEL_MAP`
(see `load_label_map`).

Design notes
------------
- The detector NEVER raises on a malformed input; an unreadable file
  returns `ClassificationResult(class_id=None, reason='...')`. This keeps
  the ingestion pipeline robust: a single bad file should not crash a
  batch.
- For OOXML the parser walks ONLY the small `docProps/custom.xml` (always
  < 16 KB in practice) — we do NOT extract the main document content.
- All GUIDs are normalized to lowercase, braceless form for table lookup.
- The default mapping (when `TWIN_MIP_LABEL_MAP` is unset) treats any
  detected label as `UNKNOWN` and lets the caller decide what to do. The
  detector therefore never silently classifies a file as "Public" just
  because the tenant table is missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, cast
from xml.etree import ElementTree as ET

log = logging.getLogger("twin.classification")

# OOXML namespaces used by the custom-properties part.
_CUSTOM_NS: Final = {
    "cp": "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
    "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
}

# Pattern for MSIP custom property names. The site GUID is the label
# identifier; the suffix is the field name (Name, Enabled, SetDate, ...).
_MSIP_PROP_RE: Final = re.compile(
    r"^MSIP_Label_(?P<guid>[0-9a-fA-F-]{36})_(?P<field>\w+)$"
)

# OOXML container extensions we handle natively (no extra dependency).
_OOXML_EXTS: Final = frozenset({
    ".docx", ".docm", ".dotx", ".dotm",
    ".xlsx", ".xlsm", ".xltx", ".xltm",
    ".pptx", ".pptm", ".potx", ".potm",
})

# Legacy OLE binary extensions (need olefile). Outlook `.msg` is also an OLE
# container and can carry MSIP custom properties.
_OLE_EXTS: Final = frozenset({".doc", ".xls", ".ppt", ".msg"})

# PDF extension (needs pikepdf for XMP metadata).
_PDF_EXTS: Final = frozenset({".pdf"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of a classification probe on a single file.

    Attributes
    ----------
    class_id:
        Tenant-specific class identifier (e.g. ``"Internal"``) resolved from the
        MIP label GUID via the loaded label map. ``None`` when the file
        carries no MIP label, when the label is unknown to the map, or when
        extraction failed (see ``reason``).
    class_name:
        Human-readable class label (e.g. ``"Internal"``). May be the
        raw MIP label name when the GUID maps to ``UNKNOWN``.
    label_guid:
        Normalized (lowercase, braceless) MIP label GUID, when detected.
    raw_name:
        The `MSIP_Label_<GUID>_Name` value as written by the producer.
    set_date:
        ISO timestamp the label was applied (`SetDate` field).
    method:
        ``"Standard"`` or ``"Privileged"`` per Microsoft's spec.
    source_format:
        ``"ooxml"`` / ``"ole"`` / ``"pdf"`` / ``"unknown"`` — which detector
        produced the result.
    reason:
        Free-text hint when `class_id is None`: ``"no-custom-props"``,
        ``"no-msip-label"``, ``"olefile-missing"``, ``"pikepdf-missing"``,
        ``"unsupported-extension"``, ``"parse-error: <detail>"``,
        ``"unknown-label-guid"``.
    meta:
        Free-form additional metadata (e.g. all MSIP_Label_* fields for
        audit). Not used for gating, persisted alongside for trace.
    """

    class_id: str | None = None
    class_name: str | None = None
    label_guid: str | None = None
    raw_name: str | None = None
    set_date: str | None = None
    method: str | None = None
    source_format: str = "unknown"
    reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-serializable dict suitable for DocStatus.metadata."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "label_guid": self.label_guid,
            "raw_name": self.raw_name,
            "set_date": self.set_date,
            "method": self.method,
            "source_format": self.source_format,
            "reason": self.reason,
            "meta": dict(self.meta),
        }


# ---------------------------------------------------------------------------
# Label map (tenant-specific GUID → tenant class)
# ---------------------------------------------------------------------------


_DEFAULT_LABEL_MAP: dict[str, str] = {}


def _normalize_guid(guid: str) -> str:
    return guid.strip("{} \t\r\n").lower()


def load_label_map(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Load a `{guid → class_id}` map from JSON.

    Resolution order:
      1. Explicit ``path`` argument.
      2. ``TWIN_MIP_LABEL_MAP`` env var (path to JSON file).
      3. Empty map (every detected label resolves to ``UNKNOWN``).

    File format::

        {
          "guid-of-public": {"id": "Public", "name": "Public"},
          "guid-of-internal": {"id": "Internal", "name": "Internal"},
          ...
        }

    or shorthand::

        {
          "guid-of-public": "Public",
          "guid-of-internal": "Internal",
          ...
        }

    GUIDs are normalized (lowercase, braceless) on load.
    """
    src = path or os.environ.get("TWIN_MIP_LABEL_MAP")
    if not src:
        return dict(_DEFAULT_LABEL_MAP)
    p = Path(src)
    if not p.is_file():
        log.warning("MIP label map %s missing — falling back to empty map", p)
        return dict(_DEFAULT_LABEL_MAP)
    raw = json.loads(p.read_text())
    out: dict[str, str] = {}
    name_out: dict[str, str] = {}
    for guid, value in raw.items():
        nguid = _normalize_guid(guid)
        if isinstance(value, str):
            out[nguid] = value
        elif isinstance(value, dict) and "id" in value:
            out[nguid] = value["id"]
            if "name" in value:
                name_out[nguid] = value["name"]
        else:
            log.warning("Skipping malformed label map entry for %s", guid)
    # Stash human names in a module-level dict keyed by guid → name.
    _LABEL_NAMES.update(name_out)
    return out


# Module-level cache of human-readable names alongside class_ids.
_LABEL_NAMES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_classification(
    file_path: str | os.PathLike[str],
    *,
    label_map: dict[str, str] | None = None,
) -> ClassificationResult:
    """Probe a single file for a MIP sensitivity label.

    Parameters
    ----------
    file_path:
        Path to the document. Must exist and be readable.
    label_map:
        Optional explicit `{guid → class_id}` map. When `None`, the result
        of :func:`load_label_map` is used.

    Returns
    -------
    ClassificationResult
        Always returns a result — never raises on a malformed input. Check
        ``result.class_id`` for success and ``result.reason`` for failure
        explanations.
    """
    p = Path(file_path)
    ext = p.suffix.lower()
    if label_map is None:
        label_map = load_label_map()

    if ext in _OOXML_EXTS:
        return _detect_ooxml(p, label_map)
    if ext in _OLE_EXTS:
        return _detect_ole(p, label_map)
    if ext in _PDF_EXTS:
        return _detect_pdf(p, label_map)
    return ClassificationResult(
        source_format="unknown",
        reason=f"unsupported-extension: {ext}",
    )


# ---------------------------------------------------------------------------
# OOXML detector (pure stdlib)
# ---------------------------------------------------------------------------


def _read_custom_xml(p: Path) -> bytes | None:
    """Return raw `docProps/custom.xml` bytes, or `None` if absent."""
    try:
        with zipfile.ZipFile(p) as z:
            try:
                return z.read("docProps/custom.xml")
            except KeyError:
                return None
    except (zipfile.BadZipFile, OSError) as exc:
        log.debug("OOXML open failed for %s: %s", p, exc)
        raise


def _parse_msip_properties(xml_bytes: bytes) -> dict[str, dict[str, str]]:
    """Parse `docProps/custom.xml` and return `{guid → {field → value}}`.

    Only `MSIP_Label_<GUID>_<field>` properties are kept; everything else
    is ignored.
    """
    root = ET.fromstring(xml_bytes)
    out: dict[str, dict[str, str]] = {}
    for prop in root.findall("cp:property", _CUSTOM_NS):
        name = prop.get("name") or ""
        m = _MSIP_PROP_RE.match(name)
        if not m:
            continue
        guid = _normalize_guid(m.group("guid"))
        field_name = m.group("field")
        # The actual value lives in the single child element (vt:lpwstr etc.)
        value = prop[0].text if len(prop) else None
        out.setdefault(guid, {})[field_name] = value or ""
    return out


def _detect_ooxml(
    p: Path,
    label_map: dict[str, str],
) -> ClassificationResult:
    try:
        xml = _read_custom_xml(p)
    except (zipfile.BadZipFile, OSError) as exc:
        return ClassificationResult(
            source_format="ooxml",
            reason=f"parse-error: {exc.__class__.__name__}",
        )
    if xml is None:
        return ClassificationResult(
            source_format="ooxml",
            reason="no-custom-props",
        )
    try:
        labels = _parse_msip_properties(xml)
    except ET.ParseError as exc:
        return ClassificationResult(
            source_format="ooxml",
            reason=f"parse-error: {exc}",
        )
    if not labels:
        return ClassificationResult(
            source_format="ooxml",
            reason="no-msip-label",
        )
    # Practice: only one MSIP label per document. If multiple, pick the most
    # recently set (highest SetDate ISO string) for safety.
    def set_date_of(entry: tuple[str, dict[str, str]]) -> str:
        return entry[1].get("SetDate", "")
    guid, fields = max(labels.items(), key=set_date_of)
    return _result_from_msip_fields("ooxml", guid, fields, label_map)


# ---------------------------------------------------------------------------
# Legacy OLE detector (optional dep: olefile)
# ---------------------------------------------------------------------------


def _detect_ole(p: Path, label_map: dict[str, str]) -> ClassificationResult:
    try:
        import olefile  # type: ignore
    except ImportError:
        return ClassificationResult(
            source_format="ole",
            reason="olefile-missing",
        )
    try:
        ole = olefile.OleFileIO(str(p))
    except (OSError, ValueError) as exc:
        return ClassificationResult(
            source_format="ole",
            reason=f"parse-error: {exc.__class__.__name__}",
        )
    try:
        # MSIP labels in legacy OLE land in the custom properties stream
        # accessible via get_metadata() — same MSIP_Label_<GUID>_<field> keys.
        meta = ole.get_metadata()
        custom = getattr(meta, "custom", None) or {}
    finally:
        ole.close()
    labels: dict[str, dict[str, str]] = {}
    for name, value in custom.items():
        m = _MSIP_PROP_RE.match(name)
        if not m:
            continue
        guid = _normalize_guid(m.group("guid"))
        labels.setdefault(guid, {})[m.group("field")] = str(value or "")
    if not labels:
        return ClassificationResult(
            source_format="ole",
            reason="no-msip-label",
        )
    guid, fields = max(labels.items(), key=lambda kv: kv[1].get("SetDate", ""))
    return _result_from_msip_fields("ole", guid, fields, label_map)


# ---------------------------------------------------------------------------
# PDF detector (optional dep: pikepdf)
# ---------------------------------------------------------------------------


_PDF_XMP_MSIP_RE: Final = re.compile(
    r"<msip:Label_(?P<guid>[0-9a-fA-F-]{36})_(?P<field>\w+)>"
    r"(?P<value>[^<]*)</msip:Label_[^>]+>",
    re.IGNORECASE,
)


def _detect_pdf(p: Path, label_map: dict[str, str]) -> ClassificationResult:
    try:
        import pikepdf  # type: ignore
    except ImportError:
        return ClassificationResult(
            source_format="pdf",
            reason="pikepdf-missing",
        )
    try:
        pdf = pikepdf.Pdf.open(p)
    except (OSError, RuntimeError) as exc:
        return ClassificationResult(
            source_format="pdf",
            reason=f"parse-error: {exc.__class__.__name__}",
        )
    try:
        with pdf.open_metadata() as xmp:
            raw_xmp = str(xmp)
    finally:
        pdf.close()
    labels: dict[str, dict[str, str]] = {}
    for m in _PDF_XMP_MSIP_RE.finditer(raw_xmp):
        guid = _normalize_guid(m.group("guid"))
        labels.setdefault(guid, {})[m.group("field")] = m.group("value")
    if not labels:
        return ClassificationResult(
            source_format="pdf",
            reason="no-msip-label",
        )
    guid, fields = max(labels.items(), key=lambda kv: kv[1].get("SetDate", ""))
    return _result_from_msip_fields("pdf", guid, fields, label_map)


# ---------------------------------------------------------------------------
# Shared assembly from parsed MSIP fields
# ---------------------------------------------------------------------------


def _result_from_msip_fields(
    source_format: str,
    guid: str,
    fields: dict[str, str],
    label_map: dict[str, str],
) -> ClassificationResult:
    class_id = label_map.get(guid)
    raw_name = fields.get("Name")
    class_name = _LABEL_NAMES.get(guid) or raw_name
    reason = None if class_id else "unknown-label-guid"
    if class_id is None:
        # Default to UNKNOWN so the caller can decide (reject vs allow with
        # quarantine). We do NOT silently coerce to "internal" — the safest
        # default for a regulated tenant is "treat as unknown until proven".
        class_id = "UNKNOWN"
    return ClassificationResult(
        class_id=class_id,
        class_name=class_name,
        label_guid=guid,
        raw_name=raw_name,
        set_date=fields.get("SetDate"),
        method=fields.get("Method"),
        source_format=source_format,
        reason=reason,
        meta=dict(fields.items()),
    )


# ---------------------------------------------------------------------------
# Convenience policy helpers
# ---------------------------------------------------------------------------


_CLASS_ALIASES: Final[dict[str, str]] = {
    "c1": "Public",
    "public": "Public",
    "c2": "Internal",
    "internal": "Internal",
    "c3": "Confidential",
    "confidential": "Confidential",
    "restricted": "Confidential",
    "c4": "Secret",
    "secret": "Secret",
    "private": "Private",
}

_DEFAULT_CLASS_LADDER: Final = ("Public", "Internal", "Private", "Confidential", "Secret")


def _normalize_class_id(class_id: str | None) -> str | None:
    if class_id is None:
        return None
    return _CLASS_ALIASES.get(str(class_id).strip().lower(), class_id)


def is_above(
    class_id: str | None,
    threshold: str,
    *,
    ladder: tuple[str, ...] = _DEFAULT_CLASS_LADDER,
) -> bool:
    """Return True when `class_id` strictly outranks `threshold`.

    Unknown class ids ("UNKNOWN", or not in the ladder) are treated as ABOVE
    the threshold by default — fail-closed. ``None`` means "no MIP label
    detected" and is not above any ceiling.

    Examples
    --------
    >>> is_above("C3", "C2")
    True
    >>> is_above("C2", "C3")
    False
    >>> is_above("UNKNOWN", "C2")
    True
    """
    normalized_class = _normalize_class_id(class_id)
    normalized_threshold = _normalize_class_id(threshold)
    if normalized_class is None:
        return False
    if normalized_class not in ladder:
        return True
    if normalized_threshold not in ladder:
        raise ValueError(f"threshold {threshold!r} not in ladder {ladder!r}")
    return ladder.index(normalized_class) > ladder.index(normalized_threshold)


def apply_operator_classification(
    detected: ClassificationResult,
    operator_class: str | None,
    *,
    ladder: tuple[str, ...] = _DEFAULT_CLASS_LADDER,
) -> ClassificationResult:
    """Combine an operator-selected class with the auto-detected result.

    Policy (compliance-safe, PO decision 2026-06-24): the embedded/detected
    label is a **floor**. The operator can RAISE the classification, or set one
    when nothing was detected (e.g. a ``.md`` with no embedded MIP label), but
    can NEVER downgrade below a detected label. An operator attempt to downgrade
    is recorded in ``meta['operator_requested']`` for the audit trail without
    changing the resolved class.

    ``operator_class`` accepts either the C1-C4 ladder ids or the business names
    (``Public`` … ``Secret``); both normalise through ``_CLASS_ALIASES``. A
    falsy or unrecognised value leaves the detected result untouched.
    """
    if not operator_class:
        return detected
    op_norm = _normalize_class_id(operator_class)
    if op_norm is None or op_norm not in ladder:
        # Operator value not in the known ladder — ignore, keep auto-detection.
        return detected

    det_norm = _normalize_class_id(detected.class_id)

    # Nothing solid auto-detected (no embedded label) -> operator is the only
    # signal and becomes authoritative.
    if det_norm is None:
        return ClassificationResult(
            class_id=op_norm,
            class_name=op_norm,
            source_format="operator",
            reason="operator-set",
            meta={**detected.meta, "detected_reason": detected.reason},
        )

    # A label was detected but does not resolve into the ladder (UNKNOWN /
    # fail-closed): never let the operator downgrade an unrecognised-but-present
    # label. Keep it, note the attempted choice.
    if det_norm not in ladder:
        return cast(
            ClassificationResult,
            replace(detected, meta={**detected.meta, "operator_requested": op_norm}),
        )

    # Both resolve into the ladder: floor protection -> keep the higher class.
    if ladder.index(op_norm) > ladder.index(det_norm):
        return ClassificationResult(
            class_id=op_norm,
            class_name=op_norm,
            label_guid=detected.label_guid,
            raw_name=detected.raw_name,
            set_date=detected.set_date,
            method=detected.method,
            source_format="operator",
            reason="operator-raised",
            meta={**detected.meta, "detected_class_id": detected.class_id},
        )

    # Operator chose an equal/lower class -> the detected label is the floor.
    return cast(
        ClassificationResult,
        replace(detected, meta={**detected.meta, "operator_requested": op_norm}),
    )
