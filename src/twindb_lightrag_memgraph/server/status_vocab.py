"""Canonical document-status vocabulary + explicit per-surface projections.

Audit 2026-07-02 (``docs/audits/ingestion-reindex/audit-2026-07-02.md``,
finding **DUP-1** / remediation #9): the "document" shape was defined 8+
times across the server package with THREE live status vocabularies:

- native shim  → **lowercase** (``native_shims._project_doc``) — mirrors
  LightRAG's ``DocStatus.value`` casing;
- twin route   → **UPPERCASE** (``webui/router._webui_doc_status``) — mirrors
  the React port's ``DocumentStatus`` type;
- seed         → **legacy** ``completed`` / ``processing`` / ``failed``
  (``webui_seed.DOCUMENTS``) — mirrors the design-prototype fixtures.

This module is the single source those three surfaces now derive from.

**ZERO wire-contract change**: every surface keeps emitting exactly its
historical casing/values. What changes is that the spellings are produced by
the explicit projections below, so the vocabularies can no longer drift
apart silently.

LightRAG 1.5.x statuses (``parsing``, ``analyzing``, ``preprocessed`` —
``lightrag/base.py`` on 1.5.4) are part of the canonical enum with their
documented coercion: surfaces that only understand the four legacy states
coerce them to ``PENDING``. That is the behaviour already shipped by
``MemgraphDocStatusStorage._deserialize_status`` (unknown → PENDING,
``docstatus_impl.py``) and by the WebUI ingress normalizer
(``lightrag_webui_twin/src/lib/docStatus.ts``) — see audit finding PIPE-13.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "CanonicalDocStatus",
    "LEGACY_STATUSES",
    "LIGHTRAG_15X_STATUSES",
    "LIGHTRAG_15X_COERCION",
    "normalize_status",
    "coerce_lightrag_15x",
    "to_native_lowercase",
    "to_native_count_key",
    "to_twin_uppercase",
    "to_seed_legacy",
    "from_seed_legacy",
    "storage_status_filter",
]


class CanonicalDocStatus(str, Enum):
    """Every document status this codebase knows how to speak.

    Values are the LightRAG-native (lowercase) spellings — the storage layer
    is the ground truth, the UI casings are projections.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    # LightRAG 1.5.x pipeline states (PENDING → PARSING → ANALYZING →
    # PROCESSING → PROCESSED|FAILED, plus PREPROCESSED). Never emitted by the
    # pinned BNP runtime (1.4.9.11); see LIGHTRAG_15X_COERCION for how the
    # 4-state surfaces are expected to coerce them.
    PARSING = "parsing"
    ANALYZING = "analyzing"
    PREPROCESSED = "preprocessed"


#: The four statuses every wire surface (native shim, twin route, seed, WebUI)
#: understood historically.
LEGACY_STATUSES: frozenset[CanonicalDocStatus] = frozenset(
    {
        CanonicalDocStatus.PENDING,
        CanonicalDocStatus.PROCESSING,
        CanonicalDocStatus.PROCESSED,
        CanonicalDocStatus.FAILED,
    }
)

#: Statuses introduced by the LightRAG 1.5.x state machine.
LIGHTRAG_15X_STATUSES: frozenset[CanonicalDocStatus] = frozenset(
    {
        CanonicalDocStatus.PARSING,
        CanonicalDocStatus.ANALYZING,
        CanonicalDocStatus.PREPROCESSED,
    }
)

#: Documented coercion for 4-state consumers. PENDING (not PROCESSING) is
#: deliberate: it matches the two normalizers already in production —
#: ``docstatus_impl._deserialize_status`` falls back to ``DocStatus.PENDING``
#: on unknown values, and the WebUI coerces unknown statuses to ``'PENDING'``
#: at ingress. Changing this mapping is a wire-behaviour change; don't.
LIGHTRAG_15X_COERCION: dict[CanonicalDocStatus, CanonicalDocStatus] = {
    CanonicalDocStatus.PARSING: CanonicalDocStatus.PENDING,
    CanonicalDocStatus.ANALYZING: CanonicalDocStatus.PENDING,
    CanonicalDocStatus.PREPROCESSED: CanonicalDocStatus.PENDING,
}


def _raw_value(raw: Any) -> Any:
    """Unwrap enum-ish inputs (``DocStatus``/``CanonicalDocStatus``) to their value."""
    return raw.value if hasattr(raw, "value") else raw


def normalize_status(raw: Any) -> CanonicalDocStatus | None:
    """Best-effort mapping of any spelling to the canonical enum.

    Accepts enum members (anything with ``.value``), any-cased strings
    (``"PROCESSED"``, ``"Processed"``, ``"processed"``) and the seed-legacy
    spelling ``"completed"``. Returns ``None`` for unknown/empty input —
    callers decide their own fallback (the shipped ones use PENDING).
    """
    value = _raw_value(raw)
    if not isinstance(value, str) or not value:
        return None
    lowered = value.lower()
    if lowered == "completed":  # seed-legacy alias for PROCESSED
        return CanonicalDocStatus.PROCESSED
    try:
        return CanonicalDocStatus(lowered)
    except ValueError:
        return None


def coerce_lightrag_15x(member: CanonicalDocStatus) -> CanonicalDocStatus:
    """Project a canonical status onto the 4-state legacy vocabulary."""
    return LIGHTRAG_15X_COERCION.get(member, member)


# ---------------------------------------------------------------------------
# Wire projections — one per surface. Each is byte-identical to the historical
# inline expression it replaced (named in the docstring). Do not "clean up"
# their edge-case behaviour without an explicit wire-contract decision.
# ---------------------------------------------------------------------------


def to_native_lowercase(raw: Any) -> str:
    """Native-shim wire spelling (``native_shims._project_doc``).

    Lowercase in practice because ``DocStatus.value`` is lowercase; by design
    this is a *pass-through* (enum → ``.value``, falsy → ``""``) so the shim
    never rewrites an exotic backend value it did not produce. Historical
    expression: ``doc.get("status") or ""`` after the ``.value`` unwrap in
    ``_project_doc_tuples``.
    """
    return _raw_value(raw) or ""


def to_native_count_key(raw: Any) -> str:
    """``status_counts`` key on the native shim (lowercased, may be empty).

    Historical expression (``_status_counts_for_projected_docs``):
    ``str(doc.get("status") or "").lower()``.
    """
    return str(_raw_value(raw) or "").lower()


def to_twin_uppercase(raw: Any) -> str:
    """Twin overlay wire spelling (``webui/router._webui_doc_status``).

    UPPERCASE — mirror of the React port's ``DocumentStatus`` type.
    Historical expression: ``(raw.value if hasattr(raw, "value") else
    str(raw or "")).upper()``.
    """
    value = raw.value if hasattr(raw, "value") else str(raw or "")
    return value.upper()


#: Seed-legacy spellings (design-prototype vocabulary used by
#: ``webui_seed.DOCUMENTS`` and the TS fixtures' ancestors).
_SEED_LEGACY_BY_CANONICAL: dict[CanonicalDocStatus, str] = {
    CanonicalDocStatus.PENDING: "pending",
    CanonicalDocStatus.PROCESSING: "processing",
    CanonicalDocStatus.PROCESSED: "completed",
    CanonicalDocStatus.FAILED: "failed",
}


def to_seed_legacy(member: CanonicalDocStatus) -> str:
    """Seed wire spelling (``completed``/``processing``/``failed``/``pending``).

    1.5.x members are coerced to the legacy 4-state vocabulary first
    (documented coercion → PENDING → ``"pending"``).
    """
    return _SEED_LEGACY_BY_CANONICAL[coerce_lightrag_15x(member)]


def from_seed_legacy(value: str) -> CanonicalDocStatus | None:
    """Inverse of :func:`to_seed_legacy` (``"completed"`` → PROCESSED)."""
    return normalize_status(value)


def storage_status_filter(status: str | None) -> str | None:
    """UI status-filter string → LightRAG ``DocStatus`` value, or ``None``.

    Byte-identical port of ``webui/router._status_filter_for_doc_status``:
    accepts the seed-legacy ``completed`` alias, the lowercase native
    values, UPPERCASE UI spellings and ``"DocStatus.PROCESSED"``-style
    reprs. Unknown values (including the 1.5.x statuses — deliberately, to
    preserve the "unknown filter → no filter" behaviour) return ``None``.
    """
    if not status or status.lower() == "all":
        return None
    normalized = status.lower()
    if normalized in ("completed", "processed"):
        return "processed"
    if normalized in ("pending", "processing", "failed"):
        return normalized
    upper_map = {
        "processed": "processed",
        "pending": "pending",
        "processing": "processing",
        "failed": "failed",
    }
    return upper_map.get(status.upper().removeprefix("DOCSTATUS.").lower())
