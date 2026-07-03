"""Pins the canonical status vocabulary + its per-surface projections.

Audit 2026-07-02 (DUP-1 / remediation #9): the three server surfaces (native
shim, twin route, seed) each spelled the document status their own way. They
now all derive from ``server.status_vocab``. These tests pin BOTH the module
itself and the wire spellings of the rewired surfaces, so a vocabulary drift
turns red here instead of leaking to the WebUI.

ZERO wire-contract change was the constraint of the rewiring — every
assertion below states the HISTORICAL spelling.
"""

from __future__ import annotations

from lightrag.base import DocStatus

from twindb_lightrag_memgraph.server import webui_seed
from twindb_lightrag_memgraph.server.native_shims import (
    _project_doc,
    _project_doc_tuples,
    _status_counts_for_projected_docs,
)
from twindb_lightrag_memgraph.server.status_vocab import (
    LEGACY_STATUSES,
    LIGHTRAG_15X_COERCION,
    LIGHTRAG_15X_STATUSES,
    CanonicalDocStatus,
    coerce_lightrag_15x,
    from_seed_legacy,
    normalize_status,
    storage_status_filter,
    to_native_count_key,
    to_native_lowercase,
    to_seed_legacy,
    to_twin_uppercase,
)
from twindb_lightrag_memgraph.server.webui.router import (
    _status_filter_for_doc_status,
    _webui_doc_status,
)


class TestCanonicalEnum:
    def test_legacy_values_are_lightrag_native_lowercase(self):
        assert {m.value for m in LEGACY_STATUSES} == {
            "pending",
            "processing",
            "processed",
            "failed",
        }

    def test_15x_statuses_present_with_documented_coercion_to_pending(self):
        assert {m.value for m in LIGHTRAG_15X_STATUSES} == {
            "parsing",
            "analyzing",
            "preprocessed",
        }
        for member in LIGHTRAG_15X_STATUSES:
            assert LIGHTRAG_15X_COERCION[member] is CanonicalDocStatus.PENDING
            assert coerce_lightrag_15x(member) is CanonicalDocStatus.PENDING

    def test_coercion_is_identity_on_legacy_statuses(self):
        for member in LEGACY_STATUSES:
            assert coerce_lightrag_15x(member) is member

    def test_normalize_accepts_any_casing_enum_and_seed_alias(self):
        assert normalize_status("PROCESSED") is CanonicalDocStatus.PROCESSED
        assert normalize_status("processed") is CanonicalDocStatus.PROCESSED
        assert normalize_status(DocStatus.PROCESSED) is (CanonicalDocStatus.PROCESSED)
        assert normalize_status("completed") is CanonicalDocStatus.PROCESSED
        assert normalize_status("parsing") is CanonicalDocStatus.PARSING
        assert normalize_status(None) is None
        assert normalize_status("") is None
        assert normalize_status("weird") is None


class TestProjections:
    def test_native_lowercase_is_a_passthrough(self):
        # Historical: enum → .value, falsy → "", strings untouched — the shim
        # never rewrites an exotic backend value it did not produce.
        assert to_native_lowercase(DocStatus.PROCESSED) == "processed"
        assert to_native_lowercase("processed") == "processed"
        assert to_native_lowercase("PROCESSED") == "PROCESSED"  # untouched
        assert to_native_lowercase(None) == ""
        assert to_native_lowercase("") == ""

    def test_native_count_key_lowercases(self):
        assert to_native_count_key(DocStatus.FAILED) == "failed"
        assert to_native_count_key("PENDING") == "pending"
        assert to_native_count_key(None) == ""

    def test_twin_uppercase(self):
        assert to_twin_uppercase(DocStatus.PROCESSED) == "PROCESSED"
        assert to_twin_uppercase("processed") == "PROCESSED"
        assert to_twin_uppercase(None) == ""
        assert to_twin_uppercase("parsing") == "PARSING"

    def test_seed_legacy_round_trip(self):
        assert to_seed_legacy(CanonicalDocStatus.PROCESSED) == "completed"
        assert to_seed_legacy(CanonicalDocStatus.PROCESSING) == "processing"
        assert to_seed_legacy(CanonicalDocStatus.FAILED) == "failed"
        assert to_seed_legacy(CanonicalDocStatus.PENDING) == "pending"
        # 1.5.x members coerce (→ PENDING → "pending") instead of raising.
        assert to_seed_legacy(CanonicalDocStatus.PARSING) == "pending"
        assert from_seed_legacy("completed") is CanonicalDocStatus.PROCESSED

    def test_storage_status_filter_table(self):
        # Byte-identical port of the historical
        # webui/router._status_filter_for_doc_status mapping.
        assert storage_status_filter(None) is None
        assert storage_status_filter("") is None
        assert storage_status_filter("all") is None
        assert storage_status_filter("ALL") is None
        assert storage_status_filter("completed") == "processed"
        assert storage_status_filter("PROCESSED") == "processed"
        assert storage_status_filter("DocStatus.PROCESSED") == "processed"
        assert storage_status_filter("pending") == "pending"
        assert storage_status_filter("processing") == "processing"
        assert storage_status_filter("failed") == "failed"
        # Unknown → None (deliberately includes 1.5.x statuses: an unknown
        # filter means "no filter", the historical behaviour).
        assert storage_status_filter("parsing") is None
        assert storage_status_filter("weird") is None


class TestRewiredSurfacesKeepTheirWireSpelling:
    """The three surfaces must keep emitting exactly their historical casing."""

    def test_seed_documents_speak_seed_legacy(self):
        statuses = [d["status"] for d in webui_seed.DOCUMENTS]
        assert statuses == [
            "completed",
            "completed",
            "failed",
            "completed",
            "completed",
            "processing",
        ]

    def test_twin_route_speaks_uppercase(self):
        assert _webui_doc_status(DocStatus.PROCESSED) == "PROCESSED"
        assert _webui_doc_status("failed") == "FAILED"
        assert _webui_doc_status(None) == ""

    def test_twin_route_filter_delegates_to_vocab(self):
        for raw in (None, "all", "completed", "PROCESSED", "failed", "weird"):
            assert _status_filter_for_doc_status(raw) == storage_status_filter(raw)

    def test_native_shim_speaks_lowercase(self):
        projected = _project_doc({"id": "d1", "status": DocStatus.PROCESSED})
        assert projected["status"] == "processed"
        rows = _project_doc_tuples(
            [("d1", {"status": DocStatus.FAILED}), ("d2", {"status": None})]
        )
        assert [row["status"] for row in rows] == ["failed", ""]

    def test_native_shim_status_counts_keys_are_lowercase(self):
        counts = _status_counts_for_projected_docs(
            [
                {"status": DocStatus.PROCESSED},
                {"status": "failed"},
                {"status": "PENDING"},
                {"status": None},
            ]
        )
        assert counts == {"processed": 1, "failed": 1, "pending": 1}
