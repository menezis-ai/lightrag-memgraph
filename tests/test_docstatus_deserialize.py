"""Unit tests for MemgraphDocStatusStorage._deserialize_status.

These do not require a live Memgraph — they exercise the pure
property-dict-to-DocProcessingStatus projection.
"""

from __future__ import annotations

from lightrag.base import DocStatus

from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


def _deserialize(props: dict) -> DocStatus:
    return MemgraphDocStatusStorage._deserialize_status(props).status


class TestStatusNormalization:
    def test_lowercase_processed_keeps_enum_value(self):
        assert _deserialize({"status": "processed"}) == DocStatus.PROCESSED

    def test_uppercase_processed_normalized_to_enum(self):
        # Seeded / imported nodes carrying uppercase status used to fall
        # back to PENDING — root cause of the 2026-06-08 "all docs show
        # pending" warning chain in prod.
        assert _deserialize({"status": "PROCESSED"}) == DocStatus.PROCESSED

    def test_mixed_case_normalized(self):
        assert _deserialize({"status": "Processing"}) == DocStatus.PROCESSING
        assert _deserialize({"status": "FaILeD"}) == DocStatus.FAILED

    def test_truly_unknown_falls_back_to_pending(self):
        assert _deserialize({"status": "spaghetti"}) == DocStatus.PENDING

    def test_missing_status_defaults_to_pending(self):
        assert _deserialize({}) == DocStatus.PENDING
