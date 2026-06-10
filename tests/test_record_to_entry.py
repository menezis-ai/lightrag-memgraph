"""Unit tests for MemgraphVectorDBStorage._record_to_entry.

Validates that all declared meta_fields are always present in the
returned dict (set to None when absent from node properties), preventing
KeyError in LightRAG's operate._find_most_related_edges_from_entities.
"""

from unittest.mock import MagicMock, patch

import pytest

from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


def _make_store(meta_fields):
    """Create a MemgraphVectorDBStorage with patched __init__."""
    with patch.object(MemgraphVectorDBStorage, "__init__", lambda self, **kw: None):
        store = MemgraphVectorDBStorage()
    store.meta_fields = meta_fields
    return store


class TestRecordToEntry:
    def test_all_meta_fields_present(self):
        """When node has all meta fields, they are returned."""
        store = _make_store({"src_id", "tgt_id", "content"})
        record = {
            "id": "r1",
            "similarity": 0.95,
            "props": {"src_id": "A", "tgt_id": "B", "content": "edge text"},
        }
        entry = store._record_to_entry(record)
        assert entry["id"] == "r1"
        assert entry["src_id"] == "A"
        assert entry["tgt_id"] == "B"
        assert entry["content"] == "edge text"
        assert entry["similarity"] == 0.95
        assert entry["distance"] == pytest.approx(0.05)

    def test_missing_meta_fields_are_none(self):
        """When node is missing meta fields, they are None (not omitted)."""
        store = _make_store({"src_id", "tgt_id", "source_id", "content"})
        record = {
            "id": "r2",
            "similarity": 0.8,
            "props": {"content": "partial data"},
        }
        entry = store._record_to_entry(record)
        assert entry["src_id"] is None
        assert entry["tgt_id"] is None
        assert entry["source_id"] is None
        assert entry["content"] == "partial data"

    def test_json_meta_field_parsed(self):
        """JSON-encoded dict values in meta fields are deserialized."""
        store = _make_store({"metadata"})
        record = {
            "id": "r3",
            "similarity": 0.7,
            "props": {"metadata": '{"key": "value"}'},
        }
        entry = store._record_to_entry(record)
        assert entry["metadata"] == {"key": "value"}

    def test_empty_meta_fields(self):
        """With no meta_fields declared, only base fields are returned."""
        store = _make_store(set())
        record = {
            "id": "r4",
            "similarity": 0.5,
            "props": {"src_id": "X", "extra": "data"},
        }
        entry = store._record_to_entry(record)
        assert set(entry.keys()) == {"id", "distance", "similarity"}
