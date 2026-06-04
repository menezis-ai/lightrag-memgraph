"""Unit tests for ``server.graph_reader`` — type mapping + layout +
shape projection from Memgraph rows to WebUI GraphEntity/Relation.

Integration coverage (real Memgraph) lives in
``tests/test_server/test_webui_router_graph.py`` (marked ``integration``).
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server.graph_reader import (
    _edge_record_to_relation,
    _entity_id_to_node_id,
    _node_record_to_entity,
    layout_position,
    map_entity_type,
)


class TestMapEntityType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("organization", "ORG"),
            ("ORG", "ORG"),
            ("Company", "ORG"),
            ("Person", "PERSON"),
            ("PEOPLE", "PERSON"),
            ("geo", "LOCATION"),
            ("Datacenter", "LOCATION"),
            ("product", "PRODUCT"),
            ("Database", "PRODUCT"),
            ("technology", "TECHNOLOGY"),
            ("PROTOCOL", "TECHNOLOGY"),
            ("concept", "CONCEPT"),
            ("event", "CONCEPT"),
        ],
    )
    def test_maps_common_lightrag_types(self, raw, expected):
        assert map_entity_type(raw) == expected

    def test_falls_back_to_concept_on_unknown(self):
        assert map_entity_type("random-bullshit-bingo") == "CONCEPT"

    def test_falls_back_to_concept_on_empty_or_none(self):
        assert map_entity_type("") == "CONCEPT"
        assert map_entity_type(None) == "CONCEPT"
        assert map_entity_type("   ") == "CONCEPT"


class TestLayoutPosition:
    def test_same_id_same_position(self):
        # Determinism is the whole point — page reloads must not shuffle.
        assert layout_position("Oracle Database", "PRODUCT") == layout_position(
            "Oracle Database", "PRODUCT"
        )

    def test_different_ids_get_different_positions(self):
        # Probabilistic but the hash space is huge — for 4 distinct ids
        # we shouldn't see a collision.
        positions = {
            layout_position(eid, "PRODUCT")
            for eid in ("a", "b", "c", "d")
        }
        assert len(positions) == 4

    def test_different_types_use_different_centroids(self):
        # Same id, different type → different cluster.
        p1 = layout_position("same-id", "PRODUCT")
        p2 = layout_position("same-id", "ORG")
        assert p1 != p2

    def test_positions_stay_in_canvas(self):
        for eid in ("foo", "bar", "baz", "very-long-entity-id-that-may-stress-layout"):
            for t in ("PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"):
                x, y = layout_position(eid, t)
                # canvas 960x620 with 40 margin → coords ∈ [40, 920]×[40, 580]
                assert 40 <= x <= 920
                assert 40 <= y <= 580


class TestNodeRecordProjection:
    def test_full_record_projects_to_webui_entity(self):
        row = {
            "entity_id": "Oracle Database",
            "entity_type": "Database",
            "description": "Relational engine.",
            "source_id": "chunk-1,chunk-2,chunk-3",
        }
        out = _node_record_to_entity(row)
        assert out["id"] == "kg_Oracle Database"
        assert out["name"] == "Oracle Database"
        assert out["type"] == "PRODUCT"
        assert out["summary"] == "Relational engine."
        assert out["mentions"] == 3
        assert out["sources"] == 3  # chunk count proxy
        # Position deterministic + clamped
        assert 40 <= out["x"] <= 920
        assert 40 <= out["y"] <= 580

    def test_handles_missing_fields_gracefully(self):
        row = {"entity_id": "lone"}
        out = _node_record_to_entity(row)
        assert out["name"] == "lone"
        assert out["type"] == "CONCEPT"  # fallback
        assert out["summary"] == ""
        assert out["mentions"] == 0
        assert out["sources"] == 0

    def test_source_id_with_sep_separator(self):
        # LightRAG sometimes joins source ids with `<SEP>` instead of comma.
        row = {
            "entity_id": "x",
            "entity_type": "concept",
            "source_id": "chunk-a<SEP>chunk-b",
        }
        out = _node_record_to_entity(row)
        assert out["mentions"] == 2
        # "sources" tracks distinct chunks until we wire DocStatus join
        assert out["sources"] == 2

    def test_source_id_deduplicates_repeated_chunks(self):
        row = {
            "entity_id": "x",
            "entity_type": "concept",
            "source_id": "chunk-a,chunk-a,chunk-b",
        }
        out = _node_record_to_entity(row)
        # de-dup: same chunk listed twice doesn't double-count
        assert out["mentions"] == 2

    def test_summary_truncated_at_600_chars(self):
        row = {
            "entity_id": "wordy",
            "entity_type": "concept",
            "description": "x" * 1000,
        }
        out = _node_record_to_entity(row)
        assert len(out["summary"]) == 600


class TestEdgeRecordProjection:
    def test_full_edge_projects(self):
        row = {
            "source_id": "Oracle Database",
            "target_id": "RHEL 9",
            "keywords": "runs on",
            "weight": 0.88,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["id"] == "kr_000000"
        assert out["source"] == "kg_Oracle Database"
        assert out["target"] == "kg_RHEL 9"
        assert out["label"] == "RUNS_ON"
        assert out["strength"] == 0.88

    def test_weight_above_one_is_normalised(self):
        # LightRAG sometimes outputs weights on a 0..10 scale.
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "uses",
            "weight": 9,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["strength"] == 0.9

    def test_missing_weight_defaults_to_05(self):
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "uses",
            "weight": None,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["strength"] == 0.5

    def test_empty_keywords_default_label(self):
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "",
            "weight": 0.5,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["label"] == "RELATED_TO"

    def test_node_id_namespacing_is_consistent(self):
        # Ensure the same prefix is applied on both sides so endpoints
        # match the entities returned by `_node_record_to_entity`.
        assert _entity_id_to_node_id("x") == "kg_x"
