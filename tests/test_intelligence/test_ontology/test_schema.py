"""Tests for ontology schema definitions."""

from twindb_lightrag_memgraph.intelligence.ontology.schema import (
    NODE_TYPES,
    RELATION_PROPERTIES,
    RELATION_TYPES,
    SEED_ENVIRONMENTS,
    SEED_METHODOLOGIES,
    SEED_SLAS,
)


class TestNodeTypes:
    def test_all_11_node_types_defined(self):
        expected = {
            "Term",
            "Role",
            "Team",
            "Tool",
            "Process",
            "Domain",
            "Document",
            "Methodology",
            "Environment",
            "SLA",
            "Asset",
        }
        assert set(NODE_TYPES.keys()) == expected

    def test_each_type_has_label(self):
        for name, node_type in NODE_TYPES.items():
            assert node_type.label == name

    def test_each_type_has_required_properties(self):
        for node_type in NODE_TYPES.values():
            assert len(node_type.required_properties) >= 1

    def test_term_properties(self):
        term = NODE_TYPES["Term"]
        assert "name" in term.required_properties
        assert "definition" in term.optional_properties
        assert "confidence" in term.optional_properties

    def test_sla_properties(self):
        sla = NODE_TYPES["SLA"]
        assert "priority" in sla.required_properties
        assert "gtr_hours" in sla.optional_properties


class TestRelationTypes:
    def test_all_16_relation_types(self):
        assert len(RELATION_TYPES) == 16

    def test_key_relations_present(self):
        assert "SYNONYM" in RELATION_TYPES
        assert "RELATED_TO" in RELATION_TYPES
        assert "CAUSED_BY" in RELATION_TYPES
        assert "DEPENDS_ON" in RELATION_TYPES
        assert "PART_OF" in RELATION_TYPES

    def test_relation_properties(self):
        assert "confidence" in RELATION_PROPERTIES
        assert "source_doc" in RELATION_PROPERTIES
        assert "created_at" in RELATION_PROPERTIES


class TestSeedData:
    def test_methodologies_count(self):
        assert len(SEED_METHODOLOGIES) == 9

    def test_methodology_names(self):
        names = {m["name"] for m in SEED_METHODOLOGIES}
        assert "ITIL" in names
        assert "SAFe" in names
        assert "DevOps" in names
        assert "SRE" in names
        assert "ISO 27001" in names
        assert "TOGAF" in names

    def test_methodology_has_required_fields(self):
        for m in SEED_METHODOLOGIES:
            assert "name" in m
            assert "version" in m
            assert "framework" in m

    def test_slas_count(self):
        assert len(SEED_SLAS) == 4

    def test_sla_priorities(self):
        priorities = [s["priority"] for s in SEED_SLAS]
        assert priorities == ["P1", "P2", "P3", "P4"]

    def test_sla_gtr_hours(self):
        gtr = {s["priority"]: s["gtr_hours"] for s in SEED_SLAS}
        assert gtr["P1"] == 1
        assert gtr["P2"] == 4
        assert gtr["P3"] == 8
        assert gtr["P4"] == 48

    def test_environments_count(self):
        assert len(SEED_ENVIRONMENTS) == 5

    def test_environment_tiers(self):
        tiers = {e["tier"] for e in SEED_ENVIRONMENTS}
        assert tiers == {"prod", "preprod", "uat", "dev", "dr"}
