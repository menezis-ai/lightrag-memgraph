"""Tests for ontology configuration loading."""

import json

import pytest

from twindb_lightrag_memgraph.intelligence.ontology.config import (
    OntologyConfig,
    load_ontology_config,
)


class TestOntologyConfig:
    def test_load_missing_json_returns_none(self, tmp_path):
        result = load_ontology_config(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_valid_json(self, tmp_path):
        config_data = {
            "enabled": True,
            "confidence_threshold": 0.8,
            "require_review": True,
            "dsep_enabled": True,
            "workspaces": {
                "oracle_ws": {
                    "mode": "dedicated",
                    "subject": "Oracle Database",
                    "context": "Oracle DBA knowledge base",
                },
                "commons": {
                    "mode": "emergence",
                    "subject": "IT Operations",
                    "context": "General docs",
                },
            },
        }
        json_file = tmp_path / "ontology.json"
        json_file.write_text(json.dumps(config_data))

        config = load_ontology_config(json_file)

        assert config is not None
        assert config.enabled is True
        assert config.confidence_threshold == 0.8
        assert config.require_review is True
        assert config.dsep_enabled is True
        assert len(config.workspaces) == 2
        assert config.workspaces["oracle_ws"].mode == "dedicated"
        assert config.workspaces["oracle_ws"].subject == "Oracle Database"
        assert config.workspaces["commons"].mode == "emergence"

    def test_load_invalid_json_raises(self, tmp_path):
        json_file = tmp_path / "ontology.json"
        json_file.write_text("{invalid json: [")

        with pytest.raises(ValueError, match="Invalid ontology.json"):
            load_ontology_config(json_file)

    def test_require_review_false_is_forbidden(self, tmp_path):
        json_file = tmp_path / "ontology.json"
        json_file.write_text(json.dumps({"enabled": True, "require_review": False}))

        with pytest.raises(ValueError, match="require_review=false is forbidden"):
            load_ontology_config(json_file)

    def test_direct_auto_approval_configuration_is_forbidden(self):
        with pytest.raises(ValueError, match="require_review=false is forbidden"):
            OntologyConfig(require_review=False)

    def test_load_non_object_json_raises(self, tmp_path):
        json_file = tmp_path / "ontology.json"
        json_file.write_text(json.dumps(["just", "a", "list"]))

        with pytest.raises(ValueError, match="must be a JSON object"):
            load_ontology_config(json_file)

    def test_load_invalid_mode_raises(self, tmp_path):
        config_data = {
            "enabled": True,
            "workspaces": {
                "bad_ws": {"mode": "invalid_mode"},
            },
        }
        json_file = tmp_path / "ontology.json"
        json_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="Invalid mode"):
            load_ontology_config(json_file)

    def test_default_values(self):
        config = OntologyConfig()
        assert config.enabled is False
        assert config.confidence_threshold == 0.7
        assert config.require_review is True
        assert config.dsep_enabled is True
        assert config.workspaces == {}

    def test_dual_pass_config_fields(self, tmp_path):
        config_data = {
            "enabled": True,
            "dual_pass": True,
            "global_max_tokens": 15000,
            "workspaces": {
                "ws": {
                    "mode": "emergence",
                    "dsep_operators_global": ["structural_analysis", "bounded_context"],
                    "dsep_operators_local": ["entity_definition", "gap_analysis"],
                },
            },
        }
        json_file = tmp_path / "ontology.json"
        json_file.write_text(json.dumps(config_data))

        config = load_ontology_config(json_file)

        assert config is not None
        assert config.dual_pass is True
        assert config.global_max_tokens == 15000
        ws = config.workspaces["ws"]
        assert ws.dsep_operators_global == ["structural_analysis", "bounded_context"]
        assert ws.dsep_operators_local == ["entity_definition", "gap_analysis"]

    def test_dual_pass_defaults(self):
        config = OntologyConfig()
        assert config.dual_pass is False
        assert config.global_max_tokens == 20000

    def test_deep_extraction_mode(self, tmp_path):
        config_data = {
            "enabled": True,
            "workspaces": {
                "deep": {
                    "mode": "deep_extraction",
                    "subject": "Symbolic exploration",
                    "dsep_operators": ["structural_analysis", "gap_analysis"],
                },
            },
        }
        json_file = tmp_path / "ontology.json"
        json_file.write_text(json.dumps(config_data))

        config = load_ontology_config(json_file)
        ws = config.workspaces["deep"]
        assert ws.mode == "deep_extraction"
        assert ws.dsep_operators == ["structural_analysis", "gap_analysis"]
