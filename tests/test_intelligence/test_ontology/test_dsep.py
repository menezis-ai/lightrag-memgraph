"""Tests for DSEP (Domain-Specific Extraction Profile)."""

from twindb_lightrag_memgraph.intelligence.ontology.dsep import (
    DSEP_OPERATORS,
    build_dsep_block,
    get_mode_defaults,
    get_pass_defaults,
)


class TestDSEPOperators:
    def test_all_6_operators_defined(self):
        assert len(DSEP_OPERATORS) == 6

    def test_operator_keys(self):
        expected = {
            "structural_analysis",
            "scope_exclusion",
            "gap_analysis",
            "bounded_context",
            "entity_definition",
            "convergence",
        }
        assert set(DSEP_OPERATORS.keys()) == expected

    def test_each_operator_has_symbol(self):
        for op in DSEP_OPERATORS.values():
            assert op.symbol
            assert len(op.symbol) == 1

    def test_each_operator_has_directive(self):
        for op in DSEP_OPERATORS.values():
            assert op.directive
            assert len(op.directive) > 10


class TestModeDefaults:
    def test_dedicated_defaults(self):
        defaults = get_mode_defaults("dedicated")
        assert defaults == [
            "structural_analysis",
            "bounded_context",
            "entity_definition",
        ]

    def test_emergence_defaults(self):
        defaults = get_mode_defaults("emergence")
        assert defaults == [
            "structural_analysis",
            "gap_analysis",
            "entity_definition",
            "convergence",
        ]

    def test_deep_extraction_defaults_all_operators(self):
        defaults = get_mode_defaults("deep_extraction")
        assert len(defaults) == 6
        assert set(defaults) == set(DSEP_OPERATORS.keys())

    def test_unknown_mode_returns_empty(self):
        defaults = get_mode_defaults("unknown")
        assert defaults == []

    def test_get_pass_defaults_global(self):
        defaults = get_pass_defaults("global")
        assert defaults == [
            "structural_analysis",
            "bounded_context",
            "scope_exclusion",
        ]

    def test_get_pass_defaults_local(self):
        defaults = get_pass_defaults("local")
        assert defaults == [
            "entity_definition",
            "gap_analysis",
            "convergence",
        ]

    def test_get_pass_defaults_unknown(self):
        defaults = get_pass_defaults("unknown")
        assert defaults == []


class TestBuildDSEPBlock:
    def test_build_dedicated_block(self):
        block = build_dsep_block([], "dedicated")
        assert "Mode: dedicated" in block
        assert "Structural Analysis" in block
        assert "Bounded Context" in block
        assert "Entity Definition" in block
        # Should NOT contain non-default operators
        assert "Gap Analysis" not in block
        assert "Scope Exclusion" not in block

    def test_build_emergence_block(self):
        block = build_dsep_block([], "emergence")
        assert "Mode: emergence" in block
        assert "Structural Analysis" in block
        assert "Gap Analysis" in block
        assert "Entity Definition" in block
        assert "Migration / Mapping" in block

    def test_build_deep_extraction_block_all_operators(self):
        block = build_dsep_block([], "deep_extraction")
        assert "Mode: deep_extraction" in block
        for op in DSEP_OPERATORS.values():
            assert op.name in block

    def test_symbols_preserved_in_output(self):
        block = build_dsep_block([], "deep_extraction")
        for op in DSEP_OPERATORS.values():
            assert op.symbol in block

    def test_custom_operator_selection(self):
        custom = ["structural_analysis", "gap_analysis"]
        block = build_dsep_block(custom, "dedicated")
        assert "Structural Analysis" in block
        assert "Gap Analysis" in block
        assert "Bounded Context" not in block

    def test_dsep_block_structure(self):
        block = build_dsep_block([], "emergence")
        assert block.startswith("=== DSEP (Domain-Specific Extraction Profile)")
        assert block.endswith("=== END DSEP ===")
        # Mode propagation + the closing instruction added by the prompt
        # security refactor (Red Team hardening 2026-06-02): the block
        # now signs off with a final-JSON gate referencing the operators.
        assert "Mode: emergence" in block
        assert "Before final JSON" in block

    def test_invalid_operator_ignored(self):
        block = build_dsep_block(["nonexistent", "structural_analysis"], "dedicated")
        assert "Structural Analysis" in block
