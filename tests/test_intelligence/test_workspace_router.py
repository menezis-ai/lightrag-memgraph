"""Tests for F06 Workspace Router."""

import json

from twindb_lightrag_memgraph.intelligence.features.workspace_router import (
    TopologyContext,
    WorkspaceRouter,
)


class TestWorkspaceRouter:
    """Tests for WorkspaceRouter cascade logic."""

    # -- Construction --

    def test_from_json_loads_rules(self, routing_rules_json):
        router = WorkspaceRouter.from_json(routing_rules_json)
        assert len(router._rules) == 3

    def test_from_json_empty_rules(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"rules": []}))
        router = WorkspaceRouter.from_json(path)
        assert len(router._rules) == 0

    def test_from_json_custom_default_workspace(self, tmp_path):
        path = tmp_path / "custom.json"
        path.write_text(json.dumps({"default_workspace": "my_default", "rules": []}))
        router = WorkspaceRouter.from_json(path)
        assert router._default_workspace == "my_default"

    # -- Cascade: L4 Override (Priority 1) --

    async def test_l4_override_bypasses_all(self, router):
        result = await router.route(
            "Probleme RMAN Oracle",
            provided_workspaces=["demo"],
        )
        assert result.strategy == "l4_override"
        assert result.workspaces == ["demo"]
        assert result.confidence == 1.0

    async def test_l4_override_with_publics(self, router):
        result = await router.route(
            "test",
            provided_workspaces=["demo"],
            provided_workspaces_publics=["commons", "commons_oracle"],
        )
        assert result.strategy == "l4_override"
        assert result.workspaces_publics == ["commons", "commons_oracle"]

    async def test_l4_override_without_publics_uses_default(self, router):
        result = await router.route(
            "test",
            provided_workspaces=["demo"],
        )
        assert result.workspaces_publics == ["commons"]

    # -- Cascade: Topology Context (Priority 2) --

    async def test_topology_context_overrides_keywords(
        self, router, topology_context_demo
    ):
        result = await router.route(
            "Probleme Oracle",  # Would match keyword, but topology takes priority
            topology_context=topology_context_demo,
        )
        assert result.strategy == "topology"
        assert result.workspaces == ["demo"]
        assert result.workspaces_publics == ["commons", "commons_oracle"]

    async def test_topology_context_empty_workspaces_falls_through(self, router):
        empty_topo = TopologyContext(workspaces=[], servers=["srv-01"])
        result = await router.route(
            "Probleme RMAN",
            topology_context=empty_topo,
        )
        # Empty topology workspaces -> falls through to keyword
        assert result.strategy == "keyword"

    # -- Cascade: Keyword Match (Priority 4 -- MVP) --

    async def test_keyword_match_single_domain(self, router):
        result = await router.route("Probleme RMAN backup")
        assert result.strategy == "keyword"
        assert "commons_oracle" in result.workspaces_publics

    async def test_keyword_match_multi_domain(self, router):
        result = await router.route("Probleme Oracle sur RedHat")
        assert result.strategy == "keyword"
        assert "commons_oracle" in result.workspaces_publics
        assert "commons_linux" in result.workspaces_publics

    async def test_keyword_match_case_insensitive(self, router):
        result = await router.route("oracle database issue")
        assert result.strategy == "keyword"
        assert "commons_oracle" in result.workspaces_publics

    async def test_keyword_match_partial(self, router):
        result = await router.route("Erreur ORA-04030 sur le serveur")
        assert result.strategy == "keyword"
        assert "commons_oracle" in result.workspaces_publics

    async def test_keyword_match_confidence(self, router):
        result = await router.route("RMAN backup failure on RHEL")
        # Oracle has confidence 1.0, Linux has 0.9 -> max is 1.0
        assert result.confidence == 1.0

    async def test_keyword_match_always_includes_default(self, router):
        result = await router.route("Oracle problem")
        assert "commons" in result.workspaces_publics

    async def test_keyword_match_dedup(self, router):
        result = await router.route("Oracle ORA-04030 RMAN")
        # All match commons_oracle -> should appear only once
        assert result.workspaces_publics.count("commons_oracle") == 1

    async def test_keyword_match_private_workspace(self, router):
        result = await router.route("Demo application issue")
        assert "demo" in result.workspaces
        assert result.strategy == "keyword"

    # -- Cascade: Default Fallback --

    async def test_default_fallback_no_match(self, router):
        result = await router.route("Comment faire du cafe ?")
        assert result.strategy == "default"
        assert result.workspaces == []
        assert result.workspaces_publics == ["commons"]

    async def test_default_fallback_confidence_low(self, router):
        result = await router.route("Question generique")
        assert result.confidence == 0.5

    # -- Traceability --

    async def test_matched_keywords_in_result(self, router):
        result = await router.route("Probleme RMAN sur RHEL 8")
        assert "RMAN" in result.matched_keywords
        assert "RHEL" in result.matched_keywords
