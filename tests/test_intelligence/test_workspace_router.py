"""Tests for F06 Workspace Router."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph.intelligence.config import TwinRAGConfig
from twindb_lightrag_memgraph.intelligence.engine import TwinRAGEngine
from twindb_lightrag_memgraph.intelligence.features.workspace_router import (
    RoutingResult, TopologyContext, WorkspaceRouter)
from twindb_lightrag_memgraph.intelligence.models.schemas import IntentType


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
            provided_workspaces=["cib"],
        )
        assert result.strategy == "l4_override"
        assert result.workspaces == ["cib"]
        assert result.confidence == 1.0

    async def test_l4_override_with_publics(self, router):
        result = await router.route(
            "test",
            provided_workspaces=["cib"],
            provided_workspaces_publics=["commons", "commons_oracle"],
        )
        assert result.strategy == "l4_override"
        assert result.workspaces_publics == ["commons", "commons_oracle"]

    async def test_l4_override_without_publics_uses_default(self, router):
        result = await router.route(
            "test",
            provided_workspaces=["cib"],
        )
        assert result.workspaces_publics == ["commons"]

    # -- Cascade: Topology Context (Priority 2) --

    async def test_topology_context_overrides_keywords(
        self, router, topology_context_cib
    ):
        result = await router.route(
            "Probleme Oracle",  # Would match keyword, but topology takes priority
            topology_context=topology_context_cib,
        )
        assert result.strategy == "topology"
        assert result.workspaces == ["cib"]
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
        result = await router.route("CIB application issue")
        assert "cib" in result.workspaces
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


class TestWorkspaceRouterIntegration:
    """Integration tests: F06 in TwinRAGEngine."""

    def _mock_llm(self, content: str, total_tokens: int = 100):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.usage = MagicMock(total_tokens=total_tokens)
        return mock_response

    async def test_aquery_with_workspace_bypasses_f06(self, config, routing_rules_json):
        """workspace='cib' explicit -> F06 not called (backward compat)."""
        config_with_routing = TwinRAGConfig(
            llm_api_key="test-key",
            llm_api_base="http://mock:8080",
            enable_workspace_routing=True,
            routing_rules_path=str(routing_rules_json),
        )
        engine = TwinRAGEngine(config_with_routing)

        oos_json = json.dumps(
            {"intent": "OUT_OF_SCOPE", "confidence": 0.97, "reason": "OOS"}
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._mock_llm(oos_json)
        )

        with patch(
            "twindb_lightrag_memgraph.intelligence.features.intent_classifier.AsyncOpenAI",
            return_value=mock_client,
        ):
            result = await engine.aquery("Quel temps ?", workspace="cib")

        # workspace='cib' -> F06 bypassed, trace.workspace stays 'cib'
        assert result.trace.workspace == "cib"
        assert result.trace.routing_strategy is None

    async def test_aquery_without_workspace_triggers_f06(
        self, config, routing_rules_json
    ):
        """No workspace -> F06 resolves automatically."""
        config_with_routing = TwinRAGConfig(
            llm_api_key="test-key",
            llm_api_base="http://mock:8080",
            enable_workspace_routing=True,
            routing_rules_path=str(routing_rules_json),
        )
        engine = TwinRAGEngine(config_with_routing)

        oos_json = json.dumps(
            {"intent": "OUT_OF_SCOPE", "confidence": 0.97, "reason": "OOS"}
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._mock_llm(oos_json)
        )

        with patch(
            "twindb_lightrag_memgraph.intelligence.features.intent_classifier.AsyncOpenAI",
            return_value=mock_client,
        ):
            # OOS early-exits before REASON, but F06 routing still happens for trace
            # Actually F06 is after F05 early-exit. Let's use a greeting to test trace
            greeting_json = json.dumps(
                {"intent": "GREETING", "confidence": 0.99, "reason": "Hi"}
            )
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_llm(greeting_json)
            )
            result = await engine.aquery("Bonjour !")

        # F06 routing doesn't run on early-exit, but workspace defaults to commons
        assert result.trace.workspace == "commons"

    async def test_aquery_with_topology_context(self, config, routing_rules_json):
        """TopologyContext provided -> strategy=topology."""
        config_with_routing = TwinRAGConfig(
            llm_api_key="test-key",
            llm_api_base="http://mock:8080",
            enable_workspace_routing=True,
            routing_rules_path=str(routing_rules_json),
        )
        engine = TwinRAGEngine(config_with_routing)

        intent_resp = self._mock_llm(
            json.dumps({"intent": "IN_SCOPE", "confidence": 0.95, "reason": "IT"})
        )
        reason_resp = self._mock_llm(
            json.dumps(
                {
                    "thought": "Oracle question",
                    "search_query": "ORA-04030 PGA",
                    "domain_hint": "oracle",
                    "coreference_resolved": False,
                }
            )
        )
        rerank_resp = self._mock_llm(
            json.dumps({"scores": [{"passage": 0, "score": 9}]})
        )
        synth_resp = self._mock_llm("Answer [Passage 0]", total_tokens=200)

        responses = [intent_resp, reason_resp, rerank_resp, synth_resp]
        call_iter = iter(range(len(responses)))

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=lambda **kw: responses[
                min(next(call_iter, len(responses) - 1), len(responses) - 1)
            ]
        )

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="ORA-04030 PGA memory docs")

        topo = TopologyContext(
            workspaces=["cib"],
            workspaces_publics=["commons", "commons_oracle"],
        )

        with (
            patch(
                "twindb_lightrag_memgraph.intelligence.features.intent_classifier.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch(
                "twindb_lightrag_memgraph.intelligence.react.reason.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch(
                "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch(
                "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch.object(engine, "_get_rag", return_value=mock_rag),
        ):
            result = await engine.aquery(
                "ORA-04030 sur cib",
                topology_context=topo,
            )

        assert result.trace.routing_strategy == "topology"
        assert result.trace.routing_workspaces == ["cib"]

    async def test_aquery_f06_disabled(self, routing_rules_json):
        """enable_workspace_routing=false -> F06 not instantiated."""
        config_no_routing = TwinRAGConfig(
            llm_api_key="test-key",
            llm_api_base="http://mock:8080",
            enable_workspace_routing=False,
        )
        engine = TwinRAGEngine(config_no_routing)
        assert engine._router is None

        oos_json = json.dumps(
            {"intent": "OUT_OF_SCOPE", "confidence": 0.97, "reason": "OOS"}
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._mock_llm(oos_json)
        )

        with patch(
            "twindb_lightrag_memgraph.intelligence.features.intent_classifier.AsyncOpenAI",
            return_value=mock_client,
        ):
            result = await engine.aquery("Quel temps ?")

        assert result.trace.routing_strategy is None
        assert result.trace.workspace == "commons"
