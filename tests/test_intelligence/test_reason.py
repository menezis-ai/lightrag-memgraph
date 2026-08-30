"""Tests for REASON phase."""

from unittest.mock import patch

import pytest

from twindb_lightrag_memgraph.intelligence.react.reason import ReasoningEngine


class TestReasoningEngine:
    """REASON phase tests."""

    @pytest.fixture
    def engine(self, config):
        return ReasoningEngine(config)

    async def test_analyze_basic_question(
        self, engine, mock_openai_client, reasoning_result_json
    ):
        client = mock_openai_client(reasoning_result_json)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.reason.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.analyze("Pourquoi ORA-04030 sur srv-demo-01 ?", [])
        assert "ORA-04030" in result.search_query
        assert result.domain_hint == "oracle"
        assert result.coreference_resolved is False

    async def test_analyze_with_coreference(
        self, engine, mock_openai_client, reasoning_coref_json
    ):
        history = [
            {"role": "user", "content": "J'ai une erreur ORA-04030"},
            {"role": "assistant", "content": "ORA-04030 est lie a la memoire PGA."},
        ]
        client = mock_openai_client(reasoning_coref_json)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.reason.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.analyze("Pourquoi ca a plante ?", history)
        assert result.coreference_resolved is True
        assert "ORA-04030" in result.search_query

    async def test_analyze_no_history(
        self, engine, mock_openai_client, reasoning_result_json
    ):
        client = mock_openai_client(reasoning_result_json)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.reason.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.analyze("Comment resoudre ORA-04030 ?", [])
        assert result.search_query != ""
        assert result.original_question == "Comment resoudre ORA-04030 ?"

    async def test_analyze_fallback_on_error(self, engine):
        """On LLM error, should use raw question as search query."""
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.reason.AsyncOpenAI",
            side_effect=Exception("Timeout"),
        ):
            result = await engine.analyze("Pourquoi ORA-04030 ?", [])
        assert result.search_query == "Pourquoi ORA-04030 ?"
        assert "Fallback" in result.thought

    async def test_analyze_long_history_truncated(
        self, engine, mock_openai_client, reasoning_result_json
    ):
        """History should be truncated to conversation_memory_depth."""
        history = [{"role": "user", "content": f"Message {i}"} for i in range(20)]
        client = mock_openai_client(reasoning_result_json)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.reason.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.analyze("Question", history)
        assert result.search_query != ""
