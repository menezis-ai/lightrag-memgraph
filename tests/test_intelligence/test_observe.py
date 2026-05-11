"""Tests for OBSERVE phase (synthesis + citations)."""

from unittest.mock import patch

import pytest

from twindb_lightrag_memgraph.intelligence.react.observe import SynthesisEngine


class TestSynthesisEngine:
    """OBSERVE phase tests."""

    @pytest.fixture
    def engine(self, config):
        return SynthesisEngine(config)

    async def test_synthesize_with_chunks(self, engine, sample_chunks, mock_openai_client):
        answer_text = (
            "La memoire PGA est insuffisante [Passage 0]. "
            "Utilisez V$PROCESS_MEMORY pour diagnostiquer [Passage 1]."
        )
        client = mock_openai_client(answer_text, total_tokens=300)
        with patch("twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI", return_value=client):
            result = await engine.synthesize("Pourquoi ORA-04030 ?", sample_chunks)

        assert result.answer != ""
        assert len(result.citations) == 2
        assert result.citations[0].passage_index == 0
        assert result.citations[1].passage_index == 1
        assert result.tokens_used == 300

    async def test_synthesize_no_chunks(self, engine):
        result = await engine.synthesize("Question", [])
        assert "aucune information" in result.answer.lower()
        assert result.citations == []

    async def test_synthesize_citations_extraction(self, engine, sample_chunks, mock_openai_client):
        answer_text = "Info from [Passage 0] and [Passage 2] and again [Passage 0]."
        client = mock_openai_client(answer_text)
        with patch("twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI", return_value=client):
            result = await engine.synthesize("Question", sample_chunks)
        # Should deduplicate: Passage 0 and Passage 2 only
        assert len(result.citations) == 2
        indices = [c.passage_index for c in result.citations]
        assert 0 in indices
        assert 2 in indices

    async def test_synthesize_cleans_passage_refs(self, engine, sample_chunks, mock_openai_client):
        answer_text = "La reponse est X [Passage 0]."
        client = mock_openai_client(answer_text)
        with patch("twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI", return_value=client):
            result = await engine.synthesize("Question", sample_chunks)
        assert "[Passage" not in result.answer

    async def test_synthesize_with_conversation_history(self, engine, sample_chunks, mock_openai_client):
        history = [
            {"role": "user", "content": "Previous question about Oracle"},
            {"role": "assistant", "content": "Previous answer about PGA"},
        ]
        answer_text = "Based on context [Passage 0]."
        client = mock_openai_client(answer_text)
        with patch("twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI", return_value=client):
            result = await engine.synthesize("Follow-up question", sample_chunks, history)
        assert result.answer != ""

    async def test_synthesize_fallback_on_error(self, engine, sample_chunks):
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            side_effect=Exception("API Error"),
        ):
            result = await engine.synthesize("Question", sample_chunks)
        assert "Erreur" in result.answer
        assert result.citations == []
