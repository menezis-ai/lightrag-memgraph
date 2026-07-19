"""Tests for OBSERVE phase (synthesis + citations)."""

from unittest.mock import patch

import pytest

from twindb_lightrag_memgraph.intelligence.models.schemas import AnswerStatus
from twindb_lightrag_memgraph.intelligence.react.observe import SynthesisEngine


class TestSynthesisEngine:
    """OBSERVE phase tests."""

    @pytest.fixture
    def engine(self, config):
        return SynthesisEngine(config)

    async def test_synthesize_with_chunks(
        self, engine, sample_chunks, mock_openai_client
    ):
        answer_text = (
            "La memoire PGA est insuffisante [Passage 0]. "
            "Utilisez V$PROCESS_MEMORY pour diagnostiquer [Passage 1]."
        )
        client = mock_openai_client(answer_text, total_tokens=300)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Pourquoi ORA-04030 ?", sample_chunks)

        assert result.answer != ""
        assert len(result.citations) == 2
        assert result.citations[0].passage_index == 0
        assert result.citations[1].passage_index == 1
        assert result.tokens_used == 300
        assert result.answer_status == AnswerStatus.GROUNDED

    async def test_synthesize_no_chunks(self, engine):
        result = await engine.synthesize("Question", [])
        assert "aucune information" in result.answer.lower()
        assert result.citations == []
        assert result.answer_status == AnswerStatus.INSUFFICIENT_INFORMATION

    async def test_synthesize_citations_extraction(
        self, engine, sample_chunks, mock_openai_client
    ):
        answer_text = "Info from [Passage 0] and [Passage 2] and again [Passage 0]."
        client = mock_openai_client(answer_text)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Question", sample_chunks)
        # Should deduplicate: Passage 0 and Passage 2 only
        assert len(result.citations) == 2
        indices = [c.passage_index for c in result.citations]
        assert 0 in indices
        assert 2 in indices

    async def test_synthesize_rejects_phantom_citations(
        self, engine, sample_chunks, mock_openai_client
    ):
        answer_text = "Unsupported claim [Passage 99]. Real fact [Passage 1]."
        client = mock_openai_client(answer_text)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Question", sample_chunks)
        assert result.citations == []
        assert result.answer_status == AnswerStatus.CITATION_VALIDATION_FAILED
        assert "[Passage" not in result.answer

    async def test_synthesize_flags_missing_citations(
        self, engine, sample_chunks, mock_openai_client
    ):
        client = mock_openai_client("Une reponse confiante sans preuve.")
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Question", sample_chunks)

        assert result.citations == []
        assert result.answer_status == AnswerStatus.CITATION_VALIDATION_FAILED

    async def test_synthesize_rejects_valid_and_malformed_citation_mix(
        self, engine, sample_chunks, mock_openai_client
    ):
        client = mock_openai_client("Fait [Passage 0]. Autre fait [Passage -1].")
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Question", sample_chunks)

        assert result.citations == []
        assert result.answer_status == AnswerStatus.CITATION_VALIDATION_FAILED
        assert "[Passage" not in result.answer

    async def test_synthesize_cleans_passage_refs(
        self, engine, sample_chunks, mock_openai_client
    ):
        answer_text = "La reponse est X [Passage 0]."
        client = mock_openai_client(answer_text)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Question", sample_chunks)
        assert "[Passage" not in result.answer

    async def test_synthesize_with_conversation_history(
        self, engine, sample_chunks, mock_openai_client
    ):
        history = [
            {"role": "user", "content": "Previous question about Oracle"},
            {"role": "assistant", "content": "Previous answer about PGA"},
        ]
        answer_text = "Based on context [Passage 0]."
        client = mock_openai_client(answer_text)
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize(
                "Follow-up question", sample_chunks, history
            )
        assert result.answer != ""

    async def test_synthesize_fallback_on_error(self, engine, sample_chunks):
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            side_effect=Exception("API Error"),
        ):
            result = await engine.synthesize("Question", sample_chunks)
        assert "Erreur" in result.answer
        assert result.citations == []
        assert result.answer_status == AnswerStatus.QUERY_FAILED

    async def test_synthesize_error_hides_exception_details(
        self, engine, sample_chunks, caplog
    ):
        secret = "SECRET_TOKEN=observe-exception-secret-123"
        with (
            caplog.at_level("ERROR", logger="twin_rag_intelligence.observe"),
            patch(
                "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
                side_effect=RuntimeError(f"upstream echoed {secret}"),
            ),
        ):
            result = await engine.synthesize("Question", sample_chunks)

        assert "Erreur" in result.answer
        assert secret not in result.answer
        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text
        assert result.citations == []
        assert result.answer_status == AnswerStatus.QUERY_FAILED

    async def test_zero_rerank_score_is_preserved(
        self, engine, sample_chunks, mock_openai_client
    ):
        sample_chunks[0].score = 0.8
        sample_chunks[0].rerank_score = 0.0
        client = mock_openai_client("Fait supporte [Passage 0].")
        with patch(
            "twindb_lightrag_memgraph.intelligence.react.observe.AsyncOpenAI",
            return_value=client,
        ):
            result = await engine.synthesize("Question", sample_chunks)

        assert result.answer_status == AnswerStatus.GROUNDED
        assert result.citations[0].score == 0.0


class TestPassageMarkerPattern:
    """ReDoS hardening of _PASSAGE_MARKER_PATTERN (Sonar S5852).

    The captured language must stay identical to the historical
    ``\\[Passage\\s+([^\\]]+)\\]`` for every matched form the synthesis
    validation relies on, while adversarial unclosed-bracket input scans
    in linear time.
    """

    def test_marker_forms_match_like_before(self):
        from twindb_lightrag_memgraph.intelligence.react.observe import (
            _PASSAGE_MARKER_PATTERN,
        )

        assert _PASSAGE_MARKER_PATTERN.findall("ok [Passage 3].") == ["3"]
        assert _PASSAGE_MARKER_PATTERN.findall("[Passage   7]") == ["7"]
        assert _PASSAGE_MARKER_PATTERN.findall("[Passage abc]") == ["abc"]
        assert _PASSAGE_MARKER_PATTERN.findall("[Passage 1] x [Passage 2]") == [
            "1",
            "2",
        ]
        # Whitespace-only content still COUNTS as a (malformed) marker so
        # the citation validation mismatch behavior is preserved.
        assert len(_PASSAGE_MARKER_PATTERN.findall("[Passage   ]")) == 1
        # Non-markers stay non-markers.
        assert _PASSAGE_MARKER_PATTERN.findall("[Passage]") == []
        assert _PASSAGE_MARKER_PATTERN.findall("[Passages 1]") == []

    def test_sub_strips_markers(self):
        from twindb_lightrag_memgraph.intelligence.react.observe import (
            _PASSAGE_MARKER_PATTERN,
        )

        assert (
            _PASSAGE_MARKER_PATTERN.sub("", "Fait A [Passage 1] fait B [Passage x]")
            == "Fait A  fait B "
        )

    def test_adversarial_space_run_completes_fast(self):
        import time

        from twindb_lightrag_memgraph.intelligence.react.observe import (
            _PASSAGE_MARKER_PATTERN,
        )

        # Old pattern was polynomial here: "[Passage" + N spaces, no "]".
        hostile = "[Passage" + " " * 50_000 + "x" * 1_000
        start = time.perf_counter()
        assert _PASSAGE_MARKER_PATTERN.findall(hostile) == []
        assert time.perf_counter() - start < 0.5
