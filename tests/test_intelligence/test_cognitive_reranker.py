"""Tests for F04: Cognitive Reranker."""

import json
from unittest.mock import patch

import pytest

from twindb_lightrag_memgraph.intelligence.features.cognitive_reranker import (
    CognitiveReranker,
)


class TestCognitiveReranker:
    """F04: Cognitive reranking tests."""

    @pytest.fixture
    def reranker(self, config):
        return CognitiveReranker(config)

    async def test_rerank_filters_low_scores(
        self, reranker, sample_chunks, mock_openai_client, reranking_scores_json
    ):
        client = mock_openai_client(reranking_scores_json)
        with patch(
            "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
            return_value=client,
        ):
            result = await reranker.rerank("ORA-04030 memory", sample_chunks)
        # Chunks with score >= 7.0 should be kept (passage 0: 9, passage 1: 8)
        assert len(result) == 2
        assert all((c.rerank_score or 0) >= 7.0 for c in result)

    async def test_rerank_sorted_by_score(
        self, reranker, sample_chunks, mock_openai_client, reranking_scores_json
    ):
        client = mock_openai_client(reranking_scores_json)
        with patch(
            "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
            return_value=client,
        ):
            result = await reranker.rerank("ORA-04030 memory", sample_chunks)
        scores = [c.rerank_score for c in result]
        assert scores == sorted(scores, reverse=True)

    async def test_rerank_fallback_on_strict_filter(
        self, reranker, sample_chunks, mock_openai_client
    ):
        """If all scores below threshold, should fallback to top-K."""
        all_low = json.dumps(
            {"scores": [{"passage": i, "score": 2} for i in range(len(sample_chunks))]}
        )
        client = mock_openai_client(all_low)
        with patch(
            "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
            return_value=client,
        ):
            result = await reranker.rerank("anything", sample_chunks)
        # Should fallback to top final_limit
        assert len(result) <= reranker.config.final_limit
        assert len(result) > 0

    async def test_rerank_accepts_compact_scores(
        self, reranker, sample_chunks, mock_openai_client
    ):
        compact = json.dumps(
            {"s": [{"p": 0, "v": 9}, {"p": 1, "v": 8}, {"p": 2, "v": 3}]}
        )
        client = mock_openai_client(compact)
        with patch(
            "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
            return_value=client,
        ):
            result = await reranker.rerank("ORA-04030 memory", sample_chunks)
        assert [c.chunk_id for c in result] == ["chunk_0", "chunk_1"]

    async def test_rerank_fallback_on_invalid_scores_payload(
        self, reranker, sample_chunks, mock_openai_client
    ):
        client = mock_openai_client(json.dumps({"scores": {"passage": 0, "score": 9}}))
        with patch(
            "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
            return_value=client,
        ):
            result = await reranker.rerank("ORA-04030 memory", sample_chunks)
        assert [c.chunk_id for c in result] == [
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "chunk_3",
        ][: reranker.config.final_limit]

    async def test_rerank_empty_chunks(self, reranker):
        result = await reranker.rerank("ORA-04030", [])
        assert result == []

    async def test_rerank_fallback_on_error(self, reranker, sample_chunks):
        """On LLM error, should fallback to top-K by raw score."""
        with patch(
            "twindb_lightrag_memgraph.intelligence.features.cognitive_reranker.AsyncOpenAI",
            side_effect=Exception("Connection refused"),
        ):
            result = await reranker.rerank("ORA-04030", sample_chunks)
        assert len(result) <= reranker.config.final_limit
        assert len(result) > 0
