"""Scientific contracts for structured retrieval and workspace fusion."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from twindb_lightrag_memgraph.intelligence.react.act import (
    ChunkResult,
    SearchEngine,
)


def _chunk(chunk_id: str, score: float, workspace: str = "upstream") -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        text=f"Evidence for {chunk_id}",
        score=score,
        source_workspace=workspace,
    )


async def test_hybrid_search_uses_structured_chunks_with_provenance(config):
    engine = SearchEngine(config)
    rag = MagicMock(workspace="demo")
    rag.aquery_data = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "chunks": [
                    {
                        "chunk_id": "chunk-17",
                        "content": "The exact passage retrieved by LightRAG.",
                        "score": 0.91,
                        "full_doc_id": "doc-4",
                        "file_path": "/runbook/oracle.pdf",
                        "reference_id": "7",
                    }
                ]
            },
        }
    )
    rag.aquery = AsyncMock(return_value="A generated answer must not be used.")

    chunks = await engine.hybrid_search(rag, "ORA-04030", config)

    rag.aquery_data.assert_awaited_once()
    rag.aquery.assert_not_awaited()
    assert chunks == [
        ChunkResult(
            chunk_id="chunk-17",
            text="The exact passage retrieved by LightRAG.",
            score=0.91,
            source_workspace="demo",
            document_id="doc-4",
            document_path="/runbook/oracle.pdf",
            metadata={"retrieval_rank": 1, "reference_id": "7"},
        )
    ]


async def test_answer_only_lightrag_is_not_converted_to_a_passage(config):
    engine = SearchEngine(config)

    class AnswerOnlyRag:
        workspace = "legacy"

        def __init__(self):
            self.aquery = AsyncMock(return_value="Confident generated synthesis")

    rag = AnswerOnlyRag()

    chunks = await engine.hybrid_search(rag, "query", config)  # type: ignore[arg-type]

    assert chunks == []
    rag.aquery.assert_not_awaited()
    assert engine._parse_lightrag_result("Generated synthesis", rag) == []  # type: ignore[arg-type]


def test_rrf_uses_rank_not_incommensurable_raw_score(config):
    engine = SearchEngine(config)
    cosine = _chunk("vector-099", 0.99)
    lexical_leader = _chunk("lexical-leader", 30.0)
    bm25 = _chunk("bm25-25", 25.0)

    fused = engine.fuse_and_dedup(
        [[cosine], [lexical_leader, bm25]],
        ["vector-workspace", "lexical-workspace"],
    )

    by_id = {chunk.chunk_id: chunk for chunk in fused}
    assert by_id["vector-099"].score == pytest.approx(1 / 61)
    assert by_id["bm25-25"].score == pytest.approx(1 / 62)
    assert [chunk.chunk_id for chunk in fused].index("vector-099") < [
        chunk.chunk_id for chunk in fused
    ].index("bm25-25")


def test_rrf_aggregates_same_chunk_across_workspaces(config):
    engine = SearchEngine(config)
    shared_private = _chunk("shared", 0.99, "private-raw")
    shared_public = _chunk("shared", 25.0, "public-raw")

    fused = engine.fuse_and_dedup(
        [
            [shared_private],
            [_chunk("public-first", 100.0), shared_public],
        ],
        ["private", "public"],
    )

    shared = next(chunk for chunk in fused if chunk.chunk_id == "shared")
    assert shared.score == pytest.approx((1 / 61) + (1 / 62))
    assert shared.source_workspace == "private"
    assert shared.metadata["source_workspaces"] == ["private", "public"]
    assert shared.metadata["rrf_ranks"] == {"private": 1, "public": 2}
    assert shared.metadata["retrieval_scores"] == {
        "private": 0.99,
        "public": 25.0,
    }
    assert shared_private.score == 0.99
    assert shared_public.score == 25.0


def test_rrf_tie_order_is_deterministic_across_workspace_order(config):
    engine = SearchEngine(config)
    alpha = _chunk("alpha", 0.1)
    beta = _chunk("beta", 999.0)

    forward = engine.fuse_and_dedup([[beta], [alpha]], ["zeta", "alpha"])
    reversed_input = engine.fuse_and_dedup([[alpha], [beta]], ["alpha", "zeta"])

    assert [chunk.chunk_id for chunk in forward] == ["alpha", "beta"]
    assert [chunk.chunk_id for chunk in reversed_input] == ["alpha", "beta"]
