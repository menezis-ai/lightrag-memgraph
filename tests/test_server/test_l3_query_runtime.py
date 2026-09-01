"""Runtime-selection and host-RAG adapter contract for Forgejo #117."""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from twindb_lightrag_memgraph._constants import (
    get_active_retrieval_filters,
    get_active_storage_folder,
)
from twindb_lightrag_memgraph.vector_impl import _capture_chunk_retrieval_scores
from twindb_lightrag_memgraph.server.query.l3_runtime import (
    L3QueryRuntime,
    build_l3_query_runtime,
    query_engine_mode,
)
from twindb_lightrag_memgraph.server.query.models import TwinQueryBody
from twindb_lightrag_memgraph.intelligence.models.schemas import (
    AnswerStatus,
    Citation,
    QueryResult,
    QueryTrace,
)
from twindb_lightrag_memgraph.intelligence.react.act import ChunkResult


def test_query_engine_defaults_to_l2_and_rejects_unknown_values():
    assert query_engine_mode({}) == "l2"
    assert query_engine_mode({"TWIN_RAG_QUERY_ENGINE": " L3 "}) == "l3"
    with pytest.raises(RuntimeError, match="must be 'l2' or 'l3'"):
        query_engine_mode({"TWIN_RAG_QUERY_ENGINE": "auto"})


def test_l2_mode_never_attempts_an_intelligence_import(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if "twindb_lightrag_memgraph.intelligence" in name:
            raise AssertionError(f"unexpected L3 import in flag-off mode: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert build_l3_query_runtime(lambda: object(), environ={}) is None


@pytest.mark.asyncio
async def test_l3_retrieval_reuses_host_rag_and_canonical_request_scope():
    observations = {}

    class HostRag:
        workspace = "physical-workspace"

        async def aquery_data(self, query, *, param):
            observations.update(
                query=query,
                rag=self,
                folder=get_active_storage_folder(),
                filters=get_active_retrieval_filters(),
                mode=param.mode,
                top_k=param.top_k,
            )
            _capture_chunk_retrieval_scores(
                "chunks",
                [{"id": "chunk-1", "similarity": 0.81}],
            )
            return {
                "status": "success",
                "data": {
                    "chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "content": "Scoped evidence",
                            "score": 0.8,
                        }
                    ]
                },
            }

    host_rag = HostRag()
    runtime = L3QueryRuntime(lambda: host_rag)

    class Search:
        @staticmethod
        def _parse_lightrag_result(result, rag, source_folder=None):
            observations["parsed_rag"] = rag
            observations["source_folder"] = source_folder
            item = result["data"]["chunks"][0]
            return [
                ChunkResult(
                    chunk_id=item["chunk_id"],
                    text=item["content"],
                    score=item["score"],
                    source_workspace=source_folder,
                )
            ]

    runtime._engine = type("Engine", (), {"search": Search()})()
    body = TwinQueryBody(
        query="Scoped question",
        mode="mix",
        top_k=7,
        min_score=0.42,
        tag_filter={"all": ["approved"]},
        doc_filter={"any": ["policy.pdf"]},
    )

    chunks = await runtime.retrieve(body=body, folder="finance", query=body.query)

    assert chunks == [
        ChunkResult(
            chunk_id="chunk-1",
            text="Scoped evidence",
            score=0.8,
            source_workspace="finance",
            metadata={"measured_retrieval_score": 0.81},
        )
    ]
    assert observations["rag"] is host_rag
    assert observations["parsed_rag"] is host_rag
    assert observations["source_folder"] == "finance"
    assert observations["folder"] == "finance"
    assert observations["mode"] == "mix"
    assert observations["top_k"] == 7
    assert observations["filters"].min_score == 0.42
    assert observations["filters"].tag_all == frozenset({"approved"})
    assert observations["filters"].doc_any == frozenset({"policy.pdf"})
    assert get_active_storage_folder() is None
    assert get_active_retrieval_filters() is None


@pytest.mark.asyncio
async def test_query_result_projection_is_enriched_scored_and_non_sensitive(
    monkeypatch,
):
    trace = QueryTrace(
        question="SECRET raw question",
        workspace="finance",
        thought="SECRET model thought",
        resolved_query="SECRET rewritten query",
        early_exit=None,
        fallbacks=["reason_fallback"],
    )
    result = QueryResult(
        answer="Supported fact [1]",
        answer_status=AnswerStatus.GROUNDED,
        citations=[
            Citation(
                passage_index=0,
                text="x" * 500,
                document_id=None,
                document_path="policy.pdf",
                source_workspace="finance",
                score=9.0,
                retrieval_score=0.82,
                chunk_id="chunk-1",
            )
        ],
        trace=trace,
    )

    async def enrich_links(sources, folder):
        assert folder == "finance"
        sources[0]["source_links"] = [{"url": "https://provenance.test/policy"}]

    from twindb_lightrag_memgraph.server.query import l3_runtime as l3_module

    monkeypatch.setattr(
        l3_module,
        "_enrich_sources_with_source_links",
        enrich_links,
    )
    host_rag = SimpleNamespace(
        text_chunks=SimpleNamespace(
            get_by_ids=AsyncMock(
                return_value=[
                    {
                        "chunk_id": "chunk-1",
                        "full_doc_id": "doc-1",
                        "content": (
                            "Irrelevant context.\n\n"
                            "Supported fact from SECRET source excerpt."
                        ),
                    }
                ]
            )
        )
    )
    runtime = L3QueryRuntime(lambda: host_rag)
    projected = await runtime.project(result, folder="finance")

    assert projected["sources"] == [
        {
            "n": 1,
            "type": "file",
            "name": "policy.pdf",
            "meta": "finance",
            "score": 0.82,
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "source_links": [{"url": "https://provenance.test/policy"}],
            "anchor": {
                "start": 21,
                "end": 63,
                "paragraph_idx": 1,
                "paragraph_count": 2,
                "confidence": 1.0,
                "method": "lexical_overlap",
            },
        }
    ]
    assert projected["trace"] == {
        "engine": "l3",
        "degraded": True,
        "fallbacks": ["reason_fallback"],
        "early_exit": None,
    }
    assert "SECRET" not in json.dumps(projected)


@pytest.mark.asyncio
async def test_l3_stream_uses_frontend_stages_and_bounded_public_trace(monkeypatch):
    from twindb_lightrag_memgraph.server.query import router

    trace = QueryTrace(
        question="SECRET raw question",
        workspace="finance",
        thought="SECRET thought",
        fallbacks=["intent_fallback"],
    )
    result = QueryResult(
        answer="Streamed answer [1]",
        answer_status=AnswerStatus.GROUNDED,
        citations=[
            Citation(
                passage_index=0,
                text="private excerpt",
                document_id="doc-1",
                document_path="policy.pdf",
                source_workspace="finance",
                retrieval_score=0.7,
                chunk_id="chunk-1",
            )
        ],
        trace=trace,
    )

    class Runtime:
        async def aquery(self, *, on_token, on_stage, **_kwargs):
            await on_stage("generation")
            await on_token("Streamed answer ")
            await on_token("[1]")
            return result

        async def project(self, value, *, folder):
            assert value is result
            assert folder == "finance"
            return {
                "response": result.answer,
                "sources": [{"n": 1, "name": "policy.pdf"}],
                "answer_status": "grounded",
                "trace": {
                    "engine": "l3",
                    "degraded": True,
                    "fallbacks": ["intent_fallback"],
                    "early_exit": None,
                },
            }

    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    monkeypatch.setattr(router, "_record_retrieval_activity", AsyncMock())
    lines = [
        json.loads(line)
        async for line in router._generate_l3_query_stream(
            Runtime(),
            TwinQueryBody(query="question"),
            request,
            "finance",
            "chat-model",
        )
    ]

    assert [event["value"] for event in lines if event["type"] == "stage"] == [
        "retrieval",
        "generation",
        "sources",
    ]
    assert (
        "".join(event["value"] for event in lines if event["type"] == "token")
        == "Streamed answer [1]"
    )
    assert (
        next(event for event in lines if event["type"] == "status")["value"]
        == "grounded"
    )
    public_trace = next(event for event in lines if event["type"] == "trace")["value"]
    assert public_trace["fallbacks"] == ["intent_fallback"]
    assert "SECRET" not in json.dumps(lines)


@pytest.mark.asyncio
async def test_l3_stream_early_exit_emits_generation_before_token(monkeypatch):
    from twindb_lightrag_memgraph.server.query import router

    result = QueryResult(
        answer="Bonjour.",
        answer_status=AnswerStatus.NO_RETRIEVAL,
        citations=[],
        trace=QueryTrace(
            question="Bonjour",
            workspace="finance",
            early_exit="GREETING",
        ),
    )

    class Runtime:
        async def aquery(self, **_kwargs):
            return result

        async def project(self, _value, *, folder):
            assert folder == "finance"
            return {
                "response": "Bonjour.",
                "sources": [],
                "answer_status": "no_retrieval",
                "trace": {
                    "engine": "l3",
                    "degraded": False,
                    "fallbacks": [],
                    "early_exit": "GREETING",
                },
            }

    monkeypatch.setattr(router, "_record_retrieval_activity", AsyncMock())
    lines = [
        json.loads(line)
        async for line in router._generate_l3_query_stream(
            Runtime(),
            TwinQueryBody(query="Bonjour"),
            SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
            "finance",
            None,
        )
    ]

    generation_index = next(
        index
        for index, event in enumerate(lines)
        if event == {"type": "stage", "value": "generation"}
    )
    token_index = next(
        index for index, event in enumerate(lines) if event["type"] == "token"
    )
    assert generation_index < token_index


@pytest.mark.asyncio
async def test_l3_stream_projection_error_emits_terminal_degradation(monkeypatch):
    from twindb_lightrag_memgraph.server.query import router, streaming

    class Runtime:
        async def aquery(self, *, on_stage, **_kwargs):
            await on_stage("generation")
            return object()

        async def project(self, _value, *, folder):
            assert folder == "finance"
            raise RuntimeError("projection unavailable")

    record_activity = AsyncMock()
    monkeypatch.setattr(streaming, "_record_retrieval_activity", record_activity)
    lines = [
        json.loads(line)
        async for line in router._generate_l3_query_stream(
            Runtime(),
            TwinQueryBody(query="question"),
            SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
            "finance",
            None,
        )
    ]

    assert [event["value"] for event in lines if event["type"] == "stage"] == [
        "retrieval",
        "generation",
        "sources",
    ]
    assert lines[-2:] == [
        {"type": "status", "value": "source_projection_failed"},
        {"type": "sources", "value": []},
    ]
    record_activity.assert_awaited_once()
    assert record_activity.await_args.kwargs["sources_count"] == 0


@pytest.mark.asyncio
async def test_l3_runtime_propagates_response_type():
    runtime = L3QueryRuntime(lambda: object())
    runtime._engine = SimpleNamespace(aquery=AsyncMock(return_value=object()))
    body = TwinQueryBody(query="question", response_type="Bullet Points")

    await runtime.aquery(body=body, folder="finance")

    assert runtime._engine.aquery.await_args.kwargs["response_type"] == "Bullet Points"


def test_l3_runtime_neutralizes_current_and_legacy_routing_flags(monkeypatch):
    monkeypatch.setenv("TWIN_RAG_ENABLE_FOLDER_ROUTING", "true")
    monkeypatch.setenv("TWIN_RAG_ENABLE_WORKSPACE_ROUTING", "true")

    engine = L3QueryRuntime(lambda: object()).engine()

    assert engine.config.enable_folder_routing is False
    assert engine.config.enable_workspace_routing is False
    assert engine.config.effective_enable_folder_routing is False
    assert engine.folder_router is None
