"""Tests for the Twin overlay POST /query route.

The endpoint wraps LightRAG's `aquery` and pairs the synthesised
response with a structured `sources` list pulled from `chunks_vdb`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server.twin_query_routes import (
    _normalize_answer,
    build_twin_query_router,
)


class TestNormalizeAnswer:
    def test_strips_references_block(self):
        text = "Answer body.\n\n### References - [1] runbook.pdf"
        assert _normalize_answer(text) == "Answer body."

    def test_strips_h2_references_too(self):
        text = "Answer.\n## References\n[1] foo"
        assert _normalize_answer(text) == "Answer."

    def test_case_insensitive(self):
        assert _normalize_answer("x\n### references - [1] a") == "x"

    def test_passthrough_when_no_references(self):
        assert _normalize_answer("Just an answer.") == "Just an answer."

    def test_inline_brackets_preserved(self):
        text = "See [3] for the runbook.\n\n### References - [3] foo"
        assert _normalize_answer(text) == "See [3] for the runbook."

    def test_strips_french_references_with_accent(self):
        text = "Réponse en français.\n\n### Références - Aucun document de référence."
        assert _normalize_answer(text) == "Réponse en français."

    def test_strips_french_references_without_accent(self):
        # LLM sometimes drops the accent (encoding round-trip)
        text = "Reponse.\n\n### References - [1] foo"
        assert _normalize_answer(text) == "Reponse."


class FakeChunksVdb:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self.last_query = query
        self.last_top_k = top_k
        return self.rows


class FakeDocStatus:
    def __init__(
        self,
        mapping: dict[str, str],
        docs: dict[str, dict[str, Any]] | None = None,
    ):
        self.mapping = mapping
        self.docs = docs or {}

    async def get_docs_by_chunks(self, chunk_ids):
        return {
            self.mapping[cid]: object()
            for cid in chunk_ids
            if cid in self.mapping
        }

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)


class FakeRag:
    def __init__(
        self,
        *,
        answer: str = "synthesised answer",
        query_data: dict[str, Any] | None = None,
        stream_chunks: list[Any] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        chunk_to_doc: dict[str, str] | None = None,
        docs: dict[str, dict[str, Any]] | None = None,
    ):
        self.answer = answer
        self.query_data = query_data or {
            "status": "success",
            "message": "Query executed successfully",
            "data": {},
            "metadata": {},
        }
        self.stream_chunks = stream_chunks
        self.calls: list[tuple[str, Any]] = []
        self.data_calls: list[tuple[str, Any]] = []
        self.chunks_vdb = FakeChunksVdb(chunks or [])
        self.doc_status = FakeDocStatus(chunk_to_doc or {}, docs)

    async def aquery(self, query: str, *, param):
        self.calls.append((query, param))
        if getattr(param, "stream", False) and self.stream_chunks is not None:
            async def gen():
                for chunk in self.stream_chunks or []:
                    yield chunk

            return gen()
        return self.answer

    async def aquery_data(self, query: str, *, param):
        self.data_calls.append((query, param))
        return self.query_data


@pytest.fixture()
async def make_client():
    async def _make(rag: FakeRag):
        app = FastAPI()
        app.include_router(build_twin_query_router(lambda: rag))
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    return _make


class TestQueryEndpoint:
    async def test_returns_response_and_sources(self, make_client):
        rag = FakeRag(
            answer="Restart Oracle by …",
            chunks=[
                {
                    "id": "chunk-aa",
                    "file_path": "/cib/runbooks/oracle.pdf",
                    "chunk_order_index": 3,
                    "score": 0.92,
                },
                {
                    "id": "chunk-bb",
                    "file_path": "/cib/runbooks/rhel.pdf",
                    "chunk_order_index": 7,
                    "score": 0.88,
                },
            ],
            chunk_to_doc={"chunk-aa": "doc-oracle", "chunk-bb": "doc-rhel"},
        )

        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "How do I restart Oracle?", "mode": "mix", "top_k": 5},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["response"] == "Restart Oracle by …"
        assert len(body["sources"]) == 2

        first, second = body["sources"]
        assert first["n"] == 1
        assert first["name"] == "/cib/runbooks/oracle.pdf"
        assert first["meta"] == "chunk 3"
        assert first["score"] == pytest.approx(0.92)
        assert first["doc_id"] == "doc-oracle"
        assert first["chunk_id"] == "chunk-aa"

        assert second["n"] == 2
        assert second["doc_id"] == "doc-rhel"

        # The endpoint forwarded the requested top_k to chunks_vdb
        assert rag.chunks_vdb.last_top_k == 5

    async def test_strips_trailing_references_block_from_response(
        self, make_client
    ):
        rag = FakeRag(
            answer=(
                "Restart Oracle by stopping listeners [1] then `shutdown immediate` [2].\n\n"
                "### References - [1] runbook.pdf [2] rhel.pdf"
            ),
            chunks=[
                {"id": "c1", "file_path": "runbook.pdf", "score": 0.9},
                {"id": "c2", "file_path": "rhel.pdf", "score": 0.8},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "q", "top_k": 2})
        assert r.status_code == 200
        body = r.json()
        assert "### References" not in body["response"]
        assert body["response"].endswith("[2].")
        assert len(body["sources"]) == 2

    async def test_only_need_context_skips_source_enrichment(self, make_client):
        rag = FakeRag(answer="raw context blob")
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "anything", "only_need_context": True},
            )

        assert r.status_code == 200
        assert r.json() == {"response": "raw context blob", "sources": []}
        # chunks_vdb should not have been touched in context-only mode
        assert rag.chunks_vdb.last_query is None

    async def test_advanced_query_params_forward_to_aquery(self, make_client):
        rag = FakeRag(answer="advanced")
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={
                    "query": "advanced retrieval",
                    "mode": "hybrid",
                    "top_k": 11,
                    "chunk_top_k": 7,
                    "max_total_tokens": 1234,
                    "history_turns": 2,
                    "user_prompt": "prefer runbook citations",
                    "enable_rerank": False,
                },
            )

        assert r.status_code == 200
        assert len(rag.calls) == 1
        query, param = rag.calls[0]
        assert query == "advanced retrieval"
        assert param.mode == "hybrid"
        assert param.top_k == 11
        assert param.chunk_top_k == 7
        assert param.max_total_tokens == 1234
        assert param.history_turns == 2
        assert param.user_prompt == "prefer runbook citations"
        assert param.enable_rerank is False
        assert param.stream is False

    async def test_tag_filter_forwards_to_aquery(self, make_client):
        rag = FakeRag(answer="tagged")
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={
                    "query": "tagged retrieval",
                    "tag_filter": {"all": ["oracle", "rman"], "any": []},
                },
            )

        assert r.status_code == 200
        assert len(rag.calls) == 1
        _query, param = rag.calls[0]
        assert param.tag_filter == {"all": ["oracle", "rman"], "any": []}

    async def test_tag_filter_is_absent_when_omitted(self, make_client):
        rag = FakeRag(answer="untagged")
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "untagged retrieval"})

        assert r.status_code == 200
        assert len(rag.calls) == 1
        _query, param = rag.calls[0]
        assert getattr(param, "tag_filter", None) is None

    async def test_invalid_tag_filter_returns_422(self, make_client):
        rag = FakeRag(answer="invalid")
        client = await make_client(rag)
        async with client:
            unknown_key = await client.post(
                "/query",
                json={
                    "query": "bad",
                    "tag_filter": {"none": ["oracle"]},
                },
            )
            non_list_value = await client.post(
                "/query",
                json={
                    "query": "bad",
                    "tag_filter": {"all": "oracle"},
                },
            )

        assert unknown_key.status_code == 422
        assert non_list_value.status_code == 422

    async def test_only_need_prompt_skips_source_enrichment(self, make_client):
        rag = FakeRag(answer="prompt that would be sent")
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "anything", "only_need_prompt": True},
            )

        assert r.status_code == 200
        assert r.json() == {
            "response": "prompt that would be sent",
            "sources": [],
        }

    async def test_chunks_vdb_failure_returns_empty_sources_not_500(
        self, make_client
    ):
        rag = FakeRag(answer="ok", chunks=[])

        async def boom(*_a, **_kw):
            raise RuntimeError("memgraph down")

        rag.chunks_vdb.query = boom  # type: ignore[assignment]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "anything"})

        assert r.status_code == 200
        assert r.json() == {"response": "ok", "sources": []}

    async def test_aquery_failure_returns_500(self, make_client):
        rag = FakeRag(answer="never returned")

        async def boom(*_a, **_kw):
            raise RuntimeError("LLM down")

        rag.aquery = boom  # type: ignore[assignment]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "anything"})

        assert r.status_code == 500
        assert "LLM down" in r.json()["detail"]

    async def test_stream_endpoint_emits_ndjson_tokens_then_sources(
        self, make_client
    ):
        import json as _json

        rag = FakeRag(
            stream_chunks=[
                "Restart ",
                {"response": "Oracle "},
                b"safely",
            ],
            chunks=[
                {"id": "c1", "file_path": "/a/runbook.pdf", "score": 0.9},
                {"id": "c2", "file_path": "/a/rhel.pdf", "score": 0.7},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/stream",
                json={
                    "query": "How do I restart Oracle?",
                    "mode": "mix",
                    "top_k": 2,
                    "chunk_top_k": 4,
                    "enable_rerank": True,
                    "user_prompt": "short answer",
                },
            )

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")

        events = [_json.loads(line) for line in r.text.splitlines() if line.strip()]
        token_events = [e for e in events if e["type"] == "token"]
        source_events = [e for e in events if e["type"] == "sources"]
        assert "".join(e["value"] for e in token_events) == "Restart Oracle safely"
        assert len(source_events) == 1
        sources = source_events[0]["value"]
        assert [s["name"] for s in sources] == ["/a/runbook.pdf", "/a/rhel.pdf"]
        assert [s["n"] for s in sources] == [1, 2]

        # The original aquery stream-flag plumbing still works.
        query, param = rag.calls[0]
        assert query == "How do I restart Oracle?"
        assert param.stream is True
        assert param.chunk_top_k == 4
        assert param.enable_rerank is True
        assert param.user_prompt == "short answer"

    async def test_query_data_returns_structured_lightrag_payload(
        self, make_client
    ):
        payload = {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "chunks": [
                    {
                        "chunk_id": "chunk-aa",
                        "full_doc_id": "doc-oracle",
                        "content": "Oracle RMAN restart",
                        "reference_id": "1",
                    }
                ],
                "references": [{"reference_id": "1", "file_path": "oracle.pdf"}],
            },
            "metadata": {"query_mode": "mix"},
        }
        rag = FakeRag(query_data=payload)
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={"query": "structured retrieval", "mode": "mix"},
            )

        assert r.status_code == 200
        assert r.json() == payload
        assert len(rag.data_calls) == 1
        query, param = rag.data_calls[0]
        assert query == "structured retrieval"
        assert param.mode == "mix"
        assert param.stream is False

    async def test_query_data_forwards_extended_query_params(
        self, make_client
    ):
        rag = FakeRag()
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={
                    "query": "advanced structured retrieval",
                    "mode": "hybrid",
                    "response_type": "Bullet Points",
                    "top_k": 12,
                    "chunk_top_k": 6,
                    "max_entity_tokens": 1000,
                    "max_relation_tokens": 2000,
                    "max_total_tokens": 3000,
                    "hl_keywords": ["backup"],
                    "ll_keywords": ["rman"],
                    "conversation_history": [
                        {"role": "user", "content": "previous question"}
                    ],
                    "history_turns": 1,
                    "user_prompt": "return concise evidence",
                    "enable_rerank": False,
                    "tag_filter": {"all": ["rman"], "any": []},
                },
            )

        assert r.status_code == 200
        assert len(rag.data_calls) == 1
        _query, param = rag.data_calls[0]
        assert param.mode == "hybrid"
        assert param.response_type == "Bullet Points"
        assert param.top_k == 12
        assert param.chunk_top_k == 6
        assert param.max_entity_tokens == 1000
        assert param.max_relation_tokens == 2000
        assert param.max_total_tokens == 3000
        assert param.hl_keywords == ["backup"]
        assert param.ll_keywords == ["rman"]
        assert param.conversation_history == [
            {"role": "user", "content": "previous question"}
        ]
        assert param.history_turns == 1
        assert param.user_prompt == "return concise evidence"
        assert param.enable_rerank is False
        assert param.tag_filter == {"all": ["rman"], "any": []}

    async def test_query_data_tag_filter_filters_chunks_and_references(
        self, make_client
    ):
        rag = FakeRag(
            query_data={
                "status": "success",
                "message": "Query executed successfully",
                "data": {
                    "chunks": [
                        {
                            "chunk_id": "chunk-rman",
                            "full_doc_id": "doc-rman",
                            "content": "RMAN runbook",
                            "reference_id": "1",
                        },
                        {
                            "chunk_id": "chunk-vmware",
                            "full_doc_id": "doc-vmware",
                            "content": "VMware runbook",
                            "reference_id": "2",
                        },
                    ],
                    "entities": [
                        {
                            "entity_name": "RMAN",
                            "source_id": "chunk-rman",
                            "reference_id": "1",
                        },
                        {
                            "entity_name": "vSphere",
                            "source_id": "chunk-vmware",
                            "reference_id": "2",
                        },
                    ],
                    "relationships": [],
                    "references": [
                        {"reference_id": "1", "file_path": "oracle.pdf"},
                        {"reference_id": "2", "file_path": "vmware.pdf"},
                    ],
                },
                "metadata": {"query_mode": "mix"},
            },
            chunk_to_doc={
                "chunk-rman": "doc-rman",
                "chunk-vmware": "doc-vmware",
            },
            docs={
                "doc-rman": {"metadata": {"tags": ["rman", "oracle"]}},
                "doc-vmware": {"metadata": {"tags": ["vmware"]}},
            },
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={
                    "query": "tagged structured retrieval",
                    "tag_filter": {"all": ["rman"], "any": []},
                },
            )

        assert r.status_code == 200
        body = r.json()
        assert [c["chunk_id"] for c in body["data"]["chunks"]] == ["chunk-rman"]
        assert [e["entity_name"] for e in body["data"]["entities"]] == ["RMAN"]
        assert body["data"]["references"] == [
            {"reference_id": "1", "file_path": "oracle.pdf"}
        ]
        assert body["metadata"]["tag_filter"] == {"all": ["rman"], "any": []}

    async def test_query_data_empty_tag_filter_does_not_filter_unknown_rows(
        self, make_client
    ):
        payload = {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "chunks": [
                    {
                        "chunk_id": "chunk-without-doc-map",
                        "content": "still visible",
                        "reference_id": "1",
                    }
                ],
                "references": [{"reference_id": "1", "file_path": "loose.txt"}],
            },
            "metadata": {"query_mode": "naive"},
        }
        rag = FakeRag(query_data=payload)
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={
                    "query": "unfiltered structured retrieval",
                    "tag_filter": {"all": [], "any": []},
                },
            )

        assert r.status_code == 200
        assert r.json() == payload

    async def test_query_data_aquery_data_failure_returns_500(
        self, make_client
    ):
        rag = FakeRag()

        async def boom(*_a, **_kw):
            raise RuntimeError("KG unavailable")

        rag.aquery_data = boom  # type: ignore[assignment]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query/data", json={"query": "anything"})

        assert r.status_code == 500
        assert "KG unavailable" in r.json()["detail"]

    async def test_score_falls_back_to_rank_when_absent(self, make_client):
        rag = FakeRag(
            answer="x",
            chunks=[
                {"id": "a", "file_path": "/a"},
                {"id": "b", "file_path": "/b"},
                {"id": "c", "file_path": "/c"},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query", json={"query": "x", "top_k": 3}
            )
        scores = [s["score"] for s in r.json()["sources"]]
        # Rank-derived scores: 0.95, 0.725, 0.50
        assert scores == [pytest.approx(0.95), pytest.approx(0.725), pytest.approx(0.50)]

    async def test_missing_file_path_falls_back_to_chunk_id(self, make_client):
        rag = FakeRag(
            answer="x",
            chunks=[{"id": "chunk-no-path"}],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "x"})
        body = r.json()
        assert body["sources"][0]["name"] == "chunk-no-path"

    async def test_500_when_rag_not_captured(self, make_client):
        def boom():
            raise RuntimeError("rag not captured")

        app = FastAPI()
        app.include_router(build_twin_query_router(boom))
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            r = await c.post("/query", json={"query": "x"})
        assert r.status_code == 500
        assert "rag not captured" in r.json()["detail"]
