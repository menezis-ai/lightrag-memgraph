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
    build_twin_query_router,
)


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
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    async def get_docs_by_chunks(self, chunk_ids):
        return {
            self.mapping[cid]: object()
            for cid in chunk_ids
            if cid in self.mapping
        }


class FakeRag:
    def __init__(
        self,
        *,
        answer: str = "synthesised answer",
        chunks: list[dict[str, Any]] | None = None,
        chunk_to_doc: dict[str, str] | None = None,
    ):
        self.answer = answer
        self.calls: list[tuple[str, Any]] = []
        self.chunks_vdb = FakeChunksVdb(chunks or [])
        self.doc_status = FakeDocStatus(chunk_to_doc or {})

    async def aquery(self, query: str, *, param):
        self.calls.append((query, param))
        return self.answer


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
