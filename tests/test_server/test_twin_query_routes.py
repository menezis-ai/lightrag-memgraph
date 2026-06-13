"""Tests for the Twin overlay POST /query route.

The endpoint wraps LightRAG's `aquery` and pairs the synthesised
response with a structured `sources` list pulled from `chunks_vdb`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

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
    """Test double for ``LightRAG`` covering the three APIs the Twin
    overlay consumes: ``aquery`` (legacy / only_need_context paths),
    ``aquery_llm`` (nominal /query and /stream since TR-RET-02 step 2),
    and ``aquery_data`` (structured-only endpoint).

    The ``chunks`` fixture passed at construction time doubles as
    ``aquery_llm``'s ``data.references``/``data.chunks`` source — one
    reference per chunk with ``reference_id = str(i+1)`` and the chunk
    fields mirrored. This keeps the tests focused on the route's
    projection logic without duplicating chunk fixtures.
    """

    def __init__(
        self,
        *,
        answer: str = "synthesised answer",
        query_data: dict[str, Any] | None = None,
        stream_chunks: list[Any] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        chunk_to_doc: dict[str, str] | None = None,
        docs: dict[str, dict[str, Any]] | None = None,
        envelope_status: str = "success",
        envelope_failure_reason: str | None = None,
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
        self.llm_calls: list[tuple[str, Any]] = []
        self.data_calls: list[tuple[str, Any]] = []
        self.chunks_vdb = FakeChunksVdb(chunks or [])
        self.doc_status = FakeDocStatus(chunk_to_doc or {}, docs)
        self._chunks_fixture = chunks or []
        self._envelope_status = envelope_status
        self._envelope_failure_reason = envelope_failure_reason

    async def aquery(self, query: str, *, param):
        self.calls.append((query, param))
        if getattr(param, "stream", False) and self.stream_chunks is not None:
            async def gen():
                for chunk in self.stream_chunks or []:
                    yield chunk

            return gen()
        return self.answer

    def _build_envelope(self, *, is_streaming: bool) -> dict[str, Any]:
        """Synthesize an aquery_llm envelope from the fixture chunks."""
        references: list[dict[str, Any]] = []
        envelope_chunks: list[dict[str, Any]] = []
        for i, chunk in enumerate(self._chunks_fixture, start=1):
            ref_id = str(i)
            file_path = chunk.get("file_path") or ""
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or ""
            references.append(
                {"reference_id": ref_id, "file_path": file_path}
            )
            envelope_chunks.append(
                {
                    "reference_id": ref_id,
                    "content": chunk.get("content", ""),
                    "file_path": file_path,
                    "chunk_id": chunk_id,
                }
            )
        envelope: dict[str, Any] = {
            "status": self._envelope_status,
            "message": "Query processed",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": envelope_chunks,
                "references": references,
            },
            "metadata": (
                {"failure_reason": self._envelope_failure_reason}
                if self._envelope_failure_reason
                else {}
            ),
        }
        if is_streaming and self.stream_chunks is not None:
            async def gen():
                for chunk in self.stream_chunks or []:
                    yield chunk

            envelope["llm_response"] = {
                "content": None,
                "response_iterator": gen(),
                "is_streaming": True,
            }
        else:
            envelope["llm_response"] = {
                "content": self.answer,
                "response_iterator": None,
                "is_streaming": False,
            }
        return envelope

    async def aquery_llm(self, query: str, *, param):
        self.llm_calls.append((query, param))
        return self._build_envelope(
            is_streaming=bool(getattr(param, "stream", False))
        )

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

        # TR-RET-02 step 2 / audit C3: ``n`` mirrors the LightRAG
        # ``reference_id`` so the React port's ``[N]`` citation parser
        # stays aligned with the sources list.
        first, second = body["sources"]
        assert first["n"] == 1
        assert first["name"] == "/cib/runbooks/oracle.pdf"
        assert first["meta"] == "1 chunk"
        # ``score`` is no longer populated by the route (no second
        # vector pass) — kept at zero for now; can be filled from the
        # aquery_llm chunk metrics in a future iteration.
        assert first["score"] == 0.0
        assert first["doc_id"] == "doc-oracle"
        assert first["chunk_id"] == "chunk-aa"

        assert second["n"] == 2
        assert second["doc_id"] == "doc-rhel"

        # STRUCTURAL GUARD (audit C3): chunks_vdb MUST NOT be touched
        # on the nominal /query path. The whole point of the
        # aquery_llm migration is that sources come from the LightRAG
        # retrieval pipeline itself, not from a second vector pass.
        assert rag.chunks_vdb.last_query is None

    async def test_records_retrieval_activity(self, make_client):
        rag = FakeRag(
            answer="Restart Oracle by …",
            chunks=[
                {
                    "id": "chunk-aa",
                    "file_path": "/cib/runbooks/oracle.pdf",
                    "score": 0.92,
                },
            ],
        )
        store = type("Store", (), {"record_activity": AsyncMock()})()

        client = await make_client(rag)
        with patch(
            "twindb_lightrag_memgraph.server.webui_router.get_store",
            return_value=store,
        ):
            async with client:
                r = await client.post(
                    "/query",
                    json={
                        "query": "How do I restart Oracle?",
                        "actor": "claire.benoit",
                        "top_k": 1,
                    },
                )

        assert r.status_code == 200
        store.record_activity.assert_awaited_once()
        event = store.record_activity.await_args.args[0]
        assert event["kind"] == "retrieval"
        assert event["actor"]["user"] == "claire.benoit"
        assert event["target"]["type"] == "query"
        assert event["meta"]["sources_count"] == 1
        assert event["meta"]["query"] == "How do I restart Oracle?"

    async def test_preserves_lightrag_answer_text_with_references_block(
        self, make_client
    ):
        answer = (
            "Restart Oracle by stopping listeners [1] then `shutdown immediate` [2].\n\n"
            "### References - [1] runbook.pdf [2] rhel.pdf"
        )
        rag = FakeRag(
            answer=answer,
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
        assert body["response"] == answer
        assert "### References" in body["response"]
        assert len(body["sources"]) == 2

    async def test_default_top_k_is_20(self, make_client):
        rag = FakeRag(answer="defaults", chunks=[])
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "default retrieval"})

        assert r.status_code == 200
        # aquery_llm is the API the nominal path calls now — the
        # legacy ``aquery()`` is only hit in only_need_context /
        # only_need_prompt mode.
        _query, param = rag.llm_calls[0]
        assert param.top_k == 20
        # chunks_vdb is never touched — sources come from aquery_llm.
        assert rag.chunks_vdb.last_query is None

    async def test_only_need_context_skips_source_enrichment(self, make_client):
        rag = FakeRag(answer="raw context blob")
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "anything", "only_need_context": True},
            )

        assert r.status_code == 200
        assert r.json() == {
            "response": "raw context blob",
            "sources": [],
            "answer_status": "grounded",
        }
        # chunks_vdb should not have been touched in context-only mode
        assert rag.chunks_vdb.last_query is None

    async def test_advanced_query_params_forward_to_aquery_llm(self, make_client):
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
        assert len(rag.llm_calls) == 1
        query, param = rag.llm_calls[0]
        assert query == "advanced retrieval"
        assert param.mode == "hybrid"
        assert param.top_k == 11
        assert param.chunk_top_k == 7
        assert param.max_total_tokens == 1234
        assert param.history_turns == 2
        assert param.user_prompt == "prefer runbook citations"
        assert param.enable_rerank is False
        assert param.stream is False

    async def test_tag_filter_forwards_to_aquery_llm(self, make_client):
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
        assert len(rag.llm_calls) == 1
        _query, param = rag.llm_calls[0]
        assert param.tag_filter == {"all": ["oracle", "rman"], "any": []}

    async def test_tag_filter_is_absent_when_omitted(self, make_client):
        rag = FakeRag(answer="untagged")
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "untagged retrieval"})

        assert r.status_code == 200
        assert len(rag.llm_calls) == 1
        _query, param = rag.llm_calls[0]
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
            "answer_status": "grounded",
        }

    async def test_chunks_vdb_is_never_called_even_when_broken(
        self, make_client
    ):
        """Audit C3 guard: the nominal /query path must never touch
        chunks_vdb, even as a defensive fallback. The previous behaviour
        called chunks_vdb.query and caught the failure to return empty
        sources; the new contract is "sources only ever come from
        aquery_llm references", so chunks_vdb is irrelevant.
        """
        rag = FakeRag(answer="ok", chunks=[])

        async def boom(*_a, **_kw):
            raise RuntimeError("memgraph down")

        rag.chunks_vdb.query = boom  # type: ignore[assignment]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "anything"})

        assert r.status_code == 200
        body = r.json()
        assert body["response"] == "ok"
        assert body["sources"] == []
        assert body["answer_status"] == "grounded"
        # chunks_vdb.query is wired to raise — if the route called
        # it, the test would surface a 500. The endpoint returning
        # 200 proves chunks_vdb is unreachable from /query.
        assert rag.chunks_vdb.last_query is None

    async def test_aquery_llm_failure_returns_500(self, make_client):
        rag = FakeRag(answer="never returned")

        async def boom(*_a, **_kw):
            raise RuntimeError("LLM down")

        rag.aquery_llm = boom  # type: ignore[assignment]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "anything"})

        assert r.status_code == 500
        assert "LLM down" in r.json()["detail"]

    async def test_none_answer_is_empty_string_not_literal_none(self, make_client):
        """LightRAG returns None on silent LLM failure — the WebUI must not
        receive a literal "None" bubble (prod incident 2026-06-11). The
        aquery_llm envelope can carry ``content: None`` for the same
        underlying reason; classify_aquery_llm_result coerces it to
        ``""`` and the route surfaces an empty response, not "None".
        """
        rag = FakeRag(answer=None)  # type: ignore[arg-type]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "quel est le rôle de LIP6 ?"})

        assert r.status_code == 200
        assert r.json()["response"] == ""

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

        # Stream plumbing: param.stream=True is forwarded to aquery_llm,
        # not the legacy aquery.
        query, param = rag.llm_calls[0]
        assert query == "How do I restart Oracle?"
        assert param.stream is True
        assert param.chunk_top_k == 4
        assert param.enable_rerank is True
        assert param.user_prompt == "short answer"
        # Audit C3 guard on the stream path: chunks_vdb stays cold.
        assert rag.chunks_vdb.last_query is None

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

    async def test_score_is_zero_baseline_post_aquery_llm(self, make_client):
        """The legacy ``_build_sources`` fabricated rank-based scores
        when chunks_vdb didn't expose any. The aquery_llm migration
        deletes that path — references don't carry a score, so the
        contract baseline is ``0.0``. This test pins that baseline so
        a future change wiring real scores from chunk metrics is
        visible at review.
        """
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
        assert scores == [0.0, 0.0, 0.0]

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


class TestAnswerStatusContract:
    """TR-RET-02: ``/twin/api/query`` must mark insufficient answers
    honestly and suppress the retrieval round-trip so the React port
    can hide the Sources panel without phrase-parsing the LLM prose.

    Two signals classify the answer (see ``_lightrag_compat``):
    - ``metadata.failure_reason == "no_results"`` — structured.
    - ``[no-context]`` marker in the LLM content — defense in depth.
    """

    LIGHTRAG_FAIL = (
        "Sorry, I'm not able to provide an answer to that question."
        "[no-context]"
    )

    async def test_non_stream_insufficient_via_failure_reason(
        self, make_client
    ):
        # Structured signal: aquery_llm envelope reports
        # ``status=failure`` with ``failure_reason=no_results``. The
        # route maps this to ``answer_status=insufficient_information``
        # without falling back to a second vector pass.
        rag = FakeRag(
            answer=self.LIGHTRAG_FAIL,
            envelope_status="failure",
            envelope_failure_reason="no_results",
            # Sources fixtures intentionally populated to verify they
            # are NOT returned even though they sit in the envelope.
            chunks=[
                {
                    "id": "chunk-aa",
                    "file_path": "/x/should-not-appear.pdf",
                },
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query", json={"query": "completely off topic"}
            )

        assert r.status_code == 200
        body = r.json()
        assert body["answer_status"] == "insufficient_information"
        # Marker stripped before reaching the operator.
        assert "[no-context]" not in body["response"]
        assert (
            body["response"]
            == "Sorry, I'm not able to provide an answer to that question."
        )
        assert body["sources"] == []
        # Audit C3 guard: never a second vector pass.
        assert rag.chunks_vdb.last_query is None

    async def test_non_stream_insufficient_via_marker_defense_in_depth(
        self, make_client
    ):
        # Defense in depth: envelope says ``status=success`` (older
        # LightRAG paths that don't set the structured failure_reason)
        # but the marker is in the content. We still detect.
        rag = FakeRag(
            answer=self.LIGHTRAG_FAIL,
            envelope_status="success",
            chunks=[
                {"id": "chunk-aa", "file_path": "/x/should-not-appear.pdf"},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query", json={"query": "off topic"}
            )

        assert r.status_code == 200
        body = r.json()
        assert body["answer_status"] == "insufficient_information"
        assert "[no-context]" not in body["response"]
        assert body["sources"] == []
        assert rag.chunks_vdb.last_query is None

    async def test_non_stream_generic_backend_failure_surfaces_as_500(
        self, make_client
    ):
        # If the envelope reports failure for a reason OTHER than
        # no_results, the route MUST NOT mask it as
        # insufficient_information — that would hide the backend
        # problem behind the React port's "no sources" copy.
        rag = FakeRag(
            answer=None,
            envelope_status="failure",
            envelope_failure_reason="query_failed",
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "x"})
        assert r.status_code == 500

    async def test_non_stream_grounded_sets_status_keeps_sources_no_chunks_vdb(
        self, make_client
    ):
        rag = FakeRag(
            answer="A real answer.",
            chunks=[
                {"id": "chunk-aa", "file_path": "/a/runbook.pdf"},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query", json={"query": "real question"}
            )

        assert r.status_code == 200
        body = r.json()
        assert body["answer_status"] == "grounded"
        assert body["response"] == "A real answer."
        assert len(body["sources"]) == 1
        # Audit C3 guard: even on the grounded path, sources come from
        # aquery_llm's references — never from a second chunks_vdb pass.
        assert rag.chunks_vdb.last_query is None

    async def test_stream_insufficient_via_marker_split_across_chunks(
        self, make_client
    ):
        import json as _json

        # The chunk-boundary case Codex flagged: the marker straddles
        # the stream chunks. The rolling buffer in AnswerMarkerStripper
        # must catch it and the route must emit
        # ``status=insufficient_information``.
        head = "Sorry, I'm not able to provide an answer to that question.[no-co"
        tail = "ntext]"
        rag = FakeRag(
            stream_chunks=[head, tail],
            chunks=[
                {"id": "chunk-aa", "file_path": "/x/should-not-appear.pdf"},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/stream",
                json={"query": "off topic", "mode": "mix"},
            )

        assert r.status_code == 200
        events = [
            _json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        token_events = [e for e in events if e["type"] == "token"]
        status_events = [e for e in events if e["type"] == "status"]
        source_events = [e for e in events if e["type"] == "sources"]

        joined = "".join(e["value"] for e in token_events)
        assert "[no-context]" not in joined
        assert (
            joined
            == "Sorry, I'm not able to provide an answer to that question."
        )
        assert len(status_events) == 1
        assert status_events[0]["value"] == "insufficient_information"
        assert len(source_events) == 1
        assert source_events[0]["value"] == []
        # Order is part of the wire contract: status before sources.
        status_pos = events.index(status_events[0])
        sources_pos = events.index(source_events[0])
        assert status_pos < sources_pos
        assert rag.chunks_vdb.last_query is None

    async def test_stream_generic_failure_surfaces_in_stream_error_token(
        self, make_client
    ):
        """Stream path symmetry of test_non_stream_generic_backend_failure
        _surfaces_as_500: once the HTTP 200 is sent we can't flip the
        status, but we MUST NOT silently emit a grounded/empty response
        for a generic backend failure. Codex review: that would mirror
        exactly the structural lie this PR is closing.
        """
        import json as _json

        rag = FakeRag(
            stream_chunks=["partial "],
            envelope_status="failure",
            envelope_failure_reason="query_failed",
            chunks=[],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/stream", json={"query": "x"}
            )

        assert r.status_code == 200
        events = [
            _json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        token_events = [e for e in events if e["type"] == "token"]
        status_events = [e for e in events if e["type"] == "status"]
        source_events = [e for e in events if e["type"] == "sources"]

        # The operator MUST see an explicit failure token, not a
        # silent grounded response with no body.
        joined = "".join(e["value"] for e in token_events)
        assert "[query failed:" in joined
        assert "query_failed" in joined
        # We emit grounded status (not insufficient_information) to
        # avoid pretending an empty sources list is the canonical
        # "no usable context" response.
        assert len(status_events) == 1
        assert status_events[0]["value"] == "grounded"
        # No fabricated sources behind a failure.
        assert source_events[0]["value"] == []
        # Audit C3 guard still holds: no second vector pass even on
        # the failure path.
        assert rag.chunks_vdb.last_query is None

    async def test_stream_grounded_emits_status_grounded_then_real_sources(
        self, make_client
    ):
        import json as _json

        rag = FakeRag(
            stream_chunks=["A ", "real ", "answer."],
            chunks=[
                {"id": "chunk-aa", "file_path": "/a/runbook.pdf"},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/stream",
                json={"query": "real question"},
            )

        assert r.status_code == 200
        events = [
            _json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        status_events = [e for e in events if e["type"] == "status"]
        source_events = [e for e in events if e["type"] == "sources"]
        token_events = [e for e in events if e["type"] == "token"]

        assert "".join(e["value"] for e in token_events) == "A real answer."
        assert len(status_events) == 1
        assert status_events[0]["value"] == "grounded"
        assert len(source_events) == 1
        assert len(source_events[0]["value"]) == 1
        assert source_events[0]["value"][0]["name"] == "/a/runbook.pdf"
        # Audit C3 guard on the stream path too.
        assert rag.chunks_vdb.last_query is None
