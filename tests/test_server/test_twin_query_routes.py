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
            envelope_chunk = {
                "reference_id": ref_id,
                "content": chunk.get("content", ""),
                "file_path": file_path,
                "chunk_id": chunk_id,
            }
            for key in ("score", "similarity", "cosine_similarity", "__metrics__"):
                if key in chunk:
                    envelope_chunk[key] = chunk[key]
            envelope_chunks.append(envelope_chunk)
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


class MalformedRefsRag(FakeRag):
    """``aquery_llm`` returns a grounded envelope (success + a real answer) whose
    ``data.references`` cannot be projected into the Twin sources contract — the
    #2 case: source projection fails *after* a usable answer was produced.

    ``data.chunks`` is left empty so the only failure is in the references
    block; the answer itself is valid and must still be returned.
    """

    def __init__(self, *, references: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bad_references = references

    def _build_envelope(self, *, is_streaming: bool) -> dict[str, Any]:
        envelope = super()._build_envelope(is_streaming=is_streaming)
        envelope["data"]["references"] = self._bad_references
        return envelope


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
        assert first["score"] == 0.92
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
        # only_need_context is sourceless by design -> no_retrieval, not the
        # grounded default (which would falsely claim a sourced answer).
        assert r.json() == {
            "response": "raw context blob",
            "sources": [],
            "answer_status": "no_retrieval",
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

    async def test_tag_filter_filters_projected_sources_on_query(self, make_client):
        rag = FakeRag(
            answer="tagged",
            chunks=[
                {"id": "chunk-a", "file_path": "/oracle", "score": 0.9},
                {"id": "chunk-b", "file_path": "/network", "score": 0.8},
            ],
            chunk_to_doc={"chunk-a": "doc-oracle", "chunk-b": "doc-network"},
        )
        client = await make_client(rag)
        async def fake_tags(doc_id: str, folder: str):
            assert folder == "default"
            return {
                "doc-oracle": {"oracle", "rman"},
                "doc-network": {"network"},
            }.get(doc_id, set())

        with patch(
            "twindb_lightrag_memgraph.server.twin_query_routes._fetch_doc_graph_tags",
            AsyncMock(side_effect=fake_tags),
        ):
            async with client:
                r = await client.post(
                    "/query",
                    json={
                        "query": "tagged retrieval",
                        "tag_filter": {"all": ["oracle", "rman"], "any": []},
                    },
                )

        assert r.status_code == 200
        assert [s["doc_id"] for s in r.json()["sources"]] == ["doc-oracle"]
        _query, param = rag.llm_calls[0]
        assert param.tag_filter == {"all": ["oracle", "rman"], "any": []}

    async def test_tag_filter_filters_projected_sources_on_query_stream(
        self, make_client
    ):
        rag = FakeRag(
            stream_chunks=["tagged"],
            chunks=[
                {"id": "chunk-a", "file_path": "/oracle", "score": 0.9},
                {"id": "chunk-b", "file_path": "/network", "score": 0.8},
            ],
            chunk_to_doc={"chunk-a": "doc-oracle", "chunk-b": "doc-network"},
        )
        client = await make_client(rag)
        async def fake_tags(doc_id: str, folder: str):
            return {
                "doc-oracle": {"oracle"},
                "doc-network": {"network"},
            }.get(doc_id, set())

        with patch(
            "twindb_lightrag_memgraph.server.twin_query_routes._fetch_doc_graph_tags",
            AsyncMock(side_effect=fake_tags),
        ):
            async with client:
                r = await client.post(
                    "/query/stream",
                    json={
                        "query": "tagged retrieval",
                        "tag_filter": {"all": ["oracle"]},
                    },
                )

        assert r.status_code == 200
        lines = [line for line in r.text.splitlines() if line.strip()]
        sources_event = next(json for json in lines if '"type": "sources"' in json)
        assert '"doc_id": "doc-oracle"' in sources_event
        assert '"doc_id": "doc-network"' not in sources_event

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

    async def test_doc_filter_filters_projected_sources(self, make_client):
        rag = FakeRag(
            answer="x",
            chunks=[
                {"id": "a", "file_path": "/a", "score": 0.91},
                {"id": "b", "file_path": "/b", "score": 0.82},
            ],
            chunk_to_doc={"a": "doc-a", "b": "doc-b"},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "x", "doc_filter": {"any": ["doc-b"]}},
            )

        assert r.status_code == 200
        assert [s["doc_id"] for s in r.json()["sources"]] == ["doc-b"]
        _query, param = rag.llm_calls[0]
        assert param.doc_filter == {"any": ["doc-b"]}

    async def test_only_need_prompt_skips_source_enrichment(self, make_client):
        rag = FakeRag(answer="prompt that would be sent")
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "anything", "only_need_prompt": True},
            )

        assert r.status_code == 200
        # only_need_prompt is sourceless by design -> no_retrieval.
        assert r.json() == {
            "response": "prompt that would be sent",
            "sources": [],
            "answer_status": "no_retrieval",
        }

    async def test_bypass_mode_reports_no_retrieval(self, make_client):
        # bypass calls the LLM directly with no retrieval. Even though the
        # fake envelope carries chunks, the route must short-circuit to
        # no_retrieval + empty sources -- never the grounded default, which
        # would falsely claim the direct answer is sourced.
        rag = FakeRag(
            answer="direct LLM answer",
            chunks=[{"id": "c1", "file_path": "/a", "score": 0.9}],
            chunk_to_doc={"c1": "doc-a"},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query",
                json={"query": "anything", "mode": "bypass"},
            )

        assert r.status_code == 200
        assert r.json() == {
            "response": "direct LLM answer",
            "sources": [],
            "answer_status": "no_retrieval",
        }

    async def test_bypass_mode_reports_no_retrieval_on_stream(self, make_client):
        rag = FakeRag(
            stream_chunks=["direct ", "answer"],
            chunks=[{"id": "c1", "file_path": "/a", "score": 0.9}],
            chunk_to_doc={"c1": "doc-a"},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/stream",
                json={"query": "anything", "mode": "bypass"},
            )

        assert r.status_code == 200
        lines = [line for line in r.text.splitlines() if line.strip()]
        status_event = next(j for j in lines if '"type": "status"' in j)
        sources_event = next(j for j in lines if '"type": "sources"' in j)
        assert '"value": "no_retrieval"' in status_event
        assert '"value": []' in sources_event

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
        self, make_client, monkeypatch
    ):
        # Audit C2 fix: tags come from the ``TAGGED_WITH`` graph
        # relation, never from ``DocStatus.metadata.tags``. The test
        # fakes the graph lookup so the fixture stays self-contained
        # without a live Memgraph.
        from twindb_lightrag_memgraph.server import twin_query_routes as tqr

        graph_tags = {
            "doc-rman": {"rman", "oracle"},
            "doc-vmware": {"vmware"},
        }

        async def fake_fetch(doc_id, folder):
            return graph_tags.get(doc_id, set())

        monkeypatch.setattr(tqr, "_fetch_doc_graph_tags", fake_fetch)

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

    async def test_query_data_tag_filter_uses_tagged_with_not_metadata_tags(
        self, make_client, monkeypatch
    ):
        """Divergence test (audit C2): a doc whose ``metadata.tags`` is
        empty but whose ``TAGGED_WITH`` edge carries the filter tag
        MUST be kept. The previous implementation rejected it (read
        the empty property), which silently disagreed with the
        retag flow that only writes the edge.
        """
        from twindb_lightrag_memgraph.server import twin_query_routes as tqr

        async def fake_fetch(doc_id, folder):
            return {"rman"} if doc_id == "doc-with-edge" else set()

        monkeypatch.setattr(tqr, "_fetch_doc_graph_tags", fake_fetch)

        rag = FakeRag(
            query_data={
                "status": "success",
                "message": "ok",
                "data": {
                    "chunks": [
                        {
                            "chunk_id": "c-edge",
                            "full_doc_id": "doc-with-edge",
                            "reference_id": "1",
                        },
                    ],
                    "references": [
                        {"reference_id": "1", "file_path": "edge.pdf"},
                    ],
                },
                "metadata": {},
            },
            chunk_to_doc={"c-edge": "doc-with-edge"},
            # Legacy property intentionally empty — the new path
            # ignores it entirely.
            docs={"doc-with-edge": {"metadata": {"tags": []}}},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={
                    "query": "x",
                    "tag_filter": {"all": ["rman"], "any": []},
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert [c["chunk_id"] for c in body["data"]["chunks"]] == ["c-edge"]

    async def test_query_data_tag_filter_rejects_doc_without_tagged_with_edge(
        self, make_client, monkeypatch
    ):
        """Divergence test (audit C2): a doc whose ``metadata.tags``
        historically carries the filter tag but which has no
        ``TAGGED_WITH`` edge MUST be rejected. The previous
        implementation kept it (read the legacy property), letting
        stale metadata leak through the filter.
        """
        from twindb_lightrag_memgraph.server import twin_query_routes as tqr

        async def fake_fetch(doc_id, folder):
            return set()  # no edge anywhere

        monkeypatch.setattr(tqr, "_fetch_doc_graph_tags", fake_fetch)

        rag = FakeRag(
            query_data={
                "status": "success",
                "message": "ok",
                "data": {
                    "chunks": [
                        {
                            "chunk_id": "c-stale",
                            "full_doc_id": "doc-stale",
                            "reference_id": "1",
                        },
                    ],
                    "references": [
                        {"reference_id": "1", "file_path": "stale.pdf"},
                    ],
                },
                "metadata": {},
            },
            chunk_to_doc={"c-stale": "doc-stale"},
            # The legacy property says "rman" but the graph disagrees.
            docs={"doc-stale": {"metadata": {"tags": ["rman", "oracle"]}}},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={
                    "query": "x",
                    "tag_filter": {"all": ["rman"], "any": []},
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert body["data"]["chunks"] == []
        assert body["data"]["references"] == []

    async def test_query_data_tag_filter_caches_per_request(
        self, make_client, monkeypatch
    ):
        """Audit C2 amendment: chunks / references / entities rows
        often reference the same doc_id. The per-request cache must
        coalesce them so ``_fetch_doc_graph_tags`` is called at most
        once per unique doc — bounded round-trips even for large
        result sets.
        """
        from twindb_lightrag_memgraph.server import twin_query_routes as tqr

        calls: list[tuple[str, str]] = []

        async def fake_fetch(doc_id, folder):
            calls.append((doc_id, folder))
            return {"rman"} if doc_id == "doc-A" else set()

        monkeypatch.setattr(tqr, "_fetch_doc_graph_tags", fake_fetch)

        rag = FakeRag(
            query_data={
                "status": "success",
                "message": "ok",
                "data": {
                    # Four rows pointing at the same doc — must
                    # trigger ONE Cypher call, not four.
                    "chunks": [
                        {"chunk_id": f"c{i}", "full_doc_id": "doc-A", "reference_id": "1"}
                        for i in range(4)
                    ],
                    "entities": [
                        {"entity_name": "X", "source_id": "c0", "reference_id": "1"},
                    ],
                    "references": [
                        {"reference_id": "1", "file_path": "a.pdf"},
                    ],
                },
                "metadata": {},
            },
            chunk_to_doc={f"c{i}": "doc-A" for i in range(4)},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={"query": "x", "tag_filter": {"all": ["rman"], "any": []}},
            )
        assert r.status_code == 200
        unique_doc_calls = {doc_id for doc_id, _folder in calls}
        # Even with 5 rows referencing doc-A, the cache collapses to one fetch.
        assert unique_doc_calls == {"doc-A"}
        assert len(calls) == 1

    async def test_query_data_tag_filter_passes_folder_from_request(
        self, make_client, monkeypatch
    ):
        """Codex review amendment: the folder MUST be passed
        explicitly from the route handler through the helper chain
        rather than recovered from ``current_folder_id()`` inside the
        helper. This test confirms the resolved folder (the catalog
        default in this test env) reaches ``_fetch_doc_graph_tags``
        verbatim, instead of e.g. a hard-coded fallback.
        """
        from twindb_lightrag_memgraph.server import twin_query_routes as tqr
        from twindb_lightrag_memgraph.server.folder import (
            load_folder_catalog,
        )

        seen_folders: list[str] = []

        async def fake_fetch(doc_id, folder):
            seen_folders.append(folder)
            return {"rman"}

        monkeypatch.setattr(tqr, "_fetch_doc_graph_tags", fake_fetch)

        rag = FakeRag(
            query_data={
                "status": "success",
                "message": "ok",
                "data": {
                    "chunks": [
                        {"chunk_id": "c1", "full_doc_id": "doc-A", "reference_id": "1"},
                    ],
                    "references": [{"reference_id": "1", "file_path": "a.pdf"}],
                },
                "metadata": {},
            },
            chunk_to_doc={"c1": "doc-A"},
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query/data",
                json={"query": "x", "tag_filter": {"all": ["rman"], "any": []}},
            )
        assert r.status_code == 200
        expected_folder = load_folder_catalog().default_folder_id
        # The folder resolved at the route boundary surfaces all the
        # way down to the helper — never the empty string, never a
        # ``current_folder_id()`` fallback inside _fetch_doc_graph_tags.
        assert set(seen_folders) == {expected_folder}

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

    async def test_scores_come_from_aquery_llm_chunk_metrics(self, make_client):
        rag = FakeRag(
            answer="x",
            chunks=[
                {"id": "a", "file_path": "/a", "similarity": 0.82},
                {"id": "b", "file_path": "/b", "__metrics__": {"score": 0.74}},
                {"id": "c", "file_path": "/c"},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query", json={"query": "x", "top_k": 3}
            )
        scores = [s["score"] for s in r.json()["sources"]]
        assert scores == [0.82, 0.74, 0.5]

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

    async def test_min_score_filters_sources_after_projection(self, make_client):
        rag = FakeRag(
            answer="x",
            chunks=[
                {"id": "a", "file_path": "/a", "score": 0.91},
                {"id": "b", "file_path": "/b", "score": 0.42},
                {"id": "c", "file_path": "/c", "score": 0.72},
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post(
                "/query", json={"query": "x", "top_k": 3, "min_score": 0.7}
            )

        assert r.status_code == 200
        assert [s["name"] for s in r.json()["sources"]] == ["/a", "/c"]

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
        # The failure_reason ("query_failed" here) is echoed in the token.
        assert "[query failed: query_failed]" in joined
        # The status carries the failure: query_failed (NOT grounded, which
        # would pretend the error notice is a sourced answer; NOT
        # insufficient_information, which would pretend retrieval ran and
        # found nothing usable).
        assert len(status_events) == 1
        assert status_events[0]["value"] == "query_failed"
        # No fabricated sources behind a failure.
        assert source_events[0]["value"] == []
        # Audit C3 guard still holds: no second vector pass even on
        # the failure path.
        assert rag.chunks_vdb.last_query is None

    async def test_stream_aquery_llm_exception_emits_query_failed(
        self, make_client
    ):
        """When aquery_llm raises mid-stream the HTTP 200 is already
        committed, so the failure is reported via a [query failed: …]
        token + a query_failed status, never a grounded lie.
        """
        import json as _json

        rag = FakeRag(answer="never returned")

        async def boom(*_a, **_kw):
            raise RuntimeError("LLM down")

        rag.aquery_llm = boom  # type: ignore[assignment]
        client = await make_client(rag)
        async with client:
            r = await client.post("/query/stream", json={"query": "x"})

        assert r.status_code == 200
        events = [
            _json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        token_events = [e for e in events if e["type"] == "token"]
        status_events = [e for e in events if e["type"] == "status"]
        source_events = [e for e in events if e["type"] == "sources"]

        joined = "".join(e["value"] for e in token_events)
        assert "[query failed: LLM down]" in joined
        assert len(status_events) == 1
        assert status_events[0]["value"] == "query_failed"
        assert source_events[0]["value"] == []

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

    # ── #2: grounded answer, source projection fails ──────────────────────
    # A successful aquery_llm envelope whose data.references can't be projected
    # must NOT surface as silent grounded+[] (reads as "no sources") nor as a
    # 500 (hides a usable answer). It surfaces the answer + sources=[] +
    # answer_status=source_projection_failed.

    BAD_REFS_CASES = [
        ("missing_reference_id", [{"file_path": "/x/a.pdf"}]),
        (
            "non_int_reference_id",
            [{"reference_id": "not-an-int", "file_path": "/x/a.pdf"}],
        ),
        ("reference_not_a_dict", ["i am not a dict"]),
    ]

    @pytest.mark.parametrize(
        "label,bad_refs",
        BAD_REFS_CASES,
        ids=[c[0] for c in BAD_REFS_CASES],
    )
    async def test_non_stream_source_projection_failed(
        self, make_client, label, bad_refs
    ):
        rag = MalformedRefsRag(answer="A grounded answer.", references=bad_refs)
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "real question"})

        assert r.status_code == 200, label
        body = r.json()
        # The grounded answer is preserved …
        assert body["response"] == "A grounded answer."
        # … but sources are empty and the status is explicit (not grounded,
        # not insufficient_information).
        assert body["sources"] == []
        assert body["answer_status"] == "source_projection_failed"
        # Audit C3 guard: never a second vector pass to "recover" sources.
        assert rag.chunks_vdb.last_query is None

    @pytest.mark.parametrize(
        "label,bad_refs",
        BAD_REFS_CASES,
        ids=[c[0] for c in BAD_REFS_CASES],
    )
    async def test_stream_source_projection_failed(
        self, make_client, label, bad_refs
    ):
        import json as _json

        rag = MalformedRefsRag(
            references=bad_refs, stream_chunks=["A ", "grounded ", "answer."]
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query/stream", json={"query": "real question"})

        assert r.status_code == 200, label
        events = [
            _json.loads(line) for line in r.text.splitlines() if line.strip()
        ]
        token_events = [e for e in events if e["type"] == "token"]
        status_events = [e for e in events if e["type"] == "status"]
        source_events = [e for e in events if e["type"] == "sources"]

        # Answer tokens still stream …
        assert "".join(e["value"] for e in token_events) == "A grounded answer."
        # … then an explicit status, then an empty sources event.
        assert len(status_events) == 1
        assert status_events[0]["value"] == "source_projection_failed"
        assert len(source_events) == 1
        assert source_events[0]["value"] == []
        assert rag.chunks_vdb.last_query is None


class TestSourceMatchesDocFilter:
    """Finding #3: the source post-filter must mirror the storage-layer
    ``doc_all`` / ``doc_any`` semantics (vector_impl._doc_conditions_set),
    not the legacy union-as-``any`` it used to conflate. It is the
    last-line guard if the aquery_llm envelope shape shifts under a
    LightRAG bump and the Cypher exclusion stops being the only gate.
    """

    @staticmethod
    def _src(doc_id: str, name: str = "") -> dict[str, Any]:
        return {"doc_id": doc_id, "name": name}

    def test_no_filter_passes(self):
        from twindb_lightrag_memgraph.server.query.router import (
            _source_matches_doc_filter,
        )

        assert _source_matches_doc_filter(self._src("docA"), None) is True
        assert _source_matches_doc_filter(self._src("docA"), {}) is True

    def test_doc_any_is_intersection(self):
        from twindb_lightrag_memgraph.server.query.router import (
            _source_matches_doc_filter,
        )

        flt = {"any": ["docA", "docB"]}
        assert _source_matches_doc_filter(self._src("docA"), flt) is True
        assert _source_matches_doc_filter(self._src("docC"), flt) is False

    def test_doc_all_is_strict_subset_not_union(self):
        from twindb_lightrag_memgraph.server.query.router import (
            _source_matches_doc_filter,
        )

        # A single-doc source can never satisfy all of TWO distinct docs --
        # the legacy union-as-``any`` wrongly passed it on "docA in {A,B}".
        two = {"all": ["docA", "docB"]}
        assert _source_matches_doc_filter(self._src("docA"), two) is False
        assert _source_matches_doc_filter(self._src("docB"), two) is False

        # all of a single requested doc == that doc is present.
        one = {"all": ["docA"]}
        assert _source_matches_doc_filter(self._src("docA"), one) is True
        assert _source_matches_doc_filter(self._src("docC"), one) is False

    def test_doc_all_matches_against_name_candidate(self):
        from twindb_lightrag_memgraph.server.query.router import (
            _source_matches_doc_filter,
        )

        # candidates = {doc_id, name}; ``all`` of two docs is satisfiable only
        # when both requested values are among the source's own identifiers.
        flt = {"all": ["docA", "/path/a.pdf"]}
        src = self._src("docA", "/path/a.pdf")
        assert _source_matches_doc_filter(src, flt) is True

    def test_doc_all_and_any_are_anded(self):
        from twindb_lightrag_memgraph.server.query.router import (
            _source_matches_doc_filter,
        )

        flt = {"all": ["docA"], "any": ["docB", "docA"]}
        # required docA present AND (docB|docA) intersect -> pass.
        assert _source_matches_doc_filter(self._src("docA"), flt) is True
        # required docA absent -> fail even though ``any`` would match.
        assert _source_matches_doc_filter(self._src("docB"), flt) is False
