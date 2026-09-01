"""Real overlay/L3/Memgraph acceptance gate for Forgejo #117.

The LLM-facing phases are deterministic fakes.  The host ``LightRAG`` 1.5.6,
its ``aquery_data`` retrieval, Memgraph vector index, ``DocStatus`` nodes and
``MEMBER_OF`` traversal are real.  This is deliberately separate from the
fast unit contract in ``test_l3_query_runtime.py``.
"""

from __future__ import annotations

from importlib.metadata import version
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

from twindb_lightrag_memgraph import _pool, register
from twindb_lightrag_memgraph._constants import (
    get_active_retrieval_filters,
    get_active_storage_folder,
    storage_folder_context,
)
from twindb_lightrag_memgraph.intelligence.models.schemas import (
    AnswerStatus,
    Citation,
    IntentResult,
    IntentType,
)
from twindb_lightrag_memgraph.intelligence.react.observe import SynthesisResult
from twindb_lightrag_memgraph.intelligence.react.reason import ReasoningResult
from twindb_lightrag_memgraph.server import idp_jwt
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp

pytestmark = pytest.mark.integration

_DIM = 8


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _embedding(texts: list[str]) -> np.ndarray:
    """Fixed unit vectors make the database result deterministic."""
    vector = [1.0] + [0.0] * (_DIM - 1)
    return np.asarray([vector for _text in texts], dtype=np.float32)


async def _host_llm(*_args, **_kwargs) -> str:
    raise AssertionError("host LightRAG answer generation is outside this gate")


def _doc(doc_id: str, path: str) -> dict:
    return {
        "id": doc_id,
        "status": "processed",
        "content_summary": "integration fixture",
        "content_length": 32,
        "file_path": path,
        "created_at": "2026-09-01T00:00:00",
        "updated_at": "2026-09-01T00:00:00",
        "content_hash": doc_id,
    }


async def _cleanup_workspace(workspace: str) -> None:
    labels = [
        *(
            f"KV_{workspace}_{namespace}"
            for namespace in (
                "full_docs",
                "text_chunks",
                "full_entities",
                "full_relations",
                "entity_chunks",
                "relation_chunks",
                "llm_response_cache",
            )
        ),
        f"DocStatus_{workspace}",
        f"Folder_{workspace}",
        workspace,
    ]
    async with _pool.get_session() as session:
        for namespace in ("chunks", "entities", "relationships"):
            label = f"Vec_{workspace}_{namespace}"
            result = await session.run(
                f"MATCH (n:`{label}`) REMOVE n:`{label}` " "WITH n DETACH DELETE n"
            )
            await result.consume()
            try:
                result = await session.run(
                    f"DROP VECTOR INDEX `vec_{workspace}_{namespace}`"
                )
                await result.consume()
            except Exception as exc:
                message = str(exc).lower()
                if "does not exist" not in message and "doesn't exist" not in message:
                    raise
        for label in labels:
            result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
            await result.consume()


@pytest.fixture
async def real_host_rag(monkeypatch, tmp_path):
    assert version("lightrag-hku") == "1.5.6"
    register()
    workspace = f"l3gate_{uuid4().hex[:10]}"
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", workspace)

    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    finalize_share_data()
    initialize_share_data()

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=8192,
            func=_embedding,
        ),
        llm_model_func=_host_llm,
        tokenizer=Tokenizer("l3-gate-char", _CharTokenizer()),
        enable_llm_cache=False,
    )
    await rag.initialize_storages()
    try:
        with storage_folder_context("alpha"):
            await rag.doc_status.upsert(
                {"doc-alpha": _doc("doc-alpha", "alpha-policy.pdf")}
            )
        with storage_folder_context("beta"):
            await rag.doc_status.upsert(
                {"doc-beta": _doc("doc-beta", "beta-policy.pdf")}
            )
        vector = [1.0] + [0.0] * (_DIM - 1)
        await rag.chunks_vdb.upsert(
            {
                "chunk-alpha": {
                    "full_doc_id": "doc-alpha",
                    "content": "Evidence visible only from alpha.",
                    "file_path": "alpha-policy.pdf",
                    "embedding": vector,
                },
                "chunk-beta": {
                    "full_doc_id": "doc-beta",
                    "content": "Evidence visible only from beta.",
                    "file_path": "beta-policy.pdf",
                    "embedding": vector,
                },
            }
        )
        yield rag
    finally:
        await rag.finalize_storages()
        await _cleanup_workspace(workspace)
        finalize_share_data()


async def test_overlay_l3_reuses_real_host_rag_and_enforces_member_of(
    real_host_rag, monkeypatch
):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "alpha")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        "["
        '{"id":"alpha","label":"Alpha","kind":"kb"},'
        '{"id":"beta","label":"Beta","kind":"kb"},'
        '{"id":"gamma","label":"Gamma","kind":"kb"}'
        "]",
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "5")
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    monkeypatch.setenv("TWIN_RAG_ENABLE_QUERY_EXPANSION", "false")
    monkeypatch.setenv("TWIN_RAG_ENABLE_COGNITIVE_RERANKING", "false")
    monkeypatch.setenv("TWIN_RAG_QUERY_ENGINE", "l3")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "l3-gate-root")
    monkeypatch.setenv("TWIN_IDP_JWKS_URL", "https://idp.test/.well-known/jwks.json")
    monkeypatch.setenv("TWIN_IDP_ISSUER", "https://idp.test")
    monkeypatch.setenv("TWIN_IDP_AUDIENCE", "twin")
    monkeypatch.setattr(
        idp_jwt,
        "require_idp_user",
        lambda _request: {"folders": ["alpha", "beta"]},
    )

    import twindb_lightrag_memgraph as twin_package
    from twindb_lightrag_memgraph.server.query import l3_runtime as l3_runtime_module

    captured_runtime = {}
    real_runtime_builder = l3_runtime_module.build_l3_query_runtime

    def observed_runtime_builder(get_rag, *, environ=None):
        runtime = real_runtime_builder(get_rag, environ=environ)
        captured_runtime["runtime"] = runtime
        captured_runtime["host_rag"] = get_rag()
        return runtime

    monkeypatch.setattr(
        l3_runtime_module,
        "build_l3_query_runtime",
        observed_runtime_builder,
    )
    monkeypatch.setitem(twin_package._twindb_state, "rag", real_host_rag)

    app = FastAPI()
    twin_package._mount_twin_subapp(
        app,
        "/twin/api",
        webui_stores="memgraph",
    )
    runtime = captured_runtime["runtime"]
    assert runtime is not None
    assert captured_runtime["host_rag"] is real_host_rag
    engine = runtime.engine()
    engine.intent_classifier.classify = AsyncMock(
        return_value=IntentResult(intent=IntentType.IN_SCOPE, confidence=1.0)
    )
    engine.reasoning.analyze = AsyncMock(
        side_effect=lambda question, _history: ReasoningResult(
            thought="deterministic test phase",
            search_query=question,
            original_question=question,
        )
    )

    async def synthesize(*, chunks, on_token=None, **_kwargs):
        assert len(chunks) == 1
        chunk = chunks[0]
        answer = f"Scoped answer from {chunk.source_workspace} [1]"
        if on_token is not None:
            await on_token(answer)
        return SynthesisResult(
            answer=answer,
            citations=[
                Citation(
                    passage_index=0,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    document_path=chunk.document_path,
                    source_workspace=chunk.source_workspace,
                    score=chunk.score,
                    retrieval_score=chunk.metadata["measured_retrieval_score"],
                    chunk_id=chunk.chunk_id,
                )
            ],
            answer_status=AnswerStatus.GROUNDED,
        )

    engine.synthesis.synthesize = AsyncMock(side_effect=synthesize)

    retrieval_calls = []
    original_aquery_data = real_host_rag.aquery_data

    async def observed_aquery_data(query, param):
        retrieval_calls.append(
            (
                get_active_storage_folder(),
                get_active_retrieval_filters(),
                param.mode,
            )
        )
        return await original_aquery_data(query, param)

    monkeypatch.setattr(real_host_rag, "aquery_data", observed_aquery_data)

    try:
        with (
            patch.object(
                engine,
                "_get_rag",
                side_effect=AssertionError("L3 must not create a folder RAG"),
            ) as legacy_get_rag,
            patch(
                "twindb_lightrag_memgraph.intelligence.engine.LightRAG",
                side_effect=AssertionError("a second LightRAG was constructed"),
            ) as constructor,
            patch(
                "twindb_lightrag_memgraph.server.query.router._record_retrieval_activity",
                new=AsyncMock(),
            ),
        ):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={"Authorization": "Bearer l3-gate-root"},
            ) as client:
                responses = {}
                for folder in ("alpha", "beta"):
                    responses[folder] = await client.post(
                        "/twin/api/query",
                        headers={"X-Twin-Folder": folder},
                        json={
                            "query": "shared deterministic query",
                            "mode": "naive",
                            "top_k": 5,
                            "min_score": 0.1,
                            "enable_rerank": False,
                        },
                    )

                calls_before_denial = len(retrieval_calls)
                denied = await client.post(
                    "/twin/api/query",
                    headers={"X-Twin-Folder": "gamma"},
                    json={"query": "must be refused", "mode": "naive"},
                )

            assert responses["alpha"].status_code == 200
            assert responses["beta"].status_code == 200
            for folder, expected_chunk, forbidden_chunk in (
                ("alpha", "chunk-alpha", "chunk-beta"),
                ("beta", "chunk-beta", "chunk-alpha"),
            ):
                payload = responses[folder].json()
                assert payload["trace"]["engine"] == "l3"
                assert [source["chunk_id"] for source in payload["sources"]] == [
                    expected_chunk
                ]
                assert forbidden_chunk not in responses[folder].text

            assert denied.status_code == 403
            assert len(retrieval_calls) == calls_before_denial == 2
            assert [call[0] for call in retrieval_calls] == ["alpha", "beta"]
            assert all(call[1].min_score == 0.1 for call in retrieval_calls)
            assert all(call[2] == "naive" for call in retrieval_calls)
            legacy_get_rag.assert_not_called()
            constructor.assert_not_called()
    finally:
        configure_idp(None)
