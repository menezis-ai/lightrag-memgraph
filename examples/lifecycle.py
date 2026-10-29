"""Real Memgraph lifecycle; fixture models by default, optional live inference."""

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import tempfile
from uuid import uuid4

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import (
    finalize_share_data,
    initialize_pipeline_status,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

from lightrag_memgraph import _pool, register

DOCUMENT = "Paris is the capital of France. The Eiffel Tower is located in Paris."
STORES = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "doc_status",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
)


class CharacterTokenizer:
    """Stateless, download-free tokenizer for the fixture, not real model limits."""

    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


async def fixture_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    if kwargs.get("keyword_extraction") or "high_level_keywords" in prompt:
        return json.dumps(
            {
                "high_level_keywords": ["geography"],
                "low_level_keywords": ["Paris", "France"],
            }
        )
    if "extract" in prompt.lower() or "entity_types" in prompt.lower():
        return "\n".join(
            [
                "entity<|#|>PARIS<|#|>location<|#|>Paris is the capital of France.",
                "entity<|#|>FRANCE<|#|>location<|#|>France is a country in Europe.",
                "relation<|#|>PARIS<|#|>FRANCE<|#|>geography, capital<|#|>Paris is the capital of France.",
                "<|COMPLETE|>",
            ]
        )
    return "Fixture response: Paris is the capital of France."


async def fixture_embedding(texts):
    vectors = []
    for text in texts:
        values = np.frombuffer(
            hashlib.sha256(text.encode()).digest() * 12, dtype=np.uint8
        ).astype(float)
        vectors.append(values / np.linalg.norm(values))
    return np.asarray(vectors)


def models(provider):
    if provider == "fixture":
        print(
            "MODE: fixture — fixed extraction and hash vectors; no answer-quality claim."
        )
        return (
            fixture_llm,
            EmbeddingFunc(embedding_dim=384, func=fixture_embedding),
            {"tokenizer": Tokenizer("fixture-character", CharacterTokenizer())},
        )

    # Live calls happen only after explicit --provider openai selection.
    import httpx

    required = (
        "OPENAI_API_KEY",
        "DEMO_LLM_MODEL",
        "DEMO_EMBEDDING_MODEL",
        "DEMO_EMBEDDING_DIM",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("Set live-provider variables: " + ", ".join(missing))
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    async def post(path, payload):
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                base + path,
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages += list(history_messages or []) + [{"role": "user", "content": prompt}]
        result = await post(
            "/chat/completions",
            {"model": os.environ["DEMO_LLM_MODEL"], "messages": messages},
        )
        return result["choices"][0]["message"]["content"]

    async def embed(texts):
        result = await post(
            "/embeddings", {"model": os.environ["DEMO_EMBEDDING_MODEL"], "input": texts}
        )
        return np.asarray(
            [
                row["embedding"]
                for row in sorted(result["data"], key=lambda row: row["index"])
            ]
        )

    print("MODE: live — document and prompts are sent to the configured endpoint.")
    return (
        complete,
        EmbeddingFunc(embedding_dim=int(os.environ["DEMO_EMBEDDING_DIM"]), func=embed),
        {},
    )


async def counts(rag):
    labels = {
        store._label() if hasattr(store, "_label") else store.workspace
        for name in STORES
        if (store := getattr(rag, name, None)) is not None
    }
    async with _pool.get_read_session() as session:
        result = await session.run(
            "MATCH (n) UNWIND labels(n) AS label WITH label WHERE label IN $labels "
            "RETURN label, count(*) AS count",
            labels=list(labels),
        )
        values = {row["label"]: row["count"] async for row in result}
        await result.consume()
        return values


async def cleanup(rag):
    labels = set()
    for name in (*STORES, "llm_response_cache"):
        store = getattr(rag, name, None)
        if store is not None:
            labels.add(store._label() if hasattr(store, "_label") else store.workspace)
            await store.drop()
    # Remove only property indexes on the randomly allocated demo labels.
    async with _pool.get_session() as session:
        rows = await (await session.run("SHOW INDEX INFO")).data()
        for row in rows:
            if row.get("label") not in labels:
                continue
            prop = row.get("property")
            prop = prop[0] if isinstance(prop, list) and len(prop) == 1 else prop
            if isinstance(prop, str):
                await (
                    await session.run(f"DROP INDEX ON :`{row['label']}`(`{prop}`)")
                ).consume()


async def run(provider):
    os.environ.setdefault("MEMGRAPH_URI", "bolt://localhost:7687")
    complete, embedding, options = models(provider)
    initialize_share_data()
    register()
    for attempt in range(30):
        try:
            async with _pool.get_read_session() as session:
                await (await session.run("RETURN 1")).consume()
            break
        except Exception:
            if attempt == 29:
                raise
            await asyncio.sleep(1)
    workspace = "demo_" + uuid4().hex
    with tempfile.TemporaryDirectory(prefix="lightrag_memgraph_demo_") as directory:
        rag = LightRAG(
            working_dir=directory,
            workspace=workspace,
            kv_storage="MemgraphKVStorage",
            vector_storage="MemgraphVectorDBStorage",
            doc_status_storage="MemgraphDocStatusStorage",
            graph_storage="MemgraphStorage",
            llm_model_func=complete,
            embedding_func=embedding,
            enable_llm_cache=False,
            chunk_token_size=1024,
            chunk_overlap_token_size=0,
            **options,
        )
        try:
            await rag.initialize_storages()
            kwargs = (
                {"workspace": workspace}
                if "workspace"
                in inspect.signature(initialize_pipeline_status).parameters
                else {}
            )
            await initialize_pipeline_status(**kwargs)
            await rag.ainsert(DOCUMENT, ids=["demo-document"], file_paths=["paris.txt"])
            processed = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            assert "demo-document" in processed, "Document ingestion did not complete"
            before = await counts(rag)
            for prefix in ("KV_", "Vec_", "DocStatus_", workspace):
                assert any(
                    label.startswith(prefix) and count > 0
                    for label, count in before.items()
                ), prefix
            print("INSERT:", json.dumps(before, sort_keys=True))
            context = await rag.aquery(
                "What is Paris?",
                param=QueryParam(mode="local", only_need_context=True, top_k=5),
            )
            assert context and "paris" in str(context).lower(), (
                "No Paris context retrieved"
            )
            print("QUERY: retrieved context containing Paris")
            result = await rag.adelete_by_doc_id("demo-document")
            assert result.status == "success", result.message
            after = await counts(rag)
            assert not after, f"Document data remains: {after}"
            print("DELETE: document, chunks, vectors, status and graph verified empty")
        finally:
            try:
                await cleanup(rag)
            finally:
                await rag.finalize_storages()
                await _pool.close_driver()
                finalize_share_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("fixture", "openai"), default="fixture")
    asyncio.run(run(parser.parse_args().provider))
