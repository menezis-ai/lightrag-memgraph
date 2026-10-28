"""Real upstream merges must publish graph updates before releasing keyed locks."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from lightrag import operate
from lightrag.kg.memgraph_impl import MemgraphStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

from lightrag_memgraph import register


class _WordTokenizer:
    def encode(self, text):
        return text.split()


@pytest.fixture
async def graph(monkeypatch):
    finalize_share_data()
    initialize_share_data()
    register()
    workspace = f"merge_concurrency_{uuid4().hex}"
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", workspace)
    config = {
        "workspace": workspace,
        "tokenizer": _WordTokenizer(),
        "summary_context_size": 8192,
        "summary_max_tokens": 8192,
        "force_llm_summary_on_merge": 100,
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 100,
        "max_source_ids_per_relation": 100,
    }
    storage = MemgraphStorage("graph", config, embedding_func=None, workspace=workspace)
    await storage.initialize()
    try:
        yield storage
    finally:
        try:
            async with storage._driver.session() as session:
                result = await session.run(f"MATCH (n:`{workspace}`) DETACH DELETE n")
                await result.consume()
                result = await session.run(f"DROP INDEX ON :{workspace}(entity_id)")
                await result.consume()
        finally:
            await storage.finalize()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["entity", "relationship"])
async def test_overlapping_documents_preserve_shared_graph_state(graph, kind):
    """Pause A after its keyed locks are released, let B finish, then resume A.

    This invokes the actual installed LightRAG merge and real graph writes.
    Only document bookkeeping and the vector stores are replaced with test
    doubles, so the graph's read/merge/write and lock behavior stay intact.
    """
    if kind == "relationship":
        for name in ("SHARED", "TARGET"):
            await graph.upsert_node(
                name,
                {
                    "entity_id": name,
                    "entity_type": "TEST",
                    "description": name,
                    "source_id": "seed",
                    "file_path": "seed.txt",
                },
            )

    paused = asyncio.Event()
    resume = asyncio.Event()
    full_entities = AsyncMock()
    full_relations = AsyncMock()

    async def pause_document_a(data):
        if "doc_a" in data:
            paused.set()
            await resume.wait()

    bookkeeping = full_entities if kind == "entity" else full_relations
    bookkeeping.upsert.side_effect = pause_document_a

    async def merge(doc_id):
        data = {
            "description": f"Description from {doc_id}",
            "source_id": f"chunk_{doc_id}",
            "file_path": f"{doc_id}.txt",
        }
        if kind == "entity":
            chunks = [({"SHARED": [{**data, "entity_type": "TEST"}]}, {})]
        else:
            chunks = [
                ({}, {("SHARED", "TARGET"): [{**data, "weight": 1.0, "keywords": doc_id}]})
            ]
        await operate.merge_nodes_and_edges(
            chunk_results=chunks,
            knowledge_graph_inst=graph,
            entity_vdb=AsyncMock(),
            relationships_vdb=AsyncMock(),
            global_config=graph.global_config,
            full_entities_storage=full_entities,
            full_relations_storage=full_relations,
            doc_id=doc_id,
            pipeline_status={"history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
        )

    task_a = asyncio.create_task(merge("doc_a"))
    try:
        await asyncio.wait_for(paused.wait(), timeout=10)
        await asyncio.wait_for(merge("doc_b"), timeout=10)
        resume.set()
        await asyncio.wait_for(task_a, timeout=10)
    finally:
        resume.set()
        if not task_a.done():
            task_a.cancel()
        with suppress(asyncio.CancelledError):
            await task_a

    if kind == "entity":
        stored = await graph.get_node("SHARED")
    else:
        stored = await graph.get_edge("SHARED", "TARGET")

    assert set(stored["source_id"].split(operate.GRAPH_FIELD_SEP)) == {
        "chunk_doc_a",
        "chunk_doc_b",
    }
    assert set(stored["description"].split(operate.GRAPH_FIELD_SEP)) == {
        "Description from doc_a",
        "Description from doc_b",
    }
    if kind == "relationship":
        assert stored["weight"] == 2.0
