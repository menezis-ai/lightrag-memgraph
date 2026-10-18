"""Exercise real public LightRAG lifecycles and native Memgraph persistence."""

import asyncio
from contextlib import asynccontextmanager
import inspect
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from lightrag import LightRAG, QueryParam
from lightrag.kg.memgraph_impl import MemgraphStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

from tests.test_e2e import EMBEDDING_DIM, SAMPLE_DOC, _mock_embedding, _mock_llm
from twindb_lightrag_memgraph import _pool, register
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage


@pytest.mark.parametrize("fallback", [None, "environment_workspace"])
def test_all_backends_honor_the_explicit_workspace(monkeypatch, tmp_path, fallback):
    if fallback is None:
        monkeypatch.delenv("MEMGRAPH_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", fallback)
    monkeypatch.delenv("WORKSPACE", raising=False)
    register()
    # Real LightRAG constructors with a local tokenizer to avoid downloads.
    from tests.test_merge_concurrency import _WordTokenizer

    for workspace in ("alpha", "beta"):
        rag = LightRAG(
            working_dir=str(tmp_path / workspace),
            workspace=workspace,
            kv_storage="MemgraphKVStorage",
            vector_storage="MemgraphVectorDBStorage",
            doc_status_storage="MemgraphDocStatusStorage",
            graph_storage="MemgraphStorage",
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=_mock_embedding
            ),
            llm_model_func=_mock_llm,
            tokenizer=_WordTokenizer(),
        )
        for name in (
            "full_docs",
            "text_chunks",
            "doc_status",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
        ):
            assert getattr(rag, name).workspace == workspace, name
    assert __import__("os").environ.get("MEMGRAPH_WORKSPACE") == fallback


@pytest.mark.parametrize(
    "props",
    [
        {"chunks_list": "not JSON"},
        {"chunks_list": '"not a list"'},
        {"chunks_list": "null"},
        {"metadata": "not JSON"},
    ],
)
async def test_destructive_point_reads_reject_corrupt_json(monkeypatch, props):
    store = MemgraphDocStatusStorage("status", {}, None, workspace="integrity")
    result = AsyncMock()
    result.single.return_value = {"props": props}

    @asynccontextmanager
    async def read():
        yield MagicMock(run=AsyncMock(return_value=result))

    monkeypatch.setattr(_pool, "get_read_session", read)
    with pytest.raises(ValueError):
        await store.get_by_id("document")


async def test_kv_propagates_deferred_write_failure(monkeypatch):
    result = AsyncMock()
    result.consume.side_effect = RuntimeError("deferred commit failure")

    @asynccontextmanager
    async def session():
        yield MagicMock(run=AsyncMock(return_value=result))

    monkeypatch.setattr(_pool, "get_session", session)
    with pytest.raises(RuntimeError, match="deferred commit failure"):
        await MemgraphKVStorage("chunks", {}, None, workspace="integrity").upsert(
            {"id": {"content": "value"}}
        )


@pytest.mark.parametrize("batch", [False, True])
async def test_legacy_graph_writer_rejects_entity_type_delimiters_before_io(batch):
    register()
    store = MemgraphStorage("graph", {}, None, workspace="integrity")
    store._driver = MagicMock()
    node = {"entity_id": "SHARED", "entity_type": "bad` MATCH (n) DETACH DELETE n //"}
    with pytest.raises(ValueError, match="entity type"):
        if batch and hasattr(store, "upsert_nodes_batch"):
            await store.upsert_nodes_batch([("SHARED", node)])
        else:
            await store.upsert_node("SHARED", node)
    store._driver.session.assert_not_called()


@pytest.fixture
async def tenants(monkeypatch, tmp_path):
    finalize_share_data()
    initialize_share_data()
    register()
    # An existing deployment default must not override tenant arguments.
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "environment_fallback")
    workspaces = ["tenant_" + uuid4().hex for _ in range(2)]
    instances = []
    from lightrag.kg.shared_storage import initialize_pipeline_status

    for workspace in workspaces:
        rag = LightRAG(
            working_dir=str(tmp_path / workspace),
            workspace=workspace,
            kv_storage="MemgraphKVStorage",
            vector_storage="MemgraphVectorDBStorage",
            doc_status_storage="MemgraphDocStatusStorage",
            graph_storage="MemgraphStorage",
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=_mock_embedding
            ),
            llm_model_func=_mock_llm,
            enable_llm_cache=False,
            chunk_token_size=200,
            chunk_overlap_token_size=50,
        )
        for storage in (
            rag.text_chunks,
            rag.chunks_vdb,
            rag.doc_status,
            rag.chunk_entity_relation_graph,
        ):
            assert storage.workspace == workspace
        await rag.initialize_storages()
        if "workspace" in inspect.signature(initialize_pipeline_status).parameters:
            await initialize_pipeline_status(workspace=workspace)
        else:
            await initialize_pipeline_status()
        instances.append(rag)
    try:
        yield instances
    finally:
        for rag in instances:
            for name in (
                "llm_response_cache",
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
            ):
                storage = getattr(rag, name, None)
                if storage is not None:
                    await storage.drop()
            async with _pool.get_session() as session:
                rows = await (await session.run("SHOW INDEX INFO")).data()
                for row in rows:
                    label, prop = row.get("label"), row.get("property")
                    if not isinstance(label, str) or not (
                        label == rag.workspace
                        or label.startswith(
                            (
                                f"KV_{rag.workspace}_",
                                f"Vec_{rag.workspace}_",
                                f"DocStatus_{rag.workspace}",
                            )
                        )
                    ):
                        continue
                    prop = (
                        prop[0] if isinstance(prop, list) and len(prop) == 1 else prop
                    )
                    if isinstance(prop, str):
                        await (
                            await session.run(f"DROP INDEX ON :`{label}`(`{prop}`)")
                        ).consume()
            await rag.finalize_storages()
        await _pool.close_driver()


@pytest.mark.integration
async def test_concurrent_tenants_and_complete_document_deletion(tenants):
    alpha, beta = tenants
    await asyncio.gather(
        *(
            rag.ainsert(
                SAMPLE_DOC, ids=["same-document-id"], file_paths=[f"{name}.txt"]
            )
            for rag, name in ((alpha, "alpha"), (beta, "beta"))
        )
    )
    from lightrag.kg.shared_storage import initialize_pipeline_status

    if "workspace" not in inspect.signature(initialize_pipeline_status).parameters:
        # LightRAG 1.4.9 has one process-wide busy flag. Concurrent ainsert can
        # leave another tenant's queue pending; explicitly drain each queue.
        for rag in tenants:
            await rag.apipeline_process_enqueue_documents()
    for rag, name in ((alpha, "alpha"), (beta, "beta")):
        status = await rag.doc_status.get_by_id("same-document-id")
        assert status["status"] == "processed", status
        assert status["file_path"] == f"{name}.txt"
        assert isinstance(status["chunks_list"], list) and status["chunks_list"]
        assert isinstance(status["metadata"], dict)
        assert await rag.chunks_vdb.query(SAMPLE_DOC, top_k=3)
        assert await rag.aquery(
            SAMPLE_DOC, param=QueryParam(mode="naive", only_need_context=True)
        )

    chunk_ids = (await alpha.doc_status.get_by_id("same-document-id"))["chunks_list"]
    result = await alpha.adelete_by_doc_id("same-document-id")
    assert result.status == "success", result
    assert await alpha.doc_status.get_by_id("same-document-id") is None
    assert await alpha.full_docs.get_by_id("same-document-id") is None
    assert not any(await alpha.text_chunks.get_by_ids(chunk_ids))
    assert not any(await alpha.chunks_vdb.get_by_ids(chunk_ids))
    async with _pool.get_read_session() as session:
        for label in (
            alpha.chunks_vdb._label(),
            alpha.entities_vdb._label(),
            alpha.relationships_vdb._label(),
            alpha.workspace,
        ):
            result = await session.run(f"MATCH (n:`{label}`) RETURN count(n) AS count")
            assert (await result.single())["count"] == 0, label
            await result.consume()
    if alpha.chunks_vdb.capabilities.version_tuple >= (3, 4, 0):
        assert await alpha.chunks_vdb.query(SAMPLE_DOC, top_k=3) == []
    # The other tenant uses exactly the same document/chunk/vector identifiers.
    assert (await beta.doc_status.get_by_id("same-document-id"))[
        "status"
    ] == "processed"
    assert all(await beta.text_chunks.get_by_ids(chunk_ids))
    assert await beta.chunks_vdb.query(SAMPLE_DOC, top_k=3)


@pytest.mark.integration
async def test_corrupt_chunk_list_cannot_report_successful_deletion(tenants):
    rag, _ = tenants
    await rag.ainsert(SAMPLE_DOC, ids=["corrupt-status"])
    chunk_ids = (await rag.doc_status.get_by_id("corrupt-status"))["chunks_list"]
    assert chunk_ids
    async with _pool.get_session() as session:
        result = await session.run(
            f"MATCH (n:`{rag.doc_status._label()}` {{id: $id}}) SET n.chunks_list = $bad",
            id="corrupt-status",
            bad="not JSON",
        )
        await result.consume()
    try:
        result = await rag.adelete_by_doc_id("corrupt-status")
    except UnboundLocalError as exc:
        # Early upstream releases reference file_path before assigning it when
        # the status read raises. They fail by exception instead of a result.
        assert "file_path" in str(exc)
    else:
        assert result.status != "success", result
    assert await rag.full_docs.get_by_id("corrupt-status") is not None
    assert all(await rag.text_chunks.get_by_ids(chunk_ids))
    assert await rag.chunks_vdb.get_by_ids(chunk_ids)
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`{rag.doc_status._label()}` {{id: $id}}) RETURN count(n) AS count",
            id="corrupt-status",
        )
        assert (await result.single())["count"] == 1
        await result.consume()
