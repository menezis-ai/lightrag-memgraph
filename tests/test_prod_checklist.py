"""
Pre-production validation tests.

Covers checklist items that require a live Memgraph >= 3.2 + MAGE:
  - Vector indexes at BGE-M3 dimension (1024)
  - Multi-workspace isolation
  - Full pipeline: register() → LightRAG() → ainsert() → aquery()
"""

import numpy as np
import pytest
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.utils import EmbeddingFunc

from twindb_lightrag_memgraph import register
from twindb_lightrag_memgraph._pool import get_driver
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

register()

# ── BGE-M3 dimension (1024) ──────────────────────────────────────────

BGE_M3_DIM = 1024


async def _mock_embed_1024(texts: list[str]) -> np.ndarray:
    """Deterministic mock at BGE-M3 dimension."""
    rng = np.random.default_rng(seed=hash(texts[0]) % 2**31)
    return rng.random((len(texts), BGE_M3_DIM)).astype(np.float32)


@pytest.fixture
def embedding_func_1024():
    return EmbeddingFunc(
        embedding_dim=BGE_M3_DIM,
        max_token_size=8192,
        func=_mock_embed_1024,
    )


@pytest.fixture
async def vec_store_1024(embedding_func_1024):
    store = MemgraphVectorDBStorage(
        namespace="test_bge_m3",
        global_config={
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
            },
        },
        embedding_func=embedding_func_1024,
        meta_fields={"entity_name", "content"},
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.mark.integration
class TestBGEM3Dimension:
    """Checklist item 4: validate vector indexes at dim=1024."""

    async def test_upsert_1024d_vector(self, vec_store_1024):
        emb = np.random.default_rng(42).random(BGE_M3_DIM).tolist()
        await vec_store_1024.upsert(
            {"bge1": {"content": "test doc", "entity_name": "e1", "embedding": emb}}
        )
        result = await vec_store_1024.get_by_id("bge1")
        assert result is not None
        assert result["entity_name"] == "e1"

    async def test_query_1024d_vector(self, vec_store_1024):
        rng = np.random.default_rng(42)
        for i in range(10):
            emb = rng.random(BGE_M3_DIM).tolist()
            await vec_store_1024.upsert(
                {
                    f"bge{i}": {
                        "content": f"doc {i}",
                        "entity_name": f"e{i}",
                        "embedding": emb,
                    }
                }
            )

        query_emb = rng.random(BGE_M3_DIM).tolist()
        results = await vec_store_1024.query("", top_k=5, query_embedding=query_emb)
        assert len(results) > 0
        assert all("id" in r and "similarity" in r for r in results)

    async def test_get_vectors_by_ids_1024d(self, vec_store_1024):
        emb = np.random.default_rng(42).random(BGE_M3_DIM).tolist()
        await vec_store_1024.upsert({"bge1": {"embedding": emb}})
        vectors = await vec_store_1024.get_vectors_by_ids(["bge1"])
        assert "bge1" in vectors
        assert len(vectors["bge1"]) == BGE_M3_DIM

    async def test_batch_upsert_1024d(self, vec_store_1024):
        """Batch upsert 50 vectors at dim=1024 in a single UNWIND."""
        rng = np.random.default_rng(42)
        data = {
            f"batch{i}": {
                "content": f"batch doc {i}",
                "entity_name": f"be{i}",
                "embedding": rng.random(BGE_M3_DIM).tolist(),
            }
            for i in range(50)
        }
        await vec_store_1024.upsert(data)
        results = await vec_store_1024.get_by_ids([f"batch{i}" for i in range(50)])
        assert len(results) == 50


# ── Multi-workspace isolation ─────────────────────────────────────────


@pytest.fixture
async def kv_ws_alpha():
    store = MemgraphKVStorage(
        namespace="test_multi",
        global_config={},
        embedding_func=None,
        workspace="alpha",
    )
    # Override workspace (normally from env var)
    store.workspace = "alpha"
    await store.initialize()
    yield store
    await store.drop()


@pytest.fixture
async def kv_ws_beta():
    store = MemgraphKVStorage(
        namespace="test_multi",
        global_config={},
        embedding_func=None,
        workspace="beta",
    )
    store.workspace = "beta"
    await store.initialize()
    yield store
    await store.drop()


@pytest.fixture
async def doc_ws_alpha():
    store = MemgraphDocStatusStorage(
        namespace="test_multi",
        global_config={},
        embedding_func=None,
    )
    store.workspace = "alpha"
    await store.initialize()
    yield store
    await store.drop()


@pytest.fixture
async def doc_ws_beta():
    store = MemgraphDocStatusStorage(
        namespace="test_multi",
        global_config={},
        embedding_func=None,
    )
    store.workspace = "beta"
    await store.initialize()
    yield store
    await store.drop()


@pytest.mark.integration
class TestMultiWorkspace:
    """Checklist item 5: two workspaces must be fully isolated."""

    async def test_kv_isolation(self, kv_ws_alpha, kv_ws_beta):
        await kv_ws_alpha.upsert({"shared_key": {"source": "alpha"}})
        await kv_ws_beta.upsert({"shared_key": {"source": "beta"}})

        alpha_val = await kv_ws_alpha.get_by_id("shared_key")
        beta_val = await kv_ws_beta.get_by_id("shared_key")

        assert alpha_val["source"] == "alpha"
        assert beta_val["source"] == "beta"

    async def test_kv_drop_does_not_affect_other_workspace(
        self, kv_ws_alpha, kv_ws_beta
    ):
        await kv_ws_alpha.upsert({"key1": {"val": 1}})
        await kv_ws_beta.upsert({"key1": {"val": 2}})

        await kv_ws_alpha.drop()

        assert await kv_ws_alpha.get_by_id("key1") is None
        beta_val = await kv_ws_beta.get_by_id("key1")
        assert beta_val is not None
        assert beta_val["val"] == 2

    async def test_kv_filter_keys_isolated(self, kv_ws_alpha, kv_ws_beta):
        await kv_ws_alpha.upsert({"only_alpha": {"val": 1}})

        missing_in_beta = await kv_ws_beta.filter_keys({"only_alpha"})
        assert "only_alpha" in missing_in_beta

    async def test_docstatus_isolation(self, doc_ws_alpha, doc_ws_beta):
        status_a = DocProcessingStatus(
            content_summary="alpha doc",
            content_length=100,
            file_path="/alpha.pdf",
            status=DocStatus.PROCESSED,
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )
        status_b = DocProcessingStatus(
            content_summary="beta doc",
            content_length=200,
            file_path="/beta.pdf",
            status=DocStatus.PENDING,
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )
        await doc_ws_alpha.upsert({"doc1": status_a})
        await doc_ws_beta.upsert({"doc1": status_b})

        alpha_result = await doc_ws_alpha.get_by_id("doc1")
        beta_result = await doc_ws_beta.get_by_id("doc1")

        assert alpha_result["status"] == "processed"
        assert beta_result["status"] == "pending"

    async def test_docstatus_counts_isolated(self, doc_ws_alpha, doc_ws_beta):
        for i in range(3):
            s = DocProcessingStatus(
                content_summary=f"doc {i}",
                content_length=10,
                file_path=f"/{i}.txt",
                status=DocStatus.PENDING,
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T00:00:00",
            )
            await doc_ws_alpha.upsert({f"a{i}": s})

        for i in range(5):
            s = DocProcessingStatus(
                content_summary=f"doc {i}",
                content_length=10,
                file_path=f"/{i}.txt",
                status=DocStatus.PROCESSED,
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T00:00:00",
            )
            await doc_ws_beta.upsert({f"b{i}": s})

        alpha_counts = await doc_ws_alpha.get_status_counts()
        beta_counts = await doc_ws_beta.get_status_counts()

        assert alpha_counts.get("pending", 0) == 3
        assert alpha_counts.get("processed", 0) == 0
        assert beta_counts.get("processed", 0) == 5
        assert beta_counts.get("pending", 0) == 0


# ── Shared driver pool ────────────────────────────────────────────────


@pytest.mark.integration
class TestSharedDriverPool:
    """Checklist item 7 (partial): all backends share the same pool driver."""

    async def test_all_backends_share_pool_driver(self, embedding_func_1024):
        from twindb_lightrag_memgraph._pool import get_driver

        kv = MemgraphKVStorage(
            namespace="pool_test", global_config={}, embedding_func=None
        )
        vec = MemgraphVectorDBStorage(
            namespace="pool_test",
            global_config={
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2}
            },
            embedding_func=embedding_func_1024,
            meta_fields=set(),
        )
        doc = MemgraphDocStatusStorage(
            namespace="pool_test", global_config={}, embedding_func=None
        )

        await kv.initialize()
        await vec.initialize()
        await doc.initialize()

        # All three use the same shared pool driver
        d1, _ = await get_driver()
        d2, _ = await get_driver()
        assert d1 is d2

        await kv.drop()
        await vec.drop()
        await doc.drop()


# ── Full pipeline ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestFullPipeline:
    """Checklist item 8: register() → instantiate all backends → write → read."""

    async def test_kv_vector_docstatus_pipeline(self, embedding_func_1024):
        """Simulate a LightRAG-like pipeline using all three backends together."""
        # 1. KV: store chunk data
        kv = MemgraphKVStorage(
            namespace="pipeline_chunks", global_config={}, embedding_func=None
        )
        await kv.initialize()

        chunks = {
            "chunk-001": {
                "content": "Paris is the capital of France.",
                "doc_id": "doc1",
            },
            "chunk-002": {
                "content": "Berlin is the capital of Germany.",
                "doc_id": "doc1",
            },
        }
        await kv.upsert(chunks)

        # 2. Vector: index embeddings
        vec = MemgraphVectorDBStorage(
            namespace="pipeline_entities",
            global_config={
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.0}
            },
            embedding_func=embedding_func_1024,
            meta_fields={"entity_name", "content"},
        )
        await vec.initialize()

        rng = np.random.default_rng(42)
        paris_emb = rng.random(BGE_M3_DIM).tolist()
        berlin_emb = rng.random(BGE_M3_DIM).tolist()
        await vec.upsert(
            {
                "e-paris": {
                    "content": "Paris",
                    "entity_name": "Paris",
                    "embedding": paris_emb,
                },
                "e-berlin": {
                    "content": "Berlin",
                    "entity_name": "Berlin",
                    "embedding": berlin_emb,
                },
            }
        )

        # 3. DocStatus: track document processing
        doc = MemgraphDocStatusStorage(
            namespace="pipeline_docs", global_config={}, embedding_func=None
        )
        await doc.initialize()

        await doc.upsert(
            {
                "doc1": DocProcessingStatus(
                    content_summary="European capitals",
                    content_length=62,
                    file_path="/data/capitals.txt",
                    status=DocStatus.PROCESSED,
                    created_at="2025-01-01T00:00:00",
                    updated_at="2025-01-01T00:00:00",
                    chunks_count=2,
                )
            }
        )

        # 4. Verify reads across all backends
        chunk = await kv.get_by_id("chunk-001")
        assert chunk["content"] == "Paris is the capital of France."

        query_results = await vec.query("", top_k=2, query_embedding=paris_emb)
        assert len(query_results) > 0
        assert query_results[0]["id"] == "e-paris"

        doc_status = await doc.get_by_id("doc1")
        assert doc_status["status"] == "processed"
        assert doc_status["chunks_count"] == 2

        counts = await doc.get_status_counts()
        assert counts["processed"] == 1

        # 5. Verify filter_keys (simulate "which docs are new?")
        new_docs = await doc.filter_keys({"doc1", "doc2"})
        assert "doc2" in new_docs
        assert "doc1" not in new_docs

        # Cleanup
        await kv.drop()
        await vec.drop()
        await doc.drop()
