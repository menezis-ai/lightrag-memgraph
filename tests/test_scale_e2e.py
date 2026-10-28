"""Opt-in real Memgraph latency qualification, shared with the stable adapter.

No register() call: these tests exercise KV/vector/status storage directly and
do not install graph/runtime patches from a legacy branch.
"""

import asyncio
import importlib
import importlib.metadata
import json
import os
import time
from uuid import uuid4

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

# Resolve historical wheels by their installed namespace without rewriting them.
_namespace = "lightrag_memgraph"
if _distribution_name := os.environ.get("SCALE_ADAPTER_DISTRIBUTION"):
    _distribution = importlib.metadata.distribution(_distribution_name)
    _namespaces = (_distribution.read_text("top_level.txt") or "").split()
    if len(_namespaces) != 1 or not _namespaces[0].isidentifier():
        raise RuntimeError("Scale adapter must expose exactly one Python package")
    _namespace = _namespaces[0]

_pool = importlib.import_module(f"{_namespace}._pool")
MemgraphDocStatusStorage = importlib.import_module(
    f"{_namespace}.docstatus_impl"
).MemgraphDocStatusStorage
MemgraphVectorDBStorage = importlib.import_module(
    f"{_namespace}.vector_impl"
).MemgraphVectorDBStorage

pytestmark = [pytest.mark.integration, pytest.mark.scale]


async def _no_embedding_api(texts):
    raise AssertionError(
        "Scale tests supply deterministic embeddings; no provider may be called"
    )


@pytest.fixture
async def stores(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "scale_" + uuid4().hex)
    status = MemgraphDocStatusStorage("doc_status", {}, None)
    vectors = MemgraphVectorDBStorage(
        "entities",
        {"vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2}},
        EmbeddingFunc(embedding_dim=128, max_token_size=8192, func=_no_embedding_api),
        meta_fields={"content", "entity_name"},
    )
    try:
        yield status, vectors
    finally:
        for store in (vectors, status):
            # Only this test's randomly allocated labels/indexes are removed.
            await store.drop()
            async with _pool.get_session() as session:
                rows = await (await session.run("SHOW INDEX INFO")).data()
                for row in rows:
                    if row.get("label") != store._label():
                        continue
                    prop = row.get("property")
                    prop = (
                        prop[0] if isinstance(prop, list) and len(prop) == 1 else prop
                    )
                    if isinstance(prop, str):
                        result = await session.run(
                            f"DROP INDEX ON :`{store._label()}`(`{prop}`)"
                        )
                        await result.consume()
        await _pool.close_driver()


def _record(request, operation, size, latencies, budget, concurrency):
    values = np.asarray(latencies) * 1000
    metrics = {
        "operation": operation,
        "dataset_size": size,
        "samples": len(values),
        "concurrency": concurrency,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
        "budget_ms": budget,
        "passed": bool(values.max() < budget),
    }
    request.node.user_properties.append(("scale_metrics", json.dumps(metrics)))
    assert metrics["passed"], json.dumps(metrics, indent=2)


@pytest.mark.parametrize("size,budget_ms", [(10_000, 2_000), (100_000, 10_000)])
async def test_document_pagination_at_scale(stores, request, size, budget_ms):
    store, _ = stores
    await store.initialize()
    for start in range(0, size, 1000):
        await store.upsert(
            {
                f"doc-{i:06d}": {
                    "status": "processed",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": f"2026-01-01T00:00:{i % 60:02d}Z",
                    "file_path": f"report-{i}.pdf",
                    "content_summary": "Scale qualification document",
                    "content_length": 10_000,
                    "chunks_list": [f"chunk-{i}"],
                    "metadata": {},
                }
                for i in range(start, min(start + 1000, size))
            }
        )

    async def query(page):
        started = time.perf_counter()
        docs, total = await store.get_docs_paginated(page=page, page_size=50)
        elapsed = time.perf_counter() - started
        assert total == size
        assert len(docs) == 50
        assert len({doc_id for doc_id, _ in docs}) == 50
        return elapsed

    await query(1)  # Warm the query plan outside the measured repetitions.
    latencies = []
    for _ in range(3):
        latencies.extend(
            await asyncio.gather(
                *(query(page) for page in (1, 2, 3, 4, size // 50, 5, 6, 7))
            )
        )
    _record(request, "paginated", size, latencies, budget_ms, concurrency=8)


async def test_vector_query_at_10k(stores, request):
    _, store = stores
    await store.initialize()
    size = 10_000
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(size, 128)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    for start in range(0, size, 500):
        await store.upsert(
            {
                f"vec-{i:06d}": {
                    "embedding": embeddings[i].tolist(),
                    "content": f"entity {i}",
                    "entity_name": f"entity_{i}",
                }
                for i in range(start, min(start + 500, size))
            }
        )

    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`{store._label()}`) RETURN count(n) AS count"
        )
        assert (await result.single())["count"] == size
        await result.consume()

    async def query(index):
        started = time.perf_counter()
        results = await store.query(
            "", top_k=10, query_embedding=embeddings[index].tolist()
        )
        elapsed = time.perf_counter() - started
        assert any(hit["id"] == f"vec-{index:06d}" for hit in results)
        return elapsed

    await query(0)
    latencies = []
    for start in (0, 100, 1000):
        latencies.extend(
            await asyncio.gather(*(query(i) for i in range(start, start + 8)))
        )
    _record(request, "vector-query", size, latencies, 500, concurrency=8)
