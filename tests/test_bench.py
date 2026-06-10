"""
Performance benchmarks for twindb-lightrag-memgraph backends.

Measures latency (p50/p95/p99), throughput (ops/sec), and
scaling behavior at 100 / 1,000 / 10,000 items.

Run:
    MEMGRAPH_URI=bolt://localhost:7687 python -m pytest tests/test_bench.py -v -s
"""

import asyncio
import json
import statistics
import time
import uuid
from dataclasses import dataclass

import numpy as np
import pytest
from lightrag.base import BaseGraphStorage, DocProcessingStatus, DocStatus
from lightrag.utils import EmbeddingFunc

from twindb_lightrag_memgraph import register
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

register()

EMBEDDING_DIM = 128
SCALES = [100, 1_000, 10_000]


# ── Helpers ────────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    count: int
    total_s: float
    latencies_ms: list[float]

    @property
    def ops_per_sec(self) -> float:
        return self.count / self.total_s if self.total_s > 0 else 0

    @property
    def p50(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def p99(self) -> float:
        return _percentile(self.latencies_ms, 99)

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    def report(self) -> str:
        return (
            f"  {self.name:.<45s} "
            f"n={self.count:>6d}  "
            f"total={self.total_s:>7.2f}s  "
            f"ops/s={self.ops_per_sec:>8.0f}  "
            f"mean={self.mean:>7.2f}ms  "
            f"p50={self.p50:>7.2f}ms  "
            f"p95={self.p95:>7.2f}ms  "
            f"p99={self.p99:>7.2f}ms"
        )


def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def _random_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    return np.random.default_rng().random(dim).tolist()


def _make_doc_status(idx: int) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary=f"Document {idx} summary text for testing",
        content_length=1000 + idx,
        file_path=f"/docs/file_{idx}.pdf",
        status=DocStatus.PENDING,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
        track_id=f"batch-{idx % 10}",
    )


async def _mock_embed(texts: list[str]) -> np.ndarray:
    return (
        np.random.default_rng(seed=42)
        .random((len(texts), EMBEDDING_DIM))
        .astype(np.float32)
    )


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def embedding_func():
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=_mock_embed,
    )


@pytest.fixture
async def kv_store():
    store = MemgraphKVStorage(
        namespace="bench_kv",
        global_config={},
        embedding_func=None,
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.fixture
async def vec_store(embedding_func):
    store = MemgraphVectorDBStorage(
        namespace="bench_vec",
        global_config={
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
            },
        },
        embedding_func=embedding_func,
        meta_fields={"entity_name", "content"},
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.fixture
async def doc_store():
    store = MemgraphDocStatusStorage(
        namespace="bench_doc",
        global_config={},
        embedding_func=None,
    )
    await store.initialize()
    yield store
    await store.drop()


# ── KV Benchmarks ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestKVBenchmark:
    @pytest.mark.parametrize("n", SCALES)
    async def test_batch_upsert(self, kv_store, n):
        """Batch upsert of n items in a single call."""
        data = {
            f"key-{i}": {"value": f"data-{i}", "idx": i, "payload": "x" * 200}
            for i in range(n)
        }
        t0 = time.perf_counter()
        await kv_store.upsert(data)
        elapsed = time.perf_counter() - t0
        r = BenchResult("KV batch_upsert", n, elapsed, [elapsed * 1000])
        print(f"\n{r.report()}")
        await kv_store.drop()

    @pytest.mark.parametrize("n", SCALES)
    async def test_sequential_get_by_id(self, kv_store, n):
        """Sequential get_by_id for n items."""
        data = {f"key-{i}": {"value": i} for i in range(n)}
        await kv_store.upsert(data)

        latencies = []
        t0 = time.perf_counter()
        for i in range(n):
            t_start = time.perf_counter()
            result = await kv_store.get_by_id(f"key-{i}")
            latencies.append((time.perf_counter() - t_start) * 1000)
            assert result is not None
        total = time.perf_counter() - t0

        r = BenchResult("KV get_by_id (sequential)", n, total, latencies)
        print(f"\n{r.report()}")
        await kv_store.drop()

    @pytest.mark.parametrize("n", SCALES)
    async def test_batch_get_by_ids(self, kv_store, n):
        """Batch get_by_ids for n items in a single call."""
        data = {f"key-{i}": {"value": i} for i in range(n)}
        await kv_store.upsert(data)

        ids = [f"key-{i}" for i in range(n)]
        t0 = time.perf_counter()
        results = await kv_store.get_by_ids(ids)
        elapsed = time.perf_counter() - t0
        assert len(results) == n

        r = BenchResult("KV batch_get_by_ids", n, elapsed, [elapsed * 1000])
        print(f"\n{r.report()}")
        await kv_store.drop()

    @pytest.mark.parametrize("n", SCALES)
    async def test_filter_keys(self, kv_store, n):
        """filter_keys with n existing + n missing keys."""
        data = {f"key-{i}": {"value": i} for i in range(n)}
        await kv_store.upsert(data)

        all_keys = {f"key-{i}" for i in range(n)} | {f"missing-{i}" for i in range(n)}
        t0 = time.perf_counter()
        missing = await kv_store.filter_keys(all_keys)
        elapsed = time.perf_counter() - t0
        assert len(missing) == n

        r = BenchResult("KV filter_keys", len(all_keys), elapsed, [elapsed * 1000])
        print(f"\n{r.report()}")
        await kv_store.drop()

    async def test_concurrent_upsert(self, kv_store):
        """10 concurrent upsert batches of 100 items each."""
        n_workers = 10
        batch_size = 100

        async def worker(worker_id):
            data = {
                f"w{worker_id}-{i}": {"worker": worker_id, "idx": i}
                for i in range(batch_size)
            }
            await kv_store.upsert(data)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(w) for w in range(n_workers)])
        total = time.perf_counter() - t0

        total_items = n_workers * batch_size
        r = BenchResult(
            "KV concurrent_upsert (10x100)", total_items, total, [total * 1000]
        )
        print(f"\n{r.report()}")
        await kv_store.drop()


# ── Vector Benchmarks ──────────────────────────────────────────────────


@pytest.mark.integration
class TestVectorBenchmark:
    @pytest.mark.parametrize("n", SCALES)
    async def test_upsert(self, vec_store, n):
        """Sequential upsert of n vectors (with pre-computed embeddings)."""
        latencies = []
        t0 = time.perf_counter()
        for i in range(n):
            data = {
                f"vec-{i}": {
                    "content": f"entity {i} description text",
                    "entity_name": f"entity_{i}",
                    "embedding": _random_embedding(),
                }
            }
            t_start = time.perf_counter()
            await vec_store.upsert(data)
            latencies.append((time.perf_counter() - t_start) * 1000)
        total = time.perf_counter() - t0

        r = BenchResult("Vec upsert (sequential)", n, total, latencies)
        print(f"\n{r.report()}")
        await vec_store.drop()

    @pytest.mark.parametrize("n", [100, 1_000])
    async def test_query_latency(self, vec_store, n):
        """Query latency against n indexed vectors (50 queries)."""
        # Seed data
        for i in range(n):
            await vec_store.upsert(
                {
                    f"vec-{i}": {
                        "content": f"entity {i}",
                        "entity_name": f"e_{i}",
                        "embedding": _random_embedding(),
                    }
                }
            )

        # Run queries
        n_queries = 50
        latencies = []
        t0 = time.perf_counter()
        for _ in range(n_queries):
            qe = _random_embedding()
            t_start = time.perf_counter()
            results = await vec_store.query("", top_k=10, query_embedding=qe)
            latencies.append((time.perf_counter() - t_start) * 1000)
        total = time.perf_counter() - t0

        r = BenchResult(f"Vec query top_k=10 (n={n})", n_queries, total, latencies)
        print(f"\n{r.report()}")
        await vec_store.drop()

    async def test_get_vectors_by_ids(self, vec_store):
        """Retrieve 100 vectors by IDs from a 1000-vector index."""
        n = 1000
        for i in range(n):
            await vec_store.upsert({f"vec-{i}": {"embedding": _random_embedding()}})

        ids = [f"vec-{i}" for i in range(0, 100)]
        t0 = time.perf_counter()
        vectors = await vec_store.get_vectors_by_ids(ids)
        elapsed = time.perf_counter() - t0
        assert len(vectors) == 100

        r = BenchResult(
            "Vec get_vectors_by_ids (100/1000)", 100, elapsed, [elapsed * 1000]
        )
        print(f"\n{r.report()}")
        await vec_store.drop()

    async def test_delete_entity_at_scale(self, vec_store):
        """Delete entities from a 1000-vector index."""
        n = 1000
        for i in range(n):
            await vec_store.upsert(
                {
                    f"vec-{i}": {
                        "entity_name": f"entity_{i % 50}",
                        "embedding": _random_embedding(),
                    }
                }
            )

        latencies = []
        t0 = time.perf_counter()
        for i in range(50):
            t_start = time.perf_counter()
            await vec_store.delete_entity(f"entity_{i}")
            latencies.append((time.perf_counter() - t_start) * 1000)
        total = time.perf_counter() - t0

        r = BenchResult("Vec delete_entity (50 from 1000)", 50, total, latencies)
        print(f"\n{r.report()}")
        await vec_store.drop()


# ── DocStatus Benchmarks ───────────────────────────────────────────────


@pytest.mark.integration
class TestDocStatusBenchmark:
    @pytest.mark.parametrize("n", SCALES)
    async def test_upsert(self, doc_store, n):
        """Upsert n DocProcessingStatus entries."""
        latencies = []
        t0 = time.perf_counter()
        for i in range(n):
            t_start = time.perf_counter()
            await doc_store.upsert({f"doc-{i}": _make_doc_status(i)})
            latencies.append((time.perf_counter() - t_start) * 1000)
        total = time.perf_counter() - t0

        r = BenchResult("DocStatus upsert (sequential)", n, total, latencies)
        print(f"\n{r.report()}")
        await doc_store.drop()

    @pytest.mark.parametrize("n", [100, 1_000])
    async def test_get_docs_by_status(self, doc_store, n):
        """Query docs by status from n entries."""
        for i in range(n):
            status = _make_doc_status(i)
            if i % 3 == 0:
                status.status = DocStatus.PROCESSED
            await doc_store.upsert({f"doc-{i}": status})

        t0 = time.perf_counter()
        docs = await doc_store.get_docs_by_status(DocStatus.PENDING)
        elapsed = time.perf_counter() - t0

        r = BenchResult(
            f"DocStatus get_by_status (n={n})",
            len(docs),
            elapsed,
            [elapsed * 1000],
        )
        print(f"\n{r.report()}")
        await doc_store.drop()

    @pytest.mark.parametrize("n", [100, 1_000])
    async def test_paginated_query(self, doc_store, n):
        """Paginated query (10 pages of 50) from n entries."""
        for i in range(n):
            await doc_store.upsert({f"doc-{i}": _make_doc_status(i)})

        latencies = []
        pages = min(10, n // 50 + 1)
        t0 = time.perf_counter()
        for page in range(1, pages + 1):
            t_start = time.perf_counter()
            docs, total = await doc_store.get_docs_paginated(page=page, page_size=50)
            latencies.append((time.perf_counter() - t_start) * 1000)
        total_time = time.perf_counter() - t0

        r = BenchResult(f"DocStatus paginated (n={n})", pages, total_time, latencies)
        print(f"\n{r.report()}")
        await doc_store.drop()

    async def test_get_status_counts_at_scale(self, doc_store):
        """get_status_counts with 5000 entries across 5 statuses."""
        n = 5000
        statuses = list(DocStatus)
        for i in range(n):
            s = _make_doc_status(i)
            s.status = statuses[i % len(statuses)]
            await doc_store.upsert({f"doc-{i}": s})

        latencies = []
        for _ in range(20):
            t_start = time.perf_counter()
            counts = await doc_store.get_status_counts()
            latencies.append((time.perf_counter() - t_start) * 1000)
        total = sum(latencies) / 1000

        r = BenchResult("DocStatus status_counts (n=5000)", 20, total, latencies)
        print(f"\n{r.report()}")
        await doc_store.drop()

    async def test_concurrent_mixed_ops(self, doc_store):
        """Concurrent read/write: 5 writers + 5 readers."""
        n_per_worker = 100

        async def writer(wid):
            for i in range(n_per_worker):
                await doc_store.upsert({f"w{wid}-{i}": _make_doc_status(i)})

        async def reader(wid):
            results = []
            for i in range(n_per_worker):
                r = await doc_store.get_by_id(f"w{wid}-{i}")
                results.append(r)
            return results

        # Phase 1: concurrent writes
        t0 = time.perf_counter()
        await asyncio.gather(*[writer(w) for w in range(5)])
        write_time = time.perf_counter() - t0

        # Phase 2: concurrent reads
        t1 = time.perf_counter()
        await asyncio.gather(*[reader(w) for w in range(5)])
        read_time = time.perf_counter() - t1

        total_ops = 5 * n_per_worker
        print(
            f"\n  DocStatus concurrent_mixed_ops:"
            f"\n    Writers: 5x{n_per_worker} = {total_ops} ops in {write_time:.2f}s "
            f"({total_ops / write_time:.0f} ops/s)"
            f"\n    Readers: 5x{n_per_worker} = {total_ops} ops in {read_time:.2f}s "
            f"({total_ops / read_time:.0f} ops/s)"
        )
        await doc_store.drop()


# ── Graph Batch Benchmarks ────────────────────────────────────────────
#
# Compare sequential (base class N-query fallback) vs batched (UNWIND patch).
# Saves the base class methods before they're overridden, then calls both
# on the same data to get a fair A/B comparison.

# Keep references to the unpatched sequential implementations
_seq_get_nodes_batch = BaseGraphStorage.get_nodes_batch
_seq_node_degrees_batch = BaseGraphStorage.node_degrees_batch
_seq_edge_degrees_batch = BaseGraphStorage.edge_degrees_batch
_seq_get_edges_batch = BaseGraphStorage.get_edges_batch
_seq_get_nodes_edges_batch = BaseGraphStorage.get_nodes_edges_batch


@pytest.fixture
async def graph_store():
    """Initialize MemgraphStorage directly (bypasses USE DATABASE for Community)."""
    import os

    from lightrag.kg.memgraph_impl import MemgraphStorage
    from neo4j import AsyncGraphDatabase as _AsyncGD

    ws = f"bench_{uuid.uuid4().hex[:8]}"
    uri = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    driver = _AsyncGD.driver(uri, auth=("", ""))

    store = MemgraphStorage.__new__(MemgraphStorage)
    store.workspace = ws
    store.namespace = ws
    store._driver = driver
    store._DATABASE = ""

    # Create index for the workspace label
    async with driver.session() as session:
        try:
            await session.run(f"CREATE INDEX ON :`{ws}`(entity_id)")
        except Exception:
            pass
        await session.run("RETURN 1")

    yield store
    # Cleanup
    try:
        async with driver.session() as session:
            await session.run(f"MATCH (n:`{ws}`) DETACH DELETE n")
    except Exception:
        pass
    await driver.close()


async def _seed_graph(store, n_nodes: int, n_edges_per_node: int = 2):
    """Insert n_nodes with edges forming a chain + extra connections."""
    node_ids = [f"entity_{i}" for i in range(n_nodes)]
    for nid in node_ids:
        await store.upsert_node(
            nid,
            {
                "entity_id": nid,
                "entity_type": "concept",
                "description": f"Test entity {nid}",
                "source_id": "bench",
            },
        )
    # Create edges: each node connects to its next `n_edges_per_node` neighbors
    edge_pairs = []
    for i in range(n_nodes):
        for j in range(1, n_edges_per_node + 1):
            tgt = (i + j) % n_nodes
            if tgt != i:
                src_id, tgt_id = node_ids[i], node_ids[tgt]
                await store.upsert_edge(
                    src_id,
                    tgt_id,
                    {
                        "weight": 1.0,
                        "description": f"{src_id} -> {tgt_id}",
                        "keywords": "test",
                        "source_id": "bench",
                    },
                )
                edge_pairs.append((src_id, tgt_id))
    return node_ids, edge_pairs


GRAPH_SCALES = [10, 40, 100]


@pytest.mark.integration
class TestGraphBatchBenchmark:
    """A/B benchmark: sequential (N queries) vs batched (single UNWIND)."""

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_get_nodes_batch(self, graph_store, n):
        node_ids, _ = await _seed_graph(graph_store, n)

        # Sequential (base class fallback)
        t0 = time.perf_counter()
        seq_result = await _seq_get_nodes_batch(graph_store, node_ids)
        seq_ms = (time.perf_counter() - t0) * 1000

        # Batched (UNWIND patch)
        t0 = time.perf_counter()
        batch_result = await graph_store.get_nodes_batch(node_ids)
        batch_ms = (time.perf_counter() - t0) * 1000

        assert len(seq_result) == len(batch_result) == n
        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        print(
            f"\n  get_nodes_batch (n={n:>3d}): "
            f"seq={seq_ms:>7.1f}ms  batch={batch_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_node_degrees_batch(self, graph_store, n):
        node_ids, _ = await _seed_graph(graph_store, n)

        t0 = time.perf_counter()
        seq_result = await _seq_node_degrees_batch(graph_store, node_ids)
        seq_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        batch_result = await graph_store.node_degrees_batch(node_ids)
        batch_ms = (time.perf_counter() - t0) * 1000

        assert len(seq_result) == len(batch_result) == n
        # Verify same values
        for nid in node_ids:
            assert seq_result[nid] == batch_result[nid], f"Mismatch for {nid}"
        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        print(
            f"\n  node_degrees_batch (n={n:>3d}): "
            f"seq={seq_ms:>7.1f}ms  batch={batch_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_get_edges_batch(self, graph_store, n):
        node_ids, edge_pairs = await _seed_graph(graph_store, n)
        pairs = [{"src": s, "tgt": t} for s, t in edge_pairs[:n]]

        t0 = time.perf_counter()
        seq_result = await _seq_get_edges_batch(graph_store, pairs)
        seq_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        batch_result = await graph_store.get_edges_batch(pairs)
        batch_ms = (time.perf_counter() - t0) * 1000

        assert len(seq_result) == len(batch_result)
        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        print(
            f"\n  get_edges_batch (n={len(pairs):>3d}): "
            f"seq={seq_ms:>7.1f}ms  batch={batch_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_edge_degrees_batch(self, graph_store, n):
        node_ids, edge_pairs = await _seed_graph(graph_store, n)
        pairs = edge_pairs[:n]

        t0 = time.perf_counter()
        seq_result = await _seq_edge_degrees_batch(graph_store, pairs)
        seq_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        batch_result = await graph_store.edge_degrees_batch(pairs)
        batch_ms = (time.perf_counter() - t0) * 1000

        assert len(seq_result) == len(batch_result)
        for key in seq_result:
            assert seq_result[key] == batch_result[key], f"Mismatch for {key}"
        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        print(
            f"\n  edge_degrees_batch (n={len(pairs):>3d}): "
            f"seq={seq_ms:>7.1f}ms  batch={batch_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_get_nodes_edges_batch(self, graph_store, n):
        node_ids, _ = await _seed_graph(graph_store, n)

        t0 = time.perf_counter()
        seq_result = await _seq_get_nodes_edges_batch(graph_store, node_ids)
        seq_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        batch_result = await graph_store.get_nodes_edges_batch(node_ids)
        batch_ms = (time.perf_counter() - t0) * 1000

        assert len(seq_result) == len(batch_result) == n
        speedup = seq_ms / batch_ms if batch_ms > 0 else float("inf")
        print(
            f"\n  get_nodes_edges_batch (n={n:>3d}): "
            f"seq={seq_ms:>7.1f}ms  batch={batch_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )

    # -- Fused benchmarks: gather(2 queries) vs single fused query --

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_fused_get_nodes_with_degrees(self, graph_store, n):
        node_ids, _ = await _seed_graph(graph_store, n)

        # gather(get_nodes_batch, node_degrees_batch) — 2 queries
        t0 = time.perf_counter()
        nodes_g, degrees_g = await asyncio.gather(
            graph_store.get_nodes_batch(node_ids),
            graph_store.node_degrees_batch(node_ids),
        )
        gather_ms = (time.perf_counter() - t0) * 1000

        # Fused single query
        t0 = time.perf_counter()
        nodes_f, degrees_f = await graph_store.get_nodes_with_degrees_batch(node_ids)
        fused_ms = (time.perf_counter() - t0) * 1000

        assert len(nodes_g) == len(nodes_f) == n
        for nid in node_ids:
            assert degrees_g[nid] == degrees_f[nid], f"Degree mismatch for {nid}"
        speedup = gather_ms / fused_ms if fused_ms > 0 else float("inf")
        print(
            f"\n  nodes+degrees FUSED (n={n:>3d}): "
            f"gather={gather_ms:>7.1f}ms  fused={fused_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )

    @pytest.mark.parametrize("n", GRAPH_SCALES)
    async def test_fused_get_edges_with_degrees(self, graph_store, n):
        node_ids, edge_pairs = await _seed_graph(graph_store, n)
        pairs = [{"src": s, "tgt": t} for s, t in edge_pairs[:n]]
        tuples = edge_pairs[:n]

        # gather(get_edges_batch, edge_degrees_batch) — 2 queries
        t0 = time.perf_counter()
        edges_g, edeg_g = await asyncio.gather(
            graph_store.get_edges_batch(pairs),
            graph_store.edge_degrees_batch(tuples),
        )
        gather_ms = (time.perf_counter() - t0) * 1000

        # Fused single query
        t0 = time.perf_counter()
        edges_f, edeg_f = await graph_store.get_edges_with_degrees_batch(pairs)
        fused_ms = (time.perf_counter() - t0) * 1000

        assert len(edges_g) == len(edges_f)
        speedup = gather_ms / fused_ms if fused_ms > 0 else float("inf")
        print(
            f"\n  edges+degrees FUSED (n={len(pairs):>3d}): "
            f"gather={gather_ms:>7.1f}ms  fused={fused_ms:>7.1f}ms  "
            f"speedup={speedup:>5.1f}x"
        )
