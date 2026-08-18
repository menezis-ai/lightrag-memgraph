"""Query post-processing N+1 benchmark.

Run as a script:
```
uv run python tests/benchmarks/query_postprocessing_batching.py
```
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from collections import Counter
from typing import Any

from twindb_lightrag_memgraph.server.query import doc_lookup
from twindb_lightrag_memgraph.server.query import response_sources

ITERATIONS = 60
ITEM_COUNT = 32
READ_DELAY_SECONDS = 0.002
READ_CAPACITY = 20
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 24


class _ReadPool:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(READ_CAPACITY)
        self.queries: Counter[str] = Counter()

    async def query(self, kind: str) -> None:
        async with self._semaphore:
            self.queries[kind] += 1
            await asyncio.sleep(READ_DELAY_SECONDS)


class _DocStatus:
    def __init__(self, pool: _ReadPool) -> None:
        self.pool = pool

    async def get_docs_by_chunks(self, chunk_ids: list[str]):
        await self.pool.query("chunk")
        return {f"doc:{chunk_id}": object() for chunk_id in chunk_ids}

    async def get_doc_by_file_path(self, file_path: str):
        await self.pool.query("path")
        return {"id": f"doc:{file_path}", "file_path": file_path}

    async def get_docs_by_file_paths(self, file_paths: list[str]):
        await self.pool.query("path")
        return {
            file_path: {"id": f"doc:{file_path}", "file_path": file_path}
            for file_path in file_paths
        }


class _TextChunks:
    def __init__(self, pool: _ReadPool) -> None:
        self.pool = pool

    async def get_by_ids(self, chunk_ids: list[str]):
        await self.pool.query("chunk")
        return [
            {
                "chunk_id": chunk_id,
                "full_doc_id": f"doc:{chunk_id}",
            }
            for chunk_id in chunk_ids
        ]


class _Rag:
    def __init__(self, pool: _ReadPool) -> None:
        self.doc_status = _DocStatus(pool)
        self.text_chunks = _TextChunks(pool)


class _Tags:
    def __init__(self, pool: _ReadPool) -> None:
        self.pool = pool

    async def one(self, doc_id: str, _folder: str) -> set[str]:
        await self.pool.query("tag")
        return {"keep"} if not doc_id.endswith("-drop") else {"drop"}

    async def batch(self, doc_ids: list[str], _folder: str) -> dict[str, set[str]]:
        await self.pool.query("tag")
        return {
            doc_id: {"keep"} if not doc_id.endswith("-drop") else {"drop"}
            for doc_id in doc_ids
        }


def _chunk_ids() -> list[str]:
    return [f"chunk-{idx:02d}" for idx in range(ITEM_COUNT)]


def _file_paths() -> list[str]:
    return [f"file-{idx:02d}.pdf" for idx in range(ITEM_COUNT)]


async def _baseline_chunk_docs(rag: Any) -> dict[str, str]:
    chunk_ids = _chunk_ids()
    doc_ids = await asyncio.gather(
        *(doc_lookup._resolve_doc_for_chunk(rag, chunk_id) for chunk_id in chunk_ids)
    )
    return dict(zip(chunk_ids, doc_ids))


async def _baseline_path_docs(rag: Any) -> dict[str, str]:
    file_paths = _file_paths()
    doc_ids = await asyncio.gather(
        *(
            doc_lookup._resolve_doc_for_file_path(rag, file_path)
            for file_path in file_paths
        )
    )
    return dict(zip(file_paths, doc_ids))


def _sources() -> list[dict[str, str]]:
    return [
        {
            "doc_id": f"doc-{idx:02d}{'-drop' if idx == ITEM_COUNT - 1 else ''}",
            "name": f"file-{idx:02d}.pdf",
        }
        for idx in range(ITEM_COUNT)
    ]


async def _one_request(optimized: bool, pool: _ReadPool) -> None:
    rag = _Rag(pool)
    tags = _Tags(pool)
    if optimized:
        chunk_docs = await doc_lookup._resolve_chunk_to_doc_id(rag, _chunk_ids())
        path_docs = await doc_lookup._resolve_file_paths_to_doc_ids(rag, _file_paths())
        kept, incomplete = await response_sources._filter_sources_by_advanced_filters(
            _sources(),
            tag_filter={"all": ["keep"]},
            doc_filter=None,
            folder="default",
            fetch_doc_tags=tags.one,
            fetch_doc_tags_batch=tags.batch,
        )
    else:
        chunk_docs = await _baseline_chunk_docs(rag)
        path_docs = await _baseline_path_docs(rag)
        kept, incomplete = await response_sources._filter_sources_by_advanced_filters(
            _sources(),
            tag_filter={"all": ["keep"]},
            doc_filter=None,
            folder="default",
            fetch_doc_tags=tags.one,
        )

    assert len(chunk_docs) == ITEM_COUNT
    assert len(path_docs) == ITEM_COUNT
    assert len(kept) == ITEM_COUNT - 1
    assert incomplete is False


async def _one_subpart(kind: str, optimized: bool, pool: _ReadPool) -> None:
    rag = _Rag(pool)
    tags = _Tags(pool)
    if kind == "chunk":
        if optimized:
            resolved = await doc_lookup._resolve_chunk_to_doc_id(rag, _chunk_ids())
        else:
            resolved = await _baseline_chunk_docs(rag)
        assert len(resolved) == ITEM_COUNT
        return
    if kind == "path":
        if optimized:
            resolved = await doc_lookup._resolve_file_paths_to_doc_ids(
                rag, _file_paths()
            )
        else:
            resolved = await _baseline_path_docs(rag)
        assert len(resolved) == ITEM_COUNT
        return
    if kind == "tag":
        kept, incomplete = await response_sources._filter_sources_by_advanced_filters(
            _sources(),
            tag_filter={"all": ["keep"]},
            doc_filter=None,
            folder="default",
            fetch_doc_tags=tags.one,
            fetch_doc_tags_batch=tags.batch if optimized else None,
        )
        assert len(kept) == ITEM_COUNT - 1
        assert incomplete is False
        return
    raise ValueError(f"unknown subpart: {kind}")


def _percentile(values: list[float], n: int, index: int) -> float:
    return statistics.quantiles(values, n=n)[index]


async def _measure_load(
    label: str,
    *,
    optimized: bool,
    requests: int,
    concurrency: int,
    subpart: str | None = None,
) -> dict[str, Any]:
    pool = _ReadPool()
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()

    async def worker() -> None:
        started = time.perf_counter()
        if subpart is None:
            await _one_request(optimized, pool)
        else:
            await _one_subpart(subpart, optimized, pool)
        durations_ms.append((time.perf_counter() - started) * 1000)

    for offset in range(0, requests, concurrency):
        batch_size = min(concurrency, requests - offset)
        await asyncio.gather(*(worker() for _ in range(batch_size)))

    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "requests": len(durations_ms),
        "concurrency": concurrency,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": _percentile(durations_ms, 20, 18),
        "p99_ms": _percentile(durations_ms, 100, 98),
        "req_per_s": len(durations_ms) / elapsed,
        "peak_mb": peak / 1024 / 1024,
        "queries": dict(pool.queries),
    }


async def _structural_counts(optimized: bool) -> Counter[str]:
    pool = _ReadPool()
    await _one_request(optimized, pool)
    return pool.queries


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entrypoint returning one ratio and exact round-trip guards."""
    requests = iterations or ITERATIONS
    baseline = await _measure_load(
        "baseline",
        optimized=False,
        requests=requests,
        concurrency=1,
    )
    optimized = await _measure_load(
        "optimized",
        optimized=True,
        requests=requests,
        concurrency=1,
    )
    subpart_cases: list[dict[str, Any]] = []
    for subpart in ("chunk", "path", "tag"):
        subpart_before = await _measure_load(
            f"baseline_{subpart}",
            optimized=False,
            requests=requests,
            concurrency=1,
            subpart=subpart,
        )
        subpart_after = await _measure_load(
            f"optimized_{subpart}",
            optimized=True,
            requests=requests,
            concurrency=1,
            subpart=subpart,
        )
        subpart_cases.append(
            {
                "name": f"query post-processing {subpart} batching",
                "kind": "ratio",
                "baseline_ms": subpart_before["mean_ms"],
                "optimized_ms": subpart_after["mean_ms"],
            }
        )
    baseline_counts = await _structural_counts(optimized=False)
    counts = await _structural_counts(optimized=True)
    return [
        {
            "name": "query post-processing composite N+1 batching",
            "kind": "ratio",
            "baseline_ms": baseline["mean_ms"],
            "optimized_ms": optimized["mean_ms"],
        },
        *subpart_cases,
        {
            "name": "chunk-to-document set-based read",
            "kind": "structural",
            "passed": (baseline_counts["chunk"] == ITEM_COUNT and counts["chunk"] == 1),
            "detail": (
                f"expected {ITEM_COUNT}->1 chunk queries, observed "
                f"{baseline_counts['chunk']}->{counts['chunk']}"
            ),
        },
        {
            "name": "file-path-to-document set-based read",
            "kind": "structural",
            "passed": (baseline_counts["path"] == ITEM_COUNT and counts["path"] == 1),
            "detail": (
                f"expected {ITEM_COUNT}->1 path queries, observed "
                f"{baseline_counts['path']}->{counts['path']}"
            ),
        },
        {
            "name": "document-tag set-based read",
            "kind": "structural",
            "passed": baseline_counts["tag"] == ITEM_COUNT and counts["tag"] == 1,
            "detail": (
                f"expected {ITEM_COUNT}->1 tag queries, observed "
                f"{baseline_counts['tag']}->{counts['tag']}"
            ),
        },
    ]


def _print_result(result: dict[str, Any]) -> None:
    print(
        f"{result['label']}: mean={result['mean_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms p99={result['p99_ms']:.3f}ms "
        f"throughput={result['req_per_s']:.1f} req/s "
        f"peak={result['peak_mb']:.3f}MB queries={result['queries']}"
    )


async def main() -> None:
    baseline = await _measure_load(
        "baseline_isolated",
        optimized=False,
        requests=ITERATIONS,
        concurrency=1,
    )
    optimized = await _measure_load(
        "optimized_isolated",
        optimized=True,
        requests=ITERATIONS,
        concurrency=1,
    )
    sustained_before = await _measure_load(
        "baseline_sustained",
        optimized=False,
        requests=ITERATIONS,
        concurrency=SUSTAINED_CONCURRENCY,
    )
    sustained_after = await _measure_load(
        "optimized_sustained",
        optimized=True,
        requests=ITERATIONS,
        concurrency=SUSTAINED_CONCURRENCY,
    )
    peak_requests = PEAK_CONCURRENCY * 2
    peak_before = await _measure_load(
        "baseline_peak",
        optimized=False,
        requests=peak_requests,
        concurrency=PEAK_CONCURRENCY,
    )
    peak_after = await _measure_load(
        "optimized_peak",
        optimized=True,
        requests=peak_requests,
        concurrency=PEAK_CONCURRENCY,
    )
    subpart_results: list[dict[str, Any]] = []
    for subpart in ("chunk", "path", "tag"):
        subpart_results.extend(
            [
                await _measure_load(
                    f"baseline_{subpart}",
                    optimized=False,
                    requests=ITERATIONS,
                    concurrency=1,
                    subpart=subpart,
                ),
                await _measure_load(
                    f"optimized_{subpart}",
                    optimized=True,
                    requests=ITERATIONS,
                    concurrency=1,
                    subpart=subpart,
                ),
            ]
        )

    for result in (
        baseline,
        optimized,
        sustained_before,
        sustained_after,
        peak_before,
        peak_after,
        *subpart_results,
    ):
        _print_result(result)

    gain = (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    throughput = (
        (optimized["req_per_s"] - baseline["req_per_s"]) / baseline["req_per_s"] * 100
    )
    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> "
        f"{optimized['mean_ms']:.3f}ms ({gain:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['req_per_s']:.1f} req/s -> "
        f"{optimized['req_per_s']:.1f} req/s (+{throughput:.1f}%)"
    )
    print(
        f"round-trips/request: {sum(baseline['queries'].values()) // ITERATIONS} "
        f"-> {sum(optimized['queries'].values()) // ITERATIONS}"
    )


if __name__ == "__main__":
    asyncio.run(main())
