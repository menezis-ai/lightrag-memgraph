"""Micro-benchmark: tag-filter doc-id resolution for /query/data rows.

Run as a script:
``python tests/benchmarks/query_data_row_doc_ids.py``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.query import query_data_filters as qdf

ITERATIONS = 40
LOOKUP_DELAY_SECONDS = 0.01


class _Rag:
    doc_status = object()


async def _fake_chunk(_rag: Any, chunk_id: str) -> str:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return f"doc:{chunk_id}"


async def _fake_file(_rag: Any, file_path: str) -> str:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return f"doc:{file_path}"


async def _baseline_doc_ids_for_query_data_row(
    rag: Any, row: dict[str, Any]
) -> set[str]:
    doc_ids = qdf._direct_doc_ids_for_query_data_row(row)
    for chunk_id in qdf._chunk_ids_for_query_data_row(row):
        doc_id = await qdf._resolve_doc_for_chunk(rag, chunk_id)
        if doc_id:
            doc_ids.add(doc_id)
    for file_path in qdf._file_path_candidates_for_query_data_row(row):
        doc_id = await qdf._resolve_doc_for_file_path(rag, file_path)
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


async def _measure(label: str, fn, row: dict[str, Any]) -> dict[str, float | str | int]:
    durations_ms: list[float] = []
    rag = _Rag()
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        doc_ids = await fn(rag, row)
        assert len(doc_ids) == 7
        durations_ms.append((time.perf_counter() - start) * 1000)
    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "iterations": ITERATIONS,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "ops_per_s": ITERATIONS / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


async def main() -> None:
    qdf._resolve_doc_for_chunk = _fake_chunk
    qdf._resolve_doc_for_file_path = _fake_file
    row = {
        "source_id": "chunk-a,chunk-b,chunk-c,chunk-d",
        "file_path": "file-a.pdf",
        "source": "file-b.pdf",
        "name": "file-c.pdf",
    }

    baseline = await _measure(
        "baseline_sequential", _baseline_doc_ids_for_query_data_row, row
    )
    optimized = await _measure(
        "optimized_parallel", qdf._doc_ids_for_query_data_row, row
    )

    for result in (baseline, optimized):
        print(result)

    speedup = (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    throughput_delta = (
        (optimized["ops_per_s"] - baseline["ops_per_s"]) / baseline["ops_per_s"] * 100
    )
    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> "
        f"{optimized['mean_ms']:.3f}ms ({speedup:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['ops_per_s']:.1f} req/s -> "
        f"{optimized['ops_per_s']:.1f} req/s ({throughput_delta:.1f}%)"
    )
    print(f"peak_mem: {baseline['peak_mb']:.3f}MB -> " f"{optimized['peak_mb']:.3f}MB")


if __name__ == "__main__":
    asyncio.run(main())
