"""Micro-benchmark: legacy fallback source doc-id resolution.

Run as a script:
``python tests/benchmarks/legacy_fallback_doc_ids.py``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.query import response_sources as rs

ITERATIONS = 50
TOP_K = 20
LOOKUP_DELAY_SECONDS = 0.01


class _ChunksVdb:
    async def query(self, _query: str, *, top_k: int) -> list[dict[str, Any]]:
        return [
            {
                "id": f"chunk-{idx:02d}",
                "file_path": f"doc-{idx:02d}.pdf",
                "chunk_order_index": idx,
            }
            for idx in range(top_k)
        ]


class _Rag:
    chunks_vdb = _ChunksVdb()


async def _fake_resolve_doc_for_chunk(_rag: Any, chunk_id: str) -> str:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return f"doc:{chunk_id}"


async def _baseline_build_sources_legacy_fallback(
    rag: Any, query: str, top_k: int
) -> list[dict[str, Any]]:
    chunks_vdb = getattr(rag, "chunks_vdb", None)
    if chunks_vdb is None:
        return []
    raw = await chunks_vdb.query(query, top_k=top_k)
    if not isinstance(raw, list):
        raw = []

    sources: list[dict[str, Any]] = []
    total = len(raw)
    for rank, chunk in enumerate(raw[:top_k]):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("id") or chunk.get("chunk_id") or ""
        file_path = (
            chunk.get("file_path")
            or chunk.get("source")
            or chunk_id
            or rs.UNKNOWN_SOURCE_NAME
        )
        doc_id = await rs._resolve_doc_for_chunk(rag, str(chunk_id))
        sources.append(
            {
                "n": rank + 1,
                "type": "file",
                "name": str(file_path),
                "meta": rs._chunk_to_meta(chunk),
                "score": rs._safe_get_score(chunk, rank, total),
                "doc_id": doc_id,
                "chunk_id": str(chunk_id) or None,
            }
        )
    return sources


async def _measure(label: str, fn) -> dict[str, float | str | int]:
    durations_ms: list[float] = []
    rag = _Rag()
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        sources = await fn(rag, "query", TOP_K)
        assert len(sources) == TOP_K
        assert sources[0]["doc_id"] == "doc:chunk-00"
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
    rs._resolve_doc_for_chunk = _fake_resolve_doc_for_chunk
    baseline = await _measure(
        "baseline_sequential", _baseline_build_sources_legacy_fallback
    )
    optimized = await _measure("optimized_parallel", rs._build_sources_legacy_fallback)

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
