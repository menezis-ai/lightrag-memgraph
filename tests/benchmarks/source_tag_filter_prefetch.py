"""Micro-benchmark: source tag filtering prefetch for /query responses.

Run as a script:
``python tests/benchmarks/source_tag_filter_prefetch.py``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.query import response_sources as rs

ITERATIONS = 50
SOURCE_COUNT = 20
LOOKUP_DELAY_SECONDS = 0.01


async def _baseline_filter_sources_by_advanced_filters(
    sources: list[dict[str, Any]],
    *,
    tag_filter: dict[str, list[str]] | None,
    doc_filter: dict[str, list[str]] | None,
    folder: str,
    fetch_doc_tags: Any,
) -> tuple[list[dict[str, Any]], bool]:
    tag_required, tag_optional = rs._tag_filter_terms(tag_filter)
    doc_required, doc_optional = rs._doc_filter_terms(doc_filter)
    if not tag_required and not tag_optional and not doc_required and not doc_optional:
        return sources, False

    tags_cache: dict[str, set[str]] = {}
    kept: list[dict[str, Any]] = []
    has_unverified = False
    for source in sources:
        if not rs._source_matches_doc_filter(source, doc_filter):
            if not rs._source_doc_candidates(source):
                has_unverified = True
            continue
        if not await rs._source_matches_tag_filter(
            source, tag_filter, folder, tags_cache, fetch_doc_tags
        ):
            if not source.get("doc_id"):
                has_unverified = True
            continue
        kept.append(source)
    return kept, has_unverified


async def _fetch_doc_tags(doc_id: str, _folder: str) -> set[str]:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return {"keep"} if not doc_id.endswith("-drop") else {"drop"}


async def _measure(
    label: str, fn, sources: list[dict[str, Any]]
) -> dict[str, float | str | int]:
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        kept, incomplete = await fn(
            sources,
            tag_filter={"all": ["keep"]},
            doc_filter=None,
            folder="default",
            fetch_doc_tags=_fetch_doc_tags,
        )
        assert len(kept) == SOURCE_COUNT - 2
        assert incomplete is False
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
    sources = [
        {
            "doc_id": f"doc-{idx:02d}{'-drop' if idx in {5, 13} else ''}",
            "name": f"source-{idx:02d}.pdf",
        }
        for idx in range(SOURCE_COUNT)
    ]

    baseline = await _measure(
        "baseline_sequential", _baseline_filter_sources_by_advanced_filters, sources
    )
    optimized = await _measure(
        "optimized_prefetch", rs._filter_sources_by_advanced_filters, sources
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
