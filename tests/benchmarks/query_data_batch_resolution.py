"""Benchmark: batch doc-id resolution for /query/data tag filtering.

Run as a script:
``python tests/benchmarks/query_data_batch_resolution.py``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.query import query_data_filters as qdf
from twindb_lightrag_memgraph.server.query.source_filters import _tag_filter_terms

ITERATIONS = 80
SMALL_ROW_COUNT = 4
SERIAL_GUARD_ROW_COUNT = qdf._PARALLEL_RESOLVE_ID_BUDGET // 2 + 1
LOOKUP_DELAY_SECONDS = 0.004
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 32


class _Rag:
    doc_status = object()


async def _fake_chunk_batch(_rag: Any, chunk_ids: set[str]) -> dict[str, str]:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return {chunk_id: f"doc:{chunk_id}" for chunk_id in chunk_ids}


async def _fake_file_batch(_rag: Any, file_paths: set[str]) -> dict[str, str]:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return {file_path: f"doc:{file_path}" for file_path in file_paths}


async def _fetch_doc_tags(doc_id: str, _folder: str) -> set[str]:
    return {"keep"}


def _rows(row_count: int) -> list[dict[str, Any]]:
    return [
        {
            "source_id": f"chunk-{idx}",
            "file_path": f"file-{idx}.pdf",
            "reference_id": f"ref-{idx}",
        }
        for idx in range(row_count)
    ]


# Pre-7ff89ff body kept here as a stable before/after comparator.
async def _baseline_filter_rows_by_tags(
    rag: Any,
    rows: list,
    tag_filter,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> tuple[list, set[str]]:
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return rows, set()

    rows_for_filter: list[dict[str, Any]] = []
    row_direct_doc_ids: list[set[str]] = []
    row_chunk_ids: list[set[str]] = []
    row_file_paths: list[set[str]] = []

    chunk_ids: set[str] = set()
    file_paths: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        direct_doc_ids, row_chunks, row_paths = qdf._query_data_row_doc_candidates(row)
        rows_for_filter.append(row)
        row_direct_doc_ids.append(direct_doc_ids)
        row_chunk_ids.append(row_chunks)
        row_file_paths.append(row_paths)
        chunk_ids.update(row_chunks)
        file_paths.update(row_paths)

    if not rows_for_filter:
        return [], set()

    chunk_docs = await qdf._resolve_doc_ids_for_chunk_ids(rag, chunk_ids)
    file_docs = await qdf._resolve_doc_ids_for_file_paths(rag, file_paths)

    row_doc_ids, all_doc_ids = qdf._combine_row_doc_ids(
        row_direct_doc_ids,
        row_chunk_ids,
        row_file_paths,
        chunk_docs,
        file_docs,
    )

    unresolved_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in tags_cache]
    if unresolved_doc_ids:
        resolved_tags = await asyncio.gather(
            *(fetch_doc_tags(doc_id, folder) for doc_id in unresolved_doc_ids)
        )
        tags_cache.update(zip(unresolved_doc_ids, resolved_tags))

    kept_rows = []
    kept_reference_ids: set[str] = set()
    for row, doc_ids in zip(rows_for_filter, row_doc_ids):
        if not doc_ids:
            continue
        if not any(
            qdf._doc_tags_match_filter(tags_cache[doc_id], tag_filter)
            for doc_id in doc_ids
        ):
            continue
        kept_rows.append(row)
        ref_id = row.get("reference_id")
        if isinstance(ref_id, str) and ref_id:
            kept_reference_ids.add(ref_id)
    return kept_rows, kept_reference_ids


async def _one_request(fn, rows: list[dict[str, Any]]) -> int:
    kept_rows, kept_refs = await fn(
        _Rag(),
        rows,
        {"all": ["keep"]},
        "default",
        {},
        _fetch_doc_tags,
    )
    assert len(kept_rows) == len(rows)
    assert len(kept_refs) == len(rows)
    return len(kept_rows)


async def _measure_isolated(
    label: str, fn, rows: list[dict[str, Any]], iterations: int = ITERATIONS
) -> dict[str, Any]:
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(iterations):
        start = time.perf_counter()
        await _one_request(fn, rows)
        durations_ms.append((time.perf_counter() - start) * 1000)
    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "iterations": iterations,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "p99_ms": statistics.quantiles(durations_ms, n=100)[98],
        "req_per_s": iterations / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


async def _measure_load(
    label: str,
    fn,
    rows: list[dict[str, Any]],
    concurrency: int,
) -> dict[str, Any]:
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()

    async def worker() -> None:
        start = time.perf_counter()
        await _one_request(fn, rows)
        durations_ms.append((time.perf_counter() - start) * 1000)

    for _ in range(ITERATIONS // concurrency):
        await asyncio.gather(*(worker() for _ in range(concurrency)))

    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "concurrency": concurrency,
        "requests": len(durations_ms),
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "p99_ms": statistics.quantiles(durations_ms, n=100)[98],
        "req_per_s": len(durations_ms) / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


def _print_result(result: dict[str, Any]) -> None:
    print(
        f"{result['label']}: mean={result['mean_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms p99={result['p99_ms']:.3f}ms "
        f"throughput={result['req_per_s']:.1f} req/s peak={result['peak_mb']:.3f}MB"
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: the ratio case for the small mixed-row parallel path.

    Returns the shared contract shape (see ``tests/benchmarks/_perf_contract``).
    The structural guard — that the two resolver groups actually overlap — is
    already covered by ``tests/test_query_data_filters.py``
    (``test_filter_rows_by_tags_resolves_chunk_and_file_batches_concurrently``),
    so this benchmark contributes the ratio case only and does not duplicate it.

    Patches the module-level chunk/file batch resolvers to sleep-only fakes and
    restores them, so calling this from the pytest process leaves no global
    residue (the CLI ``main()`` deliberately keeps its patch — the process exits
    right after).
    """
    n = iterations or ITERATIONS
    orig_chunk = qdf._resolve_doc_ids_for_chunk_ids
    orig_file = qdf._resolve_doc_ids_for_file_paths
    qdf._resolve_doc_ids_for_chunk_ids = _fake_chunk_batch
    qdf._resolve_doc_ids_for_file_paths = _fake_file_batch
    try:
        rows = _rows(SMALL_ROW_COUNT)
        baseline = await _measure_isolated(
            "baseline", _baseline_filter_rows_by_tags, rows, iterations=n
        )
        optimized = await _measure_isolated(
            "optimized", qdf._filter_rows_by_tags, rows, iterations=n
        )
    finally:
        qdf._resolve_doc_ids_for_chunk_ids = orig_chunk
        qdf._resolve_doc_ids_for_file_paths = orig_file
    return [
        {
            "name": (
                "query_data_filters._filter_rows_by_tags "
                "(small mixed-row parallel resolve)"
            ),
            "kind": "ratio",
            "baseline_ms": baseline["mean_ms"],
            "optimized_ms": optimized["mean_ms"],
        }
    ]


async def main() -> None:
    qdf._resolve_doc_ids_for_chunk_ids = _fake_chunk_batch
    qdf._resolve_doc_ids_for_file_paths = _fake_file_batch
    small_rows = _rows(SMALL_ROW_COUNT)
    serial_guard_rows = _rows(SERIAL_GUARD_ROW_COUNT)

    baseline = await _measure_isolated(
        "small_baseline_serial_batches",
        _baseline_filter_rows_by_tags,
        small_rows,
    )
    optimized = await _measure_isolated(
        "small_optimized_parallel_batches",
        qdf._filter_rows_by_tags,
        small_rows,
    )
    baseline_sustained = await _measure_load(
        "small_baseline_sustained_load",
        _baseline_filter_rows_by_tags,
        small_rows,
        SUSTAINED_CONCURRENCY,
    )
    optimized_sustained = await _measure_load(
        "small_optimized_sustained_load",
        qdf._filter_rows_by_tags,
        small_rows,
        SUSTAINED_CONCURRENCY,
    )
    baseline_peak = await _measure_load(
        "small_baseline_peak_load",
        _baseline_filter_rows_by_tags,
        small_rows,
        PEAK_CONCURRENCY,
    )
    optimized_peak = await _measure_load(
        "small_optimized_peak_load",
        qdf._filter_rows_by_tags,
        small_rows,
        PEAK_CONCURRENCY,
    )
    serial_guard_baseline = await _measure_isolated(
        "large_baseline_serial_batches",
        _baseline_filter_rows_by_tags,
        serial_guard_rows,
    )
    serial_guard_optimized = await _measure_isolated(
        "large_optimized_serial_guard",
        qdf._filter_rows_by_tags,
        serial_guard_rows,
    )

    for result in (
        baseline,
        optimized,
        baseline_sustained,
        optimized_sustained,
        baseline_peak,
        optimized_peak,
        serial_guard_baseline,
        serial_guard_optimized,
    ):
        _print_result(result)

    latency_gain = (
        (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    )
    throughput_gain = (
        (optimized["req_per_s"] - baseline["req_per_s"]) / baseline["req_per_s"] * 100
    )
    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> "
        f"{optimized['mean_ms']:.3f}ms ({latency_gain:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['req_per_s']:.1f} req/s -> "
        f"{optimized['req_per_s']:.1f} req/s ({throughput_gain:.1f}%)"
    )
    print(f"peak_mem: {baseline['peak_mb']:.3f}MB -> {optimized['peak_mb']:.3f}MB")
    print()
    print("## SERIAL GUARD")
    print(
        f"row_count={SERIAL_GUARD_ROW_COUNT} "
        f"ids_per_request={SERIAL_GUARD_ROW_COUNT * 2} "
        f"budget={qdf._PARALLEL_RESOLVE_ID_BUDGET}"
    )
    print(
        "large guarded path: "
        f"{serial_guard_baseline['mean_ms']:.3f}ms baseline -> "
        f"{serial_guard_optimized['mean_ms']:.3f}ms optimized"
    )


if __name__ == "__main__":
    asyncio.run(main())
