"""Composite load benchmark for the ingestion/catalog performance epic.

One representative cycle performs the three optimized stages in order:
classification probe, accepted DocStatus enrichment, then a tag-catalog
refresh. The baseline functions are faithful copies of the pre-optimization
paths and run against the same bounded fake downstream capacities.

Run standalone with::

    uv run python tests/benchmarks/ingestion_catalog_epic.py
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

try:
    from tests.benchmarks import classification_ingestion_pipeline as classification
    from tests.benchmarks import tag_catalog_query_batch as tag_catalog
except ModuleNotFoundError:  # Standalone execution from tests/benchmarks.
    import classification_ingestion_pipeline as classification
    import tag_catalog_query_batch as tag_catalog

from twindb_lightrag_memgraph.server import webui_tagstore as store_mod

ITERATIONS = 48
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 24


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    peak_mb: float


@dataclass
class _Structure:
    metadata_reads: int
    metadata_writes: int
    metadata_upserts: int
    tag_reads: int
    probe_threads: set[int]


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index]


async def _one_cycle(
    *,
    partition: Callable[..., Awaitable[Any]],
    metadata_request: Callable[..., Awaitable[None]],
    tag_request: Callable[..., Awaitable[list[dict[str, Any]]]],
    active_hook: Callable[[str], dict[str, Any]],
    probe_threads: set[int],
) -> tuple[float, _Structure]:
    started = time.perf_counter()
    accepted, rejected = await partition(
        active_hook, ["first.docx", "second.docx", "third.docx"]
    )
    assert len(accepted) == 3
    assert rejected == []
    _, metadata_state = await classification._one_metadata_request(metadata_request)
    _, tag_state = await tag_catalog._one_request(tag_request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return (
        elapsed_ms,
        _Structure(
            metadata_reads=metadata_state.read_queries,
            metadata_writes=metadata_state.write_queries,
            metadata_upserts=metadata_state.upsert_calls,
            tag_reads=tag_state.query_count,
            probe_threads=set(probe_threads),
        ),
    )


async def _time_cycles(
    *,
    partition: Callable[..., Awaitable[Any]],
    metadata_request: Callable[..., Awaitable[None]],
    tag_request: Callable[..., Awaitable[list[dict[str, Any]]]],
    iterations: int,
    concurrency: int,
) -> tuple[_Sample, _Structure]:
    durations: list[float] = []
    structures: list[_Structure] = []
    probe_threads: set[int] = set()
    probe_lock = threading.Lock()
    active_hook = classification._blocking_probe(probe_threads, probe_lock)
    context_token = classification._probe_context.set("copied")
    tracemalloc.start()
    started = time.perf_counter()

    async def worker() -> None:
        elapsed_ms, structure = await _one_cycle(
            partition=partition,
            metadata_request=metadata_request,
            tag_request=tag_request,
            active_hook=active_hook,
            probe_threads=probe_threads,
        )
        durations.append(elapsed_ms)
        structures.append(structure)

    try:
        for offset in range(0, iterations, concurrency):
            batch_size = min(concurrency, iterations - offset)
            await asyncio.gather(*(worker() for _ in range(batch_size)))
    finally:
        classification._probe_context.reset(context_token)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latest = structures[-1]
    latest.probe_threads = set(probe_threads)
    return (
        _Sample(
            mean_ms=statistics.fmean(durations),
            p95_ms=_percentile(durations, 0.95),
            p99_ms=_percentile(durations, 0.99),
            requests_per_second=iterations / elapsed,
            peak_mb=peak_bytes / 1024 / 1024,
        ),
        latest,
    )


async def _measure_pair(
    *,
    iterations: int,
    concurrency: int,
) -> tuple[_Sample, _Sample, _Structure, _Structure]:
    original_session = store_mod._pool.get_read_session
    store_mod._pool.get_read_session = tag_catalog._read_session
    try:
        baseline, baseline_structure = await _time_cycles(
            partition=classification._baseline_partition,
            metadata_request=classification._baseline_metadata_request,
            tag_request=tag_catalog._baseline_list_tags,
            iterations=iterations,
            concurrency=concurrency,
        )
        optimized, optimized_structure = await _time_cycles(
            partition=classification._live_partition,
            metadata_request=classification._live_metadata_request,
            tag_request=tag_catalog._live_list_tags,
            iterations=iterations,
            concurrency=concurrency,
        )
    finally:
        store_mod._pool.get_read_session = original_session
    return baseline, optimized, baseline_structure, optimized_structure


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: mixed-load ratio plus end-to-end structural guard."""
    count = iterations or ITERATIONS
    baseline, optimized, baseline_structure, optimized_structure = await _measure_pair(
        iterations=count,
        concurrency=min(SUSTAINED_CONCURRENCY, count),
    )
    main_thread = threading.get_ident()
    return [
        {
            "name": "ingestion/catalog epic sustained-load latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "ingestion/catalog epic optimized structure",
            "kind": "structural",
            "passed": (
                baseline_structure.metadata_reads == classification.DOCUMENT_COUNT
                and baseline_structure.metadata_writes
                == classification.DOCUMENT_COUNT * 2
                and baseline_structure.tag_reads == 2
                and optimized_structure.metadata_reads == 1
                and optimized_structure.metadata_writes == 2
                and optimized_structure.metadata_upserts == 1
                and optimized_structure.tag_reads == 1
                and bool(optimized_structure.probe_threads - {main_thread})
            ),
            "detail": (
                "baseline metadata reads/writes + tag reads="
                f"{baseline_structure.metadata_reads}/"
                f"{baseline_structure.metadata_writes} + "
                f"{baseline_structure.tag_reads}; optimized="
                f"{optimized_structure.metadata_reads}/"
                f"{optimized_structure.metadata_writes} + "
                f"{optimized_structure.tag_reads}; probe threads="
                f"{sorted(optimized_structure.probe_threads)}"
            ),
        },
    ]


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"peak={sample.peak_mb:.3f}MB"
    )


async def main() -> None:
    isolated = await _measure_pair(iterations=ITERATIONS, concurrency=1)
    sustained = await _measure_pair(
        iterations=ITERATIONS,
        concurrency=SUSTAINED_CONCURRENCY,
    )
    peak = await _measure_pair(
        iterations=ITERATIONS,
        concurrency=PEAK_CONCURRENCY,
    )
    _print_sample("epic_baseline", isolated[0])
    _print_sample("epic_optimized", isolated[1])
    _print_sample("epic_baseline_sustained", sustained[0])
    _print_sample("epic_optimized_sustained", sustained[1])
    _print_sample("epic_baseline_peak", peak[0])
    _print_sample("epic_optimized_peak", peak[1])


if __name__ == "__main__":
    asyncio.run(main())
