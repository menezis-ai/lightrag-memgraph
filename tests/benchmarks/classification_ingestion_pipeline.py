"""Benchmark classification-gate I/O offload and DocStatus batch enrichment.

Run standalone with::

    uv run python tests/benchmarks/classification_ingestion_pipeline.py

The in-process baselines preserve the pre-optimization behavior:

* classification probes execute synchronously on the event-loop thread;
* accepted documents are read and upserted one at a time.
"""

from __future__ import annotations

import asyncio
import copy
import statistics
import threading
import time
import tracemalloc
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from twindb_lightrag_memgraph import _classification_hook as hook_mod

ITERATIONS = 60
DOCUMENT_COUNT = 24
DB_RTT_SECONDS = 0.0015
CLASSIFY_DELAY_SECONDS = 0.002
READ_CAPACITY = 20
WRITE_CAPACITY = 8
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 24

_read_slots = asyncio.Semaphore(READ_CAPACITY)
_write_slots = asyncio.Semaphore(WRITE_CAPACITY)
_probe_context: ContextVar[str] = ContextVar(
    "classification_benchmark_context", default="missing"
)


@dataclass
class _MetadataState:
    docs: dict[str, dict[str, Any]]
    read_queries: int = 0
    write_queries: int = 0
    upsert_calls: int = 0


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    peak_mb: float
    mean_reads: float = 0.0
    mean_writes: float = 0.0


class _FakeDocStatus:
    def __init__(self, state: _MetadataState) -> None:
        self._state = state

    async def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        self._state.read_queries += 1
        async with _read_slots:
            await asyncio.sleep(DB_RTT_SECONDS)
        row = self._state.docs.get(doc_id)
        return copy.deepcopy(row) if row is not None else None

    async def get_by_ids(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        self._state.read_queries += 1
        async with _read_slots:
            await asyncio.sleep(DB_RTT_SECONDS)
        return [
            copy.deepcopy(self._state.docs[doc_id])
            for doc_id in doc_ids
            if doc_id in self._state.docs
        ]

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # MemgraphDocStatusStorage.upsert persists node properties and folder
        # membership with two write queries.
        self._state.upsert_calls += 1
        self._state.write_queries += 2
        async with _write_slots:
            await asyncio.sleep(DB_RTT_SECONDS * 2)
        for doc_id, row in data.items():
            self._state.docs[doc_id] = copy.deepcopy(row)


class _FakeRag:
    def __init__(self, state: _MetadataState) -> None:
        self.doc_status = _FakeDocStatus(state)


def _accepted() -> list[tuple[int, dict[str, Any]]]:
    return [
        (
            index,
            {
                "class_id": "C2",
                "source_format": "ooxml",
                "label_guid": f"guid-{index:04d}",
            },
        )
        for index in range(DOCUMENT_COUNT)
    ]


def _inputs() -> list[str]:
    return [f"document body {index}" for index in range(DOCUMENT_COUNT)]


def _ids() -> list[str]:
    return [f"doc-{index:04d}" for index in range(DOCUMENT_COUNT)]


def _initial_docs() -> dict[str, dict[str, Any]]:
    return {
        doc_id: {
            "id": doc_id,
            "status": "processed",
            "file_path": f"{doc_id}.docx",
            "metadata": {"existing": index},
        }
        for index, doc_id in enumerate(_ids())
    }


def _assert_metadata_parity(state: _MetadataState) -> None:
    assert len(state.docs) == DOCUMENT_COUNT
    for index, doc_id in enumerate(_ids()):
        metadata = state.docs[doc_id]["metadata"]
        assert metadata["existing"] == index
        assert metadata["classification"] == _accepted()[index][1]


async def _baseline_metadata_request(rag: _FakeRag) -> None:
    """Pre-optimization N × read/upsert loop."""
    for index, payload in _accepted():
        await hook_mod._merge_classification_metadata(rag, _ids()[index], payload)


async def _live_metadata_request(rag: _FakeRag) -> None:
    await hook_mod._apply_classification_metadata(
        rag,
        _accepted(),
        _inputs(),
        _ids(),
        True,
        paths=[f"doc-{index:04d}.docx" for index in range(DOCUMENT_COUNT)],
    )


async def _one_metadata_request(
    request: Callable[[_FakeRag], Awaitable[None]],
) -> tuple[float, _MetadataState]:
    state = _MetadataState(docs=_initial_docs())
    started = time.perf_counter()
    await request(_FakeRag(state))
    elapsed_ms = (time.perf_counter() - started) * 1000
    _assert_metadata_parity(state)
    return elapsed_ms, state


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index]


async def _time_metadata(
    request: Callable[[_FakeRag], Awaitable[None]],
    *,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    durations: list[float] = []
    reads: list[int] = []
    writes: list[int] = []
    tracemalloc.start()
    started = time.perf_counter()

    async def worker() -> None:
        elapsed_ms, state = await _one_metadata_request(request)
        durations.append(elapsed_ms)
        reads.append(state.read_queries)
        writes.append(state.write_queries)

    for offset in range(0, iterations, concurrency):
        batch_size = min(concurrency, iterations - offset)
        await asyncio.gather(*(worker() for _ in range(batch_size)))

    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _Sample(
        mean_ms=statistics.fmean(durations),
        p95_ms=_percentile(durations, 0.95),
        p99_ms=_percentile(durations, 0.99),
        requests_per_second=iterations / elapsed,
        peak_mb=peak_bytes / 1024 / 1024,
        mean_reads=statistics.fmean(reads),
        mean_writes=statistics.fmean(writes),
    )


def _blocking_probe(
    thread_ids: set[int],
    lock: threading.Lock,
) -> Callable[[str], dict[str, Any]]:
    def probe(path: str) -> dict[str, Any]:
        assert _probe_context.get() == "copied"
        with lock:
            thread_ids.add(threading.get_ident())
        time.sleep(CLASSIFY_DELAY_SECONDS)
        return {"class_id": "C2", "path": path}

    setattr(probe, "_twin_classification_probe", probe)
    setattr(probe, "_twin_classification_evaluate", lambda _path, result: result)
    return probe


async def _baseline_partition(
    active_hook: Callable[[str], dict[str, Any]],
    paths: list[str],
) -> Any:
    return hook_mod._partition_inputs(active_hook, paths)


async def _live_partition(
    active_hook: Callable[[str], dict[str, Any]],
    paths: list[str],
) -> Any:
    async_partition = getattr(hook_mod, "_partition_inputs_async", None)
    if async_partition is None:
        return hook_mod._partition_inputs(active_hook, paths)
    return await async_partition(active_hook, paths)


async def _time_event_loop_responsiveness(
    partition: Callable[[Callable[[str], dict[str, Any]], list[str]], Awaitable[Any]],
    *,
    request_count: int,
) -> tuple[_Sample, set[int]]:
    heartbeat_durations: list[float] = []
    thread_ids: set[int] = set()
    lock = threading.Lock()
    active_hook = _blocking_probe(thread_ids, lock)
    paths = ["first.docx", "second.docx", "third.docx"]
    token = _probe_context.set("copied")
    tracemalloc.start()
    started = time.perf_counter()

    async def heartbeat() -> None:
        heartbeat_started = time.perf_counter()
        await asyncio.sleep(0.001)
        heartbeat_durations.append((time.perf_counter() - heartbeat_started) * 1000)

    async def classify() -> None:
        accepted, rejected = await partition(active_hook, paths)
        assert len(accepted) == len(paths)
        assert rejected == []

    tasks: list[asyncio.Task[Any]] = []
    try:
        # Schedule each heartbeat before its classification task so the timer
        # is already pending when synchronous probe I/O blocks the loop.
        for _ in range(request_count):
            tasks.append(asyncio.create_task(heartbeat()))
            tasks.append(asyncio.create_task(classify()))
        await asyncio.gather(*tasks)
    finally:
        _probe_context.reset(token)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (
        _Sample(
            mean_ms=statistics.fmean(heartbeat_durations),
            p95_ms=_percentile(heartbeat_durations, 0.95),
            p99_ms=_percentile(heartbeat_durations, 0.99),
            requests_per_second=request_count / elapsed,
            peak_mb=peak_bytes / 1024 / 1024,
        ),
        thread_ids,
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: latency ratios plus deterministic structural guards."""
    count = iterations or ITERATIONS
    _, baseline_state = await _one_metadata_request(_baseline_metadata_request)
    _, live_state = await _one_metadata_request(_live_metadata_request)
    baseline_metadata = await _time_metadata(
        _baseline_metadata_request, iterations=count
    )
    optimized_metadata = await _time_metadata(_live_metadata_request, iterations=count)
    main_thread = threading.get_ident()
    baseline_loop, _ = await _time_event_loop_responsiveness(
        _baseline_partition, request_count=8
    )
    optimized_loop, optimized_threads = await _time_event_loop_responsiveness(
        _live_partition, request_count=8
    )
    return [
        {
            "name": "classification DocStatus batch enrichment latency",
            "kind": "ratio",
            "baseline_ms": baseline_metadata.mean_ms,
            "optimized_ms": optimized_metadata.mean_ms,
            "detail": (
                f"p95 {baseline_metadata.p95_ms:.3f}->"
                f"{optimized_metadata.p95_ms:.3f}ms"
            ),
        },
        {
            "name": "classification DocStatus enrichment uses one batch",
            "kind": "structural",
            "passed": (
                baseline_state.read_queries == DOCUMENT_COUNT
                and baseline_state.write_queries == DOCUMENT_COUNT * 2
                and live_state.read_queries == 1
                and live_state.write_queries == 2
                and live_state.upsert_calls == 1
            ),
            "detail": (
                f"baseline reads/writes={baseline_state.read_queries}/"
                f"{baseline_state.write_queries}; live reads/writes="
                f"{live_state.read_queries}/{live_state.write_queries}, "
                f"upserts={live_state.upsert_calls}"
            ),
        },
        {
            "name": "classification probe event-loop heartbeat latency",
            "kind": "ratio",
            "baseline_ms": baseline_loop.mean_ms,
            "optimized_ms": optimized_loop.mean_ms,
            "detail": (
                f"p95 {baseline_loop.p95_ms:.3f}->{optimized_loop.p95_ms:.3f}ms"
            ),
        },
        {
            "name": "classification probes leave the event-loop thread",
            "kind": "structural",
            "passed": bool(optimized_threads - {main_thread}),
            "detail": (
                f"main thread={main_thread}; probe threads="
                f"{sorted(optimized_threads)}"
            ),
        },
    ]


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"peak={sample.peak_mb:.3f}MB "
        f"reads={sample.mean_reads:.1f} writes={sample.mean_writes:.1f}"
    )


async def main() -> None:
    baseline = await _time_metadata(_baseline_metadata_request, iterations=ITERATIONS)
    optimized = await _time_metadata(_live_metadata_request, iterations=ITERATIONS)
    baseline_sustained = await _time_metadata(
        _baseline_metadata_request,
        iterations=ITERATIONS,
        concurrency=SUSTAINED_CONCURRENCY,
    )
    optimized_sustained = await _time_metadata(
        _live_metadata_request,
        iterations=ITERATIONS,
        concurrency=SUSTAINED_CONCURRENCY,
    )
    baseline_peak = await _time_metadata(
        _baseline_metadata_request,
        iterations=ITERATIONS,
        concurrency=PEAK_CONCURRENCY,
    )
    optimized_peak = await _time_metadata(
        _live_metadata_request,
        iterations=ITERATIONS,
        concurrency=PEAK_CONCURRENCY,
    )
    baseline_loop, _ = await _time_event_loop_responsiveness(
        _baseline_partition, request_count=8
    )
    optimized_loop, optimized_threads = await _time_event_loop_responsiveness(
        _live_partition, request_count=8
    )
    _print_sample("metadata_baseline", baseline)
    _print_sample("metadata_optimized", optimized)
    _print_sample("metadata_baseline_sustained", baseline_sustained)
    _print_sample("metadata_optimized_sustained", optimized_sustained)
    _print_sample("metadata_baseline_peak", baseline_peak)
    _print_sample("metadata_optimized_peak", optimized_peak)
    _print_sample("event_loop_baseline", baseline_loop)
    _print_sample("event_loop_optimized", optimized_loop)
    print(f"event_loop_probe_threads={sorted(optimized_threads)}")


if __name__ == "__main__":
    asyncio.run(main())
