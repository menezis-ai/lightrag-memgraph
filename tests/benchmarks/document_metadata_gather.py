"""Bolt micro-benchmark: overlap the three independent reads behind
``GET /documents/{doc_id}/metadata`` (``routes_documents._doc_tags_and_source_links``).

The document detail panel calls this route every time a document is opened.
Its body needs the DocStatus record and its folder membership (the
request-critical read: it decides the 404), the ``TAGGED_WITH`` tag lookup and
the source-link list — three reads that share nothing but ``doc_id``. Before
the optimization they ran one after another: four sequential round-trips
(``get_by_id`` → ``get_folders_for_doc`` → graph tags → source links). Now the
two optional reads start alongside the critical one, and are cancelled and
reaped if it raises.

The harness drives the REAL route function against a fake RAG / fake read
helpers that sleep a fixed RTT per read through a capacity-bounded pool
(``READ_CAPACITY`` models ``MEMGRAPH_READ_POOL_SIZE``, default 20). "before" is
reproduced by binding the module's ``asyncio`` proxy so ``ensure_future`` is a
lazy holder — the same function body, only the overlap toggled off — which is
also what the structural cases detect.

Run standalone with::

    uv run python tests/benchmarks/document_metadata_gather.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import folder as folder_mod
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.webui import routes_documents

ITERATIONS = 80
RTT_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))
FOLDER = "f-bench"
DOC_ID = "doc-bench"


class _ReadMeter:
    """Count reads, track concurrency, and record intervals for depth."""

    def __init__(self, capacity: int) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.reads = 0
        self.intervals: list[tuple[float, float]] = []
        self._capacity = asyncio.Semaphore(capacity)

    async def read(self, value: Any) -> Any:
        async with self._capacity:  # the Bolt driver blocks on an exhausted pool
            self.reads += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            started = time.perf_counter()
            try:
                await asyncio.sleep(RTT_SECONDS)
            finally:
                self.in_flight -= 1
                self.intervals.append((started, time.perf_counter()))
        return value

    def round_trip_depth(self) -> int:
        """Longest chain of strictly non-overlapping reads (load-independent)."""
        depth = 0
        chain_end = float("-inf")
        for start, end in sorted(self.intervals, key=lambda iv: iv[1]):
            if start >= chain_end:
                depth += 1
                chain_end = end
        return depth

    def reset(self) -> None:
        self.reads = 0
        self.max_in_flight = 0
        self.intervals = []


_DOC = {
    "id": DOC_ID,
    "file_path": "/kb/bench.md",
    "metadata": {
        "folder": FOLDER,
        "review": {"state": "approved", "actor": "steward"},
        "classification": {"class_id": "C2"},
    },
}


class _DocStatus:
    def __init__(self, meter: _ReadMeter) -> None:
        self._meter = meter

    async def get_by_id(self, doc_id: str):
        return await self._meter.read(dict(_DOC) if doc_id == DOC_ID else None)

    async def get_folders_for_doc(self, _doc_id: str) -> list[str]:
        return await self._meter.read([FOLDER])


class _Rag:
    def __init__(self, meter: _ReadMeter) -> None:
        self.doc_status = _DocStatus(meter)


class _SequentialAsyncio:
    """``asyncio`` proxy whose ``ensure_future`` is a lazy holder.

    Bound onto ``routes_documents.asyncio`` only — rebinding the real module
    would serialize this harness's own concurrent load driver. The holder does
    not start its coroutine until awaited, which is exactly the
    pre-optimization ordering (each read after the previous one completed).
    """

    class _Deferred:
        def __init__(self, coro):
            self._coro = coro
            self._result = None
            self._done = False

        def __await__(self):
            return self._run().__await__()

        async def _run(self):
            if not self._done:
                self._result = await self._coro
                self._done = True
            return self._result

        def cancel(self):
            if not self._done:
                self._coro.close()
                self._done = True

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def ensure_future(self, coro):
        return self._Deferred(coro)


@asynccontextmanager
async def _bound(sequential: bool, meter: _ReadMeter):
    """Bind the folder, the fake RAG and the fake read helpers."""

    async def _graph_tags(_doc_id: str) -> list[str] | None:
        return await meter.read(["oracle", "rman"])

    async def _source_links(_doc_id: str) -> list[dict[str, Any]]:
        return await meter.read([{"doc_id": DOC_ID, "url": "https://x.invalid/1"}])

    token = folder_mod._active_folder_id.set(FOLDER)
    previous_rag = _twindb_state.get("rag")
    original_tags = webui_router._graph_tags_for_doc_or_none
    original_links = routes_documents._source_links_for_document
    original_asyncio = routes_documents.asyncio
    _twindb_state["rag"] = _Rag(meter)
    webui_router._graph_tags_for_doc_or_none = _graph_tags
    routes_documents._source_links_for_document = _source_links
    if sequential:
        routes_documents.asyncio = _SequentialAsyncio(original_asyncio)
    try:
        yield
    finally:
        routes_documents.asyncio = original_asyncio
        routes_documents._source_links_for_document = original_links
        webui_router._graph_tags_for_doc_or_none = original_tags
        if previous_rag is None:
            _twindb_state.pop("rag", None)
        else:
            _twindb_state["rag"] = previous_rag
        folder_mod._active_folder_id.reset(token)


async def _run_once() -> dict[str, Any]:
    return await routes_documents.get_document_metadata(DOC_ID)


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    max_in_flight: int


async def _time_requests(
    *, sequential: bool, iterations: int, concurrency: int = 1
) -> _Sample:
    meter = _ReadMeter(READ_CAPACITY)
    durations: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with _bound(sequential, meter):
        for _ in range(3):  # warmup, outside the measured window
            await _run_once()

        async def run_one() -> None:
            async with semaphore:
                started = time.perf_counter()
                await _run_once()
                durations.append((time.perf_counter() - started) * 1000)

        meter.reset()
        started = time.perf_counter()
        await asyncio.gather(*(run_one() for _ in range(iterations)))
        elapsed = time.perf_counter() - started

    ordered = sorted(durations)

    def percentile(percent: float) -> float:
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percent) - 1))
        return ordered[index]

    return _Sample(
        mean_ms=statistics.mean(durations),
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        requests_per_second=iterations / elapsed,
        max_in_flight=meter.max_in_flight,
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS

    baseline_meter = _ReadMeter(READ_CAPACITY)
    async with _bound(True, baseline_meter):
        baseline_body = await _run_once()
    live_meter = _ReadMeter(READ_CAPACITY)
    async with _bound(False, live_meter):
        live_body = await _run_once()

    baseline = await _time_requests(sequential=True, iterations=count)
    optimized = await _time_requests(sequential=False, iterations=count)

    return [
        {
            "name": "document metadata route latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "the three reads overlap with an identical read count",
            "kind": "structural",
            "passed": (
                live_meter.max_in_flight == 3
                and baseline_meter.max_in_flight == 1
                and live_meter.reads == baseline_meter.reads == 4
            ),
            "detail": (
                "expected 3 concurrent reads on the live path (serial baseline 1) "
                "and 4 reads on both; observed in-flight "
                f"live={live_meter.max_in_flight} baseline={baseline_meter.max_in_flight}, "
                f"reads live={live_meter.reads} baseline={baseline_meter.reads}"
            ),
        },
        {
            "name": "round-trip depth reduced",
            "kind": "structural",
            "passed": live_meter.round_trip_depth() < baseline_meter.round_trip_depth(),
            "detail": (
                "expected fewer sequential round-trips than the serial body; "
                f"observed depth {baseline_meter.round_trip_depth()} -> "
                f"{live_meter.round_trip_depth()}"
            ),
        },
        {
            "name": "route payload unchanged",
            "kind": "structural",
            "passed": live_body == baseline_body and live_body["tags_status"] == "ok",
            "detail": f"expected identical payloads; observed {live_body == baseline_body}",
        },
    ]


def _pool(samples: list[_Sample]) -> _Sample:
    return _Sample(
        mean_ms=statistics.mean(s.mean_ms for s in samples),
        p50_ms=statistics.mean(s.p50_ms for s in samples),
        p95_ms=statistics.mean(s.p95_ms for s in samples),
        p99_ms=statistics.mean(s.p99_ms for s in samples),
        requests_per_second=statistics.mean(s.requests_per_second for s in samples),
        max_in_flight=max(s.max_in_flight for s in samples),
    )


async def _load_test(
    *, iterations: int, concurrency: int, rounds: int = 5
) -> tuple[_Sample, _Sample]:
    baselines: list[_Sample] = []
    optimizeds: list[_Sample] = []
    for _ in range(rounds):
        baselines.append(
            await _time_requests(
                sequential=True, iterations=iterations, concurrency=concurrency
            )
        )
        optimizeds.append(
            await _time_requests(
                sequential=False, iterations=iterations, concurrency=concurrency
            )
        )
    return _pool(baselines), _pool(optimizeds)


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms p50={sample.p50_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"max_reads_in_flight={sample.max_in_flight}"
    )


async def main() -> None:
    cases = await measure()
    ratio = cases[0]
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"sequential: {ratio['baseline_ms']:.3f}ms -> "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster) [{ratio['detail']}]"
    )
    for case in cases[1:]:
        print(f"structural [{case['name']}]: {case['passed']} ({case['detail']})")

    for label, concurrency, requests in (("sustained", 8, 80), ("peak", 20, 160)):
        baseline, optimized = await _load_test(
            iterations=requests, concurrency=concurrency
        )
        _print_sample(f"{label} baseline", baseline)
        _print_sample(f"{label} optimized", optimized)


if __name__ == "__main__":
    asyncio.run(main())
