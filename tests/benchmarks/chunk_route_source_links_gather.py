"""Bolt micro-benchmark: overlap the chunk fetch and the source-link read in the
chunk expansion routes.

``GET /chunks/{id}/context``, ``GET /chunks/{id}/document`` and
``GET /documents/{id}/chunks`` each loaded the chunk records and then the
document's source links — two independent reads against different stores,
neither consuming the other's result.

Scope of the measurement (deliberately explicit — an earlier version drove only
``/documents/{doc_id}/chunks`` while quoting its numbers for all three):

* **All three routes are driven and timed separately.** The two ``/chunks/...``
  routes additionally pay the anchor read that resolves the chunk to its parent
  document, so their proportional gain is necessarily smaller; reporting one
  number for all three would have overstated them.
* **Every read that would draw a pool connection is metered**, not just the two
  being overlapped: the anchor read, the DocStatus record, the folder-membership
  lookup, the chunk batch and the source-link read all pass through the same
  capacity semaphore. Leaving the authorization reads outside it modelled a pool
  that was emptier than the real one and flattered the optimized arm.

"before" is reproduced by monkeypatching the module's ``asyncio`` proxy so
``gather`` runs serially — same route body, only the overlap toggled off.

Run standalone with::

    uv run python tests/benchmarks/chunk_route_source_links_gather.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from twindb_lightrag_memgraph.server import chunk_routes

ITERATIONS = 80
CHUNK_COUNT = 40
READ_DELAY_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
SOURCE_LINK_COUNT = 3
# Every metered read draws from the same Bolt read pool
# (``MEMGRAPH_READ_POOL_SIZE``, default 20). Without this cap the load test
# models a fan-out the deployment cannot issue.
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))


class _ReadMeter:
    """Capacity-bounded counter for every read a request would issue."""

    def __init__(self, capacity: int = READ_CAPACITY) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.reads = 0
        self.overlapped_max = 0
        self._overlapped = 0
        self._capacity = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def track(self, *, overlapped: bool = False):
        # The Bolt driver blocks on an exhausted pool; model that back-pressure
        # rather than pretending concurrent read sessions are free.
        async with self._capacity:
            self.reads += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if overlapped:
                self._overlapped += 1
                self.overlapped_max = max(self.overlapped_max, self._overlapped)
            try:
                yield
            finally:
                self.in_flight -= 1
                if overlapped:
                    self._overlapped -= 1


_meter: ContextVar[_ReadMeter] = ContextVar("chunk_source_links_meter")


class _DocStatus:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._value = {"chunks_list": chunk_ids, "folder": "default"}

    async def get_by_id(self, _doc_id: str) -> dict[str, Any]:
        async with _meter.get().track():
            await asyncio.sleep(READ_DELAY_SECONDS)
            return self._value

    async def get_folders_for_doc(self, _doc_id: str) -> list[str]:
        async with _meter.get().track():
            await asyncio.sleep(READ_DELAY_SECONDS)
            return ["default"]


class _TextChunks:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._chunks = {
            chunk_id: {
                "_id": chunk_id,
                "content": f"content-{chunk_id}",
                "full_doc_id": "doc-1",
                "file_path": "document.pdf",
                "chunk_order_index": index,
                "tokens": 20,
            }
            for index, chunk_id in enumerate(chunk_ids)
        }

    async def get_by_id(self, chunk_id: str) -> dict[str, Any]:
        """Anchor read — only the two /chunks/... routes pay this."""
        async with _meter.get().track():
            await asyncio.sleep(READ_DELAY_SECONDS)
            return self._chunks[chunk_id]

    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        async with _meter.get().track(overlapped=True):
            await asyncio.sleep(READ_DELAY_SECONDS)
            return [self._chunks[chunk_id] for chunk_id in chunk_ids]


class _Rag:
    def __init__(self) -> None:
        chunk_ids = [f"chunk-{index}" for index in range(CHUNK_COUNT)]
        self.doc_status = _DocStatus(chunk_ids)
        self.text_chunks = _TextChunks(chunk_ids)


async def _fake_source_links(doc_id: str) -> list[dict[str, Any]]:
    async with _meter.get().track(overlapped=True):
        await asyncio.sleep(READ_DELAY_SECONDS)
        return [
            {
                "id": f"link-{index}",
                "doc_id": doc_id,
                "url": f"https://example.invalid/{index}",
                "label": f"source {index}",
            }
            for index in range(SOURCE_LINK_COUNT)
        ]


class _SequentialAsyncio:
    """``asyncio`` proxy whose ``gather``/task pair runs serially.

    Bound onto ``chunk_routes.asyncio`` only — rebinding the real
    ``asyncio.gather`` would serialize this harness's own concurrent load
    driver and fabricate a baseline running at concurrency 1.

    ``ensure_future`` is shimmed to a lazy holder rather than a real task so the
    second read does not start until the first is awaited; that is exactly the
    pre-optimization ordering.
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

    async def gather(self, *coros, **_kwargs):
        return [await coro for coro in coros]


_active_rag: ContextVar[_Rag] = ContextVar("chunk_source_links_rag")
_endpoints: dict[str, Callable[..., Awaitable[Any]]] | None = None


def _ensure_endpoints() -> dict[str, Callable[..., Awaitable[Any]]]:
    global _endpoints
    if _endpoints is None:
        chunk_routes.create_chunk_routes(lambda: _active_rag.get())
        by_path = {}
        for route in chunk_routes.router.routes:
            path = getattr(route, "path", None)
            if path in (
                "/chunks/{chunk_id}/context",
                "/chunks/{chunk_id}/document",
                "/documents/{doc_id}/chunks",
            ):
                by_path[path] = route.endpoint
        missing = {
            "/chunks/{chunk_id}/context",
            "/chunks/{chunk_id}/document",
            "/documents/{doc_id}/chunks",
        } - set(by_path)
        if missing:
            raise RuntimeError(f"chunk routes not registered: {sorted(missing)}")
        _endpoints = by_path
    return _endpoints


#: (label, callable taking the endpoint map -> awaitable response)
ROUTES: tuple[tuple[str, Callable[[dict], Awaitable[Any]]], ...] = (
    ("chunks/context", lambda e: e["/chunks/{chunk_id}/context"]("chunk-5", 3)),
    ("chunks/document", lambda e: e["/chunks/{chunk_id}/document"]("chunk-5")),
    (
        "documents/chunks",
        lambda e: e["/documents/{doc_id}/chunks"]("doc-1", None, None),
    ),
)


@asynccontextmanager
async def _bound(sequential: bool, meter: _ReadMeter):
    original_source_links = chunk_routes._source_links_for_doc
    original_asyncio = chunk_routes.asyncio
    chunk_routes._source_links_for_doc = _fake_source_links
    if sequential:
        chunk_routes.asyncio = _SequentialAsyncio(original_asyncio)
    meter_token = _meter.set(meter)
    try:
        yield
    finally:
        chunk_routes._source_links_for_doc = original_source_links
        chunk_routes.asyncio = original_asyncio
        _meter.reset(meter_token)


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    max_in_flight: int


async def _request(route_call) -> Any:
    endpoints = _ensure_endpoints()
    token = _active_rag.set(_Rag())
    try:
        return await route_call(endpoints)
    finally:
        _active_rag.reset(token)


async def _time_requests(
    route_call, *, sequential: bool, iterations: int, concurrency: int = 1
) -> _Sample:
    meter = _ReadMeter()
    durations: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with _bound(sequential, meter):
        for _ in range(3):  # warmup, outside the measured window
            await _request(route_call)

        async def run_one() -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await _request(route_call)
                durations.append((time.perf_counter() - started) * 1000)
                assert len(response.source_links) == SOURCE_LINK_COUNT

        meter.max_in_flight = 0
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
    cases: list[dict[str, Any]] = []

    for label, route_call in ROUTES:
        baseline_meter = _ReadMeter()
        async with _bound(True, baseline_meter):
            baseline_response = await _request(route_call)
        live_meter = _ReadMeter()
        async with _bound(False, live_meter):
            live_response = await _request(route_call)

        baseline = await _time_requests(route_call, sequential=True, iterations=count)
        optimized = await _time_requests(route_call, sequential=False, iterations=count)

        cases.append(
            {
                "name": f"{label} latency",
                "kind": "ratio",
                "baseline_ms": baseline.mean_ms,
                "optimized_ms": optimized.mean_ms,
                "detail": (
                    f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                    f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
                ),
            }
        )
        cases.append(
            {
                "name": f"{label} overlaps its two independent reads",
                "kind": "structural",
                "passed": (
                    live_meter.overlapped_max == 2
                    and baseline_meter.overlapped_max == 1
                    and live_meter.reads == baseline_meter.reads
                ),
                "detail": (
                    "expected the chunk batch and source-link reads concurrent "
                    "on the live path (serialized baseline 1) with an identical "
                    f"total read count; observed live={live_meter.overlapped_max} "
                    f"baseline={baseline_meter.overlapped_max}, total reads "
                    f"live={live_meter.reads} baseline={baseline_meter.reads}"
                ),
            }
        )
        cases.append(
            {
                "name": f"{label} response unchanged",
                "kind": "structural",
                "passed": live_response == baseline_response,
                "detail": (
                    "expected an identical ordered ChunkContextResponse; "
                    f"observed {len(live_response.chunks)} vs "
                    f"{len(baseline_response.chunks)} chunks"
                ),
            }
        )
    return cases


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
    route_call, *, iterations: int, concurrency: int, rounds: int = 5
) -> tuple[_Sample, _Sample]:
    baselines: list[_Sample] = []
    optimizeds: list[_Sample] = []
    for _ in range(rounds):
        baselines.append(
            await _time_requests(
                route_call,
                sequential=True,
                iterations=iterations,
                concurrency=concurrency,
            )
        )
        optimizeds.append(
            await _time_requests(
                route_call,
                sequential=False,
                iterations=iterations,
                concurrency=concurrency,
            )
        )
    return _pool(baselines), _pool(optimizeds)


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms p50={sample.p50_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"max_reads={sample.max_in_flight}"
    )


async def main() -> None:
    cases = await measure()
    for case in cases:
        if case["kind"] == "ratio":
            gain = (
                (case["baseline_ms"] - case["optimized_ms"]) / case["baseline_ms"] * 100
            )
            print(
                f"{case['name']}: {case['baseline_ms']:.3f}ms -> "
                f"{case['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
            )
        else:
            print(f"structural [{case['name']}]: {case['passed']} ({case['detail']})")

    for label, route_call in ROUTES:
        for phase, concurrency, requests in (("sustained", 8, 80), ("peak", 32, 160)):
            baseline, optimized = await _load_test(
                route_call, iterations=requests, concurrency=concurrency
            )
            _print_sample(f"{label} {phase} baseline", baseline)
            _print_sample(f"{label} {phase} optimized", optimized)


if __name__ == "__main__":
    asyncio.run(main())
