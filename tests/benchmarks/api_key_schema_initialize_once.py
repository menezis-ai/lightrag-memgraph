"""Benchmark one-time API-key schema initialization.

Run standalone with::

    .venv/bin/python tests/benchmarks/api_key_schema_initialize_once.py

The in-process baseline preserves the pre-optimization request path: every
API-key list refresh enters the write pool and submits both idempotent
``CREATE INDEX`` statements before the list read.  The live path calls
``api_key_store.initialize`` so the benchmark and its structural guard follow
the production implementation.
"""

from __future__ import annotations

import asyncio
import itertools
import statistics
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator

from twindb_lightrag_memgraph.server import api_key_store

ITERATIONS = 80
DDL_RTT_SECONDS = 0.004
READ_RTT_SECONDS = 0.004
WRITE_CAPACITY = 8
READ_CAPACITY = 20

_WORKSPACES = itertools.count()
_LIST_PAYLOAD = [
    {
        "id": "bench-key",
        "name": "benchmark",
        "prefix": "twk_bench…",
        "scopes": ["api:*"],
        "folders": [],
        "created_at": 1,
        "created_by": "benchmark",
        "last_used_at": None,
        "revoked_at": None,
    }
]


class _Result:
    async def consume(self) -> None:
        return None


class _Session:
    def __init__(self, harness: "_Harness") -> None:
        self._harness = harness

    async def run(self, query: str) -> _Result:
        self._harness.ddl_queries += 1
        await asyncio.sleep(DDL_RTT_SECONDS)
        # The real server reports this after the first successful CREATE.  The
        # request path still pays the complete round-trip before it can catch
        # the idempotent result.
        if query in self._harness.created_indexes:
            raise RuntimeError("index already exists")
        self._harness.created_indexes.add(query)
        return _Result()


class _Harness:
    def __init__(self) -> None:
        self.write_slots = 0
        self.ddl_queries = 0
        self.list_reads = 0
        self.created_indexes: set[str] = set()
        self._write_capacity = asyncio.Semaphore(WRITE_CAPACITY)
        self._read_capacity = asyncio.Semaphore(READ_CAPACITY)

    @asynccontextmanager
    async def acquire_write_slot(self):
        async with self._write_capacity:
            self.write_slots += 1
            yield

    @asynccontextmanager
    async def get_session(self):
        yield _Session(self)

    async def list_keys(self) -> list[dict[str, Any]]:
        async with self._read_capacity:
            self.list_reads += 1
            await asyncio.sleep(READ_RTT_SECONDS)
            return [dict(item) for item in _LIST_PAYLOAD]


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    harness: _Harness


@contextmanager
def _patched_pool(harness: _Harness) -> Iterator[None]:
    original_write_slot = api_key_store._pool.acquire_write_slot
    original_session = api_key_store._pool.get_session
    api_key_store._pool.acquire_write_slot = harness.acquire_write_slot
    api_key_store._pool.get_session = harness.get_session
    try:
        yield
    finally:
        api_key_store._pool.acquire_write_slot = original_write_slot
        api_key_store._pool.get_session = original_session


async def _baseline_initialize(workspace: str, harness: _Harness) -> None:
    """Pre-optimization body: two DDL round-trips on every request."""
    label = api_key_store._label(workspace)
    async with harness.acquire_write_slot(), harness.get_session() as session:
        for field in ("id", "hash"):
            try:
                result = await session.run(f"CREATE INDEX ON :`{label}`({field})")
                await result.consume()
            except Exception as exc:  # noqa: BLE001 - mirrors production
                if "already exists" in str(exc).lower():
                    continue
                raise


async def _baseline_request(workspace: str, harness: _Harness) -> list[dict[str, Any]]:
    await _baseline_initialize(workspace, harness)
    return await harness.list_keys()


async def _live_request(workspace: str, harness: _Harness) -> list[dict[str, Any]]:
    await api_key_store.initialize(workspace)
    return await harness.list_keys()


async def _time_requests(
    request: Callable[[str, _Harness], Awaitable[list[dict[str, Any]]]],
    *,
    iterations: int,
    concurrency: int,
) -> _Sample:
    harness = _Harness()
    workspace = f"bolt_api_keys_{next(_WORKSPACES)}"
    durations: list[float] = []
    admission = asyncio.Semaphore(concurrency)

    async def run_one() -> None:
        async with admission:
            started = time.perf_counter()
            response = await request(workspace, harness)
            durations.append((time.perf_counter() - started) * 1000)
            assert response == _LIST_PAYLOAD

    started = time.perf_counter()
    with _patched_pool(harness):
        await asyncio.gather(*(run_one() for _ in range(iterations)))
    elapsed = time.perf_counter() - started
    ordered = sorted(durations)

    def percentile(percent: float) -> float:
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percent) - 1))
        return ordered[index]

    return _Sample(
        mean_ms=statistics.mean(durations),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        requests_per_second=iterations / elapsed,
        harness=harness,
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS
    baseline = await _time_requests(
        _baseline_request,
        iterations=count,
        concurrency=1,
    )
    optimized = await _time_requests(
        _live_request,
        iterations=count,
        concurrency=1,
    )

    structural_requests = max(8, min(count, 20))
    structural = await _time_requests(
        _live_request,
        iterations=structural_requests,
        concurrency=structural_requests,
    )
    return [
        {
            "name": "API-key list refresh latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "schema DDL is single-flight and cached",
            "kind": "structural",
            "passed": (
                structural.harness.write_slots == 1
                and structural.harness.ddl_queries == 2
                and structural.harness.list_reads == structural_requests
            ),
            "detail": (
                "expected one write slot and two DDL queries for "
                f"{structural_requests} concurrent refreshes; observed "
                f"slots={structural.harness.write_slots}, "
                f"ddl={structural.harness.ddl_queries}, "
                f"reads={structural.harness.list_reads}"
            ),
        },
    ]


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"write_slots={sample.harness.write_slots} "
        f"ddl={sample.harness.ddl_queries} reads={sample.harness.list_reads}"
    )


async def main() -> None:
    cases = await measure()
    ratio, structural = cases
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"isolated: {ratio['baseline_ms']:.3f}ms -> "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
    )
    print(f"structural: {structural['passed']} ({structural['detail']})")

    for label, concurrency, requests in (
        ("sustained", 8, 160),
        ("peak", 32, 320),
    ):
        baseline = await _time_requests(
            _baseline_request,
            iterations=requests,
            concurrency=concurrency,
        )
        optimized = await _time_requests(
            _live_request,
            iterations=requests,
            concurrency=concurrency,
        )
        _print_sample(f"{label} baseline", baseline)
        _print_sample(f"{label} optimized", optimized)


if __name__ == "__main__":
    asyncio.run(main())
