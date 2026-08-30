"""Bolt micro-benchmark: overlap the two storage-info probes in ``quota.snapshot``.

``snapshot()`` reads Memgraph's instance-level metrics and the current
database's metrics — two independent probes, each on its own read session and
each fail-open on its own error. It backs ``GET /twin/api/quota`` (polled by
every open WebUI tab) and ``enforce_instance_quota()``, which gates every
ingestion POST, so the serialized pair cost two round-trips on both.

"before" is reproduced by monkeypatching the module's ``asyncio`` proxy so
``gather`` runs serially — same function body, only the overlap toggled off.
``READ_CAPACITY`` models the Bolt read pool so the load test shows the probes
queueing on the pool rather than pretending sessions are free.

Run standalone with::

    uv run python tests/benchmarks/quota_snapshot_gather.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from twindb_lightrag_memgraph.server import quota

ITERATIONS = 80
READ_DELAY_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))

# Field names are the Memgraph 3.12 contract (quota._GRAPH_KEYS /
# _VECTOR_KEYS / _USED_KEYS / _RAM_LIMIT_KEYS / _LICENSE_LIMIT_KEYS). An earlier
# version used the pre-3.11 `memory_graph` / `memory_vector_index` spellings,
# which `_pick` simply misses: `graph` came back None, `billed` stayed None, and
# the whole license-wall branch was never exercised — the payload parity case
# was comparing a degenerate snapshot. Keep these in sync with quota.py.
_INSTANCE_ROWS = [
    {"storage info": "memory_tracked", "value": "3.5GiB"},
    {"storage info": "memory_limit", "value": "8.0GiB"},
    {"storage info": "license_memory_limit", "value": "4.0GiB"},
]
_DATABASE_ROWS = [
    {"storage info": "graph_memory_tracked", "value": "2.5GiB"},
    {"storage info": "vector_index_memory_tracked", "value": "0.5GiB"},
]


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows

    async def consume(self) -> None:
        return None


class _ProbeMeter:
    def __init__(self, capacity: int = READ_CAPACITY) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.probes = 0
        self._capacity = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def session(self):
        # The Bolt driver blocks on an exhausted pool; model that back-pressure.
        async with self._capacity:
            self.probes += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                yield _FakeSession()
            finally:
                self.in_flight -= 1


class _FakeSession:
    async def run(self, query: str, **_params: Any) -> _FakeResult:
        await asyncio.sleep(READ_DELAY_SECONDS)
        if "DATABASE" in query.upper():
            return _FakeResult(list(_DATABASE_ROWS))
        return _FakeResult(list(_INSTANCE_ROWS))


class _SequentialAsyncio:
    """``asyncio`` proxy whose ``gather`` runs serially.

    Bound onto ``quota.asyncio`` only — rebinding ``asyncio.gather`` itself
    would serialize this harness's own concurrent load driver.
    """

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def gather(self, *coros, **_kwargs):
        return [await coro for coro in coros]


class _FakePool:
    def __init__(self, meter: _ProbeMeter) -> None:
        self._meter = meter

    def get_read_session(self):
        return self._meter.session()


@asynccontextmanager
async def _bound(sequential: bool, meter: _ProbeMeter):
    original_pool = quota._pool
    original_asyncio = quota.asyncio
    quota._pool = _FakePool(meter)
    if sequential:
        quota.asyncio = _SequentialAsyncio(original_asyncio)
    try:
        yield
    finally:
        quota._pool = original_pool
        quota.asyncio = original_asyncio


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    max_in_flight: int


async def _time_snapshots(
    *, sequential: bool, iterations: int, concurrency: int = 1
) -> _Sample:
    meter = _ProbeMeter()
    durations: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with _bound(sequential, meter):
        for _ in range(3):  # warmup, outside the measured window
            await quota.snapshot()

        async def run_one() -> None:
            async with semaphore:
                started = time.perf_counter()
                snap = await quota.snapshot()
                durations.append((time.perf_counter() - started) * 1000)
                assert snap["status"] in {"ok", "warning", "blocked"}

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

    baseline_meter = _ProbeMeter()
    async with _bound(True, baseline_meter):
        baseline_snapshot = await quota.snapshot()
    live_meter = _ProbeMeter()
    async with _bound(False, live_meter):
        live_snapshot = await quota.snapshot()

    baseline = await _time_snapshots(sequential=True, iterations=count)
    optimized = await _time_snapshots(sequential=False, iterations=count)

    return [
        {
            "name": "quota snapshot latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "storage-info probes overlap",
            "kind": "structural",
            "passed": (
                live_meter.max_in_flight == 2
                and baseline_meter.max_in_flight == 1
                and live_meter.probes == baseline_meter.probes == 2
            ),
            "detail": (
                "expected 2 concurrent probes on the live path (serialized "
                "baseline 1) and exactly 2 probes each; observed live="
                f"{live_meter.max_in_flight} baseline="
                f"{baseline_meter.max_in_flight}, probes live={live_meter.probes} "
                f"baseline={baseline_meter.probes}"
            ),
        },
        {
            "name": "snapshot payload unchanged",
            "kind": "structural",
            "passed": live_snapshot == baseline_snapshot,
            "detail": (
                "expected an identical snapshot payload; differing keys: "
                + ", ".join(
                    sorted(
                        key
                        for key in set(live_snapshot) | set(baseline_snapshot)
                        if live_snapshot.get(key) != baseline_snapshot.get(key)
                    )
                )
                or "none"
            ),
        },
        {
            # Without this, a benchmark that misspells the database-metric keys
            # still "passes" payload parity — by comparing two snapshots that
            # both silently dropped the billed footprint. Pin that the fixture
            # really drives the database read and the license wall.
            "name": "database metrics and license wall exercised",
            "kind": "structural",
            "passed": (
                live_snapshot.get("graph_bytes") is not None
                and live_snapshot.get("vector_bytes") is not None
                and live_snapshot.get("binding") == "license"
                and live_snapshot.get("used_pct") is not None
            ),
            "detail": (
                "expected the current-database probe to populate graph/vector "
                "bytes and the license wall to bind (3.0GiB billed vs 4.0GiB "
                f"license beats 3.5/8.0 RAM); observed graph_bytes="
                f"{live_snapshot.get('graph_bytes')}, vector_bytes="
                f"{live_snapshot.get('vector_bytes')}, binding="
                f"{live_snapshot.get('binding')!r}"
            ),
        },
    ]


def _pool_samples(samples: list[_Sample]) -> _Sample:
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
            await _time_snapshots(
                sequential=True, iterations=iterations, concurrency=concurrency
            )
        )
        optimizeds.append(
            await _time_snapshots(
                sequential=False, iterations=iterations, concurrency=concurrency
            )
        )
    return _pool_samples(baselines), _pool_samples(optimizeds)


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms p50={sample.p50_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"max_probes={sample.max_in_flight}"
    )


async def main() -> None:
    cases = await measure()
    ratio = cases[0]
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"sequential: {ratio['baseline_ms']:.3f}ms -> "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
    )
    for case in cases[1:]:
        print(f"structural [{case['name']}]: {case['passed']} ({case['detail']})")

    for label, concurrency, requests in (
        ("sustained", 8, 80),
        ("peak", 32, 160),
    ):
        baseline, optimized = await _load_test(
            iterations=requests, concurrency=concurrency
        )
        _print_sample(f"{label} baseline", baseline)
        _print_sample(f"{label} optimized", optimized)


if __name__ == "__main__":
    asyncio.run(main())
