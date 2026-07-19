"""Benchmark: batch WebUI notification mark-all-read writes.

Run as a script:
``python tests/benchmarks/notifications_mark_all_read_batch.py``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import tracemalloc
from contextvars import ContextVar
from contextlib import asynccontextmanager
from typing import Any

from twindb_lightrag_memgraph.server import webui_notificationstore as store_mod

ITERATIONS = 40
NOTIFICATION_COUNT = 200
RTT_SECONDS = 0.001
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 20

_ACTIVE_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "notifications_bench_state", default=None
)


def _build_state() -> dict[str, Any]:
    items = {
        f"n-{idx:04d}": {
            "id": f"n-{idx:04d}",
            "kind": "tag-mutation",
            "title": "Tag",
            "tagname": f"tag-{idx % 32}",
            "suffix": "requested",
            "sub": "bench",
            "rel": "now",
            "read": False,
        }
        for idx in range(NOTIFICATION_COUNT)
    }
    return {
        "items": items,
        "query_count": 0,
        "write_query_count": 0,
        "written_rows": 0,
        "rows_per_write": [],
    }


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def data(self) -> list[dict[str, Any]]:
        return self._rows

    async def consume(self) -> None:
        return None


class _FakeSession:
    async def run(self, query: str, **params: Any) -> _FakeResult:
        state = _ACTIVE_STATE.get()
        if state is None:
            raise AssertionError("fake pool used without active benchmark state")
        await asyncio.sleep(RTT_SECONDS)
        state["query_count"] += 1

        if "RETURN n.data AS data" in query:
            rows = [
                {"data": json.dumps(item, sort_keys=True)}
                for item in state["items"].values()
            ]
            return _FakeResult(rows)

        if "UNWIND $rows AS row" in query:
            rows = params["rows"]
            state["write_query_count"] += 1
            state["written_rows"] += len(rows)
            state["rows_per_write"].append(len(rows))
            for row in rows:
                item_id = row["id"]
                if item_id in state["items"]:
                    state["items"][item_id] = json.loads(row["data"])
            return _FakeResult()

        if "MATCH (m:" in query and "SET m.data = $data" in query:
            state["write_query_count"] += 1
            state["written_rows"] += 1
            state["rows_per_write"].append(1)
            item_id = params["id"]
            if item_id in state["items"]:
                state["items"][item_id] = json.loads(params["data"])
            return _FakeResult()

        raise AssertionError(f"unexpected query in benchmark fake: {query}")


@asynccontextmanager
async def _fake_read_session():
    yield _FakeSession()


@asynccontextmanager
async def _fake_write_session():
    yield _FakeSession()


@asynccontextmanager
async def _fake_write_slot():
    yield None


async def _baseline_mark_all_read(store: store_mod.MemgraphNotificationStore) -> None:
    items = await store.list()
    async with store_mod._pool.acquire_write_slot():
        async with store_mod._pool.get_session() as session:
            for notification in items:
                notification["read"] = True
                result = await session.run(
                    f"MATCH (m:`{store._label}` {{id: $id}}) "
                    "SET m.data = $data, m.`__updated_at` = timestamp()",
                    id=str(notification["id"]),
                    data=json.dumps(notification, sort_keys=True),
                )
                await result.consume()


def _assert_parity(state: dict[str, Any]) -> None:
    assert len(state["items"]) == NOTIFICATION_COUNT
    assert all(item["read"] is True for item in state["items"].values())
    assert state["written_rows"] == NOTIFICATION_COUNT


async def _one_request(fn) -> dict[str, Any]:
    state = _build_state()
    token = _ACTIVE_STATE.set(state)
    try:
        await fn(store_mod.MemgraphNotificationStore(workspace="bench"))
        _assert_parity(state)
    finally:
        _ACTIVE_STATE.reset(token)
    return state


async def _measure_isolated(
    label: str, fn, iterations: int = ITERATIONS
) -> dict[str, Any]:
    durations_ms: list[float] = []
    write_query_counts: list[int] = []
    rows_per_write: list[int] = []
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(iterations):
        start = time.perf_counter()
        state = await _one_request(fn)
        durations_ms.append((time.perf_counter() - start) * 1000)
        write_query_counts.append(state["write_query_count"])
        rows_per_write.extend(state["rows_per_write"])
    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _result(
        label,
        durations_ms,
        elapsed,
        iterations,
        peak,
        write_query_counts,
        rows_per_write,
    )


async def _measure_load(label: str, fn, concurrency: int) -> dict[str, Any]:
    durations_ms: list[float] = []
    write_query_counts: list[int] = []
    rows_per_write: list[int] = []
    tracemalloc.start()
    start_total = time.perf_counter()

    async def worker() -> None:
        start = time.perf_counter()
        state = await _one_request(fn)
        durations_ms.append((time.perf_counter() - start) * 1000)
        write_query_counts.append(state["write_query_count"])
        rows_per_write.extend(state["rows_per_write"])

    for _ in range(ITERATIONS // concurrency):
        await asyncio.gather(*(worker() for _ in range(concurrency)))

    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _result(
        label,
        durations_ms,
        elapsed,
        len(durations_ms),
        peak,
        write_query_counts,
        rows_per_write,
    )


def _percentile(samples: list[float], pct: int) -> float:
    ordered = sorted(samples)
    idx = max(min(int(len(ordered) * pct / 100) - 1, len(ordered) - 1), 0)
    return ordered[idx]


def _result(
    label: str,
    durations_ms: list[float],
    elapsed: float,
    iterations: int,
    peak_bytes: int,
    write_query_counts: list[int],
    rows_per_write: list[int],
) -> dict[str, Any]:
    return {
        "label": label,
        "iterations": iterations,
        "mean_ms": statistics.fmean(durations_ms),
        "p95_ms": _percentile(durations_ms, 95),
        "p99_ms": _percentile(durations_ms, 99),
        "req_per_s": iterations / elapsed,
        "peak_mb": peak_bytes / 1024 / 1024,
        "write_query_count": (
            statistics.fmean(write_query_counts) if write_query_counts else 0
        ),
        "max_rows_per_write": max(rows_per_write, default=0),
    }


def _print_result(result: dict[str, Any]) -> None:
    print(
        f"{result['label']}: mean={result['mean_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms p99={result['p99_ms']:.3f}ms "
        f"throughput={result['req_per_s']:.1f} req/s "
        f"peak={result['peak_mb']:.3f}MB "
        f"writes={result['write_query_count']:.1f} "
        f"max_rows_per_write={result['max_rows_per_write']}"
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: ratio + structural guard for the batched write."""
    n = iterations or ITERATIONS
    orig_read_session = store_mod._pool.get_read_session
    orig_write_session = store_mod._pool.get_session
    orig_write_slot = store_mod._pool.acquire_write_slot
    store_mod._pool.get_read_session = _fake_read_session
    store_mod._pool.get_session = _fake_write_session
    store_mod._pool.acquire_write_slot = _fake_write_slot
    try:
        baseline = await _measure_isolated(
            "baseline_per_notification_writes", _baseline_mark_all_read, iterations=n
        )
        optimized = await _measure_isolated(
            "optimized_unwind_batch_write",
            store_mod.MemgraphNotificationStore.mark_all_read,
            iterations=n,
        )
    finally:
        store_mod._pool.get_read_session = orig_read_session
        store_mod._pool.get_session = orig_write_session
        store_mod._pool.acquire_write_slot = orig_write_slot
    return [
        {
            "name": "MemgraphNotificationStore.mark_all_read (batched write)",
            "kind": "ratio",
            "baseline_ms": baseline["mean_ms"],
            "optimized_ms": optimized["mean_ms"],
        },
        {
            "name": "MemgraphNotificationStore.mark_all_read writes one batch",
            "kind": "structural",
            "passed": (
                optimized["write_query_count"] == 1
                and optimized["max_rows_per_write"] == NOTIFICATION_COUNT
            ),
            "detail": (
                f"write queries={optimized['write_query_count']}, "
                f"max rows/write={optimized['max_rows_per_write']}"
            ),
        },
    ]


async def main() -> None:
    orig_read_session = store_mod._pool.get_read_session
    orig_write_session = store_mod._pool.get_session
    orig_write_slot = store_mod._pool.acquire_write_slot
    store_mod._pool.get_read_session = _fake_read_session
    store_mod._pool.get_session = _fake_write_session
    store_mod._pool.acquire_write_slot = _fake_write_slot
    try:
        baseline = await _measure_isolated(
            "baseline_per_notification_writes", _baseline_mark_all_read
        )
        optimized = await _measure_isolated(
            "optimized_unwind_batch_write",
            store_mod.MemgraphNotificationStore.mark_all_read,
        )
        baseline_sustained = await _measure_load(
            "baseline_sustained_load",
            _baseline_mark_all_read,
            SUSTAINED_CONCURRENCY,
        )
        optimized_sustained = await _measure_load(
            "optimized_sustained_load",
            store_mod.MemgraphNotificationStore.mark_all_read,
            SUSTAINED_CONCURRENCY,
        )
        baseline_peak = await _measure_load(
            "baseline_peak_load",
            _baseline_mark_all_read,
            PEAK_CONCURRENCY,
        )
        optimized_peak = await _measure_load(
            "optimized_peak_load",
            store_mod.MemgraphNotificationStore.mark_all_read,
            PEAK_CONCURRENCY,
        )
    finally:
        store_mod._pool.get_read_session = orig_read_session
        store_mod._pool.get_session = orig_write_session
        store_mod._pool.acquire_write_slot = orig_write_slot

    for result in (
        baseline,
        optimized,
        baseline_sustained,
        optimized_sustained,
        baseline_peak,
        optimized_peak,
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
    print(
        "write round-trips: "
        f"{baseline['write_query_count']:.1f} -> {optimized['write_query_count']:.1f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
