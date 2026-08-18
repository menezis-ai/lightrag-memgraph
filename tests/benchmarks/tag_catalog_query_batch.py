"""Benchmark the set-based WebUI tag catalog/usage query.

Run standalone with::

    uv run python tests/benchmarks/tag_catalog_query_batch.py

The in-process baseline preserves the pre-optimization implementation: one
catalog read followed by one folder-scoped usage aggregation read.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import tracemalloc
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from twindb_lightrag_memgraph.server import webui_tagstore as store_mod

ITERATIONS = 80
TAG_COUNT = 96
READ_RTT_SECONDS = 0.003
READ_CAPACITY = 20
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 20


@dataclass
class _State:
    query_count: int = 0
    combined_query_count: int = 0
    last_query: str = ""


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    peak_mb: float
    mean_queries: float


_active_state: ContextVar[_State | None] = ContextVar(
    "tag_catalog_benchmark_state", default=None
)
_read_slots = asyncio.Semaphore(READ_CAPACITY)


def _tag_entry(index: int) -> dict[str, Any]:
    return {
        "tag": f"tag-{index:04d}",
        "status": "active" if index % 11 else "deleted",
        "category": f"category-{index % 8}",
        "sources_count": -1,
        "chunks_count": -1,
        "aliases": [f"alias-{index}"],
    }


def _catalog_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"tag-{index:04d}",
            "data": json.dumps(_tag_entry(index), sort_keys=True),
        }
        for index in range(TAG_COUNT)
    ]


def _usage_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": f"tag-{index:04d}",
            "sources_count": index % 7,
            "chunks_count": (index % 7) * 3,
        }
        for index in range(TAG_COUNT)
        if index % 5
    ]


def _combined_rows() -> list[dict[str, Any]]:
    usage = {row["id"]: row for row in _usage_rows()}
    return [
        {
            **row,
            "sources_count": usage.get(row["id"], {}).get("sources_count", 0),
            "chunks_count": usage.get(row["id"], {}).get("chunks_count", 0),
        }
        for row in _catalog_rows()
    ]


def _expected_tags() -> list[dict[str, Any]]:
    usage = {row["id"]: row for row in _usage_rows()}
    out: list[dict[str, Any]] = []
    for row in _catalog_rows():
        item = json.loads(row["data"])
        item["sources_count"] = usage.get(row["id"], {}).get("sources_count", 0)
        item["chunks_count"] = usage.get(row["id"], {}).get("chunks_count", 0)
        if store_mod._visible_catalog_tag(item):
            out.append(item)
    return out


class _Result:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows

    async def consume(self) -> None:
        return None


class _Session:
    async def run(self, query: str, **_params: Any) -> _Result:
        state = _active_state.get()
        if state is None:
            raise AssertionError("benchmark state is required")
        state.query_count += 1
        state.last_query = query
        async with _read_slots:
            await asyncio.sleep(READ_RTT_SECONDS)

        if "OPTIONAL MATCH" in query and "t.data AS data" in query:
            state.combined_query_count += 1
            return _Result(_combined_rows())
        if "t.data AS data" in query:
            return _Result(_catalog_rows())
        if "sources_count" in query and "chunks_count" in query:
            return _Result(_usage_rows())
        raise AssertionError(f"unexpected tag benchmark query: {query}")


@asynccontextmanager
async def _read_session():
    yield _Session()


async def _baseline_list_tags(
    store: store_mod.MemgraphTagStore,
) -> list[dict[str, Any]]:
    async with store_mod._pool.get_read_session() as session:
        result = await session.run(f"""
            MATCH (t:`{store._tag_label}`)
            RETURN t.id AS id, t.data AS data
            ORDER BY t.`__created_at`, t.id
            """)
        rows = await result.data()
        await result.consume()
        usage_result = await session.run(
            f"""
            MATCH (d:`DocStatus_benchmark`)
                  -[:TAGGED_WITH]->(t:`{store._tag_label}`)
            WHERE EXISTS(
                (d)-[:MEMBER_OF]->(:`Folder_benchmark` {{id: $folder}})
            )
            RETURN t.id AS id,
                   count(DISTINCT d) AS sources_count,
                   sum(coalesce(d.chunks_count, 0)) AS chunks_count
            """,
            folder="benchmark",
        )
        usage_rows = await usage_result.data()
        await usage_result.consume()

    usage_by_id = {
        row["id"]: {
            "sources_count": int(row.get("sources_count") or 0),
            "chunks_count": int(row.get("chunks_count") or 0),
        }
        for row in usage_rows
        if row.get("id")
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        item = json.loads(row["data"])
        usage = usage_by_id.get(row["id"], {})
        item["sources_count"] = usage.get("sources_count", 0)
        item["chunks_count"] = usage.get("chunks_count", 0)
        if store_mod._visible_catalog_tag(item):
            out.append(item)
    return out


async def _live_list_tags(
    store: store_mod.MemgraphTagStore,
) -> list[dict[str, Any]]:
    return await store.list_tags()


async def _one_request(
    request: Callable[[store_mod.MemgraphTagStore], Awaitable[list[dict[str, Any]]]],
) -> tuple[float, _State]:
    state = _State()
    token = _active_state.set(state)
    try:
        started = time.perf_counter()
        tags = await request(store_mod.MemgraphTagStore(workspace="benchmark"))
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        _active_state.reset(token)
    assert tags == _expected_tags()
    return elapsed_ms, state


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index]


async def _time_requests(
    request: Callable[[store_mod.MemgraphTagStore], Awaitable[list[dict[str, Any]]]],
    *,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    durations: list[float] = []
    query_counts: list[int] = []
    tracemalloc.start()
    started = time.perf_counter()

    async def worker() -> None:
        elapsed_ms, state = await _one_request(request)
        durations.append(elapsed_ms)
        query_counts.append(state.query_count)

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
        mean_queries=statistics.fmean(query_counts),
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: latency ratio plus deterministic one-query guard."""
    count = iterations or ITERATIONS
    original_session = store_mod._pool.get_read_session
    store_mod._pool.get_read_session = _read_session
    try:
        _, baseline_state = await _one_request(_baseline_list_tags)
        _, live_state = await _one_request(_live_list_tags)
        baseline = await _time_requests(_baseline_list_tags, iterations=count)
        optimized = await _time_requests(_live_list_tags, iterations=count)
    finally:
        store_mod._pool.get_read_session = original_session

    return [
        {
            "name": "tag catalog set-based query latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"reads {baseline.mean_queries:.1f}->{optimized.mean_queries:.1f}"
            ),
        },
        {
            "name": "tag catalog uses one folder-scoped aggregate query",
            "kind": "structural",
            "passed": (
                baseline_state.query_count == 2
                and live_state.query_count == 1
                and live_state.combined_query_count == 1
                and "OPTIONAL MATCH" in live_state.last_query
                and "count(DISTINCT d)" in live_state.last_query
            ),
            "detail": (
                f"baseline reads={baseline_state.query_count}; "
                f"live reads={live_state.query_count}; "
                f"combined={live_state.combined_query_count}"
            ),
        },
    ]


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"peak={sample.peak_mb:.3f}MB reads={sample.mean_queries:.1f}"
    )


async def main() -> None:
    original_session = store_mod._pool.get_read_session
    store_mod._pool.get_read_session = _read_session
    try:
        baseline = await _time_requests(_baseline_list_tags, iterations=ITERATIONS)
        optimized = await _time_requests(_live_list_tags, iterations=ITERATIONS)
        baseline_sustained = await _time_requests(
            _baseline_list_tags,
            iterations=ITERATIONS,
            concurrency=SUSTAINED_CONCURRENCY,
        )
        optimized_sustained = await _time_requests(
            _live_list_tags,
            iterations=ITERATIONS,
            concurrency=SUSTAINED_CONCURRENCY,
        )
        baseline_peak = await _time_requests(
            _baseline_list_tags,
            iterations=ITERATIONS,
            concurrency=PEAK_CONCURRENCY,
        )
        optimized_peak = await _time_requests(
            _live_list_tags,
            iterations=ITERATIONS,
            concurrency=PEAK_CONCURRENCY,
        )
    finally:
        store_mod._pool.get_read_session = original_session
    _print_sample("tag_catalog_baseline", baseline)
    _print_sample("tag_catalog_optimized", optimized)
    _print_sample("tag_catalog_baseline_sustained", baseline_sustained)
    _print_sample("tag_catalog_optimized_sustained", optimized_sustained)
    _print_sample("tag_catalog_baseline_peak", baseline_peak)
    _print_sample("tag_catalog_optimized_peak", optimized_peak)


if __name__ == "__main__":
    asyncio.run(main())
