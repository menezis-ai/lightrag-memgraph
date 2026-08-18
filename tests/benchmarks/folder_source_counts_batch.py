"""Benchmark batched DocStatus source counts for the folder selector.

Run standalone with::

    uv run python tests/benchmarks/folder_source_counts_batch.py

The in-process baseline preserves the pre-optimization route body: call
``get_status_counts(folder=...)`` once per configured folder.  The live path
calls :func:`server.webui.routes_folders._folder_source_counts`.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from twindb_lightrag_memgraph import docstatus_impl
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.webui import routes_folders

ITERATIONS = 80
FOLDER_COUNT = 16
READ_DELAY_SECONDS = 0.002


@dataclass
class _State:
    status_count_calls: int = 0
    folder_count_calls: int = 0
    batch_query: str = ""

    @property
    def read_queries(self) -> int:
        return self.status_count_calls + self.folder_count_calls


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    mean_queries: float


_active_state: ContextVar[_State | None] = ContextVar(
    "folder_source_counts_benchmark_state", default=None
)


def _folders() -> list[dict[str, Any]]:
    return [{"id": f"folder_{index:02d}"} for index in range(FOLDER_COUNT)]


def _status_counts(folder: str) -> dict[str, int]:
    index = int(folder.rsplit("_", 1)[1])
    return {"processed": index + 1, "failed": index % 3}


def _expected_counts() -> dict[str, int]:
    return {
        folder["id"]: sum(_status_counts(folder["id"]).values())
        for folder in _folders()
    }


class _DocStatus:
    @staticmethod
    def _label() -> str:
        return "DocStatus_benchmark"

    @staticmethod
    def _folder_label() -> str:
        return "Folder_benchmark"

    async def get_status_counts(self, folder: str | None = None) -> dict[str, int]:
        state = _active_state.get()
        if state is None or folder is None:
            raise AssertionError("benchmark state/folder is required")
        state.status_count_calls += 1
        await asyncio.sleep(READ_DELAY_SECONDS)
        return _status_counts(folder)

    get_folder_counts = MemgraphDocStatusStorage.get_folder_counts


class _Result:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = records

    def __aiter__(self):
        async def rows():
            for record in self._records:
                yield record

        return rows()

    async def consume(self) -> None:
        return None


class _Session:
    async def run(self, query: str, **params: Any) -> _Result:
        state = _active_state.get()
        if state is None:
            raise AssertionError("benchmark state is required")
        state.folder_count_calls += 1
        state.batch_query = query
        await asyncio.sleep(READ_DELAY_SECONDS)
        return _Result(
            [
                {
                    "folder": folder,
                    "cnt": sum(_status_counts(folder).values()),
                }
                for folder in params["folders"]
            ]
        )


@asynccontextmanager
async def _read_session():
    yield _Session()


class _Rag:
    doc_status = _DocStatus()


_rag = _Rag()


async def _baseline_request(folders: list[dict[str, Any]]) -> dict[str, int]:
    """Pre-optimization route body with one read per folder."""
    counts: dict[str, int] = {}
    for folder in folders:
        by_status = await _rag.doc_status.get_status_counts(folder=folder["id"])
        counts[folder["id"]] = sum(int(value or 0) for value in by_status.values())
    return counts


async def _live_request(folders: list[dict[str, Any]]) -> dict[str, int]:
    return await routes_folders._folder_source_counts(folders)


async def _one_request(
    request: Callable[[list[dict[str, Any]]], Awaitable[dict[str, int]]],
) -> tuple[float, _State]:
    state = _State()
    token = _active_state.set(state)
    try:
        started = time.perf_counter()
        counts = await request(_folders())
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        _active_state.reset(token)
    assert counts == _expected_counts()
    return elapsed_ms, state


async def _time_requests(
    request: Callable[[list[dict[str, Any]]], Awaitable[dict[str, int]]],
    *,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    durations: list[float] = []
    query_counts: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one() -> None:
        async with semaphore:
            elapsed_ms, state = await _one_request(request)
            durations.append(elapsed_ms)
            query_counts.append(state.read_queries)

    started = time.perf_counter()
    await asyncio.gather(*(run_one() for _ in range(iterations)))
    elapsed = time.perf_counter() - started
    ordered = sorted(durations)

    def percentile(percent: float) -> float:
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percent) - 1))
        return ordered[index]

    return _Sample(
        mean_ms=statistics.fmean(durations),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        requests_per_second=iterations / elapsed,
        mean_queries=statistics.fmean(query_counts),
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: self-normalizing ratio plus deterministic query guard."""
    count = iterations or ITERATIONS
    original_get_rag = webui_router._get_rag
    original_read_session = docstatus_impl._pool.get_read_session
    webui_router._get_rag = lambda: _rag
    docstatus_impl._pool.get_read_session = _read_session
    try:
        parity_state = _State()
        token = _active_state.set(parity_state)
        try:
            baseline_counts = await _baseline_request(_folders())
        finally:
            _active_state.reset(token)
        live_elapsed, live_state = await _one_request(_live_request)
        baseline = await _time_requests(_baseline_request, iterations=count)
        optimized = await _time_requests(_live_request, iterations=count)
    finally:
        webui_router._get_rag = original_get_rag
        docstatus_impl._pool.get_read_session = original_read_session

    return [
        {
            "name": "folder selector source-count latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "folder selector batches DocStatus counts",
            "kind": "structural",
            "passed": (
                live_state.folder_count_calls == 1
                and live_state.status_count_calls == 0
                and baseline_counts == _expected_counts()
                and "WHERE f.id IN $folders" in live_state.batch_query
                and "count(DISTINCT n) AS cnt" in live_state.batch_query
                and "UNWIND" not in live_state.batch_query
            ),
            "detail": (
                "expected one set-based DISTINCT query and no per-folder reads "
                f"with identical counts; observed batch={live_state.folder_count_calls}, "
                f"per-folder={live_state.status_count_calls}, set_query="
                f"{'WHERE f.id IN $folders' in live_state.batch_query}, "
                f"distinct={'count(DISTINCT n) AS cnt' in live_state.batch_query}, "
                f"elapsed={live_elapsed:.3f}ms"
            ),
        },
    ]


async def _load_test(iterations: int, concurrency: int) -> tuple[_Sample, _Sample]:
    baseline = await _time_requests(
        _baseline_request, iterations=iterations, concurrency=concurrency
    )
    optimized = await _time_requests(
        _live_request, iterations=iterations, concurrency=concurrency
    )
    return baseline, optimized


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"reads/request={sample.mean_queries:.1f}"
    )


async def main() -> None:
    original_get_rag = webui_router._get_rag
    original_read_session = docstatus_impl._pool.get_read_session
    webui_router._get_rag = lambda: _rag
    docstatus_impl._pool.get_read_session = _read_session
    try:
        cases = await measure()
        ratio = cases[0]
        structural = cases[1]
        gain = (
            (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
        )
        print(
            f"sequential: {ratio['baseline_ms']:.3f}ms -> "
            f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
        )
        print(f"structural: {structural['passed']} ({structural['detail']})")

        for label, concurrency, requests in (
            ("sustained", 8, 160),
            ("peak", 32, 320),
        ):
            baseline, optimized = await _load_test(requests, concurrency)
            _print_sample(f"{label} baseline", baseline)
            _print_sample(f"{label} optimized", optimized)
    finally:
        webui_router._get_rag = original_get_rag
        docstatus_impl._pool.get_read_session = original_read_session


if __name__ == "__main__":
    asyncio.run(main())
