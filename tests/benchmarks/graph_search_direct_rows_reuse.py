"""Benchmark request-local reuse of direct-member rows in graph label search.

Run standalone with::

    uv run python tests/benchmarks/graph_search_direct_rows_reuse.py

The in-process baseline preserves the pre-optimization search body, which loads
the same direct ``GRAPH_MEMBER_OF`` entity rows once for direct label matches
and again to build the override membership set. The live path calls
``graph_reader._search_labels_scoped``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from twindb_lightrag_memgraph.server import folder as folder_mod
from twindb_lightrag_memgraph.server import graph_reader

ITERATIONS = 80
READ_DELAY_SECONDS = 0.004
READ_CAPACITY = 20
WORKSPACE = "benchmark"
FOLDER = "folder-a"
QUERY = "oracle"
MEMBER_CHUNKS = {"chunk-a"}
EXPECTED_LABELS = ["native-entity", "direct-entity", "override-entity"]


@dataclass
class _State:
    reads: dict[str, int] = field(default_factory=dict)

    def record(self, name: str) -> None:
        self.reads[name] = self.reads.get(name, 0) + 1

    @property
    def total_reads(self) -> int:
        return sum(self.reads.values())


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    mean_reads: float
    durations_ms: tuple[float, ...]
    elapsed_seconds: float


_active_state: ContextVar[_State | None] = ContextVar(
    "graph_search_direct_rows_benchmark_state", default=None
)
_read_slots = asyncio.Semaphore(READ_CAPACITY)


async def _read(name: str) -> None:
    state = _active_state.get()
    if state is None:
        raise AssertionError("benchmark state is required")
    state.record(name)
    async with _read_slots:
        await asyncio.sleep(READ_DELAY_SECONDS)


async def _search_member_chunk_labels(**_kwargs: Any) -> list[str]:
    await _read("member_search")
    return ["native-entity"]


async def _load_folder_overrides(
    _workspace: str, _folder: str
) -> dict[str, dict[str, Any]]:
    await _read("overrides")
    return {
        "override-entity": {
            "display_name": "Oracle Database",
            "deleted": False,
        }
    }


async def _load_direct_member_entity_rows(
    _workspace: str, _folder: str
) -> list[dict[str, Any]]:
    await _read("direct_members")
    return [
        {
            "entity_id": "direct-entity",
            "display_name": "Oracle Operator Entity",
        }
    ]


async def _load_chunk_to_doc_index(_workspace: str) -> dict[str, str]:
    await _read("chunk_index")
    return {"chunk-a": "doc-a"}


async def _load_member_docs(_workspace: str, _folder: str) -> set[str]:
    await _read("member_docs")
    return {"doc-a"}


async def _entity_mutation_gate(
    _workspace: str,
    _entity_id: str,
    _chunk_to_doc: dict[str, str],
    _member_docs: set[str],
) -> str:
    await _read("entity_gate")
    return graph_reader._GATE_MEMBER


async def _baseline_request() -> list[str]:
    """Pre-optimization body with two identical direct-member reads."""
    out = await graph_reader._search_member_chunk_labels(
        workspace=WORKSPACE,
        q=QUERY,
        member_chunks=MEMBER_CHUNKS,
        limit=50,
        query="benchmark",
    )
    overrides = await graph_reader._load_folder_overrides(WORKSPACE, FOLDER)
    tombstoned = {
        entity_id
        for entity_id, override in overrides.items()
        if override.get("deleted")
    }
    out = [entity_id for entity_id in out if entity_id not in tombstoned]

    direct_rows = await graph_reader._load_direct_member_entity_rows(WORKSPACE, FOLDER)
    for row in direct_rows:
        entity_id = str(row.get("entity_id") or "")
        if not entity_id:
            continue
        override = overrides.get(entity_id) or {}
        graph_reader._append_label_match(
            out,
            eid=entity_id,
            labels=(
                row.get("entity_id"),
                row.get("display_name"),
                override.get("display_name"),
            ),
            q=QUERY,
            limit=50,
            tombstoned=tombstoned,
        )

    direct_rows = await graph_reader._load_direct_member_entity_rows(WORKSPACE, FOLDER)
    direct_members = {
        str(row["entity_id"]) for row in direct_rows if row.get("entity_id")
    }
    chunk_to_doc = await graph_reader._load_chunk_to_doc_index(WORKSPACE)
    member_docs = await graph_reader._load_member_docs(WORKSPACE, FOLDER)
    for entity_id, override in overrides.items():
        if len(out) >= 50:
            break
        if entity_id in direct_members or override.get("deleted"):
            continue
        if QUERY.lower() not in str(override.get("display_name") or "").lower():
            continue
        verdict = await graph_reader._entity_mutation_gate(
            WORKSPACE, entity_id, chunk_to_doc, member_docs
        )
        if verdict != graph_reader._GATE_ABSENT:
            graph_reader._append_label_match(
                out,
                eid=entity_id,
                labels=(override.get("display_name"),),
                q=QUERY,
                limit=50,
            )
    return out


async def _live_request() -> list[str]:
    return await graph_reader._search_labels_scoped(WORKSPACE, QUERY, MEMBER_CHUNKS, 50)


async def _one_request(
    request: Callable[[], Awaitable[list[str]]],
) -> tuple[float, _State]:
    state = _State()
    token = _active_state.set(state)
    try:
        started = time.perf_counter()
        labels = await request()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        _active_state.reset(token)
    assert labels == EXPECTED_LABELS
    return elapsed_ms, state


async def _time_requests(
    request: Callable[[], Awaitable[list[str]]],
    *,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    durations: list[float] = []
    read_counts: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one() -> None:
        async with semaphore:
            elapsed_ms, state = await _one_request(request)
            durations.append(elapsed_ms)
            read_counts.append(state.total_reads)

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
        mean_reads=statistics.fmean(read_counts),
        durations_ms=tuple(durations),
        elapsed_seconds=elapsed,
    )


def _patch_graph_reads() -> dict[str, Any]:
    replacements = {
        "_search_member_chunk_labels": _search_member_chunk_labels,
        "_load_folder_overrides": _load_folder_overrides,
        "_load_direct_member_entity_rows": _load_direct_member_entity_rows,
        "_load_chunk_to_doc_index": _load_chunk_to_doc_index,
        "_load_member_docs": _load_member_docs,
        "_entity_mutation_gate": _entity_mutation_gate,
    }
    originals = {name: getattr(graph_reader, name) for name in replacements}
    for name, replacement in replacements.items():
        setattr(graph_reader, name, replacement)
    return originals


def _restore_graph_reads(originals: dict[str, Any]) -> None:
    for name, original in originals.items():
        setattr(graph_reader, name, original)


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entry point: self-normalizing ratio plus deterministic read guard."""
    count = iterations or ITERATIONS
    originals = _patch_graph_reads()
    folder_token = folder_mod._active_folder_id.set(FOLDER)
    try:
        _, baseline_state = await _one_request(_baseline_request)
        _, live_state = await _one_request(_live_request)
        baseline = await _time_requests(_baseline_request, iterations=count)
        optimized = await _time_requests(_live_request, iterations=count)
    finally:
        folder_mod._active_folder_id.reset(folder_token)
        _restore_graph_reads(originals)

    return [
        {
            "name": "folder-scoped graph label search latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "direct-member rows are loaded once per search",
            "kind": "structural",
            "passed": (
                live_state.reads.get("direct_members") == 1
                and live_state.total_reads == 6
                and baseline_state.reads.get("direct_members") == 2
            ),
            "detail": (
                "expected one direct-member read and six total reads with "
                f"identical ordered labels; observed reads={live_state.reads}"
            ),
        },
    ]


def _combine_samples(samples: list[_Sample]) -> _Sample:
    durations = tuple(
        duration for sample in samples for duration in sample.durations_ms
    )
    ordered = sorted(durations)
    elapsed = sum(sample.elapsed_seconds for sample in samples)

    def percentile(percent: float) -> float:
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percent) - 1))
        return ordered[index]

    return _Sample(
        mean_ms=statistics.fmean(durations),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        requests_per_second=len(durations) / elapsed,
        mean_reads=statistics.fmean(sample.mean_reads for sample in samples),
        durations_ms=durations,
        elapsed_seconds=elapsed,
    )


async def _load_test(
    iterations: int, concurrency: int, *, rounds: int = 5
) -> tuple[_Sample, _Sample]:
    """Pool alternating rounds so phase-local host load cannot pick a winner."""
    baseline_samples: list[_Sample] = []
    optimized_samples: list[_Sample] = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.append(
                await _time_requests(
                    _baseline_request,
                    iterations=iterations,
                    concurrency=concurrency,
                )
            )
            optimized_samples.append(
                await _time_requests(
                    _live_request,
                    iterations=iterations,
                    concurrency=concurrency,
                )
            )
        else:
            optimized_samples.append(
                await _time_requests(
                    _live_request,
                    iterations=iterations,
                    concurrency=concurrency,
                )
            )
            baseline_samples.append(
                await _time_requests(
                    _baseline_request,
                    iterations=iterations,
                    concurrency=concurrency,
                )
            )
    return _combine_samples(baseline_samples), _combine_samples(optimized_samples)


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s "
        f"reads/request={sample.mean_reads:.1f}"
    )


async def main() -> None:
    originals = _patch_graph_reads()
    folder_token = folder_mod._active_folder_id.set(FOLDER)
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
            ("peak", 20, 200),
        ):
            baseline, optimized = await _load_test(requests, concurrency)
            _print_sample(f"{label} baseline", baseline)
            _print_sample(f"{label} optimized", optimized)
    finally:
        folder_mod._active_folder_id.reset(folder_token)
        _restore_graph_reads(originals)


if __name__ == "__main__":
    asyncio.run(main())
