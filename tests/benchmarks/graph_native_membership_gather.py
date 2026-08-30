"""Bolt micro-benchmark: parallelize the five independent membership/override
reads in ``graph_reader.read_graph_native``.

``read_graph_native`` is the body behind ``GET /graph/entities`` and
``GET /graph/relations`` — the Graph tab calls both, so the folder-bound path
runs twice per graph render. Before the optimization it serialized five
independent Memgraph round-trips (``_load_chunk_to_doc_index``,
``_load_member_docs``, ``_load_folder_overrides``,
``_load_folder_rel_overrides``, ``_load_direct_member_entity_rows``); each opens
its own read session, consumes only ``(workspace, folder)``, and absorbs its own
error into its established fallback. Those fallbacks are not uniformly
fail-closed: override-load failures can re-surface the base record. None of the
five reads can observe another's result.

The harness drives the REAL ``read_graph_native`` against a fake read session
that sleeps a fixed RTT per query. "before" is reproduced by monkeypatching the
module's ``asyncio.gather`` to a sequential shim — same function body, only the
overlap toggled off — which is also what the structural case detects.

``READ_CAPACITY`` models the Bolt read pool (``MEMGRAPH_READ_POOL_SIZE``,
default 20) so the load test shows the fan-out queueing on the pool rather than
pretending sessions are free.

Run standalone with::

    uv run python tests/benchmarks/graph_native_membership_gather.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from twindb_lightrag_memgraph.server import folder as folder_mod
from twindb_lightrag_memgraph.server import graph_reader

# --- tunables (simulate a realistic Memgraph deployment) ---------------------
ITERATIONS = 80
RTT_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
ENTITY_COUNT = 300  # nodes returned by the native knowledge-graph call
DOC_COUNT = 300  # DocStatus rows feeding the chunk->doc index
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))
WORKSPACE = "benchws"
FOLDER = "f-bench"

# The five membership reads overlap, but deliberately NOT all at once: the read
# pool is shared with every other route, so `read_graph_native` caps its own
# burst (`graph_reader._membership_fanout()`, default 2). The structural case
# asserts BOTH halves of that contract — the overlap exists (so a revert to the
# serial body is caught) and it never exceeds the cap (so a well-meaning
# "just gather all five" is caught too). See
# tests/benchmarks/shared_read_pool_interference.py for why the cap exists.
EXPECTED_FANOUT = graph_reader._membership_fanout()


class _FakeResult:
    """Async result stub matching the surface graph_reader touches."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __aiter__(self):
        async def gen():
            for record in self._records:
                yield record

        return gen()

    async def single(self):
        return self._records[0] if self._records else None

    async def consume(self):
        return None


class _SessionMeter:
    """Track concurrent read sessions so the structural case can see fan-out.

    Also records each read's [start, end) interval, which yields the metric that
    actually matters here and — unlike wall-clock — does not move with machine
    load: the **round-trip depth**, i.e. the longest chain of reads that had to
    happen one after another. Serializing five reads has depth 5; overlapping
    them two at a time has depth 3. That is the latency the optimization
    removes, expressed deterministically.
    """

    def __init__(self, capacity: int) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.queries = 0
        self.intervals: list[tuple[float, float]] = []
        self._capacity = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def session(self):
        # The Bolt driver blocks on an exhausted pool; model that back-pressure
        # instead of pretending concurrent sessions are free.
        async with self._capacity:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            started = time.perf_counter()
            try:
                yield _FakeSession(self)
            finally:
                self.in_flight -= 1
                self.intervals.append((started, time.perf_counter()))

    def round_trip_depth(self) -> int:
        """Longest chain of strictly non-overlapping reads.

        Greedy over intervals sorted by end time: each time a read starts after
        the current chain's last read ended, the chain deepens. Two reads that
        overlap at all cost one level between them, which is precisely what
        gathering buys.
        """
        depth = 0
        chain_end = float("-inf")
        for start, end in sorted(self.intervals, key=lambda iv: iv[1]):
            if start >= chain_end:
                depth += 1
                chain_end = end
        return depth


class _FakeSession:
    def __init__(self, meter: _SessionMeter) -> None:
        self._meter = meter

    async def run(self, query: str, **_params: Any) -> _FakeResult:
        self._meter.queries += 1
        await asyncio.sleep(RTT_SECONDS)
        if "MEMBER_OF" in query and "collect(" in query:  # _load_member_docs
            return _FakeResult([{"ids": [f"doc-{i}" for i in range(DOC_COUNT)]}])
        if "doc_id" in query and "chunks_list" in query:  # chunk->doc index
            return _FakeResult(
                [
                    {
                        "doc_id": f"doc-{i}",
                        "chunks_list": f'["chunk-{i}-a","chunk-{i}-b"]',
                    }
                    for i in range(DOC_COUNT)
                ]
            )
        if "-[r:DIRECTED]->" in query:  # stored rows for the visible endpoints
            return _FakeResult(
                [
                    {
                        "source_id": f"ent-{i}",
                        "target_id": f"ent-{(i + 1) % ENTITY_COUNT}",
                        "relation_id": f"rel-{i}",
                        "keywords": "k",
                        "weight": 1.0,
                        "chunk_source_id": f"chunk-{i % DOC_COUNT}-a",
                        "twin_folder_json": None,
                        "twin_props_json": None,
                    }
                    for i in range(ENTITY_COUNT)
                ]
            )
        if "GraphRelOverride" in query:  # rel overrides -> one live overlay
            return _FakeResult(
                [
                    {
                        "src": "ent-0",
                        "tgt": "ent-1",
                        "keywords": "overridden",
                        "weight": 9.0,
                        "twin_props_json": None,
                        "deleted": None,
                    }
                ]
            )
        # folder entity overrides / direct-member rows
        return _FakeResult([])


class _FakeNode:
    def __init__(self, index: int) -> None:
        self.id = f"ent-{index}"
        self.labels = [f"ent-{index}"]
        self.properties = {
            "entity_id": f"ent-{index}",
            "entity_type": "concept",
            "description": f"desc {index}",
            "source_id": f"chunk-{index % DOC_COUNT}-a",
            "display_name": f"Ent {index}",
        }


class _FakeEdge:
    def __init__(self, index: int) -> None:
        self.id = f"rel-{index}"
        self.type = "DIRECTED"
        self.source = f"ent-{index}"
        self.target = f"ent-{(index + 1) % ENTITY_COUNT}"
        self.properties = {
            "keywords": "k",
            "weight": 1.0,
            "source_id": f"chunk-{index % DOC_COUNT}-a",
        }


class _FakeKnowledgeGraph:
    def __init__(self) -> None:
        self.nodes = [_FakeNode(i) for i in range(ENTITY_COUNT)]
        self.edges = [_FakeEdge(i) for i in range(ENTITY_COUNT)]


class _Rag:
    async def get_knowledge_graph(self, **_kwargs: Any) -> _FakeKnowledgeGraph:
        await asyncio.sleep(RTT_SECONDS)
        return _FakeKnowledgeGraph()


class _SequentialAsyncio:
    """``asyncio`` proxy whose ``gather`` runs its coroutines serially.

    Bound onto ``graph_reader.asyncio`` only — patching ``asyncio.gather``
    itself would rebind the shared module attribute and serialize this
    harness's own concurrent load driver, fabricating a baseline that runs at
    concurrency 1.
    """

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def gather(self, *coros, **_kwargs):
        """Reproduce the pre-optimization serialized behaviour."""
        return [await coro for coro in coros]


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    max_in_flight: int
    entity_count: int


async def _run_once(meter: _SessionMeter) -> tuple[list, list]:
    result = await graph_reader.read_graph_native(
        _Rag(),
        WORKSPACE,
        node_label="*",
        max_nodes=ENTITY_COUNT,
    )
    assert result is not None, "native graph read returned unavailable"
    return result


@asynccontextmanager
async def _bound(sequential: bool, meter: _SessionMeter):
    """Bind the folder + fake read session, optionally toggling the overlap off."""
    token = folder_mod._active_folder_id.set(FOLDER)
    original_session = graph_reader.get_read_session
    original_asyncio = graph_reader.asyncio
    graph_reader.get_read_session = meter.session
    if sequential:
        graph_reader.asyncio = _SequentialAsyncio(original_asyncio)
    try:
        yield
    finally:
        graph_reader.get_read_session = original_session
        graph_reader.asyncio = original_asyncio
        folder_mod._active_folder_id.reset(token)


async def _time_requests(
    *,
    sequential: bool,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    meter = _SessionMeter(READ_CAPACITY)
    durations: list[float] = []
    entity_counts: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with _bound(sequential, meter):
        # warmup, outside the measured window
        for _ in range(3):
            await _run_once(meter)

        async def run_one() -> None:
            async with semaphore:
                started = time.perf_counter()
                entities, _relations = await _run_once(meter)
                durations.append((time.perf_counter() - started) * 1000)
                entity_counts.append(len(entities))

        meter.max_in_flight = 0
        started = time.perf_counter()
        await asyncio.gather(*(run_one() for _ in range(iterations)))
        elapsed = time.perf_counter() - started

    ordered = sorted(durations)

    def percentile(percent: float) -> float:
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percent) - 1))
        return ordered[index]

    assert len(set(entity_counts)) == 1, "entity count drifted between iterations"
    return _Sample(
        mean_ms=statistics.mean(durations),
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        requests_per_second=iterations / elapsed,
        max_in_flight=meter.max_in_flight,
        entity_count=entity_counts[0],
    )


def _projection(entities: list, relations: list) -> tuple:
    """Order-sensitive FULL comparison of a native graph result.

    Both halves are plain JSON-able dicts, so compare them whole rather than
    hand-picking fields: an earlier version sampled four keys and one of them
    (``label``) does not exist on an entity at all — entities carry ``name``
    (see the TypeScript ``GraphEntity`` contract), so that term compared
    ``None == None`` on every row and silently weakened the guard.
    """
    return (tuple(entities), tuple(relations))


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS

    # --- parity: identical ordered output, both toggles -----------------
    parity_meter = _SessionMeter(READ_CAPACITY)
    async with _bound(True, parity_meter):
        baseline_entities, baseline_relations = await _run_once(parity_meter)
    baseline_fanout = parity_meter.max_in_flight
    baseline_queries = parity_meter.queries
    baseline_depth = parity_meter.round_trip_depth()

    live_meter = _SessionMeter(READ_CAPACITY)
    async with _bound(False, live_meter):
        live_entities, live_relations = await _run_once(live_meter)
    live_fanout = live_meter.max_in_flight
    live_queries = live_meter.queries
    live_depth = live_meter.round_trip_depth()

    baseline = await _time_requests(sequential=True, iterations=count)
    optimized = await _time_requests(sequential=False, iterations=count)

    return [
        {
            "name": "native graph membership load latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "membership reads overlap, bounded by the fan-out cap",
            "kind": "structural",
            "passed": (
                live_fanout > 1
                and live_fanout <= EXPECTED_FANOUT
                and baseline_fanout == 1
                and live_queries == baseline_queries
            ),
            "detail": (
                f"expected >1 and <={EXPECTED_FANOUT} concurrent read sessions "
                f"on the live path (serialized baseline 1) with an identical "
                f"query count; observed live={live_fanout} "
                f"baseline={baseline_fanout}, queries live={live_queries} "
                f"baseline={baseline_queries}"
            ),
        },
        {
            # Load-independent statement of the win. Wall-clock on a shared
            # workstation is not reproducible (observed +55%, +25% and -13% for
            # the same code within minutes at load average 122); read-ordering
            # is. The serial body needed `baseline_depth` sequential round-trips
            # before it could project; the bounded gather needs fewer.
            "name": "round-trip depth reduced",
            "kind": "structural",
            "passed": live_depth < baseline_depth,
            "detail": (
                f"expected fewer sequential read round-trips than the serial "
                f"body; observed depth {baseline_depth} -> {live_depth} "
                f"(fan-out cap {EXPECTED_FANOUT})"
            ),
        },
        {
            "name": "graph projection unchanged",
            "kind": "structural",
            "passed": (
                _projection(live_entities, live_relations)
                == _projection(baseline_entities, baseline_relations)
                and len(live_entities) == ENTITY_COUNT
            ),
            "detail": (
                "expected identical ordered entities/relations; observed "
                f"{len(live_entities)} vs {len(baseline_entities)} entities, "
                f"{len(live_relations)} vs {len(baseline_relations)} relations"
            ),
        },
    ]


def _pool(samples: list[_Sample]) -> _Sample:
    """Pool alternating rounds so one load spike can't own a result."""
    return _Sample(
        mean_ms=statistics.mean(s.mean_ms for s in samples),
        p50_ms=statistics.mean(s.p50_ms for s in samples),
        p95_ms=statistics.mean(s.p95_ms for s in samples),
        p99_ms=statistics.mean(s.p99_ms for s in samples),
        requests_per_second=statistics.mean(s.requests_per_second for s in samples),
        max_in_flight=max(s.max_in_flight for s in samples),
        entity_count=samples[0].entity_count,
    )


async def _load_test(
    *, iterations: int, concurrency: int, rounds: int = 5
) -> tuple[_Sample, _Sample]:
    """Alternate baseline/optimized rounds and pool them."""
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
        f"max_sessions={sample.max_in_flight}"
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
        ("peak", 20, 160),
    ):
        baseline, optimized = await _load_test(
            iterations=requests, concurrency=concurrency
        )
        _print_sample(f"{label} baseline", baseline)
        _print_sample(f"{label} optimized", optimized)


if __name__ == "__main__":
    asyncio.run(main())
