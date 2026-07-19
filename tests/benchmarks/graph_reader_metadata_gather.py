"""Bolt micro-benchmark: parallelize the two independent metadata reads in
``graph_reader.read_graph_entities`` / ``read_graph_relations``.

The optimization gathers ``_load_chunk_to_doc_index`` and ``_active_member_docs``
(two independent Memgraph round-trips) instead of serializing them. This harness
exercises the REAL ``read_graph_entities`` against a mocked read session that
sleeps a fixed RTT per query, on the folder-bound path (both loads issue a real
query there). "before" is reproduced by monkeypatching the module's
``asyncio.gather`` to a sequential shim — same function, only the overlap toggled.

Run: .venv/bin/python tests/benchmarks/graph_reader_metadata_gather.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from contextlib import asynccontextmanager

from twindb_lightrag_memgraph.server import graph_reader
from twindb_lightrag_memgraph.server import folder as folder_mod

# --- tunables (simulate a realistic Memgraph deployment) ---------------------
ITERATIONS = 200
RTT_SECONDS = (
    float(__import__("os").environ.get("RTT_MS", "4")) / 1000.0
)  # per-query RTT
ENTITY_COUNT = 400  # nodes returned by the main MATCH
DOC_COUNT = 300  # DocStatus rows feeding the chunk->doc index
WORKSPACE = "benchws"
FOLDER = "f-bench"


class _FakeResult:
    """Async result stub matching the (run -> async-for -> single -> consume)
    surface graph_reader touches."""

    def __init__(self, records):
        self._records = records

    def __aiter__(self):
        async def gen():
            for r in self._records:
                yield r

        return gen()

    async def single(self):
        return self._records[0] if self._records else None

    async def consume(self):
        return None


class _FakeSession:
    def __init__(self, rtt):
        self._rtt = rtt

    async def run(self, query, **params):
        # Simulate the round-trip cost of every query.
        await asyncio.sleep(self._rtt)
        q = query
        if "MEMBER_OF" in q and "collect(" in q:  # _load_member_docs
            return _FakeResult([{"ids": [f"doc-{i}" for i in range(DOC_COUNT)]}])
        if "MEMBER_OF" in q and "chunks_list" in q:  # _load_member_chunks
            return _FakeResult(
                [{"chunks_list": f'["chunk-{i}-a"]'} for i in range(DOC_COUNT)]
            )
        if "doc_id" in q and "chunks_list" in q:  # _load_chunk_to_doc_index
            recs = [
                {"doc_id": f"doc-{i}", "chunks_list": f'["chunk-{i}-a","chunk-{i}-b"]'}
                for i in range(DOC_COUNT)
            ]
            return _FakeResult(recs)
        if q.strip().startswith("MATCH (n:"):  # main entity MATCH
            recs = [
                {
                    "entity_id": f"ent-{i}",
                    "entity_type": "concept",
                    "description": f"desc {i}",
                    "source_id": f"chunk-{i % DOC_COUNT}-a",
                    "display_name": f"Ent {i}",
                    "twin_tags_json": None,
                    "twin_props_json": None,
                }
                for i in range(ENTITY_COUNT)
            ]
            return _FakeResult(recs)
        # direct-member rows / overrides -> empty
        return _FakeResult([])


@asynccontextmanager
async def _fake_read_session():
    yield _FakeSession(RTT_SECONDS)


async def _sequential_gather(*coros):
    """Reproduce the pre-optimization sequential behavior."""
    return [await c for c in coros]


async def _measure(label, *, sequential, iterations=ITERATIONS):
    # Bind the active folder so both metadata loads issue real queries.
    token = folder_mod._active_folder_id.set(FOLDER)
    orig_gather = graph_reader.asyncio.gather
    if sequential:
        graph_reader.asyncio.gather = _sequential_gather
    try:
        # warmup
        for _ in range(5):
            await graph_reader.read_graph_entities(WORKSPACE, max_nodes=ENTITY_COUNT)

        tracemalloc.start()
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = await graph_reader.read_graph_entities(
                WORKSPACE, max_nodes=ENTITY_COUNT
            )
            samples.append((time.perf_counter() - t0) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        graph_reader.asyncio.gather = orig_gather
        folder_mod._active_folder_id.reset(token)

    samples.sort()
    mean = statistics.fmean(samples)
    p95 = samples[int(len(samples) * 0.95) - 1]
    p99 = samples[int(len(samples) * 0.99) - 1]
    reqs = 1000.0 / mean
    print(
        f"{label:9s} mean={mean:7.3f}ms  p50={samples[len(samples)//2]:7.3f}ms  "
        f"p95={p95:7.3f}ms  p99={p99:7.3f}ms  {reqs:6.1f} req/s  "
        f"peak={peak/1e6:5.3f}MB  n_out={len(out)}"
    )
    return mean, p95, p99


class _TrackingSession(_FakeSession):
    """Session that counts concurrent in-flight ``run`` calls, to prove the two
    metadata loads actually overlap on the optimized path."""

    def __init__(self, rtt, tracker):
        super().__init__(rtt)
        self._tracker = tracker

    async def run(self, query, **params):
        self._tracker["in_flight"] += 1
        self._tracker["peak"] = max(self._tracker["peak"], self._tracker["in_flight"])
        try:
            return await super().run(query, **params)
        finally:
            self._tracker["in_flight"] -= 1


async def _observe_overlap() -> int:
    """Run the real (non-sequential) path once and return the peak number of
    concurrent read sessions. >=2 means the gathered metadata loads overlapped;
    1 means they were serialized (optimization reverted)."""
    tracker = {"in_flight": 0, "peak": 0}

    @asynccontextmanager
    async def _tracking_read_session():
        yield _TrackingSession(RTT_SECONDS, tracker)

    orig_session = graph_reader.get_read_session
    graph_reader.get_read_session = _tracking_read_session
    token = folder_mod._active_folder_id.set(FOLDER)
    try:
        await graph_reader.read_graph_entities(WORKSPACE, max_nodes=ENTITY_COUNT)
    finally:
        graph_reader.get_read_session = orig_session
        folder_mod._active_folder_id.reset(token)
    return tracker["peak"]


async def measure(iterations: int | None = None) -> list[dict]:
    """CI entry point: one ratio case + one structural case.

    See ``tests/benchmarks/_perf_contract`` for the contract. Saves and restores
    ``graph_reader.get_read_session`` so the pytest process is left clean (the
    CLI ``main()`` keeps its patch — the process exits right after).
    """
    n = iterations or ITERATIONS
    orig_session = graph_reader.get_read_session
    graph_reader.get_read_session = _fake_read_session
    try:
        before = await _measure("BEFORE", sequential=True, iterations=n)
        after = await _measure("AFTER", sequential=False, iterations=n)
        peak = await _observe_overlap()
    finally:
        graph_reader.get_read_session = orig_session
    return [
        {
            "name": "graph_reader.read_graph_entities (parallel metadata reads)",
            "kind": "ratio",
            "baseline_ms": before[0],
            "optimized_ms": after[0],
        },
        {
            "name": "graph_reader metadata reads overlap (>=2 concurrent reads)",
            "kind": "structural",
            "passed": peak >= 2,
            "detail": f"peak concurrent read sessions={peak} (expected >=2)",
        },
    ]


async def main():
    # patch the read session used by every loader
    graph_reader.get_read_session = _fake_read_session  # type: ignore[assignment]

    before = await _measure("BEFORE", sequential=True)
    after = await _measure("AFTER", sequential=False)

    gain = (before[0] - after[0]) / before[0] * 100.0
    thr = (1000.0 / after[0]) / (1000.0 / before[0]) - 1.0
    print("-" * 78)
    print(
        f"Gain: {gain:5.1f}% latency reduced | +{thr*100:5.1f}% throughput | "
        f"p95 {before[1]:.3f}->{after[1]:.3f}ms | p99 {before[2]:.3f}->{after[2]:.3f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
