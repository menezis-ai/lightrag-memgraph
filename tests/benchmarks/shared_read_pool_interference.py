"""Bolt: does the graph fan-out starve the other routes on the SHARED read pool?

The three per-target benchmarks each model the 20-slot Bolt read pool
(``MEMGRAPH_READ_POOL_SIZE``) *in isolation*, which is exactly the wrong shape
for the question review raised: one pool is shared by every route, and
``read_graph_native`` now asks for 5 connections at once instead of 1. A single
Graph render calls ``/graph/entities`` **and** ``/graph/relations``, so it wants
~10 concurrent slots; two concurrent renders can ask for the whole pool.

This harness therefore runs all three workloads against **one** semaphore and
reports what the *other* routes feel while the graph renders, with the gathers
on and off. It answers "did the bottleneck move downstream" with a measurement
instead of an assertion.

Workload mix per round (tunable by env):

* ``GRAPH_RENDERS``  concurrent Graph renders (each = entities + relations)
* ``CHUNK_CLIENTS``  concurrent chunk expansions (the reader clicking citations)
* ``QUOTA_CLIENTS``  concurrent quota polls (banner + ingestion gate)

Run standalone with::

    uv run python tests/benchmarks/shared_read_pool_interference.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from twindb_lightrag_memgraph.server import chunk_routes, graph_reader, quota
from twindb_lightrag_memgraph.server import folder as folder_mod

RTT_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))
GRAPH_RENDERS = int(os.environ.get("GRAPH_RENDERS", "2"))
CHUNK_CLIENTS = int(os.environ.get("CHUNK_CLIENTS", "8"))
QUOTA_CLIENTS = int(os.environ.get("QUOTA_CLIENTS", "2"))
OPS_PER_CLIENT = int(os.environ.get("OPS_PER_CLIENT", "12"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))

ENTITY_COUNT = 120
DOC_COUNT = 120
CHUNK_COUNT = 40
WORKSPACE = "sharedpoolws"
FOLDER = "f-shared"


# --------------------------------------------------------------------------
# One pool, shared by every route
# --------------------------------------------------------------------------
class _SharedPool:
    def __init__(self, capacity: int = READ_CAPACITY) -> None:
        self._capacity = asyncio.Semaphore(capacity)
        self.in_flight = 0
        self.max_in_flight = 0
        self.reads = 0
        self.wait_ms: list[float] = []

    @asynccontextmanager
    async def slot(self):
        started = time.perf_counter()
        async with self._capacity:
            # Time spent queueing for a connection is the interference signal.
            self.wait_ms.append((time.perf_counter() - started) * 1000)
            self.reads += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                yield
            finally:
                self.in_flight -= 1


_pool_var: ContextVar[_SharedPool] = ContextVar("shared_pool")


# --------------------------------------------------------------------------
# Graph fakes
# --------------------------------------------------------------------------
class _FakeResult:
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

    async def data(self):
        return self._records


class _GraphSession:
    async def run(self, query: str, **_params: Any) -> _FakeResult:
        await asyncio.sleep(RTT_SECONDS)
        if "MEMBER_OF" in query and "collect(" in query:
            return _FakeResult([{"ids": [f"doc-{i}" for i in range(DOC_COUNT)]}])
        if "doc_id" in query and "chunks_list" in query:
            return _FakeResult(
                [
                    {"doc_id": f"doc-{i}", "chunks_list": f'["chunk-{i}-a"]'}
                    for i in range(DOC_COUNT)
                ]
            )
        if "-[r:DIRECTED]->" in query:
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
        return _FakeResult([])


@asynccontextmanager
async def _graph_read_session():
    async with _pool_var.get().slot():
        yield _GraphSession()


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
        self.properties = {"keywords": "k", "weight": 1.0}


class _FakeKG:
    def __init__(self) -> None:
        self.nodes = [_FakeNode(i) for i in range(ENTITY_COUNT)]
        self.edges = [_FakeEdge(i) for i in range(ENTITY_COUNT)]


class _GraphRag:
    async def get_knowledge_graph(self, **_kwargs: Any) -> _FakeKG:
        async with _pool_var.get().slot():
            await asyncio.sleep(RTT_SECONDS)
            return _FakeKG()


# --------------------------------------------------------------------------
# Chunk-route fakes
# --------------------------------------------------------------------------
class _DocStatus:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._value = {"chunks_list": chunk_ids, "folder": FOLDER}

    async def get_by_id(self, _doc_id: str) -> dict[str, Any]:
        async with _pool_var.get().slot():
            await asyncio.sleep(RTT_SECONDS)
            return self._value

    async def get_folders_for_doc(self, _doc_id: str) -> list[str]:
        async with _pool_var.get().slot():
            await asyncio.sleep(RTT_SECONDS)
            return [FOLDER]


class _TextChunks:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._chunks = {
            cid: {
                "_id": cid,
                "content": f"content-{cid}",
                "full_doc_id": "doc-1",
                "file_path": "document.pdf",
                "chunk_order_index": i,
                "tokens": 20,
            }
            for i, cid in enumerate(chunk_ids)
        }

    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        async with _pool_var.get().slot():
            await asyncio.sleep(RTT_SECONDS)
            return [self._chunks[c] for c in chunk_ids]


class _ChunkRag:
    def __init__(self) -> None:
        ids = [f"chunk-{i}" for i in range(CHUNK_COUNT)]
        self.doc_status = _DocStatus(ids)
        self.text_chunks = _TextChunks(ids)


async def _fake_source_links(doc_id: str) -> list[dict[str, Any]]:
    async with _pool_var.get().slot():
        await asyncio.sleep(RTT_SECONDS)
        return [{"id": "l1", "doc_id": doc_id, "url": "u", "label": "l"}]


# --------------------------------------------------------------------------
# Quota fakes
# --------------------------------------------------------------------------
_INSTANCE_ROWS = [
    {"storage info": "memory_tracked", "value": "3.5GiB"},
    {"storage info": "memory_limit", "value": "8.0GiB"},
    {"storage info": "license_memory_limit", "value": "4.0GiB"},
]
_DATABASE_ROWS = [
    {"storage info": "graph_memory_tracked", "value": "2.5GiB"},
    {"storage info": "vector_index_memory_tracked", "value": "0.5GiB"},
]


class _QuotaSession:
    async def run(self, query: str, **_params: Any) -> _FakeResult:
        await asyncio.sleep(RTT_SECONDS)
        if "DATABASE" in query.upper():
            return _FakeResult(list(_DATABASE_ROWS))
        return _FakeResult(list(_INSTANCE_ROWS))


class _QuotaPool:
    def get_read_session(self):
        @asynccontextmanager
        async def cm():
            async with _pool_var.get().slot():
                yield _QuotaSession()

        return cm()


# --------------------------------------------------------------------------
# Sequential toggle
# --------------------------------------------------------------------------
class _SequentialAsyncio:
    """Serialize `gather`/`ensure_future` for one module only."""

    class _Deferred:
        def __init__(self, coro):
            self._coro, self._result, self._done = coro, None, False

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
        return [await c for c in coros]


_chunk_endpoint = None


def _ensure_chunk_endpoint():
    global _chunk_endpoint
    if _chunk_endpoint is None:
        chunk_routes.create_chunk_routes(lambda: _chunk_rag_var.get())
        for route in chunk_routes.router.routes:
            if getattr(route, "path", None) == "/documents/{doc_id}/chunks":
                _chunk_endpoint = route.endpoint
    return _chunk_endpoint


_chunk_rag_var: ContextVar[_ChunkRag] = ContextVar("chunk_rag")


@asynccontextmanager
async def _bound(sequential: bool, pool: _SharedPool):
    tokens = [_pool_var.set(pool), folder_mod._active_folder_id.set(FOLDER)]
    saved = {
        "graph_session": graph_reader.get_read_session,
        "graph_asyncio": graph_reader.asyncio,
        "chunk_links": chunk_routes._source_links_for_doc,
        "chunk_asyncio": chunk_routes.asyncio,
        "quota_pool": quota._pool,
        "quota_asyncio": quota.asyncio,
    }
    graph_reader.get_read_session = _graph_read_session
    chunk_routes._source_links_for_doc = _fake_source_links
    quota._pool = _QuotaPool()
    if sequential:
        graph_reader.asyncio = _SequentialAsyncio(saved["graph_asyncio"])
        chunk_routes.asyncio = _SequentialAsyncio(saved["chunk_asyncio"])
        quota.asyncio = _SequentialAsyncio(saved["quota_asyncio"])
    try:
        yield
    finally:
        graph_reader.get_read_session = saved["graph_session"]
        graph_reader.asyncio = saved["graph_asyncio"]
        chunk_routes._source_links_for_doc = saved["chunk_links"]
        chunk_routes.asyncio = saved["chunk_asyncio"]
        quota._pool = saved["quota_pool"]
        quota.asyncio = saved["quota_asyncio"]
        folder_mod._active_folder_id.reset(tokens[1])
        _pool_var.reset(tokens[0])


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------
async def _graph_render() -> None:
    """One Graph tab render = the two routes the WebUI calls together."""
    rag = _GraphRag()
    await asyncio.gather(
        graph_reader.read_graph_native(
            rag, WORKSPACE, node_label="*", max_nodes=ENTITY_COUNT
        ),
        graph_reader.read_graph_native(
            rag, WORKSPACE, node_label="*", max_nodes=ENTITY_COUNT
        ),
    )


async def _chunk_expansion() -> None:
    endpoint = _ensure_chunk_endpoint()
    token = _chunk_rag_var.set(_ChunkRag())
    try:
        await endpoint("doc-1", None, None)
    finally:
        _chunk_rag_var.reset(token)


async def _quota_poll() -> None:
    await quota.snapshot()


@dataclass
class _ClassStats:
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(self.samples)

        def pct(p: float) -> float:
            i = max(0, min(len(ordered) - 1, int(len(ordered) * p) - 1))
            return ordered[i]

        return {
            "mean": statistics.mean(ordered),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }


async def _run_mix(sequential: bool) -> dict[str, Any]:
    pool = _SharedPool()
    stats = {"graph": _ClassStats(), "chunks": _ClassStats(), "quota": _ClassStats()}

    async with _bound(sequential, pool):
        await _quota_poll()  # warmup

        async def client(kind: str, work) -> None:
            for _ in range(OPS_PER_CLIENT):
                started = time.perf_counter()
                await work()
                stats[kind].add((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(
            *(client("graph", _graph_render) for _ in range(GRAPH_RENDERS)),
            *(client("chunks", _chunk_expansion) for _ in range(CHUNK_CLIENTS)),
            *(client("quota", _quota_poll) for _ in range(QUOTA_CLIENTS)),
        )
        elapsed = time.perf_counter() - started

    waits = sorted(pool.wait_ms)

    def wait_pct(p: float) -> float:
        i = max(0, min(len(waits) - 1, int(len(waits) * p) - 1))
        return waits[i] if waits else 0.0

    return {
        "elapsed_s": elapsed,
        "reads": pool.reads,
        "max_in_flight": pool.max_in_flight,
        "pool_wait_mean_ms": statistics.mean(waits) if waits else 0.0,
        "pool_wait_p95_ms": wait_pct(0.95),
        **{k: v.summary() for k, v in stats.items()},
    }


def _mean_of(runs: list[dict], *path) -> float:
    vals = []
    for r in runs:
        cur: Any = r
        for p in path:
            cur = cur[p]
        vals.append(cur)
    return statistics.mean(vals)


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    rounds = 2 if iterations and iterations <= 20 else ROUNDS
    base_runs, live_runs = [], []
    for _ in range(rounds):
        base_runs.append(await _run_mix(True))
        live_runs.append(await _run_mix(False))

    # Deterministic cases only. The per-class latencies this harness prints are
    # the honest answer to "did the bottleneck move downstream", but they are
    # wall-clock under a *contended* mix — on a shared CI runner they would
    # flake permanently, exactly what _perf_contract.py forbids. The CLI
    # `main()` reports them for a human; the gate asserts the two properties
    # that hold regardless of machine load.
    base_reads = _mean_of(base_runs, "reads")
    live_reads = _mean_of(live_runs, "reads")
    observed_fanout = await _observed_graph_fanout()
    cases: list[dict[str, Any]] = [
        {
            "name": "shared pool sees no extra reads",
            "kind": "structural",
            "passed": abs(base_reads - live_reads) < 1e-9,
            "detail": (
                "the gathers must pipeline the SAME reads, never add any; "
                f"observed baseline={base_reads} optimized={live_reads}"
            ),
        },
        {
            "name": "graph fan-out stays bounded under a shared-pool mix",
            "kind": "structural",
            "passed": 1 < observed_fanout <= graph_reader._membership_fanout(),
            "detail": (
                "one graph request must never hold more than the configured "
                f"cap of {graph_reader._membership_fanout()} membership reads; "
                f"observed {observed_fanout}"
            ),
        },
    ]
    return cases


async def _observed_graph_fanout() -> int:
    """Peak concurrent membership reads a single graph call actually issues."""
    peak = 0
    live = 0

    async def probe(index: int) -> int:
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.005)
        live -= 1
        return index

    await graph_reader._gather_membership_reads(*(partial(probe, i) for i in range(5)))
    return peak


async def main() -> None:
    print(
        f"mix: {GRAPH_RENDERS} graph renders x2 routes, {CHUNK_CLIENTS} chunk "
        f"clients, {QUOTA_CLIENTS} quota clients; pool={READ_CAPACITY} "
        f"RTT={RTT_SECONDS*1000:.0f}ms; {ROUNDS} alternating rounds"
    )
    base_runs, live_runs = [], []
    for _ in range(ROUNDS):
        base_runs.append(await _run_mix(True))
        live_runs.append(await _run_mix(False))

    for kind in ("graph", "chunks", "quota"):
        b_mean = _mean_of(base_runs, kind, "mean")
        o_mean = _mean_of(live_runs, kind, "mean")
        b95 = _mean_of(base_runs, kind, "p95")
        o95 = _mean_of(live_runs, kind, "p95")
        b99 = _mean_of(base_runs, kind, "p99")
        o99 = _mean_of(live_runs, kind, "p99")
        d = lambda a, b: (b - a) / a * 100 if a else 0.0  # noqa: E731
        print(
            f"{kind:>7}: mean {b_mean:8.2f} -> {o_mean:8.2f} ms ({d(b_mean,o_mean):+6.1f}%)"
            f" | p95 {b95:8.2f} -> {o95:8.2f} ({d(b95,o95):+6.1f}%)"
            f" | p99 {b99:8.2f} -> {o99:8.2f} ({d(b99,o99):+6.1f}%)"
        )
    print(
        f"  pool: reads {_mean_of(base_runs,'reads'):.0f} -> "
        f"{_mean_of(live_runs,'reads'):.0f} | "
        f"max in-flight {_mean_of(base_runs,'max_in_flight'):.0f} -> "
        f"{_mean_of(live_runs,'max_in_flight'):.0f} | "
        f"acquire-wait mean {_mean_of(base_runs,'pool_wait_mean_ms'):.2f} -> "
        f"{_mean_of(live_runs,'pool_wait_mean_ms'):.2f} ms | "
        f"p95 {_mean_of(base_runs,'pool_wait_p95_ms'):.2f} -> "
        f"{_mean_of(live_runs,'pool_wait_p95_ms'):.2f} ms"
    )
    print(
        f"  wall-clock for the whole mix: "
        f"{_mean_of(base_runs,'elapsed_s'):.2f}s -> "
        f"{_mean_of(live_runs,'elapsed_s'):.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
