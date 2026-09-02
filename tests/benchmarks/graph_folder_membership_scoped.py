"""Bolt micro-benchmark: member-scoped folder membership read on the
folder-bound graph paths (``graph_reader._load_folder_membership``).

Every folder-bound graph path — ``read_graph_native`` behind
``GET /graph/entities`` and ``GET /graph/relations`` (two calls per Graph
render), ``search_graph_labels`` behind the search box (per keystroke) and
``_member_context`` behind every gated PATCH/POST/DELETE — needs two facts about
the active folder: which documents are members, and which chunk belongs to
which member document. Before this change the second fact came from a FULL
``DocStatus_{ws}`` scan (``_load_chunk_to_doc_index``: every row of the
workspace streamed, every ``chunks_list`` JSON payload parsed in Python) and the
first from a separate read (``_load_member_docs``) — although the index is only
ever consulted through ``cd.get(chunk) in member_docs``, so every entry built
for a non-member document was discarded work. ``_load_folder_membership`` reads
the member docs' rows once and derives both facts (plus the member chunk set).

The harness drives the REAL ``read_graph_native`` against a fake read session
that sleeps a fixed RTT per query and streams realistic ``chunks_list`` JSON
payloads. "before" is reproduced by binding ``_load_folder_membership`` to the
pre-optimization composition, which still lives in the module: the unscoped
``_load_chunk_to_doc_index`` (kept for the global read) plus
``_load_member_docs`` (kept for the flat readers). Only that seam differs
between the two runs; the request body is the same code.

Wire-transfer cost is NOT modelled: on a live Memgraph every extra row of the
full scan is also serialized over Bolt, so the gain measured here is a floor.
``READ_CAPACITY`` models the Bolt read pool (``MEMGRAPH_READ_POOL_SIZE``,
default 20) so the load test shows back-pressure rather than free sessions.

Run standalone with::

    uv run python tests/benchmarks/graph_folder_membership_scoped.py
"""

from __future__ import annotations

import asyncio
import json
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
DOC_COUNT = int(os.environ.get("DOC_COUNT", "2000"))  # DocStatus rows, whole KB
MEMBER_DOCS = int(os.environ.get("MEMBER_DOCS", "200"))  # of which in the folder
CHUNKS_PER_DOC = int(os.environ.get("CHUNKS_PER_DOC", "20"))
ENTITY_COUNT = 300  # nodes returned by the native knowledge-graph call
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))
WORKSPACE = "benchws"
FOLDER = "f-bench"

assert MEMBER_DOCS <= DOC_COUNT


def _chunk_id(doc: int, index: int) -> str:
    # LightRAG chunk ids are ``chunk-`` + 32 hex chars; keep the payload the
    # same width so the JSON parse cost is representative.
    return f"chunk-{doc:06d}{index:02d}{'0' * 24}"


# Pre-rendered DocStatus payloads: what the fake Bolt result streams back.
_CHUNKS_JSON = [
    json.dumps([_chunk_id(doc, j) for j in range(CHUNKS_PER_DOC)])
    for doc in range(DOC_COUNT)
]
_FULL_SCAN_ROWS = [
    {"doc_id": f"doc-{doc}", "chunks_list": _CHUNKS_JSON[doc]}
    for doc in range(DOC_COUNT)
]
_MEMBERSHIP_ROWS = _FULL_SCAN_ROWS[:MEMBER_DOCS]
_MEMBER_DOC_IDS = [f"doc-{doc}" for doc in range(MEMBER_DOCS)]

# Query shapes, classified so the structural case can count them.
FULL_SCAN = "full_scan"  # unscoped DocStatus scan (_load_chunk_to_doc_index)
MEMBERSHIP = "membership"  # member-scoped rows (_load_folder_membership)
MEMBER_DOCS_COLLECT = "member_docs"  # collect(d.id) (_load_member_docs)
OTHER = "other"


def _classify(query: str) -> str:
    if "MEMBER_OF" in query and "collect(" in query:
        return MEMBER_DOCS_COLLECT
    if "MEMBER_OF" in query and "doc_id" in query and "chunks_list" in query:
        return MEMBERSHIP
    if "DocStatus_" in query and "chunks_list" in query:
        return FULL_SCAN
    return OTHER


class _FakeResult:
    """Async result stub matching the surface graph_reader touches."""

    def __init__(
        self, records: list[dict[str, Any]], meter: "_SessionMeter", kind: str
    ) -> None:
        self._records = records
        self._meter = meter
        self._kind = kind

    def __aiter__(self):
        async def gen():
            for record in self._records:
                self._meter.rows[self._kind] = self._meter.rows.get(self._kind, 0) + 1
                yield record

        return gen()

    async def single(self):
        return self._records[0] if self._records else None

    async def consume(self):
        return None


class _SessionMeter:
    """Count read sessions, classify queries, and count rows streamed."""

    def __init__(self, capacity: int) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.queries: dict[str, int] = {}
        self.rows: dict[str, int] = {}  # rows streamed per query kind
        self._capacity = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def session(self):
        async with self._capacity:  # the Bolt driver blocks on an exhausted pool
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                yield _FakeSession(self)
            finally:
                self.in_flight -= 1

    def reset_counters(self) -> None:
        self.queries = {}
        self.rows = {}
        self.max_in_flight = 0


class _FakeSession:
    def __init__(self, meter: _SessionMeter) -> None:
        self._meter = meter

    async def run(self, query: str, **_params: Any) -> _FakeResult:
        kind = _classify(query)
        self._meter.queries[kind] = self._meter.queries.get(kind, 0) + 1
        await asyncio.sleep(RTT_SECONDS)
        if kind == MEMBER_DOCS_COLLECT:
            return _FakeResult([{"ids": list(_MEMBER_DOC_IDS)}], self._meter, kind)
        if kind == MEMBERSHIP:
            return _FakeResult(_MEMBERSHIP_ROWS, self._meter, kind)
        if kind == FULL_SCAN:
            return _FakeResult(_FULL_SCAN_ROWS, self._meter, kind)
        if "-[r:DIRECTED]->" in query:  # stored rows for the visible endpoints
            return _FakeResult(
                [
                    {
                        "source_id": f"ent-{i}",
                        "target_id": f"ent-{(i + 1) % ENTITY_COUNT}",
                        "relation_id": f"rel-{i}",
                        "keywords": "k",
                        "weight": 1.0,
                        "chunk_source_id": _chunk_id(i % DOC_COUNT, 0),
                        "twin_folder_json": None,
                        "twin_props_json": None,
                    }
                    for i in range(ENTITY_COUNT)
                ],
                self._meter,
                kind,
            )
        # folder entity/relation overrides, direct-member rows, manual relations
        return _FakeResult([], self._meter, kind)


class _FakeNode:
    def __init__(self, index: int) -> None:
        self.id = f"ent-{index}"
        self.labels = [f"ent-{index}"]
        source = _chunk_id(index % DOC_COUNT, 0)
        if index % 3 == 0:
            # Every third entity also cites a chunk of a NON-member document →
            # exercises the "mixed provenance" branch (visible but masked).
            source += graph_reader._GRAPH_FIELD_SEP + _chunk_id(
                (index + MEMBER_DOCS) % DOC_COUNT, 1
            )
        self.properties = {
            "entity_id": f"ent-{index}",
            "entity_type": "concept",
            "description": f"desc {index}",
            "source_id": source,
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
            "source_id": _chunk_id(index % DOC_COUNT, 0),
        }


class _FakeKnowledgeGraph:
    def __init__(self) -> None:
        self.nodes = [_FakeNode(i) for i in range(ENTITY_COUNT)]
        self.edges = [_FakeEdge(i) for i in range(ENTITY_COUNT)]


class _Rag:
    async def get_knowledge_graph(self, **_kwargs: Any) -> _FakeKnowledgeGraph:
        await asyncio.sleep(RTT_SECONDS)
        return _FakeKnowledgeGraph()


async def _legacy_membership(workspace: str, folder: str) -> tuple[set[str], dict]:
    """Pre-optimization composition: full-workspace index + separate member read.

    Both halves are the module's own, unchanged functions — this shim only
    restores how the folder-bound paths used to combine them.
    """
    member_docs = await graph_reader._load_member_docs(workspace, folder)
    chunk_to_doc = await graph_reader._load_chunk_to_doc_index(workspace)
    return member_docs, chunk_to_doc


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    max_in_flight: int
    entity_count: int


async def _run_once() -> tuple[list, list]:
    result = await graph_reader.read_graph_native(
        _Rag(), WORKSPACE, node_label="*", max_nodes=ENTITY_COUNT
    )
    assert result is not None, "native graph read returned unavailable"
    return result


@asynccontextmanager
async def _bound(legacy: bool, meter: _SessionMeter):
    """Bind the folder + fake read session, optionally restoring the old seam."""
    token = folder_mod._active_folder_id.set(FOLDER)
    original_session = graph_reader.get_read_session
    original_membership = graph_reader._load_folder_membership
    graph_reader.get_read_session = meter.session
    if legacy:
        graph_reader._load_folder_membership = _legacy_membership
    try:
        yield
    finally:
        graph_reader.get_read_session = original_session
        graph_reader._load_folder_membership = original_membership
        folder_mod._active_folder_id.reset(token)


async def _time_requests(
    *, legacy: bool, iterations: int, concurrency: int = 1
) -> _Sample:
    meter = _SessionMeter(READ_CAPACITY)
    durations: list[float] = []
    entity_counts: list[int] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with _bound(legacy, meter):
        for _ in range(3):  # warmup, outside the measured window
            await _run_once()

        async def run_one() -> None:
            async with semaphore:
                started = time.perf_counter()
                entities, _relations = await _run_once()
                durations.append((time.perf_counter() - started) * 1000)
                entity_counts.append(len(entities))

        meter.reset_counters()
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
    """Order-sensitive FULL comparison of a native graph result."""
    return (tuple(entities), tuple(relations))


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS

    # --- parity + structure: one request per seam --------------------------
    legacy_meter = _SessionMeter(READ_CAPACITY)
    async with _bound(True, legacy_meter):
        legacy_entities, legacy_relations = await _run_once()
    live_meter = _SessionMeter(READ_CAPACITY)
    async with _bound(False, live_meter):
        live_entities, live_relations = await _run_once()

    baseline = await _time_requests(legacy=True, iterations=count)
    optimized = await _time_requests(legacy=False, iterations=count)

    visible = sum(1 for e in legacy_entities if e["sources"] > 0)
    masked = sum(
        1
        for e in legacy_entities
        if e["summary"] == graph_reader._MASKED_ENTITY_SUMMARY
    )
    return [
        {
            "name": "folder-bound native graph read latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms; "
                f"{DOC_COUNT} docs, {MEMBER_DOCS} members, {CHUNKS_PER_DOC} chunks/doc"
            ),
        },
        {
            # THE revert guard: a folder-bound read must never scan the whole
            # DocStatus label again, and must load membership in one read.
            "name": "folder-bound read never scans the whole DocStatus label",
            "kind": "structural",
            "passed": (
                live_meter.queries.get(FULL_SCAN, 0) == 0
                and live_meter.queries.get(MEMBERSHIP, 0) == 1
                and live_meter.queries.get(MEMBER_DOCS_COLLECT, 0) == 0
                and legacy_meter.queries.get(FULL_SCAN, 0) == 1
            ),
            "detail": (
                "expected live: 0 full scans, 1 membership read, 0 collect "
                f"reads (legacy: 1 full scan); observed live={live_meter.queries} "
                f"legacy={legacy_meter.queries}"
            ),
        },
        {
            # Load-independent statement of the win: DocStatus rows streamed
            # (and JSON-parsed) per folder-bound read drop from the whole
            # workspace to the folder's member subset.
            "name": "DocStatus rows streamed drop to the member subset",
            "kind": "structural",
            "passed": (
                live_meter.rows.get(FULL_SCAN, 0) == 0
                and live_meter.rows.get(MEMBERSHIP, 0) == MEMBER_DOCS
                and legacy_meter.rows.get(FULL_SCAN, 0) == DOC_COUNT
            ),
            "detail": (
                f"expected {MEMBER_DOCS} membership rows and no full-scan rows on "
                f"the live path, {DOC_COUNT} full-scan rows on the legacy path; "
                f"observed rows live={live_meter.rows} legacy={legacy_meter.rows}"
            ),
        },
        {
            "name": "graph projection unchanged",
            "kind": "structural",
            "passed": (
                _projection(live_entities, live_relations)
                == _projection(legacy_entities, legacy_relations)
                and 0 < visible < ENTITY_COUNT  # both branches exercised
                and masked > 0  # mixed-provenance branch exercised
            ),
            "detail": (
                "expected identical ordered entities/relations with both the "
                f"hidden and the masked branch exercised; observed "
                f"{len(live_entities)} vs {len(legacy_entities)} entities "
                f"({visible} visible, {masked} masked), {len(live_relations)} vs "
                f"{len(legacy_relations)} relations"
            ),
        },
    ]


def _pool(samples: list[_Sample]) -> _Sample:
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
    """Alternate legacy/live rounds and pool them so one spike can't win."""
    legacy: list[_Sample] = []
    live: list[_Sample] = []
    for _ in range(rounds):
        legacy.append(
            await _time_requests(
                legacy=True, iterations=iterations, concurrency=concurrency
            )
        )
        live.append(
            await _time_requests(
                legacy=False, iterations=iterations, concurrency=concurrency
            )
        )
    return _pool(legacy), _pool(live)


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
        f"legacy: {ratio['baseline_ms']:.3f}ms -> live: "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster) [{ratio['detail']}]"
    )
    for case in cases[1:]:
        print(f"structural [{case['name']}]: {case['passed']} ({case['detail']})")

    for label, concurrency, requests in (("sustained", 8, 80), ("peak", 20, 160)):
        legacy, live = await _load_test(iterations=requests, concurrency=concurrency)
        _print_sample(f"{label} legacy", legacy)
        _print_sample(f"{label} live", live)


if __name__ == "__main__":
    asyncio.run(main())
