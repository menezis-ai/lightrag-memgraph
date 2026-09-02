"""Bolt micro-benchmark: one post-delete hygiene pass per bulk-delete batch
(``routes_documents._run_bulk_delete_batch`` → ``router._post_delete_hygiene``).

``POST /documents/bulk-delete`` deletes serially by decision (write-conflict
storms under operator load, 28afb6c). Every physical delete then ran the
workspace hygiene inline: a query-LLM-cache drop and the source_id sweep —
three workspace-wide scans (every chunk id, every entity with ``source_id``,
every relation with ``source_id``) streamed into Python and partitioned. The
sweep's coalescing front door was written for a concurrent loop and never
engages inside a serial one, so a batch of N physical deletes paid N sweeps
and N cache drops. It also re-read the DocStatus record the visibility check
had just fetched. The batch now defers the hygiene to ONE pass after its last
delete and hands the record's source path down.

The harness drives the REAL chain — batch runner, per-document delete,
membership claim, ``router._delete_doc_from_rag``, ``_post_delete_hygiene``,
``graph_reader.request_source_ref_sweep`` / ``sweep_stale_source_refs`` —
against a fake RAG and fake Memgraph sessions that pay a fixed RTT per
round-trip and stream realistic sweep rows. "before" is the pre-optimization
per-document body, preserved in this file (the old ``_delete_one_document``:
visibility read, then ``_apply_membership_delete`` with inline hygiene and
its own DocStatus re-read) in the same serial loop. Only that composition
differs between the two runs; every function underneath is the module's own.

This file replaces ``bulk_delete_documents.py``, which modelled the parallel
batch body that 28afb6c reverted and was never enrolled.

Run standalone with::

    uv run python tests/benchmarks/bulk_delete_hygiene_once.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from twindb_lightrag_memgraph import _import_cleanup, _twindb_state
from twindb_lightrag_memgraph.server import folder as folder_mod
from twindb_lightrag_memgraph.server import graph_reader, webui_router
from twindb_lightrag_memgraph.server.webui import routes_documents as rd

ITERATIONS = 20
RTT_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
DOC_COUNT = int(os.environ.get("DOC_COUNT", "40"))  # route caps at 500
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "4000"))  # KV text_chunks ids
ENTITY_ROWS = int(os.environ.get("ENTITY_ROWS", "2000"))
RELATION_ROWS = int(os.environ.get("RELATION_ROWS", "3000"))
READ_CAPACITY = int(os.environ.get("READ_CAPACITY", "20"))
WORKSPACE = "bulkws"
FOLDER = "f-bulk"
SEP = graph_reader._GRAPH_FIELD_SEP

_LIVE_CHUNKS = [f"chunk-{i:06d}{'0' * 26}" for i in range(CHUNK_ROWS)]
_CHUNK_ROWS = [{"id": c} for c in _LIVE_CHUNKS]
_ENTITY_ROWS = [
    {
        "entity_id": f"ent-{i}",
        "source_id": SEP.join(_LIVE_CHUNKS[(i * 3 + j) % CHUNK_ROWS] for j in range(3)),
    }
    for i in range(ENTITY_ROWS)
]
_RELATION_ROWS = [
    {
        "src": f"ent-{i % ENTITY_ROWS}",
        "tgt": f"ent-{(i + 1) % ENTITY_ROWS}",
        "source_id": _LIVE_CHUNKS[i % CHUNK_ROWS],
    }
    for i in range(RELATION_ROWS)
]


class _Meter:
    """Count every round-trip by kind and the rows the sweep streams."""

    def __init__(self, capacity: int) -> None:
        self.calls: dict[str, int] = {}
        self.rows = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._capacity = asyncio.Semaphore(capacity)

    async def rtt(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        async with self._capacity:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                await asyncio.sleep(RTT_SECONDS)
            finally:
                self.in_flight -= 1

    def reset(self) -> None:
        self.calls = {}
        self.rows = 0
        self.max_in_flight = 0


class _FakeResult:
    def __init__(self, records: list[dict[str, Any]], meter: _Meter) -> None:
        self._records = records
        self._meter = meter

    def __aiter__(self):
        async def gen():
            for record in self._records:
                self._meter.rows += 1
                yield record

        return gen()

    async def consume(self):
        return None


class _FakeReadSession:
    def __init__(self, meter: _Meter) -> None:
        self._meter = meter

    async def run(self, query: str, **_params: Any) -> _FakeResult:
        if "_text_chunks" in query:
            await self._meter.rtt("sweep_scan_chunks")
            return _FakeResult(_CHUNK_ROWS, self._meter)
        if "-[r]->" in query:
            await self._meter.rtt("sweep_scan_relations")
            return _FakeResult(_RELATION_ROWS, self._meter)
        if "n.source_id IS NOT NULL" in query:
            await self._meter.rtt("sweep_scan_entities")
            return _FakeResult(_ENTITY_ROWS, self._meter)
        await self._meter.rtt("other_read")
        return _FakeResult([], self._meter)


class _FakeDocStatus:
    def __init__(self, meter: _Meter, docs: list[str]) -> None:
        self._meter = meter
        self.records = {
            doc: {"id": doc, "file_path": f"/kb/{doc}.md", "metadata": {}}
            for doc in docs
        }
        self.folders = {doc: [FOLDER] for doc in docs}

    async def get_by_id(self, doc_id: str):
        await self._meter.rtt("docstatus_get_by_id")
        record = self.records.get(doc_id)
        return dict(record) if record else None

    async def get_folders_for_doc(self, doc_id: str):
        await self._meter.rtt("docstatus_get_folders")
        folders = self.folders.get(doc_id)
        return list(folders) if folders is not None else None

    async def claim_last_membership_delete(self, doc_id, folder, claim) -> bool:
        await self._meter.rtt("docstatus_claim")
        return doc_id in self.folders

    async def release_delete_claim(self, doc_id, claim) -> None:
        await self._meter.rtt("docstatus_release")

    async def remove_from_folder(self, doc_id: str, folder: str) -> None:
        await self._meter.rtt("docstatus_remove_from_folder")


class _FakeRag:
    workspace = WORKSPACE

    def __init__(self, meter: _Meter, docs: list[str]) -> None:
        self._meter = meter
        self.doc_status = _FakeDocStatus(meter, docs)
        self.deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str):
        await self._meter.rtt("lightrag_adelete")
        self.deleted.append(doc_id)
        self.doc_status.records.pop(doc_id, None)
        self.doc_status.folders.pop(doc_id, None)
        return SimpleNamespace(status="success", message="")

    async def aclear_cache(self) -> None:
        await self._meter.rtt("llm_cache_drop")


async def _legacy_delete_one(legacy, rag, doc_id):
    """Pre-optimization ``_delete_one_document``: inline hygiene, own re-read."""
    active = legacy.current_folder_id()
    doc = await legacy._get_doc_for_active_folder(doc_id)
    physically_deleted = await rd._apply_membership_delete(legacy, rag, doc_id, active)
    if physically_deleted is None:
        return None
    return {
        "doc_id": doc_id,
        "label": doc.get("file_path") or doc_id,
        "folder": active,
        "physically_deleted": physically_deleted,
    }


async def _legacy_batch(legacy, rag, doc_ids):
    """Pre-optimization batch body: the same serial loop, hygiene per document."""
    results = []
    for doc_id in doc_ids:
        outcome = await _legacy_delete_one(legacy, rag, doc_id)
        if outcome is not None:
            results.append(outcome)
    return results, [], []


async def _live_batch(legacy, rag, doc_ids):
    return await rd._run_bulk_delete_batch(legacy, rag, doc_ids)


@asynccontextmanager
async def _bound(meter: _Meter, *, doc_count: int = DOC_COUNT):
    """Bind the folder, the fake RAG and the fake Memgraph sessions.

    ONE binding per measurement, even under concurrency: the route resolves
    the RAG through the process-global ``_twindb_state``, so concurrent
    requests must share one fake RAG and delete disjoint slices of its docs.
    """

    @asynccontextmanager
    async def _read_session():
        yield _FakeReadSession(meter)

    @asynccontextmanager
    async def _noop_cm():
        yield None

    async def _no_cleanup(paths):
        return None

    docs = [f"doc-{i:05d}" for i in range(doc_count)]
    rag = _FakeRag(meter, docs)
    token = folder_mod._active_folder_id.set(FOLDER)
    previous_rag = _twindb_state.get("rag")
    saved = (
        graph_reader.get_read_session,
        graph_reader.get_session,
        graph_reader.acquire_write_slot,
        _import_cleanup.cleanup_import_paths,
    )
    _twindb_state["rag"] = rag
    graph_reader.get_read_session = _read_session
    graph_reader.get_session = _noop_cm
    graph_reader.acquire_write_slot = _noop_cm
    _import_cleanup.cleanup_import_paths = _no_cleanup
    try:
        yield rag, docs
    finally:
        (
            graph_reader.get_read_session,
            graph_reader.get_session,
            graph_reader.acquire_write_slot,
            _import_cleanup.cleanup_import_paths,
        ) = saved
        if previous_rag is None:
            _twindb_state.pop("rag", None)
        else:
            _twindb_state["rag"] = previous_rag
        folder_mod._active_folder_id.reset(token)


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float


async def _time_requests(
    *, legacy: bool, iterations: int, concurrency: int = 1
) -> _Sample:
    batch = _legacy_batch if legacy else _live_batch
    durations: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)
    meter = _Meter(READ_CAPACITY)

    async with _bound(meter, doc_count=DOC_COUNT * iterations) as (rag, docs):
        slices = [docs[i * DOC_COUNT : (i + 1) * DOC_COUNT] for i in range(iterations)]

        async def run_one(doc_ids: list[str]) -> None:
            async with semaphore:
                started = time.perf_counter()
                results, _failed, _busy = await batch(webui_router, rag, doc_ids)
                durations.append((time.perf_counter() - started) * 1000)
                assert len(results) == DOC_COUNT

        started = time.perf_counter()
        await asyncio.gather(*(run_one(doc_ids) for doc_ids in slices))
        elapsed = time.perf_counter() - started
        assert sorted(rag.deleted) == docs
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
    )


async def _one(legacy: bool) -> tuple[_Meter, list, list[str]]:
    meter = _Meter(READ_CAPACITY)
    async with _bound(meter) as (rag, docs):
        results, _failed, _busy = await (_legacy_batch if legacy else _live_batch)(
            webui_router, rag, docs
        )
        return meter, results, list(rag.deleted)


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS
    legacy_meter, legacy_results, legacy_deleted = await _one(legacy=True)
    live_meter, live_results, live_deleted = await _one(legacy=False)
    baseline = await _time_requests(legacy=True, iterations=count)
    optimized = await _time_requests(legacy=False, iterations=count)

    lc, vc = legacy_meter.calls, live_meter.calls
    return [
        {
            "name": f"bulk delete of {DOC_COUNT} last-membership docs",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms; sweep rows "
                f"{CHUNK_ROWS}+{ENTITY_ROWS}+{RELATION_ROWS} per scan set"
            ),
        },
        {
            "name": "N sweeps and N cache drops collapse to one per batch",
            "kind": "structural",
            "passed": (
                vc.get("sweep_scan_chunks", 0) == 1
                and vc.get("sweep_scan_entities", 0) == 1
                and vc.get("sweep_scan_relations", 0) == 1
                and vc.get("llm_cache_drop", 0) == 1
                and lc.get("sweep_scan_chunks", 0) == DOC_COUNT
                and lc.get("llm_cache_drop", 0) == DOC_COUNT
            ),
            "detail": (
                f"expected 1 sweep (3 scans) + 1 cache drop on the live path vs "
                f"{DOC_COUNT} each on the legacy path; observed live={vc} legacy={lc}"
            ),
        },
        {
            "name": "the DocStatus record is read once per document, not twice",
            "kind": "structural",
            "passed": (
                vc.get("docstatus_get_by_id", 0) == DOC_COUNT
                and lc.get("docstatus_get_by_id", 0) == 2 * DOC_COUNT
            ),
            "detail": (
                f"expected {DOC_COUNT} get_by_id reads live vs {2 * DOC_COUNT} legacy; "
                f"observed live={vc.get('docstatus_get_by_id')} "
                f"legacy={lc.get('docstatus_get_by_id')}"
            ),
        },
        {
            "name": "sweep rows streamed drop from N sets to one",
            "kind": "structural",
            "passed": (
                live_meter.rows == CHUNK_ROWS + ENTITY_ROWS + RELATION_ROWS
                and legacy_meter.rows
                == DOC_COUNT * (CHUNK_ROWS + ENTITY_ROWS + RELATION_ROWS)
            ),
            "detail": (
                f"observed rows live={live_meter.rows} legacy={legacy_meter.rows}"
            ),
        },
        {
            "name": "batch outcome and physical deletes unchanged",
            "kind": "structural",
            "passed": (
                live_results == legacy_results
                and live_deleted == legacy_deleted
                and len(live_results) == DOC_COUNT
                and all(r["physically_deleted"] for r in live_results)
            ),
            "detail": (
                f"expected identical ordered results and deletions; observed "
                f"{len(live_results)} vs {len(legacy_results)} results"
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
    )


async def _load_test(*, iterations: int, concurrency: int, rounds: int = 3):
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
        f"throughput={sample.requests_per_second:.2f} req/s"
    )


async def main() -> None:
    cases = await measure()
    ratio = cases[0]
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"legacy: {ratio['baseline_ms']:.3f}ms -> live: {ratio['optimized_ms']:.3f}ms "
        f"({gain:.1f}% faster) [{ratio['detail']}]"
    )
    for case in cases[1:]:
        print(f"structural [{case['name']}]: {case['passed']} ({case['detail']})")
    # Concurrency = simultaneous bulk-delete requests (an admin action; 2 is
    # already unusual, 4 is a stress case).
    for label, concurrency, requests in (("sustained", 2, 6), ("peak", 4, 8)):
        legacy, live = await _load_test(iterations=requests, concurrency=concurrency)
        _print_sample(f"{label} legacy", legacy)
        _print_sample(f"{label} live", live)


if __name__ == "__main__":
    asyncio.run(main())
