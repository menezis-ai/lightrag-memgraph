"""Bolt micro-benchmark: one ledger write for the bulk-retag activity events
(``routes_tags._emit_bulk_retag_events`` → ``WebuiStore.record_activities`` →
``MemgraphActivityStore.append_many``).

``POST /documents/_bulk-retag`` accepts up to 500 documents and applies the
tag mutation in ONE Memgraph transaction — then used to record its activity
trail with one ``record_activity`` per document: N write-slot acquisitions on
the global write semaphore (shared with ingestion) and N ``MERGE``
round-trips, serialized. The batch path builds every event first and lands
them in a single ``UNWIND`` write; the regulatory audit projection still
receives each event, in order.

The harness drives the REAL emitter and the REAL ``WebuiStore.record_activities``
against a fake activity backend that pays a fixed RTT per write under a
write-slot semaphore (``WRITE_CAPACITY`` models ``MEMGRAPH_WRITE_CONCURRENCY``,
default 8). "before" is the pre-optimization emitter body, preserved in this
file, driving the same store's per-event ``record_activity``.

Run standalone with::

    uv run python tests/benchmarks/bulk_retag_activity_batch.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from twindb_lightrag_memgraph.server import audit
from twindb_lightrag_memgraph.server.webui import routes_tags
from twindb_lightrag_memgraph.server.webui.events import _make_event
from twindb_lightrag_memgraph.server.webui.store import WebuiStore

ITERATIONS = 40
RTT_SECONDS = float(os.environ.get("RTT_MS", "4")) / 1000.0
DOC_COUNT = int(os.environ.get("DOC_COUNT", "200"))  # route caps at 500
WRITE_CAPACITY = int(os.environ.get("WRITE_CAPACITY", "8"))
ACTOR = "bench.steward"
ADDS = ["rmf-validated"]
REMOVES = ["draft"]
DOC_IDS = [f"doc-{i:04d}" for i in range(DOC_COUNT)]
EXISTING = {doc_id: f"{doc_id}.md" for doc_id in DOC_IDS}
RESULTING = {doc_id: ["rmf-validated", "oracle"] for doc_id in DOC_IDS}


class _FakeActivityBackend:
    """Ledger stub: same surface as the Memgraph backend, RTT per write."""

    def __init__(self, capacity: int) -> None:
        self.writes = 0
        self.slot_acquisitions = 0
        self.ledger: list[dict[str, Any]] = []  # newest first, like the stores
        self.in_flight = 0
        self.max_in_flight = 0
        self._slots = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def _slot(self):
        async with self._slots:
            self.slot_acquisitions += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                yield
            finally:
                self.in_flight -= 1

    async def append(self, event: dict[str, Any]) -> dict[str, Any]:
        async with self._slot():
            self.writes += 1
            await asyncio.sleep(RTT_SECONDS)
            self.ledger.insert(0, dict(event))
        return dict(event)

    async def append_many(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not events:
            return []
        async with self._slot():
            self.writes += 1
            await asyncio.sleep(RTT_SECONDS)
            for event in events:
                self.ledger.insert(0, dict(event))
        return [dict(event) for event in events]

    def reset(self) -> None:
        self.writes = 0
        self.slot_acquisitions = 0
        self.ledger = []
        self.max_in_flight = 0


async def _legacy_emit(doc_ids, resulting_by_doc, existing, adds, removes, actor):
    """Pre-optimization body: one ``record_activity`` per document."""
    for doc_id in doc_ids:
        new_tags = resulting_by_doc.get(doc_id, [])
        event = _make_event(
            kind="doc-retagged",
            sev="info",
            actor=actor,
            target_label=existing[doc_id] or doc_id,
            summary=f"tags: +{','.join(adds) or '∅'} -{','.join(removes) or '∅'}",
            meta={
                "doc_id": doc_id,
                "adds": adds,
                "removes": removes,
                "resulting_tags": new_tags,
            },
            target_type="document",
            target_id=doc_id,
        )
        await routes_tags.get_store().record_activity(event)


@dataclass
class _Harness:
    backend: _FakeActivityBackend
    audited: list[dict[str, Any]]


@asynccontextmanager
async def _bound():
    """Bind a seed store with the fake ledger, and count audit submissions."""
    backend = _FakeActivityBackend(WRITE_CAPACITY)
    store = WebuiStore.from_seed()
    store._activity_backend = backend
    audited: list[dict[str, Any]] = []

    def _sink(event) -> None:
        audited.append(event)

    original_get_store = routes_tags.get_store
    original_sink = audit._audit_event_sink
    routes_tags.get_store = lambda: store
    audit.set_audit_event_sink(_sink)
    try:
        yield _Harness(backend, audited)
    finally:
        audit.set_audit_event_sink(original_sink)
        routes_tags.get_store = original_get_store


async def _emit(legacy: bool) -> None:
    emitter = _legacy_emit if legacy else routes_tags._emit_bulk_retag_events
    await emitter(DOC_IDS, RESULTING, EXISTING, ADDS, REMOVES, ACTOR)


def _normalized(ledger: list[dict[str, Any]]) -> list[tuple]:
    """Ledger projection minus the per-run fields (random id, wall-clock ts)."""
    return [
        (
            e["kind"],
            e["sev"],
            e["actor"]["user"],
            e["target"]["type"],
            e["target"].get("id"),
            e["target"]["label"],
            e["summary"],
            tuple(sorted(e["meta"].items(), key=lambda kv: kv[0])),
        )
        for e in ledger
    ]


def _freeze_meta(ledger: list[dict[str, Any]]) -> list[tuple]:
    out = []
    for e in ledger:
        meta = {k: tuple(v) if isinstance(v, list) else v for k, v in e["meta"].items()}
        out.append(_normalized([{**e, "meta": meta}])[0])
    return out


@dataclass
class _Sample:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    max_in_flight: int


async def _time_requests(
    *, legacy: bool, iterations: int, concurrency: int = 1
) -> _Sample:
    durations: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)
    async with _bound() as harness:
        for _ in range(2):  # warmup
            await _emit(legacy)

        async def run_one() -> None:
            async with semaphore:
                started = time.perf_counter()
                await _emit(legacy)
                durations.append((time.perf_counter() - started) * 1000)

        harness.backend.reset()
        started = time.perf_counter()
        await asyncio.gather(*(run_one() for _ in range(iterations)))
        elapsed = time.perf_counter() - started
        max_in_flight = harness.backend.max_in_flight

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
        max_in_flight=max_in_flight,
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS

    async with _bound() as legacy_h:
        await _emit(legacy=True)
    async with _bound() as live_h:
        await _emit(legacy=False)

    baseline = await _time_requests(legacy=True, iterations=count)
    optimized = await _time_requests(legacy=False, iterations=count)

    return [
        {
            "name": f"bulk retag activity trail latency ({DOC_COUNT} docs)",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "N ledger writes collapse to one",
            "kind": "structural",
            "passed": (
                live_h.backend.writes == 1
                and live_h.backend.slot_acquisitions == 1
                and legacy_h.backend.writes == DOC_COUNT
                and legacy_h.backend.slot_acquisitions == DOC_COUNT
            ),
            "detail": (
                f"expected 1 write / 1 write-slot on the live path against "
                f"{DOC_COUNT} on the legacy path; observed live="
                f"{live_h.backend.writes}/{live_h.backend.slot_acquisitions} "
                f"legacy={legacy_h.backend.writes}/{legacy_h.backend.slot_acquisitions}"
            ),
        },
        {
            "name": "ledger and audit trail unchanged",
            "kind": "structural",
            "passed": (
                _freeze_meta(live_h.backend.ledger)
                == _freeze_meta(legacy_h.backend.ledger)
                and len(live_h.backend.ledger) == DOC_COUNT
                and len(live_h.audited) == len(legacy_h.audited) == DOC_COUNT
                and [e.twin.resource for e in live_h.audited]
                == [e.twin.resource for e in legacy_h.audited]
            ),
            "detail": (
                "expected the same newest-first ledger (ids/timestamps aside) and "
                f"{DOC_COUNT} audit submissions in the same order on both paths; "
                f"observed ledger {len(live_h.backend.ledger)} vs "
                f"{len(legacy_h.backend.ledger)}, audited {len(live_h.audited)} vs "
                f"{len(legacy_h.audited)}"
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
    )


async def _load_test(
    *, iterations: int, concurrency: int, rounds: int = 3
) -> tuple[_Sample, _Sample]:
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
        f"max_write_slots={sample.max_in_flight}"
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

    # Concurrency here = simultaneous bulk-retag requests contending for the
    # write slots (an admin action; 4 is already generous, 8 saturates them).
    for label, concurrency, requests in (("sustained", 4, 16), ("peak", 8, 24)):
        legacy, live = await _load_test(iterations=requests, concurrency=concurrency)
        _print_sample(f"{label} legacy", legacy)
        _print_sample(f"{label} live", live)


if __name__ == "__main__":
    asyncio.run(main())
