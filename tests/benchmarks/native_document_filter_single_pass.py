"""Benchmark the native document shim's local filtering pass.

Run standalone with::

    uv run python tests/benchmarks/native_document_filter_single_pass.py

The in-process baseline preserves the pre-optimization body: one list pass for
the folder and another pass for every active optional filter.  The live path is
``native_shims._filter_docs``.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from twindb_lightrag_memgraph.server import native_shims

ITERATIONS = 80
DOC_COUNT = 5_000
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 24


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    operations_per_second: float


def _documents() -> list[dict[str, Any]]:
    return [
        {
            "doc_id": f"doc-{index:05d}",
            "file_path": (
                f"/runbooks/oracle-{index:05d}.pdf"
                if index % 2 == 0
                else f"/guides/postgres-{index:05d}.pdf"
            ),
            "content_summary": (
                "Oracle database recovery runbook"
                if index % 2 == 0
                else "Postgres maintenance guide"
            ),
            "tags": ["approved", "database"] if index % 3 else ["draft", "database"],
            "folder": "operations" if index % 11 else "archive",
            "metadata": {"folder": "legacy-value"},
        }
        for index in range(DOC_COUNT)
    ]


def _baseline_filter_docs(
    items: list[dict[str, Any]],
    q: str | None,
    tag: str | None,
    folder: str,
    source: str | None = None,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Pre-optimization multi-pass body kept beside the live path."""
    from twindb_lightrag_memgraph.server.folder import load_folder_catalog

    default_folder = load_folder_catalog().default_folder_id
    out = [
        d
        for d in items
        if (
            d.get("folder") or (d.get("metadata") or {}).get("folder") or default_folder
        )
        == folder
    ]
    if q:
        needle = q.lower()
        out = [
            d
            for d in out
            if needle
            in " ".join(
                str(d.get(key) or "")
                for key in (
                    "doc_id",
                    "id",
                    "file_path",
                    "source",
                    "content_summary",
                    "summary",
                )
            ).lower()
        ]
    if source:
        source_needle = source.lower()
        out = [
            d
            for d in out
            if source_needle in str(d.get("file_path") or d.get("source") or "").lower()
        ]
    if doc_id:
        out = [d for d in out if doc_id == str(d.get("doc_id") or d.get("id") or "")]
    if tag:
        out = [d for d in out if tag in (d.get("tags") or [])]
    return out


def _filtered(
    function: Callable[..., list[dict[str, Any]]],
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return function(
        docs,
        q="oracle",
        tag="approved",
        folder="operations",
        source="runbooks",
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _time_requests(
    function: Callable[..., list[dict[str, Any]]],
    *,
    docs: list[dict[str, Any]],
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    expected_ids = [doc["doc_id"] for doc in _filtered(_baseline_filter_docs, docs)]
    durations: list[float] = []

    def run_one() -> None:
        started = time.perf_counter()
        result = _filtered(function, docs)
        durations.append((time.perf_counter() - started) * 1000)
        assert [doc["doc_id"] for doc in result] == expected_ids

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(lambda _index: run_one(), range(iterations)))
    elapsed = time.perf_counter() - started
    return _Sample(
        mean_ms=statistics.mean(durations),
        p95_ms=_percentile(durations, 0.95),
        p99_ms=_percentile(durations, 0.99),
        operations_per_second=iterations / elapsed,
    )


def _peak_memory_mb(
    function: Callable[..., list[dict[str, Any]]], docs: list[dict[str, Any]]
) -> float:
    tracemalloc.start()
    _filtered(function, docs)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def _live_filter_is_single_pass() -> tuple[bool, str]:
    tree = ast.parse(inspect.getsource(native_shims._filter_docs))
    loops = sum(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    list_comps = sum(isinstance(node, ast.ListComp) for node in ast.walk(tree))
    passed = loops == 1 and list_comps == 0
    return (
        passed,
        "expected one loop and no document-list comprehensions; "
        f"got {loops} loop(s), {list_comps} list comprehension(s)",
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entrypoint returning a same-run ratio and a deterministic guard."""
    count = iterations or ITERATIONS
    docs = _documents()

    # Full parity matrix includes false-y values, ID aliases, absent metadata,
    # every individual predicate, and the combined hot-path predicate set.
    parity_docs = docs[:30] + [
        {"id": "alias-id", "file_path": "", "metadata": None, "tags": None},
        {"doc_id": "empty-fields", "metadata": {}, "folder": "operations"},
    ]
    cases = (
        {"q": None, "tag": None, "folder": "operations"},
        {"q": "ORACLE", "tag": None, "folder": "operations"},
        {"q": None, "tag": "approved", "folder": "operations"},
        {"q": None, "tag": None, "folder": "operations", "source": "PDF"},
        {"q": None, "tag": None, "folder": "operations", "doc_id": "doc-00002"},
        {
            "q": "oracle",
            "tag": "approved",
            "folder": "operations",
            "source": "runbooks",
        },
    )
    parity = all(
        native_shims._filter_docs(parity_docs, **kwargs)
        == _baseline_filter_docs(parity_docs, **kwargs)
        for kwargs in cases
    )

    # Warm both paths before timing to reduce import/cache noise.
    _filtered(_baseline_filter_docs, docs)
    _filtered(native_shims._filter_docs, docs)
    baseline = _time_requests(_baseline_filter_docs, docs=docs, iterations=count)
    optimized = _time_requests(native_shims._filter_docs, docs=docs, iterations=count)
    single_pass, detail = _live_filter_is_single_pass()
    return [
        {
            "name": "native document local filter latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "native document filter remains single-pass with parity",
            "kind": "structural",
            "passed": single_pass and parity,
            "detail": f"{detail}; functional parity={parity}",
        },
    ]


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.operations_per_second:.1f} ops/s"
    )


async def main() -> None:
    docs = _documents()
    cases = await measure()
    ratio, structural = cases
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"sequential: {ratio['baseline_ms']:.3f}ms -> "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
    )
    print(f"structural: {structural['passed']} ({structural['detail']})")
    print(
        "memory: "
        f"{_peak_memory_mb(_baseline_filter_docs, docs):.3f}MB -> "
        f"{_peak_memory_mb(native_shims._filter_docs, docs):.3f}MB"
    )
    for label, concurrency, requests in (
        ("sustained", SUSTAINED_CONCURRENCY, 160),
        ("peak", PEAK_CONCURRENCY, 240),
    ):
        before = _time_requests(
            _baseline_filter_docs,
            docs=docs,
            iterations=requests,
            concurrency=concurrency,
        )
        after = _time_requests(
            native_shims._filter_docs,
            docs=docs,
            iterations=requests,
            concurrency=concurrency,
        )
        _print_sample(f"{label} baseline", before)
        _print_sample(f"{label} optimized", after)


if __name__ == "__main__":
    asyncio.run(main())
