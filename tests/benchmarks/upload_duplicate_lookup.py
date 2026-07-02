"""Micro-benchmark: upload duplicate-filename lookup for the LightRAG upload path.

Run as a script: ``python tests/benchmarks/upload_duplicate_lookup.py``.

Importing this module (e.g. pytest collecting under ``tests/``) must NOT run
``register()`` or import ``lightrag.api`` — that triggers LightRAG's argv-based
config init and would abort collection of the whole suite. All such side
effects therefore live under ``__main__``.
"""

import random
import shutil
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

# Bound in __main__ only (see module docstring).
dr = None

FILE_COUNT = 20_000
ITERATIONS = 200


def _baseline_lookup(input_dir: Path, file_path: str) -> "Path | None":
    """Baseline implementation mirroring upstream (O(n) directory scan)."""
    if not file_path or file_path == dr.UNKNOWN_FILE_SOURCE:
        return None
    try:
        for candidate in input_dir.iterdir():
            if not candidate.is_file():
                continue
            if dr.normalize_file_path(candidate.name) == file_path:
                return candidate
    except FileNotFoundError:
        return None
    return None


def _build_dir(path: Path, count: int) -> list[str]:
    for i in range(count):
        (path / f"doc_{i:05d}.txt").write_text("x", encoding="utf-8")
    return [f"doc_{i:05d}.txt" for i in range(count)]


def _run_case(label: str, fn, input_dir: Path, queries: list[str]) -> dict:
    # Warm once so the optimized path's cache-fill is excluded from the lookup.
    fn(input_dir, "_warmup_missing.txt")

    durations_ms: list[float] = []
    tracemalloc.start()
    t0 = time.perf_counter()
    for query in queries:
        start = time.perf_counter()
        fn(input_dir, query)
        durations_ms.append((time.perf_counter() - start) * 1000)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "label": label,
        "iterations": len(queries),
        "total_ms": elapsed * 1000,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "p99_ms": statistics.quantiles(durations_ms, n=100)[98],
        "ops_per_s": len(queries) / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


if __name__ == "__main__":
    from twindb_lightrag_memgraph.patches.registry import (
        _patch_upload_duplicate_lookup,
    )

    import lightrag.api.routers.document_routes as dr  # noqa: F811

    _patch_upload_duplicate_lookup()

    base_dir = Path(tempfile.mkdtemp(prefix="bolt-upload-bench-"))
    try:
        existing_names = _build_dir(base_dir, FILE_COUNT)
        random.seed(42)
        existing_set = set(existing_names)

        # Missing names force the baseline's worst-case full scan — the "small
        # file still slow" path users reported when uploads are unique.
        queries: list[str] = []
        for idx in range(ITERATIONS):
            candidate = f"incoming_{idx:04d}.txt"
            while candidate in existing_set:
                candidate = f"incoming_{random.randrange(1_000_000):04d}.txt"
            queries.append(candidate)

        baseline = _run_case("baseline", _baseline_lookup, base_dir, queries)
        optimized = _run_case(
            "cached", dr.find_existing_file_by_file_path, base_dir, queries
        )

        for row in (baseline, optimized):
            print(row)

        speedup = (
            (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
        )
        mb_saved = baseline["peak_mb"] - optimized["peak_mb"]
        throughput_delta = (
            (optimized["ops_per_s"] - baseline["ops_per_s"])
            / baseline["ops_per_s"]
            * 100
        )

        print()
        print("## SUMMARY")
        print(
            f"mean: {baseline['mean_ms']:.3f}ms -> "
            f"{optimized['mean_ms']:.3f}ms ({speedup:.1f}% faster)"
        )
        print(
            f"throughput: {baseline['ops_per_s']:.1f} req/s -> "
            f"{optimized['ops_per_s']:.1f} req/s ({throughput_delta:.1f}%)"
        )
        print(
            f"peak_mem: {baseline['peak_mb']:.3f}MB -> "
            f"{optimized['peak_mb']:.3f}MB (saved {mb_saved:.3f}MB)"
        )
        print(f"FILE_COUNT={FILE_COUNT} ITERATIONS={ITERATIONS}")
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
