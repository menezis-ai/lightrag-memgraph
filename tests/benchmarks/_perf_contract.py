"""Shared contract for CI-enrolled performance benchmarks.

Each enrolled benchmark module under ``tests/benchmarks/`` exposes::

    async def measure(iterations: int | None = None) -> list[dict]

and is listed in ``PERF_BENCHMARKS`` below. ``tests/test_perf_regression.py``
imports each, runs ``measure()`` with low CI iterations, validates every
returned dict into a :class:`BenchCase`, and turns it into an assertion.

The scripts return plain dicts (not ``BenchCase``) on purpose: every benchmark
must stay runnable standalone via its ``Bench:`` command in ``.menezis/bolt.md``
(``python tests/benchmarks/<name>.py``), where the ``tests`` package is not on
``sys.path``. Keeping the shared import on the test side only preserves that.

Two case kinds, two failure modes — both needed, neither redundant:

* ``ratio``      guards against a *silent slowdown*: the optimized path getting
                 materially slower than its pre-optimization baseline (e.g. a
                 redundant call sneaks into the loop while the parallelism stays
                 in place). Self-normalizing — baseline and optimized run in the
                 same process/run, so the *ratio* is stable across machines even
                 though absolute ms are not. Never assert absolute ms in CI: the
                 shared bunker runner pool makes wall-clock unstable, and an
                 absolute threshold would flake in permanence (doctrine §6).
* ``structural`` guards against a *revert*: someone removes the ``gather`` /
                 cache, so ``optimized ≈ baseline`` and the ratio check passes
                 anyway. A deterministic boolean (parallel fan-out observed, N
                 round-trips, cache hit) catches that with zero timing flake.
                 This is the real presence-of-optimization guard.

``.menezis/bolt.md`` stays human documentation (the perf changelog + the repro
command). Thresholds live here, in code, versioned with the benchmark — never
parsed out of that prose.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ratio ceiling: optimized must not exceed baseline * this. Generous so runner
# noise doesn't flake the gate; still catches a >15% slowdown past the old code.
RATIO_TOLERANCE = 1.15

# Low iteration count for CI: enough signal for a ratio/structural check, cheap
# enough to keep the perf job off the token/wall-clock budget's radar (§7). The
# CLI ``main()`` of each script keeps its own (higher) default for local runs.
CI_ITERATIONS = 20

# Enrolled benchmarks (module name under ``tests.benchmarks``). Opt-in: a script
# joins this list only once it exposes ``measure()`` and runs Memgraph-free with
# fakes. The remaining scripts in this directory stay CLI-only until enrolled.
PERF_BENCHMARKS = [
    "query_data_batch_resolution",
    "graph_reader_metadata_gather",
    "notifications_mark_all_read_batch",
]


@dataclass
class BenchCase:
    """One CI assertion produced by a benchmark's ``measure()``.

    ``ratio`` cases populate ``baseline_ms`` / ``optimized_ms``; ``structural``
    cases populate ``passed`` / ``detail``. Constructed from the dicts the
    scripts return via ``BenchCase(**case)``, which also validates their shape.
    """

    name: str
    kind: str = "ratio"  # "ratio" | "structural"
    baseline_ms: float = 0.0
    optimized_ms: float = 0.0
    passed: bool | None = None
    detail: str = ""
