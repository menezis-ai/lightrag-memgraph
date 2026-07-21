"""CI performance-regression gate.

Runs each enrolled benchmark (``tests.benchmarks._perf_contract.PERF_BENCHMARKS``)
with low iterations and turns its ``BenchCase`` results into assertions:

* ``ratio``      — optimized must not exceed baseline * ``RATIO_TOLERANCE``
                   (self-normalizing; never an absolute-ms threshold).
* ``structural`` — a deterministic boolean must hold (parallelism present, cache
                   hit, N round-trips) — catches a revert the ratio can't.

Skipped by default; runs only when ``RUN_PERF`` is set (see ``conftest.py``), so
it never slows the standard unit run. Rationale for the two case kinds and why
``.menezis/bolt.md`` is documentation rather than a threshold source lives in
``tests/benchmarks/_perf_contract.py``.
"""

from __future__ import annotations

import importlib

import pytest

from tests.benchmarks._perf_contract import (
    CI_ITERATIONS,
    PERF_BENCHMARKS,
    RATIO_TOLERANCE,
    BenchCase,
)

#: Ratio failures re-measure up to this many times before failing the gate.
#: The two measurement phases (baseline then optimized) run sequentially, so
#: a shared-runner load spike landing on ONE phase fabricates a regression —
#: observed ~25% false-fail across 2026-07-19..21 (runs 1861/1852-era, 1881:
#: same commit green on the push run, red on the PR run). A REAL regression
#: is deterministic in-code and fails every attempt; independent noise has to
#: strike all attempts in a row (~1.5%). Tolerance stays untouched and
#: structural cases stay strict-on-first-failure — they are deterministic.
PERF_MEASUREMENT_ATTEMPTS = 3


@pytest.mark.perf
@pytest.mark.parametrize("module_name", PERF_BENCHMARKS)
async def test_perf_no_regression(module_name):
    module = importlib.import_module(f"tests.benchmarks.{module_name}")

    ratio_failures: list[str] = []
    for attempt in range(1, PERF_MEASUREMENT_ATTEMPTS + 1):
        raw_cases = await module.measure(iterations=CI_ITERATIONS)
        assert raw_cases, f"{module_name}.measure() returned no BenchCase"

        ratio_failures = []
        # BenchCase(**case) also validates the dict shape the script returned.
        for case in (BenchCase(**raw) for raw in raw_cases):
            if case.kind == "structural":
                assert case.passed, f"[{module_name}] {case.name}: {case.detail}"
            elif case.kind == "ratio":
                ceiling = case.baseline_ms * RATIO_TOLERANCE
                if case.optimized_ms > ceiling:
                    ratio_failures.append(
                        f"[{module_name}] {case.name}: perf regression — "
                        f"optimized {case.optimized_ms:.3f}ms > baseline "
                        f"{case.baseline_ms:.3f}ms x{RATIO_TOLERANCE} "
                        f"({ceiling:.3f}ms)"
                    )
            else:
                pytest.fail(f"[{module_name}] {case.name}: unknown kind {case.kind!r}")
        if not ratio_failures:
            return

    pytest.fail(
        "\n".join(ratio_failures)
        + f"\n(persisted across {PERF_MEASUREMENT_ATTEMPTS} independent "
        "measurement attempts — not runner noise)"
    )
