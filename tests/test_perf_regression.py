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


@pytest.mark.perf
@pytest.mark.parametrize("module_name", PERF_BENCHMARKS)
async def test_perf_no_regression(module_name):
    module = importlib.import_module(f"tests.benchmarks.{module_name}")
    raw_cases = await module.measure(iterations=CI_ITERATIONS)
    assert raw_cases, f"{module_name}.measure() returned no BenchCase"

    # BenchCase(**case) also validates the dict shape the script returned.
    for case in (BenchCase(**raw) for raw in raw_cases):
        if case.kind == "structural":
            assert case.passed, f"[{module_name}] {case.name}: {case.detail}"
        elif case.kind == "ratio":
            ceiling = case.baseline_ms * RATIO_TOLERANCE
            assert case.optimized_ms <= ceiling, (
                f"[{module_name}] {case.name}: perf regression — optimized "
                f"{case.optimized_ms:.3f}ms > baseline {case.baseline_ms:.3f}ms "
                f"x{RATIO_TOLERANCE} ({ceiling:.3f}ms)"
            )
        else:
            pytest.fail(f"[{module_name}] {case.name}: unknown kind {case.kind!r}")
