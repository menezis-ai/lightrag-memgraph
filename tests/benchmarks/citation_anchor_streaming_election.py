"""Benchmark citation-anchor scoring and winner election.

Run standalone with::

    uv run python tests/benchmarks/citation_anchor_streaming_election.py

The in-process baseline preserves the pre-optimization implementation: build a
set of every token in every segment, materialize every segment score, then scan
that score table twice to select the winner and runner-up.  The live path is
``paragraph_anchor.compute_best_anchor``.
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

from twindb_lightrag_memgraph.server.query import paragraph_anchor

ITERATIONS = 60
CHUNK_COUNT = 32
PARAGRAPHS_PER_CHUNK = 20
TOKENS_PER_PARAGRAPH = 160
SUSTAINED_CONCURRENCY = 8
PEAK_CONCURRENCY = 24


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    operations_per_second: float


def _workload() -> tuple[list[tuple[str, str]], paragraph_anchor.CitationEvidence]:
    evidence = paragraph_anchor.CitationEvidence(
        frozenset(f"signal{index}" for index in range(24))
    )
    candidates: list[tuple[str, str]] = []
    for chunk_index in range(CHUNK_COUNT):
        paragraphs = []
        for paragraph_index in range(PARAGRAPHS_PER_CHUNK):
            tokens = [
                f"noise{(chunk_index * 97 + paragraph_index * 31 + index) % 4096}"
                for index in range(TOKENS_PER_PARAGRAPH)
            ]
            # A deterministic winner and runner-up exercise confidence scoring;
            # the remaining segments model long, mostly irrelevant source text.
            if chunk_index == 19 and paragraph_index == 7:
                tokens[:18] = [f"signal{index}" for index in range(18)]
            elif chunk_index == 4 and paragraph_index == 11:
                tokens[:8] = [f"signal{index}" for index in range(8)]
            paragraphs.append(" ".join(tokens))
        candidates.append((f"chunk-{chunk_index:03d}", "\n\n".join(paragraphs)))
    return candidates, evidence


def _baseline_containment(
    evidence_tokens: frozenset[str], paragraph_text: str
) -> float:
    paragraph_tokens = set(paragraph_anchor._normalized_tokens(paragraph_text))
    if not evidence_tokens:
        return 0.0
    return len(evidence_tokens & paragraph_tokens) / len(evidence_tokens)


def _baseline_compute_best_anchor(
    candidates: list[tuple[str, str]],
    evidence: paragraph_anchor.CitationEvidence,
) -> tuple[str, dict[str, Any]] | None:
    """Pre-optimization score-table body kept beside the live path."""
    if evidence.incomplete or not evidence.tokens:
        return None
    segmented = [
        (
            chunk_id,
            content,
            [
                (paragraph.start, paragraph.end)
                for paragraph in paragraph_anchor.split_paragraphs(content)
            ],
        )
        for chunk_id, content in candidates
        if content
    ]
    scored: list[tuple[float, int, int, str, tuple[int, int], int]] = []
    for chunk_index, (chunk_id, content, segments) in enumerate(segmented):
        for segment_index, (start, end) in enumerate(segments):
            score = _baseline_containment(evidence.tokens, content[start:end])
            scored.append(
                (
                    score,
                    chunk_index,
                    segment_index,
                    chunk_id,
                    (start, end),
                    len(segments),
                )
            )
    if not scored:
        return None
    best_position = min(
        range(len(scored)),
        key=lambda index: (-scored[index][0], scored[index][1], scored[index][2]),
    )
    best_score, _, segment_index, chunk_id, span, segment_count = scored[best_position]
    if best_score <= 0.0:
        return None
    second = max(
        (entry[0] for index, entry in enumerate(scored) if index != best_position),
        default=0.0,
    )
    confidence = best_score - 0.5 * second
    if confidence < paragraph_anchor.MIN_ANCHOR_CONFIDENCE:
        return None
    return chunk_id, {
        "start": span[0],
        "end": span[1],
        "paragraph_idx": segment_index,
        "paragraph_count": segment_count,
        "confidence": round(confidence, 4),
        "method": paragraph_anchor.ANCHOR_METHOD_LEXICAL,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _time_requests(
    function: Callable[..., tuple[str, dict[str, Any]] | None],
    *,
    candidates: list[tuple[str, str]],
    evidence: paragraph_anchor.CitationEvidence,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    expected = _baseline_compute_best_anchor(candidates, evidence)
    durations: list[float] = []

    def run_one(_index: int) -> None:
        started = time.perf_counter()
        result = function(candidates, evidence)
        durations.append((time.perf_counter() - started) * 1000)
        assert result == expected

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(run_one, range(iterations)))
    elapsed = time.perf_counter() - started
    return _Sample(
        mean_ms=statistics.mean(durations),
        p95_ms=_percentile(durations, 0.95),
        p99_ms=_percentile(durations, 0.99),
        operations_per_second=iterations / elapsed,
    )


def _peak_memory_mb(
    function: Callable[..., tuple[str, dict[str, Any]] | None],
    candidates: list[tuple[str, str]],
    evidence: paragraph_anchor.CitationEvidence,
) -> float:
    tracemalloc.start()
    function(candidates, evidence)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def _live_path_streams_scores() -> tuple[bool, str]:
    election_tree = ast.parse(inspect.getsource(paragraph_anchor._elect_best_segment))
    containment_tree = ast.parse(inspect.getsource(paragraph_anchor._containment))
    list_comprehensions = sum(
        isinstance(node, ast.ListComp) for node in ast.walk(election_tree)
    )
    append_calls = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        for node in ast.walk(election_tree)
    )
    intersects_from_evidence = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "intersection"
        for node in ast.walk(containment_tree)
    )
    passed = list_comprehensions == 0 and append_calls == 0 and intersects_from_evidence
    return (
        passed,
        "expected no materialized score table and evidence-sized token "
        f"intersection; list comprehensions={list_comprehensions}, "
        f"append calls={append_calls}, intersection={intersects_from_evidence}",
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    """CI entrypoint returning a same-run ratio and deterministic guard."""
    count = iterations or ITERATIONS
    candidates, evidence = _workload()

    # Lock output, ordering, tie-breaking, confidence floor, and fail-soft cases.
    tie_evidence = paragraph_anchor.CitationEvidence(
        frozenset(f"tok{index}" for index in range(10))
    )
    tie_supporting = " ".join(f"tok{index}" for index in range(10))
    parity_cases = [
        (candidates, evidence),
        ([candidates[0]], evidence),
        (
            [("first", tie_supporting), ("second", tie_supporting)],
            tie_evidence,
        ),
        ([], evidence),
        (candidates, paragraph_anchor.CitationEvidence(frozenset())),
        (
            candidates,
            paragraph_anchor.CitationEvidence(evidence.tokens, incomplete=True),
        ),
    ]
    parity = all(
        paragraph_anchor.compute_best_anchor(case_candidates, case_evidence)
        == _baseline_compute_best_anchor(case_candidates, case_evidence)
        for case_candidates, case_evidence in parity_cases
    )

    _baseline_compute_best_anchor(candidates, evidence)
    paragraph_anchor.compute_best_anchor(candidates, evidence)
    baseline = _time_requests(
        _baseline_compute_best_anchor,
        candidates=candidates,
        evidence=evidence,
        iterations=count,
    )
    optimized = _time_requests(
        paragraph_anchor.compute_best_anchor,
        candidates=candidates,
        evidence=evidence,
        iterations=count,
    )
    streaming, detail = _live_path_streams_scores()
    return [
        {
            "name": "citation anchor scoring and election latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "citation anchor scores remain streaming with parity",
            "kind": "structural",
            "passed": streaming and parity,
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
    candidates, evidence = _workload()
    ratio, structural = await measure()
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"sequential: {ratio['baseline_ms']:.3f}ms -> "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
    )
    print(f"structural: {structural['passed']} ({structural['detail']})")
    print(
        "memory: "
        f"{_peak_memory_mb(_baseline_compute_best_anchor, candidates, evidence):.3f}MB "
        f"-> {_peak_memory_mb(paragraph_anchor.compute_best_anchor, candidates, evidence):.3f}MB"
    )
    for label, concurrency, requests in (
        ("sustained", SUSTAINED_CONCURRENCY, 80),
        ("peak", PEAK_CONCURRENCY, 120),
    ):
        before = _time_requests(
            _baseline_compute_best_anchor,
            candidates=candidates,
            evidence=evidence,
            iterations=requests,
            concurrency=concurrency,
        )
        after = _time_requests(
            paragraph_anchor.compute_best_anchor,
            candidates=candidates,
            evidence=evidence,
            iterations=requests,
            concurrency=concurrency,
        )
        _print_sample(f"{label} baseline", before)
        _print_sample(f"{label} optimized", after)


if __name__ == "__main__":
    asyncio.run(main())
