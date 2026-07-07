"""Micro-benchmark: exact cosine candidate ranking with/without precomputed
query-vector norm.

This benchmark isolates the exact-search scoring hot-loop introduced by
`MemgraphVectorDBStorage`’s exact path.
"""

from __future__ import annotations

import math
import random
import statistics
import time

ITERATIONS = 80
EMBEDDING_DIM = 384
TOP_K = 25
CANDIDATE_COUNT = 6000


def _build_query() -> list[float]:
    rng = random.Random(42)
    return [rng.random() for _ in range(EMBEDDING_DIM)]


def _build_candidates() -> list[list[float]]:
    rng = random.Random(7)
    return [
        [rng.random() for _ in range(EMBEDDING_DIM)] for _ in range(CANDIDATE_COUNT)
    ]


def _score_with_inline_norm(
    query_embedding: list[float], candidates: list[list[float]]
) -> list[float]:
    scores: list[float] = []
    for embedding in candidates:
        dot = 0.0
        qnorm = 0.0
        enorm = 0.0
        for q, c in zip(query_embedding, embedding):
            dot += c * q
            qnorm += q * q
            enorm += c * c
        if qnorm == 0.0 or enorm == 0.0:
            score = 0.0
        else:
            score = dot / (math.sqrt(qnorm) * math.sqrt(enorm))
        scores.append(score)
    scores.sort(reverse=True)
    return scores[:TOP_K]


def _score_with_precomputed_norm(
    query_embedding: list[float], candidates: list[list[float]]
) -> list[float]:
    scores: list[float] = []
    query_norm = math.sqrt(sum(value * value for value in query_embedding))
    for embedding in candidates:
        dot = 0.0
        enorm = 0.0
        for q, c in zip(query_embedding, embedding):
            dot += c * q
            enorm += c * c
        if query_norm == 0.0 or enorm == 0.0:
            score = 0.0
        else:
            score = dot / (query_norm * math.sqrt(enorm))
        scores.append(score)
    scores.sort(reverse=True)
    return scores[:TOP_K]


def _measure(
    label: str, fn, query_embedding: list[float], candidates: list[list[float]]
) -> dict[str, float | str | int]:
    durations_ms: list[float] = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        top = fn(query_embedding, candidates)
        assert len(top) == TOP_K
        durations_ms.append((time.perf_counter() - start) * 1000)

    return {
        "label": label,
        "iterations": ITERATIONS,
        "mean_ms": statistics.mean(durations_ms),
        "p50_ms": statistics.median(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "p99_ms": statistics.quantiles(durations_ms, n=100)[98],
        "ops_per_s": ITERATIONS / (sum(durations_ms) / 1000),
    }


def main() -> None:
    query_embedding = _build_query()
    candidates = _build_candidates()

    baseline = _measure(
        "baseline_per_row_query_norm",
        _score_with_inline_norm,
        query_embedding,
        candidates,
    )
    optimized = _measure(
        "optimized_precomputed_query_norm",
        _score_with_precomputed_norm,
        query_embedding,
        candidates,
    )

    for result in (baseline, optimized):
        print(result)

    speedup = (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    throughput_delta = (
        (optimized["ops_per_s"] - baseline["ops_per_s"]) / baseline["ops_per_s"] * 100
    )

    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> {optimized['mean_ms']:.3f}ms "
        f"({speedup:.1f}% faster)"
    )
    print(f"p50: {baseline['p50_ms']:.3f} -> {optimized['p50_ms']:.3f}")
    print(f"p95: {baseline['p95_ms']:.3f} -> {optimized['p95_ms']:.3f}")
    print(f"p99: {baseline['p99_ms']:.3f} -> {optimized['p99_ms']:.3f}")
    print(
        f"throughput: {baseline['ops_per_s']:.1f} -> "
        f"{optimized['ops_per_s']:.1f} req/s ({throughput_delta:.1f}%)"
    )


if __name__ == "__main__":
    main()
