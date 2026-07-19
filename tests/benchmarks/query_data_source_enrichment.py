"""Benchmark: dense KG source-id enrichment for ``/query/data``.

Run as a script:
``python tests/benchmarks/query_data_source_enrichment.py``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.query import query_data

ITERATIONS = 32
KG_ROW_COUNT = 2000
UNIQUE_CHUNK_COUNT = 480
EXISTING_CHUNK_COUNT = UNIQUE_CHUNK_COUNT
LOAD_CONCURRENCY = 4


class _Store:
    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": chunk_id,
                "full_doc_id": f"doc-{chunk_id}",
                "file_path": f"/kb/{chunk_id}.pdf",
            }
            for chunk_id in chunk_ids
        ]


class _Rag:
    text_chunks = _Store()
    chunks_vdb = _Store()


def _build_response() -> dict[str, Any]:
    existing_chunks = [
        {
            "chunk_id": f"chunk-{idx}",
            "full_doc_id": f"doc-chunk-{idx}",
            "file_path": f"/kb/chunk-{idx}.pdf",
            "reference_id": str(idx + 1),
        }
        for idx in range(EXISTING_CHUNK_COUNT)
    ]
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for idx in range(KG_ROW_COUNT):
        row = {
            "source_id": "<SEP>".join(
                [
                    f"chunk-{idx % UNIQUE_CHUNK_COUNT}",
                    f"chunk-{(idx + 17) % UNIQUE_CHUNK_COUNT}",
                    f"chunk-{(idx + 113) % UNIQUE_CHUNK_COUNT}",
                ]
            ),
            "similarity": 1.0 - (idx % 100) / 1000,
            "reference_id": f"kg-{idx}",
        }
        if idx % 2:
            relationships.append(row)
        else:
            entities.append(row)
    return {
        "status": "success",
        "data": {
            "chunks": existing_chunks,
            "entities": entities,
            "relationships": relationships,
            "references": [
                {"reference_id": str(idx + 1), "file_path": f"/kb/chunk-{idx}.pdf"}
                for idx in range(EXISTING_CHUNK_COUNT)
            ],
        },
        "metadata": {},
    }


def _baseline_query_data_source_chunk_ids(data: dict[str, Any]) -> list[str]:
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for row in query_data._iter_kg_rows(data):
        for chunk_id in query_data._split_source_ids(row.get("source_id")):
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
    return chunk_ids


def _baseline_query_data_existing_chunk_ids(data: dict[str, Any]) -> set[str]:
    rows = data.get("chunks")
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("chunk_id", "id", "_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                out.add(value)
    return out


def _baseline_query_data_source_chunk_scores(data: dict[str, Any]) -> dict[str, float]:
    total = sum(1 for _ in query_data._iter_kg_rows(data))
    if total == 0:
        return {}

    scores: dict[str, float] = {}
    for rank, row in enumerate(query_data._iter_kg_rows(data)):
        chunk_ids = query_data._split_source_ids(row.get("source_id"))
        if not chunk_ids:
            continue
        score = query_data._safe_get_score(row, rank, total)
        for chunk_id in chunk_ids:
            scores[chunk_id] = max(scores.get(chunk_id, score), score)
    return scores


def _baseline_annotate_query_data_chunk_scores(
    data: dict[str, Any],
) -> dict[str, Any]:
    rows = data.get("chunks")
    if not isinstance(rows, list):
        return data

    source_scores = _baseline_query_data_source_chunk_scores(data)
    total = len(rows)
    annotated_rows: list[Any] = []
    changed = False
    for rank, row in enumerate(rows):
        if not isinstance(row, dict):
            annotated_rows.append(row)
            continue
        annotated = dict(row)
        chunk_id = query_data._query_data_chunk_id(annotated)
        if not isinstance(annotated.get("score"), (int, float)):
            if chunk_id and chunk_id in source_scores:
                annotated["score"] = source_scores[chunk_id]
            else:
                annotated["score"] = query_data._safe_get_score(annotated, rank, total)
            changed = True
        annotated_rows.append(annotated)

    if not changed:
        return data
    annotated_data = dict(data)
    annotated_data["chunks"] = annotated_rows
    return annotated_data


async def _baseline_enrich_query_data_chunks_from_source_ids(
    rag: Any,
    response: dict[str, Any],
) -> dict[str, Any]:
    data = response.get("data")
    if not isinstance(data, dict):
        return response

    missing = [
        chunk_id
        for chunk_id in _baseline_query_data_source_chunk_ids(data)
        if chunk_id not in _baseline_query_data_existing_chunk_ids(data)
    ]
    if not missing:
        enriched = dict(response)
        enriched["data"] = _baseline_annotate_query_data_chunk_scores(data)
        return enriched

    records = await query_data._fetch_chunk_records_by_id(rag, missing)
    if not records:
        enriched = dict(response)
        enriched["data"] = _baseline_annotate_query_data_chunk_scores(data)
        return enriched

    enriched_data = dict(data)
    source_scores = _baseline_query_data_source_chunk_scores(data)
    raw_chunks = data.get("chunks")
    raw_references = data.get("references")
    chunks = list(raw_chunks) if isinstance(raw_chunks, list) else []
    references = list(raw_references) if isinstance(raw_references, list) else []
    next_ref = query_data._next_query_data_reference_id(data)
    chunk_rows_to_fetch: list[Any] = []
    for chunk_id in missing:
        raw = records.get(chunk_id)
        if not isinstance(raw, dict):
            continue
        ref_id = str(next_ref)
        next_ref += 1
        chunk_rows_to_fetch.append(
            query_data._query_data_chunk_row(
                rag,
                chunk_id,
                raw,
                ref_id,
                source_scores.get(chunk_id),
            )
        )
    if chunk_rows_to_fetch:
        for chunk_row, reference in await asyncio.gather(*chunk_rows_to_fetch):
            chunks.append(chunk_row)
            references.append(reference)

    enriched_data["chunks"] = chunks
    enriched_data["references"] = references
    enriched_data = _baseline_annotate_query_data_chunk_scores(enriched_data)
    enriched = dict(response)
    enriched["data"] = enriched_data
    return enriched


def _assert_parity(result: dict[str, Any]) -> None:
    data = result["data"]
    assert len(data["chunks"]) == UNIQUE_CHUNK_COUNT
    assert len(data["references"]) == UNIQUE_CHUNK_COUNT
    assert {chunk["chunk_id"] for chunk in data["chunks"]} == {
        f"chunk-{idx}" for idx in range(UNIQUE_CHUNK_COUNT)
    }
    assert all(isinstance(chunk.get("score"), float) for chunk in data["chunks"])


async def _one_request(fn, response: dict[str, Any]) -> None:
    result = await fn(_Rag(), response)
    _assert_parity(result)


async def _measure_isolated(label: str, fn) -> dict[str, Any]:
    durations_ms: list[float] = []
    response = _build_response()
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        await _one_request(fn, response)
        durations_ms.append((time.perf_counter() - start) * 1000)
    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _result(label, durations_ms, elapsed, ITERATIONS)


async def _measure_load(label: str, fn, concurrency: int) -> dict[str, Any]:
    durations_ms: list[float] = []
    response = _build_response()
    tracemalloc.start()
    start_total = time.perf_counter()

    async def worker() -> None:
        start = time.perf_counter()
        await _one_request(fn, response)
        durations_ms.append((time.perf_counter() - start) * 1000)

    for _ in range(ITERATIONS // concurrency):
        await asyncio.gather(*(worker() for _ in range(concurrency)))

    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _result(label, durations_ms, elapsed, len(durations_ms), concurrency)


def _result(
    label: str,
    durations_ms: list[float],
    elapsed: float,
    requests: int,
    concurrency: int | None = None,
) -> dict[str, Any]:
    durations = sorted(durations_ms)
    return {
        "label": label,
        "requests": requests,
        "concurrency": concurrency,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": durations[max(int(len(durations) * 0.95) - 1, 0)],
        "p99_ms": durations[max(int(len(durations) * 0.99) - 1, 0)],
        "req_per_s": requests / elapsed,
    }


def _print_result(result: dict[str, Any]) -> None:
    load = (
        f" concurrency={result['concurrency']}"
        if result.get("concurrency") is not None
        else ""
    )
    print(
        f"{result['label']}:{load} mean={result['mean_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms p99={result['p99_ms']:.3f}ms "
        f"throughput={result['req_per_s']:.1f} req/s"
    )


async def main() -> None:
    baseline = await _measure_isolated(
        "baseline_repeated_kg_scans",
        _baseline_enrich_query_data_chunks_from_source_ids,
    )
    optimized = await _measure_isolated(
        "optimized_reused_kg_projection",
        query_data._enrich_query_data_chunks_from_source_ids,
    )
    baseline_load = await _measure_load(
        "baseline_sustained_load",
        _baseline_enrich_query_data_chunks_from_source_ids,
        LOAD_CONCURRENCY,
    )
    optimized_load = await _measure_load(
        "optimized_sustained_load",
        query_data._enrich_query_data_chunks_from_source_ids,
        LOAD_CONCURRENCY,
    )

    for result in (baseline, optimized, baseline_load, optimized_load):
        _print_result(result)

    speedup = (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    throughput_delta = (
        (optimized["req_per_s"] - baseline["req_per_s"]) / baseline["req_per_s"] * 100
    )
    load_speedup = (
        (baseline_load["mean_ms"] - optimized_load["mean_ms"])
        / baseline_load["mean_ms"]
        * 100
    )
    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> "
        f"{optimized['mean_ms']:.3f}ms ({speedup:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['req_per_s']:.1f} req/s -> "
        f"{optimized['req_per_s']:.1f} req/s ({throughput_delta:.1f}%)"
    )
    print(
        f"load mean: {baseline_load['mean_ms']:.3f}ms -> "
        f"{optimized_load['mean_ms']:.3f}ms ({load_speedup:.1f}% faster)"
    )


if __name__ == "__main__":
    asyncio.run(main())
