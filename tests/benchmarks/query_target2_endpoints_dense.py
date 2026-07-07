"""Target 2 micro-benchmarks for dense `/twin/api/query*` payloads.

Compares legacy-style baseline implementations (in-script) against the
optimized in-tree implementations after this change.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.query import query_data, query_data_filters as qdf, response_sources as rs
from twindb_lightrag_memgraph.server.query import doc_lookup

ITERATIONS = 80
SOURCE_COUNT = 2000
ROW_COUNT = 1200
LOOKUP_DELAY_SECONDS = 0.002


async def fake_fetch_doc_tags(doc_id: str, _folder: str) -> set[str]:
    await asyncio.sleep(LOOKUP_DELAY_SECONDS)
    return {"keep" if not doc_id.endswith("-drop") else "drop"}


class Rag:
    def __init__(self) -> None:
        self.seen_chunks: list[str] = []
        self.seen_file_paths: list[str] = []

    async def _resolve_doc_for_chunk(self, chunk_id: str) -> str | None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        self.seen_chunks.append(chunk_id)
        if chunk_id.endswith("-missing"):
            return None
        return f"doc-{chunk_id}"

    async def _resolve_doc_for_file_path(self, file_path: str) -> str | None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        self.seen_file_paths.append(file_path)
        if file_path.endswith("/missing"):
            return None
        return f"doc-{file_path.split('/')[-1]}"

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for chunk_id in ids:
            await asyncio.sleep(0)
            out.append(
                {
                    "chunk_id": chunk_id,
                    "file_path": f"doc-{chunk_id}.pdf",
                }
            )
        return out


# --- Payload builders --------------------------------------------------------


def build_dense_sources() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(SOURCE_COUNT):
        doc_idx = i % 120
        out.append(
            {
                "doc_id": f"doc-{doc_idx}" if i % 17 else f"doc-{doc_idx}-drop",
                "name": f"doc-{doc_idx}.pdf",
                "score": 0.9,
                "chunk_id": f"chunk-{i % 80}",
                "n": i + 1,
            }
        )
    return out


def build_dense_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(ROW_COUNT):
        chunks = [f"chunk-{i % 90}", f"chunk-{(i + 1) % 90}"]
        rows.append(
            {
                "chunk_id": f"chunk-{i}",
                "source_id": ",".join(chunks),
                "file_path": f"/kb/file-{i % 150}.pdf",
                "source": f"/kb/source-{i % 90}.pdf",
                "name": f"{i}.pdf",
                "reference_id": str(i),
                "doc_id": f"doc-{i % 120}" if i % 11 else None,
            }
        )
    return rows


def build_query_data_payload() -> dict[str, Any]:
    chunks = [
        {
            "chunk_id": f"chunk-{i}",
            "full_doc_id": f"doc-chunk-{i}",
            "file_path": f"/kb/c-{i}.pdf",
            "reference_id": str(i),
        }
        for i in range(25)
    ]
    source_ids = [",".join([f"chunk-{i}", f"chunk-{i + 1}", f"chunk-{i + 200}"]) for i in range(90)]
    return {
        "chunks": chunks,
        "entities": [
            {"source_id": source_ids[0], "reference_id": "e0", "score": 0.9},
            {"source_id": source_ids[1], "reference_id": "e1", "score": 0.6},
            {"source_id": source_ids[2], "reference_id": "e2", "score": 0.4},
        ],
        "relationships": [],
        "references": [{"reference_id": "0"}, {"reference_id": "1"}],
    }


def generate_query_response() -> dict[str, Any]:
    return {
        "status": "success",
        "data": build_query_data_payload(),
        "metadata": {},
    }


# --- Baselines --------------------------------------------------------------

async def baseline_filter_sources_by_advanced_filters(
    sources: list[dict[str, Any]],
    *,
    tag_filter: dict[str, list[str]] | None,
    doc_filter: dict[str, list[str]] | None,
    folder: str,
    fetch_doc_tags: Any,
) -> tuple[list[dict[str, Any]], bool]:
    from twindb_lightrag_memgraph.server.query.source_filters import (
        _tag_filter_terms,
        _doc_filter_terms,
        _source_matches_doc_filter,
        _doc_tags_match_filter,
    )

    tag_required, tag_optional = _tag_filter_terms(tag_filter)
    doc_required, doc_optional = _doc_filter_terms(doc_filter)
    if not tag_required and not tag_optional and not doc_required and not doc_optional:
        return sources, False

    async def source_matches_tag_filter(source: dict[str, Any], tags_cache: dict[str, set[str]]):
        if not tag_required and not tag_optional:
            return True
        doc_id = source.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            return False
        if doc_id not in tags_cache:
            tags_cache[doc_id] = await fetch_doc_tags(doc_id, folder)
        return _doc_tags_match_filter(tags_cache[doc_id], tag_filter)

    tags_cache: dict[str, set[str]] = {}
    kept = []
    has_unverified = False
    for source in sources:
        if not _source_matches_doc_filter(source, doc_filter):
            if not source.get("doc_id"):
                has_unverified = True
            continue
        if not await source_matches_tag_filter(source, tags_cache):
            if not source.get("doc_id"):
                has_unverified = True
            continue
        kept.append(source)
    return kept, has_unverified


async def baseline_filter_rows_by_tags(
    rag: Any,
    rows: list[dict[str, Any]],
    tag_filter: dict[str, list[str]] | None,
    folder: str,
    tags_cache: dict[str, set[str]],
    fetch_doc_tags: Any,
) -> tuple[list[dict[str, Any]], set[str]]:
    kept: list[dict[str, Any]] = []
    kept_reference_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if await qdf._row_matches_tag_filter(rag, row, tag_filter, folder, tags_cache, fetch_doc_tags):
            kept.append(row)
            ref_id = row.get("reference_id")
            if isinstance(ref_id, str) and ref_id:
                kept_reference_ids.add(ref_id)
    return kept, kept_reference_ids


async def baseline_enrich_query_data_chunks(rag: Any, response: dict[str, Any]) -> dict[str, Any]:
    data = response["data"]
    missing = query_data._query_data_source_chunk_ids(data)
    source_scores = query_data._query_data_source_chunk_scores(data)

    records = await query_data._fetch_chunk_records_by_id(rag, missing)
    if not records:
        enriched = dict(response)
        enriched["data"] = query_data._annotate_query_data_chunk_scores(data)
        return enriched

    enriched_data = dict(data)
    chunks = list(data.get("chunks") or [])
    references = list(data.get("references") or [])
    next_ref = query_data._next_query_data_reference_id(data)

    for chunk_id in missing:
        raw = records.get(chunk_id)
        if not isinstance(raw, dict):
            continue
        ref_id = str(next_ref)
        next_ref += 1
        chunk_row, reference = await query_data._query_data_chunk_row(
            rag,
            chunk_id,
            raw,
            ref_id,
            source_scores.get(chunk_id),
        )
        chunks.append(chunk_row)
        references.append(reference)

    enriched_data["chunks"] = chunks
    enriched_data["references"] = references
    enriched_data = query_data._annotate_query_data_chunk_scores(enriched_data)
    enriched = dict(response)
    enriched["data"] = enriched_data
    return enriched


# --- Benchmark harness -------------------------------------------------------


def _measure_once(coro):
    return asyncio.create_task(coro)


async def measure(label: str, fn, *args, **kwargs) -> dict[str, float | int | str]:
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        out = await fn(*args, **kwargs)
        assert out is not None
        durations_ms.append((time.perf_counter() - start) * 1000)
    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    durations = sorted(durations_ms)
    idx95 = max(int(len(durations) * 0.95) - 1, 0)
    idx99 = max(int(len(durations) * 0.99) - 1, 0)
    return {
        "label": label,
        "iterations": ITERATIONS,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": durations[idx95],
        "p99_ms": durations[idx99],
        "ops_per_s": ITERATIONS / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


async def bench_query_like():
    print("# /twin/api/query (+ /query/stream shared filtering path)")
    sources = build_dense_sources()

    base = await measure(
        "query_baseline",
        baseline_filter_sources_by_advanced_filters,
        sources,
        tag_filter={"all": ["keep"]},
        doc_filter=None,
        folder="default",
        fetch_doc_tags=fake_fetch_doc_tags,
    )
    optim = await measure(
        "query_optimized",
        rs._filter_sources_by_advanced_filters,
        sources,
        tag_filter={"all": ["keep"]},
        doc_filter=None,
        folder="default",
        fetch_doc_tags=fake_fetch_doc_tags,
    )
    print(base)
    print(optim)

    speedup = (base["mean_ms"] - optim["mean_ms"]) / base["mean_ms"] * 100
    print(f"mean: {base['mean_ms']:.3f} -> {optim['mean_ms']:.3f} ({speedup:.1f}% faster)")
    print(f"p99: {base['p99_ms']:.3f} -> {optim['p99_ms']:.3f}")


async def bench_query_data_filters():
    print("\n# /twin/api/query/data")
    rows = build_dense_rows()
    rag = Rag()

    base = await measure(
        "query_data_filter_baseline",
        baseline_filter_rows_by_tags,
        rag,
        rows,
        tag_filter={"all": ["keep"]},
        folder="default",
        tags_cache={},
        fetch_doc_tags=fake_fetch_doc_tags,
    )

    async def row_filter(rag_in: Any):
        return await qdf._filter_rows_by_tags(
            rag_in,
            rows,
            tag_filter={"all": ["keep"]},
            folder="default",
            tags_cache={},
            fetch_doc_tags=fake_fetch_doc_tags,
        )

    optim = await measure("query_data_filter_optimized", row_filter, Rag())

    print(base)
    print(optim)
    speedup = (base["mean_ms"] - optim["mean_ms"]) / base["mean_ms"] * 100
    print(f"mean: {base['mean_ms']:.3f} -> {optim['mean_ms']:.3f} ({speedup:.1f}% faster)")


async def bench_query_data_enrich():
    print("\n# /twin/api/query/data chunk enrichment")
    rag = Rag()
    response = generate_query_response()

    base = await measure(
        "query_data_enrich_baseline",
        baseline_enrich_query_data_chunks,
        rag,
        response,
    )
    optim = await measure(
        "query_data_enrich_optimized",
        query_data._enrich_query_data_chunks_from_source_ids,
        rag,
        response,
    )

    print(base)
    print(optim)
    speedup = (base["mean_ms"] - optim["mean_ms"]) / base["mean_ms"] * 100
    print(f"mean: {base['mean_ms']:.3f} -> {optim['mean_ms']:.3f} ({speedup:.1f}% faster)")


async def main() -> None:
    async def fake_resolve_chunk(_rag: Any, chunk_id: str) -> str | None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        if chunk_id.endswith("-missing"):
            return None
        return f"doc-{chunk_id}"

    async def fake_resolve_file_path(_rag: Any, file_path: str) -> str | None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        if file_path.endswith("/missing"):
            return None
        return f"doc-{file_path.split('/')[-1]}"

    qdf._resolve_doc_for_chunk = fake_resolve_chunk
    qdf._resolve_doc_for_file_path = fake_resolve_file_path
    doc_lookup._resolve_doc_for_chunk = fake_resolve_chunk
    doc_lookup._resolve_doc_for_file_path = fake_resolve_file_path

    await bench_query_like()
    await bench_query_data_filters()
    await bench_query_data_enrich()


if __name__ == "__main__":
    asyncio.run(main())
