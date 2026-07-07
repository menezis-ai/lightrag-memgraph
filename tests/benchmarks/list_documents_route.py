"""Micro-benchmark: WebUI document listing pipeline in doc-heavy folders.

Baseline mirrors the previous in-memory flow (query filter before membership filter),
and optimized applies `q` filtering first, then active-folder membership checks.

Run as a script:
```
python tests/benchmarks/list_documents_route.py
```
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from twindb_lightrag_memgraph.server.webui import router as webui_router

ITERATIONS = 60
DOC_COUNT = 3000
MEMBERSHIP_LOOKUP_SECONDS = 0.001
TAG_BATCH_BASE_SECONDS = 0.0008
TAG_DOC_SECOND = 0.00002


class FakeDocStatus:
    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self._membership: dict[str, list[str]] = {}
        for idx in range(DOC_COUNT):
            doc_id = f"doc-{idx:04d}"
            file_path = f"document-{idx:04d}"
            if idx % 20 == 0:
                file_path += " keep"
            self._docs[doc_id] = {
                "status": "processed",
                "file_path": file_path,
                "content_summary": f"document {idx} summary",
                "chunks_count": 1,
                "metadata": {"folder": "default" if idx % 3 else "shared"},
            }
            if idx % 3 == 0:
                self._membership[doc_id] = ["default"]
            elif idx % 7 == 0:
                self._membership[doc_id] = ["default", "sandbox"]
            else:
                self._membership[doc_id] = []

    async def get_docs_paginated(self, **kwargs: Any):
        # emulate current caller path: page=1, page_size=500
        page = int(kwargs.get("page", 1))
        page_size = int(kwargs.get("page_size", 500))
        status_filter = kwargs.get("status_filter")
        if status_filter is not None and str(status_filter) not in {"processed", "PROCESSED"}:
            return [], 0
        rows = list(self._docs.items())
        start = (page - 1) * page_size
        stop = start + page_size
        return rows[start:stop], len(rows)

    async def get_folders_for_doc(self, doc_id: str) -> list[str] | None:
        await asyncio.sleep(MEMBERSHIP_LOOKUP_SECONDS)
        return self._membership.get(doc_id)


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()


async def _fake_attach_graph_tags_for_documents(docs: list[dict[str, Any]]) -> None:
    if not docs:
        return
    await asyncio.sleep(
        TAG_BATCH_BASE_SECONDS + TAG_DOC_SECOND * len(docs)
    )
    for idx, doc in enumerate(docs):
        doc["tags"] = ["mock", str(idx)] if idx % 2 else ["keep"]


async def _baseline_list_documents(route: Any, *, q: str | None = None) -> list[dict[str, Any]]:
    rag = route._get_rag()
    folder = route.current_folder_id()
    docs_tuples, _total = await rag.doc_status.get_docs_paginated(page=1, page_size=500)

    rows: list[dict[str, Any]] = []
    for doc_id, raw in docs_tuples:
        payload = route._status_to_dict(raw)
        payload["id"] = doc_id
        rows.append(payload)

    if folder is not None:
        memberships_by_doc = await asyncio.gather(
            *(rag.doc_status.get_folders_for_doc(str(doc.get("doc_id") or doc.get("id") or "")) for doc in rows)
        )
        filtered_rows: list[dict[str, Any]] = []
        for row, memberships in zip(rows, memberships_by_doc):
            if memberships is None:
                if route._doc_row_has_active_folder_hint(row, folder=folder):
                    filtered_rows.append(row)
                continue
            if folder in memberships:
                filtered_rows.append(row)
        rows = filtered_rows

    docs = [route._project_doc_status_for_webui(row, visible_folder=folder) for row in rows]
    await _fake_attach_graph_tags_for_documents(docs)
    return route._filter_doc_status_rows(docs, q=q, tag=None, folder=None)


async def _optimized_list_documents(route: Any, *, q: str | None = None) -> list[dict[str, Any]]:
    rag = route._get_rag()
    folder = route.current_folder_id()
    docs_tuples, _total = await rag.doc_status.get_docs_paginated(page=1, page_size=500)

    docs = [
        route._project_doc_status_for_webui(
            {**route._status_to_dict(raw), "id": doc_id},
            visible_folder=folder,
        )
        for doc_id, raw in docs_tuples
    ]

    if q:
        docs = [doc for doc in docs if route._doc_matches_query(doc, q=q)]
    if folder is not None:
        docs = await route._filter_docs_to_active_folder(docs, folder=folder, rag=rag)

    await _fake_attach_graph_tags_for_documents(docs)
    return route._filter_doc_status_rows(docs, q=None, tag=None, folder=None)


async def _measure(label: str, fn) -> dict[str, float | str | int]:
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()
    route = webui_router
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        output = await fn(route, q="keep")
        assert len(output) > 0
        durations_ms.append((time.perf_counter() - start) * 1000)
    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    sorted_ms = sorted(durations_ms)
    p50_idx = max(int(len(sorted_ms) * 0.5) - 1, 0)
    p95_idx = max(int(len(sorted_ms) * 0.95) - 1, 0)
    p99_idx = max(int(len(sorted_ms) * 0.99) - 1, 0)
    return {
        "label": label,
        "iterations": ITERATIONS,
        "mean_ms": statistics.mean(durations_ms),
        "p50_ms": sorted_ms[p50_idx],
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "p99_ms": sorted_ms[p99_idx],
        "ops_per_s": ITERATIONS / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


async def main() -> None:
    route = webui_router

    route._get_rag = lambda: FakeRag()  # type: ignore[method-assign]
    route.current_folder_id = lambda: "default"  # type: ignore[method-assign]

    baseline = await _measure("baseline_query_then_tags", _baseline_list_documents)
    optimized = await _measure("optimized_query_first", _optimized_list_documents)

    for result in (baseline, optimized):
        print(result)

    speedup = (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    throughput_delta = (
        (optimized["ops_per_s"] - baseline["ops_per_s"]) / baseline["ops_per_s"] * 100
    )
    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> {optimized['mean_ms']:.3f}ms ({speedup:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['ops_per_s']:.1f} req/s -> {optimized['ops_per_s']:.1f} req/s ({throughput_delta:.1f}%)"
    )
    print(f"peak_mem: {baseline['peak_mb']:.3f}MB -> {optimized['peak_mb']:.3f}MB")


if __name__ == "__main__":
    asyncio.run(main())
