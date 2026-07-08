"""Micro-benchmark: sequential vs parallel bulk document deletion.

Run as a script:
```
python tests/benchmarks/bulk_delete_documents.py
```
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from typing import Any

from fastapi import HTTPException

from twindb_lightrag_memgraph.server.webui import routes_documents as rd

ITERATIONS = 40
DOC_COUNT = 100
LOOKUP_DELAY_SECONDS = 0.004


class _FakeDocStatus:
    def __init__(self, docs: list[str], folder: str = "default") -> None:
        self._folders: dict[str, list[str]] = {doc_id: [folder] for doc_id in docs}

    async def get_folders_for_doc(self, doc_id: str) -> list[str] | None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        return self._folders.get(doc_id)

    async def remove_from_folder(self, doc_id: str, folder: str) -> None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        folders = self._folders.get(doc_id)
        if folders is None:
            return
        self._folders[doc_id] = [value for value in folders if value != folder]
        if not self._folders[doc_id]:
            self._folders.pop(doc_id, None)

    async def delete(self, ids: list[str]) -> None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        for doc_id in ids:
            self._folders.pop(doc_id, None)


class _FakeRag:
    def __init__(self, docs: list[str], folder: str = "default") -> None:
        self.doc_status = _FakeDocStatus(docs, folder)


class _FakeLegacy:
    def __init__(self, rag: _FakeRag, folder: str = "default") -> None:
        self._rag = rag
        self._folder = folder

    def current_folder_id(self) -> str:
        return self._folder

    async def _get_doc_for_active_folder(self, doc_id: str) -> dict[str, Any]:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        folders = await self._rag.doc_status.get_folders_for_doc(doc_id)
        if not folders or self._folder not in folders:
            raise HTTPException(404, f"Document '{doc_id}' not found.")
        return {"id": doc_id, "file_path": f"/kb/{doc_id}.pdf"}

    async def _delete_doc_from_rag(self, rag: _FakeRag, doc_id: str) -> None:
        await asyncio.sleep(LOOKUP_DELAY_SECONDS)
        await rag.doc_status.delete([doc_id])


def _build_payload(doc_count: int = DOC_COUNT) -> list[str]:
    return [f"doc-{idx:03d}" for idx in range(doc_count)]


async def _make_env(
    doc_count: int = DOC_COUNT,
) -> tuple[_FakeLegacy, _FakeRag, list[str]]:
    doc_ids = _build_payload(doc_count)
    rag = _FakeRag(doc_ids)
    legacy = _FakeLegacy(rag)
    return legacy, rag, doc_ids


async def _baseline_bulk_delete(
    legacy: _FakeLegacy, rag: _FakeRag, doc_ids: list[str]
) -> list[dict[str, Any]]:
    del rag
    results: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        try:
            result = await rd._delete_one_document(legacy, legacy._rag, doc_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                continue
            raise
        if result is not None:
            results.append(result)
    return results


async def _optimized_bulk_delete(
    legacy: _FakeLegacy, rag: _FakeRag, doc_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    del rag
    return await rd._run_bulk_delete_batch(legacy, legacy._rag, doc_ids)


async def _measure(label: str, fn) -> dict[str, float | str | int]:
    durations_ms: list[float] = []
    tracemalloc.start()
    start_total = time.perf_counter()

    for _ in range(ITERATIONS):
        legacy, rag, doc_ids = await _make_env()
        start = time.perf_counter()
        result = await fn(legacy, rag, doc_ids)
        assert len(doc_ids) >= 0
        if isinstance(result, tuple):
            deleted, _failed = result
            assert len(deleted) == DOC_COUNT
            assert not _failed
        else:
            assert len(result) == DOC_COUNT
        durations_ms.append((time.perf_counter() - start) * 1000)

    elapsed = time.perf_counter() - start_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "iterations": ITERATIONS,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "ops_per_s": ITERATIONS / elapsed,
        "peak_mb": peak / 1024 / 1024,
    }


async def main() -> None:
    baseline = await _measure("baseline_sequential", _baseline_bulk_delete)
    optimized = await _measure("optimized_parallel", _optimized_bulk_delete)

    for result in (baseline, optimized):
        print(result)

    mean_delta = (
        (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    )
    throughput_delta = (
        (optimized["ops_per_s"] - baseline["ops_per_s"]) / baseline["ops_per_s"] * 100
    )
    print()
    print("## SUMMARY")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> {optimized['mean_ms']:.3f}ms "
        f"({mean_delta:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['ops_per_s']:.1f} req/s -> "
        f"{optimized['ops_per_s']:.1f} req/s ({throughput_delta:.1f}%)"
    )
    print(f"peak_mem: {baseline['peak_mb']:.3f}MB -> {optimized['peak_mb']:.3f}MB")


if __name__ == "__main__":
    asyncio.run(main())
