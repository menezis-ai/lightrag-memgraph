"""Benchmark reuse of the authorized DocStatus in chunk expansion routes.

Run standalone with::

    uv run python tests/benchmarks/chunk_route_docstatus_reuse.py

The in-process baseline preserves the pre-optimization route body: authorize the
document, then fetch the same DocStatus again for ``chunks_list``.  The live path
calls the route endpoint registered from :mod:`server.chunk_routes`.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from twindb_lightrag_memgraph.server import chunk_routes

ITERATIONS = 80
CHUNK_COUNT = 40
READ_DELAY_SECONDS = 0.004


class _DocStatus:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._value = {"chunks_list": chunk_ids, "folder": "default"}
        self.get_by_id_calls = 0
        self.get_folders_calls = 0

    async def get_by_id(self, _doc_id: str) -> dict[str, Any]:
        self.get_by_id_calls += 1
        await asyncio.sleep(READ_DELAY_SECONDS)
        return self._value

    async def get_folders_for_doc(self, _doc_id: str) -> list[str]:
        self.get_folders_calls += 1
        await asyncio.sleep(READ_DELAY_SECONDS)
        return ["default"]


class _TextChunks:
    def __init__(self, chunk_ids: list[str]) -> None:
        self._chunks = {
            chunk_id: {
                "_id": chunk_id,
                "content": f"content-{chunk_id}",
                "full_doc_id": "doc-1",
                "file_path": "document.pdf",
                "chunk_order_index": index,
                "tokens": 20,
            }
            for index, chunk_id in enumerate(chunk_ids)
        }

    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        await asyncio.sleep(READ_DELAY_SECONDS)
        return [self._chunks[chunk_id] for chunk_id in chunk_ids]


class _Rag:
    def __init__(self) -> None:
        chunk_ids = [f"chunk-{index}" for index in range(CHUNK_COUNT)]
        self.doc_status = _DocStatus(chunk_ids)
        self.text_chunks = _TextChunks(chunk_ids)


_active_rag: ContextVar[_Rag] = ContextVar("chunk_benchmark_rag")
_live_endpoint: Callable[..., Awaitable[Any]] | None = None


@dataclass
class _Sample:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float


def _document_chunks_endpoint() -> Callable[..., Awaitable[Any]]:
    for route in chunk_routes.router.routes:
        if getattr(route, "path", None) == "/documents/{doc_id}/chunks":
            return route.endpoint
    raise RuntimeError("document chunks route was not registered")


def _ensure_live_endpoint() -> Callable[..., Awaitable[Any]]:
    global _live_endpoint
    if _live_endpoint is None:
        chunk_routes.create_chunk_routes(lambda: _active_rag.get())
        _live_endpoint = _document_chunks_endpoint()
    return _live_endpoint


async def _baseline_request(rag: _Rag) -> chunk_routes.ChunkContextResponse:
    """Pre-optimization route body with two DocStatus reads."""
    doc_id = "doc-1"
    await chunk_routes._require_doc_in_active_folder(rag, doc_id)
    ordered_ids = await chunk_routes._get_ordered_chunk_ids(rag, doc_id)
    items = await chunk_routes._fetch_chunks_by_ids(rag, ordered_ids)
    return chunk_routes.ChunkContextResponse(
        chunks=items,
        doc_id=doc_id,
        file_path=items[0].file_path if items else "",
        total_chunks_in_doc=len(ordered_ids),
    )


async def _live_request(rag: _Rag) -> chunk_routes.ChunkContextResponse:
    endpoint = _ensure_live_endpoint()
    token = _active_rag.set(rag)
    try:
        return await endpoint("doc-1", None, None)
    finally:
        _active_rag.reset(token)


async def _time_requests(
    request: Callable[[_Rag], Awaitable[chunk_routes.ChunkContextResponse]],
    *,
    iterations: int,
    concurrency: int = 1,
) -> _Sample:
    durations: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one() -> None:
        async with semaphore:
            rag = _Rag()
            started = time.perf_counter()
            response = await request(rag)
            elapsed_ms = (time.perf_counter() - started) * 1000
            assert response.total_chunks_in_doc == CHUNK_COUNT
            assert [item.chunk_id for item in response.chunks] == [
                f"chunk-{index}" for index in range(CHUNK_COUNT)
            ]
            durations.append(elapsed_ms)

    started = time.perf_counter()
    await asyncio.gather(*(run_one() for _ in range(iterations)))
    elapsed = time.perf_counter() - started
    ordered = sorted(durations)

    def percentile(percent: float) -> float:
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percent) - 1))
        return ordered[index]

    return _Sample(
        mean_ms=statistics.mean(durations),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        requests_per_second=iterations / elapsed,
    )


async def measure(iterations: int | None = None) -> list[dict[str, Any]]:
    count = iterations or ITERATIONS

    parity_rag = _Rag()
    baseline_response = await _baseline_request(parity_rag)
    live_rag = _Rag()
    live_response = await _live_request(live_rag)

    baseline = await _time_requests(_baseline_request, iterations=count)
    optimized = await _time_requests(_live_request, iterations=count)

    return [
        {
            "name": "chunk route latency",
            "kind": "ratio",
            "baseline_ms": baseline.mean_ms,
            "optimized_ms": optimized.mean_ms,
            "detail": (
                f"p95 {baseline.p95_ms:.3f}->{optimized.p95_ms:.3f}ms; "
                f"p99 {baseline.p99_ms:.3f}->{optimized.p99_ms:.3f}ms"
            ),
        },
        {
            "name": "authorized DocStatus is reused",
            "kind": "structural",
            "passed": (
                live_rag.doc_status.get_by_id_calls == 1
                and live_response == baseline_response
            ),
            "detail": (
                "expected one DocStatus lookup with identical response; "
                f"observed {live_rag.doc_status.get_by_id_calls} lookup(s)"
            ),
        },
    ]


async def _load_test(iterations: int, concurrency: int) -> tuple[_Sample, _Sample]:
    baseline = await _time_requests(
        _baseline_request,
        iterations=iterations,
        concurrency=concurrency,
    )
    optimized = await _time_requests(
        _live_request,
        iterations=iterations,
        concurrency=concurrency,
    )
    return baseline, optimized


def _print_sample(label: str, sample: _Sample) -> None:
    print(
        f"{label}: mean={sample.mean_ms:.3f}ms "
        f"p95={sample.p95_ms:.3f}ms p99={sample.p99_ms:.3f}ms "
        f"throughput={sample.requests_per_second:.1f} req/s"
    )


async def main() -> None:
    cases = await measure()
    ratio = cases[0]
    structural = cases[1]
    gain = (ratio["baseline_ms"] - ratio["optimized_ms"]) / ratio["baseline_ms"] * 100
    print(
        f"sequential: {ratio['baseline_ms']:.3f}ms -> "
        f"{ratio['optimized_ms']:.3f}ms ({gain:.1f}% faster)"
    )
    print(f"structural: {structural['passed']} ({structural['detail']})")

    for label, concurrency, requests in (
        ("sustained", 8, 160),
        ("peak", 32, 320),
    ):
        baseline, optimized = await _load_test(requests, concurrency)
        _print_sample(f"{label} baseline", baseline)
        _print_sample(f"{label} optimized", optimized)


if __name__ == "__main__":
    asyncio.run(main())
