"""Micro-benchmark: document membership lock map growth and lock reuse.

Run as a script:
```
python tests/benchmarks/membership_lock_cache.py
```
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import statistics
import time
import tracemalloc
from typing import Dict


ITERATIONS = 10
DOC_COUNT = 3000
_MAX_MEMBERSHIP_LOCKS = 2048
_MEMBERSHIP_LOCK_CLEANUP_EVERY = 1024


class _BaselineLocker:
    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}

    def lock(self, doc_id: str) -> asyncio.Lock:
        lock = self._locks.get(doc_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[doc_id] = lock
        return lock

    @property
    def size(self) -> int:
        return len(self._locks)


class _BoundedLocker:
    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = OrderedDict()
        self._access = 0

    def _evict(self) -> None:
        if len(self._locks) <= _MAX_MEMBERSHIP_LOCKS:
            return
        for doc_id in list(self._locks.keys()):
            if len(self._locks) <= _MAX_MEMBERSHIP_LOCKS:
                return
            lock = self._locks[doc_id]
            if lock.locked():
                continue
            self._locks.pop(doc_id, None)

    def lock(self, doc_id: str) -> asyncio.Lock:
        lock = self._locks.get(doc_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[doc_id] = lock
        else:
            self._locks.move_to_end(doc_id)

        self._access += 1
        if self._access % _MEMBERSHIP_LOCK_CLEANUP_EVERY == 0:
            self._evict()

        return lock

    @property
    def size(self) -> int:
        return len(self._locks)


async def _acquire_once(locker, doc_id: str) -> None:
    lock = locker.lock(doc_id)
    async with lock:
        return


async def _measure(locker, label: str) -> dict[str, float | int]:
    durations_ms: list[float] = []
    for run in range(ITERATIONS):
        start = time.perf_counter()
        for idx in range(DOC_COUNT):
            await _acquire_once(locker, f"doc-{run}-{idx}")
        await asyncio.gather(*(_acquire_once(locker, "doc-shared") for _ in range(16)))
        durations_ms.append((time.perf_counter() - start) * 1000)

    return {
        "label": label,
        "iterations": ITERATIONS,
        "doc_count": DOC_COUNT,
        "size": locker.size,
        "mean_ms": statistics.mean(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "ops_per_s": ITERATIONS / (sum(durations_ms) / 1000),
    }


async def main() -> None:
    tracemalloc.start()
    baseline = await _measure(_BaselineLocker(), "baseline_unbounded")
    _, peak_base = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    optimized = await _measure(_BoundedLocker(), "optimized_bounded")
    _, peak_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for result in (baseline, optimized):
        print(result)

    size_delta = (baseline["size"] - optimized["size"]) / baseline["size"] * 100
    mean_delta = (baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"] * 100
    throughput_delta = (
        (optimized["ops_per_s"] - baseline["ops_per_s"]) / baseline["ops_per_s"] * 100
    )

    print()
    print("## SUMMARY")
    print(f"lock_map_size: {baseline['size']} -> {optimized['size']} ({size_delta:.1f}% reduction)")
    print(
        f"mean: {baseline['mean_ms']:.3f}ms -> {optimized['mean_ms']:.3f}ms "
        f"({mean_delta:.1f}% faster)"
    )
    print(
        f"throughput: {baseline['ops_per_s']:.1f} -> {optimized['ops_per_s']:.1f} "
        f"req/s ({throughput_delta:.1f}%)"
    )
    print(f"peak_mem: {peak_base / 1024 / 1024:.3f}MB -> {peak_opt / 1024 / 1024:.3f}MB")


if __name__ == "__main__":
    asyncio.run(main())
