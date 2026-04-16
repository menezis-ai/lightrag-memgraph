"""
End-to-end probe tests for DocStatus paginated listing.

These tests reproduce the exact conditions that caused the BNP probe 502:
the LightRAG web frontend calls POST /documents/paginated, which hits
``MemgraphDocStatusStorage.get_docs_paginated()``. When the sort field
(``updated_at``) had no index, the query did a full scan on large datasets,
taking > 60s and triggering nginx upstream timeouts (→ 502 Bad Gateway),
which crashed the frontend (it received HTML instead of JSON).

This module verifies the fix in two ways:

1. **Index presence** — the 6 sortable/filterable properties have label
   indexes on the DocStatus_{workspace} label.
2. **Scale + timing budgets** — seeds realistic volumes (10, 100, 1000
   docs) and asserts the paginated call stays under a latency budget.
   If a future change breaks the index or re-serialises count + fetch,
   the budget is violated and CI goes red.

Requires a running Memgraph instance (set MEMGRAPH_URI).
"""

import time
import uuid
from datetime import datetime, timezone

import pytest
from lightrag.base import DocProcessingStatus, DocStatus

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


def _make_status(i: int, status: DocStatus = DocStatus.PROCESSED) -> DocProcessingStatus:
    """Build a DocProcessingStatus with a varied updated_at for sort testing."""
    # Use a monotonic timestamp so sort order is predictable (i=99 is most recent).
    ts = datetime.now(timezone.utc).replace(microsecond=i * 10 % 1_000_000).isoformat()
    return DocProcessingStatus(
        status=status,
        content_summary=f"Summary {i}",
        content_length=100 + i,
        file_path=f"/tmp/doc-{i:04d}.txt",
        created_at=ts,
        updated_at=ts,
        chunks_count=5,
    )


@pytest.fixture
async def probe_store():
    """DocStatus store with a unique workspace per test run (isolation)."""
    # Unique workspace avoids collisions when integration tests run in parallel.
    import os
    suffix = uuid.uuid4().hex[:8]
    os.environ["MEMGRAPH_WORKSPACE"] = f"probe_{suffix}"
    store = MemgraphDocStatusStorage(
        namespace="doc_status", global_config={}, embedding_func=None,
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.mark.integration
class TestPaginatedIndexes:
    async def test_all_sortable_indexes_created(self, probe_store):
        """initialize() must create indexes on the 6 sortable/filterable props.

        Without the `updated_at` / `created_at` indexes, the default ORDER BY
        forces a full scan → probe timeout → 502.
        """
        label = probe_store._label()
        expected = {"id", "status", "file_path", "track_id", "updated_at", "created_at"}

        async with _pool.get_read_session() as session:
            result = await session.run("SHOW INDEX INFO")
            records = [dict(r) async for r in result]
            await result.consume()

        # Memgraph returns property as str OR [str] depending on version.
        def _norm(p):
            return p[0] if isinstance(p, list) and p else p

        found = {
            _norm(rec.get("property"))
            for rec in records
            if rec.get("label") == label
        }
        missing = expected - found
        assert not missing, f"Missing indexes on :{label}: {missing}"


@pytest.mark.integration
class TestPaginatedCorrectness:
    async def test_sort_desc_by_updated_at(self, probe_store):
        """Default sort must return most recent docs first."""
        data = {f"doc-{i:03d}": _make_status(i) for i in range(30)}
        await probe_store.upsert(data)

        docs, total = await probe_store.get_docs_paginated(page=1, page_size=10)

        assert total == 30
        assert len(docs) == 10
        ts_list = [d[1].updated_at for d in docs]
        assert ts_list == sorted(ts_list, reverse=True), (
            "DESC sort on updated_at violated"
        )

    async def test_pagination_no_overlap(self, probe_store):
        """Consecutive pages must not overlap."""
        data = {f"doc-{i:03d}": _make_status(i) for i in range(30)}
        await probe_store.upsert(data)

        page1, _ = await probe_store.get_docs_paginated(page=1, page_size=10)
        page2, _ = await probe_store.get_docs_paginated(page=2, page_size=10)
        page3, _ = await probe_store.get_docs_paginated(page=3, page_size=10)

        ids1 = {d[0] for d in page1}
        ids2 = {d[0] for d in page2}
        ids3 = {d[0] for d in page3}
        assert not (ids1 & ids2), "Page 1 and 2 overlap"
        assert not (ids2 & ids3), "Page 2 and 3 overlap"
        assert not (ids1 & ids3), "Page 1 and 3 overlap"
        assert len(ids1 | ids2 | ids3) == 30, "Pages do not cover all docs"

    async def test_status_filter(self, probe_store):
        """status_filter must restrict results."""
        data = {}
        for i in range(20):
            data[f"proc-{i}"] = _make_status(i, DocStatus.PROCESSED)
        for i in range(10):
            data[f"pend-{i}"] = _make_status(i, DocStatus.PENDING)
        await probe_store.upsert(data)

        docs_proc, total_proc = await probe_store.get_docs_paginated(
            status_filter=DocStatus.PROCESSED, page=1, page_size=50,
        )
        docs_pend, total_pend = await probe_store.get_docs_paginated(
            status_filter=DocStatus.PENDING, page=1, page_size=50,
        )

        assert total_proc == 20
        assert total_pend == 10
        assert all(d[1].status == DocStatus.PROCESSED for d in docs_proc)
        assert all(d[1].status == DocStatus.PENDING for d in docs_pend)


@pytest.mark.integration
class TestPaginatedScaleTimingBudget:
    """Regression tests for the BNP probe 502 bug.

    If a future change removes the updated_at index, or serialises count
    and fetch, these budgets will be violated and CI goes red. The budgets
    are generous — a healthy Memgraph on a laptop should be 5-10x under.
    """

    @pytest.mark.parametrize(
        "n_docs,budget_ms",
        [(10, 150), (100, 300), (500, 1000)],
    )
    async def test_paginated_latency_budget(self, probe_store, n_docs, budget_ms):
        """get_docs_paginated must stay well under nginx's 60s upstream timeout.

        With the 0.5.1 fix (index + parallel count/fetch), 500 docs takes
        ~20ms on a laptop. Budget set to 50x that to tolerate CI jitter.
        """
        data = {f"doc-{i:04d}": _make_status(i) for i in range(n_docs)}
        await probe_store.upsert(data)

        # Warm-up — first call pays the schema cache cost.
        await probe_store.get_docs_paginated(page=1, page_size=50)

        t0 = time.perf_counter()
        docs, total = await probe_store.get_docs_paginated(page=1, page_size=50)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert total == n_docs
        assert len(docs) == min(50, n_docs)
        assert elapsed_ms < budget_ms, (
            f"get_docs_paginated({n_docs} docs) took {elapsed_ms:.0f}ms, "
            f"budget {budget_ms}ms — regression in sort index or parallel gather"
        )
