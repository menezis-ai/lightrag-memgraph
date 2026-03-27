"""
Tests for the _memory module: size parsing, cost estimation, and live introspection.

Unit tests (no Memgraph needed) for:
- _parse_size() -- human-readable size string parsing
- estimate_insert_cost() -- pre-insert memory cost estimation
- estimate_batch_insert_cost() -- batch estimation
- check_memory_budget() -- combined budget check (mocked Memgraph)
- get_memory_usage() -- live introspection (mocked Memgraph)
- get_storage_info() -- raw SHOW STORAGE INFO (mocked Memgraph)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph._memory import (
    BudgetCheck,
    InsertCostEstimate,
    MemoryUsage,
    _parse_size,
    check_memory_budget,
    estimate_batch_insert_cost,
    estimate_database_usage,
    estimate_insert_cost,
    get_memory_usage,
    get_storage_info,
    get_workspace_node_counts,
)

# ---------------------------------------------------------------------------
# _parse_size tests
# ---------------------------------------------------------------------------


class TestParseSize:
    def test_bytes(self):
        assert _parse_size("0B") == 0
        assert _parse_size("100B") == 100

    def test_bytes_word(self):
        assert _parse_size("42 bytes") == 42
        assert _parse_size("1 byte") == 1

    def test_kib(self):
        assert _parse_size("1KiB") == 1024
        assert _parse_size("2.5KiB") == 2560

    def test_mib(self):
        assert _parse_size("1MiB") == 1024 * 1024
        assert _parse_size("42MiB") == 42 * 1024 * 1024

    def test_gib(self):
        assert _parse_size("1GiB") == 1024**3
        assert _parse_size("1.5GiB") == int(1.5 * 1024**3)

    def test_tib(self):
        assert _parse_size("1TiB") == 1024**4

    def test_decimal_units(self):
        assert _parse_size("1KB") == 1000
        assert _parse_size("1MB") == 1_000_000
        assert _parse_size("1GB") == 1_000_000_000
        assert _parse_size("1TB") == 1_000_000_000_000

    def test_whitespace(self):
        assert _parse_size("  42 MiB  ") == 42 * 1024**2
        assert _parse_size("100 B") == 100

    def test_plain_integer(self):
        """Memgraph might return raw byte counts as plain integers."""
        assert _parse_size("1048576") == 1048576
        assert _parse_size("0") == 0

    def test_unparseable_returns_none(self):
        assert _parse_size("unknown") is None
        assert _parse_size("") is None
        assert _parse_size("abc MiB") is None

    def test_case_insensitive(self):
        assert _parse_size("42mib") == 42 * 1024**2
        assert _parse_size("42MIB") == 42 * 1024**2
        assert _parse_size("1gib") == 1024**3


# ---------------------------------------------------------------------------
# estimate_insert_cost tests
# ---------------------------------------------------------------------------


class TestEstimateInsertCost:
    def test_small_document(self):
        text = "Hello world" * 10  # 110 chars
        est = estimate_insert_cost(text, embedding_dim=1536)
        assert isinstance(est, InsertCostEstimate)
        assert est.total_bytes > 0
        assert est.num_chunks_estimate >= 1
        assert est.full_doc_bytes > len(text.encode("utf-8"))
        assert est.embedding_bytes > 0

    def test_large_document(self):
        text = "A" * 120_000  # ~100 chunks at default settings
        est = estimate_insert_cost(text, embedding_dim=1536)
        # ~100 chunks * 1536 * 2 bytes (f16 default) = ~300 KiB for embeddings
        assert est.embedding_bytes > 250_000
        assert est.num_chunks_estimate > 50

    def test_embedding_dim_affects_cost(self):
        text = "A" * 12_000
        est_small = estimate_insert_cost(text, embedding_dim=384)
        est_large = estimate_insert_cost(text, embedding_dim=3072)
        assert est_large.embedding_bytes > est_small.embedding_bytes
        # Embedding is proportional to dim
        # (excluding node overhead which is constant per chunk)
        assert est_large.total_bytes > est_small.total_bytes

    def test_custom_chunk_size(self):
        text = "A" * 12_000
        est_default = estimate_insert_cost(text)
        est_small_chunks = estimate_insert_cost(text, chunk_size=500, chunk_overlap=50)
        # Smaller chunks = more chunks = more embeddings
        assert est_small_chunks.num_chunks_estimate > est_default.num_chunks_estimate
        assert est_small_chunks.embedding_bytes > est_default.embedding_bytes

    def test_utf8_multibyte(self):
        """Non-ASCII text should estimate based on UTF-8 byte length."""
        text = "caf\u00e9" * 1000  # 4 bytes per "cafe" + 2 for e-acute
        est = estimate_insert_cost(text)
        assert est.full_doc_bytes > len(text)  # UTF-8 is longer than len()

    def test_empty_text(self):
        est = estimate_insert_cost("")
        assert est.num_chunks_estimate == 1  # minimum 1 chunk
        assert est.total_bytes > 0  # overhead alone

    def test_human_readable(self):
        est = estimate_insert_cost("A" * 12_000, embedding_dim=1536)
        readable = est.human_readable()
        assert "TOTAL:" in readable
        assert "chunks" in readable.lower()

    def test_total_bytes_is_sum(self):
        est = estimate_insert_cost("A" * 5000, embedding_dim=768)
        expected = (
            est.full_doc_bytes
            + est.text_chunks_bytes
            + est.embedding_bytes
            + est.graph_overhead_bytes
        )
        assert est.total_bytes == expected


# ---------------------------------------------------------------------------
# estimate_batch_insert_cost tests
# ---------------------------------------------------------------------------


class TestEstimateBatchInsertCost:
    def test_batch_sums_correctly(self):
        texts = ["A" * 5000, "B" * 10000, "C" * 3000]
        batch = estimate_batch_insert_cost(texts, embedding_dim=1536)
        individual_total = sum(
            estimate_insert_cost(t, embedding_dim=1536).total_bytes for t in texts
        )
        assert batch.total_bytes == individual_total

    def test_batch_chunks_sum(self):
        texts = ["A" * 5000, "B" * 10000]
        batch = estimate_batch_insert_cost(texts)
        individual_chunks = sum(
            estimate_insert_cost(t).num_chunks_estimate for t in texts
        )
        assert batch.num_chunks_estimate == individual_chunks

    def test_empty_batch(self):
        batch = estimate_batch_insert_cost([])
        assert batch.total_bytes == 0
        assert batch.num_chunks_estimate == 0


# ---------------------------------------------------------------------------
# get_storage_info / get_memory_usage tests (mocked Memgraph)
# ---------------------------------------------------------------------------


def _make_mock_session(rows: list[tuple[str, str]]):
    """Create a mock session that returns rows from SHOW STORAGE INFO."""
    mock_result = AsyncMock()
    mock_records = []
    for k, v in rows:
        record = MagicMock()
        record.keys.return_value = ["storage info", "value"]
        record.__getitem__ = lambda self, idx, _k=k, _v=v: (
            _k if idx == "storage info" or idx == 0 else _v
        )
        mock_records.append(record)

    mock_result.__aiter__ = lambda self: _async_iter(mock_records)
    mock_result.consume = AsyncMock()

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    return mock_session


async def _async_iter(items):
    for item in items:
        yield item


class TestGetStorageInfo:
    async def test_returns_dict(self):
        rows = [
            ("vertex_count", "1234"),
            ("edge_count", "5678"),
            ("memory_usage", "42MiB"),
        ]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            info = await get_storage_info()

        assert info["vertex_count"] == "1234"
        assert info["edge_count"] == "5678"
        assert info["memory_usage"] == "42MiB"

    async def test_lowercases_keys(self):
        rows = [("Memory_Usage", "100MiB")]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            info = await get_storage_info()

        assert "memory_usage" in info


class TestGetMemoryUsage:
    async def test_legacy_memory_usage_fallback(self):
        """Older Memgraph (< v3.9): only memory_usage available."""
        rows = [
            ("vertex_count", "100"),
            ("edge_count", "200"),
            ("memory_usage", "42MiB"),
        ]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await get_memory_usage()

        assert isinstance(usage, MemoryUsage)
        assert usage.used_bytes == 42 * 1024**2
        assert usage.vertex_count == 100
        assert usage.edge_count == 200
        assert usage.graph_memory_bytes is None
        assert usage.vector_index_memory_bytes is None

    async def test_memory_tracked_preferred_over_memory_usage(self):
        """memory_tracked is the Enterprise billing metric — takes priority."""
        rows = [
            ("memory_usage", "100MiB"),
            ("memory_tracked", "80MiB"),
        ]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await get_memory_usage()

        assert usage.used_bytes == 80 * 1024**2  # memory_tracked, not memory_usage

    async def test_v39_granular_fields(self):
        """v3.9+: graph_memory_tracked + vector_index_memory_tracked."""
        rows = [
            ("memory_tracked", "10GiB"),
            ("graph_memory_tracked", "7GiB"),
            ("vector_index_memory_tracked", "3GiB"),
            ("vertex_count", "500000"),
            ("edge_count", "1200000"),
        ]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await get_memory_usage()

        assert usage.used_bytes == 10 * 1024**3
        assert usage.graph_memory_bytes == 7 * 1024**3
        assert usage.vector_index_memory_bytes == 3 * 1024**3
        assert usage.vertex_count == 500000
        assert usage.edge_count == 1200000

    async def test_granular_sum_when_no_memory_tracked(self):
        """If memory_tracked absent but graph+vector present, sum them."""
        rows = [
            ("graph_memory_tracked", "6GiB"),
            ("vector_index_memory_tracked", "2GiB"),
        ]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await get_memory_usage()

        assert usage.used_bytes == 8 * 1024**3
        assert usage.graph_memory_bytes == 6 * 1024**3
        assert usage.vector_index_memory_bytes == 2 * 1024**3

    async def test_handles_missing_fields(self):
        """Gracefully handle a minimal SHOW STORAGE INFO response."""
        rows = [("memory_usage", "1GiB")]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await get_memory_usage()

        assert usage.used_bytes == 1024**3
        assert usage.vertex_count == 0
        assert usage.edge_count == 0
        assert usage.memory_res_bytes is None
        assert usage.graph_memory_bytes is None
        assert usage.vector_index_memory_bytes is None

    async def test_preserves_raw_dict(self):
        rows = [
            ("memory_usage", "10MiB"),
            ("custom_key", "custom_value"),
        ]
        mock_session = _make_mock_session(rows)

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await get_memory_usage()

        assert "custom_key" in usage.raw
        assert usage.raw["custom_key"] == "custom_value"


# ---------------------------------------------------------------------------
# check_memory_budget tests (mocked get_memory_usage)
# ---------------------------------------------------------------------------


class TestCheckMemoryBudget:
    """check_memory_budget uses per-database estimate_database_usage
    (not instance-wide SHOW STORAGE INFO) so the limit is per-database."""

    async def test_fits_within_budget(self):
        with patch(
            "twindb_lightrag_memgraph._memory.estimate_database_usage",
            new_callable=AsyncMock,
            return_value=1 * 1024**3,  # 1 GiB used in this database
        ):
            result = await check_memory_budget(
                text="A" * 10_000,
                embedding_dim=1536,
                memory_limit_bytes=2 * 1024**3,  # 2 GiB per-database limit
            )

        assert isinstance(result, BudgetCheck)
        assert result.fits is True
        assert result.used_bytes == 1 * 1024**3

    async def test_exceeds_budget(self):
        with patch(
            "twindb_lightrag_memgraph._memory.estimate_database_usage",
            new_callable=AsyncMock,
            return_value=int(1.9 * 1024**3),  # 1.9 GiB used
        ):
            result = await check_memory_budget(
                text="A" * 50_000_000,  # ~50 MB document
                embedding_dim=3072,
                memory_limit_bytes=2 * 1024**3,  # 2 GiB limit
                safety_margin=0.05,
            )

        assert result.fits is False

    async def test_empty_database_allows_insert(self):
        """Regression: an empty database (0 bytes used) must allow inserts
        even if the Memgraph instance is 4+ GiB total."""
        with patch(
            "twindb_lightrag_memgraph._memory.estimate_database_usage",
            new_callable=AsyncMock,
            return_value=0,  # database is empty
        ):
            result = await check_memory_budget(
                text="A" * 10_000,
                embedding_dim=1536,
                memory_limit_bytes=2 * 1024**3,
            )

        assert result.fits is True
        assert result.used_bytes == 0

    async def test_env_var_limit(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "10GiB")

        with patch(
            "twindb_lightrag_memgraph._memory.estimate_database_usage",
            new_callable=AsyncMock,
            return_value=5 * 1024**3,
        ):
            result = await check_memory_budget(text="A" * 1000)

        assert result.limit_bytes == 10 * 1024**3

    async def test_batch_budget_check(self):
        with patch(
            "twindb_lightrag_memgraph._memory.estimate_database_usage",
            new_callable=AsyncMock,
            return_value=500 * 1024**2,  # 500 MiB
        ):
            result = await check_memory_budget(
                texts=["A" * 5000, "B" * 10000],
                memory_limit_bytes=2 * 1024**3,
            )

        assert result.fits is True
        assert result.estimated_cost_bytes > 0

    async def test_raises_without_text(self):
        with pytest.raises(ValueError, match="Provide either"):
            await check_memory_budget(memory_limit_bytes=1024**3)

    async def test_human_readable(self):
        with patch(
            "twindb_lightrag_memgraph._memory.estimate_database_usage",
            new_callable=AsyncMock,
            return_value=100 * 1024**2,  # 100 MiB
        ):
            result = await check_memory_budget(
                text="A" * 10_000,
                memory_limit_bytes=2 * 1024**3,
            )

        readable = result.human_readable()
        assert "OK" in readable
        assert "Current usage:" in readable
        assert "Headroom:" in readable


# ---------------------------------------------------------------------------
# estimate_database_usage tests (per-database memory estimation)
# ---------------------------------------------------------------------------


def _make_query_mock_session(query_results: dict[str, dict]):
    """Create a mock session where run() returns different results per query pattern.

    query_results maps a substring of the Cypher query to a dict of column->value.
    """
    mock_session = AsyncMock()

    async def _run(query, **kwargs):
        result = AsyncMock()
        for pattern, row_data in query_results.items():
            if pattern in query:
                record = MagicMock()
                record.__getitem__ = lambda self, key, _d=row_data: _d[key]
                result.single = AsyncMock(return_value=record)
                result.consume = AsyncMock()
                return result
        # No match — empty result
        record = MagicMock()
        record.__getitem__ = lambda self, key: 0
        result.single = AsyncMock(return_value=record)
        result.consume = AsyncMock()
        return result

    mock_session.run = _run
    return mock_session


class TestEstimateDatabaseUsage:
    async def test_empty_database_returns_zero(self):
        mock_session = _make_query_mock_session({
            "n.data IS NOT NULL": {"cnt": 0, "data_bytes": 0},
            "n.embedding IS NOT NULL": {"cnt": 0},
            "n.data IS NULL AND n.embedding IS NULL": {"cnt": 0},
            "()-[r]->()": {"cnt": 0},
        })

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await estimate_database_usage(embedding_dim=1536)

        assert usage == 0

    async def test_kv_nodes_counted(self):
        mock_session = _make_query_mock_session({
            "n.data IS NOT NULL": {"cnt": 100, "data_bytes": 500_000},
            "n.embedding IS NOT NULL": {"cnt": 0},
            "n.data IS NULL AND n.embedding IS NULL": {"cnt": 0},
            "()-[r]->()": {"cnt": 0},
        })

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await estimate_database_usage(embedding_dim=1536)

        # 500_000 data bytes + 100 * 128 node overhead
        assert usage == 500_000 + 100 * 128

    async def test_vec_nodes_counted(self):
        mock_session = _make_query_mock_session({
            "n.data IS NOT NULL": {"cnt": 0, "data_bytes": 0},
            "n.embedding IS NOT NULL": {"cnt": 50},
            "n.data IS NULL AND n.embedding IS NULL": {"cnt": 0},
            "()-[r]->()": {"cnt": 0},
        })

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await estimate_database_usage(embedding_dim=1536)

        # 50 * (1536 * 2 bytes f16 + 128 overhead + 256 content snippet)
        expected_vec = 50 * (1536 * 2 + 128 + 256)
        assert usage == expected_vec

    async def test_fallback_to_zero_on_error(self):
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=RuntimeError("Memgraph down"))

        with patch(
            "twindb_lightrag_memgraph._memory._pool.get_read_session"
        ) as mock_ctx:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            usage = await estimate_database_usage()

        assert usage == 0  # fallback, don't block inserts


# ---------------------------------------------------------------------------
# MemoryUsage / InsertCostEstimate dataclass sanity
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_memory_usage_defaults(self):
        usage = MemoryUsage(used_bytes=42)
        assert usage.vertex_count == 0
        assert usage.edge_count == 0
        assert usage.memory_res_bytes is None
        assert usage.graph_memory_bytes is None
        assert usage.vector_index_memory_bytes is None
        assert usage.raw == {}

    def test_insert_cost_total(self):
        est = InsertCostEstimate(
            full_doc_bytes=100,
            text_chunks_bytes=200,
            embedding_bytes=300,
            graph_overhead_bytes=400,
            num_chunks_estimate=5,
        )
        assert est.total_bytes == 1000

    def test_budget_check_human_readable_exceeds(self):
        check = BudgetCheck(
            fits=False,
            used_bytes=70 * 1024**3,
            limit_bytes=75 * 1024**3,
            estimated_cost_bytes=10 * 1024**3,
            remaining_bytes=0,
            headroom_ratio=0.0,
        )
        readable = check.human_readable()
        assert "WOULD EXCEED BUDGET" in readable
