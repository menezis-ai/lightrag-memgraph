"""Unit tests for the instance quota module.

Pure logic (parse / index / status mapping) + a mocked snapshot path —
no Memgraph required. The version-compat tests feed the **real** rows
captured from ``SHOW STORAGE INFO`` on Memgraph 3.9.0 and 3.10.1, so a
field rename in either edition is caught here.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server import quota

GIB = 1024 ** 3
MIB = 1024 ** 2


def _rows(pairs: list[tuple[str, object]]) -> list[dict[str, object]]:
    """Build ``SHOW STORAGE INFO`` rows from (key, value) pairs."""
    return [{"storage info": k, "value": v} for k, v in pairs]


# Verbatim from `SHOW STORAGE INFO` on the two versions in play.
REAL_3_9_0 = _rows(
    [
        ("memory_res", "741.18MiB"),
        ("memory_tracked", "252.97MiB"),
        ("allocation_limit", "7.82GiB"),
        ("graph_memory_tracked", "252.97MiB"),
        ("vector_index_memory_tracked", "0B"),
        ("storage_mode", "IN_MEMORY_TRANSACTIONAL"),
    ]
)
REAL_3_10_1 = _rows(
    [
        ("memory_res", "925.48MiB"),
        ("global_memory_tracked", "409.72MiB"),
        ("global_runtime_allocation_limit", "2.00GiB"),
        ("global_license_allocation_limit", "unlimited"),
        ("db_memory_tracked", "86.32MiB"),
        ("db_storage_memory_tracked", "56.64MiB"),
        ("db_embedding_memory_tracked", "29.68MiB"),
        ("storage_mode", "IN_MEMORY_TRANSACTIONAL"),
    ]
)


class TestParseSize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("409.72MiB", int(409.72 * MIB)),
            ("2.00GiB", 2 * GIB),
            ("0B", 0),
            ("5.44KiB", int(5.44 * 1024)),
            ("2048", 2048),
            (2048, 2048),
            (1024.0, 1024),
            ("unlimited", None),
            ("UNLIMITED", None),
            ("", None),
            (None, None),
            ("not-a-size", None),
            (True, None),  # bool is not a size
        ],
    )
    def test_parse(self, raw, expected):
        assert quota._parse_size(raw) == expected


class TestParseMemgraphLimit:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2GiB", 2 * GIB),
            ("2 GiB", 2 * GIB),
            ("2048MiB", 2048 * MIB),
            ("1.5GiB", int(1.5 * GIB)),
            ("2GB", 2 * 10 ** 9),
            ("2000000000", 2 * 10 ** 9),
            ("512KiB", 512 * 1024),
        ],
    )
    def test_recognised(self, raw, expected):
        assert quota.parse_memgraph_limit(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "lol", "GiB"])
    def test_unparseable_returns_none(self, raw):
        assert quota.parse_memgraph_limit(raw) is None


class TestVersionFieldCompat:
    """The whole point: the same picks work on 3.9 and 3.10 field sets."""

    def test_memgraph_3_9_0(self):
        idx = quota._index_rows(REAL_3_9_0)
        assert quota._pick(idx, quota._USED_KEYS) == quota._parse_size("252.97MiB")
        assert quota._pick(idx, quota._LIMIT_KEYS) == quota._parse_size("7.82GiB")
        assert quota._pick(idx, quota._GRAPH_KEYS) == quota._parse_size("252.97MiB")
        assert quota._pick(idx, quota._VECTOR_KEYS) == 0  # "0B"

    def test_memgraph_3_10_1(self):
        idx = quota._index_rows(REAL_3_10_1)
        assert quota._pick(idx, quota._USED_KEYS) == quota._parse_size("409.72MiB")
        assert quota._pick(idx, quota._LIMIT_KEYS) == quota._parse_size("2.00GiB")
        assert quota._pick(idx, quota._GRAPH_KEYS) == quota._parse_size("56.64MiB")
        assert quota._pick(idx, quota._VECTOR_KEYS) == quota._parse_size("29.68MiB")

    def test_used_is_tracked_not_rss(self):
        # The fix: track the allocation Memgraph enforces, NOT process RSS.
        idx = quota._index_rows(REAL_3_10_1)
        assert quota._pick(idx, quota._USED_KEYS) == quota._parse_size("409.72MiB")
        assert quota._pick(idx, quota._USED_RSS_KEYS) == quota._parse_size("925.48MiB")

    def test_license_unlimited_is_not_a_limit(self):
        idx = quota._index_rows(REAL_3_10_1)
        assert "global_license_allocation_limit" not in idx  # "unlimited" → dropped

    def test_rss_only_build_falls_back(self):
        idx = quota._index_rows(_rows([("memory_res", "500MiB")]))
        assert quota._pick(idx, quota._USED_KEYS) is None
        assert quota._pick(idx, quota._USED_RSS_KEYS) == 500 * MIB

    def test_empty_and_non_size_rows(self):
        assert quota._index_rows([]) == {}
        idx = quota._index_rows(_rows([("storage_mode", "IN_MEMORY_TRANSACTIONAL")]))
        assert idx == {}

    def test_alternate_key_column(self):
        idx = quota._index_rows([{"name": "memory_tracked", "value": 1234}])
        assert quota._pick(idx, quota._USED_KEYS) == 1234


class TestStatusMapping:
    @pytest.mark.parametrize(
        "pct,expected",
        [
            (None, "ok"), (0.0, "ok"), (0.84, "ok"),
            (0.85, "warning"), (0.999, "warning"),
            (1.0, "blocked"), (1.5, "blocked"),
        ],
    )
    def test_thresholds(self, pct, expected):
        assert quota._status_from(pct) == expected


class TestSnapshot:
    def _patch_storage(self, monkeypatch, indexed):
        async def _fake():
            return indexed
        monkeypatch.setattr(quota, "_read_storage_info", _fake)

    async def test_3_10_real_numbers(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        self._patch_storage(monkeypatch, quota._index_rows(REAL_3_10_1))
        snap = await quota.snapshot()
        assert snap["configured"] is True
        assert snap["used_bytes"] == quota._parse_size("409.72MiB")
        assert snap["limit_bytes"] == 2 * GIB
        assert snap["used_basis"] == "tracked"
        assert snap["limit_source"] == "memgraph"
        assert snap["graph_bytes"] == quota._parse_size("56.64MiB")
        assert snap["vector_bytes"] == quota._parse_size("29.68MiB")
        assert 0.19 < snap["used_pct"] < 0.21  # ~20%, NOT the 45% RSS would give
        assert snap["status"] == "ok"

    async def test_3_9_real_numbers(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        self._patch_storage(monkeypatch, quota._index_rows(REAL_3_9_0))
        snap = await quota.snapshot()
        assert snap["used_bytes"] == quota._parse_size("252.97MiB")
        assert snap["limit_bytes"] == quota._parse_size("7.82GiB")
        assert snap["limit_source"] == "memgraph"
        assert snap["status"] == "ok"

    async def test_env_fallback_when_memgraph_reports_no_limit(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")
        # tracked present, but no allocation_limit field
        self._patch_storage(monkeypatch, {"memory_tracked": int(0.9 * GIB)})
        snap = await quota.snapshot()
        assert snap["limit_bytes"] == GIB
        assert snap["limit_source"] == "env"
        assert snap["status"] == "warning"

    async def test_rss_basis_when_only_rss(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")
        self._patch_storage(monkeypatch, {"memory_res": 100})
        snap = await quota.snapshot()
        assert snap["used_basis"] == "rss"
        assert snap["used_bytes"] == 100

    async def test_no_limit_anywhere_is_unconfigured_ok(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        self._patch_storage(monkeypatch, {"memory_tracked": 1234})
        snap = await quota.snapshot()
        assert snap["configured"] is False
        assert snap["limit_bytes"] is None
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"

    async def test_probe_failure_keeps_ok(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "2GiB")
        self._patch_storage(monkeypatch, {})  # fail-open → empty
        snap = await quota.snapshot()
        assert snap["used_bytes"] is None
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"

    async def test_blocked(self, monkeypatch):
        self._patch_storage(
            monkeypatch,
            {"memory_tracked": GIB + 1, "allocation_limit": GIB},
        )
        snap = await quota.snapshot()
        assert snap["status"] == "blocked"
        assert snap["used_pct"] >= 1.0

    async def test_zero_limit_no_div_zero(self, monkeypatch):
        self._patch_storage(
            monkeypatch, {"memory_tracked": 100, "allocation_limit": 0}
        )
        # 0 from Memgraph isn't a real cap → falls through to env (none here)
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        snap = await quota.snapshot()
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"


class TestEnforceDependency:
    async def test_ok_passes(self, monkeypatch):
        async def _snap():
            return {"status": "ok", "used_bytes": 100, "limit_bytes": 1000}
        monkeypatch.setattr(quota, "snapshot", _snap)
        await quota.enforce_instance_quota()

    async def test_blocked_raises_507(self, monkeypatch):
        from fastapi import HTTPException

        async def _snap():
            return {
                "status": "blocked",
                "used_bytes": 2 * GIB + 1,
                "limit_bytes": 2 * GIB,
            }
        monkeypatch.setattr(quota, "snapshot", _snap)
        with pytest.raises(HTTPException) as exc:
            await quota.enforce_instance_quota()
        assert exc.value.status_code == 507
        assert "quota reached" in exc.value.detail and "GiB" in exc.value.detail
