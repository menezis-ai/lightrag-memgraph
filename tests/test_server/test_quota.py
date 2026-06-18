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
        assert quota._pick(idx, quota._RAM_LIMIT_KEYS) == quota._parse_size("7.82GiB")
        assert quota._pick(idx, quota._GRAPH_KEYS) == quota._parse_size("252.97MiB")
        assert quota._pick(idx, quota._VECTOR_KEYS) == 0  # "0B"

    def test_memgraph_3_10_1(self):
        idx = quota._index_rows(REAL_3_10_1)
        assert quota._pick(idx, quota._USED_KEYS) == quota._parse_size("409.72MiB")
        assert quota._pick(idx, quota._RAM_LIMIT_KEYS) == quota._parse_size("2.00GiB")
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


def _patch_storage(monkeypatch, indexed):
    async def _fake():
        return indexed
    monkeypatch.setattr(quota, "_read_storage_info", _fake)


class TestSnapshotRamWall:
    """Community (OVH) / no license → the headline binds on the RAM wall."""

    async def test_3_10_community(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        monkeypatch.delenv("MEMGRAPH_BUDGET_ENFORCE", raising=False)
        # REAL_3_10_1 has license "unlimited" → no license wall → RAM binds.
        _patch_storage(monkeypatch, quota._index_rows(REAL_3_10_1))
        snap = await quota.snapshot()
        assert snap["binding"] == "ram"
        assert snap["budget_enforce"] == "reject"
        assert snap["used_bytes"] == quota._parse_size("409.72MiB")
        assert snap["limit_bytes"] == 2 * GIB
        assert snap["ram_basis"] == "tracked"
        assert snap["license_limit_bytes"] is None  # "unlimited"
        assert snap["graph_bytes"] == quota._parse_size("56.64MiB")
        assert snap["vector_bytes"] == quota._parse_size("29.68MiB")
        assert 0.19 < snap["used_pct"] < 0.21  # ~20% tracked, NOT 45% RSS
        assert snap["status"] == "ok"

    async def test_3_9(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        _patch_storage(monkeypatch, quota._index_rows(REAL_3_9_0))
        snap = await quota.snapshot()
        assert snap["binding"] == "ram"
        assert snap["used_bytes"] == quota._parse_size("252.97MiB")
        assert snap["limit_bytes"] == quota._parse_size("7.82GiB")

    async def test_env_fallback_limit(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "reject")
        _patch_storage(monkeypatch, {"memory_tracked": int(0.9 * GIB)})
        snap = await quota.snapshot()
        assert snap["limit_bytes"] == GIB
        assert snap["budget_enforce"] == "reject"
        assert snap["status"] == "warning"

    async def test_budget_enforce_mode_is_normalized(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "  WARN  ")
        assert quota.budget_enforce_mode() == "warn"

    async def test_rss_basis_when_only_rss(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")
        _patch_storage(monkeypatch, {"memory_res": 100})
        snap = await quota.snapshot()
        assert snap["ram_basis"] == "rss"
        assert snap["used_bytes"] == 100

    async def test_no_limit_anywhere(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        _patch_storage(monkeypatch, {"memory_tracked": 1234})
        snap = await quota.snapshot()
        assert snap["configured"] is False
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"

    async def test_probe_failure_fail_open(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "2GiB")
        _patch_storage(monkeypatch, {})
        snap = await quota.snapshot()
        assert snap["used_pct"] is None and snap["status"] == "ok"


class TestSnapshotLicenseWall:
    """Enterprise → the billed footprint (graph [+vec]) vs the license cap."""

    # license binds: data ~98% of a 1 GiB license, RAM far below its 4 GiB.
    ENT = {
        "global_memory_tracked": int(1.1 * GIB),
        "global_runtime_allocation_limit": 4 * GIB,
        "global_license_allocation_limit": GIB,
        "db_storage_memory_tracked": 800 * MIB,   # graph
        "db_embedding_memory_tracked": 200 * MIB,  # vectors
    }

    async def test_graph_analytics_bills_graph_plus_vectors(self, monkeypatch):
        monkeypatch.delenv("TWIN_MEMGRAPH_LICENSE_PLAN", raising=False)  # default
        _patch_storage(monkeypatch, dict(self.ENT))
        snap = await quota.snapshot()
        assert snap["binding"] == "license"
        assert snap["vectors_billed"] is True
        assert snap["billed_bytes"] == 1000 * MIB  # graph + vectors
        assert snap["license_limit_bytes"] == GIB
        assert snap["used_bytes"] == 1000 * MIB  # headline = billed
        assert snap["limit_bytes"] == GIB
        assert snap["status"] == "warning"  # ~97.6%

    async def test_ai_platform_excludes_vectors(self, monkeypatch):
        monkeypatch.setenv("TWIN_MEMGRAPH_LICENSE_PLAN", "ai-platform")
        _patch_storage(monkeypatch, dict(self.ENT))
        snap = await quota.snapshot()
        assert snap["vectors_billed"] is False
        assert snap["billed_bytes"] == 800 * MIB  # graph only — vectors free
        assert snap["billed_pct"] == 800 * MIB / GIB
        # 800/1024 = 0.78 (ok) < ram 1.1/4 = 0.275 → license still binds but ok
        assert snap["status"] == "ok"

    async def test_ram_binds_when_closer_than_license(self, monkeypatch):
        monkeypatch.delenv("TWIN_MEMGRAPH_LICENSE_PLAN", raising=False)
        _patch_storage(monkeypatch, {
            "global_memory_tracked": int(3.5 * GIB),
            "global_runtime_allocation_limit": 4 * GIB,  # ram 87.5%
            "global_license_allocation_limit": 10 * GIB,
            "db_storage_memory_tracked": 800 * MIB,
            "db_embedding_memory_tracked": 200 * MIB,    # billed 1G/10G = 10%
        })
        snap = await quota.snapshot()
        assert snap["binding"] == "ram"
        assert snap["used_bytes"] == int(3.5 * GIB)
        assert snap["status"] == "warning"


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
                "budget_enforce": "reject",
            }
        monkeypatch.setattr(quota, "snapshot", _snap)
        with pytest.raises(HTTPException) as exc:
            await quota.enforce_instance_quota()
        assert exc.value.status_code == 507
        assert "quota reached" in exc.value.detail and "GiB" in exc.value.detail

    async def test_blocked_warn_mode_does_not_raise(self, monkeypatch):
        async def _snap():
            return {
                "status": "blocked",
                "used_bytes": 2 * GIB + 1,
                "limit_bytes": 2 * GIB,
                "budget_enforce": "warn",
            }
        monkeypatch.setattr(quota, "snapshot", _snap)
        await quota.enforce_instance_quota()
