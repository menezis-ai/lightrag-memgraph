"""Unit tests for the instance quota module.

These exercise the pure logic (parse / extract / status mapping) and a
mocked probe path — no Memgraph required.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server import quota


class TestParseMemgraphLimit:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2GiB", 2 * 1024 ** 3),
            ("2 GiB", 2 * 1024 ** 3),
            ("2gib", 2 * 1024 ** 3),
            ("2048MiB", 2048 * 1024 ** 2),
            ("1.5GiB", int(1.5 * 1024 ** 3)),
            ("2GB", 2 * 10 ** 9),
            ("2000000000", 2 * 10 ** 9),
            ("2147483648B", 2 * 1024 ** 3),
            ("2147483648", 2 * 1024 ** 3),
            ("512KiB", 512 * 1024),
            ("1TiB", 1024 ** 4),
        ],
    )
    def test_recognised_units(self, raw, expected):
        assert quota.parse_memgraph_limit(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "lol", "2 zigabytes", "GiB"])
    def test_unparseable_returns_none(self, raw):
        assert quota.parse_memgraph_limit(raw) is None


class TestExtractMemoryBytes:
    def test_prefers_memory_res_over_others(self):
        rows = [
            {"storage info": "memory_allocated", "value": 100},
            {"storage info": "memory_res", "value": 999},
            {"storage info": "memory_tracked", "value": 555},
        ]
        assert quota._extract_memory_bytes(rows) == 999

    def test_falls_back_to_memory_tracked(self):
        rows = [
            {"storage info": "memory_allocated", "value": 100},
            {"storage info": "memory_tracked", "value": 555},
        ]
        assert quota._extract_memory_bytes(rows) == 555

    def test_falls_back_to_memory_allocated(self):
        rows = [{"storage info": "memory_allocated", "value": 100}]
        assert quota._extract_memory_bytes(rows) == 100

    def test_returns_none_when_no_memory_row(self):
        rows = [{"storage info": "edges_count", "value": 42}]
        assert quota._extract_memory_bytes(rows) is None

    def test_returns_none_on_empty(self):
        assert quota._extract_memory_bytes([]) is None

    def test_accepts_alternate_key_columns(self):
        rows = [{"name": "memory_res", "value": 1234}]
        assert quota._extract_memory_bytes(rows) == 1234

    def test_ignores_non_numeric_values(self):
        rows = [{"storage info": "memory_res", "value": "not-a-number"}]
        assert quota._extract_memory_bytes(rows) is None


class TestStatusMapping:
    @pytest.mark.parametrize(
        "pct,expected",
        [
            (None, "ok"),
            (0.0, "ok"),
            (0.5, "ok"),
            (0.84, "ok"),
            (0.85, "warning"),
            (0.95, "warning"),
            (0.999, "warning"),
            (1.0, "blocked"),
            (1.5, "blocked"),
        ],
    )
    def test_thresholds(self, pct, expected):
        assert quota._status_from(pct) == expected


class TestSnapshot:
    async def test_no_limit_configured_yields_ok_and_unconfigured(
        self, monkeypatch
    ):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)

        async def _fake_probe():
            return 1234

        monkeypatch.setattr(quota, "get_used_bytes", _fake_probe)
        snap = await quota.snapshot()
        assert snap["configured"] is False
        assert snap["limit_bytes"] is None
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"
        assert snap["warn_threshold"] == quota.WARN_THRESHOLD

    async def test_probe_failure_keeps_status_ok(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "2GiB")

        async def _fake_probe():
            return None

        monkeypatch.setattr(quota, "get_used_bytes", _fake_probe)
        snap = await quota.snapshot()
        assert snap["configured"] is True
        assert snap["limit_bytes"] == 2 * 1024 ** 3
        assert snap["used_bytes"] is None
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"

    async def test_warning_state(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")

        async def _fake_probe():
            return int(0.9 * 1024 ** 3)

        monkeypatch.setattr(quota, "get_used_bytes", _fake_probe)
        snap = await quota.snapshot()
        assert snap["status"] == "warning"
        assert 0.85 <= snap["used_pct"] < 1.0

    async def test_blocked_state(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")

        async def _fake_probe():
            return 1024 ** 3 + 1

        monkeypatch.setattr(quota, "get_used_bytes", _fake_probe)
        snap = await quota.snapshot()
        assert snap["status"] == "blocked"
        assert snap["used_pct"] >= 1.0

    async def test_zero_limit_protects_against_div_zero(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "0B")

        async def _fake_probe():
            return 100

        monkeypatch.setattr(quota, "get_used_bytes", _fake_probe)
        snap = await quota.snapshot()
        assert snap["used_pct"] is None
        assert snap["status"] == "ok"


class TestEnforceDependency:
    async def test_ok_passes(self, monkeypatch):
        async def _fake_snapshot():
            return {"status": "ok", "used_bytes": 100, "limit_bytes": 1000}

        monkeypatch.setattr(quota, "snapshot", _fake_snapshot)
        await quota.enforce_instance_quota()  # no raise

    async def test_warning_passes(self, monkeypatch):
        async def _fake_snapshot():
            return {"status": "warning", "used_bytes": 850, "limit_bytes": 1000}

        monkeypatch.setattr(quota, "snapshot", _fake_snapshot)
        await quota.enforce_instance_quota()  # no raise

    async def test_blocked_raises_507_with_gib_in_detail(self, monkeypatch):
        from fastapi import HTTPException

        async def _fake_snapshot():
            return {
                "status": "blocked",
                "used_bytes": 2 * 1024 ** 3 + 1,
                "limit_bytes": 2 * 1024 ** 3,
            }

        monkeypatch.setattr(quota, "snapshot", _fake_snapshot)
        with pytest.raises(HTTPException) as exc_info:
            await quota.enforce_instance_quota()
        assert exc_info.value.status_code == 507
        assert "quota reached" in exc_info.value.detail
        assert "GiB" in exc_info.value.detail
