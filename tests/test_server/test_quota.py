"""Unit tests for the Memgraph 3.12 instance quota module.

The contract tests feed the real rows captured from both Memgraph 3.12.0
storage-info queries. This catches regressions in the instance/database field
split without keeping compatibility logic for superseded Memgraph releases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pytest

from twindb_lightrag_memgraph.server import quota

GIB = 1024**3
MIB = 1024**2


def _rows(pairs: list[tuple[str, object]]) -> list[dict[str, object]]:
    """Build ``SHOW STORAGE INFO`` rows from (key, value) pairs."""
    return [{"storage info": k, "value": v} for k, v in pairs]


# Verbatim from Memgraph 3.12.0 Community Edition.
REAL_3_12_INSTANCE = _rows(
    [
        ("vm_max_map_count", 262144),
        ("memory_res", "94.26MiB"),
        ("peak_memory_res", "94.26MiB"),
        ("disk_usage", "227.43KiB"),
        ("memory_tracked", "27.61MiB"),
        ("memory_limit", "7.82GiB"),
        ("license_memory_limit", "unlimited"),
        ("query+graph_memory_tracked", "27.61MiB"),
        ("vector_index_memory_tracked", "0B"),
        ("global_storage_mode", "IN_MEMORY_TRANSACTIONAL"),
    ]
)
REAL_3_12_DATABASE = _rows(
    [
        ("name", "memgraph"),
        ("vertex_count", 0),
        ("edge_count", 0),
        ("disk_usage", "82.23KiB"),
        ("graph_memory_tracked", "112.00KiB"),
        ("query_memory_tracked", "5.44KiB"),
        ("vector_index_memory_tracked", "0B"),
        ("tenant_memory_tracked", "117.44KiB"),
        ("tenant_memory_limit", "7.82GiB"),
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
            (True, None),
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
            ("2GB", 2 * 10**9),
            ("2000000000", 2 * 10**9),
            ("512KiB", 512 * 1024),
        ],
    )
    def test_recognised(self, raw, expected):
        assert quota.parse_memgraph_limit(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "lol", "GiB"])
    def test_unparseable_returns_none(self, raw):
        assert quota.parse_memgraph_limit(raw) is None


class TestMemgraph312FieldContract:
    """Memgraph 3.12 splits instance limits from database footprint."""

    def test_instance_metrics(self):
        idx = quota._index_rows(REAL_3_12_INSTANCE)
        assert quota._pick(idx, quota._USED_KEYS) == quota._parse_size("27.61MiB")
        assert quota._pick(idx, quota._RAM_LIMIT_KEYS) == quota._parse_size("7.82GiB")
        assert quota._pick(idx, quota._GRAPH_KEYS) is None

    def test_current_database_metrics(self):
        idx = quota._index_rows(REAL_3_12_DATABASE)
        assert quota._pick(idx, quota._GRAPH_KEYS) == 112 * 1024
        assert quota._pick(idx, quota._VECTOR_KEYS) == 0
        assert quota._pick(idx, quota._RAM_LIMIT_KEYS) is None

    def test_used_is_reclamation_aware_tracker_not_rss(self):
        idx = quota._index_rows(REAL_3_12_INSTANCE)
        assert quota._pick(idx, quota._USED_KEYS) == quota._parse_size("27.61MiB")
        assert idx["memory_res"] == quota._parse_size("94.26MiB")

    def test_license_unlimited_is_not_a_limit(self):
        idx = quota._index_rows(REAL_3_12_INSTANCE)
        assert "license_memory_limit" not in idx

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
            (None, "ok"),
            (0.0, "ok"),
            (0.84, "ok"),
            (0.85, "warning"),
            (0.999, "warning"),
            (1.0, "blocked"),
            (1.5, "blocked"),
        ],
    )
    def test_thresholds(self, pct, expected):
        assert quota._status_from(pct) == expected


def _patch_storage(monkeypatch, indexed, database_indexed=None):
    async def _fake_instance():
        return indexed

    async def _fake_database():
        return database_indexed or {}

    monkeypatch.setattr(quota, "_read_storage_info", _fake_instance)
    monkeypatch.setattr(quota, "_read_database_storage_info", _fake_database)


class TestSnapshotRamWall:
    """Community / no license → the headline binds on the RAM wall."""

    async def test_3_12_community(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_MEMORY_LIMIT", raising=False)
        monkeypatch.delenv("MEMGRAPH_BUDGET_ENFORCE", raising=False)
        _patch_storage(
            monkeypatch,
            quota._index_rows(REAL_3_12_INSTANCE),
            quota._index_rows(REAL_3_12_DATABASE),
        )
        snap = await quota.snapshot()
        assert snap["binding"] == "ram"
        assert snap["budget_enforce"] == "reject"
        assert snap["used_bytes"] == quota._parse_size("27.61MiB")
        assert snap["limit_bytes"] == quota._parse_size("7.82GiB")
        assert snap["ram_basis"] == "tracked"
        assert snap["license_limit_bytes"] is None
        assert snap["graph_bytes"] == 112 * 1024
        assert snap["vector_bytes"] == 0
        assert 0.003 < snap["used_pct"] < 0.004
        assert snap["status"] == "ok"

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

    async def test_missing_tracker_does_not_fall_back_to_rss(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MEMORY_LIMIT", "1GiB")
        _patch_storage(monkeypatch, {"memory_res": 100})
        snap = await quota.snapshot()
        assert snap["ram_basis"] is None
        assert snap["used_bytes"] is None
        assert snap["used_pct"] is None

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

    ENT = {
        "memory_tracked": int(1.1 * GIB),
        "memory_limit": 4 * GIB,
        "license_memory_limit": GIB,
    }
    DB = {
        "graph_memory_tracked": 800 * MIB,
        "vector_index_memory_tracked": 200 * MIB,
    }

    async def test_graph_analytics_bills_graph_plus_vectors(self, monkeypatch):
        monkeypatch.delenv("TWIN_MEMGRAPH_LICENSE_PLAN", raising=False)
        _patch_storage(monkeypatch, dict(self.ENT), dict(self.DB))
        snap = await quota.snapshot()
        assert snap["binding"] == "license"
        assert snap["vectors_billed"] is True
        assert snap["billed_bytes"] == 1000 * MIB
        assert snap["license_limit_bytes"] == GIB
        assert snap["used_bytes"] == 1000 * MIB
        assert snap["limit_bytes"] == GIB
        assert snap["status"] == "warning"

    async def test_ai_platform_excludes_vectors(self, monkeypatch):
        monkeypatch.setenv("TWIN_MEMGRAPH_LICENSE_PLAN", "ai-platform")
        _patch_storage(monkeypatch, dict(self.ENT), dict(self.DB))
        snap = await quota.snapshot()
        assert snap["vectors_billed"] is False
        assert snap["billed_bytes"] == 800 * MIB
        assert snap["billed_pct"] == 800 * MIB / GIB
        assert snap["status"] == "ok"

    async def test_ram_binds_when_closer_than_license(self, monkeypatch):
        monkeypatch.delenv("TWIN_MEMGRAPH_LICENSE_PLAN", raising=False)
        _patch_storage(
            monkeypatch,
            {
                "memory_tracked": int(3.5 * GIB),
                "memory_limit": 4 * GIB,
                "license_memory_limit": 10 * GIB,
            },
            dict(self.DB),
        )
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


# ── Probe failure log-once (OVH maquette audit 2026-07-28) ─────────────────
#
# The WebUI polls /twin/api/quota continuously; a steadily failing probe
# (e.g. a pre-3.11 Memgraph that cannot parse the per-database query) was
# logging a full traceback per poll — 163/day on the maquette. Contract:
# full exception once per outage, DEBUG for repeats, re-armed by the next
# success. Fail-open behaviour ({} on failure) is unchanged.


class _FailingSession:
    async def run(self, *_a, **_k):
        raise RuntimeError("mismatched input 'ON' expecting")


class _OkResult:
    async def data(self):
        return []

    async def consume(self):
        return None


class _OkSession:
    async def run(self, *_a, **_k):
        return _OkResult()


class TestProbeFailureLogOnce:
    def _install_session(self, monkeypatch, holder):
        @asynccontextmanager
        async def _cm():
            yield holder["session"]

        monkeypatch.setattr(quota._pool, "get_read_session", _cm)
        monkeypatch.setattr(quota, "_PROBE_FAILURE_LOGGED", set())

    async def test_repeat_failures_log_full_exception_once(self, monkeypatch, caplog):
        holder = {"session": _FailingSession()}
        self._install_session(monkeypatch, holder)
        with caplog.at_level(logging.DEBUG, logger=quota.logger.name):
            assert await quota._read_database_storage_info() == {}
            assert await quota._read_database_storage_info() == {}
            assert await quota._read_database_storage_info() == {}
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(errors) == 1
        assert errors[0].exc_info is not None  # full traceback kept
        assert len(debugs) == 2

    async def test_success_rearms_the_loud_log(self, monkeypatch, caplog):
        holder = {"session": _FailingSession()}
        self._install_session(monkeypatch, holder)
        with caplog.at_level(logging.DEBUG, logger=quota.logger.name):
            await quota._read_database_storage_info()  # outage 1 → ERROR
            holder["session"] = _OkSession()
            await quota._read_database_storage_info()  # recovery
            holder["session"] = _FailingSession()
            await quota._read_database_storage_info()  # outage 2 → ERROR again
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 2

    async def test_instance_and_database_probes_tracked_separately(
        self, monkeypatch, caplog
    ):
        holder = {"session": _FailingSession()}
        self._install_session(monkeypatch, holder)
        with caplog.at_level(logging.DEBUG, logger=quota.logger.name):
            await quota._read_database_storage_info()
            await quota._read_storage_info()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 2  # one loud log per distinct probe query
