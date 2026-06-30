"""Unit tests for Pool Bolt tuning parameters in _pool.py.

No Memgraph required — all DB interactions are mocked.
"""

import pytest

import twindb_lightrag_memgraph._pool as pool
from twindb_lightrag_memgraph._constants import (
    CONNECTION_POOL_SIZE,
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
    DEFAULT_IDLE_DISCONNECT_SECONDS,
    DEFAULT_READ_POOL_SIZE,
)


@pytest.fixture(autouse=True)
def reset_pool_state():
    """Reset pool state between tests."""
    pool._driver = None
    pool._database = None
    pool._bound_loop_id = None
    pool._read_driver = None
    pool._read_database = None
    pool._read_bound_loop_id = None
    pool._last_driver_activity = None
    pool._last_read_driver_activity = None
    pool._write_semaphore = None
    pool._semaphore_loop_id = None
    yield
    pool._driver = None
    pool._database = None
    pool._bound_loop_id = None
    pool._read_driver = None
    pool._read_database = None
    pool._read_bound_loop_id = None
    pool._last_driver_activity = None
    pool._last_read_driver_activity = None
    pool._write_semaphore = None
    pool._semaphore_loop_id = None


# ── _read_pool_size ───────────────────────────────────────────────────


class TestReadPoolSize:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_POOL_SIZE", raising=False)
        assert pool._read_pool_size() == CONNECTION_POOL_SIZE

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_POOL_SIZE", "100")
        assert pool._read_pool_size() == 100

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_POOL_SIZE", "abc")
        assert pool._read_pool_size() == CONNECTION_POOL_SIZE

    def test_zero_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_POOL_SIZE", "0")
        assert pool._read_pool_size() == CONNECTION_POOL_SIZE

    def test_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_POOL_SIZE", "-5")
        assert pool._read_pool_size() == CONNECTION_POOL_SIZE


# ── _read_connection_acquire_timeout ──────────────────────────────────


class TestReadConnectionAcquireTimeout:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", raising=False)
        assert (
            pool._read_connection_acquire_timeout()
            == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
        )

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", "10.0")
        assert pool._read_connection_acquire_timeout() == 10.0

    def test_integer_accepted(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", "3")
        assert pool._read_connection_acquire_timeout() == 3.0

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", "nope")
        assert (
            pool._read_connection_acquire_timeout()
            == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
        )

    def test_zero_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", "0")
        assert (
            pool._read_connection_acquire_timeout()
            == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
        )

    def test_negative_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", "-1.5")
        assert (
            pool._read_connection_acquire_timeout()
            == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
        )


# ── _read_idle_disconnect_seconds ────────────────────────────────────


class TestReadIdleDisconnectSeconds:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_IDLE_DISCONNECT_SECONDS", raising=False)
        assert pool._read_idle_disconnect_seconds() == DEFAULT_IDLE_DISCONNECT_SECONDS

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_IDLE_DISCONNECT_SECONDS", "7200")
        assert pool._read_idle_disconnect_seconds() == 7200.0

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_IDLE_DISCONNECT_SECONDS", "nope")
        assert pool._read_idle_disconnect_seconds() == DEFAULT_IDLE_DISCONNECT_SECONDS

    def test_zero_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_IDLE_DISCONNECT_SECONDS", "0")
        assert pool._read_idle_disconnect_seconds() == DEFAULT_IDLE_DISCONNECT_SECONDS


# ── _read_read_pool_size ──────────────────────────────────────────────


class TestReadReadPoolSize:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_READ_POOL_SIZE", raising=False)
        assert pool._read_read_pool_size() == DEFAULT_READ_POOL_SIZE

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_READ_POOL_SIZE", "30")
        assert pool._read_read_pool_size() == 30

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_READ_POOL_SIZE", "abc")
        assert pool._read_read_pool_size() == DEFAULT_READ_POOL_SIZE

    def test_zero_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_READ_POOL_SIZE", "0")
        assert pool._read_read_pool_size() == DEFAULT_READ_POOL_SIZE


# ── _read_connection_config ───────────────────────────────────────────


class TestConnectionConfig:
    def test_timeout_in_driver_kwargs(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", "7.5")
        monkeypatch.delenv("MEMGRAPH_POOL_SIZE", raising=False)
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["connection_acquisition_timeout"] == 7.5
        assert kwargs["max_connection_pool_size"] == CONNECTION_POOL_SIZE

    def test_pool_size_in_driver_kwargs(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_POOL_SIZE", "75")
        monkeypatch.delenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", raising=False)
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["max_connection_pool_size"] == 75
        assert (
            kwargs["connection_acquisition_timeout"]
            == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
        )

    def test_pool_size_override(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_POOL_SIZE", "75")
        _, _, kwargs = pool._read_connection_config(pool_size_override=20)
        assert kwargs["max_connection_pool_size"] == 20

    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_POOL_SIZE", raising=False)
        monkeypatch.delenv("MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT", raising=False)
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["max_connection_pool_size"] == CONNECTION_POOL_SIZE
        assert (
            kwargs["connection_acquisition_timeout"]
            == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
        )

    def test_liveness_check_and_max_lifetime_set(self, monkeypatch):
        """Bolt connection resilience: idle Memgraph sockets get recycled.

        Without these two, the driver hands out defunct connections and
        the next ``session.run(...)`` fails with
        ``ConnectionResetError(104, 'Connection reset by peer')`` — the
        exact 500 chain seen on 2026-06-07 (POST /tags, GET /activity,
        GET /notifications) before the fix.
        """
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["liveness_check_timeout"] == 5
        assert kwargs["max_connection_lifetime"] == 1800

    def test_driver_kwargs_accepted_by_async_driver(self):
        """Smoke-check: the runtime neo4j driver accepts the new kwargs.

        Guards against the pinned range ``neo4j>=5,<7`` shipping a build
        that drops ``liveness_check_timeout`` (added in 5.17). We
        construct a driver against an unreachable URI — the assert is
        that ``AsyncGraphDatabase.driver(...)`` itself does not raise
        ``TypeError`` because of an unknown kwarg.
        """
        from neo4j import AsyncGraphDatabase

        _, _, kwargs = pool._read_connection_config()
        # Drop auth so the constructor is pure kwarg-validation.
        kwargs.pop("auth", None)
        driver = AsyncGraphDatabase.driver("bolt://127.0.0.1:1", **kwargs)
        # No await close — sync close on an unopened driver is a no-op
        # for our purpose (no socket was opened by the constructor).
        assert driver is not None

    async def test_write_driver_recreated_after_idle_disconnect(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_IDLE_DISCONNECT_SECONDS", "3600")
        first = object()
        second = object()
        closed = []

        class DriverFactory:
            calls = 0

            @classmethod
            def driver(cls, *_args, **_kwargs):
                cls.calls += 1
                return first if cls.calls == 1 else second

        async def close_stale(driver, label):
            closed.append((driver, label))

        monkeypatch.setattr(pool, "AsyncGraphDatabase", DriverFactory)
        monkeypatch.setattr(pool, "_close_stale_driver", close_stale)

        driver, _ = await pool.get_driver()
        assert driver is first
        pool._last_driver_activity -= 3601

        driver, _ = await pool.get_driver()

        assert driver is second
        assert closed == [(first, "write")]

    async def test_read_driver_recreated_after_idle_disconnect(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_IDLE_DISCONNECT_SECONDS", "3600")
        first = object()
        second = object()
        closed = []

        class DriverFactory:
            calls = 0

            @classmethod
            def driver(cls, *_args, **_kwargs):
                cls.calls += 1
                return first if cls.calls == 1 else second

        async def close_stale(driver, label):
            closed.append((driver, label))

        monkeypatch.setattr(pool, "AsyncGraphDatabase", DriverFactory)
        monkeypatch.setattr(pool, "_close_stale_driver", close_stale)

        driver, _ = await pool._get_read_driver()
        assert driver is first
        pool._last_read_driver_activity -= 3601

        driver, _ = await pool._get_read_driver()

        assert driver is second
        assert closed == [(first, "read")]
