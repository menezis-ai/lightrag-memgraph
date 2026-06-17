"""Unit tests for Pool Bolt tuning parameters in _pool.py.

No Memgraph required — all DB interactions are mocked.
"""

import pytest

import twindb_lightrag_memgraph._pool as pool
from twindb_lightrag_memgraph._constants import (
    CONNECTION_POOL_SIZE,
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
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
    pool._write_semaphore = None
    pool._semaphore_loop_id = None
    yield
    pool._driver = None
    pool._database = None
    pool._bound_loop_id = None
    pool._read_driver = None
    pool._read_database = None
    pool._read_bound_loop_id = None
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

    def test_require_tls_rejects_plaintext_bolt(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_REQUIRE_TLS", "true")
        monkeypatch.setenv("MEMGRAPH_URI", "bolt://memgraph.internal:7687")
        monkeypatch.delenv("MEMGRAPH_ENCRYPTED", raising=False)
        with pytest.raises(ValueError, match="MEMGRAPH_REQUIRE_TLS"):
            pool._read_connection_config()

    def test_require_tls_accepts_encrypted_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_REQUIRE_TLS", "true")
        monkeypatch.setenv("MEMGRAPH_URI", "bolt://memgraph.internal:7687")
        monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "true")
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["encrypted"] is True

    def test_require_tls_accepts_tls_scheme(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_REQUIRE_TLS", "true")
        monkeypatch.setenv("MEMGRAPH_URI", "bolt+s://memgraph.internal:7687")
        monkeypatch.delenv("MEMGRAPH_ENCRYPTED", raising=False)
        # Must not raise — TLS is carried by the +s scheme.
        uri, _, _ = pool._read_connection_config()
        assert uri == "bolt+s://memgraph.internal:7687"

    def test_plaintext_allowed_when_not_required(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_REQUIRE_TLS", raising=False)
        monkeypatch.setenv("MEMGRAPH_URI", "bolt://memgraph.internal:7687")
        monkeypatch.delenv("MEMGRAPH_ENCRYPTED", raising=False)
        # Warns (remote plaintext) but does not raise.
        uri, _, _ = pool._read_connection_config()
        assert uri == "bolt://memgraph.internal:7687"

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
