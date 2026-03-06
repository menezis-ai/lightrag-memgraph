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
