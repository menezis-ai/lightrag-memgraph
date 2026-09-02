"""Unit tests for Pool Bolt tuning parameters in _pool.py.

No Memgraph required — all DB interactions are mocked.
"""

import asyncio

import pytest
from neo4j import TrustCustomCAs

import twindb_lightrag_memgraph._pool as pool
from twindb_lightrag_memgraph._constants import (
    CONNECTION_POOL_SIZE,
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
    DEFAULT_IDLE_DISCONNECT_SECONDS,
    DEFAULT_MEMGRAPH_URI,
    DEFAULT_OPERATION_TIMEOUT,
    DEFAULT_READ_POOL_SIZE,
    DEFAULT_WRITE_SLOT_ACQUIRE_TIMEOUT,
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


class TestReadOperationTimeout:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_OPERATION_TIMEOUT", raising=False)
        assert pool._read_operation_timeout() == DEFAULT_OPERATION_TIMEOUT

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_OPERATION_TIMEOUT", "12.5")
        assert pool._read_operation_timeout() == 12.5

    @pytest.mark.parametrize("value", ["", "nope", "0", "-1"])
    def test_invalid_falls_back(self, monkeypatch, value):
        monkeypatch.setenv("MEMGRAPH_OPERATION_TIMEOUT", value)
        assert pool._read_operation_timeout() == DEFAULT_OPERATION_TIMEOUT


class TestReadWriteSlotAcquireTimeout:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT", raising=False)
        assert (
            pool._read_write_slot_acquire_timeout()
            == DEFAULT_WRITE_SLOT_ACQUIRE_TIMEOUT
        )

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT", "8.5")
        assert pool._read_write_slot_acquire_timeout() == 8.5

    @pytest.mark.parametrize("value", ["", "nope", "0", "-1"])
    def test_invalid_falls_back(self, monkeypatch, value):
        monkeypatch.setenv("MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT", value)
        assert (
            pool._read_write_slot_acquire_timeout()
            == DEFAULT_WRITE_SLOT_ACQUIRE_TIMEOUT
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
    def test_connection_identity_defaults(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_URI", raising=False)
        monkeypatch.delenv("MEMGRAPH_DATABASE", raising=False)

        assert pool.connection_identity() == (DEFAULT_MEMGRAPH_URI, "memgraph")

    def test_connection_identity_overrides(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_URI", "neo4j+s://cluster.example")
        monkeypatch.setenv("MEMGRAPH_DATABASE", "tenant_a")

        assert pool.connection_identity() == (
            "neo4j+s://cluster.example",
            "tenant_a",
        )

    def test_config_and_routing_use_connection_identity(self, monkeypatch):
        monkeypatch.setattr(
            pool,
            "connection_identity",
            lambda: ("neo4j+s://cluster.example", "tenant_a"),
        )

        uri, database, _kwargs = pool._read_connection_config()

        assert (uri, database) == ("neo4j+s://cluster.example", "tenant_a")
        assert pool._uses_routing_protocol() is True

    def test_password_read_from_docker_secret(self, monkeypatch, tmp_path):
        password_file = tmp_path / "memgraph-password"
        password_file.write_text("strong-password\n", encoding="utf-8")
        monkeypatch.delenv("MEMGRAPH_PASSWORD", raising=False)
        monkeypatch.setenv("MEMGRAPH_PASSWORD_FILE", str(password_file))

        _, _, kwargs = pool._read_connection_config()

        assert kwargs["auth"][1] == "strong-password"

    def test_direct_and_file_password_are_rejected(self, monkeypatch, tmp_path):
        password_file = tmp_path / "memgraph-password"
        password_file.write_text("file-password", encoding="utf-8")
        monkeypatch.setenv("MEMGRAPH_PASSWORD", "direct-password")
        monkeypatch.setenv("MEMGRAPH_PASSWORD_FILE", str(password_file))

        with pytest.raises(ValueError, match="either MEMGRAPH_PASSWORD"):
            pool._read_connection_config()

    def test_missing_password_file_fails_closed(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMGRAPH_PASSWORD", raising=False)
        monkeypatch.setenv("MEMGRAPH_PASSWORD_FILE", str(tmp_path / "does-not-exist"))

        with pytest.raises(RuntimeError, match="MEMGRAPH_PASSWORD_FILE"):
            pool._read_connection_config()

    def test_empty_password_file_fails_closed(self, monkeypatch, tmp_path):
        password_file = tmp_path / "memgraph-password"
        password_file.write_text("\n", encoding="utf-8")
        monkeypatch.delenv("MEMGRAPH_PASSWORD", raising=False)
        monkeypatch.setenv("MEMGRAPH_PASSWORD_FILE", str(password_file))

        with pytest.raises(ValueError, match="is empty"):
            pool._read_connection_config()

    def test_custom_ca_configures_verified_tls(self, monkeypatch, tmp_path):
        ca_file = tmp_path / "memgraph-ca.crt"
        ca_file.write_text("test-ca", encoding="utf-8")
        monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "true")
        monkeypatch.setenv("MEMGRAPH_TRUST", "TRUST_CUSTOM_CA")
        monkeypatch.setenv("MEMGRAPH_CA_FILE", str(ca_file))

        _, _, kwargs = pool._read_connection_config()

        assert kwargs["encrypted"] is True
        assert isinstance(kwargs["trusted_certificates"], TrustCustomCAs)

    def test_custom_ca_requires_existing_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "true")
        monkeypatch.setenv("MEMGRAPH_TRUST", "TRUST_CUSTOM_CA")
        monkeypatch.setenv("MEMGRAPH_CA_FILE", str(tmp_path / "missing.crt"))

        with pytest.raises(ValueError, match="readable file"):
            pool._read_connection_config()

    def test_unknown_tls_policy_fails_closed(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "true")
        monkeypatch.setenv("MEMGRAPH_TRUST", "TRUST_SOMETHING_ELSE")

        with pytest.raises(ValueError, match="MEMGRAPH_TRUST"):
            pool._read_connection_config()

    def test_invalid_encrypted_value_fails_closed(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_ENCRYPTED", "yes")

        with pytest.raises(ValueError, match="MEMGRAPH_ENCRYPTED"):
            pool._read_connection_config()

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

    def test_max_connection_lifetime_env_override(self, monkeypatch):
        """Deployments behind an idle-killing network path (Docker Swarm
        overlay/IPVS ~900s) lower the lifetime below the kill window so
        recycling happens by age check, not by a logged reset-by-peer."""
        monkeypatch.setenv("MEMGRAPH_MAX_CONNECTION_LIFETIME", "600")
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["max_connection_lifetime"] == 600.0

    def test_max_connection_lifetime_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_MAX_CONNECTION_LIFETIME", "-1")
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["max_connection_lifetime"] == 1800.0
        monkeypatch.setenv("MEMGRAPH_MAX_CONNECTION_LIFETIME", "soon")
        _, _, kwargs = pool._read_connection_config()
        assert kwargs["max_connection_lifetime"] == 1800.0

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


class TestSessionOperationDeadline:
    async def test_caller_cancellation_is_not_translated_to_timeout(self):
        entered = asyncio.Event()

        async def stalled_operation():
            async with pool._operation_deadline(60):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(stalled_operation())
        await entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_expiry_swallowed_by_body_still_raises_timeout(self):
        # A driver cleanup path can swallow the cancellation our timer
        # delivers. The deadline must still report the expiry on normal exit
        # instead of leaking a pending cancellation into the caller's next
        # await as a stray CancelledError.
        with pytest.raises(asyncio.TimeoutError):
            async with pool._operation_deadline(0.01):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass

        # The withdrawn request must not cancel unrelated follow-up awaits.
        await asyncio.sleep(0)
        cancelling = getattr(asyncio.current_task(), "cancelling", None)
        if cancelling is not None:  # Python 3.11+ tracks requests
            assert cancelling() == 0

    @pytest.mark.parametrize(
        ("getter_name", "context_name"),
        [
            ("get_driver", "get_session"),
            ("_get_read_driver", "get_read_session"),
        ],
    )
    async def test_stalled_session_body_times_out_and_closes_session(
        self, monkeypatch, getter_name, context_name
    ):
        class Session:
            exited = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                self.exited = True

        session = Session()

        class Driver:
            def session(self, **_kwargs):
                return session

        async def get_driver():
            return Driver(), "memgraph"

        monkeypatch.setattr(pool, getter_name, get_driver)
        # ``asyncio.timeout`` is absent on Python 3.10. Keep the regression
        # visible even when this test runs under a newer interpreter.
        monkeypatch.setattr(asyncio, "timeout", None, raising=False)
        monkeypatch.setenv("MEMGRAPH_OPERATION_TIMEOUT", "0.01")
        monkeypatch.setenv("MEMGRAPH_URI", "bolt://localhost:7687")

        with pytest.raises(asyncio.TimeoutError):
            async with getattr(pool, context_name)():
                await asyncio.Event().wait()

        assert session.exited is True

    @pytest.mark.parametrize(
        ("getter_name", "context_name"),
        [
            ("get_driver", "get_session"),
            ("_get_read_driver", "get_read_session"),
        ],
    )
    async def test_stalled_driver_acquisition_is_inside_operation_deadline(
        self, monkeypatch, getter_name, context_name
    ):
        entered = asyncio.Event()

        async def stalled_getter():
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(pool, getter_name, stalled_getter)
        monkeypatch.setenv("MEMGRAPH_OPERATION_TIMEOUT", "0.01")

        with pytest.raises(asyncio.TimeoutError):
            async with getattr(pool, context_name)():
                pytest.fail("a stalled driver getter must not yield a session")

        assert entered.is_set()

    @pytest.mark.parametrize(
        ("getter_name", "context_name"),
        [
            ("get_driver", "get_session"),
            ("_get_read_driver", "get_read_session"),
        ],
    )
    async def test_stalled_stale_driver_close_is_inside_operation_deadline(
        self, monkeypatch, getter_name, context_name
    ):
        stale_driver = object()
        close_started = asyncio.Event()

        async def stalled_close(driver, _label):
            assert driver is stale_driver
            close_started.set()
            await asyncio.Event().wait()

        if getter_name == "get_driver":
            pool._driver = stale_driver
            pool._bound_loop_id = -1
        else:
            pool._read_driver = stale_driver
            pool._read_bound_loop_id = -1

        class DriverFactory:
            @staticmethod
            def driver(*_args, **_kwargs):
                return object()

        monkeypatch.setattr(pool, "AsyncGraphDatabase", DriverFactory)
        monkeypatch.setattr(pool, "_close_stale_driver", stalled_close)
        monkeypatch.setenv("MEMGRAPH_OPERATION_TIMEOUT", "0.01")

        with pytest.raises(asyncio.TimeoutError):
            async with getattr(pool, context_name)():
                pytest.fail("a stalled stale-driver close must not yield a session")

        assert close_started.is_set()
