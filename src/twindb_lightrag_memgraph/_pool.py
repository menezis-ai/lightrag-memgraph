"""
Shared async connection pool for all Memgraph storage backends.

Singleton pattern: one Bolt driver per event loop, shared across
MemgraphStorage (built-in graph), KV, Vector, and DocStatus.

Handles event loop changes (e.g. between test functions) by detecting
when the loop has changed and recreating the driver.
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from typing import Any, Callable
from urllib.parse import urlparse

from neo4j import AsyncGraphDatabase, TrustAll, TrustCustomCAs, TrustSystemCAs
from neo4j.exceptions import ClientError as Neo4jClientError

from ._constants import (
    CONNECTION_POOL_SIZE,
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
    DEFAULT_IDLE_DISCONNECT_SECONDS,
    DEFAULT_MAX_CONNECTION_LIFETIME,
    DEFAULT_MEMGRAPH_URI,
    DEFAULT_OPERATION_TIMEOUT,
    DEFAULT_READ_POOL_SIZE,
    DEFAULT_WRITE_CONCURRENCY,
    DEFAULT_WRITE_SLOT_ACQUIRE_TIMEOUT,
    MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT_ENV,
    MEMGRAPH_IDLE_DISCONNECT_SECONDS_ENV,
    MEMGRAPH_MAX_CONNECTION_LIFETIME_ENV,
    MEMGRAPH_OPERATION_TIMEOUT_ENV,
    MEMGRAPH_POOL_SIZE_ENV,
    MEMGRAPH_READ_POOL_SIZE_ENV,
    MEMGRAPH_WRITE_CONCURRENCY_ENV,
    MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT_ENV,
    validate_identifier,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

_MUST_BE_POSITIVE = "must be >= 1"
_MUST_BE_STRICTLY_POSITIVE = "must be > 0"

_thread_lock = threading.Lock()
_driver = None
_database = None
_bound_loop_id = None
_last_driver_activity = None

_write_semaphore = None
_semaphore_loop_id = None

_read_driver = None
_read_database = None
_read_bound_loop_id = None
_last_read_driver_activity = None

# Enterprise multi-database detection.
# None = not yet probed, True = USE DATABASE succeeded at least once. There is
# deliberately NO ``False`` state: a refused ``USE DATABASE`` on a non-default
# database is an error (see MemgraphDatabaseUnavailableError), never a cached
# "skip from now on" — the refusal must be observed by every session so a
# restored licence recovers without a restart, and a broken one never becomes
# silent.
_enterprise_supported: bool | None = None
_storage_metric_recorder: Callable[[str], None] | None = None


def set_storage_metric_recorder(
    recorder: Callable[[str], None] | None,
) -> None:
    """Install the optional server metrics hook without adding a storage dep.

    Storage-only installs do not carry ``prometheus-client``.  The server extra
    registers this callback when its metrics module is imported; otherwise the
    write path remains behavior-identical.
    """
    global _storage_metric_recorder
    _storage_metric_recorder = recorder


def _record_storage_metric(outcome: str) -> None:
    recorder = _storage_metric_recorder
    if recorder is None:
        return
    try:
        recorder(outcome)
    except Exception:  # pragma: no cover - observability must not break storage
        logger.debug("Storage metric recorder failed", exc_info=True)


class MemgraphDatabaseUnavailableError(RuntimeError):
    """``MEMGRAPH_DATABASE`` names a database this server cannot select.

    Raised when ``USE DATABASE <name>`` is refused because the server has no
    Enterprise licence (multi-database is an Enterprise feature). Before
    2026-08-25 the package logged this at INFO and silently kept working on
    the default ``memgraph`` database — in a one-database-per-KB topology that
    merges several KBs into the default database without any error. The
    failure is now fail-closed: the session is refused, ``initialize()``
    fails at boot and ``/ready`` reports the database as unreachable.

    Remedies, in the message: unset ``MEMGRAPH_DATABASE`` (single-database
    deployment, Community edition) or install the Enterprise licence.
    """


def _is_closed_transport_error(exc: BaseException) -> bool:
    """Return True for stale/defunct Bolt transport failures worth retrying once."""
    text = str(exc).lower()
    return (
        "tcptransport closed=true" in text
        or "the handler is closed" in text
        or "connection reset by peer" in text
        or "failed to read from defunct connection" in text
    )


def is_sync_replication_error(exc: BaseException) -> bool:
    """Return True for Memgraph SYNC replication acknowledgement failures.

    In Memgraph ``SYNC`` mode the write can be committed on MAIN even when the
    client receives an error because one replica did not acknowledge in time.
    Callers must not blindly translate this class of error to "business write
    definitely failed".
    """
    message = str(exc).lower()
    if "failed to replicate" not in message:
        return False
    return "sync" in message or "replica" in message


def memgraph_exception_payload(exc: BaseException) -> dict[str, Any]:
    """Build a sanitized operator-facing payload for Memgraph exceptions."""
    if is_sync_replication_error(exc):
        return {
            "type": "MemgraphSyncReplicationWarning",
            "message": (
                "Memgraph write may have committed on MAIN but was not "
                "confirmed by every SYNC replica before the driver returned."
            ),
            "operator_action": (
                "Verify the object on MAIN / SHOW REPLICAS before retrying; "
                "a naive retry may duplicate non-idempotent writes."
            ),
            "commit_may_have_succeeded": True,
            "retry_safe": False,
            "detail": _short_exception_detail(exc),
        }
    return {
        "type": "MemgraphDependencyError",
        "message": "Memgraph dependency error.",
        "operator_action": "Check Memgraph connectivity, role, and replication health.",
        "commit_may_have_succeeded": False,
        "retry_safe": None,
        "detail": _short_exception_detail(exc),
    }


def _short_exception_detail(exc: BaseException, *, max_len: int = 500) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) <= max_len:
        return detail
    return detail[: max_len - 3] + "..."


async def _close_stale_driver(driver, label: str) -> None:
    """Best-effort close for a driver replaced after an event-loop change.

    This helper is intentionally awaited outside ``_thread_lock``. A synchronous
    ``threading.Lock`` held across an ``await`` can block the event loop if
    another coroutine reaches the same lock while the close is suspended.
    """
    try:
        await driver.close()
    except Exception as e:
        logger.debug("Error closing stale %s driver: %s", label, e)


def _read_pool_size() -> int:
    """Read MEMGRAPH_POOL_SIZE from env, default CONNECTION_POOL_SIZE (50)."""
    raw = os.environ.get(MEMGRAPH_POOL_SIZE_ENV, "")
    try:
        val = int(raw)
        if val < 1:
            raise ValueError(_MUST_BE_POSITIVE)
        return val
    except (ValueError, TypeError):
        return CONNECTION_POOL_SIZE


def _read_connection_acquire_timeout() -> float:
    """Read MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT from env, default 5.0s."""
    raw = os.environ.get(MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT_ENV, "")
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError(_MUST_BE_STRICTLY_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_CONNECTION_ACQUIRE_TIMEOUT


def _read_max_connection_lifetime() -> float:
    """Read MEMGRAPH_MAX_CONNECTION_LIFETIME from env, default 1800s.

    Lower it below the network path's idle-kill window (Docker Swarm
    overlay/IPVS resets idle TCP around 900s) to recycle pooled connections
    by age check instead of by a logged reset-by-peer on next use.
    """
    raw = os.environ.get(MEMGRAPH_MAX_CONNECTION_LIFETIME_ENV, "")
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError(_MUST_BE_STRICTLY_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_MAX_CONNECTION_LIFETIME


def _read_operation_timeout() -> float:
    """Read the deadline for work performed with an acquired Bolt session."""
    raw = os.environ.get(MEMGRAPH_OPERATION_TIMEOUT_ENV, "")
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError(_MUST_BE_STRICTLY_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_OPERATION_TIMEOUT


def _read_write_slot_acquire_timeout() -> float:
    """Read the deadline for waiting on the process-local write semaphore."""
    raw = os.environ.get(MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT_ENV, "")
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError(_MUST_BE_STRICTLY_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_WRITE_SLOT_ACQUIRE_TIMEOUT


@asynccontextmanager
async def _operation_deadline(seconds: float):
    """Bound an async block without requiring Python 3.11 ``asyncio.timeout``.

    Kept although the supported floor is now Python 3.12 (2026-07-25) —
    ``asyncio.timeout`` exists there, but this helper predates the floor
    bump and swapping it is churn without behavior change.  A loop timer
    cancels the current task when the deadline expires; only that cancellation
    is translated to ``asyncio.TimeoutError``.  Cancellation requested by a
    caller remains ``CancelledError``.
    """
    task = asyncio.current_task()
    if task is None:  # pragma: no cover - an async context always owns a task
        raise RuntimeError("operation deadline requires a running asyncio task")

    cancelling = getattr(task, "cancelling", None)
    initial_cancels = cancelling() if cancelling is not None else 0
    expired = False

    def _cancel_on_expiry() -> None:
        nonlocal expired
        expired = True
        task.cancel()

    handle = asyncio.get_running_loop().call_later(seconds, _cancel_on_expiry)
    try:
        yield
    except asyncio.CancelledError as exc:  # NOSONAR - deliberate translation:
        # external cancellation re-raises on both guarded paths below; only
        # this context's own expiry cancellation becomes TimeoutError (the
        # same contract as Python 3.11's ``asyncio.timeout``).
        if not expired:
            raise

        # Python 3.11 tracks cancellation requests. Remove only our timeout
        # cancellation; if another caller also cancelled the task, preserve it.
        uncancel = getattr(task, "uncancel", None)
        if uncancel is not None and uncancel() > initial_cancels:
            raise
        raise asyncio.TimeoutError from exc
    else:
        # The timer can fire after the body's last suspension point, or the
        # body may swallow the cancellation itself (driver cleanup paths do).
        # Without translation here the operation would "succeed" while our
        # cancellation request leaks into the caller's next await as a stray
        # CancelledError. Withdraw the request, then report the expiry.
        if expired:
            uncancel = getattr(task, "uncancel", None)
            if uncancel is not None:
                uncancel()
            raise asyncio.TimeoutError
    finally:
        handle.cancel()


def _read_idle_disconnect_seconds() -> float:
    """Read MEMGRAPH_IDLE_DISCONNECT_SECONDS, default 3600s.

    This is intentionally separate from neo4j-driver's
    ``liveness_check_timeout``. The driver can ping an idle socket before reuse,
    but BNP SREs asked us to force a disconnect after one hour of inactivity.
    """
    raw = os.environ.get(MEMGRAPH_IDLE_DISCONNECT_SECONDS_ENV, "")
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError(_MUST_BE_STRICTLY_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_IDLE_DISCONNECT_SECONDS


def _read_read_pool_size() -> int:
    """Read MEMGRAPH_READ_POOL_SIZE from env, default 20."""
    raw = os.environ.get(MEMGRAPH_READ_POOL_SIZE_ENV, "")
    try:
        val = int(raw)
        if val < 1:
            raise ValueError(_MUST_BE_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_READ_POOL_SIZE


def _read_secret(name: str, default: str = "") -> str:
    """Read a credential from ``NAME`` or Docker-style ``NAME_FILE``.

    Exactly one source may be configured. File errors and empty secret files
    are fatal: silently falling back to an empty password would disable the
    authentication posture the operator intended to deploy.
    """
    direct_value = os.environ.get(name)
    file_path = os.environ.get(f"{name}_FILE", "").strip()

    if direct_value is not None and file_path:
        raise ValueError(f"Set either {name} or {name}_FILE, not both")
    if not file_path:
        return default if direct_value is None else direct_value

    try:
        value = Path(file_path).read_text(encoding="utf-8").rstrip("\r\n")
    except OSError as exc:
        raise RuntimeError(f"Unable to read secret configured by {name}_FILE") from exc
    if not value:
        raise ValueError(f"Secret configured by {name}_FILE is empty")
    return value


def _warn_for_unencrypted_remote_connection(uri: str, encrypted_env: str) -> None:
    parsed_uri = urlparse(uri)
    localhost = {"localhost", "127.0.0.1", "::1"}
    remote_plaintext = (
        parsed_uri.scheme in ("bolt", "neo4j")
        and parsed_uri.hostname
        and parsed_uri.hostname not in localhost
        and encrypted_env != "true"
    )
    if remote_plaintext:
        logger.warning(
            "Plaintext Bolt connection to remote host %s — credentials "
            "will be sent unencrypted. Use bolt+s:// or set "
            "MEMGRAPH_ENCRYPTED=true for TLS.",
            parsed_uri.hostname,
        )


def _read_connection_config(*, pool_size_override: int | None = None):
    """Read Memgraph connection parameters from environment variables.

    All settings come from ``os.environ`` (compatible with HashiCorp Vault
    agent injection, Kubernetes secrets, systemd ``EnvironmentFile``, etc.).

    Args:
        pool_size_override: If provided, overrides the pool size from env.
            Used by the read pool to have its own size.

    Returns:
        tuple: (uri, database, driver_kwargs) where driver_kwargs is ready
        to be passed to ``AsyncGraphDatabase.driver(uri, **driver_kwargs)``.
    """
    uri = os.environ.get("MEMGRAPH_URI", DEFAULT_MEMGRAPH_URI)
    username = _read_secret("MEMGRAPH_USERNAME")
    password = _read_secret("MEMGRAPH_PASSWORD")
    database = os.environ.get("MEMGRAPH_DATABASE", "memgraph")
    validate_identifier(database, "database")

    encrypted_env = os.environ.get("MEMGRAPH_ENCRYPTED", "").strip().lower()
    if encrypted_env not in {"", "true", "false"}:
        raise ValueError("MEMGRAPH_ENCRYPTED must be 'true' or 'false'")

    _warn_for_unencrypted_remote_connection(uri, encrypted_env)

    driver_kwargs = {
        "auth": (username, password),
        "max_connection_pool_size": (
            pool_size_override if pool_size_override is not None else _read_pool_size()
        ),
        "connection_acquisition_timeout": _read_connection_acquire_timeout(),
        # Memgraph can drop idle Bolt connections (server-side reset) while
        # the driver still holds them in the pool. Without a liveness probe
        # the next `session.run(...)` fails with `ConnectionResetError(104,
        # 'Connection reset by peer')` and returns 500 to the caller. Cap
        # connection age at 30 min and ping any idle connection older than
        # 5 s before reuse so the driver transparently recycles defunct
        # sockets. (Dropped from 30 s to 5 s after a 2026-06-08 reset
        # surfaced during a doc delete that idled longer than 5 s but
        # less than 30.) Requires neo4j-driver >= 5.17 (pinned >= 5.0,<7).
        "max_connection_lifetime": _read_max_connection_lifetime(),
        "liveness_check_timeout": 5,
    }

    if encrypted_env == "true":
        driver_kwargs["encrypted"] = True
        trust_env = os.environ.get("MEMGRAPH_TRUST", "TRUST_SYSTEM_CA").upper()
        if trust_env == "TRUST_ALL":
            logger.warning(
                "Memgraph TLS trust set to TRUST_ALL — certificate "
                "verification is DISABLED. Do not use in production."
            )
            driver_kwargs["trusted_certificates"] = TrustAll()
        elif trust_env == "TRUST_SYSTEM_CA":
            driver_kwargs["trusted_certificates"] = TrustSystemCAs()
        elif trust_env == "TRUST_CUSTOM_CA":
            ca_file = os.environ.get("MEMGRAPH_CA_FILE", "").strip()
            if not ca_file:
                raise ValueError("MEMGRAPH_CA_FILE is required with TRUST_CUSTOM_CA")
            if not Path(ca_file).is_file():
                raise ValueError("MEMGRAPH_CA_FILE must reference a readable file")
            driver_kwargs["trusted_certificates"] = TrustCustomCAs(ca_file)
        else:
            raise ValueError(
                "MEMGRAPH_TRUST must be TRUST_SYSTEM_CA, TRUST_CUSTOM_CA, "
                "or TRUST_ALL"
            )
        logger.info("Memgraph TLS enabled (trust=%s)", trust_env)
    elif encrypted_env == "false":
        driver_kwargs["encrypted"] = False

    return uri, database, driver_kwargs


def _idle_expired(last_activity: float | None) -> bool:
    if last_activity is None:
        return False
    return monotonic() - last_activity >= _read_idle_disconnect_seconds()


def _mark_write_activity() -> None:
    global _last_driver_activity
    _last_driver_activity = monotonic()


def _mark_read_activity() -> None:
    global _last_read_driver_activity
    _last_read_driver_activity = monotonic()


async def get_driver():
    """Get or create the shared AsyncGraphDatabase driver.

    If the event loop has changed since the driver was created,
    the old driver is closed and a new one is created.

    Returns:
        tuple: (driver, database_name)
    """
    global _driver, _database, _bound_loop_id, _last_driver_activity

    current_loop_id = id(asyncio.get_running_loop())

    # Fast path: driver exists and is bound to the current loop
    if (
        _driver is not None
        and _bound_loop_id == current_loop_id
        and not _idle_expired(_last_driver_activity)
    ):
        return _driver, _database

    stale_driver = None
    with _thread_lock:
        # Double-check after acquiring lock
        if (
            _driver is not None
            and _bound_loop_id == current_loop_id
            and not _idle_expired(_last_driver_activity)
        ):
            return _driver, _database

        uri, database, driver_kwargs = _read_connection_config()
        new_driver = AsyncGraphDatabase.driver(uri, **driver_kwargs)
        stale_driver = _driver
        _driver = new_driver
        _database = database
        _bound_loop_id = current_loop_id
        _last_driver_activity = monotonic()
        _parsed = urlparse(uri)
        _safe_uri = f"{_parsed.scheme}://{_parsed.hostname}:{_parsed.port}"
        logger.info(
            "Memgraph Bolt driver created — uri=%s database=%s",
            _safe_uri,
            _database,
        )
        driver, database = _driver, _database
    if stale_driver is not None:
        await _close_stale_driver(stale_driver, "write")
    return driver, database


@asynccontextmanager
async def get_session():
    """Get a session with proper database routing.

    * ``neo4j://`` / ``neo4j+s://`` / ``neo4j+ssc://`` — routing protocol.
      The driver performs cluster discovery; ``database=`` is passed natively
      to ``session()`` so that routing targets the correct database.

    * ``bolt://`` / ``bolt+s://`` / ``bolt+ssc://`` — direct connection.
      Memgraph Community/Coordinator rejects ``database=`` in the Bolt
      handshake (``GQL 50N42``), so we issue ``USE DATABASE`` instead.
      On Memgraph Community (no Enterprise license), ``USE DATABASE``
      fails — the session is refused (``MemgraphDatabaseUnavailableError``),
      on every attempt, never silently redirected to the default database.
    """
    # Include driver creation/replacement and stale-driver shutdown in the
    # operation deadline.  Those paths can await I/O before the Neo4j driver's
    # connection-acquisition timeout applies, often while the caller already
    # owns a write-semaphore slot.
    async with _operation_deadline(_read_operation_timeout()):
        driver, database = await get_driver()
        if _uses_routing_protocol():
            async with driver.session(database=database) as session:
                try:
                    yield session
                finally:
                    _mark_write_activity()
        else:
            async with driver.session() as session:
                if database:
                    await _try_use_database(session, database)
                try:
                    yield session
                finally:
                    _mark_write_activity()


async def _get_read_driver():
    """Get or create the shared read-only AsyncGraphDatabase driver.

    Separate pool from the write driver, isolating reads from write pressure.
    If the event loop has changed, the old driver is closed and recreated.

    Returns:
        tuple: (driver, database_name)
    """
    global _read_driver, _read_database, _read_bound_loop_id
    global _last_read_driver_activity

    current_loop_id = id(asyncio.get_running_loop())

    if (
        _read_driver is not None
        and _read_bound_loop_id == current_loop_id
        and not _idle_expired(_last_read_driver_activity)
    ):
        return _read_driver, _read_database

    stale_driver = None
    with _thread_lock:
        if (
            _read_driver is not None
            and _read_bound_loop_id == current_loop_id
            and not _idle_expired(_last_read_driver_activity)
        ):
            return _read_driver, _read_database

        read_pool_size = _read_read_pool_size()
        uri, database, driver_kwargs = _read_connection_config(
            pool_size_override=read_pool_size,
        )
        new_driver = AsyncGraphDatabase.driver(uri, **driver_kwargs)
        stale_driver = _read_driver
        _read_driver = new_driver
        _read_database = database
        _read_bound_loop_id = current_loop_id
        _last_read_driver_activity = monotonic()
        _parsed = urlparse(uri)
        _safe_uri = f"{_parsed.scheme}://{_parsed.hostname}:{_parsed.port}"
        logger.info(
            "Memgraph READ Bolt driver created — uri=%s database=%s pool_size=%d",
            _safe_uri,
            _read_database,
            read_pool_size,
        )
        driver, database = _read_driver, _read_database
    if stale_driver is not None:
        await _close_stale_driver(stale_driver, "read")
    return driver, database


@asynccontextmanager
async def get_read_session():
    """Get a read-only session from the dedicated read pool.

    Uses the same database routing logic as ``get_session()`` but draws
    connections from a separate pool, isolating reads from write pressure.
    """
    async with _operation_deadline(_read_operation_timeout()):
        driver, database = await _get_read_driver()
        if _uses_routing_protocol():
            async with driver.session(database=database) as session:
                try:
                    yield session
                finally:
                    _mark_read_activity()
        else:
            async with driver.session() as session:
                if database:
                    await _try_use_database(session, database)
                try:
                    yield session
                finally:
                    _mark_read_activity()


async def _try_use_database(session, database: str) -> None:
    """Issue ``USE DATABASE`` when *database* is not the Community default.

    ``"memgraph"`` is the default database on every edition, so no switch is
    attempted for it. For any other name the command is issued on **every**
    session (a Bolt session starts on the default database). If the server
    refuses it because multi-database needs an Enterprise licence, raise
    :class:`MemgraphDatabaseUnavailableError` instead of falling back to the
    default database — fail-closed, and not cached, so the next session
    re-probes (a licence installed at runtime recovers without a restart).
    Other client errors propagate unchanged.
    """
    global _enterprise_supported

    # "memgraph" is the default database on Community — no need to switch.
    if database == "memgraph":
        return

    try:
        result = await session.run(f"USE DATABASE {database}")
        await result.consume()
        if _enterprise_supported is None:
            _enterprise_supported = True
            logger.debug(
                "USE DATABASE %s succeeded — Enterprise multi-database enabled",
                database,
            )
    except Neo4jClientError as exc:
        if "enterprise" in str(exc).lower() or "license" in str(exc).lower():
            logger.error(
                "MEMGRAPH_DATABASE=%s cannot be selected: the server refused "
                "USE DATABASE (%s). Refusing the session rather than silently "
                "working on the default database.",
                database,
                exc,
            )
            raise MemgraphDatabaseUnavailableError(
                f"MEMGRAPH_DATABASE={database!r} cannot be selected on this "
                "Memgraph server: USE DATABASE was refused (multi-database "
                "requires an Enterprise licence). Refusing to run on the "
                "default database instead. Fix: unset MEMGRAPH_DATABASE for a "
                "single-database (Community) deployment, or install the "
                "Enterprise licence on the server."
            ) from exc
        raise


def _uses_routing_protocol() -> bool:
    """Return True when the URI scheme uses the routing protocol.

    ``neo4j://``, ``neo4j+s://``, ``neo4j+ssc://`` use the routing
    protocol — the driver performs cluster discovery and routes queries
    to the correct server.  The ``database=`` parameter **must** be
    passed to ``session()`` so that routing happens on the right database.

    ``bolt://``, ``bolt+s://``, ``bolt+ssc://`` are direct connections —
    Memgraph Community/Coordinator rejects ``database=`` in the Bolt
    handshake (``GQL 50N42``), so we use ``USE DATABASE`` instead.
    """
    uri = os.environ.get("MEMGRAPH_URI", DEFAULT_MEMGRAPH_URI)
    return urlparse(uri).scheme.startswith("neo4j")


def _read_write_concurrency() -> int:
    """Read MEMGRAPH_WRITE_CONCURRENCY from env, default 8."""
    raw = os.environ.get(MEMGRAPH_WRITE_CONCURRENCY_ENV, "")
    try:
        val = int(raw)
        if val < 1:
            raise ValueError(_MUST_BE_POSITIVE)
        return val
    except (ValueError, TypeError):
        return DEFAULT_WRITE_CONCURRENCY


def _get_write_semaphore() -> asyncio.Semaphore:
    """Get or (re)create the write semaphore, respecting loop changes."""
    global _write_semaphore, _semaphore_loop_id

    current_loop_id = id(asyncio.get_running_loop())

    if _write_semaphore is not None and _semaphore_loop_id == current_loop_id:
        return _write_semaphore

    with _thread_lock:
        if _write_semaphore is not None and _semaphore_loop_id == current_loop_id:
            return _write_semaphore

        limit = _read_write_concurrency()
        _write_semaphore = asyncio.Semaphore(limit)
        _semaphore_loop_id = current_loop_id
        logger.debug(
            "Write semaphore (re)created — loop=%d, concurrency=%d",
            current_loop_id,
            limit,
        )
        return _write_semaphore


async def _cancel_and_reap_acquire(acquire_task: asyncio.Task[bool]) -> bool:
    """Cancel an acquire task and return whether it had already won a permit."""
    acquire_task.cancel()
    # gather(return_exceptions=True) consumes cancellation of the child only.
    # Cancellation of this cleanup coroutine still propagates normally.
    await asyncio.gather(acquire_task, return_exceptions=True)
    if acquire_task.cancelled():
        return False
    return acquire_task.result()


@asynccontextmanager
async def acquire_write_slot():
    """Gate write operations to ``MEMGRAPH_WRITE_CONCURRENCY`` slots.

    Only write operations (upsert, delete, drop) should use this.
    Read operations must NOT acquire this so they remain unthrottled.
    """
    sem = _get_write_semaphore()
    acquire_task = asyncio.create_task(sem.acquire())
    outcome = "error"
    try:
        try:
            done, _ = await asyncio.wait(
                {acquire_task}, timeout=_read_write_slot_acquire_timeout()
            )
        except asyncio.CancelledError:
            # A cancellation can race with the semaphore granting the permit. If
            # the acquire completed, return that permit before propagating.
            acquired = await _cancel_and_reap_acquire(acquire_task)
            if acquired:
                sem.release()
            raise

        if acquire_task not in done:
            acquired = await _cancel_and_reap_acquire(acquire_task)
            if acquired:
                sem.release()
            raise asyncio.TimeoutError("Memgraph write queue exhausted")

        acquired = acquire_task.result()
        try:
            yield
            outcome = "success"
        finally:
            if acquired:
                sem.release()
    finally:
        _record_storage_metric(outcome)


async def close_driver():
    """Close write and read drivers. Call on application shutdown.

    The module state is detached and reset **first**, then each driver is
    closed independently: a failing ``close()`` on the write pool must not
    leave the read pool open nor the globals pointing at a half-closed
    driver (review of #451). Every driver gets its close attempt; the first
    failure is re-raised afterwards so callers that care still see it.
    """
    global _driver, _bound_loop_id, _write_semaphore, _semaphore_loop_id
    global _last_driver_activity, _last_read_driver_activity
    global _read_driver, _read_database, _read_bound_loop_id
    global _enterprise_supported
    write_driver, read_driver = _driver, _read_driver
    _driver = None
    _bound_loop_id = None
    _last_driver_activity = None
    _write_semaphore = None
    _semaphore_loop_id = None
    _enterprise_supported = None
    _read_driver = None
    _read_database = None
    _read_bound_loop_id = None
    _last_read_driver_activity = None
    # Deferred import: _capabilities imports get_read_session from this module,
    # so a top-level import here would be circular.
    from ._capabilities import reset_capability_cache

    reset_capability_cache()

    first_error: BaseException | None = None
    for name, drv in (("write", write_driver), ("read", read_driver)):
        if drv is None:
            continue
        try:
            await drv.close()
        except Exception as exc:  # noqa: BLE001 - every driver gets its attempt
            logger.warning("Memgraph %s driver close failed: %s", name, exc)
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
