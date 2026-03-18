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
from urllib.parse import urlparse

from neo4j import AsyncGraphDatabase, TrustAll, TrustSystemCAs
from neo4j.exceptions import ClientError as Neo4jClientError

from ._constants import (
    CONNECTION_POOL_SIZE,
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
    DEFAULT_MEMGRAPH_URI,
    DEFAULT_READ_POOL_SIZE,
    DEFAULT_WRITE_CONCURRENCY,
    MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT_ENV,
    MEMGRAPH_POOL_SIZE_ENV,
    MEMGRAPH_READ_POOL_SIZE_ENV,
    MEMGRAPH_WRITE_CONCURRENCY_ENV,
    validate_identifier,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

_thread_lock = threading.Lock()
_driver = None
_database = None
_bound_loop_id = None

_write_semaphore = None
_semaphore_loop_id = None

_read_driver = None
_read_database = None
_read_bound_loop_id = None

# Enterprise multi-database detection.
# None = not yet probed, True = USE DATABASE succeeded, False = Community edition.
_enterprise_supported: bool | None = None


def _read_pool_size() -> int:
    """Read MEMGRAPH_POOL_SIZE from env, default CONNECTION_POOL_SIZE (50)."""
    raw = os.environ.get(MEMGRAPH_POOL_SIZE_ENV, "")
    try:
        val = int(raw)
        if val < 1:
            raise ValueError("must be >= 1")
        return val
    except (ValueError, TypeError):
        return CONNECTION_POOL_SIZE


def _read_connection_acquire_timeout() -> float:
    """Read MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT from env, default 5.0s."""
    raw = os.environ.get(MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT_ENV, "")
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError("must be > 0")
        return val
    except (ValueError, TypeError):
        return DEFAULT_CONNECTION_ACQUIRE_TIMEOUT


def _read_read_pool_size() -> int:
    """Read MEMGRAPH_READ_POOL_SIZE from env, default 20."""
    raw = os.environ.get(MEMGRAPH_READ_POOL_SIZE_ENV, "")
    try:
        val = int(raw)
        if val < 1:
            raise ValueError("must be >= 1")
        return val
    except (ValueError, TypeError):
        return DEFAULT_READ_POOL_SIZE


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
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")
    database = os.environ.get("MEMGRAPH_DATABASE", "memgraph")
    validate_identifier(database, "database")

    _parsed_uri = urlparse(uri)
    _localhost = {"localhost", "127.0.0.1", "::1"}
    if (
        _parsed_uri.scheme in ("bolt", "neo4j")
        and _parsed_uri.hostname
        and _parsed_uri.hostname not in _localhost
    ):
        logger.warning(
            "Plaintext Bolt connection to remote host %s — credentials "
            "will be sent unencrypted. Use bolt+s:// or set "
            "MEMGRAPH_ENCRYPTED=true for TLS.",
            _parsed_uri.hostname,
        )

    encrypted_env = os.environ.get("MEMGRAPH_ENCRYPTED", "").lower()

    driver_kwargs = {
        "auth": (username, password),
        "max_connection_pool_size": (
            pool_size_override if pool_size_override is not None else _read_pool_size()
        ),
        "connection_acquisition_timeout": _read_connection_acquire_timeout(),
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
        else:
            driver_kwargs["trusted_certificates"] = TrustSystemCAs()
        logger.info("Memgraph TLS enabled (trust=%s)", trust_env)
    elif encrypted_env == "false":
        driver_kwargs["encrypted"] = False

    return uri, database, driver_kwargs


async def get_driver():
    """Get or create the shared AsyncGraphDatabase driver.

    If the event loop has changed since the driver was created,
    the old driver is closed and a new one is created.

    Returns:
        tuple: (driver, database_name)
    """
    global _driver, _database, _bound_loop_id

    current_loop_id = id(asyncio.get_running_loop())

    # Fast path: driver exists and is bound to the current loop
    if _driver is not None and _bound_loop_id == current_loop_id:
        return _driver, _database

    with _thread_lock:
        # Double-check after acquiring lock
        if _driver is not None and _bound_loop_id == current_loop_id:
            return _driver, _database

        # Close stale driver from a previous event loop
        if _driver is not None:
            try:
                await _driver.close()
            except Exception as e:
                logger.debug("Error closing stale driver: %s", e)
            _driver = None

        uri, _database, driver_kwargs = _read_connection_config()
        _driver = AsyncGraphDatabase.driver(uri, **driver_kwargs)
        _bound_loop_id = current_loop_id
        _parsed = urlparse(uri)
        _safe_uri = f"{_parsed.scheme}://{_parsed.hostname}:{_parsed.port}"
        logger.info(
            "Memgraph Bolt driver created — uri=%s database=%s",
            _safe_uri,
            _database,
        )
        return _driver, _database


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
      fails — we detect this once and skip it for all subsequent sessions.
    """
    driver, database = await get_driver()
    if _uses_routing_protocol():
        async with driver.session(database=database) as session:
            yield session
    else:
        async with driver.session() as session:
            if database:
                await _try_use_database(session, database)
            yield session


async def _get_read_driver():
    """Get or create the shared read-only AsyncGraphDatabase driver.

    Separate pool from the write driver, isolating reads from write pressure.
    If the event loop has changed, the old driver is closed and recreated.

    Returns:
        tuple: (driver, database_name)
    """
    global _read_driver, _read_database, _read_bound_loop_id

    current_loop_id = id(asyncio.get_running_loop())

    if _read_driver is not None and _read_bound_loop_id == current_loop_id:
        return _read_driver, _read_database

    with _thread_lock:
        if _read_driver is not None and _read_bound_loop_id == current_loop_id:
            return _read_driver, _read_database

        if _read_driver is not None:
            try:
                await _read_driver.close()
            except Exception as e:
                logger.debug("Error closing stale read driver: %s", e)
            _read_driver = None

        read_pool_size = _read_read_pool_size()
        uri, _read_database, driver_kwargs = _read_connection_config(
            pool_size_override=read_pool_size,
        )
        _read_driver = AsyncGraphDatabase.driver(uri, **driver_kwargs)
        _read_bound_loop_id = current_loop_id
        _parsed = urlparse(uri)
        _safe_uri = f"{_parsed.scheme}://{_parsed.hostname}:{_parsed.port}"
        logger.info(
            "Memgraph READ Bolt driver created — uri=%s database=%s pool_size=%d",
            _safe_uri,
            _read_database,
            read_pool_size,
        )
        return _read_driver, _read_database


@asynccontextmanager
async def get_read_session():
    """Get a read-only session from the dedicated read pool.

    Uses the same database routing logic as ``get_session()`` but draws
    connections from a separate pool, isolating reads from write pressure.
    """
    driver, database = await _get_read_driver()
    if _uses_routing_protocol():
        async with driver.session(database=database) as session:
            yield session
    else:
        async with driver.session() as session:
            if database:
                await _try_use_database(session, database)
            yield session


async def _try_use_database(session, database: str) -> None:
    """Issue ``USE DATABASE`` if the server supports it (Enterprise).

    On the first call the result is probed: if the command succeeds the
    ``_enterprise_supported`` flag is set to ``True`` and all subsequent
    calls go through immediately.  If the server returns the Enterprise
    license error, the flag is set to ``False`` and all subsequent calls
    are silently skipped.

    When *database* is ``"memgraph"`` (the Community default), the command
    is skipped entirely — it is the default database and ``USE DATABASE``
    is an Enterprise-only feature on Community edition.
    """
    global _enterprise_supported

    # "memgraph" is the default database on Community — no need to switch.
    if database == "memgraph":
        return

    if _enterprise_supported is False:
        return  # Community edition — skip

    try:
        await session.run(f"USE DATABASE {database}")
        if _enterprise_supported is None:
            _enterprise_supported = True
            logger.debug(
                "USE DATABASE %s succeeded — Enterprise multi-database enabled",
                database,
            )
    except Neo4jClientError as exc:
        if "enterprise" in str(exc).lower() or "license" in str(exc).lower():
            _enterprise_supported = False
            logger.info(
                "Memgraph Community detected — USE DATABASE not available, "
                "using default database"
            )
        else:
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
    """Read MEMGRAPH_WRITE_CONCURRENCY from env, default 10."""
    raw = os.environ.get(MEMGRAPH_WRITE_CONCURRENCY_ENV, "")
    try:
        val = int(raw)
        if val < 1:
            raise ValueError("must be >= 1")
        return val
    except (ValueError, TypeError):
        return DEFAULT_WRITE_CONCURRENCY


async def _get_write_semaphore() -> asyncio.Semaphore:
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


@asynccontextmanager
async def acquire_write_slot():
    """Gate write operations to ``MEMGRAPH_WRITE_CONCURRENCY`` slots.

    Only write operations (upsert, delete, drop) should use this.
    Read operations must NOT acquire this so they remain unthrottled.
    """
    sem = await _get_write_semaphore()
    async with sem:
        yield


async def close_driver():
    """Close write and read drivers. Call on application shutdown."""
    global _driver, _bound_loop_id, _write_semaphore, _semaphore_loop_id
    global _read_driver, _read_database, _read_bound_loop_id
    global _enterprise_supported
    if _driver is not None:
        await _driver.close()
        _driver = None
        _bound_loop_id = None
    _write_semaphore = None
    _semaphore_loop_id = None
    _enterprise_supported = None
    if _read_driver is not None:
        await _read_driver.close()
        _read_driver = None
        _read_database = None
        _read_bound_loop_id = None
