"""API key store backed by Memgraph (per-workspace).

One label per workspace: ``WebuiApiKey_{workspace}`` (mirrors the
``WebuiTag_{workspace}`` / ``WebuiTagCategory_{workspace}`` overlay
namespacing — distinct from the ``KV_`` prefix reserved for LightRAG's
native KV storage).

Each node carries:

- ``id`` (string, indexed) — opaque public identifier
- ``hash`` (string, indexed) — SHA-256 of the bearer; the direct lookup
  field used by :func:`validate_bearer` (O(1) average)
- ``data`` (JSON blob) — static metadata: name, prefix, hash,
  created_at, created_by, revoked_at
- ``last_used_at_ms`` (long, mutable) — separate property so the hot
  path can update without re-serialising the blob

The full key value is **never persisted** — only its SHA-256. The raw
value is returned exactly once, at creation. Subsequent reads expose only
the prefix. General operator keys use ``twk_``; metadata-profile credentials
use the distinct ``tcp_`` prefix so a ``profile:read`` key can never fall
through the generic Twin authentication chain.

The store is FastAPI-free so unit tests can drive it without spinning the
full app. ``require_auth`` calls into ``validate_bearer`` directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import threading
import time
from typing import Any

from .. import _pool
from .._constants import validate_identifier

logger = logging.getLogger(__name__)

# Index DDL is process/schema state, not request work.  Readiness is global
# because indexes survive event-loop turnover; locks are loop-local because an
# asyncio.Lock must not be carried into a replacement loop.  Cache only
# successful initialization and serialize the first attempt per loop so a cold
# burst cannot replay the same two CREATE INDEX round-trips.  The connection
# identity is part of the key: tests and long-lived development processes may
# retarget MEMGRAPH_URI / MEMGRAPH_DATABASE without restarting Python.
_SchemaKey = tuple[str, str, str]
_schema_ready: set[_SchemaKey] = set()
_schema_locks: dict[_SchemaKey, asyncio.Lock] = {}
_schema_locks_loop_id: int | None = None
_schema_state_lock = threading.Lock()

KEY_PREFIX = "twk_"
PROFILE_KEY_PREFIX = "tcp_"
KEY_SECRET_BYTES = 24  # → ~32 url-safe base64 chars
KEY_PREVIEW_LEN = 8


def _label(workspace: str) -> str:
    validate_identifier(workspace, "workspace")
    return f"WebuiApiKey_{workspace}"


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _mint_token(prefix: str = KEY_PREFIX) -> tuple[str, str]:
    """Return ``(full_value, preview)``. ``full_value`` is shown ONCE."""
    body = secrets.token_urlsafe(KEY_SECRET_BYTES)
    full = f"{prefix}{body}"
    preview = f"{prefix}{body[:KEY_PREVIEW_LEN]}…"
    return full, preview


def _new_id() -> str:
    """Ordered-ish public id: ``<ms_hex>-<random_hex>``."""
    return f"{_now_ms():x}-{secrets.token_hex(4)}"


def _public_entry(
    entry: dict[str, Any], *, last_used_at_ms: int | None
) -> dict[str, Any]:
    out = dict(entry)
    out.pop("hash", None)
    # Entries minted before scoped keys existed remain broad operator keys.
    out.setdefault("scopes", ["api:*"])
    out.setdefault("folders", [])
    out["last_used_at"] = last_used_at_ms
    return out


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------


def _schema_key(workspace: str) -> _SchemaKey:
    uri, database = _pool.connection_identity()
    return workspace, uri, database


def _get_schema_lock(key: _SchemaKey) -> asyncio.Lock:
    """Return this event loop's single-flight lock without retaining the loop."""
    global _schema_locks_loop_id

    loop_id = id(asyncio.get_running_loop())
    with _schema_state_lock:
        if _schema_locks_loop_id != loop_id:
            _schema_locks.clear()
            _schema_locks_loop_id = loop_id
        return _schema_locks.setdefault(key, asyncio.Lock())


async def _initialize_schema(label: str) -> None:
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        for field in ("id", "hash"):
            try:
                result = await session.run(f"CREATE INDEX ON :`{label}`({field})")
                await result.consume()
                logger.info("[ApiKeyStore] Index on :%s(%s) ensured", label, field)
            except Exception as exc:  # noqa: BLE001 — narrow check below
                if "already exists" in str(exc).lower():
                    continue
                raise


async def initialize(workspace: str) -> None:
    """Ensure the workspace ``id`` + ``hash`` indexes once per process.

    Successful schema setup is cached for the active Memgraph endpoint and
    database.  Failures are deliberately not cached, so the next request can
    recover after a transient database outage.  Concurrent cold requests on
    one event loop share a lock and issue one pair of DDL statements.
    """
    label = _label(workspace)
    key = _schema_key(workspace)
    with _schema_state_lock:
        if key in _schema_ready:
            return

    # _get_schema_lock takes the non-reentrant state lock itself.
    lock = _get_schema_lock(key)

    async with lock:
        with _schema_state_lock:
            if key in _schema_ready:
                return
        await _initialize_schema(label)
        with _schema_state_lock:
            _schema_ready.add(key)
            _schema_locks.pop(key, None)


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------


async def list_keys(workspace: str) -> list[dict[str, Any]]:
    """Return every key for a workspace (revoked included). The hash is
    stripped; ``last_used_at`` is exposed."""
    label = _label(workspace)
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`{label}`) "
            f"RETURN n.id AS id, n.data AS data, n.last_used_at_ms AS last_used "
            f"ORDER BY n.`__created_at` DESC, n.id"
        )
        rows = await result.data()
        await result.consume()
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("data")
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "[ApiKeyStore] Skipping non-JSON node on :%s id=%s",
                label,
                row.get("id"),
            )
            continue
        out.append(_public_entry(entry, last_used_at_ms=row.get("last_used")))
    return out


async def get_key(workspace: str, key_id: str) -> dict[str, Any] | None:
    """Lookup a single key by id. Returns the public shape or ``None``."""
    label = _label(workspace)
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`{label}` {{id: $id}}) "
            f"RETURN n.data AS data, n.last_used_at_ms AS last_used",
            id=key_id,
        )
        record = await result.single()
        await result.consume()
    if not record or not record.get("data"):
        return None
    try:
        entry = json.loads(record["data"])
    except json.JSONDecodeError:
        return None
    return _public_entry(entry, last_used_at_ms=record.get("last_used"))


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


async def create_key(
    workspace: str,
    *,
    name: str,
    created_by: str,
    scopes: list[str] | None = None,
    folders: list[str] | None = None,
) -> dict[str, Any]:
    """Mint a new key. Returns the entry **with ``full_value``** — the only
    moment that value is ever exposed by this module."""
    label = _label(workspace)
    key_id = _new_id()
    effective_scopes = list(scopes or ["api:*"])
    effective_folders = list(folders or [])
    prefix = PROFILE_KEY_PREFIX if effective_scopes == ["profile:read"] else KEY_PREFIX
    full_value, preview = _mint_token(prefix)
    hashed = _hash_token(full_value)
    entry = {
        "id": key_id,
        "name": (name.strip()[:120] or "Unnamed key"),
        "prefix": preview,
        "hash": hashed,
        "created_at": _now_ms(),
        "created_by": created_by or "system",
        "scopes": effective_scopes,
        "folders": effective_folders,
        "revoked_at": None,
    }
    data = json.dumps(entry, sort_keys=True)
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        # ``hash`` is duplicated as a direct property so the indexed
        # lookup in :func:`validate_bearer` is O(1). Source of truth
        # stays the JSON blob (``data``) — the direct property is a
        # query optimisation, never read for content.
        result = await session.run(
            f"""
                CREATE (n:`{label}` {{id: $id}})
                SET n.data = $data,
                    n.hash = $hash,
                    n.`__created_at` = timestamp()
                """,
            id=key_id,
            data=data,
            hash=hashed,
        )
        await result.consume()
    public = _public_entry(entry, last_used_at_ms=None)
    public["full_value"] = full_value
    return public


async def revoke_key(workspace: str, key_id: str) -> dict[str, Any] | None:
    """Mark a key revoked. Returns the public entry on success, ``None``
    if the key did not exist. Re-revoking an already-revoked key returns
    its entry without mutating the timestamp."""
    label = _label(workspace)
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        res = await session.run(
            f"MATCH (n:`{label}` {{id: $id}}) "
            f"RETURN n.data AS data, n.last_used_at_ms AS last_used",
            id=key_id,
        )
        row = await res.single()
        await res.consume()
        if not row or not row.get("data"):
            return None
        try:
            entry = json.loads(row["data"])
        except json.JSONDecodeError:
            return None
        if entry.get("revoked_at"):
            return _public_entry(entry, last_used_at_ms=row.get("last_used"))
        entry["revoked_at"] = _now_ms()
        new_data = json.dumps(entry, sort_keys=True)
        res = await session.run(
            f"MATCH (n:`{label}` {{id: $id}}) "
            f"SET n.data = $data, n.`__updated_at` = timestamp()",
            id=key_id,
            data=new_data,
        )
        await res.consume()
    return _public_entry(entry, last_used_at_ms=row.get("last_used"))


# ---------------------------------------------------------------------------
# Auth-chain integration
# ---------------------------------------------------------------------------


async def validate_bearer(workspace: str, token: str) -> dict[str, Any] | None:
    """Look up a bearer against the workspace store. Returns the public
    key entry on match (and not revoked), ``None`` otherwise.

    SHA-256 hashing of the bearer is constant-time wrt the input. The
    Memgraph lookup is an indexed equality on the ``hash`` property, so
    the round-trip is O(1) regardless of how many keys live in the
    workspace. The DB-side equality test on a 64-char hex string does
    not leak useful timing info — both sides are deterministic SHA-256
    digests, not the raw secret.

    Callers MAY ``await mark_used(workspace, entry['id'])`` after a
    positive match — it is intentionally not awaited here to keep the
    auth hot path side-effect-free.
    """
    if not token or not token.startswith((KEY_PREFIX, PROFILE_KEY_PREFIX)):
        return None
    hashed = _hash_token(token)
    label = _label(workspace)
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`{label}` {{hash: $h}}) "
            f"RETURN n.data AS data, n.last_used_at_ms AS last_used "
            f"LIMIT 1",
            h=hashed,
        )
        row = await result.single()
        await result.consume()
    if not row or not row.get("data"):
        return None
    try:
        entry = json.loads(row["data"])
    except json.JSONDecodeError:
        return None
    if entry.get("revoked_at"):
        return None
    return _public_entry(entry, last_used_at_ms=row.get("last_used"))


async def mark_used(workspace: str, key_id: str) -> None:
    """Bump ``last_used_at_ms`` on a key. Fire-and-forget — errors are
    logged and swallowed so the auth path is never affected."""
    label = _label(workspace)
    now = _now_ms()
    try:
        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            res = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) SET n.last_used_at_ms = $now",
                id=key_id,
                now=now,
            )
            await res.consume()
    except Exception:
        logger.exception(
            "[ApiKeyStore] mark_used failed for key %s in workspace %s",
            key_id,
            workspace,
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def reset_workspace(workspace: str) -> None:
    """Test-only: wipe every node under :ApiKey_{workspace}."""
    label = _label(workspace)
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        res = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
        await res.consume()


__all__ = [
    "KEY_PREFIX",
    "PROFILE_KEY_PREFIX",
    "create_key",
    "get_key",
    "initialize",
    "list_keys",
    "mark_used",
    "reset_workspace",
    "revoke_key",
    "validate_bearer",
]
