"""Versioned storage identities and non-destructive legacy-startup checks."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import ClassVar

from neo4j.exceptions import ClientError, ResultFailedError, TransientError

from . import _pool
from ._constants import (
    STORAGE_LABEL_PREFIX,
    connection_fingerprint,
    validate_identifier,
)

REGISTRY_LABEL = STORAGE_LABEL_PREFIX + "REGISTRY"
KINDS = {"kv", "vector", "status", "graph"}


class StorageIdentityError(RuntimeError):
    """Refuse access when the physical storage's identity has not been verified."""


class LegacyStorageError(StorageIdentityError):
    """Legacy data/index ownership requires an explicit migration decision."""


def _physical(role: str, workspace: str, namespace: str) -> str:
    validate_identifier(workspace, "workspace")
    validate_identifier(namespace, "namespace")
    # Length prefixes keep literal underscores/markers inside components opaque.
    return f"{STORAGE_LABEL_PREFIX}{role}_w{len(workspace)}_{workspace}_n{len(namespace)}_{namespace}"


def storage_label(kind: str, workspace: str, namespace: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"Unknown storage kind: {kind!r}")
    return _physical(kind.upper(), workspace, namespace)


def storage_index_name(kind: str, workspace: str, namespace: str) -> str:
    if kind not in {"vector", "text"}:
        raise ValueError(f"Unknown index kind: {kind!r}")
    return _physical(kind.upper() + "_INDEX", workspace, namespace)


@dataclass(frozen=True)
class StorageIdentity:
    kind: str
    workspace: str
    namespace: str

    @property
    def label(self):
        return storage_label(self.kind, self.workspace, self.namespace)

    @property
    def record(self):
        return {
            "id": self.label,
            "format": 2,
            "kind": self.kind,
            "workspace": self.workspace,
            "namespace": self.namespace,
        }

    @property
    def legacy_label(self):
        if self.kind == "kv":
            return f"KV_{self.workspace}_{self.namespace}"
        if self.kind == "vector":
            return f"Vec_{self.workspace}_{self.namespace}"
        if self.kind == "status":
            return f"DocStatus_{self.workspace}"
        return self.workspace

    @property
    def legacy_indexes(self):
        if self.kind == "vector":
            return (
                f"vec_{self.workspace}_{self.namespace}",
                f"text_{self.workspace}_{self.namespace}",
            )
        return ()


def initialization_context(storage, kind):
    identity = StorageIdentity(kind, storage.workspace, storage.namespace)
    return identity.label, connection_fingerprint(), _pool._identity_epoch


def invalidate_identity(storage):
    storage._verified_storage_identity = None


def mark_identity_ready(storage, kind, context):
    if context != initialization_context(storage, kind):
        raise StorageIdentityError(
            "Storage configuration changed during initialization; recreate the storage instance"
        )
    storage._verified_storage_identity = context


def require_identity(storage, kind):
    expected = initialization_context(storage, kind)
    if getattr(storage, "_verified_storage_identity", None) != expected:
        raise StorageIdentityError(
            f"Storage {expected[0]} is not initialized for this configuration; call initialize() before accessing data"
        )


def guard_graph_method(method):
    @wraps(method)
    async def guarded(self, *args, **kwargs):
        require_identity(self, "graph")
        return await method(self, *args, **kwargs)

    guarded._lightrag_memgraph_identity_guarded = True
    return guarded


def _registered_record(props):
    try:
        identity = StorageIdentity(
            props["kind"], props["workspace"], props["namespace"]
        )
        valid = props == identity.record
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise StorageIdentityError(
            "Unrecognized data in the storage identity registry; inventory/restore it before upgrading"
        )
    return props


async def _registry_owners(session, wanted):
    # Stream validation instead of retaining every tenant's registry records.
    result = await session.run(
        f"MATCH (r:`{REGISTRY_LABEL}`) RETURN properties(r) AS props"
    )
    owners = {}
    try:
        async for row in result:
            props = _registered_record(dict(row["props"]))
            if props["id"] in wanted:
                if props["id"] in owners:
                    raise StorageIdentityError(
                        "Duplicate storage identity registry entries; restore the uniqueness constraint"
                    )
                owners[props["id"]] = props
    finally:
        await result.consume()
    return owners


async def _rows(session, query, **params):
    result = await session.run(query, **params)
    rows = await result.data()
    await result.consume()
    return [dict(row) for row in rows]


def _index_label(row):
    value = row.get("label", "")
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else ""
    return value.removeprefix(":") if isinstance(value, str) else ""


def _index_mentions(row, label, names=()):
    return (
        _index_label(row) == label
        or row.get("index_name") in names
        or any(f"(name: {name})" in row.get("index type", "") for name in names)
    )


async def _ddl(session, query):
    try:
        result = await session.run(query)
        await result.consume()
    except ClientError as exc:
        if "already exist" not in str(exc).lower():
            raise


async def initialize_identity(storage, kind, session):
    """Verify ownership before any application DDL, data reads or writes.

    No legacy data/index is changed. Re-ingest into an explicit fresh scope;
    ambiguous historical ownership cannot be inferred from a concatenated label.
    """
    identity = StorageIdentity(kind, storage.workspace, storage.namespace)
    legacy_sets = await _rows(
        session,
        f"MATCH (n:`{identity.legacy_label}`) RETURN DISTINCT labels(n) AS labels",
    )
    wanted = {identity.label, identity.legacy_label}
    wanted.update(
        label
        for row in legacy_sets
        for label in row["labels"]
        if label.startswith(STORAGE_LABEL_PREFIX)
    )
    owners = await _registry_owners(session, wanted)
    indexes = await _rows(session, "SHOW INDEX INFO")
    indexes += await _rows(session, "SHOW VECTOR INDEX INFO")
    for row in indexes:
        if _index_label(row) == REGISTRY_LABEL:
            prop = row.get("property")
            if prop not in (None, "", "id", ["id"], ("id",)):
                raise StorageIdentityError(
                    "An unclaimed index occupies the storage identity registry label; use a fresh database"
                )

    # A logical graph workspace may equal another v2 store's physical label.
    # Registered v2 nodes/type aliases are not evidence of legacy ownership.
    legacy_owned = (
        identity.legacy_label in owners or identity.legacy_label == REGISTRY_LABEL
    )
    if not legacy_owned:
        legacy_data = any(
            not any(
                label in owners or label == REGISTRY_LABEL for label in row["labels"]
            )
            for row in legacy_sets
        )
        legacy_index = any(
            _index_mentions(row, identity.legacy_label, identity.legacy_indexes)
            for row in indexes
        )
        if legacy_data or legacy_index:
            raise LegacyStorageError(
                f"Legacy storage {identity.legacy_label!r} has data or indexes with ambiguous ownership. Nothing was migrated; inventory and re-ingest into a fresh database/workspace before initializing {identity.label}"
            )

    if identity.label in owners:
        return
    occupied = await _rows(
        session, f"MATCH (n:`{identity.label}`) RETURN 1 AS present LIMIT 1"
    )
    target_names = (
        tuple(
            storage_index_name(name, identity.workspace, identity.namespace)
            for name in ("vector", "text")
        )
        if kind == "vector"
        else ()
    )
    if occupied or any(
        _index_mentions(row, identity.label, target_names) for row in indexes
    ):
        # A concurrent initializer may have committed between our read queries.
        if identity.label in await _registry_owners(session, {identity.label}):
            return
        raise StorageIdentityError(
            f"Unclaimed data or indexes occupy {identity.label}; refusing to adopt them"
        )

    for attempt in range(5):
        try:
            await _ddl(
                session,
                f"CREATE CONSTRAINT ON (r:`{REGISTRY_LABEL}`) ASSERT r.id IS UNIQUE",
            )
            await _ddl(session, f"CREATE INDEX ON :`{REGISTRY_LABEL}`(id)")
            rows = await _rows(
                session,
                f"MERGE (r:`{REGISTRY_LABEL}` {{id: $id}}) ON CREATE SET r += $props RETURN properties(r) AS props",
                id=identity.label,
                props=identity.record,
            )
            if (
                len(rows) != 1
                or _registered_record(dict(rows[0]["props"])) != identity.record
            ):
                raise StorageIdentityError(
                    "Storage identity registration did not return the expected owner"
                )
            return
        except (ClientError, TransientError, ResultFailedError) as exc:
            if identity.label in await _registry_owners(session, {identity.label}):
                return
            if attempt == 4 or not (
                isinstance(exc, TransientError)
                or "conflicting transactions" in str(exc).lower()
                or "TransientError" in str(exc)
            ):
                raise
            await asyncio.sleep(0.02 * 2**attempt)


class StorageIdentityMixin:
    _storage_kind: ClassVar[str]

    @asynccontextmanager
    async def _read_session(self):
        require_identity(self, self._storage_kind)
        async with _pool.get_read_session() as session:
            yield session

    @asynccontextmanager
    async def _write_session(self):
        require_identity(self, self._storage_kind)
        async with _pool.get_session() as session:
            yield session


@asynccontextmanager
async def initialization_session():
    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            yield session
