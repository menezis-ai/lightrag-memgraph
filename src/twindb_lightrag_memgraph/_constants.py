"""
Centralized constants and helpers shared across all Memgraph storage backends.

Single source of truth for default values, environment variable keys,
and workspace resolution logic.

## Naming: ``workspace`` here ≠ Twin folder

The token ``workspace`` in this module refers strictly to the LightRAG-core
notion: the backtick-safe identifier that becomes the Memgraph node label
(`KV_{workspace}`, `Vec_{workspace}`, `DocStatus_{workspace}`, ...). That
contract is defined upstream by LightRAG and we don't get to rename it.

The user-facing Twin sub-scope is called **folder** everywhere else in this
codebase. The two concepts overlap today because the deploy maps a single
LightRAG workspace per Twin instance, but the wording in this module
deliberately stays "workspace" so anyone touching the storage backends
recognises it as the LightRAG-aligned label and not the Twin overlay surface.
"""

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Environment variable keys.
#
# Workspace resolution chain (`resolve_workspace()`):
#   1. ``MEMGRAPH_WORKSPACE`` — historical alias kept for back-compat
#      with deploys that set it explicitly alongside ``WORKSPACE``.
#   2. ``WORKSPACE`` — the canonical LightRAG-core variable. Setting
#      this single value is now enough for both LightRAG core *and*
#      our Memgraph storage backends.
#   3. ``TWIN_DEFAULT_FOLDER`` — Twin overlay's source of truth; honoured
#      as a fallback so a "folder-only" deploy boots without setting a
#      legacy variable.
#   4. ``DEFAULT_WORKSPACE`` ("base") — the LightRAG-internal default.
#
# Aligning on the chain lets new deploys ship a single ``WORKSPACE``
# or ``TWIN_DEFAULT_FOLDER`` without the old "set both" footgun
# of earlier deployment templates.
MEMGRAPH_WORKSPACE_ENV = "MEMGRAPH_WORKSPACE"
WORKSPACE_ENV = "WORKSPACE"
TWIN_DEFAULT_FOLDER_ENV = "TWIN_DEFAULT_FOLDER"
TWIN_DEFAULT_FOLDER_LABEL_ENV = "TWIN_DEFAULT_FOLDER_LABEL"
TWIN_MAX_FOLDERS_ENV = "TWIN_MAX_FOLDERS"

# Default values
DEFAULT_WORKSPACE = "base"
DEFAULT_MEMGRAPH_URI = "bolt://localhost:7687"
CONNECTION_POOL_SIZE = 50
VECTOR_INDEX_CAPACITY = 100_000
DEFAULT_PAGE_SIZE = 50
DEFAULT_TWIN_MAX_FOLDERS = 5

# Write throttle — limits concurrent write operations (upsert, delete, drop)
# to avoid saturating the Bolt pool during bulk uploads.
MEMGRAPH_WRITE_CONCURRENCY_ENV = "MEMGRAPH_WRITE_CONCURRENCY"
DEFAULT_WRITE_CONCURRENCY = 8

# Pool Bolt tuning — configurable pool size and connection acquire timeout.
MEMGRAPH_POOL_SIZE_ENV = "MEMGRAPH_POOL_SIZE"
MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT_ENV = "MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT"
DEFAULT_CONNECTION_ACQUIRE_TIMEOUT = 5.0  # seconds — fail fast, don't hang

# Read pool — dedicated connection pool for read operations.
MEMGRAPH_READ_POOL_SIZE_ENV = "MEMGRAPH_READ_POOL_SIZE"
DEFAULT_READ_POOL_SIZE = 20

_SAFE_IDENTIFIER_RE = re.compile(r"^\w+$", re.ASCII)
_active_storage_folder: ContextVar[str | None] = ContextVar(
    "twin_active_storage_folder",
    default=None,
)
_active_operator_classification: ContextVar[str | None] = ContextVar(
    "twin_active_operator_classification",
    default=None,
)


def validate_identifier(value: str, name: str = "identifier") -> str:
    """Validate that a Cypher identifier contains only safe characters.

    Prevents Cypher injection via label names, database names, and
    relationship types that cannot use ``$param`` parameterization.

    Raises:
        ValueError: If *value* contains characters outside ``[a-zA-Z0-9_]``.
    """
    if not value or not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Invalid {name}: must be non-empty and contain only "
            f"alphanumeric characters or underscores, got {value!r}"
        )
    return value


def resolve_workspace() -> str:
    """Resolve the active workspace (Memgraph label) from the
    environment.

    Falls through the alias chain in order, returning the first
    non-empty value:

      1. ``MEMGRAPH_WORKSPACE``
      2. ``WORKSPACE`` (LightRAG-core canonical)
      3. ``TWIN_DEFAULT_FOLDER`` (Twin overlay)
      4. :data:`DEFAULT_WORKSPACE` (``"base"``)

    Raises:
        ValueError: If the resolved workspace contains unsafe
        characters.
    """
    for env_key in (
        MEMGRAPH_WORKSPACE_ENV,
        WORKSPACE_ENV,
        TWIN_DEFAULT_FOLDER_ENV,
    ):
        candidate = os.environ.get(env_key, "").strip()
        if candidate:
            return validate_identifier(candidate, "workspace")
    return validate_identifier(DEFAULT_WORKSPACE, "workspace")


def default_twin_folder() -> str:
    """Return the fallback Twin folder id used when no request scope exists."""
    candidate = (
        os.environ.get(TWIN_DEFAULT_FOLDER_ENV)
        or os.environ.get(WORKSPACE_ENV)
        or "default"
    ).strip()
    try:
        return validate_identifier(candidate, "folder")
    except ValueError:
        return "default"


def get_active_storage_folder() -> str | None:
    """Folder captured for storage writes in the current async context."""
    return _active_storage_folder.get()


@contextmanager
def storage_folder_context(folder: str | None) -> Iterator[None]:
    """Temporarily bind a Twin folder for low-level storage writes.

    This lives in the storage constants module, not in ``server.folder``, so
    Memgraph storage backends can read it without importing FastAPI/server
    code. Callers must pass an already validated folder id.
    """
    token = _active_storage_folder.set(
        validate_identifier(folder, "folder") if folder else None
    )
    try:
        yield
    finally:
        _active_storage_folder.reset(token)


def get_active_operator_classification() -> str | None:
    """Operator-selected MIP class captured for the current ingestion context.

    Set from the ``X-Twin-Classification`` upload header. ``None`` means the
    operator made no explicit choice ("no MIP"), so auto-detection alone
    decides. The combination policy (embedded label is a floor — operator can
    raise, never downgrade) lives in
    :func:`classification.apply_operator_classification`.
    """
    return _active_operator_classification.get()


@contextmanager
def operator_classification_context(class_id: str | None) -> Iterator[None]:
    """Temporarily bind an operator-selected MIP classification for ingestion.

    Lives in the storage constants module (not ``server``) so the
    classification hook can read it without importing FastAPI. The raw value is
    validated for safe characters here; ladder membership and the floor policy
    are resolved later in ``classification.apply_operator_classification``.
    An unsafe/garbage value is dropped (treated as "no operator choice").
    """
    cleaned: str | None = None
    if class_id:
        try:
            cleaned = validate_identifier(class_id, "classification")
        except ValueError:
            cleaned = None
    token = _active_operator_classification.set(cleaned)
    try:
        yield
    finally:
        _active_operator_classification.reset(token)
