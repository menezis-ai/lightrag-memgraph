"""
Centralized constants and helpers shared across all Memgraph storage backends.

Single source of truth for default values, environment variable keys,
and workspace resolution logic.
"""

import os
import re

# Environment variable keys
MEMGRAPH_WORKSPACE_ENV = "MEMGRAPH_WORKSPACE"

# Default values
DEFAULT_WORKSPACE = "base"
DEFAULT_MEMGRAPH_URI = "bolt://localhost:7687"
CONNECTION_POOL_SIZE = 50
VECTOR_INDEX_CAPACITY = 100_000

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

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_]+$")


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
    """Resolve the active workspace from the environment.

    Reads ``MEMGRAPH_WORKSPACE`` env var, strips whitespace,
    and falls back to :data:`DEFAULT_WORKSPACE` when empty or unset.

    Raises:
        ValueError: If the workspace name contains unsafe characters.
    """
    ws = os.environ.get(MEMGRAPH_WORKSPACE_ENV, "").strip()
    ws = ws if ws else DEFAULT_WORKSPACE
    return validate_identifier(ws, "workspace")
