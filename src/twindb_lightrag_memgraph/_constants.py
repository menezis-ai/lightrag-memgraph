"""
Centralized constants and helpers shared across all Memgraph storage backends.

Single source of truth for default values, environment variable keys,
and workspace resolution logic.
"""

import os
import re

# Environment variable keys.
#
# Workspace resolution chain (`resolve_workspace()`):
#   1. ``MEMGRAPH_WORKSPACE`` — historical alias kept for back-compat
#      with deploys that set it explicitly alongside ``WORKSPACE``.
#   2. ``WORKSPACE`` — the canonical LightRAG-core variable. Setting
#      this single value is now enough for both LightRAG core *and*
#      our Memgraph storage backends.
#   3. ``TWIN_DEFAULT_SPACE`` — Twin overlay's source of truth; honoured
#      as a fallback so a "space-only" deploy boots without setting a
#      legacy variable.
#   4. ``DEFAULT_WORKSPACE`` ("base") — the LightRAG-internal default.
#
# Aligning on the chain lets new deploys ship a single ``WORKSPACE``
# or ``TWIN_DEFAULT_SPACE`` without the old "set both" footgun
# documented in ``deploy/ovh-twin/stack.yml``.
MEMGRAPH_WORKSPACE_ENV = "MEMGRAPH_WORKSPACE"
WORKSPACE_ENV = "WORKSPACE"
TWIN_DEFAULT_SPACE_ENV = "TWIN_DEFAULT_SPACE"

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
    """Resolve the active workspace (Memgraph label) from the
    environment.

    Falls through the alias chain in order, returning the first
    non-empty value:

      1. ``MEMGRAPH_WORKSPACE``
      2. ``WORKSPACE`` (LightRAG-core canonical)
      3. ``TWIN_DEFAULT_SPACE`` (Twin overlay)
      4. :data:`DEFAULT_WORKSPACE` (``"base"``)

    Raises:
        ValueError: If the resolved workspace contains unsafe
        characters.
    """
    for env_key in (MEMGRAPH_WORKSPACE_ENV, WORKSPACE_ENV, TWIN_DEFAULT_SPACE_ENV):
        candidate = os.environ.get(env_key, "").strip()
        if candidate:
            return validate_identifier(candidate, "workspace")
    return validate_identifier(DEFAULT_WORKSPACE, "workspace")
