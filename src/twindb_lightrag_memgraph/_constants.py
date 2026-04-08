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

# Vector quantization — Memgraph 3.8+ supports scalar_kind in vector indexes.
# "f32" (default, full precision), "f16" (50% memory savings), "i8" (75% savings).
MEMGRAPH_VECTOR_SCALAR_KIND_ENV = "MEMGRAPH_VECTOR_SCALAR_KIND"
DEFAULT_VECTOR_SCALAR_KIND = "f16"
_VALID_SCALAR_KINDS = frozenset({"f32", "f16", "i8"})

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

# Batched delete — prevents OOM on large datasets by limiting transaction size.
MEMGRAPH_DELETE_BATCH_SIZE_ENV = "MEMGRAPH_DELETE_BATCH_SIZE"
DEFAULT_DELETE_BATCH_SIZE = 10_000

# Memory budget — LightRAG-specific cap on how much Memgraph memory it may use.
# Compared against memory_tracked (= graph_memory_tracked + vector_index_memory_tracked).
# 0 = unlimited (budget check skipped even if BUDGET_ENFORCE is warn/reject).
# Must be set explicitly when using BUDGET_ENFORCE, e.g. "2GiB".
MEMGRAPH_MEMORY_LIMIT_ENV = "MEMGRAPH_MEMORY_LIMIT"
DEFAULT_MEMORY_LIMIT_GIB = 0

# Budget enforcement mode for pre-insert memory check.
# Modes: off (default, no check), warn (log but proceed), reject (raise error).
MEMGRAPH_BUDGET_ENFORCE_ENV = "MEMGRAPH_BUDGET_ENFORCE"

# TTL — Memgraph Enterprise native TTL support.
# When set, KV upserts add a :TTL label and ttl property (Unix epoch expiry).
# Memgraph's background job deletes expired nodes automatically.
MEMGRAPH_TTL_SECONDS_ENV = "MEMGRAPH_TTL_SECONDS"
# Comma-separated list of KV namespace names that get TTL labels.
MEMGRAPH_TTL_LABELS_ENV = "MEMGRAPH_TTL_LABELS"
DEFAULT_TTL_LABELS = "full_docs,text_chunks"

# Retry — exponential backoff for Memgraph MVCC TransientError (GQL 50N42).
MEMGRAPH_RETRY_MAX_ATTEMPTS_ENV = "MEMGRAPH_RETRY_MAX_ATTEMPTS"
DEFAULT_RETRY_MAX_ATTEMPTS = 6
MEMGRAPH_RETRY_BASE_DELAY_MS_ENV = "MEMGRAPH_RETRY_BASE_DELAY_MS"
DEFAULT_RETRY_BASE_DELAY_MS = 50

# Replica-aware retry — longer budget for SYNC replica lag.
MEMGRAPH_REPLICA_RETRIES_ENV = "MEMGRAPH_REPLICA_RETRIES"
DEFAULT_REPLICA_RETRIES = 20
MEMGRAPH_REPLICA_RETRY_DELAY_MS_ENV = "MEMGRAPH_REPLICA_RETRY_DELAY_MS"
DEFAULT_REPLICA_RETRY_DELAY_MS = 2000

# Lazy full_docs — purge raw content after processing, reconstruct on demand.
MEMGRAPH_PURGE_FULL_DOCS_ENV = "MEMGRAPH_PURGE_FULL_DOCS"
# Values: "off" (default), "on" — purge full_docs content after PROCESSED status

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
