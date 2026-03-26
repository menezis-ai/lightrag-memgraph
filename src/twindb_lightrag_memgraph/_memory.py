"""
Memgraph memory introspection and pre-insert size estimation.

Provides:
- ``get_memory_usage()`` -- query Memgraph's live memory consumption via Bolt
- ``get_storage_info()`` -- full SHOW STORAGE INFO as a dict
- ``estimate_insert_cost()`` -- predict memory cost of a document BEFORE ingestion
- ``check_memory_budget()`` -- compare estimated cost against remaining capacity

Works on both Memgraph Community and Enterprise editions.

Usage::

    from twindb_lightrag_memgraph._memory import (
        get_memory_usage,
        estimate_insert_cost,
        check_memory_budget,
    )

    info = await get_memory_usage()
    print(f"Memory used: {info.used_bytes / 1024**2:.1f} MiB")

    cost = estimate_insert_cost("Some long document text...", embedding_dim=1536)
    print(f"Estimated cost: {cost.total_bytes / 1024**2:.1f} MiB")

    ok = await check_memory_budget(
        text="Some long document text...",
        embedding_dim=1536,
        memory_limit_bytes=75 * 1024**3,  # 75 GiB license
    )
    print(f"Fits in budget: {ok.fits}")
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from . import _pool
from ._constants import DEFAULT_MEMORY_LIMIT_GIB, MEMGRAPH_MEMORY_LIMIT_ENV

logger = logging.getLogger("twindb_lightrag_memgraph")


class MemoryBudgetExceeded(Exception):
    """Raised when a document insert would exceed the configured memory budget.

    Attributes:
        budget_check: The :class:`BudgetCheck` that triggered the rejection.
    """

    def __init__(self, budget_check: "BudgetCheck"):
        self.budget_check = budget_check
        super().__init__(budget_check.human_readable())


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------


@dataclass
class MemoryUsage:
    """Current memory usage reported by Memgraph.

    The billing metric for Memgraph Enterprise is ``memory_tracked``
    (= ``graph_memory_tracked`` + ``vector_index_memory_tracked``).
    On v3.9+ these are reported separately; on older versions we
    fall back to ``memory_usage`` which is a coarser estimate.

    The memory limit applies to the entire Memgraph instance, not
    per-database.  In multi-tenant mode all databases share the same
    pool — enforcement must happen on the LightRAG side.
    """

    used_bytes: int
    """Billable memory: ``memory_tracked`` when available (v3.9+),
    otherwise ``memory_usage`` from SHOW STORAGE INFO."""

    graph_memory_bytes: int | None = None
    """Graph structures memory (v3.9+).  None on older versions."""

    vector_index_memory_bytes: int | None = None
    """Vector index memory (v3.9+).  None on older versions."""

    vertex_count: int = 0
    """Number of vertices in the database."""

    edge_count: int = 0
    """Number of edges in the database."""

    memory_res_bytes: int | None = None
    """Resident set size (RSS) of the Memgraph process, if reported."""

    raw: dict[str, str] = field(default_factory=dict)
    """All key-value pairs from SHOW STORAGE INFO, as raw strings."""


@dataclass
class InsertCostEstimate:
    """Predicted memory cost of ingesting a document into LightRAG + Memgraph."""

    full_doc_bytes: int
    """Size of the full_document KV entry (raw text + JSON overhead)."""

    text_chunks_bytes: int
    """Size of all text_chunk KV entries (roughly same as raw text)."""

    embedding_bytes: int
    """Size of all vector embeddings (num_chunks * dim * 4 bytes per float32)."""

    graph_overhead_bytes: int
    """Estimated graph node/edge overhead (~20% of text size)."""

    num_chunks_estimate: int
    """Estimated number of chunks (text_len / chunk_size)."""

    @property
    def total_bytes(self) -> int:
        """Total estimated memory cost in bytes."""
        return (
            self.full_doc_bytes
            + self.text_chunks_bytes
            + self.embedding_bytes
            + self.graph_overhead_bytes
        )

    def human_readable(self) -> str:
        """Format the estimate as a human-readable breakdown."""

        def _fmt(b: int) -> str:
            if b >= 1024**3:
                return f"{b / 1024**3:.2f} GiB"
            if b >= 1024**2:
                return f"{b / 1024**2:.2f} MiB"
            if b >= 1024:
                return f"{b / 1024:.1f} KiB"
            return f"{b} B"

        lines = [
            f"Insert cost estimate (~{self.num_chunks_estimate} chunks):",
            f"  full_doc KV:      {_fmt(self.full_doc_bytes)}",
            f"  text_chunks KV:   {_fmt(self.text_chunks_bytes)}",
            f"  embeddings:       {_fmt(self.embedding_bytes)}",
            f"  graph overhead:   {_fmt(self.graph_overhead_bytes)}",
            f"  TOTAL:            {_fmt(self.total_bytes)}",
        ]
        return "\n".join(lines)


@dataclass
class BudgetCheck:
    """Result of a pre-insert budget check."""

    fits: bool
    """True if the estimated cost fits within the remaining budget."""

    used_bytes: int
    """Current memory used by Memgraph."""

    limit_bytes: int
    """Memory limit (license or configured)."""

    estimated_cost_bytes: int
    """Estimated cost of the pending insert."""

    remaining_bytes: int
    """Remaining budget after the hypothetical insert."""

    headroom_ratio: float
    """Fraction of total capacity remaining after the insert (0.0 to 1.0)."""

    def human_readable(self) -> str:
        """Format the budget check as a human-readable summary."""

        def _fmt(b: int) -> str:
            if b >= 1024**3:
                return f"{b / 1024**3:.2f} GiB"
            if b >= 1024**2:
                return f"{b / 1024**2:.2f} MiB"
            return f"{b / 1024:.1f} KiB"

        status = "OK" if self.fits else "WOULD EXCEED BUDGET"
        lines = [
            f"Budget check: {status}",
            f"  Current usage:    {_fmt(self.used_bytes)}",
            f"  Limit:            {_fmt(self.limit_bytes)}",
            f"  Insert cost:      {_fmt(self.estimated_cost_bytes)}",
            f"  After insert:     {_fmt(self.used_bytes + self.estimated_cost_bytes)}",
            f"  Remaining:        {_fmt(self.remaining_bytes)}",
            f"  Headroom:         {self.headroom_ratio:.1%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Size parsing: Memgraph returns human-readable strings like "42MiB"
# ---------------------------------------------------------------------------

_SIZE_PATTERN = re.compile(
    r"^\s*([\d.]+)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB|bytes?)\s*$",
    re.IGNORECASE,
)

_SIZE_MULTIPLIERS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kib": 1024,
    "kb": 1000,
    "mib": 1024**2,
    "mb": 1000**2,
    "gib": 1024**3,
    "gb": 1000**3,
    "tib": 1024**4,
    "tb": 1000**4,
}


def _parse_size(value: str) -> int | None:
    """Parse a human-readable size string to bytes.

    Handles Memgraph's output format: ``"42MiB"``, ``"1.5GiB"``, ``"0B"``.
    Returns None if the string cannot be parsed.
    """
    m = _SIZE_PATTERN.match(value)
    if not m:
        # Try parsing as a plain integer (some Memgraph versions return raw bytes)
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    number = float(m.group(1))
    unit = m.group(2).lower()
    multiplier = _SIZE_MULTIPLIERS.get(unit)
    if multiplier is None:
        return None
    return int(number * multiplier)


# ---------------------------------------------------------------------------
# Live memory introspection via SHOW STORAGE INFO
# ---------------------------------------------------------------------------


async def get_storage_info() -> dict[str, str]:
    """Run ``SHOW STORAGE INFO`` and return all rows as a dict.

    Returns:
        Dict mapping lowercase storage info keys to their string values.
        Example: ``{"vertex_count": "1234", "edge_count": "5678",
        "memory_usage": "42MiB", ...}``

    Raises:
        RuntimeError: If the query fails (Memgraph not running, etc.).
    """
    raw: dict[str, str] = {}
    try:
        async with _pool.get_read_session() as session:
            result = await session.run("SHOW STORAGE INFO")
            async for record in result:
                # SHOW STORAGE INFO returns (storage info, value) columns.
                # Column names vary by Memgraph version: could be
                # "storage info"/"value" or positional.
                keys = record.keys()
                if len(keys) >= 2:
                    k = str(record[keys[0]]).strip().lower()
                    v = str(record[keys[1]]).strip()
                    raw[k] = v
            await result.consume()
    except Exception as exc:
        logger.error("SHOW STORAGE INFO failed: %s", exc)
        raise RuntimeError(f"Failed to query Memgraph storage info: {exc}") from exc

    return raw


def _parse_int(raw: dict[str, str], *keys: str) -> int:
    """Try multiple key names, return first parseable int or 0."""
    for key in keys:
        if key in raw:
            try:
                return int(raw[key])
            except (ValueError, TypeError):
                pass
    return 0


def _parse_mem(raw: dict[str, str], *keys: str) -> int | None:
    """Try multiple key names, return first parseable size or None."""
    for key in keys:
        if key in raw:
            parsed = _parse_size(raw[key])
            if parsed is not None:
                return parsed
    return None


async def get_memory_usage() -> MemoryUsage:
    """Query Memgraph's current memory usage.

    Uses ``SHOW STORAGE INFO`` which is available on both Community and
    Enterprise editions.

    **Billing metric priority** (Memgraph Enterprise):

    1. ``memory_tracked`` — total tracked allocations (most accurate)
    2. ``graph_memory_tracked`` + ``vector_index_memory_tracked`` — v3.9+
       breakdown, summed if ``memory_tracked`` is absent
    3. ``memory_usage`` — legacy fallback for older versions

    The limit applies to the entire Memgraph instance (not per-database).

    Returns:
        :class:`MemoryUsage` with parsed memory values.

    Raises:
        RuntimeError: If Memgraph is unreachable or the query fails.
    """
    raw = await get_storage_info()

    # ── Parse granular memory fields (v3.9+) ──────────────────────
    graph_mem = _parse_mem(raw, "graph_memory_tracked", "graph memory tracked")
    vector_mem = _parse_mem(
        raw, "vector_index_memory_tracked", "vector index memory tracked"
    )

    # ── Resolve billable memory: memory_tracked > sum(graph+vector) > memory_usage
    used_bytes = 0
    tracked = _parse_mem(raw, "memory_tracked", "memory tracked")
    if tracked is not None:
        used_bytes = tracked
    elif graph_mem is not None or vector_mem is not None:
        used_bytes = (graph_mem or 0) + (vector_mem or 0)
    else:
        fallback = _parse_mem(raw, "memory_usage", "memory usage")
        used_bytes = fallback or 0

    return MemoryUsage(
        used_bytes=used_bytes,
        graph_memory_bytes=graph_mem,
        vector_index_memory_bytes=vector_mem,
        vertex_count=_parse_int(raw, "vertex_count", "vertex count"),
        edge_count=_parse_int(raw, "edge_count", "edge count"),
        memory_res_bytes=_parse_mem(raw, "memory_res", "memory res"),
        raw=raw,
    )


async def get_workspace_node_counts(workspace: str) -> dict[str, int]:
    """Count nodes per label prefix for a given workspace.

    Memgraph does not expose per-label memory, but we can count nodes
    per storage type to estimate relative usage.

    Returns:
        Dict like ``{"KV": 1234, "Vec": 567, "DocStatus": 89, "Graph": 456}``
    """
    from ._constants import validate_identifier

    validate_identifier(workspace, "workspace")

    counts: dict[str, int] = {}
    label_prefixes = {
        "KV": f"KV_{workspace}_",
        "Vec": f"Vec_{workspace}_",
        "DocStatus": f"DocStatus_{workspace}",
        "Graph": workspace,  # MemgraphStorage uses workspace as the label directly
    }

    try:
        async with _pool.get_read_session() as session:
            # First, get all labels in the database
            result = await session.run("CALL db.labels() YIELD label RETURN label")
            all_labels: list[str] = []
            async for record in result:
                all_labels.append(record["label"])
            await result.consume()

            # Count nodes for each label that matches our workspace prefixes
            for storage_type, prefix in label_prefixes.items():
                total = 0
                matching_labels = [lbl for lbl in all_labels if lbl.startswith(prefix)]
                for label in matching_labels:
                    count_result = await session.run(
                        f"MATCH (n:`{label}`) RETURN count(n) AS cnt"
                    )
                    record = await count_result.single()
                    if record:
                        total += record["cnt"]
                    await count_result.consume()
                counts[storage_type] = total
    except Exception as exc:
        logger.warning("Failed to count workspace nodes: %s", exc)

    return counts


# ---------------------------------------------------------------------------
# Pre-insert size estimation (no Memgraph connection needed)
# ---------------------------------------------------------------------------

# LightRAG default chunking parameters
_DEFAULT_CHUNK_SIZE = 1200  # characters per chunk (LightRAG default)
_DEFAULT_CHUNK_OVERLAP = 100  # overlap between chunks
_GRAPH_OVERHEAD_RATIO = 0.20  # ~20% of text size for graph nodes/edges
_KV_JSON_OVERHEAD = 200  # JSON serialization overhead per KV entry (bytes)
_MEMGRAPH_NODE_OVERHEAD = 128  # estimated per-node fixed overhead in Memgraph

# Bytes per scalar element by quantization kind (Memgraph 3.8+ vector indexes)
_SCALAR_BYTES: dict[str, float] = {"f32": 4.0, "f16": 2.0, "i8": 1.0}


def _resolve_scalar_bytes() -> float:
    """Return bytes-per-element for the configured vector scalar_kind."""
    import os

    from ._constants import DEFAULT_VECTOR_SCALAR_KIND, MEMGRAPH_VECTOR_SCALAR_KIND_ENV

    kind = os.environ.get(MEMGRAPH_VECTOR_SCALAR_KIND_ENV, DEFAULT_VECTOR_SCALAR_KIND)
    return _SCALAR_BYTES.get(kind, _SCALAR_BYTES[DEFAULT_VECTOR_SCALAR_KIND])


def estimate_insert_cost(
    text: str,
    *,
    embedding_dim: int = 1536,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> InsertCostEstimate:
    """Estimate memory cost of inserting a document into LightRAG + Memgraph.

    This is a heuristic estimate based on the LightRAG ingestion pipeline:

    1. **full_document KV entry**: The raw text is stored as a JSON-serialized
       node in ``KV_{workspace}_full_docs``. Cost is roughly ``len(text)``
       bytes plus JSON overhead.

    2. **text_chunk KV entries**: Each chunk is stored similarly. Total cost
       is roughly ``len(text)`` bytes (chunks cover the full text with
       overlap, so total size is close to the original).

    3. **Vector embeddings**: Each chunk gets an embedding vector. Cost
       depends on ``scalar_kind``: f32 = 4B, **f16 = 2B** (default),
       i8 = 1B per dimension. Controlled via ``MEMGRAPH_VECTOR_SCALAR_KIND``.

    4. **Graph overhead**: Entity/relationship extraction creates graph nodes
       and edges. This is hard to predict exactly, so we use a ~20% heuristic
       on top of the text size.

    Args:
        text: The raw document text to be inserted.
        embedding_dim: Dimension of the embedding model (default: 1536 for
            OpenAI text-embedding-3-small / ada-002).
        chunk_size: LightRAG chunk size in characters (default: 1200).
        chunk_overlap: LightRAG chunk overlap in characters (default: 100).

    Returns:
        :class:`InsertCostEstimate` with per-component byte estimates.
    """
    text_len = len(text.encode("utf-8"))
    bytes_per_scalar = _resolve_scalar_bytes()

    # Estimate number of chunks
    effective_chunk_size = max(chunk_size - chunk_overlap, 1)
    num_chunks = max(1, math.ceil(text_len / effective_chunk_size))

    # 1. Full document KV entry: raw text + JSON wrapper + node overhead
    full_doc_bytes = text_len + _KV_JSON_OVERHEAD + _MEMGRAPH_NODE_OVERHEAD

    # 2. Text chunks KV: each chunk is ~chunk_size bytes + overhead
    # Total chunk bytes is roughly text_len (overlap means slightly more,
    # but Memgraph deduplicates nothing here)
    text_chunks_bytes = text_len + (
        num_chunks * (_KV_JSON_OVERHEAD + _MEMGRAPH_NODE_OVERHEAD)
    )

    # 3. Embedding vectors: num_chunks * dim * bytes_per_scalar
    # Plus the Vec_ node overhead (id, content snippet, etc.)
    embedding_bytes = num_chunks * (
        int(embedding_dim * bytes_per_scalar)
        + _MEMGRAPH_NODE_OVERHEAD
        + 256  # content snippet stored alongside embedding
    )

    # 4. Graph overhead: entity extraction + relationships
    # Very document-dependent, but ~20% of text size is a reasonable heuristic.
    # More entities = more nodes + edges with properties.
    graph_overhead_bytes = int(text_len * _GRAPH_OVERHEAD_RATIO) + (
        num_chunks * _MEMGRAPH_NODE_OVERHEAD  # chunk_entity nodes
    )

    return InsertCostEstimate(
        full_doc_bytes=full_doc_bytes,
        text_chunks_bytes=text_chunks_bytes,
        embedding_bytes=embedding_bytes,
        graph_overhead_bytes=graph_overhead_bytes,
        num_chunks_estimate=num_chunks,
    )


def estimate_batch_insert_cost(
    texts: list[str],
    *,
    embedding_dim: int = 1536,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> InsertCostEstimate:
    """Estimate memory cost for a batch of documents.

    Sums individual estimates. Useful for ``ainsert()`` calls with
    multiple documents.

    Args:
        texts: List of document texts to be inserted.
        embedding_dim: Dimension of the embedding model.
        chunk_size: LightRAG chunk size in characters.
        chunk_overlap: LightRAG chunk overlap in characters.

    Returns:
        Aggregated :class:`InsertCostEstimate`.
    """
    totals = InsertCostEstimate(
        full_doc_bytes=0,
        text_chunks_bytes=0,
        embedding_bytes=0,
        graph_overhead_bytes=0,
        num_chunks_estimate=0,
    )
    for text in texts:
        est = estimate_insert_cost(
            text,
            embedding_dim=embedding_dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        totals.full_doc_bytes += est.full_doc_bytes
        totals.text_chunks_bytes += est.text_chunks_bytes
        totals.embedding_bytes += est.embedding_bytes
        totals.graph_overhead_bytes += est.graph_overhead_bytes
        totals.num_chunks_estimate += est.num_chunks_estimate
    return totals


# ---------------------------------------------------------------------------
# Combined budget check: live usage + estimated cost
# ---------------------------------------------------------------------------


async def check_memory_budget(
    text: str | None = None,
    texts: list[str] | None = None,
    *,
    embedding_dim: int = 1536,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    memory_limit_bytes: int | None = None,
    safety_margin: float = 0.05,
) -> BudgetCheck:
    """Check whether a document insert fits within the remaining memory budget.

    Combines live memory introspection with pre-insert size estimation
    to determine if the insert is safe.

    Args:
        text: Single document text (mutually exclusive with ``texts``).
        texts: Batch of document texts (mutually exclusive with ``text``).
        embedding_dim: Embedding model dimension.
        chunk_size: LightRAG chunk size.
        chunk_overlap: LightRAG chunk overlap.
        memory_limit_bytes: LightRAG memory budget in bytes. If None,
            reads from ``MEMGRAPH_MEMORY_LIMIT`` env var. If 0 or unset,
            the check always returns ``fits=True`` (unlimited).
        safety_margin: Reserve this fraction of total capacity as headroom
            (default: 5%). The insert is rejected if it would leave less
            than this margin.

    Returns:
        :class:`BudgetCheck` with the verdict and breakdown.

    Raises:
        ValueError: If neither ``text`` nor ``texts`` is provided.
        RuntimeError: If Memgraph is unreachable.
    """
    import os

    if text is None and texts is None:
        raise ValueError("Provide either 'text' or 'texts'")

    # Estimate cost
    if texts is not None:
        cost = estimate_batch_insert_cost(
            texts,
            embedding_dim=embedding_dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        cost = estimate_insert_cost(
            text,
            embedding_dim=embedding_dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Get live memory usage
    usage = await get_memory_usage()

    # Resolve memory limit
    if memory_limit_bytes is None:
        env_limit = os.environ.get(MEMGRAPH_MEMORY_LIMIT_ENV, "")
        if env_limit:
            parsed = _parse_size(env_limit)
            if parsed is not None:
                memory_limit_bytes = parsed
        if memory_limit_bytes is None:
            memory_limit_bytes = DEFAULT_MEMORY_LIMIT_GIB * 1024**3

    # 0 = unlimited — always fits
    if memory_limit_bytes <= 0:
        return BudgetCheck(
            fits=True,
            used_bytes=usage.used_bytes,
            limit_bytes=0,
            estimated_cost_bytes=cost.total_bytes,
            remaining_bytes=0,
            headroom_ratio=-1.0,
        )

    # Calculate budget
    projected = usage.used_bytes + cost.total_bytes
    remaining = memory_limit_bytes - projected
    headroom = remaining / memory_limit_bytes
    fits = headroom >= safety_margin

    return BudgetCheck(
        fits=fits,
        used_bytes=usage.used_bytes,
        limit_bytes=memory_limit_bytes,
        estimated_cost_bytes=cost.total_bytes,
        remaining_bytes=max(0, remaining),
        headroom_ratio=max(0.0, headroom),
    )
