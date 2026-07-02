"""Instance-wide Memgraph storage quota.

Memgraph rejects writes once its tracked allocations reach the effective
memory limit (``--memory-limit``, or — on Enterprise — the lower of that
and the license capacity). When the wall is hit, ingestion fails *at
write time* with no early signal. This module reads Memgraph's own
accounting and surfaces the pressure so an operator sees ``warning``
(≥85 %) before the ``blocked`` wall, and so ingestion endpoints can
refuse uploads with a clear 507 before partial writes corrupt state.

What it measures (verified empirically against Memgraph 3.9.0 **and**
3.10.1 — ``SHOW STORAGE INFO`` field names differ across versions, see
the ``_*_KEYS`` tuples):

- **used**  = ``global_memory_tracked`` (3.10) / ``memory_tracked`` (3.9)
  — the allocation count Memgraph enforces its limit against. NOT the
  process RSS (``memory_res``, which includes allocator overhead and
  over-reports ~2×), and NOT the sum of ingested file sizes (originals
  are deleted; only chunks + graph + vectors remain).
- **limit** = ``global_runtime_allocation_limit`` (3.10) /
  ``allocation_limit`` (3.9) — read straight from Memgraph, so it tracks
  the real ``--memory-limit`` and the Enterprise license cap without a
  hand-maintained env. Falls back to ``MEMGRAPH_MEMORY_LIMIT`` only when
  Memgraph reports no limit.
- **graph_bytes / vector_bytes** — the actual data footprint
  (``db_storage_memory_tracked`` + ``db_embedding_memory_tracked`` on
  3.10; ``graph_memory_tracked`` + ``vector_index_memory_tracked`` on
  3.9), surfaced for capacity visibility.

Values come back as unit strings (``"409.72MiB"``, ``"2.00GiB"``,
``"unlimited"``, ``"0B"``) — :func:`_parse_size` handles those plus raw
numbers. A probe failure is **fail-open**: the guard never blocks on its
own malfunction, only on a real over-quota condition.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Final

from fastapi import HTTPException

from .. import _pool

logger = logging.getLogger(__name__)

# Warn threshold is hardcoded (product decision 2026-06-17): operator UX
# wants a single boundary, not yet-another tunable.
WARN_THRESHOLD: Final[float] = 0.85
BLOCK_THRESHOLD: Final[float] = 1.0

_MEMGRAPH_LIMIT_ENV = "MEMGRAPH_MEMORY_LIMIT"
_BUDGET_ENFORCE_ENV = "MEMGRAPH_BUDGET_ENFORCE"
_BUDGET_REJECT = "reject"

# Accept binary (KiB/MiB/GiB/TiB) and decimal (KB/MB/GB/TB) suffixes plus
# the bare ``B``, with an optional decimal part. Case/whitespace insensitive.
_SIZE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>(?:[KMGT]i?B|B))?\s*$",
    re.IGNORECASE,
)
_UNIT_FACTORS: Final[dict[str, int]] = {
    "B": 1,
    "KB": 1_000,
    "KIB": 1024,
    "MB": 1_000_000,
    "MIB": 1024**2,
    "GB": 1_000_000_000,
    "GIB": 1024**3,
    "TB": 1_000_000_000_000,
    "TIB": 1024**4,
}

# ``SHOW STORAGE INFO`` keys, newest name first, older name(s) after.
# Verified on 3.10.1 and 3.9.0 — note graph/vector keys are *swapped*
# between the two releases, hence both must be tried.
_USED_KEYS: Final = ("global_memory_tracked", "memory_tracked")
_USED_RSS_KEYS: Final = ("memory_res",)  # coarse last resort (process RSS)
_RAM_LIMIT_KEYS: Final = ("global_runtime_allocation_limit", "allocation_limit")
# The Enterprise license cap, billed on data. 3.10+ only; absent on 3.9 and
# "unlimited" on Community (→ parsed to None, no license wall).
_LICENSE_LIMIT_KEYS: Final = ("global_license_allocation_limit",)
_GRAPH_KEYS: Final = ("db_storage_memory_tracked", "graph_memory_tracked")
_VECTOR_KEYS: Final = (
    "db_embedding_memory_tracked",
    "vector_index_memory_tracked",
    "embeddings_memory_tracked",
)

# Which Memgraph Enterprise plan BNP is on — decides whether vector index
# memory counts toward the *billed* footprint:
#   - "graph-analytics" (default): graph + vectors are billed.
#   - "ai-platform": only graph is billed; vector indexes are free.
# (SHOW STORAGE INFO doesn't expose the plan, so it's operator-declared.)
_LICENSE_PLAN_ENV = "TWIN_MEMGRAPH_LICENSE_PLAN"
_AI_PLATFORM = "ai-platform"


def _vectors_are_billed() -> bool:
    return (
        os.environ.get(_LICENSE_PLAN_ENV, "graph-analytics").strip().lower()
        != _AI_PLATFORM
    )


def budget_enforce_mode() -> str:
    """Instance budget enforcement mode.

    BNP production declares ``MEMGRAPH_BUDGET_ENFORCE=reject``. ``reject``
    remains the default to preserve the existing 507 guard whenever an
    instance limit is configured, while making the env contract explicit
    in the quota payload.
    """
    return (
        os.environ.get(_BUDGET_ENFORCE_ENV, _BUDGET_REJECT).strip().lower()
        or _BUDGET_REJECT
    )


def _parse_size(value: Any) -> int | None:
    """Parse a Memgraph size value to bytes.

    Handles raw ints/floats, unit strings (``"409.72MiB"``, ``"2.00GiB"``,
    ``"0B"``, ``"2048"``), and the sentinels Memgraph emits for "no
    limit" (``"unlimited"``). Returns ``None`` when not a size.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() in ("unlimited", "inf", "infinite", "none"):
        return None
    m = _SIZE_RE.match(text)
    if not m:
        return None
    factor = _UNIT_FACTORS.get((m.group("unit") or "B").upper())
    if factor is None:
        return None
    return int(float(m.group("value")) * factor)


def parse_memgraph_limit(raw: str | None) -> int | None:
    """Parse a ``MEMGRAPH_MEMORY_LIMIT`` env value to bytes (or ``None``).

    Used only as the fallback limit when Memgraph itself reports none.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    parsed = _parse_size(text)
    if parsed is None:
        logger.warning(
            "MEMGRAPH_MEMORY_LIMIT=%r is not a recognised size literal; "
            "instance quota fallback limit disabled",
            raw,
        )
    return parsed


def get_limit_bytes() -> int | None:
    """Env-override / fallback limit (sync). :func:`snapshot` prefers the
    allocation limit Memgraph reports itself."""
    return parse_memgraph_limit(os.environ.get(_MEMGRAPH_LIMIT_ENV))


# ---------------------------------------------------------------------------
# Memgraph probe
# ---------------------------------------------------------------------------


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Parse ``SHOW STORAGE INFO`` rows → ``{lower_key: bytes}``.

    Pure (no I/O) so version compatibility is unit-testable against the
    real 3.9 and 3.10 field sets. Each row is ``{"storage info": <key>,
    "value": <unit-string|number>}`` (older builds vary the column names).
    """
    indexed: dict[str, int] = {}
    for row in rows:
        key = next(
            (
                row[c]
                for c in ("storage info", "storage_info", "name", "key")
                if c in row
            ),
            None,
        )
        if key is None:
            continue
        raw = next((row[c] for c in ("value", "size", "bytes") if c in row), None)
        parsed = _parse_size(raw)
        if parsed is not None:
            indexed[str(key).strip().lower()] = parsed
    return indexed


async def _read_storage_info() -> dict[str, int]:
    """One ``SHOW STORAGE INFO`` probe → ``{key: bytes}`` (lower-cased keys).

    Fail-open: returns ``{}`` on any error so the guard never blocks on
    its own malfunction.
    """
    try:
        async with _pool.get_read_session() as session:
            result = await session.run("SHOW STORAGE INFO")
            rows = await result.data()
            await result.consume()
    except Exception:  # noqa: BLE001
        # Fail-open, never propagate quota guard failures.
        logger.exception("[quota] SHOW STORAGE INFO failed; guard inactive")
        return {}
    return _index_rows(rows)


def _pick(indexed: dict[str, int], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in indexed:
            return indexed[key]
    return None


async def get_used_bytes() -> int | None:
    """Tracked memory Memgraph enforces its limit against
    (``global_memory_tracked`` / ``memory_tracked``), falling back to the
    process RSS (``memory_res``) only if the tracker isn't exposed."""
    indexed = await _read_storage_info()
    used = _pick(indexed, _USED_KEYS)
    return used if used is not None else _pick(indexed, _USED_RSS_KEYS)


# ---------------------------------------------------------------------------
# Snapshot + status logic
# ---------------------------------------------------------------------------


def _status_from(used_pct: float | None) -> str:
    """Map a usage percentage to the three-state status."""
    if used_pct is None:
        return "ok"
    if used_pct >= BLOCK_THRESHOLD:
        return "blocked"
    if used_pct >= WARN_THRESHOLD:
        return "warning"
    return "ok"


def _pct(used: int | None, limit: int | None) -> float | None:
    if used is None or limit is None or limit <= 0:
        return None
    return used / limit


async def snapshot() -> dict[str, Any]:
    """Full quota payload for ``GET /twin/api/quota`` and the 507 guard.

    Two distinct walls are computed from one probe and the binding one
    becomes the headline (``used_bytes`` / ``limit_bytes`` / ``status``):

    - **license** — the *billed* footprint (graph [+ vectors, unless the
      AI-Platform plan]) vs ``global_license_allocation_limit``. This is
      what Memgraph charges for and what triggers read-only writes on
      Enterprise once exceeded. Absent on Community (license "unlimited").
    - **ram** — Memgraph's tracked allocation vs the effective
      ``--memory-limit`` (``global_runtime_allocation_limit``). Binds on
      Community, or when ``--memory-limit`` is lower than the license.

    Every field is surfaced so the operator can see *which* wall binds
    and what the data actually weighs — never a hand-waved number.
    """
    indexed = await _read_storage_info()

    graph = _pick(indexed, _GRAPH_KEYS)
    vector = _pick(indexed, _VECTOR_KEYS)
    vectors_billed = _vectors_are_billed()

    # --- RAM wall: tracked allocation vs the effective allocation limit ---
    ram_used = _pick(indexed, _USED_KEYS)
    ram_basis: str | None = "tracked" if ram_used is not None else None
    if ram_used is None:
        ram_used = _pick(indexed, _USED_RSS_KEYS)
        ram_basis = "rss" if ram_used is not None else None
    ram_limit = _pick(indexed, _RAM_LIMIT_KEYS)
    if ram_limit is None:
        ram_limit = get_limit_bytes()  # MEMGRAPH_MEMORY_LIMIT env fallback
    ram_pct = _pct(ram_used, ram_limit)

    # --- License wall (what Memgraph bills): data vs the license cap ---
    license_limit = _pick(indexed, _LICENSE_LIMIT_KEYS)
    billed: int | None = None
    if graph is not None:
        billed = graph
        if vectors_billed and vector is not None:
            billed += vector
    billed_pct = _pct(billed, license_limit)

    # Headline = the binding wall (closest to its cap) → the banner alerts
    # on whichever constraint will actually block writes first.
    if billed_pct is not None and (ram_pct is None or billed_pct >= ram_pct):
        used, limit, binding = billed, license_limit, "license"
    else:
        used, limit, binding = ram_used, ram_limit, "ram"
    used_pct = _pct(used, limit)

    return {
        "used_bytes": used,
        "limit_bytes": limit,
        "used_pct": used_pct,
        "status": _status_from(used_pct),
        "warn_threshold": WARN_THRESHOLD,
        "configured": limit is not None,
        "budget_enforce": budget_enforce_mode(),
        "binding": binding,
        # data footprint (the billed components)
        "graph_bytes": graph,
        "vector_bytes": vector,
        "vectors_billed": vectors_billed,
        # license / billed wall — the cost number
        "billed_bytes": billed,
        "license_limit_bytes": license_limit,
        "billed_pct": billed_pct,
        # RAM wall (--memory-limit; binds on Community or when RAM < license)
        "ram_used_bytes": ram_used,
        "ram_limit_bytes": ram_limit,
        "ram_pct": ram_pct,
        "ram_basis": ram_basis,
    }


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def _format_gib(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{n / (1024 ** 3):.2f}"


async def enforce_instance_quota() -> None:
    """FastAPI dependency: 507 when the Memgraph instance is at quota.

    No limit reported (Memgraph + env both silent) → no-op. Probe failure
    → no-op (fail-open). ``status == "blocked"`` → ``HTTPException(507)``
    naming the absolute usage in GiB so the operator knows how much to free.
    """
    snap = await snapshot()
    if snap.get("budget_enforce") != _BUDGET_REJECT:
        return
    if snap["status"] != "blocked":
        return
    used = _format_gib(snap.get("used_bytes"))
    limit = _format_gib(snap.get("limit_bytes"))
    raise HTTPException(
        status_code=507,
        detail=(
            f"Memgraph instance quota reached ({used}/{limit} GiB). "
            "Free space before ingesting."
        ),
    )


__all__ = [
    "BLOCK_THRESHOLD",
    "WARN_THRESHOLD",
    "budget_enforce_mode",
    "enforce_instance_quota",
    "get_limit_bytes",
    "get_used_bytes",
    "parse_memgraph_limit",
    "snapshot",
]
