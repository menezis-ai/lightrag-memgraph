"""Instance-wide Memgraph storage quota.

Memgraph caps in-process memory via ``--memory-limit`` (env
``MEMGRAPH_MEMORY_LIMIT``, value like ``2GiB``). When the cap is
reached, ingestion fails *at write time* with no early signal to the
operator. This module surfaces the cap up to the WebUI so an operator
sees pressure (≥85 %) before it becomes a wall (100 %) and so blocking
endpoints can refuse uploads with a clear 507 before partial writes
corrupt state.

Public surface:

- :func:`parse_memgraph_limit` — env-string → bytes (``GiB`` / ``MiB``
  / ``KiB`` / ``B`` / plain integer)
- :func:`snapshot` — async, returns ``{used_bytes, limit_bytes,
  used_pct, status}`` with ``status ∈ {ok, warning, blocked}``
- :func:`enforce_instance_quota` — FastAPI dependency that raises 507
  when ``status == "blocked"``

If ``MEMGRAPH_MEMORY_LIMIT`` is not set, the limit is ``None`` and
every dependency / snapshot reports ``status="ok"`` with
``limit_bytes=None`` so dev environments without a cap never trip
the guard.
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

# Accept binary (KiB / MiB / GiB / TiB) and decimal (KB / MB / GB / TB)
# suffixes plus the bare ``B``. Whitespace and case insensitive.
_LIMIT_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?i?B|B)?\s*$",
    re.IGNORECASE,
)
_UNIT_FACTORS: Final[dict[str, int]] = {
    "B": 1,
    "KB": 1_000,
    "KIB": 1024,
    "MB": 1_000_000,
    "MIB": 1024 ** 2,
    "GB": 1_000_000_000,
    "GIB": 1024 ** 3,
    "TB": 1_000_000_000_000,
    "TIB": 1024 ** 4,
}


def parse_memgraph_limit(raw: str | None) -> int | None:
    """Parse a ``MEMGRAPH_MEMORY_LIMIT`` value to bytes.

    Returns ``None`` when ``raw`` is missing, empty, or unparseable
    (the caller treats ``None`` as "no quota configured" and disables
    the guard). Accepts ``2GiB``, ``2 GiB``, ``2048MiB``, ``2GB``,
    ``2000000000``, ``2147483648B``. Floats like ``1.5GiB`` are
    rounded down to the nearest byte.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    m = _LIMIT_RE.match(text)
    if not m:
        logger.warning(
            "MEMGRAPH_MEMORY_LIMIT=%r is not a recognised size literal; "
            "instance quota guard disabled",
            raw,
        )
        return None
    value = float(m.group("value"))
    unit = (m.group("unit") or "B").upper()
    factor = _UNIT_FACTORS.get(unit)
    if factor is None:
        logger.warning(
            "MEMGRAPH_MEMORY_LIMIT=%r: unit %r not recognised; guard disabled",
            raw,
            unit,
        )
        return None
    return int(value * factor)


def get_limit_bytes() -> int | None:
    """Resolve the configured quota at call time (env reads cheap)."""
    return parse_memgraph_limit(os.environ.get(_MEMGRAPH_LIMIT_ENV))


# ---------------------------------------------------------------------------
# Memgraph probe
# ---------------------------------------------------------------------------


async def get_used_bytes() -> int | None:
    """Best-effort read of Memgraph's current memory usage in bytes.

    Uses ``SHOW STORAGE INFO`` which Memgraph (both Community and
    Enterprise) exposes as a result-set of ``(storage info, value)``
    rows. We grep for ``memory_res`` (resident set) first, then fall
    back to ``memory_allocated`` if the build emits only that key.

    Returns ``None`` on any failure — the caller treats a missing
    probe as "guard cannot enforce" and lets the request through. This
    is fail-OPEN by design: we never want a quota probe failure to
    block ingestion, only a real over-quota condition.
    """
    try:
        async with _pool.get_read_session() as session:
            result = await session.run("SHOW STORAGE INFO")
            rows = await result.data()
            await result.consume()
    except Exception:  # noqa: BLE001 — fail-open, never propagate
        logger.exception("[quota] SHOW STORAGE INFO failed; guard inactive")
        return None
    return _extract_memory_bytes(rows)


def _extract_memory_bytes(rows: list[dict[str, Any]]) -> int | None:
    """Pull the memory-usage byte count out of ``SHOW STORAGE INFO`` rows.

    Memgraph ``SHOW STORAGE INFO`` rows are typed as
    ``{"storage info": <key>, "value": <number>}`` (older builds may
    use different column names; we accept the union).

    Preference order:

      1. ``memory_res`` (resident set size — what the OS sees)
      2. ``memory_tracked`` (Memgraph's internal accounting)
      3. ``memory_allocated`` (legacy alias)
    """
    if not rows:
        return None
    preference = ("memory_res", "memory_tracked", "memory_allocated")
    indexed: dict[str, int] = {}
    for row in rows:
        key = None
        for cand in ("storage info", "storage_info", "name", "key"):
            if cand in row:
                key = row[cand]
                break
        if key is None:
            continue
        value = None
        for cand in ("value", "size", "bytes"):
            if cand in row:
                value = row[cand]
                break
        if not isinstance(value, (int, float)):
            continue
        indexed[str(key).strip().lower()] = int(value)
    for key in preference:
        if key in indexed:
            return indexed[key]
    return None


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


async def snapshot() -> dict[str, Any]:
    """Build the full quota payload consumed by ``GET /twin/api/quota``
    and the ingestion dependency.

    Shape:

    .. code-block:: json

        {
          "used_bytes": 1234567,
          "limit_bytes": 2147483648,
          "used_pct": 0.000575,
          "status": "ok",
          "warn_threshold": 0.85,
          "configured": true
        }

    Fields:

    - ``configured`` — false when ``MEMGRAPH_MEMORY_LIMIT`` is unset
      (the guard is then inert).
    - ``used_pct`` — ``null`` when either ``used_bytes`` or
      ``limit_bytes`` is ``null``.
    - ``status`` — derived from ``used_pct`` via the warn / block
      thresholds; ``"ok"`` when no quota is configured.
    """
    limit_bytes = get_limit_bytes()
    used_bytes = await get_used_bytes()
    used_pct: float | None
    if limit_bytes is None or used_bytes is None:
        used_pct = None
    elif limit_bytes <= 0:
        used_pct = None
    else:
        used_pct = used_bytes / limit_bytes
    return {
        "used_bytes": used_bytes,
        "limit_bytes": limit_bytes,
        "used_pct": used_pct,
        "status": _status_from(used_pct),
        "warn_threshold": WARN_THRESHOLD,
        "configured": limit_bytes is not None,
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

    ``configured == False`` → no-op. Probe failure → no-op (fail-open).
    ``status == "blocked"`` → ``HTTPException(507)`` with a message
    that names the absolute usage in GiB so the operator immediately
    knows how much to free.
    """
    snap = await snapshot()
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
    "enforce_instance_quota",
    "get_limit_bytes",
    "get_used_bytes",
    "parse_memgraph_limit",
    "snapshot",
]
