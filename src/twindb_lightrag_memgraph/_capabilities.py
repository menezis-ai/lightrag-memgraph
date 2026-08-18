"""Runtime capability detection for the connected Memgraph instance.

Storage backends (KV / Vector / DocStatus / graph) run on plain
``memgraph/memgraph`` — they call only core Cypher plus the core
``vector_search`` procedure (native since Memgraph 3.0). MAGE is NOT required
for that floor tier.

Some *additive, opt-in* features (graph-algorithm curation: Louvain community
detection, Katz centrality, …) do need MAGE query modules. Rather than turning
MAGE into a hard install prerequisite, we DETECT at runtime which procedures the
connected instance actually exposes, per-procedure, via ``CALL mg.procedures()``.

Tier model:

* **Floor tier** — always available. Everything BNP runs today on the base
  image. If MAGE is absent, additive features are silently skipped; nothing in
  the storage path changes.
* **MAGE tier** — additive. A feature declares the exact procedures it needs
  (e.g. ``community_detection.get``) and gates itself on
  :func:`has_procedure`. Absence degrades to the floor tier, never crashes.

Fail-closed: any probe error (old server, permissions, connectivity) resolves to
"procedure absent" → floor tier. The probe is not cached on failure, so a
transient error self-heals on the next call.

Override via ``TWIN_MAGE`` (see :data:`TWIN_MAGE_ENV`):
``auto`` (default) probes; ``off`` forces the floor tier; ``on`` trusts the
operator that MAGE is present and skips the probe entirely.
"""

from dataclasses import dataclass
import logging
import os
import threading

from ._constants import _FALSE_FLAG_VALUES, TWIN_MAGE_ENV
from ._pool import get_read_session

logger = logging.getLogger("twindb_lightrag_memgraph")

_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})

# Human-facing markers only: used to render the boot log line, NOT to gate any
# feature. Feature gating is always per-procedure via has_procedure().
_KNOWN_MAGE_MARKERS = (
    "community_detection.get",  # Louvain
    "katz_centrality.get",  # Katz
    "pagerank.get",
    "node_similarity.cosine",
    "link_prediction.predict",
)

_thread_lock = threading.Lock()
# None = not yet probed; frozenset = probed procedure-name set (may be empty).
_available_procedures: frozenset[str] | None = None
_tier_logged = False


@dataclass(frozen=True)
class MageCapabilitySnapshot:
    """Atomic MAGE diagnostics for operator-facing status surfaces.

    ``procedures=None`` means no procedure probe was performed or its result
    was unavailable. A resolved ``available`` beside that value therefore
    comes from the explicit ``TWIN_MAGE`` override, not from a fabricated
    procedure count.
    """

    available: bool | None
    procedures: frozenset[str] | None


def _mage_override() -> bool | None:
    """Resolve the ``TWIN_MAGE`` override.

    Returns ``True`` (force MAGE tier / skip probe), ``False`` (force floor
    tier), or ``None`` (``auto`` — probe the instance).
    """
    raw = os.environ.get(TWIN_MAGE_ENV, "").strip().lower()
    if raw in _FALSE_FLAG_VALUES:
        return False
    if raw in _TRUE_FLAG_VALUES:
        return True
    return None  # "auto" / unset / unrecognised


async def get_available_procedures(*, force: bool = False) -> frozenset[str]:
    """Return the set of procedure names the connected instance exposes.

    Lazily probes ``CALL mg.procedures()`` once and caches the result for the
    process lifetime (server capabilities do not change under us). On probe
    failure returns an empty set WITHOUT caching, so the next call retries.

    Args:
        force: Re-probe even if a cached result exists.
    """
    global _available_procedures, _tier_logged

    if _available_procedures is not None and not force:
        return _available_procedures

    try:
        async with get_read_session() as session:
            result = await session.run("CALL mg.procedures() YIELD name RETURN name")
            names = frozenset([record["name"] async for record in result])
            await result.consume()
    except Exception as exc:  # noqa: BLE001 — fail-closed to floor tier
        logger.warning(
            "MAGE capability probe failed (CALL mg.procedures()) — assuming "
            "floor tier (no MAGE). Will retry on next call. detail=%s: %s",
            type(exc).__name__,
            exc,
        )
        return frozenset()

    with _thread_lock:
        _available_procedures = names
        if not _tier_logged:
            _tier_logged = True
            present = sorted(m for m in _KNOWN_MAGE_MARKERS if m in names)
            if present:
                logger.info(
                    "MAGE tier: detected %s → graph-algorithm curation AVAILABLE",
                    ", ".join(present),
                )
            else:
                logger.info(
                    "MAGE absent on connected instance → floor tier "
                    "(storage + LLM curation only; base memgraph image)"
                )
    return names


async def has_procedure(name: str) -> bool:
    """Return True when the connected instance exposes procedure ``name``.

    Honors the ``TWIN_MAGE`` override before probing: ``off`` → always False,
    ``on`` → always True (operator asserts MAGE present, probe skipped).
    """
    forced = _mage_override()
    if forced is False:
        return False
    if forced is True:
        return True
    procedures = await get_available_procedures()
    return name in procedures


async def has_all_procedures(*names: str) -> bool:
    """Return True only when every procedure in ``names`` is available.

    Convenience for a feature that needs more than one MAGE procedure; a single
    probe backs all lookups.
    """
    forced = _mage_override()
    if forced is False:
        return False
    if forced is True:
        return True
    procedures = await get_available_procedures()
    return all(name in procedures for name in names)


def _has_mage_markers(procedures: frozenset[str]) -> bool:
    """Return whether a procedure snapshot contains a known MAGE marker."""
    return any(marker in procedures for marker in _KNOWN_MAGE_MARKERS)


async def is_mage_available(procedures: frozenset[str] | None = None) -> bool:
    """Return True when the instance exposes the MAGE query-module tier.

    Discriminates on :data:`_KNOWN_MAGE_MARKERS`, NOT on "the probe returned a
    non-empty set". The base ``memgraph/memgraph`` image already exposes core
    procedures (``vector_search.search``, ``mg.procedures``, ``mg.functions``),
    so a truthiness test on the probe result reports MAGE on every reachable
    instance — see the ``_CORE_ONLY`` fixture in ``tests/test_capabilities.py``,
    which pins exactly that set as the no-MAGE case.

    Honors ``TWIN_MAGE`` like the other public predicates.

    Args:
        procedures: An already-probed procedure set to answer from. Pass it
            when a caller needs the count AND the tier to describe the SAME
            instant: a probe failure is deliberately not cached, so two
            consecutive calls can straddle a reconnection and publish a
            contradictory pair (``procedures=0`` with ``mage=True``). Omit it
            for feature gating, where the self-healing re-probe is the point.
    """
    forced = _mage_override()
    if forced is not None:
        return forced
    if procedures is None:
        procedures = await get_available_procedures()
    return _has_mage_markers(procedures)


async def get_mage_capability_snapshot() -> MageCapabilitySnapshot:
    """Return one coherent MAGE tier/procedure snapshot for diagnostics.

    Explicit overrides are authoritative and skip the procedure probe, just
    like :func:`has_procedure` and :func:`is_mage_available`. In automatic
    mode, one procedure snapshot backs both the tier and the count. The probe
    API fails closed to an uncached empty set; because every reachable
    Memgraph exposes at least ``mg.procedures`` itself, that state is reported
    as unknown rather than as a confirmed floor tier.
    """
    forced = _mage_override()
    if forced is not None:
        return MageCapabilitySnapshot(available=forced, procedures=None)

    procedures = await get_available_procedures()
    if not procedures:
        return MageCapabilitySnapshot(available=None, procedures=None)
    return MageCapabilitySnapshot(
        available=_has_mage_markers(procedures),
        procedures=procedures,
    )


def reset_capability_cache() -> None:
    """Clear the probed-capability cache. Called on driver shutdown."""
    global _available_procedures, _tier_logged
    with _thread_lock:
        _available_procedures = None
        _tier_logged = False
