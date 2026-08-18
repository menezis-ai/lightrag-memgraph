"""Unit tests for runtime MAGE capability detection (_capabilities.py).

No Memgraph required — the read session is mocked. These prove the tier model:
the floor tier is the default whenever MAGE is absent or the probe fails, and
the TWIN_MAGE override wins over the probe.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from twindb_lightrag_memgraph import _capabilities

# Core-only instance (base memgraph image): vector_search is core, no MAGE.
_CORE_ONLY = ["vector_search.search", "mg.procedures", "mg.functions"]
# MAGE instance: core + the graph-algorithm modules additive features need.
_WITH_MAGE = _CORE_ONLY + ["community_detection.get", "katz_centrality.get"]


class _AsyncRecords:
    def __init__(self, names):
        self._rows = [{"name": n} for n in names]
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._rows):
            raise StopAsyncIteration
        row = self._rows[self._i]
        self._i += 1
        return row


def _patch_probe(monkeypatch, names=None, *, raise_exc=None):
    """Patch _capabilities.get_read_session; return the run AsyncMock (call count)."""
    result = _AsyncRecords(names or [])
    result.consume = AsyncMock()
    run = AsyncMock(return_value=result)
    session = AsyncMock()
    session.run = run

    @asynccontextmanager
    async def _fake_read_session():
        if raise_exc is not None:
            raise raise_exc
        yield session

    monkeypatch.setattr(_capabilities, "get_read_session", _fake_read_session)
    return run


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset the module cache and clear the override env before each test."""
    monkeypatch.delenv(_capabilities.TWIN_MAGE_ENV, raising=False)
    _capabilities.reset_capability_cache()
    yield
    _capabilities.reset_capability_cache()


async def test_probe_detects_mage(monkeypatch):
    _patch_probe(monkeypatch, _WITH_MAGE)
    assert await _capabilities.has_procedure("community_detection.get") is True
    assert await _capabilities.has_all_procedures(
        "community_detection.get", "katz_centrality.get"
    )


async def test_probe_absent_mage_is_floor(monkeypatch):
    _patch_probe(monkeypatch, _CORE_ONLY)
    assert await _capabilities.has_procedure("community_detection.get") is False
    # vector_search stays visible — the floor tier is fully functional.
    assert await _capabilities.has_procedure("vector_search.search") is True


async def test_probe_error_falls_to_floor(monkeypatch):
    _patch_probe(monkeypatch, raise_exc=RuntimeError("bolt down"))
    # No exception propagates; absence is assumed.
    assert await _capabilities.has_procedure("community_detection.get") is False
    assert await _capabilities.get_available_procedures() == frozenset()


async def test_probe_error_not_cached_retries(monkeypatch):
    _patch_probe(monkeypatch, raise_exc=RuntimeError("transient"))
    assert await _capabilities.get_available_procedures() == frozenset()
    # A later successful probe must self-heal (failure was not cached).
    run = _patch_probe(monkeypatch, _WITH_MAGE)
    assert await _capabilities.has_procedure("katz_centrality.get") is True
    assert run.await_count == 1


async def test_probe_cached_once(monkeypatch):
    run = _patch_probe(monkeypatch, _WITH_MAGE)
    await _capabilities.has_procedure("community_detection.get")
    await _capabilities.has_procedure("katz_centrality.get")
    await _capabilities.get_available_procedures()
    assert run.await_count == 1  # single probe backs all lookups


async def test_env_off_forces_floor_without_probe(monkeypatch):
    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, "off")
    run = _patch_probe(monkeypatch, _WITH_MAGE)
    assert await _capabilities.has_procedure("community_detection.get") is False
    assert await _capabilities.has_all_procedures("community_detection.get") is False
    run.assert_not_awaited()  # override short-circuits — no probe issued


async def test_env_on_forces_tier_without_probe(monkeypatch):
    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, "on")
    run = _patch_probe(monkeypatch, _CORE_ONLY)  # instance lacks MAGE...
    # ...but operator asserts present; we trust and skip the probe.
    assert await _capabilities.has_procedure("community_detection.get") is True
    run.assert_not_awaited()


async def test_reset_clears_cache(monkeypatch):
    run = _patch_probe(monkeypatch, _WITH_MAGE)
    await _capabilities.get_available_procedures()
    assert run.await_count == 1
    _capabilities.reset_capability_cache()
    await _capabilities.get_available_procedures()
    assert run.await_count == 2  # re-probed after reset


# ---------------------------------------------------------------------------
# is_mage_available — the marker-based predicate
# ---------------------------------------------------------------------------


async def test_is_mage_available_false_on_core_only_instance(monkeypatch):
    """Regression: the base image exposes procedures but NOT MAGE.

    A truthiness test on the probe result (``bool(procedures)``) reported
    MAGE on every reachable instance, because ``vector_search.search`` /
    ``mg.procedures`` / ``mg.functions`` are core.
    """
    _patch_probe(monkeypatch, _CORE_ONLY)
    assert await _capabilities.get_available_procedures()  # non-empty…
    assert await _capabilities.is_mage_available() is False  # …but not MAGE


async def test_is_mage_available_true_when_markers_present(monkeypatch):
    _patch_probe(monkeypatch, _WITH_MAGE)
    assert await _capabilities.is_mage_available() is True


async def test_is_mage_available_honors_override(monkeypatch):
    _patch_probe(monkeypatch, _WITH_MAGE)
    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, "off")
    assert await _capabilities.is_mage_available() is False

    _capabilities.reset_capability_cache()
    _patch_probe(monkeypatch, _CORE_ONLY)
    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, "on")
    assert await _capabilities.is_mage_available() is True


async def test_is_mage_available_false_when_probe_fails(monkeypatch):
    _patch_probe(monkeypatch, raise_exc=RuntimeError("no server"))
    assert await _capabilities.is_mage_available() is False


async def test_is_mage_available_answers_from_a_supplied_snapshot(monkeypatch):
    """A caller passing a probed set must not trigger a second probe.

    That second probe is what let ``/twin/api/system/about`` publish a count
    and a tier taken from two different instants.
    """
    run = _patch_probe(monkeypatch, _WITH_MAGE)
    snapshot = await _capabilities.get_available_procedures()
    before = run.await_count

    assert await _capabilities.is_mage_available(snapshot) is True
    assert await _capabilities.is_mage_available(_CORE_ONLY) is False
    assert run.await_count == before  # no re-probe on either call


async def test_supplied_snapshot_still_yields_to_the_override(monkeypatch):
    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, "off")
    assert await _capabilities.is_mage_available(_WITH_MAGE) is False


@pytest.mark.parametrize(
    ("override", "expected_availability"),
    [
        ("on", True),
        ("off", False),
    ],
)
async def test_diagnostic_snapshot_override_skips_probe(
    monkeypatch, override, expected_availability
):
    """Operator overrides are resolved without touching Memgraph."""
    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, override)
    run = _patch_probe(monkeypatch, _WITH_MAGE)

    snapshot = await _capabilities.get_mage_capability_snapshot()

    assert snapshot.available is expected_availability
    assert snapshot.procedures is None
    run.assert_not_awaited()


async def test_diagnostic_snapshot_auto_uses_one_procedure_set(monkeypatch):
    run = _patch_probe(monkeypatch, _WITH_MAGE)

    snapshot = await _capabilities.get_mage_capability_snapshot()

    assert snapshot.available is True
    assert snapshot.procedures == frozenset(_WITH_MAGE)
    assert run.await_count == 1
