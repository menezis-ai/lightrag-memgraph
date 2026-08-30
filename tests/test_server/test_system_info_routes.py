"""Contract tests for ``GET /twin/api/system/about``.

Covers the three properties the route exists for:

* the two-tier payload (reduced for any caller served by the backend, full for
  admin),
* the LightRAG native/composite version split around ``register()``'s marker,
* fail-soft behaviour — a Memgraph that is down must degrade to
  ``reachable: false``, never 500. That is the case the panel is read in.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import system_info_routes
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp

API_KEY = "about-test-key"


@pytest.fixture(autouse=True)
def _auth_state():
    """Pin the auth backend explicitly.

    ``require_auth`` only allows anonymous when NO auth backend is
    configured, and that configuration is process-global. Relying on the
    ambient default made these tests pass alone and 401 inside the full
    suite, where an earlier module had already configured auth. Set it
    here, restore it after — the reason stays next to the test.
    """
    configure_auth(api_key=API_KEY, jwt_secret=None)
    configure_idp(None)
    yield
    configure_auth(api_key=None, jwt_secret=None)
    configure_idp(None)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(system_info_routes.router, prefix="/twin/api")
    return app


async def _get(app: FastAPI, *, anonymous: bool = False):
    headers = {} if anonymous else {"Authorization": f"Bearer {API_KEY}"}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/twin/api/system/about", headers=headers)


@pytest.fixture
def non_admin(monkeypatch):
    monkeypatch.setattr(system_info_routes, "_is_admin", lambda request: False)


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(system_info_routes, "_is_admin", lambda request: True)


# --------------------------------------------------------------------------
# The route is gated
# --------------------------------------------------------------------------


async def test_anonymous_rejected_when_an_auth_backend_is_configured():
    resp = await _get(_app(), anonymous=True)
    assert resp.status_code == 401


async def test_open_mode_anonymous_gets_versions_but_never_topology():
    """Open access is deliberate LightRAG parity — do NOT gate it here.

    ``require_auth`` allows anonymous when no auth backend is configured
    (server/auth.py). Diverging from that is what crash-looped BNP on
    2026-06-10, so this route must not add its own boot-time refusal. The
    property that actually matters is the tier split holding: an anonymous
    caller in open mode is not admin, so it gets versions and NO deployment
    topology.
    """
    configure_auth(api_key=None, jwt_secret=None)
    configure_idp(None)
    resp = await _get(_app(), anonymous=True)

    assert resp.status_code == 200
    body = resp.json()
    assert body["admin"] is False
    assert body["twin"]
    assert body["memgraph"] is None
    assert body["storage"] is None
    assert body["runtime"] is None
    assert body["overlay"] is None


# --------------------------------------------------------------------------
# Tier split
# --------------------------------------------------------------------------


async def test_non_admin_gets_versions_only(non_admin):
    resp = await _get(_app())
    assert resp.status_code == 200
    body = resp.json()

    assert body["admin"] is False
    assert body["twin"]
    assert "lightrag" in body
    # The deployment shape must not leak to a non-admin caller.
    assert body["memgraph"] is None
    assert body["runtime"] is None
    assert body["storage"] is None
    assert body["overlay"] is None
    assert body["limits"] is None


async def test_admin_gets_full_payload(admin, monkeypatch):
    monkeypatch.setattr(
        system_info_routes,
        "_memgraph_info",
        _fake_memgraph(reachable=True, version="3.9.0"),
    )
    resp = await _get(_app())
    assert resp.status_code == 200
    body = resp.json()

    assert body["admin"] is True
    assert body["memgraph"]["reachable"] is True
    assert body["memgraph"]["version"] == "3.9.0"
    assert body["runtime"]["python"]
    assert body["runtime"]["platform"]
    assert isinstance(body["storage"], dict)
    assert set(body["overlay"]) == {
        "replace_ui",
        "mount_server",
        "shim_native_routes",
    }
    assert body["limits"] == {"vector_index_capacity": 100_000}


async def test_admin_limits_follow_the_configured_capacity(admin, monkeypatch):
    monkeypatch.setenv("TWIN_VECTOR_INDEX_CAPACITY", "250000")
    monkeypatch.setattr(
        system_info_routes,
        "_memgraph_info",
        _fake_memgraph(reachable=False),
    )
    body = (await _get(_app())).json()
    assert body["limits"] == {"vector_index_capacity": 250_000}


async def test_admin_limits_block_is_hidden_not_500_on_malformed_env(
    admin, monkeypatch
):
    """register() refuses to boot on this value; the About route, reached in
    a process that was configured after boot, must not turn it into a 500."""
    monkeypatch.setenv("TWIN_VECTOR_INDEX_CAPACITY", "abc")
    monkeypatch.setattr(
        system_info_routes,
        "_memgraph_info",
        _fake_memgraph(reachable=False),
    )
    resp = await _get(_app())
    assert resp.status_code == 200
    assert resp.json()["limits"] is None


def _fake_memgraph(**kwargs):
    async def _probe():
        return system_info_routes.MemgraphInfo(**kwargs)

    return _probe


# --------------------------------------------------------------------------
# LightRAG version split
# --------------------------------------------------------------------------


def test_version_split_recovers_native_from_composite(monkeypatch):
    import lightrag

    monkeypatch.setattr(lightrag, "__version__", "1.4.9.11+memgraph-1.1.0")
    versions = system_info_routes._split_lightrag_version()
    assert versions.native == "1.4.9.11"
    assert versions.composite == "1.4.9.11+memgraph-1.1.0"


def test_version_split_without_marker_is_native_only(monkeypatch):
    """register() not run (or a future upstream that drops the marker)."""
    import lightrag

    monkeypatch.setattr(lightrag, "__version__", "1.5.4")
    versions = system_info_routes._split_lightrag_version()
    assert versions.native == "1.5.4"
    assert versions.composite is None


def test_version_split_tolerates_missing_attribute(monkeypatch):
    import lightrag

    monkeypatch.delattr(lightrag, "__version__", raising=False)
    versions = system_info_routes._split_lightrag_version()
    assert versions.native is None
    assert versions.composite is None


# --------------------------------------------------------------------------
# Fail-soft — the reason this route exists
# --------------------------------------------------------------------------


async def test_memgraph_down_degrades_instead_of_500(admin, monkeypatch):
    """A dead Memgraph is exactly when the operator opens this panel."""

    def _boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(
        "twindb_lightrag_memgraph._pool.get_read_session",
        _boom,
    )
    resp = await _get(_app())
    assert resp.status_code == 200
    body = resp.json()
    assert body["memgraph"]["reachable"] is False
    assert body["memgraph"]["version"] is None
    assert body["memgraph"]["error"] == "OSError"


async def test_capability_snapshot_is_atomic(monkeypatch):
    """procedures and mage must describe the SAME probe.

    ``get_available_procedures()`` fails closed to an empty set and does NOT
    cache that failure, so two consecutive calls can straddle a reconnection.
    Probing twice therefore published ``procedures=0`` beside ``mage=True`` —
    a self-contradicting diagnostic in the panel opened to be told the truth.
    This fixture reproduces exactly that interleaving.
    """
    from twindb_lightrag_memgraph import _capabilities

    calls = {"n": 0}
    mage_set = frozenset({"pagerank.get", "community_detection.get"})

    async def _flaky_then_healthy(**kwargs):
        calls["n"] += 1
        return frozenset() if calls["n"] == 1 else mage_set

    monkeypatch.setattr(_capabilities, "get_available_procedures", _flaky_then_healthy)
    monkeypatch.setattr(
        system_info_routes, "_memgraph_version_probe", _fake_version("3.12.0")
    )
    info = await system_info_routes._memgraph_info()

    assert info.reachable is True
    # The failed probe is reported as unknown on BOTH fields, never as a
    # resolved tier standing next to a zero count.
    assert info.procedures is None
    assert info.mage is None
    assert calls["n"] == 1


async def test_unreachable_memgraph_reports_mage_unknown_not_absent(monkeypatch):
    """Absence of evidence is not evidence of the floor tier."""

    def _boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("twindb_lightrag_memgraph._pool.get_read_session", _boom)
    info = await system_info_routes._memgraph_info()

    assert info.reachable is False
    assert info.mage is None
    assert info.procedures is None


async def test_mage_resolves_on_a_healthy_probe(monkeypatch):
    from twindb_lightrag_memgraph import _capabilities

    async def _core_only(**kwargs):
        return frozenset({"vector_search.search", "mg.procedures", "mg.functions"})

    monkeypatch.setattr(_capabilities, "get_available_procedures", _core_only)
    monkeypatch.setattr(
        system_info_routes, "_memgraph_version_probe", _fake_version("3.12.0")
    )
    info = await system_info_routes._memgraph_info()

    # Core procedures ARE exposed by the base image — a resolved False, not
    # unknown, and a real count beside it.
    assert info.mage is False
    assert info.procedures == 3


@pytest.mark.parametrize(
    ("override", "expected_availability"),
    [
        ("on", True),
        ("off", False),
    ],
)
async def test_mage_override_is_reported_without_a_procedure_probe(
    monkeypatch, override, expected_availability
):
    """The diagnostic route must preserve the override's skip-probe contract."""
    from twindb_lightrag_memgraph import _capabilities

    calls = {"n": 0}

    async def _unexpected_probe(**kwargs):
        calls["n"] += 1
        return frozenset({"pagerank.get"})

    monkeypatch.setenv(_capabilities.TWIN_MAGE_ENV, override)
    monkeypatch.setattr(_capabilities, "get_available_procedures", _unexpected_probe)
    monkeypatch.setattr(
        system_info_routes, "_memgraph_version_probe", _fake_version("3.12.0")
    )

    info = await system_info_routes._memgraph_info()

    assert info.reachable is True
    assert info.mage is expected_availability
    assert info.procedures is None
    assert calls["n"] == 0


def _fake_version(version: str | None):
    async def _probe() -> str | None:
        return version

    return _probe


async def test_admin_probe_failure_falls_back_to_non_admin(monkeypatch):
    """An IdP fault must reduce the payload, never 500 the route."""

    def _boom(request):
        raise RuntimeError("jwks unreachable")

    monkeypatch.setattr(system_info_routes, "require_admin_user", _boom)
    resp = await _get(_app())
    assert resp.status_code == 200
    assert resp.json()["admin"] is False


def test_storage_classes_empty_when_rag_not_captured():
    assert system_info_routes._storage_classes(None) == {}


def test_storage_classes_read_from_the_injected_provider():
    """The factory's provider is what must be read, not _twindb_state.

    The standalone factory owns ``server.app._rag`` and never populates
    ``registry._twindb_state``; reading only the latter reported an empty
    topology on that path.
    """

    class _Chunks: ...

    class _Rag:
        text_chunks = _Chunks()

    assert system_info_routes._storage_classes(_Rag())["kv"] == "_Chunks"


async def test_router_factory_uses_the_supplied_rag(admin, monkeypatch):
    class _Vec: ...

    class _Rag:
        chunks_vdb = _Vec()

    monkeypatch.setattr(
        system_info_routes,
        "_memgraph_info",
        _fake_memgraph(reachable=True, version="3.9.0"),
    )
    app = FastAPI()
    app.include_router(
        system_info_routes.build_system_info_router(lambda: _Rag()),
        prefix="/twin/api",
    )
    resp = await _get(app)
    assert resp.json()["storage"] == {"vector": "_Vec"}


# --------------------------------------------------------------------------
# Overlay flags follow register()'s resolution, not the raw env
# --------------------------------------------------------------------------


def test_overlay_flags_prefer_the_flags_register_resolved(monkeypatch):
    """`register(mount_server=True)` with no env var must not read as off."""
    from twindb_lightrag_memgraph.patches import registry

    monkeypatch.delenv("TWIN_MOUNT_SERVER", raising=False)
    monkeypatch.setitem(
        registry._twindb_state,
        "overlay_flags",
        {"replace_ui": False, "mount_server": True, "shim_native_routes": False},
    )
    assert system_info_routes._overlay_flags()["mount_server"] is True


def test_overlay_flags_fall_back_to_env_before_register_runs(monkeypatch):
    from twindb_lightrag_memgraph.patches import registry

    # delitem, not pop(): a bare pop() leaves the process without the key a
    # later test may rely on, which is exactly the order-dependence the
    # conftest net exists to catch rather than to absorb.
    monkeypatch.delitem(registry._twindb_state, "overlay_flags", raising=False)
    monkeypatch.setenv("TWIN_REPLACE_UI", "1")
    monkeypatch.delenv("TWIN_MOUNT_SERVER", raising=False)
    flags = system_info_routes._overlay_flags()
    assert flags["replace_ui"] is True
    assert flags["mount_server"] is False
