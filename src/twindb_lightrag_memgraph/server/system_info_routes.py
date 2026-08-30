"""``GET /twin/api/system/about`` — runtime identity card for debugging.

Two-tier payload (product decision 2026-07-27):

* **Non-admin caller** sees the software identity only: the Twin package
  version and the LightRAG version it runs against.
* **Admin** additionally sees the deployment shape — Memgraph server version,
  MAGE tier, Python/platform, the storage backend classes actually bound, and
  the overlay activation flags.

The split exists because topology is reconnaissance material at BNP; the
reduced tier still answers "which build am I looking at?", which is the
question a non-admin operator actually needs when filing a ticket.

**The tier split — not the route gate — is what protects the topology.** The
route sits behind ``require_auth`` like the rest of the Twin surface, and that
dependency is LightRAG-parity: it rejects anonymous callers only when an auth
backend is actually configured. On an instance with no auth configured, an
anonymous caller therefore reaches this route and gets the reduced tier,
exactly as it reaches ``/twin/api/documents``. That is deliberate — adding a
route-local refusal here is the divergence that crash-looped BNP on
2026-06-10 (see the Auth posture section of ``CLAUDE.md``). Never describe this
route as "authenticated callers only"; describe it as "topology is admin-only".

**This route never raises.** It is read precisely when something is already
broken, so a Memgraph that is down must yield ``{"reachable": false}`` rather
than a 500 that hides the information the operator came for.

Two upstream traps, both confirmed against a live Memgraph on 2026-07-27 —
do not "simplify" either one away:

* ``SHOW VERSION`` is refused inside an explicit transaction. Memgraph answers
  *"Version information query is not allowed in multicommand transactions"*.
  ``driver.execute_query()`` opens one, so the query MUST run as an autocommit
  ``session.run()`` on a read session.
* ``driver.get_server_info().agent`` reports ``"Neo4j/v5.11.0 compatible graph
  database server - Memgraph"``. That is the Bolt compatibility level, **not**
  the Memgraph version. It looks like an answer and is not one.
"""

from __future__ import annotations

import logging
import platform
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .. import __version__ as TWIN_VERSION
from .auth import require_auth
from .idp_jwt import require_admin_user

logger = logging.getLogger(__name__)

# ``register()`` appends this marker to ``lightrag.__version__`` so the WebUI
# shows the composite build (see patches/registry.py). Splitting on it recovers
# the native LightRAG version without reaching into the registry's internals,
# and degrades correctly when register() has not run (no marker -> version
# as-is, composite None).
_MEMGRAPH_VERSION_MARKER = "+memgraph-"


class LightRagVersions(BaseModel):
    native: str | None = None
    composite: str | None = None


class MemgraphInfo(BaseModel):
    reachable: bool = False
    version: str | None = None
    #: Tri-state: True / False / None = "could not be determined". An
    #: unreachable Memgraph or a failed capability probe is an absence of
    #: evidence, not evidence of absence — reporting it as False would show
    #: "floor tier" in the panel and send the operator down the wrong path.
    mage: bool | None = None
    procedures: int | None = None
    error: str | None = None


class RuntimeInfo(BaseModel):
    python: str
    implementation: str
    platform: str


class AboutResponse(BaseModel):
    twin: str
    lightrag: LightRagVersions
    admin: bool = False
    # Admin-only blocks; absent (None) for a non-admin caller.
    memgraph: MemgraphInfo | None = None
    runtime: RuntimeInfo | None = None
    storage: dict[str, str] | None = None
    overlay: dict[str, bool] | None = None
    #: Configured storage limits (admin-only). ``vector_index_capacity`` is
    #: the ``CREATE VECTOR INDEX`` capacity this runtime would use — an
    #: existing index keeps the capacity it was created with.
    limits: dict[str, int] | None = None


def _default_get_rag() -> Any:
    """Overlay default: the instance ``register()`` captured."""
    from .. import _twindb_state

    return _twindb_state.get("rag")


def _split_lightrag_version() -> LightRagVersions:
    """Recover the native LightRAG version from the possibly-patched string."""
    try:
        import lightrag

        raw = getattr(lightrag, "__version__", None)
    except Exception:  # pragma: no cover - lightrag absent is not a 500
        logger.debug("about: lightrag import failed", exc_info=True)
        return LightRagVersions()
    if not isinstance(raw, str) or not raw:
        return LightRagVersions()
    if _MEMGRAPH_VERSION_MARKER in raw:
        native, _, _ = raw.partition(_MEMGRAPH_VERSION_MARKER)
        return LightRagVersions(native=native, composite=raw)
    return LightRagVersions(native=raw, composite=None)


async def _memgraph_version_probe() -> str | None:
    """Return the server version, or None when the query yields nothing.

    Raises on transport failure — the caller turns that into ``reachable:
    false``. Split out from :func:`_memgraph_info` so the capability half can
    be tested without a live Memgraph.
    """
    from .._pool import get_read_session

    async with get_read_session() as session:
        # Autocommit run — an explicit transaction makes Memgraph refuse
        # SHOW VERSION outright. See the module docstring.
        result = await session.run("SHOW VERSION")
        rows = [record async for record in result]
        await result.consume()
    if rows:
        value = rows[0].get("version")
        if value is not None:
            return str(value)
    return None


async def _memgraph_info() -> MemgraphInfo:
    """Probe Memgraph, keeping capability failures distinct from transport ones.

    A failed version/transport probe degrades to ``reachable=False``. Once the
    server is known to be reachable, a failed MAGE capability probe leaves the
    tier and procedure count unknown instead of changing reachability.
    """
    try:
        version = await _memgraph_version_probe()
    except Exception as exc:
        logger.warning("about: Memgraph version probe failed: %s", exc)
        return MemgraphInfo(reachable=False, error=type(exc).__name__)

    info = MemgraphInfo(reachable=True, version=version)

    # MAGE tier is additive and must never turn this route red. Two properties
    # this block has to hold, both learned the hard way:
    #
    # * `mage` comes from the marker-based predicate, NOT from "the probe
    #   returned rows": the base image already exposes core procedures, so a
    #   truthiness test would report MAGE everywhere.
    # * ONE snapshot backs both fields. The snapshot also resolves TWIN_MAGE
    #   before probing, so an explicit on/off override remains authoritative
    #   and keeps its documented "skip the probe" behaviour.
    try:
        from .._capabilities import get_mage_capability_snapshot

        snapshot = await get_mage_capability_snapshot()
        info.mage = snapshot.available
        if snapshot.procedures is not None:
            info.procedures = len(snapshot.procedures)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("about: MAGE probe failed: %s", exc)
    return info


def _storage_classes(rag: Any) -> dict[str, str]:
    """Class names of the storage backends actually bound on the live RAG."""
    if rag is None:
        return {}
    slots = {
        "kv": "text_chunks",
        "vector": "chunks_vdb",
        "docstatus": "doc_status",
        "graph": "chunk_entity_relation_graph",
    }
    bound: dict[str, str] = {}
    for label, attr in slots.items():
        target = getattr(rag, attr, None)
        if target is not None:
            bound[label] = type(target).__name__
    return bound


def _overlay_flags() -> dict[str, bool]:
    """Report the overlay activation ``register()`` actually resolved.

    Reads the flags register() published in ``_twindb_state`` rather than
    re-reading the env: explicit booleans win over env there, so a host that
    calls ``register(mount_server=True)`` without the matching env var would
    otherwise be reported as fully disabled. Falls back to the env only when
    register() has not run (standalone factory, tests).
    """
    from .. import _twindb_state

    resolved = _twindb_state.get("overlay_flags")
    if isinstance(resolved, dict):
        return {k: bool(v) for k, v in resolved.items()}

    from ..patches.registry import _env_flag

    return {
        "replace_ui": _env_flag("TWIN_REPLACE_UI"),
        "mount_server": _env_flag("TWIN_MOUNT_SERVER"),
        "shim_native_routes": _env_flag("TWIN_SHIM_NATIVE_ROUTES"),
    }


def _limits() -> dict[str, int] | None:
    """Configured storage limits. A malformed env value is a boot-time error
    elsewhere (``register()``); here it only hides the block."""
    from .._constants import resolve_vector_index_capacity

    try:
        return {"vector_index_capacity": resolve_vector_index_capacity()}
    except ValueError:
        logger.debug("about: vector index capacity unreadable", exc_info=True)
        return None


def _is_admin(request: Request) -> bool:
    """Non-gating admin probe: reuse the dependency, swallow its refusal."""
    try:
        require_admin_user(request)
    except HTTPException:
        return False
    except Exception:  # pragma: no cover - an IdP fault must not 500 this route
        logger.debug("about: admin probe failed", exc_info=True)
        return False
    return True


def build_system_info_router(get_rag: Any = None) -> APIRouter:
    """Build the About router against a caller-supplied rag provider.

    The two mount points resolve the live instance differently — the overlay
    reads ``_twindb_state["rag"]`` (captured by ``register()``), the standalone
    factory owns ``server.app._rag``. Binding the provider here keeps the panel
    from reporting an empty storage topology on the factory path.
    """
    provider = get_rag or _default_get_rag

    router = APIRouter(
        prefix="/system",
        tags=["system"],
        dependencies=[Depends(require_auth)],
    )

    @router.get(
        "/about",
        response_model=AboutResponse,
        summary="Runtime identity card",
    )
    async def get_about(request: Request) -> dict[str, Any]:
        """Return the software versions of the runtime (package and
        retrieval engine). Administrators additionally receive the
        deployment shape: database version and reachability, storage
        bindings and active overlay flags (`admin: true` marks the
        extended payload). Never fails on a down database — reachability
        is reported in the payload instead."""
        payload: dict[str, Any] = {
            "twin": TWIN_VERSION,
            "lightrag": _split_lightrag_version().model_dump(),
            "admin": False,
        }
        if not _is_admin(request):
            return payload

        try:
            rag = provider()
        except Exception:  # pragma: no cover - a 503-raising provider is fine
            rag = None

        payload["admin"] = True
        payload["memgraph"] = (await _memgraph_info()).model_dump()
        payload["runtime"] = RuntimeInfo(
            python=platform.python_version(),
            implementation=platform.python_implementation(),
            platform=platform.platform(),
        ).model_dump()
        payload["storage"] = _storage_classes(rag)
        payload["overlay"] = _overlay_flags()
        payload["limits"] = _limits()
        return payload

    return router


#: Overlay-default router (resolves the rag from ``_twindb_state``).
router = build_system_info_router()

__all__ = ["build_system_info_router", "router"]
