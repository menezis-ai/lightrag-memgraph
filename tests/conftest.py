"""
Shared fixtures for twindb-lightrag-memgraph tests.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a running Memgraph instance"
    )
    config.addinivalue_line(
        "markers", "perf: performance-regression benchmark (set RUN_PERF to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless MEMGRAPH_URI is set, and perf benchmarks
    unless RUN_PERF is set. Perf runs off by default so the standard unit run
    stays fast; CI turns it on in the dedicated (non-blocking) perf-tests job."""
    has_memgraph = bool(os.environ.get("MEMGRAPH_URI"))
    run_perf = bool(os.environ.get("RUN_PERF"))
    skip_integration = pytest.mark.skip(
        reason="MEMGRAPH_URI not set, skipping integration test"
    )
    skip_perf = pytest.mark.skip(reason="RUN_PERF not set, skipping perf benchmark")
    for item in items:
        if not has_memgraph and "integration" in item.keywords:
            item.add_marker(skip_integration)
        if not run_perf and "perf" in item.keywords:
            item.add_marker(skip_perf)


def ensure_fresh_native_document_router():
    """Rebind a fresh ``document_routes.router`` before building a native app.

    LightRAG 1.4.x declares ``router = APIRouter(...)`` at module level
    (1.4.9.11 ``document_routes.py:79``) and ``create_document_routes``
    re-decorates that shared object on every call (``return router`` at
    ``:3194``). FastAPI matches the FIRST registration, so the second
    native app built in a pytest process silently serves the first app's
    ``rag``/``doc_manager`` closures — whose tmp dirs are already torn
    down. 1.5.x builds the router inside the factory and is unaffected.

    Call this immediately before ``create_document_routes(...)`` in any
    harness that constructs a native LightRAG app.
    """
    import lightrag.api.routers.document_routes as dr

    old = getattr(dr, "router", None)
    if old is None:
        return  # 1.5.x: factory-local router, nothing shared
    from fastapi import APIRouter

    dr.router = APIRouter(prefix=old.prefix, tags=list(old.tags or []))
