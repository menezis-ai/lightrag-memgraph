"""
Shared fixtures for twindb-lightrag-memgraph tests.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a running Memgraph instance"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless MEMGRAPH_URI is set."""
    if os.environ.get("MEMGRAPH_URI"):
        return

    skip = pytest.mark.skip(reason="MEMGRAPH_URI not set, skipping integration test")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


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
