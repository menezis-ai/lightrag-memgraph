"""
Shared fixtures for twindb-lightrag-memgraph tests.
"""

import os
import tempfile

import pytest


def pytest_configure(config):
    # mutmut runs several pytest processes concurrently. Pytest's shared
    # ``pytest-current`` symlink cleanup has a TOCTOU race across those
    # processes (one unlinks it between another's exists/unlink calls), which
    # aborts an otherwise valid campaign with FileNotFoundError. An explicit
    # per-worker base temp keeps tmp_path isolation while avoiding that shared
    # cleanup namespace. Normal pytest runs retain their native temp policy.
    if os.environ.get("MUTANT_UNDER_TEST"):
        config.option.basetemp = os.path.join(
            tempfile.gettempdir(), f"twindb-mutmut-pytest-{os.getpid()}"
        )
    config.addinivalue_line(
        "markers", "integration: requires a running Memgraph instance"
    )
    config.addinivalue_line(
        "markers", "perf: performance-regression benchmark (set RUN_PERF to run)"
    )
    config.addinivalue_line(
        "markers",
        "live_vision: real RapidOCR/PDFium/OpenRouter acceptance test "
        "(set RUN_VISION_LIVE=1 to run)",
    )


def pytest_collection_modifyitems(config, items):
    """Keep external and expensive suites opt-in outside their dedicated CI."""
    has_memgraph = bool(os.environ.get("MEMGRAPH_URI"))
    run_perf = bool(os.environ.get("RUN_PERF"))
    run_live_vision = os.environ.get("RUN_VISION_LIVE") == "1"
    skip_integration = pytest.mark.skip(
        reason="MEMGRAPH_URI not set, skipping integration test"
    )
    skip_perf = pytest.mark.skip(reason="RUN_PERF not set, skipping perf benchmark")
    skip_live_vision = pytest.mark.skip(
        reason="RUN_VISION_LIVE=1 not set, skipping live OCR/Vision acceptance"
    )
    for item in items:
        if not has_memgraph and "integration" in item.keywords:
            item.add_marker(skip_integration)
        if not run_perf and "perf" in item.keywords:
            item.add_marker(skip_perf)
        if not run_live_vision and "live_vision" in item.keywords:
            item.add_marker(skip_live_vision)


_MISSING = object()


@pytest.fixture(autouse=True)
def _isolate_process_globals():
    """Net of last resort for process-global state that outlives a test.

    **This is a safety net, not the place to register new state.** A test that
    mutates a global must restore it itself -- ``monkeypatch.setattr`` does it
    for free and keeps the reason next to the test. This fixture exists because
    the four cases below already leaked across module boundaries and were
    silently changing later tests' outcomes; do not treat it as licence to skip
    local cleanup.

    Five values, four sources: a test that ran earlier used to change what a
    later one observed:

    * ``LightRAG._twin_classification_hook`` -- a class attribute on the
      upstream LightRAG class, set by ``install_classification_hook()``. While
      it is set, ``app._classification_guard_active()`` is True and raw-text
      ``POST /insert`` answers 400 by design, which broke the six
      ``test_app.py::TestInsertEndpoint`` / ``TestAuthDisabled`` cases.
    * ``utils_pipeline._DOC_STATUS_METADATA_CARRY_OVER_KEYS`` -- extended by
      the 1.5.x classification compatibility patch so security metadata
      survives pipeline transitions. The production mutation is intentional,
      but tests that install and uninstall the hook must not leak it.
    * ``server.tracing._TRACING_ENABLED`` -- module global flipped by
      ``configure_tracing()``; leaks into ``/health``'s ``tracing_enabled``.
    * ``classification._LABEL_NAMES`` -- module-global human-name cache filled
      by ``load_label_map()``; leaks into the ``class_name -> raw_name``
      fallback.

    Declaration order happened to keep the polluters after their victims, so
    the suite was green in-order and failed under any reordering (mutmut's
    stats-driven order, a reversed file list, pytest-randomly). Restoring
    rather than clearing keeps class- and module-scoped fixtures that set these
    on purpose working: whatever was in place when the test started is what it
    gets back.

    Project imports are direct and unguarded. ``lightrag`` is a hard
    dependency; ``classification`` and ``server.tracing`` are part of this
    package; ``server/__init__`` resolves its heavy names lazily through
    ``__getattr__``, and the ``test`` extra carries the metric client imported
    by ``tracing`` while missing LangSmith remains optional. Only
    ``lightrag.utils_pipeline`` is capability-
    guarded because the supported 1.4.x line does not expose that 1.5.x
    module.

    ``_MISSING`` distinguishes "attribute was absent" from "attribute was
    present and None" -- ``_classification_guard_active()`` tests
    ``is not None``, so restoring an absent attribute as None would be a
    different state, not a restoration.
    """
    saved = []

    from importlib import import_module

    from lightrag import LightRAG

    from twindb_lightrag_memgraph import classification as _classification
    from twindb_lightrag_memgraph.server import tracing as _tracing

    try:
        utils_pipeline = import_module("lightrag.utils_pipeline")
    except ModuleNotFoundError:
        utils_pipeline = None

    saved.append(
        (
            LightRAG,
            "_twin_classification_hook",
            getattr(LightRAG, "_twin_classification_hook", _MISSING),
        )
    )
    if utils_pipeline is not None:
        saved.append(
            (
                utils_pipeline,
                "_DOC_STATUS_METADATA_CARRY_OVER_KEYS",
                getattr(
                    utils_pipeline,
                    "_DOC_STATUS_METADATA_CARRY_OVER_KEYS",
                    _MISSING,
                ),
            )
        )
    saved.append((_tracing, "_TRACING_ENABLED", _tracing._TRACING_ENABLED))
    saved.append((_tracing, "_langsmith_available", _tracing._langsmith_available))

    label_names = _classification._LABEL_NAMES
    label_names_snapshot = dict(label_names)

    yield

    for owner, attr, value in saved:
        if value is _MISSING:
            # The attribute did not exist when the test started; setting it to
            # None would leave a different state behind, not the original one.
            if hasattr(owner, attr):
                delattr(owner, attr)
        else:
            setattr(owner, attr, value)

    label_names.clear()
    label_names.update(label_names_snapshot)


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


def seed_native_app_state(app):
    """Seed the app state the real LightRAG server's lifespan provides.

    LightRAG 1.5.5's upload/insert routes resolve their managed-task
    registry via ``request.app.state.background_tasks``
    (``get_managed_background_tasks``), seeded by ``lightrag_server.py``'s
    lifespan. A bare ``FastAPI()`` harness app has no lifespan, so every
    upload dies on ``AttributeError`` — call this right after constructing
    the app. No-op on already-seeded apps; harmless pre-1.5.5.
    """
    if not hasattr(app.state, "background_tasks"):
        app.state.background_tasks = set()
