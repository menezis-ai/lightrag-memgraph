"""Contract of the Gunicorn/uvicorn factory entrypoint (``asgi.py``).

Every worker process imports this module fresh, and a ``register()`` that did
not run there would serve LightRAG **unpatched** — native WebUI, no overlay,
no Memgraph storage slots — while answering 200. The module is four
statements whose whole value is their ORDER and their failure mode, so that
is what these tests pin.

Order is asserted by OBSERVING the upstream import, not by inferring it from
the end state: swapping the two statements in ``asgi.py`` leaves ``register``
called once, the factory identity unchanged and the failed-import cleanup
identical, so an end-state assertion cannot tell the correct module from the
one that boots an unpatched app. A ``builtins.__import__`` spy can.

The module is imported here with a stand-in ``register`` so no test process
ever pays the real global patch; ``sys.modules`` surgery keeps the import
repeatable.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

import twindb_lightrag_memgraph

MODULE = "twindb_lightrag_memgraph.asgi"
UPSTREAM = "lightrag.api.lightrag_server"

REGISTER = "register"
IMPORT_UPSTREAM = "import:lightrag-server"


@pytest.fixture
def fresh_asgi(monkeypatch):
    """Import ``asgi`` under a recorded ``register`` and an import spy.

    Returns ``(import_module, events)``: calling the first with a register
    stand-in imports the module and appends, in real execution order, a
    ``REGISTER`` entry per call and an ``IMPORT_UPSTREAM`` entry each time the
    LightRAG server module is imported.
    """
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    # ``asgi`` pulls LightRAG's API package, which parses process argv at
    # import time. Present the argv its real console entrypoint receives
    # instead of pytest's flags, rather than relying on whichever earlier
    # test happened to initialize ``global_args`` (house idiom, see
    # tests/test_server/test_native_query_guards.py).
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    events: list[str] = []
    real_import = builtins.__import__

    def spy(name, globals=None, locals=None, fromlist=(), level=0):
        # A cached module still goes through __import__, so the event fires
        # whether or not an earlier test already imported it.
        if name == UPSTREAM or name.startswith(UPSTREAM + "."):
            events.append(IMPORT_UPSTREAM)
        return real_import(name, globals, locals, fromlist, level)

    def _import(register):
        def recorded(*args, **kwargs):
            events.append(REGISTER)
            return register(*args, **kwargs)

        monkeypatch.setattr(twindb_lightrag_memgraph, "register", recorded)
        monkeypatch.setattr(builtins, "__import__", spy)
        try:
            return importlib.import_module(MODULE)
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)
            monkeypatch.delitem(sys.modules, MODULE, raising=False)

    return _import, events


def test_registration_happens_before_the_upstream_server_is_imported(fresh_asgi):
    """The load-bearing invariant. Importing LightRAG's server module first
    would build the app off the unpatched storage/UI registry — and would be
    invisible to every end-state assertion."""
    import_module, events = fresh_asgi
    calls = []
    import_module(lambda *a, **kw: calls.append((a, kw)))

    assert REGISTER in events, "register() never ran at import time"
    assert IMPORT_UPSTREAM in events, "the upstream server module was never imported"
    assert events.index(REGISTER) < events.index(
        IMPORT_UPSTREAM
    ), f"asgi.py imported {UPSTREAM} before register(): {events}"
    assert calls == [((), {})], "register() must run exactly once"


def test_the_factory_is_re_exported_from_the_module_it_imported(fresh_asgi):
    import_module, _ = fresh_asgi
    module = import_module(lambda *a, **kw: None)

    assert module.__all__ == ["get_application"]
    # Compare against the module the import itself loaded: re-importing it
    # here would be a second chance to initialize LightRAG's config and would
    # hide an ordering bug rather than surface it.
    upstream = sys.modules[UPSTREAM]
    assert module.get_application is upstream.get_application


def test_overlay_activation_is_left_to_the_environment(fresh_asgi):
    """Deliberate asymmetry with ``lightrag_server.main()``, which forces
    ``replace_ui``/``mount_server``/``shim_native_routes`` to True. The ASGI
    entrypoint passes NO keyword so the deployment's env vars decide
    (AGENTS.md: the two entrypoints are parallel but not identical). Asserting
    it here means a copy-paste of the explicit flag set gets caught."""
    import_module, _ = fresh_asgi
    calls = []
    import_module(lambda *a, **kw: calls.append((a, kw)))

    ((_, kwargs),) = calls
    assert kwargs == {}


def test_a_failed_registration_aborts_before_the_upstream_import(fresh_asgi):
    """Fail closed: a worker that could not patch LightRAG must die at import
    rather than come up serving an unpatched app. The upstream server module
    must never be reached once register() has raised."""
    import_module, events = fresh_asgi

    def _boom():
        raise RuntimeError("storage slots unavailable")

    with pytest.raises(RuntimeError, match="storage slots unavailable"):
        import_module(_boom)

    assert events == [
        REGISTER
    ], f"asgi.py touched {UPSTREAM} despite a failed registration: {events}"
    assert MODULE not in sys.modules
