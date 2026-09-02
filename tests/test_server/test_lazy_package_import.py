"""Server package imports must not pull standalone-only settings eagerly."""

from __future__ import annotations

import importlib
import sys


class _BlockPydanticSettings:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pydantic_settings":
            raise ModuleNotFoundError("No module named 'pydantic_settings'")
        return None


def test_importing_auth_does_not_require_pydantic_settings(monkeypatch):
    for name in list(sys.modules):
        if name == "pydantic_settings" or name.startswith(
            "twindb_lightrag_memgraph.server"
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    blocker = _BlockPydanticSettings()
    sys.meta_path.insert(0, blocker)
    try:
        auth = importlib.import_module("twindb_lightrag_memgraph.server.auth")
    finally:
        sys.meta_path.remove(blocker)

    assert hasattr(auth, "require_auth")


def test_the_lazy_getattr_resolves_every_documented_name():
    """``server/__init__`` exports its three public names through
    ``__getattr__`` so ``pydantic-settings`` stays optional. The branches are
    only taken by ``from ...server import <name>`` — the suite otherwise
    imports the submodules directly, which is why they went unexercised."""
    server = importlib.import_module("twindb_lightrag_memgraph.server")
    from twindb_lightrag_memgraph.server.app import create_app
    from twindb_lightrag_memgraph.server.settings import (
        LightRAGServerSettings,
        get_settings,
    )

    assert server.create_app is create_app
    assert server.LightRAGServerSettings is LightRAGServerSettings
    assert server.get_settings is get_settings
    assert set(server.__all__) == {
        "create_app",
        "LightRAGServerSettings",
        "get_settings",
    }


def test_the_lazy_getattr_falls_back_to_submodules():
    """``mock.patch("...server.tracing.<attr>")`` resolves the submodule
    through the same ``__getattr__``; it must not raise for a real one."""
    server = importlib.import_module("twindb_lightrag_memgraph.server")
    monkeypatched_target = getattr(server, "tracing")

    assert monkeypatched_target is importlib.import_module(
        "twindb_lightrag_memgraph.server.tracing"
    )


def test_an_unknown_name_is_an_attribute_error_not_an_import_error():
    """A typo must surface as AttributeError — ``hasattr`` and
    ``inspect`` probes swallow that, an ImportError escapes them."""
    import pytest

    server = importlib.import_module("twindb_lightrag_memgraph.server")
    with pytest.raises(AttributeError, match="no attribute 'not_a_submodule'"):
        server.not_a_submodule
