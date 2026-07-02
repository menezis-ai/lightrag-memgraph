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
