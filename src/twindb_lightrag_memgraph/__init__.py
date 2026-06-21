"""Public package entrypoint.

The LightRAG patch registry lives in :mod:`twindb_lightrag_memgraph.patches`.
This module preserves the historical root import surface, including private
helpers used by existing tests and downstream integrations.
"""

from __future__ import annotations

import inspect

from .patches import registry as _registry

globals().update(
    {
        name: getattr(_registry, name)
        for name in dir(_registry)
        if not name.startswith("__")
    }
)

_registered = _registry._registered


def _sync_registry_from_public_module() -> None:
    for name in dir(_registry):
        if name == "register" or name.startswith("__") or name not in globals():
            continue
        setattr(_registry, name, globals()[name])


def _sync_public_module_from_registry() -> None:
    globals().update(
        {
            name: getattr(_registry, name)
            for name in dir(_registry)
            if name != "register" and not name.startswith("__")
        }
    )


def register(*args, **kwargs) -> None:
    """Register TwinDB Memgraph patches through the split registry module."""
    global _registered
    _sync_registry_from_public_module()
    _registry.register(*args, **kwargs)
    _sync_public_module_from_registry()


__version__ = _registry.__version__
register.__signature__ = inspect.signature(_registry.register)
__all__ = [
    "register",
    "register_post_index_hook",
    "clear_post_index_hooks",
]
