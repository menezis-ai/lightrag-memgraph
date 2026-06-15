"""twindb-lightrag-memgraph server subpackage.

L2 patch: FastAPI server layer on top of the L1 storage patch.

Usage (as module)::

    python -m twindb_lightrag_memgraph.server

Usage (programmatic)::

    from twindb_lightrag_memgraph.server import create_app
    app = create_app()
"""

__all__ = ["create_app", "LightRAGServerSettings", "get_settings"]


def __getattr__(name: str):
    """Load standalone-server helpers lazily.

    The production LightRAG overlay imports submodules such as
    ``server.auth``. Importing ``server.app`` here would also import
    ``server.settings`` and require ``pydantic-settings`` even when the
    standalone Twin server is not being used. BNP's base LightRAG image
    does not ship that optional dependency, so keep package import cheap
    and defer standalone-only imports until explicitly requested.
    """
    if name == "create_app":
        from .app import create_app

        return create_app
    if name in {"LightRAGServerSettings", "get_settings"}:
        from .settings import LightRAGServerSettings, get_settings

        return {
            "LightRAGServerSettings": LightRAGServerSettings,
            "get_settings": get_settings,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
