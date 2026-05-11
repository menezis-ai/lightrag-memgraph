"""twindb-lightrag-memgraph server subpackage.

L2 patch: FastAPI server layer on top of the L1 storage patch.

Usage (as module)::

    python -m twindb_lightrag_memgraph.server

Usage (programmatic)::

    from twindb_lightrag_memgraph.server import create_app
    app = create_app()
"""

from .app import create_app
from .settings import LightRAGServerSettings, get_settings

__all__ = ["create_app", "LightRAGServerSettings", "get_settings"]
