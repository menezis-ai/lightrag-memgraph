"""Uvicorn entry point.

Usage::

    python -m twindb_lightrag_memgraph.server
    # or with custom port:
    LIGHTRAG_PORT=8080 python -m twindb_lightrag_memgraph.server
"""

import uvicorn

from .settings import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "twindb_lightrag_memgraph.server.app:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        log_level="info",
    )
