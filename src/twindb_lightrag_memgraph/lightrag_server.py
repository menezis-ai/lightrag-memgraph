"""Explicit TwinRAG entrypoint for LightRAG container deployments.

Do not rely on ``sitecustomize`` for production activation. Some launchers and
``python -m`` execution paths can import/execute the LightRAG server in a way
that bypasses a module patched by name. This entrypoint runs the Twin overlay
registration first, then delegates to LightRAG's own CLI main function in the
same imported module namespace.
"""

from __future__ import annotations

import importlib


def main() -> None:
    import twindb_lightrag_memgraph as twin

    twin.register(
        replace_ui=True,
        mount_server=True,
        shim_native_routes=True,
    )
    server = importlib.import_module("lightrag.api.lightrag_server")
    server.main()


if __name__ == "__main__":
    main()
