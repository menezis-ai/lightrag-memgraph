"""ASGI factory entrypoint for Gunicorn / uvicorn worker deployments.

When the server runs under Gunicorn (or any launcher that imports the
application through an import string), each worker process imports the
app module fresh — ``register()`` from a separate boot script never runs
there, so the native LightRAG app (and its native WebUI) would be served
unpatched.

This module guarantees the patch by running ``register()`` at import
time, in every worker, BEFORE LightRAG's server module is touched
(importing it earlier creates a circular import — see ``twin_main.py``).
Overlay activation is driven by environment variables:

    TWIN_REPLACE_UI=true          swap /webui for the embedded Twin WebUI
    TWIN_MOUNT_SERVER=true        mount the /twin/api/* overlay
    TWIN_SHIM_NATIVE_ROUTES=true  shadow native routes with the Twin contract

Point the launcher at this module instead of LightRAG's:

    gunicorn 'twindb_lightrag_memgraph.asgi:get_application()' \
        -k uvicorn.workers.UvicornWorker
    # or
    uvicorn --factory twindb_lightrag_memgraph.asgi:get_application
"""

from twindb_lightrag_memgraph import register

register()

from lightrag.api.lightrag_server import get_application  # noqa: E402

__all__ = ["get_application"]
