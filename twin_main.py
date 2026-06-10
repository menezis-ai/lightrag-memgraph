"""Twin entrypoint — installs the Memgraph patch then runs LightRAG.

Doctrine: ``twindb_lightrag_memgraph.register()`` MUST run *before* the
LightRAG server module is imported. Trying to call register() from
inside ``lightrag.api.lightrag_server`` (e.g. via a sed-prepended
import) creates a circular import — register() patches
``lightrag.api.lightrag_server.create_app`` but the module is still
mid-import at that point and ``create_app`` does not yet exist.

This entrypoint solves it by running register() at module top, then
importing the LightRAG server's ``main`` and invoking it. Use it as
your container CMD instead of ``python -m lightrag.api.lightrag_server``.

Flags passed to register():

- ``replace_ui=True``  — swap the bundled LightRAG /webui Mount with
  the Twin React WebUI fork (requires ``src/twindb_lightrag_memgraph/
  webui_dist/`` to be present in the installed package).
- ``mount_server=True`` — mount the Twin overlay under ``/twin/api/*``
  (folders, tags, activity, notifications, graph CRUD, query overlay).
- ``shim_native_routes=True`` — shadow LightRAG's native /documents,
  /health, /pipeline_status, /auth-status, /login, /logout, /openapi
  routes with the Twin-shaped contract.
"""

from twindb_lightrag_memgraph import register

register(
    replace_ui=True,
    mount_server=True,
    shim_native_routes=True,
)

from lightrag.api.lightrag_server import main

if __name__ == "__main__":
    main()
