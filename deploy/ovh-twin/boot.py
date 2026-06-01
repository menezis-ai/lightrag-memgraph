"""Boot wrapper for the twin-real OVH deploy.

Calls :func:`twindb_lightrag_memgraph.register` with the server-mode
flags BEFORE importing :func:`lightrag.api.lightrag_server.main`. The
order is significant — ``register()`` monkey-patches
``lightrag_server.create_app`` and ``create_document_routes``, both of
which must be wrapped before LightRAG's ``main()`` resolves them.

Env vars consumed (in addition to the LightRAG-native ones):

  MEMGRAPH_URI       Bolt URI of the Memgraph service (compose)
  WORKSPACE          Twin workspace id (default: "default")
  TWIN_API_BASE_URL  Optional override for the WebUI's apiBaseUrl
  TWIN_LIGHTRAG_BASE_URL  Optional override for lightragBaseUrl
  TWIN_IDP_LOGOUT_URL     Optional IdP logout URL (debug placeholder
                          while JWT/IdP middleware is not yet shipped)
  TWIN_CATEGORIES_CONFIG  Optional path to a JSON taxonomy file
                          (cf. docs/templates/twin-categories.schema.json)
"""

import os
import sys

from twindb_lightrag_memgraph import register

register(
    shim_native_routes=True,
    replace_ui=True,
    mount_server=True,
    webui_stores="memgraph",
    webui_dist="/app/webui_dist",
    webui_categories_config=os.environ.get("TWIN_CATEGORIES_CONFIG") or None,
)

# Import + dispatch only AFTER register() has applied its patches.
from lightrag.api.lightrag_server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main() or 0)
