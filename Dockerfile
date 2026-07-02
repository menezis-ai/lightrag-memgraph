# syntax=docker/dockerfile:1.6

# BNP runtime image. The export branch commits the prebuilt Twin WebUI under
# src/twindb_lightrag_memgraph/webui_dist/, so this Dockerfile must not run Bun
# or perform editable package installation during the BNP build.
FROM fr2.icr.io/a100575-hprd/hkuds/lightrag:v1.4.9.11

COPY src /app/src
COPY src/twindb_lightrag_memgraph /app/twindb_lightrag_memgraph
COPY pyproject.toml README.md ENV_VARIABLES.txt /app/twindb_lightrag_memgraph_bundle/
COPY requirements /app/twindb_lightrag_memgraph_bundle/requirements
COPY config/build.conf /app/twindb_lightrag_memgraph_bundle/config/build.conf

RUN test -f /app/src/twindb_lightrag_memgraph/webui_dist/index.html \
 && test -f /app/twindb_lightrag_memgraph/webui_dist/index.html

WORKDIR /app

ENV PYTHONPATH=/app:/app/src:${PYTHONPATH}

# twindb_lightrag_memgraph.lightrag_server calls:
#   register(replace_ui=True, mount_server=True, shim_native_routes=True)
# before importing LightRAG's native server entrypoint.
ENTRYPOINT ["python", "-m", "twindb_lightrag_memgraph.lightrag_server"]
