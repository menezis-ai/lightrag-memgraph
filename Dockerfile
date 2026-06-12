# syntax=docker/dockerfile:1.6

# Stage 1: build the Twin WebUI fork dist/.
# scripts/build_webui.sh is the dev/Mac equivalent; here we inline the same
# steps so the image build is self-contained and never depends on a host
# toolchain having run beforehand.
FROM oven/bun:1.3.6 AS webui-builder
WORKDIR /webui
COPY lightrag_webui_twin/package.json lightrag_webui_twin/bun.lock ./
RUN bun install --frozen-lockfile
COPY lightrag_webui_twin/ ./
RUN bun run build \
 && test -f dist/index.html \
 && rm -f dist/mockServiceWorker.js

# Stage 2: runtime image (BNP base).
FROM fr2.icr.io/a100575-hprd/hkuds/lightrag:v1.4.9.11

# pip conf
ENV PIP_CONFIG_FILE=/tmp/pip.conf
COPY config/build.conf /tmp/pip.conf
COPY pyproject.toml /app/twindb_lightrag_memgraph/pyproject.toml
COPY README.md /app/twindb_lightrag_memgraph/README.md
COPY src /app/twindb_lightrag_memgraph/src

# Embed the WebUI dist inside the package source tree BEFORE the editable
# install so package-data ([tool.setuptools.package-data] in pyproject.toml)
# picks it up and the runtime _resolve_webui_dist() finds it at
# <package>/webui_dist/index.html (the candidate-2 path in __init__.py).
COPY --from=webui-builder /webui/dist \
     /app/twindb_lightrag_memgraph/src/twindb_lightrag_memgraph/webui_dist

# Install dependencies
RUN pip install --no-cache-dir -e /app/twindb_lightrag_memgraph/

COPY src/twindb_lightrag_memgraph /app/twindb_lightrag_memgraph

# Mirror the dist under the legacy /app path too — line above flattens the
# package next to its src/ layout and the embedded dist must follow.
COPY --from=webui-builder /webui/dist \
     /app/twindb_lightrag_memgraph/webui_dist

WORKDIR /app

# Patch lightrag application
RUN mkdir -p /app/sitecustom \
 && echo 'from twindb_lightrag_memgraph import register; register(replace_ui=True, mount_server=True, shim_native_routes=True)' > /app/sitecustom/sitecustomize.py
ENV PYTHONPATH=/app/sitecustom:${PYTHONPATH}

ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
