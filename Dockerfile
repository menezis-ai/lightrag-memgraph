# syntax=docker/dockerfile:1.6

# BNP runtime-only image. The export branch commits the prebuilt Twin WebUI
# under src/twindb_lightrag_memgraph/webui_dist/; no frontend or dependency
# build is performed in the BNP runtime image.
FROM fr2.icr.io/a100575-hprd/hkuds/lightrag:v1.5.6

# Release builds pass --build-arg TWIN_RELEASE_COMMIT=<full SHA>. The private
# SBOM gate rejects an image whose immutable digest does not carry the matching
# OCI revision. "unversioned" keeps local development builds possible while
# making them deliberately ineligible for release evidence.
ARG TWIN_RELEASE_COMMIT=unversioned
LABEL org.opencontainers.image.revision="${TWIN_RELEASE_COMMIT}"

COPY src /app/src
COPY src/twindb_lightrag_memgraph /app/twindb_lightrag_memgraph
COPY pyproject.toml README.md ENV_VARIABLES.txt /app/twindb_lightrag_memgraph_bundle/
COPY requirements /app/twindb_lightrag_memgraph_bundle/requirements
COPY config/build.conf /app/twindb_lightrag_memgraph_bundle/config/build.conf

RUN test -f /app/src/twindb_lightrag_memgraph/webui_dist/index.html \
 && test -f /app/twindb_lightrag_memgraph/webui_dist/index.html

WORKDIR /app

ENV PYTHONPATH=/app:/app/src:${PYTHONPATH}

# Audit 2026-08-06, B-01: run the runtime and native parsers as an
# unprivileged user. Deployment paths outside /app must be writable by twin.
RUN useradd --system --create-home --shell /usr/sbin/nologin twin \
 && chown -R twin:twin /app
USER twin

# This entrypoint registers the Twin UI, API overlay and native route shims
# before importing LightRAG's server.
ENTRYPOINT ["python", "-m", "twindb_lightrag_memgraph.lightrag_server"]
