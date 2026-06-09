# Twin-real deploy image — LightRAG 1.4.9.11 + twindb-lightrag-memgraph
# (Couche 3 wired) + the React port bundled under /webui/.
#
# Two-stage build:
#   - Stage 1 (web)  : Bun installs + builds lightrag_webui_twin/ into dist/
#   - Stage 2 (final): Python 3.12 + LightRAG + our package + dist/ inlined
#
# The final image starts via boot.py which calls register() with the
# server-mode flags (replace_ui, mount_server, shim_native_routes,
# webui_stores="memgraph") before handing off to lightrag-server.

# ------------------------------------------------------------------ #
# Stage 1 — bundle the WebUI
# ------------------------------------------------------------------ #
FROM oven/bun:1.3.6 AS web

WORKDIR /web
COPY lightrag_webui_twin/package.json lightrag_webui_twin/bun.lock ./
RUN bun install --frozen-lockfile

COPY lightrag_webui_twin/ ./
# VITE_FORCE_MSW=false (default) — bundle expects real backend.
RUN bun run build && ls -la dist/

# ------------------------------------------------------------------ #
# Stage 2 — runtime
# ------------------------------------------------------------------ #
FROM python:3.12-slim AS runtime

# System deps: build-essential for any wheels that compile, curl for
# healthcheck, dumb-init for clean PID-1 signal handling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    dumb-init \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in isolated layers so iterative rebuilds are fast.
# LightRAG version pinned to BNP target (1.4.9.11); see CI matrix in
# .forgejo/workflows/ci.yml for the supported range.
#
# Runtime dependencies of LightRAG that aren't declared as hard requirements
# in lightrag-hku's pyproject — pip-master would install them at startup,
# but our security_baseline patch disables that path. Pinning explicitly
# (using the versions that work locally on our .venv) avoids the runtime
# `ModuleNotFoundError: No module named 'aiofiles'` surprise.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    "lightrag-hku[api]==1.4.9.11" \
    "neo4j>=5,<7" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydantic>=2" \
    "pydantic-settings>=2" \
    "PyJWT>=2.8" \
    "python-multipart>=0.0.9" \
    "pypdf>=6.1.0" \
    "python-docx>=0.8.11,<2.0.0" \
    "python-pptx>=0.6.21,<2.0.0" \
    "openpyxl>=3.0.0,<4.0.0" \
    "openai>=1.50" \
    "tiktoken>=0.6" \
    "aiofiles>=25" \
    "ascii_colors>=0.11" \
    "httpx>=0.28" \
    "json_repair>=0.57" \
    "nano-vectordb>=0.0.4" \
    "networkx>=3" \
    "numpy>=1.26,<2" \
    "pipmaster>=1.1" \
    "tenacity>=9" \
    "tqdm>=4.67" \
    "future>=1"

# Copy our package source — installed editable so the boot script can
# import it cleanly. The pyproject.toml declares the entry points.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-cache-dir -e .

# Bundle the WebUI dist from stage 1 into a deterministic path that
# boot.py passes to register(webui_dist=...).
COPY --from=web /web/dist /app/webui_dist

# Boot wrapper (calls register() then hands off to lightrag-server).
COPY deploy/ovh-twin/boot.py /app/boot.py

# Working dir for LightRAG (kv files, logs); volume-mounted at runtime.
RUN mkdir -p /data/rag_storage /data/inputs

# LightRAG listens on 9621 by default; we keep that.
EXPOSE 9621

# Sensible defaults; the stack.yml overrides everything that matters
# (MEMGRAPH_URI, OPENAI_API_KEY, WORKSPACE, etc.).
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=9621 \
    WORKING_DIR=/data/rag_storage \
    INPUT_DIR=/data/inputs \
    LIGHTRAG_KV_STORAGE=MemgraphKVStorage \
    LIGHTRAG_VECTOR_STORAGE=MemgraphVectorDBStorage \
    LIGHTRAG_DOC_STATUS_STORAGE=MemgraphDocStatusStorage \
    LIGHTRAG_GRAPH_STORAGE=MemgraphStorage \
    LLM_BINDING=openai \
    LLM_MODEL=gpt-4o-mini \
    EMBEDDING_BINDING=openai \
    EMBEDDING_MODEL=text-embedding-3-small \
    EMBEDDING_DIM=1536

# Healthcheck hits the LightRAG-native /health (NOT the shimmed root
# /health which only exists once register(shim_native_routes=True) has
# wrapped create_app — by the time we curl it, both paths are alive).
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:9621/health || exit 1

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "/app/boot.py"]
