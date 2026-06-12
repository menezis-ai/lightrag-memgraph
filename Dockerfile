FROM fr2.icr.io/a100575-hprd/hkuds/lightrag:v1.4.9.11

# pip conf
ENV PIP_CONFIG_FILE=/tmp/pip.conf
COPY config/build.conf /tmp/pip.conf
COPY pyproject.toml /app/twindb_lightrag_memgraph/pyproject.toml
COPY README.md /app/twindb_lightrag_memgraph/README.md
COPY src /app/twindb_lightrag_memgraph/src

# Install dependencies
RUN pip install --no-cache-dir -e /app/twindb_lightrag_memgraph/

COPY src/twindb_lightrag_memgraph /app/twindb_lightrag_memgraph

WORKDIR /app

# Patch lightrag application
RUN mkdir -p /app/sitecustom \
 && echo 'from twindb_lightrag_memgraph import register; register(replace_ui=True, mount_server=True, shim_native_routes=True)' > /app/sitecustom/sitecustomize.py
ENV PYTHONPATH=/app/sitecustom:${PYTHONPATH}

ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
