# Twin demo backend — FastAPI + SQLite.
#
# Persists the operator-side mutations the maquette emits (doc approve /
# reject, tag governance, activity log, notification clears) in a real
# SQLite file mounted from a Docker volume. Replaces the client-side
# sql.js WASM approach — BNP infra won't accept a CDN-fetched WASM blob
# and a backend with audited code + real DB is the production-shaped
# answer.
#
# Image stays small: distroless-ish Python:slim, no compiler toolchain
# needed (FastAPI/uvicorn/aiosqlite are pure-Python wheels).

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.32" \
      "pydantic>=2.9" \
      "aiosqlite>=0.20"

COPY twin_demo_api /app/twin_demo_api
COPY seed-data /app/seed-data

# SQLite file lives under /data — bind-mounted from a Docker volume so
# state survives container rebuilds.
ENV TWIN_DEMO_DB_PATH=/data/twin-demo.sqlite \
    TWIN_DEMO_SEED_DIR=/app/seed-data \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "twin_demo_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
