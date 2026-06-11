#!/usr/bin/env bash
# Build the WebUI fork TS and embed dist/ inside the Python package.
#
# Required step before `python -m build` so that the resulting wheel ships
# the UI assets and can be `pip install`-ed in production without any
# Node/Bun toolchain on the target host (TwinRAG security baseline + DORA
# art. 9 — hermetic artefacts only).
#
# Output: src/twindb_lightrag_memgraph/webui_dist/
#         (declared in pyproject.toml [tool.setuptools.package-data])
#
# Strips dev-only artefacts:
#   - mockServiceWorker.js (audit WebUI fork TS §11 — would expose
#     mock-service-worker in prod, audit red flag)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBUI_SRC="$REPO_ROOT/lightrag_webui_twin"
PKG_DIST="$REPO_ROOT/src/twindb_lightrag_memgraph/webui_dist"

if [ ! -d "$WEBUI_SRC" ]; then
    echo "ERROR: WebUI source not found at $WEBUI_SRC" >&2
    exit 1
fi

if ! command -v bun >/dev/null 2>&1; then
    echo "ERROR: bun not on PATH. Install: https://bun.sh/" >&2
    exit 1
fi

echo "==> Building WebUI fork TS (bun ${BUN_VERSION:-1.3.6})"
cd "$WEBUI_SRC"
bun install --frozen-lockfile
bun run build

if [ ! -f "$WEBUI_SRC/dist/index.html" ]; then
    echo "ERROR: bun run build did not produce dist/index.html" >&2
    exit 1
fi

echo "==> Embedding dist/ → $PKG_DIST"
rm -rf "$PKG_DIST"
mkdir -p "$PKG_DIST"
cp -R "$WEBUI_SRC/dist/." "$PKG_DIST/"

# Strip dev-only artefacts (cf. audit WebUI fork §11)
if [ -f "$PKG_DIST/mockServiceWorker.js" ]; then
    rm -f "$PKG_DIST/mockServiceWorker.js"
    echo "    stripped: mockServiceWorker.js (dev-only, audit red flag)"
fi

# Sanity check: required files present
for required in index.html assets; do
    if [ ! -e "$PKG_DIST/$required" ]; then
        echo "ERROR: missing $required in embedded dist" >&2
        exit 1
    fi
done

echo
echo "==> Embedded WebUI ready ($(du -sh "$PKG_DIST" | cut -f1))"
ls -la "$PKG_DIST"
