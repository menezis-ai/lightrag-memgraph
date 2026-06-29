# BNP Export Procedure

This procedure rebuilds the GitHub `export-1.0.0` branch from the private
`main` integration branch (the active full-runtime line).

BNP does not have Bun/Node in the runtime path. The export must therefore ship
the prebuilt WebUI assets under `src/twindb_lightrag_memgraph/webui_dist/`.
The frontend source tree `lightrag_webui_twin/` must also stay in the export for
audit/review needs, but the BNP Dockerfile must not build it.

The export is also the BNP validation handoff. It must include the Python test
tree, the `requirements/` constraints, and a lightweight GitHub Actions workflow
that builds the BNP image and runs the stdlib smoke runner against the container.

## Branches

- Source branch: private integration `main`
- Export target: GitHub `origin/export-1.0.0`
- Do not push `export-1.0.0` to the private integration remote.

## Build The WebUI Once

Run this from the private source checkout:

```bash
SOURCE_REPO=/path/to/internal/twindb-lightrag-memgraph
EXPORT_TREE=/private/tmp/twindb-export-1.0.0
PRIVATE_REMOTE=internal

cd "$SOURCE_REPO"
git fetch "$PRIVATE_REMOTE" main
git switch main
git merge --ff-only "$PRIVATE_REMOTE/main"

cd lightrag_webui_twin
bun install --frozen-lockfile
bun run build
cd ..
```

## Rebuild The Export Tree

```bash
git fetch origin export-1.0.0
git worktree add -B export-1.0.0 "$EXPORT_TREE" origin/export-1.0.0
cd "$EXPORT_TREE"
git rm -r .

rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
  "$SOURCE_REPO/src" ./
rsync -a --exclude='__pycache__/' --exclude='*.pyc' \
  "$SOURCE_REPO/tests" ./
rsync -a --exclude='node_modules/' --exclude='dist/' \
  --exclude='coverage/' --exclude='playwright-report/' \
  --exclude='test-results/' --exclude='qa-screenshots/' \
  --exclude='scripts/' \
  "$SOURCE_REPO/lightrag_webui_twin" ./
rsync -a \
  "$SOURCE_REPO/pyproject.toml" \
  "$SOURCE_REPO/MANIFEST.in" \
  "$SOURCE_REPO/README.md" \
  "$SOURCE_REPO/ENV_VARIABLES.txt" \
  "$SOURCE_REPO/docker-compose.yml" \
  "$SOURCE_REPO/.gitignore" \
  "$SOURCE_REPO/Dockerfile" \
  ./

mkdir -p config requirements .github/workflows src/twindb_lightrag_memgraph/webui_dist
rsync -a "$SOURCE_REPO/config/build.conf" config/
rsync -a "$SOURCE_REPO/requirements/" requirements/
rsync -a "$SOURCE_REPO/.github/workflows/bnp-export-smoke.yml" .github/workflows/
rsync -a --delete \
  "$SOURCE_REPO/lightrag_webui_twin/dist/" \
  src/twindb_lightrag_memgraph/webui_dist/
rm -f src/twindb_lightrag_memgraph/webui_dist/mockServiceWorker.js
rm -rf src/twindb_lightrag_memgraph.egg-info
```

## Export Dockerfile Rule

Start from the BNP Dockerfile in the source repo, then remove only the frontend
builder stage (`oven/bun`, `bun install`, `bun run build`). The export Dockerfile
must keep the same BNP runtime base image and runtime registration logic, but it
must use the committed prebuilt assets:

```text
src/twindb_lightrag_memgraph/webui_dist/index.html
```

The runtime image should copy `src`, assert that `webui_dist/index.html` exists,
mirror `src/twindb_lightrag_memgraph` under `/app/twindb_lightrag_memgraph`,
put `/app` on `PYTHONPATH`, and register the Twin overlay with:

```python
from twindb_lightrag_memgraph import register
register(replace_ui=True, mount_server=True, shim_native_routes=True)
```

Do not run `pip install -e "/app/twindb_lightrag_memgraph/[server]"` in the BNP
export Dockerfile. BNP builds can run without outbound package access; editable
installs trigger isolated build dependency resolution (`setuptools`, `wheel`) and
can fail before the application code is even imported.

## Keep

- `src/twindb_lightrag_memgraph/**`
- `src/twindb_lightrag_memgraph/webui_dist/**`
- `lightrag_webui_twin/**`
- `tests/**`
- `requirements/**`
- `.github/workflows/bnp-export-smoke.yml`
- `pyproject.toml`
- `MANIFEST.in`
- `Dockerfile`
- `docker-compose.yml`
- `config/build.conf`
- `README.md`
- `ENV_VARIABLES.txt`

## Exclude

- `docs/`
- `.forgejo/`
- `scripts/`
- `.github/workflows/*` except `.github/workflows/bnp-export-smoke.yml`
- `lightrag_webui_twin/node_modules/`
- `lightrag_webui_twin/dist/`
- `lightrag_webui_twin/coverage/`
- `lightrag_webui_twin/playwright-report/`
- `lightrag_webui_twin/test-results/`
- `lightrag_webui_twin/qa-screenshots/`
- `lightrag_webui_twin/scripts/`
- `__pycache__/`
- `*.pyc`
- `*.egg-info`

## Verify Before Push

```bash
test -f src/twindb_lightrag_memgraph/webui_dist/index.html
test -f lightrag_webui_twin/package.json
test -f tests/smoke/run_smoke.py
test -f requirements/constraints-prod.txt
test -f .github/workflows/bnp-export-smoke.yml
find . -path '*/__pycache__/*' -o -name '*.pyc'
find . -maxdepth 2 \( -name docs -o -name scripts -o -name node_modules \)
grep -n 'pip install' Dockerfile
git status --short
```

The two `find` commands and the `grep` command must print nothing.

## BNP Export CI

The export branch carries `.github/workflows/bnp-export-smoke.yml`. This is not
the development CI matrix. It is a release-consumer smoke:

- build the export Dockerfile directly;
- start Memgraph 3.9.0;
- start the BNP runtime image with local JWT auth;
- run `python tests/smoke/run_smoke.py tests/smoke/runtime-smoke-ci-http.json`;
- upload the smoke report and HTTP trace.

If `fr2.icr.io/a100575-hprd/hkuds/lightrag:v1.4.9.11` requires registry auth
on the GitHub runner, configure repository secrets `BNP_REGISTRY_USERNAME` and
`BNP_REGISTRY_PASSWORD`. If the base image is publicly pullable, no secrets are
needed.

## Push

```bash
git commit -m "Export BNP runtime bundle 1.0.0"
git push origin export-1.0.0
```

Do not push this branch to the private integration remote.
