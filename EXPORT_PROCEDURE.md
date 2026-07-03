# BNP Export Procedure

This procedure rebuilds the GitHub `export-1.0.0` branch from the private
`main` integration branch (the active full-runtime line).

BNP does not have Bun/Node in the runtime path. The export must therefore ship
the prebuilt WebUI assets under `src/twindb_lightrag_memgraph/webui_dist/`.
The frontend source tree `lightrag_webui_twin/` must also stay in the export for
audit/review needs, but the BNP Dockerfile must not build it.

The export is also the BNP validation handoff. It must include the Python test
tree (including the stdlib smoke runner `tests/smoke/run_smoke.py`, used for
manual validation against a deployed instance) and the `requirements/`
constraints. The export carries **no CI workflow**: the GitHub Actions smoke
job and the in-image CVE remediation were dropped (decision 2026-07-03) — OS
and Python package remediation is owned by the BNP-side delivery pipeline.

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

mkdir -p config requirements src/twindb_lightrag_memgraph/webui_dist
rsync -a "$SOURCE_REPO/config/build.conf" config/
rsync -a "$SOURCE_REPO/requirements/" requirements/
rsync -a --delete \
  "$SOURCE_REPO/lightrag_webui_twin/dist/" \
  src/twindb_lightrag_memgraph/webui_dist/
rm -f src/twindb_lightrag_memgraph/webui_dist/mockServiceWorker.js
rm -rf src/twindb_lightrag_memgraph.egg-info
```

## Export Dockerfile Rule

Start from the BNP Dockerfile in the source repo, then strip it down to the
runtime stage only:

- remove the frontend builder stage (`oven/bun`, `bun install`, `bun run build`);
- remove every `pip install` layer (BNP builds can run without outbound package
  access; editable installs trigger isolated build dependency resolution and
  can fail before the application code is even imported);
- remove the apt CVE-remediation layer (`apt-get ... --only-upgrade`): BNP
  build hosts may have no apt access and the `t64` package names may not exist
  on the base's Debian release — package remediation is owned by the BNP-side
  pipeline, not by this image.

The export Dockerfile must keep the same BNP runtime base image and runtime
registration logic, but it must use the committed prebuilt assets:

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

## Keep

- `src/twindb_lightrag_memgraph/**`
- `src/twindb_lightrag_memgraph/webui_dist/**`
- `lightrag_webui_twin/**`
- `tests/**` (includes `tests/smoke/run_smoke.py`, the manual validation runner)
- `requirements/**`
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
- `.github/` (the export carries no CI workflow — dropped 2026-07-03)
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
find . -path '*/__pycache__/*' -o -name '*.pyc'
find . -maxdepth 2 \( -name docs -o -name scripts -o -name node_modules -o -name .github \)
grep -n 'pip install\|apt-get' Dockerfile
git status --short
```

The two `find` commands and the `grep` command must print nothing.

## Stage And Push

```bash
git add -A
git add -f src/twindb_lightrag_memgraph/webui_dist/
git commit -m "Export BNP runtime bundle 1.0.0"
git push origin export-1.0.0
```

**The `git add -f` is load-bearing.** The `.gitignore` copied from the source
repo excludes `src/twindb_lightrag_memgraph/webui_dist/` (on `main` the dist
is a build artifact), so a plain `git add -A` silently stages the export
WITHOUT the WebUI assets — the resulting image then fails its own
`RUN test -f .../webui_dist/index.html` assertion at BNP build time. Check
`git diff --cached --stat` includes `webui_dist/` paths before committing.

Do not push this branch to the private integration remote.
