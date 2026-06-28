# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This package provides Memgraph storage backends (KV, Vector, DocStatus) for [LightRAG](https://github.com/HKUDS/LightRAG) **without modifying LightRAG's source code**. LightRAG ships a graph backend; this package fills the remaining 3 slots so an entire instance runs on a single Memgraph DB.

### Branch policy (updated 2026-06-28)

- `main` — **active full-runtime line** (storage backends + intelligence + server overlay + WebUI fork + doctrine docs). The **default working branch**; day-to-day dev merges here, and it carries `pyproject` version `1.0.0`. It is ~130 commits ahead of `stable/0.6.x`.
- `stable/0.6.x` — **superseded by `main`.** Was the active dev line through mid-2026; now trails `main` and is no longer the working branch. Kept for history.
- `stable/0.5.x` — **LTS, storage backends only** (intent). WebUI/runtime work is pollution here. NB: the branch currently still carries `lightrag_webui_twin/` + `server/` paths despite the PR #30 (2026-05-20) revert — branch-hygiene cleanup pending, not yet done.
- `stable/0.3.2-lts` — frozen LTS = 0.3.2 + auto-create vector index.
- `0.4.x` was abandoned.

`pyproject.toml` reports version `1.0.0` and `register()` patches `lightrag.__version__` to `vX.Y.Z+memgraph-1.0.0` so the WebUI shows the composite version. See `changelog.md` for what's in vs. out of stable.

## Distribution

**Three-surface model** (re-articulated 2026-06-16, supersedes the looser "Forgejo-first / GitHub archived" framing of 2026-05-11). Each surface has a strict file-level scope; the source-of-truth flow is always **bunker → (GitHub `main` xor GitHub `export-1.0.0`)**, never the reverse.

- **`bunker` (Forgejo, `192.168.1.61`) — full source of truth.** Storage backends + intelligence layer (TwinRAGEngine, ReAct, DSEP) + server module (FastAPI overlay, classification, folders, idp_jwt, native_shims) + WebUI fork (`lightrag_webui_twin/`) + doctrine docs (`CLAUDE.md`, `DOCTRINE.md`, `WEBUI-WIRING-*`). Day-to-day dev pushes here. Active branch: `main`. The L1/L2/L3 ZIP-delivery split is retired here; intelligence/server are tracked normally and no longer gitignored.
- **GitHub `origin/main` (`menezis-ai/lightrag-memgraph`) — public backend patch only.** Strictly the storage adapter slice: `src/twindb_lightrag_memgraph/{__init__,_buffered_graph,_constants,_hooks,_pool,kv_impl,vector_impl,docstatus_impl}.py` + `tests/` for those + `pyproject.toml` + `README.md` + `sonar-project.properties`. **No `server/`, no `intelligence/`, no `classification*.py`, no `_folders.py`, no `asgi.py`, no `lightrag_server.py`, no `lightrag_webui_twin/`, no CLAUDE.md.** Used (a) for visibility on the Memgraph adapter, (b) as the receiving surface for Claude-action PRs from the GitHub integration when they target storage code, (c) as the substrate `export-1.0.0` rebuilds from.
- **GitHub `origin/export-1.0.0` — BNP delivery snapshot.** Full repo with prebuilt WebUI assets bundled under `src/twindb_lightrag_memgraph/webui_dist/` (no Bun/Node in the BNP runtime path). Rebuilt from `main` via `EXPORT_PROCEDURE.md`. **Never** pushed to `bunker`. Nothing else is pushed to `origin`.

**Flow direction is INTO bunker only — do not propose bunker → GitHub propagation as a routine step.** Clarified 2026-06-16 after a session over-invested in mirror-pushing a bunker fix back to GitHub `main`:

- **Bunker is the source of truth, period.** GitHub `origin/main` is a derived public-visibility slice; it can drift behind bunker. There is no per-fix back-port doctrine.
- **GitHub PRs flow INTO bunker**: Claude-action PRs on `menezis-ai/lightrag-memgraph` are triaged → absorbed into bunker (cherry-pick / port) → close the GitHub PR with the absorption note. Do NOT then mirror bunker's accepted version back to GitHub via a follow-up PR.
- A fix that touches `intelligence/`, `server/`, `classification*.py`, `_folders.py`, `asgi.py`, `lightrag_server.py`, `lightrag_webui_twin/`, or doctrine docs → bunker only (GitHub `origin/main`'s file scope simply excludes them).
- A fix that touches the public-backend files (`__init__.py`, `_buffered_graph.py`, `_constants.py`, `_hooks.py`, `_pool.py`, `kv_impl.py`, `vector_impl.py`, `docstatus_impl.py`) → still bunker only by default. If the public slice on `origin/main` needs to be re-synced, that's a deliberate batch operation (re-publish, release), not a per-fix flow.
- The only sanctioned bunker → GitHub flow is the `export-1.0.0` rebuild via `EXPORT_PROCEDURE.md` for BNP delivery.

The global directive §7 ("push to all remotes") does **not** apply here.

## Doctrine layer

`DOCTRINE.md` (repo root) is the strategic-intent layer that sits **above** this file. It explains why the architecture takes the shape it does — the non-fork doctrine, the additivity/idempotence/graceful-degradation contract, the duality narrative (Salah's "façade unifiée" vs Louis/Eric's "extension du patch déjà en prod"), and the catalogue raisonné of inscribed intentions. Reading order when inheriting an unfamiliar zone of code: code + header comment → introducing commit → `CLAUDE.md` (operational posture) → `DOCTRINE.md` (strategic frame) → project memories. Do not "simplify" an inscription whose intent you have not first decoded through that sequence — the canonical example is forking LightRAG, which would erase four distinct intentions (political, operational, cognitive, economic) at once.

## Audits

`docs/audits/<area>/audit-<date>.md` is the convention for honest cross-cutting reviews. `docs/audits/lightrag-interactions/audit-2026-06-13.md` is the reference review for `/twin/api/query` and retrieval. Both of its priorities are now **closed** (keep this in mind so you don't "re-fix" them): (1) the nominal `/query` path no longer reconstructs sources via a second vector search — it grounds through `aquery_llm()` and projects sources from `data.references` (the chunks LightRAG actually used); (2) the WebUI `tag_filter` / `doc_filter` / `min_score` are no longer a retrieval no-op — they are enforced at the Memgraph storage layer (`vector_impl._build_search_cypher`, bound via `storage_filter_context`), so an excluded chunk/entity never enters the prompt rather than being trimmed from the Sources panel afterwards. **Known residual:** `tag_filter`/`doc_filter` on the entity/relation vdb scope the *selection*, not the LLM-aggregated `content` of a kept record (same residual as folder scoping — see `test_retrieval_filters_scoping.py` and `vector_impl._build_search_cypher`). Read the relevant audit before claiming a fix in those areas.

Other open audit areas under `docs/audits/`: `intelligence-layer/`, `lightrag-1.4.9.11/`, `process-install-bnp/`, `retrieval-tuning/`, `sonarqube/`. Same `audit-<date>.md` convention. Consult the relevant area before claiming a fix in code it covers.

Three release/ops docs added for the 1.0.0 cut (merged to `main`; the `production-readiness-p0-risk-accept-lightrag-cves` branch is retired):
- `docs/security/lightrag-1.4.9.11-risk-acceptance.md` — documented risk-acceptance for the LightRAG CVEs in the pinned `1.4.9.11`. Read before raising a CVE on that dependency.
- `docs/operations/install-runbook.md` — BNP install runbook.
- `docs/qa/release-candidate-1.0.0-checklist.md` — the 1.0.0 RC checklist.

## Commands

### Install

```bash
pip install -e ".[test]"          # storage backends only (kv/vector/docstatus + tests/test_*.py at the top level)
pip install -e ".[test-server]"   # adds server/ deps (fastapi, uvicorn, PyJWT, pypdf, …) for tests/test_server/
pip install -e ".[test-intelligence]"  # adds intelligence/ deps (openai, pydantic-settings, respx) for tests/test_intelligence/
pip install -e ".[all]" -e ".[test]"   # everything — what the local .venv carries
```

**Extras matrix (`pyproject.toml`):** `[test]` ships `fastapi`+`httpx` but **not** `openai`/`pydantic-settings`. Since `pytest tests/` collects the `tests/test_intelligence/` and `tests/test_server/` trees, a bare `.[test]` install fails at *import* on those trees (e.g. `intelligence/config.py` needs `pydantic-settings`). For a full local run install the matching test extra (`[test-server]`, `[test-intelligence]`) or `[all]`. Runtime-only extras: `[server]`, `[intelligence]`, `[tracing]`.

`uv.lock` exists locally but is **gitignored and untracked** (commit `1325b47`, "chore: gitignore uv.lock") — the project deliberately ships no committed lockfile and relies on `pip install -e`. Don't commit `uv.lock` or convert dependency management to a `uv sync` flow without an explicit decision; treat the editable-install path above as the source of truth. Pinned dependency sets live under `requirements/` (`constraints-dev.txt`, `constraints-prod.txt`, `prod-target.txt`).

### Run the FastAPI overlay locally

```bash
python -m twindb_lightrag_memgraph.server   # uvicorn factory, default port
LIGHTRAG_PORT=8080 python -m twindb_lightrag_memgraph.server
```

Requires `MEMGRAPH_URI` + the LLM credentials read by `server/settings.py:LightRAGServerSettings`. See `WEBUI-WIRING-PLAN.md` (entry point; status split across `WEBUI-WIRING-WIRED.md` as-built and `WEBUI-WIRING-TO-WIRE.md` remaining) for the full Couche 2 / Couche 3 contract this server exposes to the WebUI fork.

### Environment variables

`ENV_VARIABLES.txt` (repo root) is the **canonical reference** for every variable read by this package — grouped by concern (overlay activation, IdP, folders, classification, …). Read it before hunting through code for env var names or defaults. Two load-bearing groups worth knowing without opening the file:

- **Overlay activation (§0)** — `TWIN_REPLACE_UI` / `TWIN_MOUNT_SERVER` / `TWIN_SHIM_NATIVE_ROUTES`. A deployment whose boot already calls a bare `register()` activates the full overlay surface by setting these three vars — **no code change**. This is the mechanism that lets BNP flip overlays on/off without a redeploy.
- **LightRAG-native vars are unchanged** — `OPENAI_API_KEY`, `LLM_BINDING`, `EMBEDDING_*`, `SUMMARY_LANGUAGE`, `TOKEN_SECRET`, etc. keep working as documented upstream. `ENV_VARIABLES.txt` lists only the variables this package introduces or consumes.

### Production-style entrypoints

Three entrypoints, same doctrine (`register()` before LightRAG server import), different launch surfaces:

- `twin_main.py` (repo root) — readable reference entrypoint. Calls `register(replace_ui=True, mount_server=True, shim_native_routes=True)` at module top, then imports `lightrag.api.lightrag_server.main` and runs it. Use it when launching the runtime directly (e.g. `python twin_main.py`) or when the orchestrator can set `CMD ["python", "twin_main.py"]`.
- `src/twindb_lightrag_memgraph/lightrag_server.py` — the **BNP container entrypoint**, `python -m twindb_lightrag_memgraph.lightrag_server`. Same three flags, but importable as a module so the production Dockerfile's `ENTRYPOINT` is `-m`-launchable. Its module docstring spells out the why: *"Do not rely on sitecustomize for production activation. Some launchers and python -m execution paths can import/execute the LightRAG server in a way that bypasses a module patched by name."*
- `src/twindb_lightrag_memgraph/asgi.py` — **gunicorn / uvicorn-factory entrypoint** (`gunicorn 'twindb_lightrag_memgraph.asgi:get_application()' -k uvicorn.workers.UvicornWorker` or `uvicorn --factory twindb_lightrag_memgraph.asgi:get_application`). Runs a bare `register()` at import time (no flags — relies on `TWIN_REPLACE_UI` / `TWIN_MOUNT_SERVER` / `TWIN_SHIM_NATIVE_ROUTES` env activation per `ENV_VARIABLES.txt` §0), then imports `lightrag.api.lightrag_server.get_application`. Required when the launcher imports the app via an import string per worker — a separate boot script never executes inside the worker, so the native LightRAG app would otherwise be served unpatched.

If you change the `twin_main.py` / `lightrag_server.py` flag set, change both — they're parallel. `asgi.py` is env-driven by design; decide separately whether the deployment's overlay env needs to track the flag change.

The ordering is doctrine: `register()` MUST run before the LightRAG server module is imported. Calling it from inside `lightrag_server` (e.g. via a sed-prepended import) creates a circular import because `create_app` doesn't exist yet mid-import.

Two Docker assets, do not confuse them:
- `Dockerfile` (repo root) — the **BNP production image**. Two-stage build: stage 1 = `oven/bun:1.3.6` builds `lightrag_webui_twin/` and strips `mockServiceWorker.js`; stage 2 = `fr2.icr.io/a100575-hprd/hkuds/lightrag:v1.4.9.11` base, editable `pip install -e`, dist embedded at both candidate paths that `_resolve_webui_dist()` looks at (`<package>/webui_dist/index.html` and the legacy flat `/app/twindb_lightrag_memgraph/webui_dist/`). `ENTRYPOINT` is `python -m twindb_lightrag_memgraph.lightrag_server`.
- `Dockerfile.example` — illustrative minimal wiring, not the production build.

### Build the embedded WebUI

```bash
scripts/build_webui.sh
```

Builds `lightrag_webui_twin/` with Bun and copies `dist/` into `src/twindb_lightrag_memgraph/webui_dist/` (declared as package-data in `pyproject.toml`), stripping `mockServiceWorker.js` (dev-only, BNP audit red flag). **Required before building a wheel** — `replace_ui=True` fails at runtime with `FileNotFoundError` if the installed package lacks `webui_dist/index.html`. The embedded dist lets BNP `pip install` the UI with no Node/Bun on the target host.

### Tests

The local `.venv` is Python 3.14. Be aware of two known footguns recorded in project memory:

- `pytest-cov`'s `--cov` flag is broken with numpy on Python 3.14 (double-import). Use `coverage run -m pytest ...` then `coverage xml` instead.
- `pytest-asyncio` is in auto mode (`asyncio_mode = "auto"` in `pyproject.toml`). Don't add `@pytest.mark.asyncio` decorators manually.

```bash
# Unit tests only (no Memgraph) — what CI runs as the unit-tests job.
# Marker-based: pytest scans tests/, conftest.py auto-skips @pytest.mark.integration
# when MEMGRAPH_URI is unset. Picks up intelligence/ + server/ test trees too.
pytest tests/ --ignore=tests/test_bench.py -v

# All integration tests (real Memgraph required)
docker compose up -d memgraph   # uses root docker-compose.yml (memgraph/memgraph-mage:latest on 7687/7444)
# or, ad-hoc: docker run -d --name memgraph-test -p 7687:7687 memgraph/memgraph-mage:latest
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/ --ignore=tests/test_bench.py -v

# Single test
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_kv.py::TestMemgraphKVStorage::test_upsert_and_get -v

# Benchmarks (latency / throughput at 100/1K/10K)
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_bench.py -v -s

# Coverage (workaround for Py3.14 + numpy)
coverage run -m pytest tests/ --ignore=tests/test_bench.py
coverage xml  # produces coverage.xml for SonarQube
```

`tests/conftest.py` auto-skips `@pytest.mark.integration` tests when `MEMGRAPH_URI` is unset, so no need to filter manually for offline runs.

Test layout: storage tests live directly under `tests/` (`test_kv.py`, `test_vector.py`, `test_docstatus.py`, `test_buffered_writes.py`, `test_batch_patch.py`, `test_e2e.py`, …); intelligence and server suites live in `tests/test_intelligence/` and `tests/test_server/` respectively. `pytest tests/` collects all three trees.

### Restricted runtime smoke test

`tests/smoke/run_smoke.py` is a **stdlib-only** runner that executes a JSON manifest (default `tests/smoke/runtime-smoke.json`) against a deployed instance — designed for BNP-style containers where no pip install is possible. Validates `/webui` mount, local JWT login/logout, anonymous rejection on `/twin/api/*`, and that native LightRAG + Twin overlay routes are reachable post-auth. Outputs `/tmp/twin-smoke-report.json` + `/tmp/twin-smoke-http.log` without logging secrets.

```bash
export TWIN_SMOKE_BASE_URL="https://your-runtime-host"
python tests/smoke/run_smoke.py tests/smoke/runtime-smoke.json
```

`tests/test_smoke_runner.py` is the unit test for the runner itself (collected by `pytest tests/`).

### SonarQube

SonarQube is the sovereign instance at `http://192.168.1.49:9000` (version 26.5.x, autossh tunnel on ALPHA per `citadelle-infra/INFRA_MAP.md`). The old `192.168.1.212:9000` address was decommissioned (no host, no CT) and was confirmed dead on 2026-06-21 — `sonar.host.url` was corrected then. `sonar-scanner` is at `/opt/homebrew/bin/sonar-scanner`. The analysis token lives in the MCP vault (item `6f5f8b5d`, `sqa_…` on the first notes line); fetch it via the `citadelle-infra/OPS_VAULT.md` SSH+bw pipe-direct recipe and pass it as `SONAR_TOKEN` env to the scanner — never committed, never echoed.

Scanner config lives in `sonar-project.properties` at the repo root: `sonar.sources=src`, `sonar.tests=tests`, Python version matrix `3.10,3.11,3.12,3.13,3.14`, coverage report at `coverage.xml`. Reuse this file rather than passing flags ad hoc.

`aquery()` cognitive complexity must stay ≤ 15 — keep helper methods extracted (project memory).

### CI matrix

CI (`.forgejo/workflows/ci.yml`) runs on Forgejo Actions. Two runner pools:
- `[self-hosted, docker]` — Python + WebUI lint/unit jobs (docker squad 310-314 post-2026-06-11 OOM incident, bumped to 4096M).
- `[self-hosted, high]` — Playwright e2e jobs (`webui-e2e`, `webui-e2e-real`). The `high:host` pool can NOT run `setup-python` actions — see `reference_ci_runner_pools.md`. Python jobs use `docker run python:X-bookworm` against a mounted workdir instead of `setup-python`, which is why `unit-tests` / `integration-tests` look containerized inside the workflow.

Jobs:
- **unit-tests**: Python 3.10/3.11/3.12/3.13 × LightRAG 1.4.9.11 / 1.4.11 / 1.4.12 (no Memgraph). Runs `pytest tests/ --ignore=tests/test_bench.py` inside a `python:${ver}-bookworm` container — `conftest.py` auto-skips `@pytest.mark.integration` when `MEMGRAPH_URI` is unset.
- **integration-tests**: LightRAG matrix × Memgraph **3.9.0 + 3.10.1** (3.9.0 = BNP prod target after the 2026-06-19 rollback; 3.10.1 kept as forward-compat, 3.11 imminent; both pinned so `latest` can't drift the coverage point). `max-parallel: 1` — each matrix job spins its own isolated docker network + Memgraph container to avoid cross-job contention. URI inside the network: `bolt://memgraph:7687`.
- **webui-lint**: cheap ESLint gate on `lightrag_webui_twin/` (parallel to the heavier WebUI jobs).
- **webui-tests**: `bun install --frozen-lockfile && bun run typecheck && bun run test:run && bun run build`. Bun pinned to `1.3.6` (never `latest` — `setup-bun` would hit api.github.com's 60/h anonymous rate limit and the bunker runner pool shares its outbound IP).
- **webui-e2e**: Playwright operator journeys against the MSW-backed WebUI (`high` pool).
- **webui-e2e-real**: Playwright against a real Twin backend (Memgraph + LightRAG container spawned by the job, `LIGHTRAG_API_KEY` set to `real-e2e-token`). LLM retrieval cases skipped (no model creds in CI). Also runs on the `high` pool.

Matrix exclusions (don't re-add without checking):
- LightRAG `1.4.9` vanilla dropped 2026-05-29 — BNP runs `1.4.9.11`, vanilla `1.4.9` flaked on the bunker runner.
- LightRAG `1.4.10` dropped (issue #6) — intermittent failures under integration load, fixed upstream in 1.4.11+.
- Memgraph `3.7.2` / `3.8.0` / `latest` dropped — never deployed at BNP. `3.9.0` is the prod target after BNP's 2026-06-19 rollback from `3.10.1`; `3.10.1` retained as forward-compat coverage (3.11 imminent). e2e jobs run on `3.9.0` only (single BNP target — Playwright jobs aren't matrixed across versions to spare the `high` pool).

Branch protection on `main`, `stable/0.5.x`, `stable/0.3.2-lts` requires the `CI / unit-tests*`, `CI / integration-tests*`, `CI / webui-lint`, `CI / webui-tests`, `CI / webui-e2e*` checks to be green before merge.

A push triggers 1–15 min of CI (global directive §6). Run unit + integration locally first; do not push speculatively.

## Architecture

### Storage package (`src/twindb_lightrag_memgraph/`)

- `patches/registry.py` — **home of `register()`** since the P2 backend modularization (commit `cf45bfd`). `__init__.py` is now a thin re-export shim that mirrors the registry's public names into the historical root import surface (and keeps test/downstream monkeypatches working via a two-way `_sync_*` mechanism) — don't put logic there. `register()` monkey-patches three dicts in `lightrag.kg`: `STORAGE_IMPLEMENTATIONS`, `STORAGE_ENV_REQUIREMENTS`, `STORAGES`. Module paths in `STORAGES` **must be absolute** (`twindb_lightrag_memgraph.kv_impl`, not relative) because `lazy_external_import` resolves with `package="lightrag"`. Idempotent via `_registered` flag. Also patches `lightrag.__version__` to append `+memgraph-{version}` so the WebUI shows `core_version` like `v1.4.9.11+memgraph-0.5.3`. Must be called **before** `LightRAG(...)` — the `lightrag_server.py` module-level entrypoint hardens that ordering against `python -m` import-bypass paths, see §Production-style entrypoints. Beyond storage, `register()` takes the runtime-overlay flags: `replace_ui=True` swaps the native `/webui` Mount for the embedded `webui_dist/` build; `mount_server=True` mounts the Twin sub-app under `twin_api_prefix` (default `/twin/api`) with chained lifespans; `shim_native_routes=True` prepends the `native_shims` router; `webui_stores` picks `"memgraph"` (default, persistent, needs `MEMGRAPH_URI`) vs `"seed"` (in-memory demo fixtures); `webui_categories_config` mirrors a JSON tag-category taxonomy on every boot (replace-not-merge, Config-as-Code); `classify`/`classification_*` args wire the MIP classification hook.
- `_pool.py` — Two independent async Bolt drivers (write + read) as module-level singletons; both detect event-loop changes and rebuild on a thread-safe lock. `acquire_write_slot()` is an `asyncio.Semaphore` (default 8, `MEMGRAPH_WRITE_CONCURRENCY`) wrapping every write. Read sessions (`get_read_session()`) are never throttled. URI scheme determines whether `database=` is passed natively (`neo4j://...`) or via `USE DATABASE` (`bolt://...`); on Memgraph Community, `USE DATABASE` fails on first attempt and is silently skipped thereafter. The graph backend (built into LightRAG) keeps **its own** driver — production has 3 pools by design.
- `_buffered_graph.py` — `_BufferedGraphProxy` wraps the graph storage during `merge_nodes_and_edges`, accumulating `upsert_node`/`upsert_edge`, then flushes nodes-then-edges as 2 UNWIND queries. Supports read-your-own-writes via in-memory buffer checks before delegating. Reduces ~130 round-trips/doc to 2–3.
- `_hooks.py` — Post-indexation hooks.
- `_constants.py` — Validators, defaults, env var names. **Identifier validator** here is the canonical place for any new label / database / namespace input (Cassandre flagged Cypher injection on f-string interpolation of these — labels use backticks, but `USE DATABASE` and ontology `rel_type` from LLM output do not).
- `kv_impl.py` — `MemgraphKVStorage`. Label `KV_{workspace}_{namespace}`. Value dict serialized to single `data` JSON string. Batch via UNWIND + MERGE.
- `vector_impl.py` — `MemgraphVectorDBStorage`. Label `Vec_{workspace}_{namespace}`, vector index via `CREATE VECTOR INDEX ... WITH CONFIG {dimension, capacity, metric: "cos"}`. `query()` uses MAGE `vector_search.search()` and **auto-creates the index if missing then retries once** (added in 0.5.1 — silent `[no-context]` fix). `cosine_better_than_threshold` read from `global_config["vector_db_storage_cls_kwargs"]`, default 0.2.
- `docstatus_impl.py` — `MemgraphDocStatusStorage`. Label `DocStatus_{workspace}` (no namespace suffix). Document identity stays on the single physical `DocStatus` node; folder membership is the many-to-many relation `(:DocStatus_{workspace})-[:MEMBER_OF]->(:Folder_{workspace} {id})`. The legacy `folder` property is still dual-written as a migration safety net only; new reads/list/counts/delete-refcount semantics are membership-authoritative. Indexes on `id`, `status`, `file_path`, `folder`, `track_id`, `updated_at`, `created_at`, `content_hash`. Sequential upserts (per-item `DocProcessingStatus`-vs-dict serialization). `get_docs_paginated` runs count + fetch in parallel via `asyncio.gather` over two read sessions. Sort fields are whitelisted against injection.
- `_folders.py` — Env-only Twin folder catalog parser (`TwinFolder`, `TwinFolderCatalog`, `load_folder_catalog`). Reads `TWIN_DEFAULT_FOLDER` / `TWIN_FOLDERS_JSON` / `TWIN_MAX_FOLDERS`. Deliberately FastAPI-free so it can feed `_build_runtime_config()` even when only `replace_ui=True`. Server-side runtime-mutable persistence sits in `server/folder_store.py`.

The package also patches `MemgraphGraphStorage` with batch read methods (`get_nodes_batch`, `node_degrees_batch`, `get_edges_batch`, `get_nodes_with_degrees_batch`, etc.) replacing N sequential queries with single UNWIND queries.

### Server module (`src/twindb_lightrag_memgraph/server/`)

FastAPI app factory that sits on top of `register()` and gives the WebUI fork a real backend (Couche 3 of `WEBUI-WIRING-PLAN.md`). Run with `python -m twindb_lightrag_memgraph.server` (uvicorn factory mode; `LIGHTRAG_PORT=8080` to override). Layout:

**Post-P2 modularization (commit `cf45bfd`):** two heavy route modules were split into subpackages, with the old flat modules kept as **compat shims** so historical import paths and their monkeypatches still resolve — do not delete the shims:
- `twin_query_routes.py` → real code in `server/query/router.py` (the shim aliases `sys.modules[__name__]` to it).
- `webui_router.py` (now thin) → routes split across `server/webui/`: `router.py` + per-domain `routes_{tags,documents,folders,graph,activity,notifications}.py`, plus `store.py` and `events.py`.
Other server modules not listed below: `_lightrag_compat.py` (LightRAG version-shim helpers), `api_wiring.py` (route assembly), `api_key_routes.py` + `api_key_store.py` (operator API-key CRUD), `quota.py` + `quota_routes.py` (per-folder quota enforcement), `document_hash.py`.

- `app.py` — `create_app()` factory. Mounts CORS, triple auth (static API key + legacy CFT JWT + local `POST /login` / `POST /logout` that issues a `twin_local_token` HttpOnly secure cookie, added 2026-06-09 in commit `0a68c02` — driven by `LIGHTRAG_JWT_SECRET` + `LIGHTRAG_JWT_PASSWORD`), LangSmith tracing, `/query`, `/insert`, `/health`, and the chunk/document routes. `settings.py` (`LightRAGServerSettings`) provides runtime config; `auth.py` is the auth router; `tracing.py` does trace-parent propagation.
- `chunk_routes.py` — `/chunks/*` and `/documents/*` endpoints (P3 context expansion) that the WebUI's DocDetailPanel + RetrievalTab consume.
- `webui_router.py` + `webui_models.py` + `webui_*store.py` (`webui_tagstore`, `webui_activitystore`, `webui_notificationstore`) — Twin overlay endpoints (`/twin/api/{tags,activity,notifications,...}`). Pydantic models in `webui_models.py` are the Python side of the contract whose TypeScript twin lives in `lightrag_webui_twin/src/fixtures/`. `webui_seed.py` seeds them for demo/test runs.
- `folder.py` + `folder_store.py` — Twin folder request binding + admin CRUD. `folder.py` exposes `bind_request_folder()` / `resolve_folder_from_headers()` (reads `X-Twin-Folder`, raises 400/403 on invalid or unknown folders) and a `ContextVar` so downstream code reads the active folder without thread argument-passing. `folder_store.py` persists operator-added folders in-memory by default, or to JSON when `TWIN_FOLDERS_RUNTIME_FILE` is set. Env-seeded folders always win id collisions; the runtime store can't shadow the SRE-provisioned default.
- `idp_jwt.py` — IdP middleware (Couche 3 §3.3). Verifies JWTs from cookie or `Authorization` header against a JWKS endpoint (TTL-cached) and projects claims into the `AuthenticatedUser` shape the React port consumes (`lightrag_webui_twin/src/types/auth.ts`). Claim names are fully env-mapped so BNP-specific bindings don't need a code change. `require_admin_user()` gates admin routes through the `admin:folders` gateway scope, granted from `TWIN_IDP_ADMIN_GROUPS`. Dormant when `TWIN_IDP_JWKS_URL` is unset — auth falls back to the static API key / legacy JWT branches in `auth.py`, and `_build_runtime_config` keeps `debugUser` alive for local/dev standalone runs.
- `native_shims.py` — Shadow router that re-shapes LightRAG's native FastAPI surface to the React port's contract (`GET /documents`, `GET /documents/{id}/chunks`, projected `/health`, `/pipeline_status`, etc.). Also shadows LightRAG's native auth routes (`/auth-status`, `/login`, `/logout`) by delegating to `server/auth.py` — LightRAG 1.4.9.11 mints guest JWTs from `/auth-status`, which the Twin runtime must not do. Mounted by `register(shim_native_routes=True)` at the **head** of `app.router.routes` so shims win the match against LightRAG's later `include_router(...)`. Doctrine: WebUI contract is source of truth; LightRAG routes get translated, not vice-versa. Coverage table is in the module docstring.
- `graph_reader.py` — Cypher → `GraphEntity`/`GraphRelation` mapper for the WebUI Knowledge Graph tab. Workspace-scoped queries, best-effort `entity_type` → closed enum mapping, deterministic hash-bucket layout so positions are stable across page loads without a force simulation. Also contains the M12 batch 2 write helpers (entity/relation patch, create, delete) that back the PATCH/POST/DELETE `/twin/api/graph/*` routes in `webui_router.py`.
- `twin_query_routes.py` — `POST /twin/api/query` wrapper around `aquery()`. Adds a cheap vector-only retrieval against `chunks_vdb`, joins to `DocStatus` for parent doc paths, and returns `{response, sources}` so the React port can render a sources panel. Inline `{cite:N}` markers in the response still need a prompt-side change.

### Classification (Couche 2 — BNP MIP labels)

- `classification.py` — `detect_classification()` extracts Microsoft Information Protection (MIP/AIP) sensitivity labels from OOXML (`docx/xlsx/pptx` via stdlib), legacy OLE (`doc/xls/ppt` via optional `olefile`), and PDF (XMP via optional `pikepdf`). Maps GUID → tenant-specific class (BNP `C1`/`C2`/`C3`/`C4`) via a JSON map at `TWIN_MIP_LABEL_MAP`. Gracefully degrades when optional deps or labels are missing.
- `_classification_hook.py` — `install_classification_hook()` patches LightRAG's ingestion path: runs `detect_classification()` on the source path *before* chunking, writes the payload to `DocStatus.metadata.classification`, and optionally REJECTS docs above `TWIN_MIP_MAX_CLASSIFICATION` (default `C2`). Rejections emit a `classification-rejected` audit event (callback injected to stay decoupled from the overlay store). **Opt-in** — must be called after `register()`. Never raises into the caller; failures yield `class_id="UNKNOWN"` and let LightRAG decide.

### Intelligence package (`src/twindb_lightrag_memgraph/intelligence/`)

L3 layer between LightRAG retriever (L2) and the agentic platform (L4). Single entry point: `TwinRAGEngine` in `engine.py`.

Pipeline: `F05 Intent → REASON (coref + F03 expansion) → ACT (search + F04 rerank) → OBSERVE (synthesis)`.

- `config.py` — `TwinRAGConfig` (pydantic-settings, `TWIN_RAG_` env prefix). Separate chat LLM and indexing LLM credentials so dev queries don't compete with prod ingestion GPU. Reasoning effort tuned per phase (intent=low, reason=medium, reranker=low, synthesis=high). Feature flags for OOS detection, query expansion, cognitive reranking, feedback, workspace routing.
- `features/` — `intent_classifier.py` (F05), `query_expander.py` (F03), `cognitive_reranker.py` (F04), `feedback.py`, `workspace_router.py` (F06, routes by domain).
- `react/` — `reason.py`, `act.py`, `observe.py` (the ReAct loop).
- `ontology/` — DSEP (Domain-Specific Extraction Profile, **not "SCG"** per user preference) pipeline. 4 steps: `extract` → `cluster` → `enrich` → `validate`. `pipeline.run()` is dry-run by default; call `pipeline.approve(result, workspace)` to persist. `ontology.json` absence = feature disabled, zero behavior change. `expand_v2()` does NOT call `initialize()` in hot path — uses lightweight `has_data()` count query.
- `prompts/` — Prompt templates loaded via `.format()`. **JSON braces inside prompts must be escaped `{{` `}}`** (project memory).
- `routing/routing_rules.json` — Embedded default for the workspace router.
- `thesaurus/it_ops_thesaurus.json` — Term expansion source for IT Ops domain.
- `models/schemas.py` — Pydantic models for every data structure in the intelligence package (`IntentType`, `IntentResult`, `Citation`, `QueryTrace`, `QueryResult`, `FeedbackEntry`, …). Typed contracts passed across the ReAct phases.
- `json_utils.py` — Tolerant JSON helpers for LLM responses (`load_json_object` with fenced-```json``` fallback, `clamp_float`, `coerce_str`). Use these instead of bare `json.loads` on model output.
- `prompt_security.py` — Prompt-boundary guard. `neutralize_reserved_tags()` stops untrusted user/document text from closing or forging the `<UNTRUSTED_*>` / `<USER_QUESTION>` prompt delimiter tags. Apply to untrusted text spliced into prompts.

LLM call pattern (project memory): `AsyncOpenAI(api_key=..., base_url=...)` with `response_format={"type": "json_object"}`. Patch target for tests is the **exact module import path** (e.g., `twindb_lightrag_memgraph.intelligence.ontology.steps.extract.AsyncOpenAI`), not the original `openai.AsyncOpenAI`.

User preference: config files in **JSON, not YAML** (no extra dependency, ecosystem consistency).

### WebUI fork (`lightrag_webui_twin/`)

Sibling Vite + Bun + React 19 + TypeScript strict + Tailwind v3 sub-project. Ports the design proto at `/Users/julien/Desktop/UI/` (untouched reference) into a typed, tested codebase that will eventually serve as the Twin operator console (citations cliquables, UI tag rétroactif, sous-graphe filtré par tag, source isolation badge).

**Roadmap** — S1/S2/S3/S4a/S4b/S4c are landed (see memory `project_webui_fork.md`). The live end-to-end mutation loop (WebUI → FastAPI → Memgraph store → cache invalidation → refetch) is in place. **Couche 3** progress: runtime config injection + frontend folder cutover are landed; backend `X-Twin-Folder` scoping guard + admin folder CRUD live in `server/folder.py` and `server/folder_store.py`. The Memgraph graph reader **and** the M12 batch 2 write persistence both live in `server/graph_reader.py`. **Authoritative status lives in the WEBUI-WIRING docs at the repo root** — `WEBUI-WIRING-PLAN.md` is the entry point, split into `WEBUI-WIRING-WIRED.md` (as-built) and `WEBUI-WIRING-TO-WIRE.md` (remaining + PO-gated items). Re-read them before claiming any Couche/Gxxx item is closed.

**Folders (2026-06-08 CTO Group doctrine, supersedes 2026-06-01 "Space"):** The canonical term is **Folder**. The Twin overlay surface (routes `/twin/api/folders`, Pydantic models, audit events, frontend props) uses `folder`. The LightRAG-aligned label that lands on Memgraph nodes (`KV_{ws}`, `Vec_{ws}`, `DocStatus_{ws}`, …) is still spelled `workspace` because that's an upstream contract we don't get to rename — see the module docstring in `src/twindb_lightrag_memgraph/_constants.py`. The React port no longer hardcodes any folder id — initial folder comes from `window.__twinConfig.defaultFolderId`. The server injects folders via `_build_runtime_config()` from env vars:

- `TWIN_DEFAULT_FOLDER` (fallback `WORKSPACE`, then `default`)
- `TWIN_DEFAULT_FOLDER_LABEL`
- `TWIN_FOLDERS_JSON` — JSON array of `{id, label, kind, description, sources}`
- `TWIN_MAX_FOLDERS` — clamped 1..5 (one SRE default + up to four admin-created)
- `TWIN_FOLDERS_RUNTIME_FILE` — operator-added folders persisted as JSON

The HTTP client sends `X-Twin-Folder` only. Empty state when no folder is provisioned: `No folder available for this KB. Please contact Twincore Team.`

**Folder membership model (2026-06-25 refactor):** Do not describe folders as UI-only filters. The `{workspace}` label remains the single physical LightRAG/Memgraph namespace, but folders now provide relational cloisonnement on top of it: one physical document/chunk/vector/KG record can belong to N folders through `MEMBER_OF`. Folder-scoped reads traverse membership, not `n.folder`: document lists/counts, query grounding (chunks + KG vector selection + KG expansion), WebUI graph `source_docs`, tags/counters, frontend caches, upload duplicate-to-share, and the admin-only "Add to folder" UI all use the membership model. This is the canonical application of the "relationship over mutable property" doctrine: `workspace` = physical namespace; `folder` = logical many-to-many membership relation.

**Vrai Graph:** The Knowledge Graph tab reads live Memgraph projections through `/twin/api/graph/entities` and `/twin/api/graph/relations` via `server/graph_reader.py`, with a seed fallback only when Memgraph is empty/unreachable so demo and pre-ingestion paths still boot. PATCH/POST/DELETE graph endpoints persist through Memgraph helpers and emit activity. The graph still lives in the single LightRAG/Memgraph `{workspace}` namespace, but folder-bound graph reads are membership-scoped: entities/relations are visible only when at least one `source_docs` chunk belongs to a doc `MEMBER_OF` the active folder, and mixed-provenance text is masked where needed. Do not invent per-folder graph labels unless a future hard-isolation tier explicitly changes the storage model.

**Graph contract testing discipline:** Any PR touching graph reads, graph mutations, document deletion cascade, folder scoping, query-cache invalidation, or `source_docs` projection must add a contract-level test, not just a screenshot or component render. Cover the four sensitive axes explicitly: frontend cache keys/refetch (`graph-entities`, `graph-relations`), no inspector fallback to the first node after a selected node disappears, folder/header behavior, and `source_docs`-based document filtering/cascade behavior. MSW handlers must mutate graph state when the real backend would cascade, otherwise e2e can go falsely green.

**LightRAG compatibility discipline:** Every Twin extension must prove that the native LightRAG path stays behavior-identical when the extension is absent or flag-off. For server/runtime changes, add or update a test that exercises the native route/shim path and, where relevant, the stdlib smoke manifest in `tests/smoke/run_smoke.py`. This is part of the BNP positioning: Twin adds folders/tags/metadata/KV around LightRAG without breaking Eric's chunks+vectors path or Louis' "extension of the patch already in prod" story.

**Stack notes:**
- Bun runs everything: `bun install`, `bun run dev`, `bun run typecheck`, `bun run test:run`, `bun run build`.
- Vitest config is inline in `vite.config.ts`. `src/test/setup.ts` provisions an **in-memory localStorage** because happy-dom 20.x on Bun does not ship a Storage implementation.
- Design tokens (`--twin-*`, light + dark) live in `src/styles/tokens.css` as plain CSS variables; `tailwind.config.js` exposes them as utility classes (`bg-twin-accent`, `text-twin-green-700`, etc.).
- Modals emit typed `*Action` payloads on submit (RetagAction, AddSourceAction) — the host (App.tsx) owns the toast queue and the network call. **No `window.*` globals** *except* `window.__twinConfig` which is the server-injected runtime config (folders, idp, debugUser). Thesaurus / folders / notifications are injected via props.
- Typed fixtures in `src/fixtures/` are the **contract template** that the Python `server/webui_models.py` honors.
- **MSW + runtime config**: dev and standalone fixture runs use MSW (mocked at the browser worker level). `resolveRuntimeConfig()` decides at boot whether to install MSW vs hit the real backend. The `VITE_FORCE_MSW=1` env flag forces MSW even in a production build. Do not strip the MSW fallback from `resolveRuntimeConfig()`; local dev, e2e, and fixture-only review builds depend on it.

**Tests pitfalls** (also in `project_webui_fork.md`):
- `userEvent.type(input, 'foo{Enter}')` races on slow CI — split into two calls + `waitFor`.
- `useModalA11y`'s 30ms autofocus can steal mid-typing keystrokes from a non-first input — wait 60ms + force `.focus()` explicitly before typing.
- ARIA live regions duplicate visible text; scope `getByText` to a specific container or use `data-testid`.

## Test doctrine

Two non-negotiable rules — both reflect product-level constraints (graph centrality + BNP audit defense), not stylistic preference. Canonical docs:

- **[`docs/test-doctrine-graph.md`](docs/test-doctrine-graph.md)** — Graph = contract, not screen. Four sensitive axes (front cache, seed fallback, folder binding, `source_docs`). Every PR touching `graph_reader.py` / `GraphTab` / `chunks_vdb` / graph cache keys must add an end-to-end contract test (Cypher → API → cache → UI invalidation). Screenshots and Cypher-only unit tests do not count.
- **[`docs/test-doctrine-lightrag-compat.md`](docs/test-doctrine-lightrag-compat.md)** — Every Twin extension must ship a test proving the LightRAG-native path behaves identically when the extension is absent or its feature flag is off. The CI matrix is the coarse net; per-extension regression tests are the fine net. New Twin extension without a compat test → reject before review.

## Auth posture (audit 2026-06-10, relaxed to LightRAG parity same day)

**History matters**: the morning P0 hardening (H1 boot refusal + H2 `changeme` `ValueError`) crash-looped the BNP deployment — their env carries `TOKEN_SECRET` but no `LIGHTRAG_JWT_PASSWORD`, so the default "changeme" raised at app creation → pod crash-loop → nginx 503. Product decision the same afternoon: **credential defaults align on LightRAG native** (warn loudly, never refuse to boot). The strict RBAC posture activates ONLY via the IdP env switch (`TWIN_IDP_JWKS_URL`).

**Boot (H1, relaxed)** — no boot-time auth check. When neither `LIGHTRAG_API_KEY` nor `LIGHTRAG_JWT_SECRET`/`TOKEN_SECRET` is set, `require_auth` returns `None` (anonymous allowed, v1.0.x behaviour) and `configure_auth` logs `auth DISABLED`. Identical to LightRAG native. Do NOT reintroduce a boot raise on missing auth env — that's the exact regression that broke BNP on 2026-06-10.

**`changeme` (H2, relaxed)** — `configure_auth` logs a `SECURITY:` warning when `jwt_password == "changeme"` (and no `AUTH_ACCOUNTS`) or when any `AUTH_ACCOUNTS` value is `"changeme"`. Never raises. When no JWT secret is configured, `/login` is disabled and the password value is ignored.

**Shim router auth (C1, unchanged)** — `native_shims.build_native_shims_router(get_rag, auth_dependency=require_auth)`. Public routes: `/auth-status`, `/login`, `/logout` (handshake). `/health` is also public (LB probes). Everything else (`/documents`, `/documents/{id}/chunks`, `/documents/{id}/scan`, `/documents/{id}`, `/pipeline_status`, `/openapi`) sits behind `Depends(require_auth)` — which gates anonymous only when an auth backend is actually configured (LightRAG parity otherwise).

**Two-tier `require_admin_user` (H4)** — `server/idp_jwt.py`:
- *Palier 1 — IdP dormant* (`TWIN_IDP_JWKS_URL` unset): returns a placeholder dict (`idp_validated=False, gateway_scopes=[]`). The route-level `require_auth` already filtered anonymous, so what reaches the handler is at least an authenticated identity. Boot emits a `WARNING` once; per-call `INFO` rate-limited to once per process.
- *Palier 2 — IdP active*: requires the `admin:folders` gateway scope (projected by `claims_to_user` when the user's `groups` intersect `IdpConfig.admin_groups`). 401 on missing/invalid token, 403 on scope-missing.

**Folder header binding (C2)** — `server/folder.py:resolve_folder_for_request(request)` (use this, NOT the legacy `resolve_folder_from_headers(headers)`):
- *Palier 1*: identical to the pure header+catalog logic.
- *Palier 2*: header is bound to `user["folders"]` (the `twin_folders` claim). Folder not in user scope → 403. Empty claim (MyAccess rollout window) → only the catalog default folder is reachable.

The `bind_request_folder` dependency (FastAPI dep used by `webui_router`) now calls `resolve_folder_for_request`. The 3 shim handlers that previously took `request.headers` were migrated.

**Test posture** — no special conftest fixture needed (open-access default means tests boot without auth env). Test files: `tests/test_server/test_auth_defaults.py` (LightRAG-parity contract: open access + changeme warnings), `test_require_admin_two_tier.py` (H4), `test_native_shims_auth.py` (C1), `test_folder_idp_binding.py` (C2).

## Storage idioms (read these before touching impls)

- `_pool.get_driver()` returns `(driver, database)`. Always `await result.consume()` after `session.run(...)` — silent error swallowing on bulk indexation was the v0.3.2 fix that made stable.
- All write paths (`upsert`, `delete`, `drop`) must be wrapped with `async with acquire_write_slot():` — read paths must use `get_read_session()` (never `get_session()`) so they bypass the write throttle.
- Workspace + namespace + database + relation_type all go into Cypher via f-string label/identifier substitution. Validate via `_constants` before interpolating. **Never** interpolate property values — those go through `$param`.
- `except Exception: pass` for "index already exists" must check the message and re-raise everything else (Cassandre incident `SILENT_EXCEPTION_SWALLOWING`).

## Git workflow specific to this repo

Default remote = `bunker` (Forgejo at `192.168.1.61`). Day-to-day branches:

```
git push bunker <branch>
```

If `bunker` is missing: `git remote add bunker http://192.168.1.61:3000/<user>/<repo>.git`.

Stable branches are named `stable/X.Y.x` and protected on both `bunker` and `origin` (pre-receive hook on bunker; branch protection on GitHub). Direct push refused → open a PR.

The global directive §7 ("push to all remotes") does **not** apply here. `origin` is split-scope (see the Distribution section): a separate PR against GitHub `main` for a storage-only fix, a separate `export-1.0.0` rebuild for a BNP delivery, nothing else. Anything touching `server/` / `intelligence/` / `lightrag_webui_twin/` / doctrine docs stays on bunker.

**Push doctrine (do not project onto the user)**: pushing to `bunker` from this Mac is normal. Do **not** say "tu pushes depuis le LAN" or any equivalent — that framing was a temporary travel-period assumption, obsolete from 2026-06-10. When closing a session with a commit, either push if it was asked / is the natural follow-up, or ask explicitly. Never assume Julien will do it himself. See memory `feedback_codex_brief_push.md` — that rule is scoped to Codex briefs only, not to Claude on this Mac.
