# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This package provides Memgraph storage backends (KV, Vector, DocStatus) for [LightRAG](https://github.com/HKUDS/LightRAG) **without modifying LightRAG's source code**. LightRAG ships a graph backend; this package fills the remaining 3 slots so an entire instance runs on a single Memgraph DB.

### Branch policy (per project memory, 2026-05-20)

- `stable/0.6.x` — **active dev** branch. Carries the WebUI fork (`lightrag_webui_twin/`) and the `maquette-deploy/` artifacts. **Default working branch.**
- `stable/0.5.x` — **LTS, storage backends only**. WebUI/maquette/deploy work is pollution here and was force-reverted from it (PR #30, 2026-05-20).
- `stable/0.3.2-lts` — frozen LTS = 0.3.2 + auto-create vector index.
- `main` — tracks 0.5.x; `0.4.x` was abandoned.

Maquette work flows through short-lived `feat/maquette-*` branches cut off `stable/0.6.x` (often a dozen in flight at once — `git branch | grep maquette`). They merge back to `0.6.x`, not to `main`.

The live maquette at https://maquette.sigilum.fr/ runs from a `docker service` on OVH `37.59.104.111` and is independent of repo state — see `maquette-deploy/` below.

`pyproject.toml` reports version `1.0.0` and `register()` patches `lightrag.__version__` to `vX.Y.Z+memgraph-1.0.0` so the WebUI shows the composite version. See `changelog.md` for what's in vs. out of stable.

## Distribution

**Forgejo only** since 2026-05-11. GitHub (`origin` = `menezis-ai/lightrag-memgraph`) is being archived; do not push there. The active remote is `bunker` (Forgejo at `192.168.1.61`). The repo contains the **full** package — storage backends (KV / Vector / DocStatus), intelligence layer (TwinRAGEngine, ReAct, DSEP ontology), and server module. The previous public/private split (L1 GitHub + L2/L3 ZIP for BNP) is retired; no more `.gitignore` exclusions for `intelligence/` or `server/`.

## Commands

### Install

```bash
pip install -e ".[test]"
```

`uv.lock` is **not** present — use `uv` defaults if creating one (per global directive §4), but the project currently relies on `pip install -e`.

### Run the FastAPI overlay locally

```bash
python -m twindb_lightrag_memgraph.server   # uvicorn factory, default port
LIGHTRAG_PORT=8080 python -m twindb_lightrag_memgraph.server
```

Requires `MEMGRAPH_URI` + the LLM credentials read by `server/settings.py:LightRAGServerSettings`. See `WEBUI-WIRING-PLAN.md` (repo root) for the full Couche 2 / Couche 3 contract this server exposes to the WebUI fork.

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

### SonarQube

SonarQube is permanently available at `http://192.168.1.212:9000` (per project memory; not the address in the global directive — that one is `192.168.1.49:9000`, a different instance). `sonar-scanner` is at `/opt/homebrew/bin/sonar-scanner`. Token must be provided via env, never committed.

Scanner config lives in `sonar-project.properties` at the repo root: `sonar.sources=src`, `sonar.tests=tests`, Python version matrix `3.10,3.11,3.12,3.13,3.14`, coverage report at `coverage.xml`. Reuse this file rather than passing flags ad hoc.

`aquery()` cognitive complexity must stay ≤ 15 — keep helper methods extracted (project memory).

### CI matrix

CI (`.forgejo/workflows/ci.yml`) runs on Forgejo Actions, all jobs on the self-hosted `ubuntu-latest-docker` runner:
- **unit-tests**: Python 3.10/3.11/3.12/3.13 × LightRAG 1.4.9 / 1.4.9.11 / 1.4.11 / 1.4.12 (no Memgraph). Runs `pytest tests/ --ignore=tests/test_bench.py` — the `@pytest.mark.integration` auto-skip in `conftest.py` keeps storage/e2e tests out when `MEMGRAPH_URI` is unset.
- **integration-tests**: LightRAG matrix × Memgraph 3.7.2 / 3.8.0 / latest, with a `memgraph` service container. `max-parallel: 5` to keep cold-start within the 60-retry health window. URI: `bolt://memgraph:7687`.
- **webui-tests**: `bun install --frozen-lockfile && bun run typecheck && bun run test:run && bun run build` on the `lightrag_webui_twin/` Vite project (see "WebUI fork" below). Bun is pinned to `1.3.6` in the workflow — never use `latest` (api.github.com rate limit).

LightRAG `1.4.10` is **dropped** from the matrix (issue #6) — intermittent test failures under integration load, fixed upstream in 1.4.11+.

Branch protection on `main`, `stable/0.5.x`, `stable/0.3.2-lts` requires patterns `CI / unit-tests*`, `CI / integration-tests*`, `CI / webui-tests*` to be green before merge.

A push triggers 1–15 min of CI (global directive §6). Run unit + integration locally first; do not push speculatively.

## Architecture

### Storage package (`src/twindb_lightrag_memgraph/`)

- `__init__.py` — `register()` monkey-patches three dicts in `lightrag.kg`: `STORAGE_IMPLEMENTATIONS`, `STORAGE_ENV_REQUIREMENTS`, `STORAGES`. Module paths in `STORAGES` **must be absolute** (`twindb_lightrag_memgraph.kv_impl`, not relative) because `lazy_external_import` resolves with `package="lightrag"`. Idempotent via `_registered` flag. Also patches `lightrag.__version__` to append `+memgraph-{version}` so the WebUI shows `core_version` like `v1.4.9.11+memgraph-0.5.3`. Must be called **before** `LightRAG(...)`.
- `_pool.py` — Two independent async Bolt drivers (write + read) as module-level singletons; both detect event-loop changes and rebuild on a thread-safe lock. `acquire_write_slot()` is an `asyncio.Semaphore` (default 10) wrapping every write. Read sessions (`get_read_session()`) are never throttled. URI scheme determines whether `database=` is passed natively (`neo4j://...`) or via `USE DATABASE` (`bolt://...`); on Memgraph Community, `USE DATABASE` fails on first attempt and is silently skipped thereafter. The graph backend (built into LightRAG) keeps **its own** driver — production has 3 pools by design.
- `_buffered_graph.py` — `_BufferedGraphProxy` wraps the graph storage during `merge_nodes_and_edges`, accumulating `upsert_node`/`upsert_edge`, then flushes nodes-then-edges as 2 UNWIND queries. Supports read-your-own-writes via in-memory buffer checks before delegating. Reduces ~130 round-trips/doc to 2–3.
- `_hooks.py` — Post-indexation hooks.
- `_constants.py` — Validators, defaults, env var names. **Identifier validator** here is the canonical place for any new label / database / namespace input (Cassandre flagged Cypher injection on f-string interpolation of these — labels use backticks, but `USE DATABASE` and ontology `rel_type` from LLM output do not).
- `kv_impl.py` — `MemgraphKVStorage`. Label `KV_{workspace}_{namespace}`. Value dict serialized to single `data` JSON string. Batch via UNWIND + MERGE.
- `vector_impl.py` — `MemgraphVectorDBStorage`. Label `Vec_{workspace}_{namespace}`, vector index via `CREATE VECTOR INDEX ... WITH CONFIG {dimension, capacity, metric: "cos"}`. `query()` uses MAGE `vector_search.search()` and **auto-creates the index if missing then retries once** (added in 0.5.1 — silent `[no-context]` fix). `cosine_better_than_threshold` read from `global_config["vector_db_storage_cls_kwargs"]`, default 0.2.
- `docstatus_impl.py` — `MemgraphDocStatusStorage`. Label `DocStatus_{workspace}` (no namespace suffix). Indexes on `id`, `status`, `file_path`, `track_id`, `updated_at`, `created_at`. Sequential upserts (per-item `DocProcessingStatus`-vs-dict serialization). `get_docs_paginated` runs count + fetch in parallel via `asyncio.gather` over two read sessions. Sort fields are whitelisted against injection.
- `_spaces.py` — Env-only Twin space catalog parser (`TwinSpace`, `TwinSpaceCatalog`, `load_space_catalog`). Deliberately FastAPI-free so it can feed `_build_runtime_config()` even when only `replace_ui=True`. Server-side runtime-mutable persistence sits in `server/space_store.py`.

The package also patches `MemgraphGraphStorage` with batch read methods (`get_nodes_batch`, `node_degrees_batch`, `get_edges_batch`, `get_nodes_with_degrees_batch`, etc.) replacing N sequential queries with single UNWIND queries.

### Server module (`src/twindb_lightrag_memgraph/server/`)

FastAPI app factory that sits on top of `register()` and gives the WebUI fork a real backend (Couche 3 of `WEBUI-WIRING-PLAN.md`). Run with `python -m twindb_lightrag_memgraph.server` (uvicorn factory mode; `LIGHTRAG_PORT=8080` to override). Layout:

- `app.py` — `create_app()` factory. Mounts CORS, dual auth (static API key + JWT for the CFT agent), LangSmith tracing, `/query`, `/insert`, `/health`, and the chunk/document routes. `settings.py` (`LightRAGServerSettings`) provides runtime config; `auth.py` is the auth router; `tracing.py` does trace-parent propagation.
- `chunk_routes.py` — `/chunks/*` and `/documents/*` endpoints (P3 context expansion) that the WebUI's DocDetailPanel + RetrievalTab consume.
- `webui_router.py` + `webui_models.py` + `webui_*store.py` (`webui_tagstore`, `webui_activitystore`, `webui_notificationstore`) — Twin overlay endpoints (`/twin/api/{tags,activity,notifications,...}`). Pydantic models in `webui_models.py` are the Python side of the contract whose TypeScript twin lives in `lightrag_webui_twin/src/fixtures/`. `webui_seed.py` seeds them for demo/test runs.
- `space.py` + `space_store.py` — Twin space request binding + admin CRUD. `space.py` exposes `require_active_space()` (resolves `X-Twin-Space` / `X-Twin-Workspace`, raises 401/404 on unknown) and a `ContextVar` so downstream code reads the active space without thread argument-passing. `space_store.py` persists operator-added spaces (in-memory by default; JSON file when `TWIN_SPACES_RUNTIME_FILE` is set). Env-seeded spaces always win id collisions; the runtime store can't shadow the SRE-provisioned default. See `_spaces.py` (storage pkg) for the env-only catalog parser.
- `idp_jwt.py` — IdP middleware (Couche 3 §3.3). Verifies JWTs from cookie or `Authorization` header against a JWKS endpoint (TTL-cached) and projects claims into the `AuthenticatedUser` shape the React port consumes (`lightrag_webui_twin/src/types/auth.ts`). Claim names are fully env-mapped so BNP-specific bindings don't need a code change. `require_admin_user()` gates admin routes through the `admin:spaces` gateway scope, granted from `TWIN_IDP_ADMIN_GROUPS`. Dormant when `TWIN_IDP_JWKS_URL` is unset — auth falls back to the static API key / legacy JWT branches in `auth.py`, and `_build_runtime_config` keeps `debugUser` alive for dev / OVH standalone.
- `native_shims.py` — Shadow router that re-shapes LightRAG's native FastAPI surface to the React port's contract (`GET /documents`, `GET /documents/{id}/chunks`, projected `/health`, `/pipeline_status`, etc.). Mounted by `register(shim_native_routes=True)` at the **head** of `app.router.routes` so shims win the match against LightRAG's later `include_router(...)`. Doctrine: WebUI contract is source of truth; LightRAG routes get translated, not vice-versa. Coverage table is in the module docstring.
- `graph_reader.py` — Read-only Cypher → `GraphEntity`/`GraphRelation` mapper for the WebUI Knowledge Graph tab (M12 batch 1). Workspace-scoped queries, best-effort `entity_type` → closed enum mapping, deterministic hash-bucket layout so positions are stable across page loads without a force simulation. PATCH persistence is M12 batch 2 (still upcoming).
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

LLM call pattern (project memory): `AsyncOpenAI(api_key=..., base_url=...)` with `response_format={"type": "json_object"}`. Patch target for tests is the **exact module import path** (e.g., `twindb_lightrag_memgraph.intelligence.ontology.steps.extract.AsyncOpenAI`), not the original `openai.AsyncOpenAI`.

User preference: config files in **JSON, not YAML** (no extra dependency, ecosystem consistency).

### WebUI fork (`lightrag_webui_twin/`)

Sibling Vite + Bun + React 19 + TypeScript strict + Tailwind v3 sub-project. Ports the design proto at `/Users/julien/Desktop/UI/` (untouched reference) into a typed, tested codebase that will eventually serve as the Twin operator console (citations cliquables, UI tag rétroactif, sous-graphe filtré par tag, source isolation badge).

**Roadmap** — S1/S2/S3/S4a/S4b/S4c are landed (see memory `project_webui_fork.md`). The live end-to-end mutation loop (WebUI → FastAPI → Memgraph store → cache invalidation → refetch) is in place. **Couche 3** progress: runtime config injection + frontend space cutover landed 2026-06-02 (PR #170); backend `X-Twin-Space` scoping guard + admin space CRUD landed via the hardening commits on this branch (`server/space.py`, `server/space_store.py`). Real Memgraph graph reader (M12 batch 1) sits in `server/graph_reader.py`; PATCH persistence (M12 batch 2) is the next chunk on this branch. **Authoritative status lives in `WEBUI-WIRING-PLAN.md` at the repo root** — re-read it before claiming any Couche/Gxxx item is closed.

**Spaces vs workspaces (2026-06-01 Fabrice doctrine, 2026-06-02 cutover):** The canonical term is **Space** — confirmed by Fabrice during the 2026-06-01 meeting ("Space alors. Juste enlève le workspace, c'est space."). The Twin overlay surface (routes, Pydantic models, audit events, frontend props) uses `space`. The LightRAG-aligned label that lands on Memgraph nodes (`KV_{ws}`, `Vec_{ws}`, `DocStatus_{ws}`, …) is still spelled `workspace` because that's an upstream contract we don't get to rename — see the module docstring in `src/twindb_lightrag_memgraph/_constants.py`. The React port no longer hardcodes any space id — initial space comes from `window.__twinConfig.defaultSpaceId`. The server injects spaces via `_build_runtime_config()` from env vars:

- `TWIN_DEFAULT_SPACE` (fallback `WORKSPACE`, then `default`)
- `TWIN_DEFAULT_SPACE_LABEL`
- `TWIN_SPACES_JSON` — JSON array of `{id, label, kind, description, sources}`
- `TWIN_MAX_SPACES` — clamped 1..5 (one SRE default + up to four admin-created)

The HTTP client sends `X-Twin-Space` AND `X-Twin-Workspace` in parallel during the migration. Don't strip the dual-header window until the backend contract has fully migrated. The Empty state when no space is provisioned: `No space available for this KB. Please contact Twincore Team.`

**Vrai Graph (M12):** The Knowledge Graph tab in the React port is UI-only (`GRAPH_ENTITY_FIXTURES` + MSW state). PATCH endpoints persist in-memory MSW only. Real Memgraph wiring (Cypher `MATCH (n)`, real PATCH, layout strategy) is tracked under milestone M12 (#25). Don't conflate the UI port (landed PR #169) with the graph being real.

**Stack notes:**
- Bun runs everything: `bun install`, `bun run dev`, `bun run typecheck`, `bun run test:run`, `bun run build`.
- Vitest config is inline in `vite.config.ts`. `src/test/setup.ts` provisions an **in-memory localStorage** because happy-dom 20.x on Bun does not ship a Storage implementation.
- Design tokens (`--twin-*`, light + dark) live in `src/styles/tokens.css` as plain CSS variables; `tailwind.config.js` exposes them as utility classes (`bg-twin-accent`, `text-twin-green-700`, etc.).
- Modals emit typed `*Action` payloads on submit (RetagAction, AddSourceAction) — the host (App.tsx) owns the toast queue and the network call. **No `window.*` globals** *except* `window.__twinConfig` which is the server-injected runtime config (spaces, idp, debugUser). Thesaurus / spaces / notifications are injected via props.
- Typed fixtures in `src/fixtures/` are the **contract template** that the Python `server/webui_models.py` honors.
- **MSW + runtime config**: dev and the OVH standalone demo run on MSW (mocked at the browser worker level). `resolveRuntimeConfig()` decides at boot whether to install MSW vs hit the real backend. The `VITE_FORCE_MSW=1` env flag forces MSW even in a production build — that is how `https://maquette.sigilum.fr/` ships a static SPA with no backend. Do not strip the MSW fallback from `resolveRuntimeConfig()`; both prod-with-backend and prod-standalone-demo depend on it.

**Tests pitfalls** (also in `project_webui_fork.md`):
- `userEvent.type(input, 'foo{Enter}')` races on slow CI — split into two calls + `waitFor`.
- `useModalA11y`'s 30ms autofocus can steal mid-typing keystrokes from a non-first input — wait 60ms + force `.focus()` explicitly before typing.
- ARIA live regions duplicate visible text; scope `getByText` to a specific container or use `data-testid`.

### Maquette deploy (`maquette-deploy/`)

OVH demo packaging at `https://maquette.sigilum.fr/` (VPS `37.59.104.111`). Two containers: Caddy + the static SPA (`source/`), and a FastAPI + SQLite persistence sibling (`backend/`). `Dockerfile`/`Caddyfile` build the legacy JSX prototype; `Dockerfile.react`/`Caddyfile.react` build the React port (the typed WebUI fork) — same SPA contract, different source tree. Full deploy procedure (stage tarball → `scp` → `docker stack deploy`) lives in `maquette-deploy/README.md`; UI conventions in `STYLES.md`; the backend contract the SPA depends on is in `BACKEND_CONTRACT.md`. **The maquette is a vitrine, not a deliverable** — its FastAPI/SQLite layer is intentionally a toy and must not be confused with `src/twindb_lightrag_memgraph/server/`. Don't audit it as a production system.

## Storage idioms (read these before touching impls)

- `_pool.get_driver()` returns `(driver, database)`. Always `await result.consume()` after `session.run(...)` — silent error swallowing on bulk indexation was the v0.3.2 fix that made stable.
- All write paths (`upsert`, `delete`, `drop`) must be wrapped with `async with acquire_write_slot():` — read paths must use `get_read_session()` (never `get_session()`) so they bypass the write throttle.
- Workspace + namespace + database + relation_type all go into Cypher via f-string label/identifier substitution. Validate via `_constants` before interpolating. **Never** interpolate property values — those go through `$param`.
- `except Exception: pass` for "index already exists" must check the message and re-raise everything else (Cassandre incident `SILENT_EXCEPTION_SWALLOWING`).

## Git workflow specific to this repo

Forgejo-only since 2026-05-11. Push to `bunker` (Forgejo at `192.168.1.61`), **not** to `origin` (GitHub is being archived):

```
git push bunker <branch>
```

If `bunker` is missing: `git remote add bunker http://192.168.1.61:3000/<user>/<repo>.git`.

Stable branches are named `stable/X.Y.x` and protected on the Forgejo remote.

The global directive §7 ("push to all remotes") does **not** apply here — the dual-remote era ended with the GitHub archive. The `origin` remote is left configured for historical pull only.
