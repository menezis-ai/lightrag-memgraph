# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This package provides Memgraph storage backends (KV, Vector, DocStatus) for [LightRAG](https://github.com/HKUDS/LightRAG) **without modifying LightRAG's source code**. LightRAG ships a graph backend; this package fills the remaining 3 slots so an entire instance runs on a single Memgraph DB.

The current working branch is `stable/0.5.x` (LTS = 0.3.2 + auto-create vector index). The `main` branch tracks 0.5.x as well; `0.4.x` was abandoned. See `changelog.md` for what's in vs. out of stable.

## Distribution

**Forgejo only** since 2026-05-11. GitHub (`origin` = `menezis-ai/lightrag-memgraph`) is being archived; do not push there. The active remote is `bunker` (Forgejo at `192.168.1.61`). The repo contains the **full** package — storage backends (KV / Vector / DocStatus), intelligence layer (TwinRAGEngine, ReAct, DSEP ontology), and server module. The previous public/private split (L1 GitHub + L2/L3 ZIP for BNP) is retired; no more `.gitignore` exclusions for `intelligence/` or `server/`.

## Commands

### Install

```bash
pip install -e ".[test]"
```

`uv.lock` is **not** present — use `uv` defaults if creating one (per global directive §4), but the project currently relies on `pip install -e`.

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
docker run -d --name memgraph-test -p 7687:7687 memgraph/memgraph-mage:latest
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

### SonarQube

SonarQube is permanently available at `http://192.168.1.212:9000` (per project memory; not the address in the global directive — that one is `192.168.1.49:9000`, a different instance). `sonar-scanner` is at `/opt/homebrew/bin/sonar-scanner`. Token must be provided via env, never committed.

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

The package also patches `MemgraphGraphStorage` with batch read methods (`get_nodes_batch`, `node_degrees_batch`, `get_edges_batch`, `get_nodes_with_degrees_batch`, etc.) replacing N sequential queries with single UNWIND queries.

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

**Roadmap** — S1 (scaffold + 3 leaves + 2 hooks + typed fixtures) and S2 (3 modals + DocumentsTab + RetrievalTab) merged 2026-05-12. S3 (tags / activity / api / graph + CSS class-body port + TweaksPanel) and S4 (real network wiring against backend phase 1) remain. See memory `project_webui_fork.md` for the detailed plan, current state, and pitfalls archive.

**Stack notes:**
- Bun runs everything: `bun install`, `bun run dev`, `bun run typecheck`, `bun run test:run`, `bun run build`.
- Vitest config is inline in `vite.config.ts`. `src/test/setup.ts` provisions an **in-memory localStorage** because happy-dom 20.x on Bun does not ship a Storage implementation.
- Design tokens (`--twin-*`, light + dark) live in `src/styles/tokens.css` as plain CSS variables; `tailwind.config.js` exposes them as utility classes (`bg-twin-accent`, `text-twin-green-700`, etc.).
- 114 unit tests across 11 test files, ~1.6s. Modals emit typed `*Action` payloads on submit (RetagAction, AddSourceAction) — the host (App.tsx) owns the toast queue and the network call. **No `window.*` globals**; thesaurus / workspaces / notifications are injected via props.
- Typed fixtures in `src/fixtures/` are the **contract template for backend phase 1**: `/documents`, `/workspaces`, `/notifications`, `/tags`, `/retrieval` endpoints will honor these shapes.

**Tests pitfalls** (also in `project_webui_fork.md`):
- `userEvent.type(input, 'foo{Enter}')` races on slow CI — split into two calls + `waitFor`.
- `useModalA11y`'s 30ms autofocus can steal mid-typing keystrokes from a non-first input — wait 60ms + force `.focus()` explicitly before typing.
- ARIA live regions duplicate visible text; scope `getByText` to a specific container or use `data-testid`.

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
