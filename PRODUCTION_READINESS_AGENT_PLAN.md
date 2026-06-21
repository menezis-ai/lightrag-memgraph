# Production Readiness Agent Plan

Date: 2026-06-21

Audience: agents and engineers working on `twindb-lightrag-memgraph`.

Purpose: track only the remaining work needed to fully close production readiness. Completed P0/P1 items have been removed from this file.

## Executive Summary

P0/P1 production-readiness hardening has been implemented on branch
`production-readiness-p0-p1`:

- Production auth now fails closed when `TWIN_ENV=production` or
  `TWIN_REQUIRE_AUTH=true`.
- Insecure production JWT defaults and weak HMAC secrets are rejected.
- Python runtime constraints are committed and wired into Docker/CI.
- CI has Python and frontend production audit gates.
- Frontend npm audit findings are cleared.
- OpenAPI duplicate operation ID warnings are fixed.
- Routine backend test warning noise is cleared.
- Container examples no longer use `memgraph/memgraph-mage:latest`.

Current residual work is:

- Resolve or formally risk-accept the deferred LightRAG 1.4.9.11 CVEs.
- Split oversized backend/frontend modules before the next feature wave.
- Add operational readiness endpoints, request-size posture, and observability baseline.
- Run final release-candidate validation from a clean checkout/image.

Recent verification:

- `uv run pytest tests/test_server -q`: 652 passed, 38 skipped, 0 warnings.
- `npm audit --omit=dev`: 0 vulnerabilities.
- `npm run lint`, `npm run typecheck`, and targeted `useUrlParam` tests passed.

## Ground Rules For Agents

Do not start with broad refactors unless assigned to a P2 modularization item. Each agent should take one workstream, keep changes scoped, and preserve existing behavior unless the task explicitly changes production policy.

Before editing:

1. Run `git status --short`.
2. Do not revert unrelated user changes.
3. Read the local tests covering the files you plan to modify.
4. Prefer existing patterns and helpers.
5. Add or update tests with every behavior change.

Definition of done for any workstream:

1. Relevant unit tests pass.
2. Relevant frontend checks pass when frontend is touched.
3. No unrelated file churn.
4. Documentation or env reference is updated when runtime behavior changes.
5. The final PR summary states risk, tests, and rollback notes.

## Priority Map

P0 is the deferred security follow-up that still blocks a clean "nothing left" production-readiness plan.

P2 improves maintainability and lowers future change risk.

P3 is operational hardening after the first release-readiness gates.

## P0. Deferred Security Follow-Up

### P0-1. Resolve LightRAG 1.4.9.11 CVE Posture

Objective: remove or formally risk-accept the known LightRAG 1.4.9.11 advisories.

Context:

- The project intentionally remains on `lightrag-hku==1.4.9.11`.
- CI currently ignores `CVE-2026-30762` and `CVE-2026-39413` for that pinned version.
- This was explicitly deferred and must be revisited before declaring the plan empty.

Likely files:

- `requirements/prod-target.txt`
- `requirements/constraints-prod.txt`
- `.forgejo/workflows/ci.yml`
- `docs/operations/install-runbook.md`
- `ENV_VARIABLES.txt` if the mitigation changes operator guidance

Options:

1. Upgrade LightRAG after compatibility validation.
2. Backport or mitigate the vulnerable surface locally.
3. Keep `1.4.9.11` and add a dated, owner-approved risk acceptance with exploitability rationale and review date.

Acceptance criteria:

- `pip-audit` production gate has no unexplained LightRAG ignores, or each ignore has a documented owner, rationale, and review date.
- Compatibility with Memgraph storage patches remains covered by tests.
- The chosen posture is documented for release reviewers.

Suggested verification:

```bash
uvx pip-audit -r requirements/constraints-prod.txt --no-deps --disable-pip
uv run pytest tests/test_register.py tests/test_upstream_compat.py tests/test_lightrag_server_entrypoint.py -q
```

## P2. Modularization And Maintainability

### P2-1. Split `webui_router.py`

Objective: reduce a 2447-line route/store module into maintainable units without changing behavior.

Current problem:

- `webui_router.py` contains models orchestration, store management, folders, documents, tags, graph, activity, notifications, and route definitions.

Recommended target structure:

```text
src/twindb_lightrag_memgraph/server/webui/
  __init__.py
  router.py
  store.py
  routes_documents.py
  routes_tags.py
  routes_graph.py
  routes_activity.py
  routes_notifications.py
  routes_folders.py
  events.py
```

Constraints:

- Preserve existing import path if external callers import `server.webui_router.router`.
- Keep a compatibility shim in `webui_router.py` initially.

Acceptance criteria:

- No route contract changes.
- Existing tests pass.
- `webui_router.py` becomes a thin compatibility module.

Suggested verification:

```bash
uv run pytest tests/test_server/test_webui_router.py tests/test_server/test_webui_router_graph.py tests/test_server/test_webui_router_mutations.py -q
```

### P2-2. Split `App.tsx`

Objective: move application orchestration out of a 1594-line component.

Recommended target:

```text
lightrag_webui_twin/src/app/
  AppShell.tsx
  queryClient.ts
  useAppData.ts
  useAppNavigation.ts
  useDocumentActions.ts
  useTagActions.ts
  useToasts.ts
```

Constraints:

- Preserve current UI behavior and tests.
- Avoid broad design changes during this refactor.

Acceptance criteria:

- `App.tsx` is below 300 lines.
- Unit tests pass.
- E2E smoke paths still pass.

Suggested verification:

```bash
cd lightrag_webui_twin
npm run typecheck
npm run lint
npm run test:run
npm run test:e2e
```

### P2-3. Split `GraphTab.tsx`

Objective: make the graph UI maintainable.

Current problem:

- `GraphTab.tsx` is approximately 2347 lines.

Recommended target:

```text
lightrag_webui_twin/src/components/Graph/
  GraphTab.tsx
  GraphCanvas.tsx
  GraphFilters.tsx
  GraphInspector.tsx
  graphLayout.ts
  graphSelection.ts
  graphTypes.ts
```

Acceptance criteria:

- Existing graph tests pass.
- No visual/layout regression in Playwright graph spec.
- Top-level `GraphTab.tsx` becomes reviewable.

Suggested verification:

```bash
cd lightrag_webui_twin
npm run test:run -- GraphTab
npm run test:e2e -- e2e/graph.spec.ts
```

### P2-4. Split LightRAG Patching From `__init__.py`

Objective: reduce risk in the package entrypoint and make patch behavior easier to test.

Current problem:

- `src/twindb_lightrag_memgraph/__init__.py` is approximately 1943 lines and owns too many patching concerns.

Recommended target:

```text
src/twindb_lightrag_memgraph/patches/
  __init__.py
  registry.py
  security_baseline.py
  builtin_memgraph.py
  merge_write_path.py
  insert_done.py
  version.py
  server_create_app.py
  native_route_capture.py
```

Constraints:

- Public API must remain `from twindb_lightrag_memgraph import register`.
- Preserve idempotency.
- Preserve import timing guarantees, especially security baseline before LightRAG API imports.

Acceptance criteria:

- `register()` public behavior unchanged.
- Storage-only mode remains unaffected.
- Tests around upstream compatibility still pass.

Suggested verification:

```bash
uv run pytest tests/test_register.py tests/test_upstream_compat.py tests/test_lightrag_server_entrypoint.py -q
```

### P2-5. Reduce Query Route Complexity

Objective: make `twin_query_routes.py` easier to reason about and test.

Current problem:

- Query, streaming, source projection, tag/doc filtering, activity recording, and LightRAG compatibility all live in one large file.

Recommended target:

```text
src/twindb_lightrag_memgraph/server/query/
  router.py
  models.py
  params.py
  sources.py
  filters.py
  stream.py
  activity.py
```

Acceptance criteria:

- `/twin/api/query`, `/twin/api/query/data`, and `/twin/api/query/stream` contracts unchanged.
- Tests for grounded vs insufficient information still pass.

Suggested verification:

```bash
uv run pytest tests/test_server/test_twin_query_routes.py tests/test_query_modes.py -q
```

## P3. Operational Hardening

### P3-1. Runtime Health And Readiness

Objective: distinguish "process alive" from "ready for traffic".

Recommended additions:

- `/health` remains lightweight.
- `/ready` verifies Memgraph connectivity, LightRAG initialized, vector index callable if practical, and auth production policy loaded.

Likely files:

- `src/twindb_lightrag_memgraph/server/app.py`
- `src/twindb_lightrag_memgraph/server/quota.py`
- `tests/test_server/test_twin_health_endpoint.py`

Acceptance criteria:

- Readiness fails when Memgraph is unreachable.
- Liveness stays cheap.
- Docs explain which endpoint to use for Kubernetes probes.

### P3-2. Request Size And Upload Limits

Objective: reduce DoS exposure from large form or upload bodies.

Likely files:

- `src/twindb_lightrag_memgraph/server/app.py`
- native LightRAG shim files
- deployment docs

Acceptance criteria:

- Upload limits are explicit and testable.
- Operators know where to configure them.

### P3-3. Observability Baseline

Objective: make production incidents diagnosable.

Recommended additions:

- Structured logs for request ID, folder, auth mode, route group, latency.
- Metric counters for ingestion failures, query failures, quota rejects, auth rejects.
- Trace correlation in retrieval routes.

Likely files:

- `src/twindb_lightrag_memgraph/server/tracing.py`
- `src/twindb_lightrag_memgraph/server/app.py`
- route modules after P2 splits

Acceptance criteria:

- Logs do not expose bearer tokens, JWTs, API keys, document body content, or raw prompts unless explicitly enabled in a secure debug mode.
- Each 5xx path logs enough context to identify failing subsystem.

## Release Candidate Checklist

Before tagging a production release candidate, all of the following must be true:

- Deferred LightRAG CVE posture is resolved or formally risk-accepted.
- Backend tests pass from a clean checkout.
- Frontend lint/typecheck/unit/build pass from a clean checkout.
- Playwright MSW E2E pass.
- Real-backend E2E pass in Forgejo.
- Docker image builds from a clean checkout.
- Runtime smoke test passes against the built image.
- `ENV_VARIABLES.txt` and `docs/operations/install-runbook.md` reflect actual production settings.
- Dependency audit reports are attached or linked.

Suggested full local verification:

```bash
uv run pytest -q
cd lightrag_webui_twin
npm run lint
npm run typecheck
npm run test:run
npm run build
npm audit --omit=dev
```

Suggested CI/staging verification:

```bash
python tests/smoke/run_smoke.py tests/smoke/runtime-smoke.json
```

## Agent Work Allocation

Suggested parallelization:

- Agent A: P0-1 LightRAG CVE posture.
- Agent B: P2-1 backend WebUI router split.
- Agent C: P2-2 and P2-3 frontend modularization.
- Agent D: P2-4 and P2-5 patch/query modularization.
- Agent E: P3 operational readiness endpoints, limits, and observability.

Coordination rules:

- P2 refactors should be split into multiple PRs with no behavior change.
- Agents should not run overlapping edits on the same large module.
- Any production behavior change must include docs and tests in the same PR.

## Known Dirty Worktree At Plan Creation

At the time this plan was created, the worktree had one unrelated untracked file:

```text
?? docs/TwinRAG - Visite guidee (standalone).html
```

Agents must not delete or modify it unless explicitly assigned.
