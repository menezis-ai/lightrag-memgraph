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
- Run final release-candidate validation from a clean checkout/image.

Recent verification:

- `uv run pytest tests/test_server -q`: 652 passed, 38 skipped, 0 warnings.
- `npm audit --omit=dev`: 0 vulnerabilities.
- `npm run lint`, `npm run typecheck`, and targeted `useUrlParam` tests passed.

## Ground Rules For Agents

Do not start with broad refactors unless explicitly assigned. Each agent should take one workstream, keep changes scoped, and preserve existing behavior unless the task explicitly changes production policy.

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

Coordination rules:

- Refactors should be split into focused PRs with no behavior change.
- Agents should not run overlapping edits on the same large module.
- Any production behavior change must include docs and tests in the same PR.

## Known Dirty Worktree At Plan Creation

At the time this plan was created, the worktree had one unrelated untracked file:

```text
?? docs/TwinRAG - Visite guidee (standalone).html
```

Agents must not delete or modify it unless explicitly assigned.
