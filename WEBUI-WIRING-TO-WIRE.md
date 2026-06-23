# Twin KMS WebUI Wiring — To Wire

This is the active backlog for the WebUI/backend contract. The live state is in
[WEBUI-WIRING-PLAN.md](WEBUI-WIRING-PLAN.md); the implemented inventory is in
[WEBUI-WIRING-WIRED.md](WEBUI-WIRING-WIRED.md).

## Priority 1 — Stabilize Real MyAccess / IdP Deployment

The code path is present. Production wiring is not finished until it is proven
against the real IdP, not only mocked JWKS tests.

Current mechanics:

- `ensure_auth_backend_configured` fails closed in production unless an auth
  backend is configured.
- `TWIN_IDP_JWKS_URL` activates IdP JWT verification.
- Missing, expired, malformed, or invalid-signature tokens return 401.
- `admin:folders` is enforced for Folder mutations once IdP claims are active.
- Folder access can be constrained by the user's `twin_folders` claim, with the
  default-folder fallback kept for rollout.
- API-key auth remains supported for CI, service use, and generated-key e2e.

Remaining work:

- Wire the real MyAccess JWKS URL, issuer, audience, cookie/header convention,
  and group-to-scope mapping in deployment config.
- Run an integration smoke against the real IdP path, not only PyJWK mocks.
- Document the exact MyAccess claims consumed by Twin KMS in the install
  runbook.
- Decide whether the default-folder fallback remains after rollout or becomes a
  hard deny when `twin_folders` is absent.

## Priority 2 — Make Deployment Smoke Boring

The smoke tooling exists, but it still needs to become the normal post-deploy
ritual.

Minimum smoke checklist:

- `/webui/` serves the Twin KMS build and contains no unresolved
  `__TWIN_CONFIG_JSON__` placeholder.
- Runtime config resolves `apiBaseUrl` to `/twin/api`.
- `/health` and `/twin/api/health` both answer as expected.
- `/twin/api/folders` returns the env-injected catalog.
- `/twin/api/settings/api-keys` is reachable for an authorized admin/operator.
- `/twin/api/quota` returns a structured snapshot.
- `/twin/api/graph/entities` returns real Memgraph-backed data or an honest
  empty result, never fixture data.
- Anonymous requests to protected Twin routes are rejected in production.

Remaining work:

- Keep `docs/operations/install-runbook.md` aligned with the smoke script.
- Record the exact command set for OVH/twin-real and the Forgejo deployment
  lane.
- Add one small "known good deploy" evidence block per release candidate.

## Priority 3 — Reduce CI Runner Surprise

The CI has the right coverage, but self-hosted Docker jobs must stay isolated.

Remaining work:

- Keep real-backend Playwright jobs on dynamically allocated host ports.
- Ensure every Docker-backed lane removes containers and networks on failure.
- Consider a lightweight preflight that prints any leftover `twin-ci-*`
  containers before starting real-backend lanes.
- Keep npm/Bun usage explicit: Bun for runner-local frontend quality/build where
  declared, npm inside Playwright containers.
- Avoid adding fixed host ports to future jobs unless the runner pool guarantees
  isolation.

## Priority 4 — Performance and Polling Cleanup

There is known useful work from the historical perf branch, but it should be
rebased and reviewed against the current Twin KMS code before merge.

Candidate items:

- Lazy-load secondary tabs and modal bodies.
- Gate inactive tab queries with `enabled` flags.
- Bound activity reads with an explicit `limit`, default, and max.
- Add or verify scalar/indexed Memgraph fields for high-volume WebUI stores.

Acceptance criteria:

- Typecheck, unit tests, and e2e still pass.
- The initial WebUI bundle meaningfully shrinks or the runtime polling load
  measurably drops.
- No query becomes stale in normal operator navigation.

## Priority 5 — Retention / Sweep Policy

Deferred by policy. Do not implement a silent default.

Required decisions:

- TTL per store: tags, activity, notifications, audit logs.
- Sweep mechanism: scheduled Cypher purge, read-time filtering, or external
  retention tooling.
- Scope: per Folder, per KB, sandbox-only, or global.
- BCE/DORA/legal-hold behavior, including how legal hold suspends deletion.

## PO-Gated / Do Not Start Without Confirmation

- BNP MIP classification and ingestion hook. Existing modules remain opt-in via
  `register(classify=True)` or `TWIN_MIP_LABEL_MAP`.
- Any durable raw-document storage, source download route, or full-document
  preview for sensitive material.
- Provider configuration panels in Settings.
- Role-perspective simulator for RBAC.
- Member invite/delete UI.
- Automatic approval of future source modifications.
