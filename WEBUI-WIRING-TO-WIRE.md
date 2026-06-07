# WebUI Wiring — To Wire

Remaining work after the as-built state captured in [WEBUI-WIRING-WIRED.md](WEBUI-WIRING-WIRED.md).

## Priority 1 — Real MyAccess / IdP JWT enforcement

PO-gated by Louis HORVAT (RBAC sign-off pending).

- Implement real MyAccess / IdP JWT enforcement before production exposure.
- Validate parent-KB access from IdP claims before returning spaces or data.
- Keep member management out of the WebUI; consume claims, do not reinvent IAM.
- Do not add steward-managed API-key distribution — BNP production target is OAuth2 / IdP-backed auth or mTLS, not user-issued API keys.
- Tests to add:
  - missing cookie/token → 401;
  - expired token → 401;
  - invalid signature → 401;
  - valid user allowed on parent KB gets configured spaces.

## Priority 2 — Deployment smoke on OVH `twin-real`

- Confirm `twin-real` stack on OVH `37.59.104.111` is actually serving `maquette.sigilum.fr` (vs. the legacy `twin-maquette` JSX stack).
- Smoke checklist post-deploy:
  - `/webui/` substitution worked (`__TWIN_CONFIG_JSON__` absent from served HTML).
  - `apiBaseUrl` resolves to `/twin/api`.
  - `/twin/api/spaces` returns the env-injected catalog.
  - `/twin/api/graph/entities` returns real Memgraph data.
- Document the runbook commands in `docs/operations/install-runbook.md`.

## Priority 3 — Retention / sweep policy (DEFERRED BY POLICY)

Not tech debt — explicit "do nothing silently" because the wrong default could violate BCE/DORA retention or wipe legal-hold evidence. Re-open when PO + compliance arbitrate the four axes in writing:

- **TTL per store**: separate values for `WebuiTag`, `ActivityEvent`, `Notification`, audit logs.
- **Sweep mechanism**: cron-style Cypher purge vs read-time predicate (Memgraph TTL primitives).
- **Scope**: per-space vs global; sandbox vs primary behavior.
- **BCE / DORA / legal-hold**: minimum-retention contracts + how a legal-hold flag suspends sweeping.

## Priority 4 — Performance frontend optimizations (local branch)

Local branch `feat/webui-perf-optimizations` (HEAD `d578ac3`) implements:

- Lazy-load secondary tabs + modal bodies + Suspense boundaries in `App.tsx`.
- `QueryGate` (`enabled` flag) on read hooks so inactive tabs don't poll.
- `limit?: number` on `api.listActivity()` + bounded `/activity` reads (default 200, max 1000) in `webui_router`.
- Scalar/indexed Memgraph fields on `WebuiActivity_{workspace}` (`kind`, `sev`, `actor_user`, `__created_at`).

Bundle: entry JS `473.67 → 278.85 kB raw`, gzip `134.55 → 85.97 kB`. Tests green (392 pytest + 396 vitest + typecheck).

To do: push the branch, open the PR, merge.

## PO-Gated / Do Not Start Without Confirmation

- **BNP MIP classification + ingestion hook.** Tabled 2026-06-07 — out of current scope ("ligne de canon digne du VASA"). Existing `classification.py` + `_classification_hook.py` modules remain in the tree as opt-in, but are not wired into the pipeline by default, and there is no `/twin/api/classification/_self_check` endpoint, no `[classification]` packaging extra, no Memgraph e2e coverage for it. Re-evaluate only if BNP product brings it back as a P0.
- Source-document lifecycle doctrine (the §1.5/§5.5 runbook formalisation prepared on a parallel branch — not landed yet).
- Auto-approve future source modifications.
- Provider configure panels in Settings.
- Role-perspective simulator (RBAC simulation in UI).
- Member invite/delete UI.
- Any durable raw-document storage or full-document preview for sensitive material.
