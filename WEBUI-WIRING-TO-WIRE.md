# WebUI Wiring — To Wire

Remaining work after the as-built state captured in [WEBUI-WIRING-WIRED.md](WEBUI-WIRING-WIRED.md).

## Priority 1 — Real MyAccess / IdP JWT enforcement

**Status 2026-06-10**: code mechanic landed (palier 1 dormant + palier 2 active), JWKS wiring still pending Louis HORVAT (RBAC sign-off pending).

The two-tier posture flips on a single env var (`TWIN_IDP_JWKS_URL`):

- **Palier 1 — dormant**: `require_auth` refuses anonymous at boot (`ensure_auth_backend_configured` raises unless `LIGHTRAG_API_KEY` / `LIGHTRAG_JWT_SECRET` / `TWIN_IDP_JWKS_URL` / `TWIN_ALLOW_OPEN_ACCESS=1` is set). `require_admin_user` returns a placeholder with `idp_validated=False`. `resolve_folder_for_request` reproduces pure header+catalog binding.
- **Palier 2 — active (auto, when `TWIN_IDP_JWKS_URL` is set)**: scope `admin:folders` enforced on folder CRUD. Folder header bound to the user's `twin_folders` claim (fallback default folder when the claim is empty, for the MyAccess rollout window).

What's done:

- ✅ Missing cookie/token → 401 (`require_idp_user` + boot fail-closed)
- ✅ Expired token → 401 (existing `test_idp_jwt.py`)
- ✅ Invalid signature → 401 (existing `test_idp_jwt.py`)
- ✅ Valid user allowed on parent KB gets configured folders (`tests/test_server/test_folder_idp_binding.py`)
- ✅ `changeme` default → loud `SECURITY:` warning (relaxed from unconditional refusal same day — LightRAG-parity product decision after the BNP crash-loop; warning becomes irrelevant once the IdP is wired).
- ✅ No steward-managed API-key distribution.

What's left for BNP:

- Wire `TWIN_IDP_JWKS_URL` to the real MyAccess JWKS endpoint (Louis HORVAT).
- Integration test against a real Keycloak/MyAccess (vs PyJWK mock) — ops/deployment, not code.

## Priority 2 — Deployment smoke

- Smoke checklist post-deploy:
  - `/webui/` substitution worked (`__TWIN_CONFIG_JSON__` absent from served HTML).
  - `apiBaseUrl` resolves to `/twin/api`.
  - `/twin/api/folders` returns the env-injected catalog.
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
