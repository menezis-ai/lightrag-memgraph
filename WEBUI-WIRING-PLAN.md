# WebUI Wiring Plan — Couches 2 & 3

> Companion to `lightrag_webui_twin/` (Vite + React 19 + TypeScript + Bun)
> and `src/twindb_lightrag_memgraph/` (Python storage backends + LightRAG
> registration). Documents what's already done (Couche 2 — Classification)
> and what remains (Couche 3 — LightRAG real-backend wiring), so anyone
> picking up this work has the full contract without reading the entire
> session log.

## TL;DR — state of play, 2026-06-02

| Couche | Scope | Status | Reference |
|---|---|---|---|
| **0** | Decisions + branch hygiene + visual snapshot | ✅ Done | session log |
| **1** | Visual port from `~/Downloads/prototype/` to React/TS | ✅ Done | PR [#158](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/158), [#159](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/159) |
| **2** | BNP classification (TS types + ClassPill + DocDetailPanel gating + Python extractor + pre-insert hook) | ✅ Done | PR [#157](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/157) (Python) + PR [#158](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/158) (TS UI) |
| **3** | LightRAG wiring real (server FastAPI sub-app, JWT, real fetch, X-Twin-Space header, drop MSW in prod) | 🚧 **Partial** — runtime config + frontend space cutover done; backend enforcement remains | this document |

The standalone OVH demo at https://maquette.sigilum.fr/ uses the React
port with **MSW client-side** (everything mocked in the browser, no
backend). Couche 3 replaces MSW with the real LightRAG + Twin overlay.

## Update 2026-06-02 — Runtime config + Twin spaces

Implementation status:

- The React client no longer hardcodes `cib`; the initial Twin space comes
  from `window.__twinConfig.defaultSpaceId` / `window.__twinConfig.spaces`.
- The server injects `defaultSpaceId`, `spaces`, and `maxSpaces` into the
  runtime config from env vars:
  - `TWIN_DEFAULT_SPACE` (fallback: `WORKSPACE`, then `default`)
  - `TWIN_DEFAULT_SPACE_LABEL`
  - `TWIN_SPACES_JSON`
  - `TWIN_MAX_SPACES` (clamped to 1..5: one default + four admin-created)
- The HTTP client sends `X-Twin-Space` on API requests. It also sends the
  legacy `X-Twin-Workspace` header during the transition so existing backend
  code keeps working while the backend contract moves to "space".
- Visible UI copy now says "Space" / "Spaces". The empty state is:
  `No space available for this KB. Please contact Twincore Team`.
- Backend phase-1 enforcement now reads `X-Twin-Space` first,
  accepts `X-Twin-Workspace` as a temporary fallback, validates the id
  against the configured catalog, and binds `request.state.space`.
- The Twin WebUI overlay has one store per space. In-memory dev stores are
  isolated; Memgraph stores are initialized per configured space.
- Native document shims keep the LightRAG workspace/KB unchanged and filter
  documents/chunks/delete by `DocStatus.metadata.space`; legacy docs with no
  space metadata remain visible only from the default space.

Env contract for SRE/devOps:

```bash
TWIN_DEFAULT_SPACE=default
TWIN_DEFAULT_SPACE_LABEL="Default space"
TWIN_MAX_SPACES=5
TWIN_SPACES_JSON='[
  {
    "id": "default",
    "label": "Default space",
    "kind": "primary",
    "description": "SRE-provisioned default space for this KB.",
    "sources": 0
  },
  {
    "id": "sandbox",
    "label": "Sandbox",
    "kind": "sandbox",
    "description": "Operator-managed test space.",
    "sources": 0
  }
]'
```

Rules:

- `TWIN_DEFAULT_SPACE` is mandatory conceptually: every KB has one default
  space provisioned by SRE/devOps. Code fallback is `WORKSPACE`, then
  `default`, only to keep dev/test bootable.
- `TWIN_MAX_SPACES` is clamped to `1..5`. Product rule = one default space
  plus up to four admin-created spaces.
- Space ids are validated with the same safe identifier rule as Memgraph
  workspace labels: non-empty `[a-zA-Z0-9_]`.
- `TWIN_SPACES_JSON` is the deployment-time source of truth until the admin
  CRUD exists.

Code touched in this update:

- `src/twindb_lightrag_memgraph/__init__.py` — runtime config now emits
  `defaultSpaceId`, `spaces`, and `maxSpaces`.
- `src/twindb_lightrag_memgraph/_spaces.py` — shared env parsing and runtime
  space catalog without FastAPI/server dependency.
- `src/twindb_lightrag_memgraph/server/space.py` — request header resolution
  and request context binding.
- `src/twindb_lightrag_memgraph/server/webui_router.py` — per-space WebUI
  stores plus `/spaces` compatibility endpoint.
- `src/twindb_lightrag_memgraph/server/native_shims.py` — document/chunk/delete
  shims filter by active Twin space without changing the LightRAG workspace.
- `lightrag_webui_twin/src/api/client.ts` — runtime URL bases plus
  `X-Twin-Space` header, with temporary `X-Twin-Workspace` compatibility.
- `lightrag_webui_twin/src/App.tsx` — active space initialized from runtime
  config and switched through `setActiveSpace`.
- `lightrag_webui_twin/src/types/auth.ts` and `src/config/devConfig.ts` —
  typed space config and dev defaults.
- UI copy under `components/` — visible vocabulary changed from workspace to
  space where it describes Twin sub-scopes.

Validated:

- `cd lightrag_webui_twin && bun run typecheck`
- `cd lightrag_webui_twin && bun run test:run` — 325 tests passed
- `cd lightrag_webui_twin && bun run build`
- `.venv/bin/pytest tests/test_register.py -q` — 9 tests passed
- `.venv/bin/pytest -q` — 575 tests passed, 147 skipped
- `git diff --check`

Still to do:

- Frontend Admin UI: add a Settings "Manage spaces" section for runtime
  create/update/delete of non-env-seeded spaces.
- Real JWT/MyAccess enforcement: validate the parent KB access through IdP
  claims before exposing the configured space list.
- Clean up remaining internal `workspace` names once the backend contract has
  fully migrated; keep compatibility until then.

## Audit 2026-06-04 — priorités de câblage réel

This section deduplicates the current Couche 3 backlog after a code audit.
The frontend client and MSW know more routes than the Python backend currently
serves; the priorities below are the shortest path from "demo green" to "real
backend green".

### Update 2026-06-04 — M12 Graph real Memgraph

M12 is functionally closed on branch `feat/webui-graph-real-memgraph`:

- `GET /twin/api/graph/entities` and `/graph/relations` read from Memgraph.
- `PATCH /twin/api/graph/entities/{id}` and `/graph/relations/{id}` persist.
- Entity/relation creation and deletion are wired end-to-end from GraphTab to
  Memgraph.
- Frontend GraphTab supports add entity, delete entity/relation, and add
  relation.
- Local validation reported for the M12 stack: backend `634/634` pytest,
  frontend `364/364` vitest.

Commits in the local M12 stack:

| Commit | Scope |
|---|---|
| `c805609` | batch 1 backend — real Memgraph GET + layout |
| `8b616d0` | batch 2 backend — PATCH persistence |
| `58f5e14` | batch 3 backend — POST/DELETE lifecycle |
| `4a0b53e` | batch 3 frontend — Add entity + Delete entity/relation |
| `7da28d3` | batch 3 frontend — Add relation form |

Remaining P0 focus after M12: document metadata, bulk delete, Twin overlay
health, route parity, production fixture fallbacks, and real auth.

### Update 2026-06-04 — Admin Space CRUD backend

Commit `173b09f` closes the backend half of the "spaces beyond
`TWIN_SPACES_JSON`" gap:

- `server/space_store.py` provides FastAPI-free runtime CRUD with optional
  atomic JSON persistence via `TWIN_SPACES_RUNTIME_FILE`.
- `load_space_catalog()` merges env seed + runtime additions; env-seeded
  spaces win on id collision and remain immutable.
- `POST /twin/api/spaces`, `PATCH /twin/api/spaces/{id}`, and
  `DELETE /twin/api/spaces/{id}` are wired in `webui_router.py`.
- Delete refuses to orphan state when the target space still has docs or tags.
- Mutations emit `settings` activity events with structured
  `operation: create | update | delete` metadata.
- Validation reported for the batch: `661/661` pytest, `97` skipped.

Still open for spaces: frontend Settings UI and JWT/MyAccess admin-only
gating.

### P0 — Contract drift blockers

- **Done — Add an automated route parity test** that compares:
  - `lightrag_webui_twin/src/api/resources.ts` expected paths,
  - `lightrag_webui_twin/src/mocks/handlers.ts` MSW paths,
  - actual FastAPI routes from `webui_router` + `native_shims`.
  Current implementation: `tests/test_server/test_route_parity.py`, with a
  short `KNOWN_BACKEND_GAPS` allow-list for the three documented Couche 3
  holes.
- **Done — MSW `/query` drift fixed.** The parity test caught that
  `resources.ts` called `POST /query` without an MSW handler; the handler now
  returns a minimal `{response}` payload.
- Implement or deliberately remove these frontend/MSW-only routes:
  - `GET /twin/api/documents/{id}/metadata`
  - `POST /twin/api/documents/bulk-delete`
  - `GET /twin/api/health`
- Add backend contract tests for those routes before expanding Playwright
  coverage. Current Playwright/MSW success is not enough because it can mask
  real-backend `404`/`405` failures.

### P1 — Remove silent fixture fallbacks in production

- Replace production fallbacks from local fixtures with explicit loading/error
  states for documents, tags, activity, thesaurus, graph, and notifications.
  In dev/MSW, fixtures remain acceptable; in prod, stale local data must not
  look like backend truth.
- Keep `VITE_FORCE_MSW=true` restricted to standalone demo builds. Production
  builds without that flag must hit real `/documents` and `/twin/api/*`.
- Add a runtime assertion that fails loudly when `window.__twinConfig` was not
  substituted outside dev/MSW demo mode.

### P2 — Persist all operator-visible mutations

- `bulk-delete`: implement the real backend path called by the UI and emit one
  activity event per deleted doc. The existing MSW handler already proves the
  frontend journey, but the Python router has no matching route yet.
- `document metadata`: serve tags, space, review, and classification from
  DocStatus + tag graph relations, not from WebUI seed data.
- `graph lifecycle`: done in M12 for read/write/create/delete against Memgraph.
  Keep it covered by route parity + regression tests.
- `tag delete migration`: current backend records migration intent, but does
  not retag affected documents. Either implement the migration/untag cascade or
  change the UI wording to say only the tag catalog changed.

### P3 — Production auth and integration confidence

- Replace local username/password JWT with real MyAccess/IdP validation:
  JWKS validation, parent KB access check, allowed spaces derived from claims,
  and fail-closed behavior.
- Run CI with a real Memgraph service for Couche 3 backend contracts. The
  current pytest setup skips integration tests when `MEMGRAPH_URI` is absent,
  so local green does not prove real persistence.
- Add one end-to-end smoke against MSW disabled (`VITE_USE_MSW=false`) and a
  running backend, covering login/auth headers, documents list, retag, bulk
  delete, and retrieval query.

## WebUI hardening backlog — 2026-06-03 audit follow-up

The 2026-06-03 async/a11y audit fixed the most urgent regressions:
bulk delete now uses `/documents/bulk-delete`, toast live-region updates
batch simultaneous announcements, tag autocomplete Escape is isolated from
modal close handlers, and the graph canvas blocks scroll chaining. The
follow-up hardening pass is now implemented as well:

- **Done — Bound bulk upload concurrency.** Upload batches now run through a
  TanStack mutation that caps concurrent `/documents/upload` calls at 4 and
  invalidates `documents` + `pipeline_status` once after the batch. Regression
  coverage asserts that 20 files do not create more than 4 simultaneous
  fetches.
- **Done — Make `useModalA11y` autofocus non-stealing.** The deferred autofocus
  now clears its timeout on cleanup and skips autofocus when focus is already
  inside the modal. The direct `AddSourceModal` / `RetagModal` sleep
  workarounds were removed.
- **Done — Harden Knowledge Graph wheel handling.** The graph canvas now uses
  a native `wheel` listener with `{ passive: false }`, cleaned up in
  `useEffect`, while keeping `touch-action: none` /
  `overscroll-behavior: none` for scroll-chain isolation.

## Plan e2e renforcé — recette v2 + Couche 3

This section translates the 2026-05-29 WebUI recipe findings into an
actionable e2e test plan. `WEBUI-WIRING-PLAN.md` remains the source of truth:
the old recipe vocabulary says "workspace", but the active contract now says
"space" for Twin sub-scopes.

### Update 2026-06-03 — recipe hardening status

The current React WebUI test plan now covers the highest-risk recipe failures
that are in scope for the standalone prototype and Couche 3 frontend contract.

Closed by focused e2e/RTL coverage:

- **RC-1 persisted mutations:** Documents retag/delete/review decisions, Tags
  request/approve/reject/edit/synonyms/delete, and Activity refresh/immutable
  ledger behavior.
- **RC-2 counters and drill-downs:** Documents counters/status/tag filters,
  Knowledge Graph exact source drill-down, entity type counters, and pinned
  KG entities restored after reload.
- **RC-3 validation:** Add source file type/size/counting, tag request required
  name, tag reject required reason, taxonomy JSON validation, and Settings
  absence assertions for MyAccess-owned member/default-tag surfaces.
- **RC-4 no-op actions:** Add source browse/file chooser, Retrieval citation
  navigation, API bearer revoke confirmation, and sign-out client cleanup.
- **RC-5/RC-7 UI regressions and wording:** bulk delete remains present,
  document Edit & Approve opens and persists, topbar brand routes back to
  Documents, Tags edit toast uses truthful generic wording, and the Tags
  pending banner uses tag-specific copy.
- **Async/a11y hardening:** bounded upload concurrency, non-stealing modal
  autofocus, toast live-region batching, tag autocomplete Escape isolation,
  and native non-passive graph wheel handling.

Still open or PO-gated in the current React port:

- `TWIN-DOC-10`: no current React component exposes the lifecycle
  "Auto-approve future modifications" checkbox; wiring it would create new
  product surface.
- `TWIN-SET-01`: provider Configure panels remain PO-gated/out of current
  Settings scope.
- `TWIN-SET-02`: sign-out local cleanup is covered; real IdP/JWT revocation
  remains Couche 3/backend contract work.
- `TWIN-TRX-01`: role perspective selector remains PO-gated because the
  current direction is real MyAccess/JWT-driven authorization, not UI role
  simulation.
- `TWIN-TRX-02`: the implemented space switch uses dynamic refetch/state
  reset; a full page reload should only be restored if PO explicitly requires
  that behavior.
- `TWIN-TRX-04` and `TWIN-TAG-11`: responsive/dark-theme visual assertions are
  still good candidates for a later visual-regression pass.

Latest frontend validation baseline:

- `npm run typecheck` — OK.
- `npm run test:run` — OK, 345 Vitest tests.
- `npm run test:e2e` — OK, 59 Playwright tests.
- `git diff --check` — OK.

### Preconditions

- Keep the existing Playwright smoke green before adding new coverage. Current
  selectors must use the active UI contract (`Switch space`, not the old
  `Switch workspace` wording).
- Run Playwright with one worker while the prototype uses MSW mutable state.
  Multiple spec files can otherwise race through shared `/__e2e/reset`
  handlers. Revisit this once e2e targets the real Couche 3 backend or each
  worker gets an isolated mock namespace.
- Split the current monolithic `lightrag_webui_twin/e2e/app.spec.ts` into
  domain files before adding many new cases:
  - `documents.spec.ts`
  - `tags.spec.ts`
  - `retrieval.spec.ts`
  - `graph.spec.ts`
  - `activity.spec.ts`
  - `settings-auth.spec.ts`
  - `spaces-runtime.spec.ts`
- Keep Playwright focused on operator journeys. Use Python `pytest` for real
  backend contracts and Memgraph isolation.

### Test layers

| Layer | Scope | Tool |
|---|---|---|
| WebUI journeys | Visible operator behavior, forms, toasts, refresh, reload persistence | Playwright + MSW |
| Front/API contract | `window.__twinConfig`, URL bases, headers, cache invalidation | Vitest + Playwright request interception |
| Route parity contract | Frontend `resources.ts` paths, MSW handlers, and real FastAPI router must match | `tests/test_server/test_route_parity.py` |
| Real backend contract | Space isolation, Memgraph stores, classification metadata, auth | `pytest tests/test_server/*` |

### Priority 0 — stabilize existing e2e

- Fix stale selectors from the workspace → space migration.
- Keep the two `@smoke` journeys green locally before expanding coverage.
- Add helper assertions in `e2e/helpers.ts`:
  - `expectToast(page, text)`
  - `expectDocumentTags(page, docId, tags)`
  - `reloadAndExpectVisible(page, locatorFactory)`
  - `expectMswMutationCount(page, resource, id, count)`

### Priority 1 — RC-1 persisted mutations

The core recipe failure is "toast success without commit". Every S1/S2
mutation e2e must prove all three outcomes:

1. The UI changes immediately.
2. The data survives a refetch or `page.reload()`.
3. The expected Activity/Notification side effect appears when the contract
   says it should.

Playwright cases:

- Documents:
  - Add source valid file → document appears with initial status → reload
    still shows it.
  - Retag one document → row tags update → reload keeps tags.
  - Bulk retag N documents → every selected row updates → failure rolls back.
  - Bulk delete → confirmation required → rows removed → counters refresh.
  - Pending approve/reject/edit-approve → item leaves pending queue → reload
    preserves review state.
- Tags:
  - Request new tag → pending queue updates.
  - Approve tag → pending item disappears, active tag appears, Activity and
    notification are created.
  - Reject tag → reason required, pending item disappears.
  - Edit tag → edited definition/synonyms persist after reload.
  - Deprecate/delete with migration → tag status/list and affected documents
    reflect the selected strategy.
- Activity:
  - Refresh loads newly available events and clears the "new events" counter.
  - Clear events never reports removed items when none were purgeable.

Backend `pytest` cases:

- Done: route parity now asserts every frontend path in `resources.ts` either exists in the
  real FastAPI router or is explicitly marked dev/MSW-only. MSW handlers must
  not be treated as the source of truth.
- `GET /twin/api/documents/{id}/metadata`
- `POST /twin/api/documents/{id}/approve`
- `POST /twin/api/documents/{id}/reject`
- `POST /twin/api/documents/bulk-delete`
- `GET /twin/api/health`
- Graph endpoints are M12-complete; keep GET/PATCH/POST/DELETE lifecycle in
  regression coverage and route parity.
- `POST /twin/api/tags`, approve/reject/edit/synonyms/delete
- Assert persisted store state per space and emitted activity events.

Progress as of 2026-06-02:

- Playwright smoke selectors were aligned with the active "space" contract.
- MSW e2e mutable stores now persist across page reloads for documents, tags,
  tag categories, notifications, and activity; `POST /__e2e/reset` clears that
  session state between tests.
- `TWIN-DOC-04` is covered by the upload-with-initial-tags journey: uploaded
  document + auto-applied tag remain visible after `page.reload()`.
- `lightrag_webui_twin/e2e/documents.spec.ts` adds focused RC-1 coverage for:
  - `TWIN-DOC-01`: bulk retag is verified through a tag-filtered document view
    and remains valid after reload.
  - `TWIN-DOC-02`: bulk delete action is present, requires the existing
    two-click confirmation, removes the row, and stays removed after reload.
  - `TWIN-DOC-03`: Edit & Approve opens the form, commits edited summary/tags,
    removes the item from pending review, and survives reload.
  - Document reject baseline: rejected pending source leaves the review queue
    and remains absent after reload.
- `TWIN-TAG-05` is covered by the tag approval journey: approved tag remains
  visible after reload, leaves pending, and keeps notification/activity side
  effects.
- `lightrag_webui_twin/e2e/tags.spec.ts` adds focused RC-1 coverage for:
  - `TWIN-TAG-01` direct tag edit updates the tag data and survives reload,
  - `TWIN-TAG-04` requested tag remains pending after reload,
  - `TWIN-TAG-06` Edit & Approve applies steward definition/category edits
    before approving the request and keeps those edits after reload; canonical
    rename is not covered by the current API contract,
  - `TWIN-TAG-08` rejected request leaves pending after reload,
  - `TWIN-TAG-09` synonym updates remain visible after reload,
  - `TWIN-TAG-10` delete/migration removes the tag after reload.
- `lightrag_webui_twin/e2e/activity.spec.ts` adds focused RC-1 coverage for:
  - `TWIN-ACT-01` explicit Refresh refetches newly available events from the
    store and the fetched event survives reload.
  - `TWIN-ACT-02` is now covered as an immutable-ledger decision: there is no
    manual "Clear activity events" / purge affordance in the UI, so the old
    contradictory purge toast cannot be triggered.

### Priority 2 — Twin spaces and runtime config

This is the 2026-06-02 Couche 3 contract and must be covered independently
from generic navigation.

Playwright cases:

- Initial active space comes from `window.__twinConfig.defaultSpaceId`.
- The space menu lists `window.__twinConfig.spaces` and uses visible "Space"
  copy.
- Switching space:
  - clears URL filters,
  - closes open modals/detail panels,
  - resets notification local overrides,
  - refetches documents, tags, activity, and notifications,
  - sends subsequent requests with the new active space.
- Empty configured space list shows exactly:
  `No space available for this KB. Please contact Twincore Team`.

Front/API contract cases:

- Every `apiFetch` request carries `X-Twin-Space`.
- During migration it also carries `X-Twin-Workspace`.
- Per-call `space` override wins over the global active space.
- `VITE_FORCE_MSW=true` keeps the standalone OVH demo on MSW; production
  without that flag hits the real backend.

Backend `pytest` cases:

- `X-Twin-Space` is read before `X-Twin-Workspace`.
- Legacy `X-Twin-Workspace` remains accepted while the compatibility window is
  open.
- Unknown space returns the configured empty-state error.
- Native document shims filter by `DocStatus.metadata.space`.
- Legacy docs without space metadata are visible only from the default space.
- Chunks and delete routes reject documents from another space.
- WebUI tag/activity/notification stores are isolated per configured space.

Progress as of 2026-06-03:

- `lightrag_webui_twin/e2e/spaces-runtime.spec.ts` covers the runtime-config
  UI contract:
  - initial active space comes from `window.__twinE2eRuntimeConfig.defaultSpaceId`
    in e2e, mirroring the server-substituted `window.__twinConfig` contract;
  - the topbar space menu renders the configured `spaces` list and visible
    "Spaces" copy;
  - switching to `sandbox` causes subsequent Twin overlay requests to carry
    both `X-Twin-Space: sandbox` and transitional `X-Twin-Workspace: sandbox`;
  - an explicitly empty configured space list shows
    `No space available for this KB. Please contact Twincore Team`.
- `api/client.test.ts` now covers per-call `space` override, legacy
  `workspace` override, and explicit `space: null` header suppression.
- `useAuth.test.tsx` covers the dev/e2e runtime override that lets Playwright
  test server-injected config despite the Vite HTML placeholder.

### Priority 3 — validations and no-op actions

Playwright cases:

- Add source:
  - unsupported file type is visibly rejected,
  - file over limit is visibly rejected,
  - mixed valid/invalid upload only counts valid files in the submit button,
  - partial upload failure reports exact `ok/ko` counts.
- Tags:
  - request tag cannot submit without required fields,
  - reject tag cannot submit without reason,
  - invalid taxonomy JSON shows a precise validation banner.
- Retrieval:
  - clicking a citation opens the referenced document/chunk source.
- Settings/Auth:
  - sign out calls `/twin/api/auth/logout`, clears local caches, and leaves no
    retrieval thread state behind.
  - local API bearer revoke in the OpenAPI explorer requires a second
    confirmation click.
  - Members/invites/deletes and editable default ingestion tags stay out of
    Settings scope; MyAccess and the controlled tag picker own those surfaces.

Progress as of 2026-06-03:

- `lightrag_webui_twin/e2e/upload.spec.ts` covers Add source validation:
  - `TWIN-DOC-05` browse opens the native file chooser and accepts selected
    files.
  - `TWIN-DOC-06` unsupported extensions and files over 50 MB are rendered as
    errors before submit.
  - `TWIN-DOC-07` mixed valid/invalid uploads count only valid files in
    `Add n sources`, submit only valid raw files, and leave invalid files out
    of the Documents table.
- `AddSourceModal.test.tsx` covers the same validation at component level for
  fast feedback.
- Tags validation is already enforced in both RTL and Playwright:
  - `TWIN-TAG-03`: `Request new tag` keeps `Submit request` disabled until
    the required proposed name is filled.
  - `TWIN-TAG-07`: `Reject request` keeps the destructive submit disabled until
    a non-empty reason is provided.
- Tags wording cleanup is covered in component/e2e tests:
  - `TWIN-TAG-02`: tag edit now uses the truthful generic toast suffix
    `updated` instead of hardcoding `definition updated`.
  - `TWIN-TAG-12`: the pending section uses tag-specific wording:
    `Tag requests` / `n tag requests awaiting review`.
- `lightrag_webui_twin/e2e/retrieval.spec.ts` covers `TWIN-RET-01`: clicking
  a citation navigates to Documents with the cited source as the search filter.
- Settings/Auth guardrails are covered without reintroducing member/admin
  surfaces into Settings:
  - `TWIN-SET-03`: `ApiTab`'s local bearer revoke is a two-step action
    (`Revoke token` → `Confirm revoke token`), covered by
    `ApiTab.test.tsx` and `lightrag_webui_twin/e2e/settings.spec.ts`.
  - `TWIN-SET-04`: default ingestion tags are not editable from Settings;
    tag entry remains constrained to thesaurus-backed pickers in Add source /
    Retag flows. Settings absence is covered by `SettingsTab.test.tsx` and
    `settings.spec.ts`.
  - `TWIN-SET-05` / `TWIN-SET-06`: member invite/delete remain MyAccess-owned
    and absent from the Settings rail/body, covered by `SettingsTab.test.tsx`
    and `settings.spec.ts`.
  - Validation run: `bun run typecheck`, `bun run test:run`, and
    `npx playwright test e2e/settings.spec.ts`.

### Priority 4 — classification Couche 2 in real wiring

Backend `pytest` cases:

- `GET /twin/api/documents/{id}/metadata` returns structured
  `classification` when present in DocStatus metadata.
- A synthetic C3/C4 document above `TWIN_MIP_MAX_CLASSIFICATION` is rejected,
  leaves DocStatus `FAILED`, and emits `classification-rejected`.
- Missing/unknown label map fails closed as documented in the activation
  matrix.

Playwright cases:

- Structured C1/C2/C3/C4 metadata renders the expected `ClassPill`.
- Legacy string classification stays silent.
- Above-internal classification gates raw/chunk access with the notice.

### Priority 5 — counters, filters, and drill-downs

Playwright cases:

- Documents counters reflect the currently filtered collection, not the global
  fixture count.
- Status and tag filters update URL state and visible rows coherently.
- Knowledge Graph entity drill-down to Documents uses a complete tag/entity
  filter, not a lossy text search.
- Entity type counters update after graph filters.

Progress as of 2026-06-03:

- `DocumentsTab` counters now derive from the search + tag filtered table
  collection instead of the global document list. The active status filter is
  then applied to rows, while all status pills keep showing the distribution
  within the same search/tag subset so operators can still switch statuses.
- `lightrag_webui_twin/src/components/DocumentsTab.test.tsx` covers:
  - default status counts,
  - search-scoped counts (`oracle`),
  - tag-scoped counts (`rman`),
  - URL-backed status and tag filters.
- `lightrag_webui_twin/e2e/documents.spec.ts` covers:
  - `TWIN-DOC-08`: visible Documents counters follow the active search filter
    instead of the global fixture count;
  - `TWIN-DOC-09`: explicit status and tag filters update URL state, visible
    rows, counters, and survive reload for tag filters.
- Nuance locked by e2e: the "Uploaded" table counters intentionally count only
  the non-pending table rows. Documents currently shown in the pending review
  panel, such as modified source `d2`, are not double-counted in the table
  counters until they leave the review queue.
- `GraphTab` entity type counters now derive from the active graph search,
  tag, and source filters before applying type toggles, so the rail reflects
  the currently inspectable graph subset.
- Graph entity drill-down now navigates to Documents with an exact
  comma-separated `source` URL filter derived from the entity source map,
  falling back to text search only when no source map exists. This avoids the
  previous lossy `q=entity name` behavior.
- `lightrag_webui_twin/src/components/GraphTab.test.tsx` covers filtered type
  counters and exact-source navigation.
- `lightrag_webui_twin/e2e/graph.spec.ts` covers:
  - `TWIN-KG-01`: entity drill-down writes the full exact source list and
    hides unrelated document rows;
  - `TWIN-KG-02`: Pin/Pinned state is persisted in localStorage and restored
    after reload;
  - `TWIN-KG-03`: entity type counts update after graph search filters.
- `TWIN-DOC-10` remains a PO/UI gate in this React port: no current component
  exposes an "Auto-approve future modifications" lifecycle checkbox to wire or
  test without inventing new product surface.
- `TWIN-TRX-03` is covered at component level: the topbar brand is now an
  accessible button that routes back to Documents.

### Recipe ticket coverage matrix

The table below maps every 2026-05-29 recipe ticket to the intended e2e
coverage. Items marked "PO gate" must first be confirmed as still in scope for
the current React port and Couche 3 contract.

| Ticket | Coverage target | Layer |
|---|---|---|
| TWIN-DOC-04 | Add source commits a valid file, updates list/counters, survives reload | Playwright + backend upload/track-status |
| TWIN-DOC-01 | Single and bulk retag persist tags, refresh documents, emit activity | Playwright + backend `_bulk-retag` |
| TWIN-DOC-02 | Bulk delete action is present after selection, requires confirmation, persists removal | Playwright + backend bulk-delete, PO gate if removal was intentional |
| TWIN-DOC-03 | Edit & Approve opens the edit form, commits edits, leaves pending queue | Playwright + backend approve-with-edits |
| TWIN-DOC-05 | Browse button opens the native file chooser and accepts selected files | Playwright `filechooser` |
| TWIN-DOC-06 | Unsupported type / oversized file shows an error and is excluded | Playwright |
| TWIN-DOC-07 | Submit button count includes valid files only | Playwright |
| TWIN-DOC-08 | Document counters reflect active filters | Playwright |
| TWIN-DOC-09 | Status and tag filters are explicit, URL-backed, and update rows/counters | Playwright |
| TWIN-DOC-10 | Lifecycle auto-approve checkbox toggles and persists | Playwright, PO gate if lifecycle is out of current UI scope |
| TWIN-TAG-01 | Tag edit persists changed fields and history | Playwright + backend tag PATCH |
| TWIN-TAG-02 | Tag edit toast is field-aware or uses a truthful generic message | Playwright/RTL |
| TWIN-TAG-03 | Request new tag blocks empty required fields | Playwright |
| TWIN-TAG-04 | Requested tag appears in pending queue and survives reload | Playwright + backend tag POST |
| TWIN-TAG-05 | Approve removes pending item, creates active tag, emits activity/notification | Playwright + backend approve |
| TWIN-TAG-06 | Approve-with-edits preserves steward edits | Playwright + backend approve/edit path, PO gate if flow is not exposed |
| TWIN-TAG-07 | Reject requires a non-empty reason | Playwright |
| TWIN-TAG-08 | Reject removes pending item and persists rejected state | Playwright + backend reject |
| TWIN-TAG-09 | Manage synonyms persists aliases and refreshes tag detail/list | Playwright + backend synonyms |
| TWIN-TAG-10 | Delete migrate/untag strategies persist tag and affected document changes | Playwright + backend delete |
| TWIN-TAG-11 | Delete strategy labels remain readable in dark theme | Playwright visual/CSS assertion |
| TWIN-TAG-12 | Pending tag banner uses tag-specific wording | Playwright/RTL copy assertion |
| TWIN-RET-01 | Citation click opens referenced source/chunk | Playwright |
| TWIN-KG-01 | Graph entity drill-down uses complete source filter | Playwright |
| TWIN-KG-02 | Pin/Pinned state persists for session or reload per PO decision | Playwright, PO gate for persistence level |
| TWIN-KG-03 | Entity type counters reflect active graph filters | Playwright |
| TWIN-ACT-01 | Refresh fetches new events and resets new-event indicator | Playwright + backend activity list |
| TWIN-ACT-02 | Clear activity message and removed count match actual purge result | Playwright + backend activity clear if implemented |
| TWIN-SET-01 | Provider Configure buttons open an editable panel | Playwright, PO gate if Providers section is out of current UI scope |
| TWIN-SET-02 | Sign out calls logout, clears client state, reaches non-auth/redirect path | Playwright + backend auth/logout |
| TWIN-SET-03 | API explorer bearer revoke requires confirmation | Playwright + RTL |
| TWIN-SET-04 | Default ingestion tags are not editable in Settings; tag entry remains constrained to thesaurus pickers | Settings absence assertion + Add source/Retag coverage |
| TWIN-SET-05 | Member invite remains out of Settings scope (MyAccess-owned) | Settings absence assertion |
| TWIN-SET-06 | Delete member remains out of Settings scope (MyAccess-owned) | Settings absence assertion |
| TWIN-TRX-01 | Role perspective selector exists and changes available actions | Playwright, PO gate because current contract may rely on real JWT/MyAccess instead |
| TWIN-TRX-02 | Space switch visibly refetches/re-evaluates state; full reload only if PO requires it | Playwright + front/API contract |
| TWIN-TRX-03 | Logo returns to Documents/home | Playwright/RTL, PO gate if no home affordance is desired |
| TWIN-TRX-04 | Header and pending cards remain usable on mobile viewport | Playwright responsive screenshot/assertions |

Baseline "do not regress" coverage should stay split across the existing
smoke journeys plus focused domain tests. Out-of-scope recipe items remain
out-of-scope unless the PO explicitly brings them into the Couche 3 contract.

### CI recommendation

- Keep existing `webui-tests` as typecheck + Vitest + build.
- Add a lightweight Playwright smoke job first: `bun run test:e2e -- --grep
  @smoke`.
- Add full Playwright later, after splitting specs and stabilizing selectors.
- Keep backend space/classification tests in the existing Python integration
  job with Memgraph service.

---

## Couche 2 — Classification BNP (as-built)

The compliance layer that reads the Microsoft Information Protection
(MIP) sensitivity label on Office documents at ingestion time, persists
it on the DocStatus, gates retrieval-time UI, and refuses anything above
the configured ceiling.

### Files delivered

| File | Role |
|---|---|
| `src/twindb_lightrag_memgraph/classification.py` | Extractor module — OOXML (stdlib), legacy OLE (`olefile`), PDF (`pikepdf`). Returns `ClassificationResult` matching the TS shape. |
| `src/twindb_lightrag_memgraph/_classification_hook.py` | Pre-insert hook + `install_classification_hook(label_map_path, ceiling, audit_emit)` factory + `ClassificationRejection` exception (fail-closed on UNKNOWN classes). |
| `scripts/extract_msip.py` | CLI wrapper — `python scripts/extract_msip.py FILE [--label-map labels.json] [--json] [--exit-code-on-above C2]`. |
| `tests/test_classification.py` | 34 offline tests (synthetic OOXML built in-memory, no real fixture files). |
| `tests/test_classification_hook.py` | 8 offline tests covering gating + audit emission. |
| `lightrag_webui_twin/src/types/classification.ts` | TS mirror of `ClassificationResult.as_dict()`. `ClassId`, `ClassificationValue`, helpers (`isStructured`, `getClassId`, `getClassName`, `isAbove`, `isAboveInternal`). |
| `lightrag_webui_twin/src/components/ClassPill.tsx` | UI badge — 5 variants (`class-c1`/`c2`/`c3`/`c4`/`unknown`), silent on legacy string shape. |
| `lightrag_webui_twin/src/types/classification.test.ts` + `ClassPill.test.tsx` | 29 frontend tests. |
| `README.md` (top-level) | New "Classification (Microsoft Information Protection)" section + env vars `TWIN_MIP_LABEL_MAP` / `TWIN_MIP_MAX_CLASSIFICATION`. |

### Wiring contract

The Twin overlay endpoint `GET /twin/api/documents/{id}/metadata` MUST
return the structured classification when present:

```json
{
  "tags": ["rman", "oracle"],
  "space": "default",
  "review": { ... },
  "classification": {
    "class_id": "C2",
    "class_name": "C2 Confidentiel",
    "label_guid": "22222222-2222-2222-2222-222222222222",
    "raw_name": "C2 Confidentiel",
    "set_date": "2026-03-12T14:22:01Z",
    "method": "Standard",
    "source_format": "ooxml",
    "reason": null,
    "meta": { "Enabled": "true", "SiteId": "{...}" }
  }
}
```

The WebUI reads `doc.metadata.classification` directly. The
`ClassPill` component renders only when the value is structured (the
legacy string shape `"internal"` / `"restricted"` is invisible — it
predates the hook). The `DocDetailPanel` chunks tab + "View raw"
notice gate on `isAboveInternal(cls)` = "above C2 on the BNP ladder".

### Activation matrix

| State | Behavior |
|---|---|
| No `TWIN_MIP_LABEL_MAP` env var | Empty map → every detected label → `class_id: "UNKNOWN"` → fail-closed reject by hook |
| `TWIN_MIP_LABEL_MAP=/etc/twin/labels.json`, GUID in map | `class_id` resolves to `"C1".."C4"`, allow / reject per `is_above(class_id, ceiling)` |
| `TWIN_MIP_MAX_CLASSIFICATION=C3` | Hook allows C1/C2/C3, rejects C4 + UNKNOWN |
| Hook not installed (default after `register()`) | No classification on `metadata.classification` — UI shows no pill |

### Activation steps (for a real BNP deploy)

1. **Acquire the tenant label map** from Louis HORVAT (Compliance Center
   → Sensitivity Labels → Export, JSON or CSV). Convert to:
   ```json
   {
     "<bnp-c1-guid>": {"id": "C1", "name": "C1 Public"},
     "<bnp-c2-guid>": {"id": "C2", "name": "C2 Confidentiel"},
     "<bnp-c3-guid>": {"id": "C3", "name": "C3 Strictement Confidentiel"},
     "<bnp-c4-guid>": {"id": "C4", "name": "C4 Secret"}
   }
   ```
   Save as `/etc/twin/labels.json` (or wherever your secret store mounts
   it). The file is **not** secret — but the GUIDs identify the tenant,
   so keep it out of public git.

2. **Set env vars** on the LightRAG host:
   ```bash
   TWIN_MIP_LABEL_MAP=/etc/twin/labels.json
   TWIN_MIP_MAX_CLASSIFICATION=C2   # adjust per KB / space policy
   ```

3. **Install the hook** (Couche 3 wiring — see below) in the FastAPI
   sub-app, calling `install_classification_hook()` once at startup
   with the audit-emit callback bound to the Twin activity store.

4. **Smoke test** with `python scripts/extract_msip.py path/to/real.docx
   --label-map /etc/twin/labels.json`. Should resolve `class_id` to
   one of C1/C2/C3/C4. If `UNKNOWN`, the GUID mapping is missing.

### What's deliberately NOT in Couche 2

- **No `register()` integration** — the hook is opt-in via explicit
  `install_classification_hook()` + manual call before `LightRAG.insert()`.
  Couche 3 wires it as part of the FastAPI sub-app boot.
- **No BNP tenant label map shipped** — needs to come from Louis HORVAT.
- **No real-document fixtures** in the test suite — tests synthesize
  minimal OOXML packages in-memory (zip + xml).

---

## Couche 3 — LightRAG wiring real (TODO)

Replaces MSW with the real LightRAG + Twin overlay. After Couche 3, the
React port runs as a sub-app mounted by `register()` on the LightRAG
FastAPI server, talks to the real `/documents` + `/twin/api/*` endpoints,
uses real JWT auth from Keycloak / BNP IdP, and ingests with the
classification hook active.

### Architecture target

```
┌─────────────────────────────────────────────────────────────┐
│  LightRAG FastAPI (the wheel: twindb-lightrag-memgraph)     │
│  ├─ Native: /documents, /documents/{id}/chunks, /health,    │
│  │          /query, /openapi, /pipeline_status              │
│  ├─ Mounted via register(replace_ui=True):                  │
│  │   /webui/  → serves React port dist/index.html           │
│  │             with __TWIN_CONFIG_JSON__ substitution       │
│  └─ Mounted via register(mount_server=True):                │
│      /twin/api/* → Twin overlay sub-app                     │
│        ├─ /workspaces, /notifications, /tags, /activity     │
│        ├─ /documents/{id}/metadata, /approve, /reject       │
│        ├─ /graph/entities, /graph/relations                 │
│        └─ /auth/logout                                      │
├─────────────────────────────────────────────────────────────┤
│  Pre-insert hook (Couche 2)                                 │
│  install_classification_hook(label_map, ceiling, audit_emit)│
│  → DocStatus.metadata.classification = ClassificationResult │
├─────────────────────────────────────────────────────────────┤
│  Memgraph (KV + Vector + DocStatus + Graph)                 │
└─────────────────────────────────────────────────────────────┘
```

### Files to create / extend

Current state after the 2026-06-04 audit: the server package and most shared
plumbing already exist. Do not recreate parallel `routes_*` modules unless a
refactor is intentional; first close the contract gaps in the existing
`webui_router.py`, `native_shims.py`, stores, and tests.

| File | Action | Why |
|---|---|---|
| `src/twindb_lightrag_memgraph/server/webui_router.py` | **EDIT** | Add missing real routes: `/documents/{id}/metadata`, `/documents/bulk-delete`, `/health`; Graph GET/PATCH/POST/DELETE lifecycle is M12-complete and should stay covered |
| `src/twindb_lightrag_memgraph/server/space_store.py` | **DONE** | Runtime CRUD catalog for non-env-seeded spaces, with optional atomic JSON persistence via `TWIN_SPACES_RUNTIME_FILE` |
| `src/twindb_lightrag_memgraph/server/native_shims.py` | **EDIT** | Keep native `/documents`, `/documents/{id}/chunks`, `/health`, `/pipeline_status`, `/openapi` aligned with React contract and space filtering |
| `src/twindb_lightrag_memgraph/server/webui_*store.py` | **EDIT** | Ensure tags, activity, notifications, document overlay metadata, and graph lifecycle mutations persist per space in Memgraph |
| `src/twindb_lightrag_memgraph/server/auth.py` | **EDIT** | Replace local JWT username/password path with MyAccess/IdP/JWKS validation for production mode |
| `src/twindb_lightrag_memgraph/server/space.py` | **EDIT** | Keep `X-Twin-Space` as canonical and retire `X-Twin-Workspace` only after all callers migrate |
| `src/twindb_lightrag_memgraph/__init__.py` | **EDIT** | `replace_ui`, `mount_server`, `shim_native_routes`, runtime config substitution, and direct Twin router mount already exist; keep extending these paths rather than booting a second LightRAG |
| `tests/test_server/` | **EDIT** | Add real backend contract tests, including route parity and Memgraph persistence. Integration tests remain `@pytest.mark.integration` |
| `lightrag_webui_twin/index.html` | **VERIFY** | Confirm the `__TWIN_CONFIG_JSON__` placeholder is still in place (already there per `useAuth.ts`) |
| `lightrag_webui_twin/src/api/client.ts` | **VERIFY** | Runtime API bases and `X-Twin-Space` are already wired; add prod error behavior for missing backend/config |
| `lightrag_webui_twin/src/api/resources.ts` | **VERIFY + TEST** | Treat as the frontend source of expected paths; route parity test must catch path drift |
| `lightrag_webui_twin/src/App.tsx` | **EDIT** | Remove silent production fallbacks to local fixtures; keep fixtures only for dev/MSW first paint |
| `pyproject.toml` | **EDIT** | Move `olefile` and `pikepdf` from optional to a new extra `pip install twindb-lightrag-memgraph[classification]` |
| `README.md` | **EDIT** | New "Couche 3 — Real backend wiring" section linking to this doc |

### Concrete tasks (ordered)

#### 3.1 — FastAPI route parity + missing routes (4-6h)

- [x] Server package, WebUI router, space binding, native shims, direct
      Twin router mount, and runtime config substitution exist.
- [x] Add route parity tests that diff frontend `resources.ts`, MSW handlers,
      and the actual FastAPI route table. MSW must never be the only
      implementation of a production path.
- [ ] Complete document overlay routes in `webui_router.py`:
  - `GET /documents/{id}/metadata` → reads DocStatus from
    `MemgraphDocStatusStorage`, returns `{tags, space, review,
    classification}` (classification from `DocStatus.metadata.classification`,
    tags from `[:TAGGED_WITH]` graph relations).
  - `POST /documents/bulk-delete` body `{doc_ids, actor}` → deletes each doc
    via LightRAG, emits one activity per doc, returns `{deleted}`.
  - `POST /documents/{id}/approve` and `/reject` already exist; harden them
    with space checks and response shape matching the frontend `Document`
    contract.
- [ ] Add Twin overlay health:
  - `GET /health` under `/twin/api` → reports overlay status and backing store
    availability separately from native LightRAG `/health`.
- [x] Complete graph lifecycle routes:
  - `GET /graph/entities` and `/graph/relations`
  - `PATCH /graph/entities/{id}` and `/graph/relations/{id}`
  - entity/relation creation and deletion
  - GraphTab add entity, delete entity/relation, and add relation forms
- [x] Complete backend runtime space CRUD:
  - `POST /spaces`
  - `PATCH /spaces/{id}`
  - `DELETE /spaces/{id}`
  - optional `TWIN_SPACES_RUNTIME_FILE` persistence
  - env-seeded spaces immutable; delete refuses spaces with docs/tags
- [ ] Keep tag/activity/notification routes in `webui_router.py`; do not
      create duplicate `routes_tags.py` etc. unless the router is intentionally
      split as a refactor after parity tests are green.

#### 3.2 — Tag + activity + notifications persistence (4h)

Current state:

- [x] Tags, activity, and notifications have in-memory and Memgraph-backed
      stores.
- [x] `register(..., mount_server=True, webui_stores="memgraph")` wires those
      stores per configured Twin space during lifespan startup.
- [x] Tag mutations emit synthesized activity and notification events.

Still to do:

- [ ] Add backend tests proving a fresh Memgraph-backed space boots without
      demo tags/activity/notifications unless explicit seed/bootstrap is
      requested.
- [ ] Add retention/sweep strategy for activity and notifications.
- [ ] Extend persistence beyond tags/activity/notifications where the UI
      currently reads seed-only surfaces: thesaurus if it becomes
      operator-editable, and document overlay metadata. Graph entities and
      relations are M12-complete.

#### 3.3 — JWT middleware + space scoping (3h)

- [ ] `middleware_jwt.py`: decode the BNP IdP / Keycloak JWT from
      cookie (HttpOnly, SameSite=Lax). Validate signature against the
      IdP's JWKS. On valid token: set `request.state.user =
      AuthenticatedUser(...)` matching the TS type.
      Failure modes: missing cookie → 401, expired → 401 with
      `WWW-Authenticate: Bearer error="expired"`, signature mismatch
      → 401.
- [x] Frontend side: `apiFetch` sets `X-Twin-Space` on every request
      from the active runtime-configured space. It also sends
      `X-Twin-Workspace` temporarily for old route code.
- [x] Backend side: read `X-Twin-Space` first, accept
      `X-Twin-Workspace` as a temporary fallback, validate the id against
      the configured space list, and set `request.state.space`.
      Downstream routes read this to scope every Twin query in the same
      Memgraph database.
- [x] Wire space binding in `register()` / WebUI router so every
      `/twin/api/*` request hits it before route handlers.

#### 3.4 — index.html substitution + WebUI mount (2h)

- [x] `register(replace_ui=True)` can replace the native `/webui` mount with
      the React `dist/` and substitute `__TWIN_CONFIG_JSON__`.
- [x] Root `/assets`, favicon/icons, and `mockServiceWorker.js` side mounts are
      handled so a Vite build with absolute asset paths can run under
      `/webui/`.
- [ ] Verify production config generation does not ship a wide-open
      `debugUser`; debug identity must remain dev/local only once MyAccess is
      wired.
- [ ] Keep the runtime config shape aligned with:
      ```python
      json.dumps({
        "apiBaseUrl": "/twin/api",
        "lightragBaseUrl": "",
        "idpLogoutUrl": os.environ["TWIN_IDP_LOGOUT_URL"],
        "defaultSpaceId": os.environ["TWIN_DEFAULT_SPACE"],
        "spaces": json.loads(os.environ["TWIN_SPACES_JSON"]),
        "maxSpaces": 5,
        "debugUser": None,  # PROD: no debug user, real JWT decoded server-side
      })
      ```
- [ ] Add a smoke test that starts the patched LightRAG app, fetches
      `/webui/`, and asserts the placeholder is gone and `apiBaseUrl` points
      to `/twin/api`.

#### 3.5 — Classification hook integration (1h)

- [ ] In the FastAPI startup (`@app.on_event("startup")` or lifespan),
      call `install_classification_hook(label_map_path,
      ceiling=os.environ.get("TWIN_MIP_MAX_CLASSIFICATION", "C2"),
      audit_emit=emit_to_memgraph_activity)`.
- [ ] Patch LightRAG's `insert()` call site to run the hook BEFORE
      passing the file to LightRAG. On `ClassificationRejection`, mark
      the DocStatus `status="FAILED"` + `error_msg=str(exc)` instead of
      ingesting.

#### 3.6 — Frontend cutover (2h)

- [x] `client.ts`: read `apiBaseUrl` from `window.__twinConfig`. The
      MSW gate in `main.tsx` already turns off MSW unless
      `VITE_FORCE_MSW=true`, so a PROD build automatically hits the
      real backend.
- [x] Add `X-Twin-Space` header to every fetch: `apiFetch` reads the
      active space set by the App/Topbar selector. `X-Twin-Workspace`
      remains as a temporary compatibility header.
- [x] `useAuth.signout()` is already wired correctly (POST
      `/twin/api/auth/logout` → `queryClient.clear()` → redirect IdP).
      Confirm the IdP URL is read from `window.__twinConfig.idpLogoutUrl`.
- [ ] Remove silent production fallbacks from `App.tsx`:
  - `DOCUMENT_FIXTURES`
  - `TAG_FIXTURES`
  - `TAG_CATEGORY_FIXTURES`
  - `ACTIVITY_FIXTURES`
  - `GRAPH_ENTITY_FIXTURES`
  - `GRAPH_RELATION_FIXTURES`
  - `THESAURUS_FIXTURES`
  - `NOTIFICATION_FIXTURES`
  Dev/MSW can keep fixture first paint, but prod must show loading/error when
  the backend is unavailable.
- [ ] Retrieval should stop displaying fixture answer sources in real mode.
      Until structured source extraction lands, show the backend response with
      an honest empty/unknown sources state.

#### 3.7 — Integration tests (3h)

- [x] `tests/test_server/` package exists.
- [x] Add `test_route_parity.py`: inspect FastAPI routes from the patched app,
      compare against `resources.ts` production paths, and flag any route that
      exists only in MSW.
- [ ] `test_metadata_endpoint.py`: insert a doc with structured
      `metadata.classification`, GET `/twin/api/documents/{id}/metadata`,
      assert the classification is in the response.
- [ ] `test_bulk_delete_endpoint.py`: insert N docs, POST
      `/twin/api/documents/bulk-delete`, assert docs disappear from
      `/documents`, chunks are not retrievable, and activity events are
      written.
- [x] Graph backend/frontend regression coverage: M12 reports real Memgraph
      GET/PATCH/POST/DELETE lifecycle covered in the `634/634` pytest and
      `364/364` vitest baseline.
- [ ] `test_twin_health_endpoint.py`: assert `/twin/api/health` reports overlay
      store status and fails/degrades when Memgraph-backed stores cannot
      initialize.
- [ ] `test_classification_rejection.py`: ingest a synthetic .docx
      tagged C3 with `TWIN_MIP_MAX_CLASSIFICATION=C2`, assert the
      DocStatus is `FAILED` with the expected `error_msg`, and an
      activity event of kind `classification-rejected` exists.
- [ ] `test_space_scoping.py`: insert docs in spaces `default` and
      `sandbox`, GET `/documents` with `X-Twin-Space: default`, assert
      only `default` docs returned.
- [ ] `test_jwt_middleware.py`: GET `/twin/api/workspaces` without
      cookie → 401; with a valid JWT for a user allowed on the parent
      KB → 200 with the configured space list.
- [ ] Add the new tests to the `integration-tests` job in
      `.forgejo/workflows/ci.yml` (already includes a Memgraph service
      container).
- [ ] Add one real-backend WebUI smoke with MSW disabled
      (`VITE_USE_MSW=false`) against a running FastAPI app.

#### 3.8 — Deployment (2h)

- [ ] Build wheel: `python -m build` produces `dist/twindb_lightrag_memgraph-1.2.0-py3-none-any.whl`.
- [ ] Build webui dist: `cd lightrag_webui_twin && bun run build`
      (NOT with `VITE_FORCE_MSW=true` — that's only for the standalone
      OVH demo).
- [ ] Ship the wheel + webui dist to the LightRAG host (BNP infra,
      OVH staging, etc.).
- [ ] On host: `pip install twindb_lightrag_memgraph-1.2.0-py3-none-any.whl[classification]`
      (extra includes `olefile` + `pikepdf` for non-OOXML formats).
- [ ] Set env vars: `MEMGRAPH_URI`, `TWIN_MIP_LABEL_MAP`,
      `TWIN_MIP_MAX_CLASSIFICATION`, `TWIN_IDP_LOGOUT_URL`,
      `TWIN_JWKS_URL`, etc.
- [ ] Start the LightRAG server with `register(replace_ui=True,
      mount_server=True, webui_dist_path="/opt/twin/webui")`.
- [ ] Smoke check `https://<host>/webui/` loads the React port +
      hits real `/twin/api/*` endpoints.

### Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Keycloak JWKS endpoint unreachable from the LightRAG host | Medium | Cache the JWKS at startup with a TTL; fail closed (401) if cache stale + refresh fails. |
| Memgraph labels for tags/activity grow unbounded | Low (90d retention per Settings) | Implement the retention sweep job — see `docs/operations/install-runbook.md` §5 |
| `register()` flag explosion (replace_ui, mount_server, classify, ...) | High if not designed carefully | Group under a single `extensions=ExtensionConfig(...)` dataclass; keep `register()` signature small |
| Frontend MSW removal breaks the dev story | Low (MSW stays on in DEV by default) | The activation matrix in `main.tsx` is already documented — don't regress |
| BNP tenant label map mismatched with Compliance Center | Medium | Add a `/twin/api/classification/_self_check` debug endpoint that returns the loaded map + lets ops validate visually |
| Real fetch + TanStack Query cache thrash on space switch | Medium | On `setActiveSpace`, call `queryClient.removeQueries()` for all `['documents', '...']` keys — already in App.tsx skeleton, just needs confirmation |

### Sequencing recommendation

```
3.1 route parity + missing routes (4-6h) ─┬─→ 3.5 classification hook (1h) ─┐
                             │                                  │
3.2 persistence (4h) ────────┤                                  │
                             │                                  ├─→ 3.8 deploy (2h)
3.3 JWT + space (3h) ────────┤                                  │
                             │                                  │
3.4 index.html mount (2h) ───┤                                  │
                             │                                  │
                             └─→ 3.6 frontend cutover (2h) ─────┤
                                                                │
                              3.7 integration tests (3h) ───────┘
```

Total estimate: **18-22h of focused work** for a single engineer who
knows the codebase. Realistic shipping window: a focused 3-day sprint
once the tenant label map is in hand from Louis HORVAT.

---

## References

- **Brief sprint coder (Couches 1+2)** : `docs/handoff/SPRINT-2026-05-30-coder-brief.md`
- **Pitch deck Fabrice 2026-06-01** : `docs/presentations/pitch-fabrice-2026-06-01.md`
- **Install runbook (production)** : `docs/operations/install-runbook.md`
- **Rapport recette Alberto** : `docs/audits/TwinRAG - Rapport de recette v2.md`
- **Design-agent prototype** : `~/Downloads/prototype/` (Tier-1 visual reference)
- **PR #157** : MSIP classification (Python module + hook + 42 tests)
- **PR #158** : React port from prototype + Couche 2 UI (320/320 tests)
- **PR #159** : 11 visual fixes from dev smoke
- **PR #160** : VITE_FORCE_MSW + Dockerfile.react + standalone OVH deploy
- **Memory** : `~/.claude/projects/-Users-julien-twindb-lightrag-memgraph/memory/`
  - `project_louis_eric_meeting_2026-05-28.md` (compliance doctrine)
  - `project_twin_myaccess_rights_model.md` (palier ↔ classification mapping)
  - `project_webui_fork.md` (history of the React port)
