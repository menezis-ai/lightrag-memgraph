# WebUI Wiring — Wired

As-built state on `stable/0.6.x` (HEAD `7302023` at split time).

## Frontend (`lightrag_webui_twin/`)

- React 19 + TypeScript + Vite + Bun. Production build via `bun run build`; tests via `bun run test:run` (33 files / 396 tests).
- Runtime config read from `window.__twinConfig` (server-injected) or e2e override: `apiBaseUrl`, `lightragBaseUrl`, `idpLogoutUrl`, `defaultSpaceId`, `spaces`, `maxSpaces`.
- Production builds hit the real backend; `VITE_FORCE_MSW=true` opts into MSW for the OVH standalone-demo path only.
- `apiFetch` sends `X-Twin-Space` (preferred) and `X-Twin-Workspace` (legacy compat window) on every request.
- Visible copy uses "Space" for Twin sub-scopes (Fabrice doctrine 2026-06-01).
- e2e Playwright suite covers documents, tags, retrieval, graph, activity, settings/auth guardrails, runtime spaces, upload validation, async/a11y hardening.

## Twin Spaces

- Env-driven catalog:
  - `TWIN_DEFAULT_SPACE` (fallback `WORKSPACE`, then `default`).
  - `TWIN_DEFAULT_SPACE_LABEL`.
  - `TWIN_SPACES_JSON` (admin/runtime catalog).
  - `TWIN_MAX_SPACES` clamped to `1..5`.
- Backend reads `X-Twin-Space` first, accepts `X-Twin-Workspace` temporarily, validates against the configured catalog, and binds the active space via `ContextVar`.
- Native document shims filter on `DocStatus.metadata.space`.
- Admin Space CRUD (`POST/PATCH/DELETE /twin/api/spaces`) gated by `admin:spaces` gateway scope, derived from `TWIN_IDP_ADMIN_GROUPS` env (default `twin-admin,twin-steward`). See commit `a62b4b4`.

## Backend Overlay

- `register(replace_ui=True, mount_server=True, shim_native_routes=True, webui_stores=..., security_baseline=True)` wires the React WebUI dist and Twin API surface into the host LightRAG app.
- `server/webui_router.py` exposes `/twin/api/{tags, activity, notifications, documents, spaces, workspaces, graph, openapi, ...}`.
- `server/native_shims.py` re-shapes LightRAG's native FastAPI surface (`/documents`, `/health`, `/pipeline_status`, `/documents/{id}/chunks`) to match the React port's contract — Twin = AI-readable surface, LightRAG gets translated.
- `server/twin_query_routes.py` adds `POST /twin/api/query` (synchronous) and `POST /twin/api/query/stream` (token streaming) with advanced controls: `chunk_top_k`, `enable_rerank`, `user_prompt`, `history_turns`, `tag_filter`. Commits `524b2a8` + `a6ff23a`.
- WebUI stores have in-memory and Memgraph variants:
  - `webui_tagstore.py`
  - `webui_activitystore.py`
  - `webui_notificationstore.py`
- Memgraph stores initialized per configured Twin space when `webui_stores="memgraph"`. Fresh spaces boot empty (no demo seed leak — see mock-kill F6).

## Mock-kill audit remediation (`docs/audits/webui-fork/mock-kill-audit-2026-06-04.md`)

Commit `731f0d1` closes:

- **F1** Settings → Space identity card reads runtime config (no more hardcoded `eu-west-3 · dc-paris` / fictional TTLs).
- **F2** Settings → API tab fetches `/openapi.json` direct (ISO LightRAG by construction); "Try it out" performs a real `fetch`; the fictional `cib-kb.twin.internal` server selector is dropped.
- **F3** Graph tab detail panel drops fixture lookups that returned empty for real Memgraph entities.
- **F5** Boot WARN when `webui_stores="seed"` runs under an active IdP (production trap).
- **F6** `WebuiStore.for_space(mode="memgraph")` so the default space doesn't expose demo `_documents` / `_graph_entities` on a real deploy.
- F4 (RetrievalTab dead fixture fallback) shipped with `524b2a8`.

## Retrieval (real backend)

- Streaming + advanced controls landed in `524b2a8` (`POST /twin/api/query/stream` via FastAPI `StreamingResponse`, frontend `useStreamQuery` consumes tokens).
- `tag_filter` end-to-end in `a6ff23a`: UI captures tags → `App.tsx` maps to `{all: [...]}` → backend `TwinQueryBody.tag_filter` → LightRAG `QueryParam`.
- Tag delete cascade fix in `7302023`: `affected_docs = (graph_affected or 0) + seed_affected` so a fresh-Memgraph CI run doesn't mask the in-memory seed cascade.

## Tests run

- `pytest tests/test_server/` — 392 passed.
- `pytest tests/ --ignore=tests/test_bench.py` — 768 passed, 97 skipped (integration auto-skip without `MEMGRAPH_URI`).
- `bun run typecheck` — OK.
- `bun run test:run` — 396 passed.
- Playwright `test:e2e` — runs against MSW dev mode + a `:real` lane against a configured backend (`REAL_BACKEND_URL`).
