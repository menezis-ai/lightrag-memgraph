# WebUI Wiring — Wired

Historical as-built state captured on `stable/0.6.x` (HEAD `7302023` at split
time). This file is retained for audit context. The live contract is now
Folder-based; use [WEBUI-WIRING-PLAN.md](WEBUI-WIRING-PLAN.md) and the tests as
the source of truth.

## Frontend (`lightrag_webui_twin/`)

- React 19 + TypeScript + Vite. The Forgejo WebUI jobs use Bun for lint/unit/build; the GitHub mirror uses npm.
- Runtime config read from `window.__twinConfig` (server-injected) or e2e override: `apiBaseUrl`, `lightragBaseUrl`, `idpLogoutUrl`, `defaultFolderId`, `folders`, `maxFolders`.
- Production builds hit the real backend; `VITE_FORCE_MSW=true` opts into MSW for the OVH standalone-demo path only.
- `apiFetch` sends `X-Twin-Folder` on requests bound to the active Twin folder.
- Visible copy uses "Folder" for Twin sub-scopes.
- e2e Playwright suite covers documents, tags, retrieval, graph, activity, settings/auth guardrails, runtime folders, upload validation, async/a11y hardening.

## Twin Folders

- Env-driven catalog:
  - `TWIN_DEFAULT_FOLDER` (fallback `WORKSPACE`, then `default`).
  - `TWIN_DEFAULT_FOLDER_LABEL`.
  - `TWIN_FOLDERS_JSON` (admin/runtime catalog).
  - `TWIN_MAX_FOLDERS` clamped to `1..5`.
- Backend reads `X-Twin-Folder`, validates against the configured catalog, and binds the active folder via `ContextVar`.
- Native document shims filter on `DocStatus.metadata.folder`.
- Admin Folder CRUD (`POST/PATCH/DELETE /twin/api/folders`) gated by `admin:folders` gateway scope.

## Backend Overlay

- `register(replace_ui=True, mount_server=True, shim_native_routes=True, webui_stores=..., security_baseline=True)` wires the React WebUI dist and Twin API surface into the host LightRAG app.
- `server/webui_router.py` exposes `/twin/api/{tags, activity, notifications, documents, folders, graph, openapi, ...}`.
- `server/native_shims.py` re-shapes LightRAG's native FastAPI surface (`/documents`, `/health`, `/pipeline_status`, `/documents/{id}/chunks`) to match the React port's contract — Twin = AI-readable surface, LightRAG gets translated.
- `server/twin_query_routes.py` adds `POST /twin/api/query` (synchronous) and `POST /twin/api/query/stream` (token streaming) with advanced controls: `chunk_top_k`, `enable_rerank`, `user_prompt`, `history_turns`, `tag_filter`. Commits `524b2a8` + `a6ff23a`.
- WebUI stores have in-memory and Memgraph variants:
  - `webui_tagstore.py`
  - `webui_activitystore.py`
  - `webui_notificationstore.py`
- Memgraph stores initialized per configured Twin folder when `webui_stores="memgraph"`. Fresh folders boot empty (no demo seed leak — see mock-kill F6).

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
