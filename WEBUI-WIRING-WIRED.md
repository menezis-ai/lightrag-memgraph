# Twin KMS WebUI Wiring — Wired

This file is the current as-built inventory for the Twin KMS WebUI and backend
overlay. It is not a backlog. Use [WEBUI-WIRING-PLAN.md](WEBUI-WIRING-PLAN.md)
for the live contract and [WEBUI-WIRING-TO-WIRE.md](WEBUI-WIRING-TO-WIRE.md)
for remaining work.

Last aligned: 2026-06-23.

## Frontend

- React 19, TypeScript, Vite.
- Product-facing brand is Twin KMS. The shorter "Twin" remains valid for the
  ecosystem, team, route prefix, and chatbot-facing offer.
- The WebUI is served at `/webui` when `register(replace_ui=True)` is active.
- Runtime config is read from `window.__twinConfig`, with e2e override support:
  `apiBaseUrl`, `lightragBaseUrl`, `idpLogoutUrl`, `defaultFolderId`,
  `folders`, `maxFolders`.
- Production builds hit the real backend by default. `VITE_FORCE_MSW=true` is
  an explicit demo/test override, not the production path.
- `apiFetch` centralizes backend calls and sends `X-Twin-Folder` for
  folder-bound API calls.
- The operator journeys covered by Playwright include documents, upload, tags,
  retrieval, graph, activity, settings, runtime folders, auth guardrails,
  responsive topbar, modal accessibility, and real-backend API coverage.

## Twin Folders

- Environment-driven catalog:
  - `TWIN_DEFAULT_FOLDER`
  - `TWIN_DEFAULT_FOLDER_LABEL`
  - `TWIN_FOLDERS_JSON`
  - `TWIN_MAX_FOLDERS`
- Backend parsing and request binding live in `server/folder.py` and
  `server/folder_store.py`.
- The browser sends the active Folder through `X-Twin-Folder`.
- Folder catalog routes live under `/twin/api/folders`.
- Admin Folder mutations require `admin:folders`.
- Env-seeded Folders are protected from deletion through the API.
- Runtime-added Folders persist only when a backing runtime store is configured.
- Native document shims respect folder metadata so `/documents` does not leak
  cross-Folder rows.

## Backend Overlay

- `register(replace_ui=True, mount_server=True, shim_native_routes=True,
  webui_stores=..., security_baseline=True)` wires the WebUI, overlay API,
  native shims, and runtime install guardrails.
- `server/webui/router.py` is the modular overlay router.
- `server/webui_router.py` remains as a compatibility wrapper for older imports.
- Overlay routes include:
  - `/twin/api/health`
  - `/twin/api/folders`
  - `/twin/api/documents`
  - `/twin/api/documents/{id}/metadata`
  - `/twin/api/documents/_bulk-retag`
  - `/twin/api/documents/bulk-delete`
  - `/twin/api/documents/{id}/approve`
  - `/twin/api/documents/{id}/reject`
  - `/twin/api/tags`
  - `/twin/api/tags/categories`
  - `/twin/api/graph/entities`
  - `/twin/api/graph/relations`
  - `/twin/api/activity`
  - `/twin/api/notifications`
  - `/twin/api/thesaurus`
  - `/twin/api/settings/api-keys`
  - `/twin/api/quota`
  - `/twin/api/openapi/groups`
- WebUI stores have in-memory and Memgraph variants for tags, activity, and
  notifications.
- Memgraph WebUI stores are initialized per configured Twin Folder when
  `webui_stores="memgraph"`.
- Fresh Memgraph-backed folders boot clean; demo seed data is not exposed on a
  real deployment.

## Native LightRAG Shims

`server/native_shims.py` adapts the native LightRAG FastAPI surface where the
React WebUI needs stable behavior:

- `/documents`
- `/documents/scan`
- `/documents/{id}/chunks`
- `/health`
- `/pipeline_status`
- `/openapi.json`
- `/logout`

The rule is translation, not takeover: native LightRAG behavior must remain
reachable unless a shim is explicitly part of the Twin KMS WebUI contract.
Route parity tests guard this boundary.

## Query and Retrieval

- `POST /twin/api/query` returns a structured non-streaming answer.
- `POST /twin/api/query/stream` streams answer events.
- `POST /twin/api/query/data` wraps `LightRAG.aquery_data()`.
- Advanced controls include `chunk_top_k`, `enable_rerank`, `user_prompt`,
  `history_turns`, and Twin-specific filter plumbing where supported.
- `tag_filter` is honored only where the backend can enforce it honestly:
  `/twin/api/query/data`.
- Routes that cannot enforce `tag_filter` reject it rather than pretending.
- The API tab reflects that split so operator "Try it out" requests do not send
  misleading filters to unsupported query routes.

## Auth and API Keys

- Production fails closed unless an auth backend is configured:
  `LIGHTRAG_API_KEY`, `LIGHTRAG_JWT_SECRET`, `TWIN_IDP_JWKS_URL`, or explicit
  `TWIN_ALLOW_OPEN_ACCESS=1`.
- `TWIN_IDP_JWKS_URL` activates IdP JWT validation.
- The IdP path supports admin group mapping and `admin:folders` enforcement.
- Generated API keys are minted through `/twin/api/settings/api-keys`.
- `webui-e2e-keygen` proves a generated key can authenticate real API requests.
- `POST /twin/api/auth/logout` exists as the Twin logout ack/future cookie hook.

## CI and Test Coverage

Forgejo is the main CI surface:

- Python unit lanes cover supported Python and LightRAG combinations.
- Integration lanes cover Memgraph-backed behavior.
- Frontend quality/build uses the workflow's declared Node/Bun setup.
- Playwright MSW e2e runs in a Playwright container.
- Real-backend Playwright e2e starts Memgraph plus a real Twin backend.
- Generated-key Playwright e2e starts an isolated real Twin backend and proves
  the API-key minting flow.
- Real-backend Playwright jobs use dynamically allocated host ports and pass the
  resulting `REAL_BACKEND_URL` to later steps.

Local validation commands that map to the contract:

```bash
uv run pytest tests/test_server/
uv run pytest tests/ --ignore=tests/test_bench.py
cd lightrag_webui_twin && npm run typecheck
cd lightrag_webui_twin && npm run test:run
cd lightrag_webui_twin && npm run test:e2e
```

## Guardrails Already in Place

- Mock-kill remediation removed fixture fallbacks from production-facing
  settings, API, graph, retrieval, and store initialization paths.
- Route parity tests prevent the frontend from quietly depending on missing
  backend routes.
- Graph tests treat create/update/delete failures as backend errors, not fake
  successes.
- Folder tests protect the `X-Twin-Folder` contract across native shims and
  overlay routes.
- Security baseline blocks runtime package installation unless explicitly
  disabled for development.
