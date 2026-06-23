# Twin KMS WebUI Wiring Plan

This page is the live map for the Twin KMS WebUI and backend overlay. The two
companion files keep the split explicit:

- [WEBUI-WIRING-WIRED.md](WEBUI-WIRING-WIRED.md) is the current as-built
  inventory.
- [WEBUI-WIRING-TO-WIRE.md](WEBUI-WIRING-TO-WIRE.md) is the remaining backlog
  and policy-gated work.

Use this file when deciding whether a change belongs in the current product
contract. Use the tests and `.forgejo/workflows/ci.yml` as the executable
contract.

## Product Contract

Twin KMS is a knowledge management system on top of LightRAG and Memgraph. It is
not a document management system. The document source of truth remains the
origin system: SharePoint, an internal DMS, mail, or another upstream source.

Twin KMS owns:

- the operator WebUI served at `/webui`;
- the Twin overlay API mounted under `/twin/api`;
- folder-scoped governance data: tags, activity, notifications, API keys,
  folder catalog, quota status, and graph projections;
- native LightRAG route shims where the React WebUI needs a stable contract;
- source metadata and retrieval traces needed to answer and audit questions.

Twin KMS must not silently become durable raw-document storage. Any route that
re-opens source download or full raw-body preview for sensitive material needs
explicit PO and compliance sign-off before implementation.

## Live Runtime Contract

The live scoping primitive is the Twin Folder:

- `TWIN_DEFAULT_FOLDER`
- `TWIN_DEFAULT_FOLDER_LABEL`
- `TWIN_FOLDERS_JSON`
- `TWIN_MAX_FOLDERS`
- browser runtime config `defaultFolderId`, `folders`, `maxFolders`
- request header `X-Twin-Folder`
- API surface `/twin/api/folders`

Folders are an operator-facing subdivision inside one deployed KB. They are not
the same thing as LightRAG's storage workspace labels. The graph still follows
the deployed LightRAG/Memgraph workspace; folder scoping applies to WebUI
governance state, document metadata, native document shims, and operator flows.

## Status Snapshot

Last aligned: 2026-06-23.

| Area | Status | Source of truth |
|---|---|---|
| Product name and UI brand | Wired as Twin KMS | `README.md`, WebUI brand tests |
| React WebUI replacement at `/webui` | Wired | `register(replace_ui=True)` and smoke tests |
| Twin overlay under `/twin/api` | Wired | `server/webui/router.py`, `server/webui_router.py` compatibility wrapper |
| Runtime Folder catalog and switcher | Wired | `server/folder.py`, `server/folder_store.py`, `FolderSwitcher` |
| `X-Twin-Folder` frontend/backend contract | Wired | route tests and e2e runtime folder specs |
| Admin Folder CRUD | Wired | `/twin/api/folders`, `admin:folders` gating |
| Memgraph WebUI stores | Wired | tag, activity, notification stores |
| Native LightRAG route shims | Wired | `server/native_shims.py` and route parity tests |
| Query, streaming query, query data | Wired | `/twin/api/query`, `/query/stream`, `/query/data` |
| Retrieval `tag_filter` | Wired only where supported | `/twin/api/query/data`; reject on routes that cannot honor it |
| API key minting and generated-key e2e | Wired | `/twin/api/settings/api-keys`, `webui-e2e-keygen` |
| Quota snapshot route | Wired | `/twin/api/quota` |
| Real-backend Playwright coverage | Wired but runner-sensitive | `webui-e2e-real`, `webui-e2e-keygen` |
| MyAccess / IdP JWT mechanics | Code-ready | `TWIN_IDP_JWKS_URL`, IdP tests |
| Real MyAccess deployment | Pending ops integration | needs real JWKS/issuer/audience |
| Deployment smoke/runbook | Partially wired | `tests/smoke`, `docs/operations/install-runbook.md` |
| Retention and sweep policy | Deferred by policy | PO/compliance decision required |
| BNP MIP classification hook | PO-gated | opt-in modules only |

## CI Contract

Forgejo is the authoritative CI for this repository.

- Python unit and integration lanes run in Python containers on the self-hosted
  runner pool.
- Frontend quality/build uses Bun on the runner where the workflow declares it.
- Playwright lanes run inside the Microsoft Playwright container and use npm.
- `webui-e2e` runs against MSW.
- `webui-e2e-real` starts a real Twin backend plus Memgraph and runs real
  backend smoke/mutation coverage.
- `webui-e2e-keygen` starts its own real Twin backend plus Memgraph and proves
  generated API keys work as real credentials.
- Real/backend Playwright jobs must not rely on fixed host ports; Docker should
  allocate host ports and propagate `REAL_BACKEND_URL` to later steps.

The GitHub workflow is a mirror-oriented convenience workflow, not the complete
contract for the Forgejo release train.

## Change Discipline

- A new frontend route must either hit a real backend route or be listed as a
  deliberate known gap in route parity tests.
- A new Twin overlay route must be covered by server tests and, when visible to
  operators, by WebUI or Playwright coverage.
- A LightRAG shim must preserve native behavior unless the Twin contract
  explicitly documents the translation.
- A new compliance/security behavior must fail closed in production or be gated
  behind an explicit opt-in env var.
- Sonar cleanup is welcome only when it preserves those contracts; suppressions
  are acceptable for framework-required async or protocol shapes.
