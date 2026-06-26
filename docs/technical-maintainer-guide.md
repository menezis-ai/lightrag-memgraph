# Technical Maintainer Guide

This guide is the handover document for engineers who need to maintain Twin KMS
without relying on implicit project knowledge. It explains the architecture,
runtime boot paths, data model, risky areas, test doctrine, and common change
playbooks.

The short version: Twin KMS is a Memgraph-backed LightRAG runtime and server
overlay. It does not fork LightRAG. It registers storage implementations and,
when requested, installs a server/WebUI overlay around LightRAG.

## 1. First Principles

Twin KMS provides:

- Memgraph storage adapters for LightRAG KV, vector, document status, and graph
  data.
- A Twin server overlay that preserves LightRAG API compatibility while adding
  folder-aware document management, auth, classification, source projection,
  graph CRUD, and WebUI routes.
- A Twin WebUI that extends the upstream LightRAG user experience for
  folder-scoped workflows.
- Test coverage that tracks upstream LightRAG compatibility across several
  LightRAG and Memgraph versions.

Twin KMS is not a hard fork of LightRAG. Avoid copying upstream LightRAG source
into this repository unless there is no cleaner option. The preferred pattern is
registration, patching, shimming, and extension at runtime.

Twin KMS is also not a second application with a fully separate domain model.
Most flows still pass through LightRAG concepts: working directory, namespace,
document status, vector storage, graph storage, query parameters, and upstream
server route shapes.

## 2. Compatibility Promise

The production baseline is:

- LightRAG `1.4.9.11`
- Memgraph MAGE `3.9.0`

Forward compatibility is tested for:

- LightRAG `1.4.11`
- LightRAG `1.4.12`
- Memgraph MAGE `3.10.1`

Do not describe LightRAG upgrades as untested by default. The Forgejo CI matrix
tests LightRAG `1.4.9.11`, `1.4.11`, and `1.4.12`, and integration tests cover
Memgraph MAGE `3.9.0` and `3.10.1`.

The public GitHub workflow is intentionally reduced. The compatibility gate is
Forgejo CI in `.forgejo/workflows/ci.yml`.

## 3. Repository Map

Start with these files:

- `README.md` - installation, runtime entrypoints, config, and operator-facing
  quick start.
- `CLAUDE.md` - detailed project doctrine and operational rules.
- `DOCTRINE.md` - strategic intent and non-negotiable project principles.
- `ENV_VARIABLES.txt` - environment variable reference.
- `docs/test-doctrine-lightrag-compat.md` - compatibility test doctrine.
- `docs/test-doctrine-graph.md` - graph-storage test doctrine.
- `docs/operations/install-runbook.md` - production install and runbook.
- `WEBUI-WIRING-PLAN.md` - WebUI wiring plan.
- `WEBUI-WIRING-WIRED.md` - wired WebUI surface.
- `WEBUI-WIRING-TO-WIRE.md` - remaining WebUI wiring work.

Core backend code lives under:

- `twindb_lightrag_memgraph/`
- `twindb_lightrag_memgraph/patches/`
- `twindb_lightrag_memgraph/server/`
- `twindb_lightrag_memgraph/server/webui/`
- `twindb_lightrag_memgraph/server/query/`
- `twindb_lightrag_memgraph/intelligence/`

Frontend code lives under:

- `lightrag_webui_twin/src/`
- `lightrag_webui_twin/e2e/`
- `lightrag_webui_twin/tests/`

Packaging and runtime entrypoints include:

- `pyproject.toml`
- `twin_main.py`
- `twindb_lightrag_memgraph/lightrag_server.py`
- `twindb_lightrag_memgraph/asgi.py`
- `Dockerfile`

CI surfaces include:

- `.forgejo/workflows/ci.yml` - primary compatibility and integration matrix.
- `.github/workflows/ci.yml` - reduced GitHub public mirror checks.

## 4. Distribution Surfaces

The project has three important distribution surfaces:

- Bunker repository: private source of truth.
- GitHub `main`: public backend patch surface.
- GitHub `export-1.0.0`: BNP delivery snapshot.

When a change affects production behavior, update the private source first and
then make sure the intended public/export surfaces receive the same effective
patch. Do not assume GitHub Actions fully represent the production test matrix;
the main matrix is in Forgejo.

## 5. Runtime Modes

### Storage-Only Mode

Use this when an application owns its own LightRAG server or script and only
needs Memgraph-backed storage:

```python
from twindb_lightrag_memgraph import register

register()
```

This registers Twin storage classes with LightRAG. It should run before the
application creates LightRAG instances that resolve storage implementation
names.

### Full Twin Overlay

Use this when the Twin server and WebUI overlay should be active:

```python
from twindb_lightrag_memgraph import register

register(
    replace_ui=True,
    mount_server=True,
    shim_native_routes=True,
)
```

This installs storage registration and server-side patches. Register before
importing or starting LightRAG server entrypoints that should receive the Twin
overlay.

Supported production entrypoints:

```bash
python twin_main.py
python -m twindb_lightrag_memgraph.lightrag_server
gunicorn 'twindb_lightrag_memgraph.asgi:get_application()' --bind 0.0.0.0:9621
```

The Docker image uses:

```bash
python -m twindb_lightrag_memgraph.lightrag_server
```

For local development:

```bash
uv run lightrag-server
```

## 6. Boot Sequence

The boot sequence matters because most integration is runtime registration or
patching:

1. Environment variables are loaded by LightRAG and Twin code.
2. `register()` installs storage implementations in LightRAG registries.
3. With the overlay flags (`mount_server`, `replace_ui`, `shim_native_routes`), Twin installs server overlay behavior.
4. The LightRAG server app is created.
5. Twin app/router code attaches compatibility routes, WebUI assets, auth
   behavior, folder logic, and API shims.
6. Requests flow through FastAPI route handlers into LightRAG and Twin storage.

When debugging unexpected upstream behavior, first confirm whether the failing
runtime was started before or after `register(...)` (with the overlay flags).

## 7. Core Architecture

### Registry And Patch Layer

Primary file:

- `twindb_lightrag_memgraph/patches/registry.py`

This is the highest-risk backend file. It is responsible for registering Twin
storage implementations and server patches without modifying upstream LightRAG
source.

Maintenance rules:

- Keep patches narrow and idempotent.
- Preserve upstream API signatures where possible.
- Add tests whenever a patch depends on an upstream symbol, route, attribute, or
  import path.
- Prefer feature detection over version checks when the upstream API has a
  stable capability to inspect.
- When a version check is unavoidable, explain why in code or tests.

### Storage Layer

Primary files:

- `twindb_lightrag_memgraph/kv_impl.py`
- `twindb_lightrag_memgraph/vector_impl.py`
- `twindb_lightrag_memgraph/docstatus_impl.py`
- `twindb_lightrag_memgraph/_pool.py`
- `twindb_lightrag_memgraph/_buffered_graph.py`
- `twindb_lightrag_memgraph/graph_impl.py`

The storage layer maps LightRAG storage contracts onto Memgraph:

- KV storage: namespace-key-value behavior.
- Vector storage: embeddings, metadata, similarity search, and namespace
  isolation.
- Document status storage: ingestion status, deduplication state, file metadata,
  and folder/workspace context.
- Graph storage: entities, relationships, labels, properties, and neighborhood
  traversal.

Maintenance rules:

- Treat LightRAG storage tests as contract tests.
- Do not change return shapes casually; route and UI code may depend on them.
- Keep Memgraph writes explicit and parameterized.
- Be careful with batch APIs. They are often where upstream compatibility breaks.
- Preserve namespace and workspace isolation.

### Server Overlay

Primary files:

- `twindb_lightrag_memgraph/server/app.py`
- `twindb_lightrag_memgraph/server/native_shims.py`
- `twindb_lightrag_memgraph/server/webui/routes_documents.py`
- `twindb_lightrag_memgraph/server/webui/router.py`
- `twindb_lightrag_memgraph/server/query/router.py`
- `twindb_lightrag_memgraph/server/graph_reader.py`
- `twindb_lightrag_memgraph/server/twin_query_routes.py`

The server overlay has two jobs:

- Preserve upstream LightRAG API compatibility.
- Add Twin behavior where product requirements need it.

Do not add product behavior directly to low-level storage code if the behavior
belongs at HTTP/API level. Keep storage adapters close to LightRAG contracts and
place route-specific policy in the server layer.

### Auth And Security

Primary files:

- `twindb_lightrag_memgraph/server/auth.py`
- `twindb_lightrag_memgraph/server/idp.py`
- `twindb_lightrag_memgraph/server/api_keys.py`
- `twindb_lightrag_memgraph/server/quota.py`

Default behavior mirrors LightRAG: the server can run without mandatory auth.
Production should fail closed by setting:

```bash
TWIN_REQUIRE_AUTH=true
```

or:

```bash
TWIN_ENV=production
```

Supported auth backends (pick one; see the auth section of `ENV_VARIABLES.txt`):

- `LIGHTRAG_API_KEY` - static bearer key.
- `LIGHTRAG_JWT_SECRET` (+ `LIGHTRAG_JWT_PASSWORD`) / `TOKEN_SECRET` - local `/login` JWT.
- `TWIN_IDP_JWKS_URL` - corporate IdP (JWKS), strict RBAC.

When none is configured the server runs open-access (LightRAG-native parity).

Maintenance rules:

- Any new mutating route must pass through the same auth expectations as similar
  existing routes.
- Do not create a second auth scheme for a single route.
- Keep dev-provider behavior explicit and non-production.
- Avoid silent auth bypasses in fallback routes and static/WebUI helpers.

### Classification And Intelligence

Primary files:

- `twindb_lightrag_memgraph/intelligence/classifier.py`
- `twindb_lightrag_memgraph/intelligence/docling.py`

Classification is optional and controlled by environment variables. It should
not be required for basic ingestion, query, or storage compatibility.

Maintenance rules:

- If classification fails, the core document flow should still have a clear and
  tested fallback unless the caller explicitly requests fail-closed behavior.
- Do not mix classifier output schemas with storage internals without a mapping
  layer.
- Keep AI/provider-specific assumptions outside route contracts where possible.

### WebUI

Primary files:

- `lightrag_webui_twin/src/app/AppShell.tsx`
- `lightrag_webui_twin/src/api/`
- `lightrag_webui_twin/src/components/`
- `lightrag_webui_twin/src/routes/`
- `lightrag_webui_twin/e2e/`
- `lightrag_webui_twin/tests/`

The WebUI is built separately and copied into the Python package:

```bash
cd lightrag_webui_twin
npm ci
npm run build
rm -rf ../twindb_lightrag_memgraph/webui_dist
mkdir -p ../twindb_lightrag_memgraph/webui_dist
cp -R dist/* ../twindb_lightrag_memgraph/webui_dist/
```

Maintenance rules:

- Keep API client changes synchronized with backend route contracts.
- Update MSW mocks and E2E tests when a route shape changes.
- Do not rely on a dev-only endpoint for packaged WebUI behavior.
- Verify built assets when changing static serving or route fallback behavior.

## 8. Data Model

### Workspace

The LightRAG working directory remains the top-level operational context.
Workspace-level state includes storage namespaces, graph data, document status,
and server configuration.

### Folder

A folder is a user-facing collection inside a workspace. A document may be linked
to multiple folders. Folder membership is represented in graph data, not only as
a scalar property.

The canonical relationship is:

```text
(:DocStatus_{workspace})-[:MEMBER_OF]->(:Folder_{workspace})
```

Some legacy surfaces still dual-write or read a `folder` property for backwards
compatibility. Treat that property as compatibility metadata, not the canonical
membership model.

### Document

Documents have ingestion status, content chunks, vector entries, graph entities,
metadata, and folder membership. Deleting or unlinking a document must respect
whether it is still referenced by another folder.

### Graph

Graph data uses Memgraph entities and relationships. Graph code must preserve
the expectations of LightRAG graph retrieval while also supporting Twin-specific
folder/source projection.

### Vector And KV Data

Vector data is stored in Memgraph-backed structures and must preserve metadata
filtering and similarity behavior expected by LightRAG. KV storage is used by
LightRAG for namespaced runtime data. Keep namespace handling strict; accidental
cross-namespace reads can produce confusing query results.

## 9. Main Request Flows

### Document Upload And Ingestion

High-level flow:

1. Client uploads a document through an upstream-compatible or Twin route.
2. Server resolves workspace and folder context.
3. Auth and quota checks run when enabled.
4. Optional classification/docling steps enrich metadata.
5. Document status is created or updated.
6. LightRAG ingestion writes chunks, vectors, graph entities, and doc status.
7. Folder membership is written through canonical graph relationships.
8. API returns a shape compatible with the caller surface.

Risk points:

- Duplicate upload behavior.
- Folder membership on already-known documents.
- Partial ingestion failure and status rollback.
- Classification failure path.
- Route shape drift between backend and WebUI.

### Query And Retrieval

High-level flow:

1. Client sends a query to a LightRAG-compatible or Twin route.
2. Server resolves query parameters, workspace, folder filters, and auth.
3. Query flows into LightRAG retrieval.
4. Twin code may project sources, folder scope, or graph context.
5. Response is returned in the requested shape, including streaming where
   supported.

Risk points:

- Folder filtering after retrieval instead of before retrieval.
- Missing source IDs or source metadata.
- Streaming route regressions.
- Response shape drift across LightRAG versions.

### Document Delete Or Unlink

High-level flow:

1. Client requests document delete, batch delete, or folder unlink.
2. Server resolves whether the operation is global delete or folder unlink.
3. Membership references are inspected.
4. If the document still belongs to another folder, only the selected membership
   is removed.
5. If no memberships remain and the operation is a delete, document storage,
   vectors, status, and graph artifacts are cleaned up.

Risk points:

- Deleting shared documents too aggressively.
- Leaving stale vector or graph records.
- Returning stale WebUI cache after mutation.

### Graph Read

High-level flow:

1. Client requests graph data.
2. Route resolves scope and query parameters.
3. `graph_reader` fetches and normalizes Memgraph entities/relationships.
4. Response is shaped for upstream compatibility or WebUI graph components.

Risk points:

- Cypher changes that break older Memgraph versions.
- Missing labels/properties expected by WebUI.
- Large graph responses without bounds.

### Auth Flow

High-level flow:

1. Request reaches FastAPI middleware or route dependency.
2. Auth provider is resolved from environment.
3. Disabled/dev/OIDC behavior is applied.
4. Route receives user context or fails.
5. Quota checks may apply for API keys or user scopes.

Risk points:

- Production accidentally using dev auth.
- Static/WebUI fallback routes bypassing auth expectations.
- Mutating routes missing dependency wiring.

## 10. Environment Variables

Use `ENV_VARIABLES.txt` as the full reference. The most important variables for
maintainers are:

```bash
LIGHTRAG_GRAPH_STORAGE=MemgraphStorage
LIGHTRAG_VECTOR_STORAGE=MemgraphVectorDBStorage
LIGHTRAG_KV_STORAGE=MemgraphKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=MemgraphDocStatusStorage
MEMGRAPH_URI=bolt://localhost:7687
MEMGRAPH_USERNAME=memgraph            # default "" = no auth
MEMGRAPH_PASSWORD=memgraph
TWIN_REQUIRE_AUTH=true
TWIN_ENV=production
TWIN_API_BASE_URL=/twin/api           # base path injected into the WebUI
# Auth backend - pick ONE (see ENV_VARIABLES.txt section auth):
LIGHTRAG_API_KEY=...                  # static bearer key, OR
LIGHTRAG_JWT_SECRET=...               # enables local /login (+ LIGHTRAG_JWT_PASSWORD), OR
TWIN_IDP_JWKS_URL=...                 # corporate IdP (strict RBAC)
```

When adding an environment variable:

1. Add the implementation.
2. Add or update tests.
3. Update `ENV_VARIABLES.txt`.
4. Update `README.md` only if it is part of the operator-facing quick path.
5. Update this guide if it changes maintenance behavior.

## 11. Test And CI Doctrine

### Primary CI Matrix

The primary matrix is in `.forgejo/workflows/ci.yml`.

Unit tests run across:

- Python `3.10`, `3.11`, `3.12`, `3.13`
- LightRAG `1.4.9.11`, `1.4.11`, `1.4.12`

Integration tests run across:

- LightRAG `1.4.9.11`, `1.4.11`, `1.4.12`
- Memgraph MAGE `3.9.0`, `3.10.1`

### Local Test Commands

Run all Python tests:

```bash
uv run pytest
```

Run compatibility tests:

```bash
uv run pytest tests/test_upstream_compat.py tests/test_batch_patch.py
```

Run server-related tests:

```bash
uv run pytest tests/test_server*.py tests/test_*routes*.py
```

Run graph tests:

```bash
uv run pytest tests/test_*graph*.py
```

Run WebUI tests:

```bash
cd lightrag_webui_twin
npm test
npm run test:e2e
```

### What To Test By Change Type

Storage adapter change:

- Unit tests for the adapter.
- Relevant LightRAG compatibility tests.
- Integration test with Memgraph when behavior depends on Cypher, indexes, or
  transaction behavior.

Registry or monkey-patch change:

- Upstream compatibility tests.
- Batch patch tests.
- At least one smoke test that imports and registers under the supported
  LightRAG versions.

Server route change:

- Route tests for success, failure, auth, and response shape.
- WebUI client/mock updates if the route is used by the frontend.

Folder behavior change:

- Membership tests.
- Query scoping tests.
- Delete/unlink tests.
- WebUI folder workflow tests if the behavior is visible.

WebUI change:

- Component/unit tests where available.
- MSW mock updates.
- E2E path for user-visible flows.
- Build verification before packaging.

## 12. Common Change Playbooks

### Upgrade LightRAG

1. Add the new LightRAG version to the Forgejo matrix.
2. Run compatibility tests locally for the new version if practical.
3. Inspect failures in registry patches, batch APIs, route imports, and response
   shapes first.
4. Prefer capability checks over version branching.
5. Update `docs/test-doctrine-lightrag-compat.md`.
6. Update README compatibility claims only after CI confirms the matrix.
7. Keep the production baseline explicit if production does not move.

### Upgrade Memgraph

1. Add the new Memgraph/MAGE version to the integration matrix.
2. Run storage and graph integration tests.
3. Check Cypher syntax, index creation, vector search behavior, and transaction
   behavior.
4. Update operations docs if deployment images change.
5. Keep production baseline separate from forward-compat claims.

### Add Or Change A Route

1. Locate the closest existing route and match its auth, error, and response
   conventions.
2. Add backend tests.
3. Update WebUI API client code if used by frontend.
4. Update MSW mocks and E2E tests if user-visible.
5. Document the route only if it becomes part of the public/operator surface.

### Change Folder Semantics

1. Read the folder doctrine in `CLAUDE.md`.
2. Preserve `(:DocStatus_{workspace})-[:MEMBER_OF]->(:Folder_{workspace})` as canonical membership (the document/DocStatus node points at the folder, not the reverse - see `docstatus_impl.py`).
3. Decide whether legacy `folder` property compatibility must be preserved.
4. Test upload, query, unlink, delete, shared document behavior, and WebUI state.
5. Avoid changing storage-level contracts for route-level policy.

### Change WebUI API Usage

1. Update the API client under `lightrag_webui_twin/src/api/`.
2. Update React Query keys and invalidation logic if data freshness changes.
3. Update MSW mocks.
4. Update E2E tests for the visible workflow.
5. Rebuild `webui_dist` before packaging.

### Touch Registry Or Monkey Patches

1. Identify the upstream symbol or behavior being patched.
2. Check whether all supported LightRAG versions expose it the same way.
3. Add a compatibility test that would fail if upstream moves it again.
4. Keep the patch idempotent.
5. Avoid importing heavy server modules during storage-only registration.

### Add A New Storage Feature

1. Confirm whether the feature belongs in storage, server policy, or WebUI.
2. Preserve LightRAG storage return shapes.
3. Add direct storage tests.
4. Add route tests only if the feature is exposed through HTTP.
5. Add integration tests if Memgraph behavior is part of the guarantee.

## 13. Debugging Checklist

### Server Starts But Routes Are Missing

Check:

- Was `register(...)` (with the overlay flags) called before server import/start?
- Is the runtime using `twindb_lightrag_memgraph.lightrag_server` or a plain
  upstream LightRAG server entrypoint?
- Did route import fail silently because an optional dependency is missing?
- Are `TWIN_API_BASE_URL` and WebUI fallback routes configured as expected?

### Storage Falls Back To Non-Memgraph

Check:

- `LIGHTRAG_GRAPH_STORAGE`
- `LIGHTRAG_VECTOR_STORAGE`
- `LIGHTRAG_KV_STORAGE`
- `LIGHTRAG_DOC_STATUS_STORAGE`
- `register()` call order
- package installation extras
- LightRAG version in the active environment

### Memgraph Connection Fails

Check:

- `MEMGRAPH_URI`
- `MEMGRAPH_USERNAME`
- `MEMGRAPH_PASSWORD`
- container/network reachability
- Memgraph MAGE image version
- whether tests expect `localhost` or Docker service DNS

### Query Returns Cross-Folder Sources

Check:

- folder context passed by route
- membership graph relationships
- legacy `folder` property fallback
- source projection code
- WebUI query parameters and cache keys

### WebUI Loads But API Calls Fail

Check:

- static asset build exists in `twindb_lightrag_memgraph/webui_dist`
- API prefix
- auth provider and cookies/tokens
- CORS only if using separate dev servers
- MSW/dev proxy assumptions that are not true in packaged runtime

### Auth Behaves Differently In Production

Check:

- `TWIN_ENV`
- `TWIN_REQUIRE_AUTH`
- the configured auth backend (`LIGHTRAG_API_KEY` / `LIGHTRAG_JWT_SECRET` / `TWIN_IDP_JWKS_URL`)
- route dependency wiring
- dev provider leakage
- API key/quota middleware

## 14. Bus-Factor Hotspots

These areas require extra care because they encode project-specific knowledge.

### `twindb_lightrag_memgraph/patches/registry.py`

Why it matters:

- It is the bridge between Twin and upstream LightRAG.
- Small upstream changes can break imports, symbols, or patch timing.

Before touching:

- Read the relevant tests.
- Check the supported LightRAG matrix.
- Keep storage-only and full-overlay modes separate.

### `twindb_lightrag_memgraph/server/webui/router.py`

Why it matters:

- It coordinates WebUI static serving and API fallback behavior.
- Mistakes can produce routes that work locally but fail in packaged runtime.

Before touching:

- Verify built asset paths.
- Test API routes and SPA fallback separately.
- Check auth expectations for protected surfaces.

### `twindb_lightrag_memgraph/server/graph_reader.py`

Why it matters:

- It normalizes graph data for API and UI consumers.
- It is sensitive to Memgraph version behavior and graph schema drift.

Before touching:

- Run graph tests.
- Check response shape consumers in the WebUI.
- Keep result bounds and filtering explicit.

### `twindb_lightrag_memgraph/docstatus_impl.py`

Why it matters:

- It carries ingestion state, deduplication behavior, and folder/workspace
  metadata.
- Regressions here often show up later as query, delete, or UI inconsistencies.

Before touching:

- Run document status tests.
- Check upload, delete, and shared-document paths.
- Preserve LightRAG status semantics.

### `lightrag_webui_twin/src/app/AppShell.tsx`

Why it matters:

- It coordinates the main product workflow and navigation state.
- It is easy for route/cache changes to create stale UI behavior.

Before touching:

- Read the API query hooks used by the affected view.
- Check E2E coverage.
- Verify responsive layout if changing visible structure.

### `lightrag_webui_twin/src/api/`

Why it matters:

- It is the contract boundary between the backend and UI.
- Backend route changes must be reflected here and in mocks.

Before touching:

- Update generated/manual types if present.
- Update MSW handlers.
- Update query invalidation rules for mutations.

## 15. Code Quality Guidance

Avoid spaghetti by keeping these boundaries:

- Registry code patches upstream integration points only.
- Storage code implements LightRAG storage contracts only.
- Server route code owns HTTP shape, auth, error handling, and route policy.
- Folder membership policy lives near document/folder route behavior, not hidden
  in low-level KV/vector calls.
- WebUI API code owns transport details; components should not assemble raw
  endpoint URLs ad hoc.
- Classification enriches documents; it should not become a required hidden
  dependency for core ingestion.

Avoid boilerplate by following existing patterns:

- Add small helpers only when two or more call sites share real behavior.
- Prefer route-local code for one-off route details.
- Do not add generic abstractions for a single upstream compatibility branch.
- Keep environment handling centralized enough that operators can audit it.
- Delete obsolete compatibility layers after the support window is explicitly
  dropped and tests prove the path is dead.

## 16. Documentation Maintenance

When behavior changes, update docs in this order:

1. Code comments only for non-obvious implementation constraints.
2. Tests, especially compatibility tests, as executable documentation.
3. `ENV_VARIABLES.txt` for configuration changes.
4. `README.md` for operator-facing setup and supported matrix changes.
5. This guide for architecture, ownership, or handover changes.
6. Doctrine docs only when project rules change.

Keep README short enough to onboard an operator. Keep this guide detailed enough
to onboard a maintainer.
