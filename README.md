# Twin KMS

Memgraph-backed runtime for [LightRAG](https://github.com/HKUDS/LightRAG), without
modifying LightRAG source code.

LightRAG already ships a graph backend for Memgraph (`MemgraphStorage`). This
package fills the three other storage slots (`KV`, `Vector`, `DocStatus`) and, in
the full Twin runtime, adds:

- a FastAPI overlay under `/twin/api`;
- an operator WebUI served from the package build;
- folder-based cloisonnement on top of one physical LightRAG/Memgraph workspace;
- tag governance, activity, notifications, graph CRUD, structured query, API keys,
  quota, auth shims, and optional MIP classification gates.

`register()` is the activation point. It patches LightRAG registries and optional
runtime surfaces before the LightRAG app or instance is created.

Maintainer handover: [docs/technical-maintainer-guide.md](docs/technical-maintainer-guide.md)
is the technical guide for architecture, request flows, tests, and bus-factor
reduction.

## Status

Active development version on `main`: `1.1.0`.

Deployed BNP production release: `1.0.0`, frozen on `stable/1.0.x` and
delivered through the `export-1.0.0` snapshot.

Production target:

- LightRAG: `lightrag-hku==1.4.9.11`
- Memgraph MAGE: `3.9.0`
- Forward-compat CI: LightRAG `1.4.11` / `1.4.12`, Memgraph MAGE `3.10.1`

The public GitHub workflow is intentionally reduced. The compatibility gate is
Forgejo CI in `.forgejo/workflows/ci.yml`.

## Compatibility Matrix

| | Memgraph MAGE 3.9.0 | Memgraph MAGE 3.10.1 |
|---|:-:|:-:|
| **LightRAG 1.4.9.11** | OK | OK |
| **LightRAG 1.4.11** | OK | OK |
| **LightRAG 1.4.12** | OK | OK |

Forgejo CI runs:

- unit tests on Python `3.10` / `3.11` / `3.12` / `3.13` across the LightRAG
  matrix;
- integration tests across the LightRAG matrix and Memgraph `3.9.0` / `3.10.1`;
- WebUI lint, typecheck, unit tests, build, MSW Playwright e2e, and real-backend
  Playwright smoke.

`1.4.10` is deliberately excluded because it had intermittent integration-load
failures fixed upstream in `1.4.11+`. Memgraph `latest` is deliberately not used:
CI pins the coverage points.

## Install

For local development with the server and tests:

```bash
pip install -c requirements/constraints-dev.txt "lightrag-hku[api]==1.4.12"
pip install -c requirements/constraints-dev.txt -e ".[server,intelligence,test]"
```

For the reproducible production target:

```bash
pip install -c requirements/constraints-prod.txt \
  -e ".[server,intelligence,tracing]"
```

`requirements/prod-target.txt` pins the BNP production LightRAG baseline.
Refresh the resolved production constraints only as part of an explicit dependency
update:

```bash
uv pip compile pyproject.toml \
  --extra intelligence --extra server --extra tracing \
  --python-version 3.12 \
  --constraints requirements/prod-target.txt \
  -o requirements/constraints-prod.txt
```

The base package depends only on `lightrag-hku` and the Neo4j Bolt driver. The
runtime extras are:

| Extra | Purpose |
|---|---|
| `server` | FastAPI/uvicorn runtime and LightRAG upload dependencies. |
| `intelligence` | OpenAI/Pydantic settings for the Twin intelligence layer. |
| `tracing` | LangSmith tracing integration. |
| `test` | Core pytest/httpx/fastapi test dependencies. |
| `test-server` / `test-intelligence` | Focused test extras for server or intelligence suites. |
| `all` | Runtime extras (`server`, `intelligence`, `tracing`). |

## Quick Start

Storage-only usage:

```python
from twindb_lightrag_memgraph import register

register()  # Call before constructing LightRAG.

from lightrag import LightRAG

rag = LightRAG(
    kv_storage="MemgraphKVStorage",
    vector_storage="MemgraphVectorDBStorage",
    doc_status_storage="MemgraphDocStatusStorage",
    graph_storage="MemgraphStorage",  # LightRAG's built-in graph backend.
)
```

Full Twin overlay usage:

```python
from twindb_lightrag_memgraph import register

register(
    replace_ui=True,
    mount_server=True,
    shim_native_routes=True,
    webui_stores="memgraph",
    security_baseline=True,
)
```

Production-style entrypoints:

```bash
# Reference script.
python twin_main.py

# BNP container entrypoint.
python -m twindb_lightrag_memgraph.lightrag_server

# Import-string launcher.
gunicorn 'twindb_lightrag_memgraph.asgi:get_application()' \
  -k uvicorn.workers.UvicornWorker
```

When using the import-string launcher, enable overlays through env vars:

```bash
TWIN_REPLACE_UI=true
TWIN_MOUNT_SERVER=true
TWIN_SHIM_NATIVE_ROUTES=true
```

## Configuration

The storage backends read Memgraph connection settings directly from environment
variables:

| Variable | Default | Description |
|---|---:|---|
| `MEMGRAPH_URI` | `bolt://localhost:7687` | Bolt endpoint. Use `bolt+s://` or `neo4j+s://` for TLS. |
| `MEMGRAPH_USERNAME` | empty | Memgraph username. |
| `MEMGRAPH_PASSWORD` | empty | Memgraph password. |
| `MEMGRAPH_DATABASE` | `memgraph` | Database name. Enterprise can route by database; Community falls back gracefully. |
| `MEMGRAPH_WORKSPACE` | `base` fallback | Physical LightRAG/Memgraph workspace label prefix. |
| `MEMGRAPH_POOL_SIZE` | `50` | Write pool size. |
| `MEMGRAPH_READ_POOL_SIZE` | `20` | Dedicated read pool size. |
| `MEMGRAPH_WRITE_CONCURRENCY` | `8` | Write semaphore limit. |
| `MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT` | `5.0` | Seconds to wait for a write semaphore slot. |
| `MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT` | `5.0` | Seconds to wait for a Bolt connection. |
| `MEMGRAPH_OPERATION_TIMEOUT` | `60.0` | Maximum seconds for one pooled or graph-storage Bolt operation, including driver/session acquisition and closure. Raise this for unusually large ingestion batches after measuring them. |

Common Twin runtime variables:

| Variable | Purpose |
|---|---|
| `TWIN_DEFAULT_FOLDER` | Default logical folder id. Also used as a fallback workspace value when no LightRAG workspace env is set. |
| `TWIN_DEFAULT_FOLDER_LABEL` | Display label for the default folder. |
| `TWIN_FOLDERS_JSON` | JSON folder catalog. Prefer this for audited provisioning. |
| `TWIN_MAX_FOLDERS` | Runtime folder cap, clamped by the implementation. |
| `TWIN_FOLDERS_RUNTIME_FILE` | Optional JSON file for runtime-created folders. |
| `TWIN_API_BASE_URL` | WebUI base for Twin overlay routes, usually `/twin/api`. |
| `TWIN_LIGHTRAG_BASE_URL` | WebUI base for native/shimmed LightRAG routes. |
| `TWIN_MIP_LABEL_MAP` | Tenant MIP GUID-to-class map. |
| `TWIN_MIP_MAX_CLASSIFICATION` | Maximum accepted class, default `C2`. |

`ENV_VARIABLES.txt` is the full reference.

## Auth

The default matches LightRAG: if no auth backend is configured, access is open and
the server logs a warning.

Configure one or more auth backends before exposing a deployment:

- static bearer token: `LIGHTRAG_API_KEY`;
- local JWT login: `LIGHTRAG_JWT_SECRET` or `TOKEN_SECRET`, plus
  `LIGHTRAG_JWT_PASSWORD` or `AUTH_ACCOUNTS`;
- IdP/JWKS validation: `TWIN_IDP_JWKS_URL` and claim mapping env vars.

Fail-closed startup is opt-in:

```bash
TWIN_REQUIRE_AUTH=true
# or
TWIN_ENV=production
```

In fail-closed mode, startup requires an API key, a JWT secret, or an IdP config.
Default `changeme` local-login passwords are rejected only in that strict mode; in
LightRAG-parity mode they warn but do not crash the process.

Admin folder operations are gated through `require_admin_user`; with an active IdP,
that means the configured `admin:folders` gateway scope/group mapping.
Without an active IdP, local-login JWTs and per-operator `twk_` keys do not carry
authoritative RBAC claims: only the separately managed `LIGHTRAG_API_KEY` may call
administrative routes or select a non-default `X-Twin-Folder`. Twin routes accept
that root key as a bearer token; LightRAG-native routes use `X-API-Key`. This is a
breaking security posture change in 1.1.0 for local-login deployments.

## Folder Model

There are two separate concepts:

- **workspace**: the physical LightRAG/Memgraph namespace used in labels such as
  `KV_base_chunks`, `Vec_base_entities`, `DocStatus_base`;
- **folder**: the operator-facing logical cloisonnement used by the WebUI and Twin
  API.

Folders are not just UI filters. A document is stored once as a
`DocStatus_{workspace}` node and can belong to multiple folders through
`MEMBER_OF`:

```cypher
(:DocStatus_base {id: "doc-1", content_hash: "..."})
  -[:MEMBER_OF]->(:Folder_base {id: "default"})

(:DocStatus_base {id: "doc-1", content_hash: "..."})
  -[:MEMBER_OF]->(:Folder_base {id: "sandbox"})
```

Folder-scoped reads traverse membership. This applies to document lists/counts,
query grounding, KG expansion, WebUI graph `source_docs`, tags/counters, frontend
caches, and duplicate-upload sharing. The legacy `folder` property is still
dual-written as a migration safety net; membership is the authoritative model.

Minimal folder config:

```bash
TWIN_DEFAULT_FOLDER=default
TWIN_DEFAULT_FOLDER_LABEL="Default"
```

Explicit catalog:

```bash
TWIN_DEFAULT_FOLDER=default
TWIN_FOLDERS_JSON='[
  {"id":"default","label":"Default","kind":"primary"},
  {"id":"sandbox","label":"Sandbox","kind":"sandbox"}
]'
```

Folder administration routes:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/twin/api/folders` | List folders. |
| `POST` | `/twin/api/folders` | Create a runtime folder. Admin-gated. |
| `PATCH` | `/twin/api/folders/{folder_id}` | Update runtime folder metadata. |
| `DELETE` | `/twin/api/folders/{folder_id}` | Delete an empty runtime folder. Env-seeded folders are not deleted through the API. |
| `GET` | `/twin/api/documents/{doc_id}/folders` | List memberships for a visible document. Admin-gated. |
| `POST` | `/twin/api/documents/{doc_id}/folders` | Add membership. Admin-gated. |
| `DELETE` | `/twin/api/documents/{doc_id}/folders/{folder_id}` | Remove membership; physical delete only when this was the last folder. Admin-gated. |

## HTTP Surfaces

Native or shimmed LightRAG routes used by the WebUI:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Projected service health. |
| `GET` | `/ready` | Readiness check for app/dependencies. |
| `GET` | `/pipeline_status` | Projected LightRAG pipeline status. |
| `GET` | `/documents` | Folder-scoped document list. |
| `GET` | `/documents/{doc_id}/chunks` | Read chunks for one visible document. |
| `POST` | `/documents/{doc_id}/scan` | Explicit 409: per-document scan is unsupported by LightRAG. |
| `DELETE` | `/documents/{doc_id}` | Delete through the active folder semantics. |
| `GET` | `/openapi` | Curated WebUI API groups. |

Twin overlay routes live under `/twin/api`. Main groups:

| Group | Routes |
|---|---|
| Query | `/query`, `/query/stream`, `/query/data` |
| Documents | `/documents/*`, membership endpoints, metadata, bulk delete, bulk retag, approval/rejection |
| Folders | `/folders` |
| Tags | `/tags`, `/tags/categories`, taxonomy import/template |
| Graph | `/graph/entities`, `/graph/relations`, `/graph/search` |
| Activity/notifications | `/activity`, `/notifications` |
| Settings | `/settings/api-keys` |
| Ops | `/quota`, `/ops/metrics`, `/health` |

For route-level contracts, prefer the tests and generated OpenAPI over copying
large tables into this README.

## How It Works

`register()` patches three LightRAG registry dictionaries:

| Dict | Purpose | Added values |
|---|---|---|
| `STORAGE_IMPLEMENTATIONS` | Valid class names by storage type. | `MemgraphKVStorage`, `MemgraphVectorDBStorage`, `MemgraphDocStatusStorage` |
| `STORAGE_ENV_REQUIREMENTS` | Env vars required by storage classes. | `MEMGRAPH_URI` |
| `STORAGES` | Class-name to module-path mapping. | Absolute `twindb_lightrag_memgraph.*` module paths |

The module paths must be absolute because LightRAG imports them relative to the
`lightrag` package when given relative paths.

Storage layout:

| Storage | Label shape | Notes |
|---|---|---|
| KV | `KV_{workspace}_{namespace}` | JSON-serialized value in `data`. |
| Vector | `Vec_{workspace}_{namespace}` | Embeddings plus native (core) vector index. |
| DocStatus | `DocStatus_{workspace}` | Document processing state, content hash, chunks list, membership edges. |
| Graph | `{workspace}` | Built-in LightRAG Memgraph graph backend, patched for TLS/multi-db and batched reads/writes. |

Connection model:

- one write Bolt pool shared by KV/Vector/DocStatus;
- one read Bolt pool for read endpoints;
- one separate graph pool owned by LightRAG's `MemgraphStorage`;
- event-loop changes are detected and pools are rebuilt;
- writes are throttled by `MEMGRAPH_WRITE_CONCURRENCY`; reads are not gated.

## WebUI Build

The production image builds `lightrag_webui_twin/` and embeds the `dist/` output
into package data under `webui_dist/`. Host deployments do not need Node/Bun at
runtime.

For local frontend work:

```bash
cd lightrag_webui_twin
npm ci
npm run lint
npm run typecheck
npm run test:run
npm run test:e2e
```

The Docker production build strips `mockServiceWorker.js` from the packaged UI.

## Tests

Python:

```bash
# Unit tests; integration tests auto-skip when MEMGRAPH_URI is unset.
pytest tests/ --ignore=tests/test_bench.py -v

# Integration tests.
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/ --ignore=tests/test_bench.py -v

# Benchmarks.
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_bench.py -v -s

# Coverage for SonarQube.
coverage run -m pytest tests/ --ignore=tests/test_bench.py
coverage xml
```

Local Memgraph:

```bash
docker run -d --name memgraph-test -p 7687:7687 memgraph/memgraph-mage:3.9.0
```

Restricted-runtime smoke:

```bash
export TWIN_SMOKE_BASE_URL="https://your-runtime-host"
python tests/smoke/run_smoke.py tests/smoke/runtime-smoke.json
```

The smoke runner is stdlib-only and writes `/tmp/twin-smoke-report.json` plus
`/tmp/twin-smoke-http.log` without logging bearer tokens.

## Classification

The optional MIP classification hook reads Microsoft sensitivity labels before
LightRAG ingestion and can reject files above a configured ceiling.

Supported inputs:

- OOXML (`.docx`, `.xlsx`, `.pptx` and macro-enabled variants) via stdlib;
- legacy OLE (`.doc`, `.xls`, `.ppt`) via optional `olefile`;
- PDF XMP via optional `pikepdf`.

Tenant label map example:

```json
{
  "11111111-2222-3333-4444-555555555555": "C1",
  "22222222-3333-4444-5555-666666666666": {"id": "C2", "name": "C2 Confidentiel"},
  "33333333-4444-5555-6666-777777777777": "C3"
}
```

CLI:

```bash
python scripts/extract_msip.py path/to/report.docx --label-map labels.json
python scripts/extract_msip.py --label-map labels.json --exit-code-on-above C2 docs/*.docx
```

Runtime env:

```bash
TWIN_MIP_LABEL_MAP=/etc/twin/labels.json
TWIN_MIP_MAX_CLASSIFICATION=C2
```

Unknown mapped labels, missing labels, unsupported inputs, and missing parser
dependencies all fail closed while the hook is active. An operator-selected
C1/C2 value may raise a trusted mapped source label, but cannot replace absent
or unparseable source provenance. Raw-text `/insert` is disabled in this mode,
even when a separate `file_path` is supplied; use `/documents/upload` so the
classified binary and the ingested content are the same source.

## Debugging

Memgraph connectivity:

```bash
python -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687'); d.verify_connectivity(); print('OK'); d.close()"
```

List labels:

```bash
mgconsole --host localhost --port 7687 \
  -c "CALL schema.node_type_properties() YIELD nodeLabels RETURN DISTINCT nodeLabels"
```

Vector search does **not** require MAGE — `CREATE VECTOR INDEX` and
`vector_search.search` are core Memgraph features (stable since 3.0.0). The
plain `memgraph/memgraph:3.9.0` image is sufficient; `memgraph/memgraph-mage`
also works (superset). This package calls no MAGE-only procedure — KV,
DocStatus and the graph backend are plain Cypher.

If LightRAG reports `Unknown storage implementation`, `register()` ran too late.
Call it before constructing `LightRAG` or before importing the LightRAG server
factory.

## File Map

```text
src/twindb_lightrag_memgraph/
  __init__.py                 Root re-export shim.
  patches/registry.py         register() and LightRAG runtime patches.
  _pool.py                    Shared read/write Bolt pools.
  _constants.py               Env names, validators, folder/workspace context.
  _buffered_graph.py          Buffered graph write proxy.
  kv_impl.py                  MemgraphKVStorage.
  vector_impl.py              MemgraphVectorDBStorage.
  docstatus_impl.py           MemgraphDocStatusStorage + folder membership.
  classification.py           MIP sensitivity-label extractor.
  _classification_hook.py     Pre-ingestion classification gate.
  server/                     FastAPI overlay, auth, folders, graph, query, shims.
  intelligence/               TwinRAG intelligence layer.

lightrag_webui_twin/          React operator WebUI.
tests/                        Python unit/integration suites.
tests/smoke/                  Stdlib deployed-runtime smoke runner.
docs/operations/              Install/runbook material.
docs/test-doctrine-*.md       Compatibility and graph test doctrine.
```

## Known Limitations

- The LightRAG graph backend owns a separate Bolt driver, so production uses three
  pools by design: write, read, graph.
- DocStatus upserts remain per-entry because they accept both LightRAG
  `DocProcessingStatus` objects and raw dict payloads.
- Folder membership is authoritative, but the legacy `folder` property is still
  dual-written as a rollback/migration safety net.
- Hard-isolated folders with separate physical graph labels are not implemented;
  current folders are relational cloisonnement over one physical workspace.
