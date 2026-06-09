# lightrag-memgraph

Memgraph storage backends (KV, Vector, DocStatus) for [LightRAG](https://github.com/HKUDS/LightRAG) **without modifying LightRAG's source code**.

LightRAG already ships with a built-in `MemgraphStorage` for the **graph** layer. This package fills the remaining 3 slots (KV, Vector, DocStatus) so that an entire LightRAG instance can run on a single Memgraph database.

## Why this exists

LightRAG has a plugin registry (`lightrag.kg`) that maps storage class names to module paths. The registry is hardcoded at import time and does not support third-party packages out of the box. This package works around that by monkey-patching the three registry dicts at runtime via a single `register()` call, before LightRAG is instantiated.

## Requirements

- Python >= 3.10
- Memgraph >= 3.2 with [MAGE](https://memgraph.com/docs/mage) (for `vector_search.search()`)
- `lightrag-hku >= 1.4.9, < 2.0.0`
- `neo4j >= 5.0.0, < 7.0.0` (Bolt driver, compatible with Memgraph)

### Tested compatibility matrix

| | Memgraph MAGE 3.7.2 | Memgraph MAGE 3.8.0 | Memgraph MAGE latest |
|---|:-:|:-:|:-:|
| **LightRAG 1.4.9** | OK | OK | OK |
| **LightRAG 1.4.9.11** | OK | OK | OK |
| **LightRAG 1.4.11** | OK | OK | OK |
| **LightRAG 1.4.12** | OK | OK | OK |

CI runs this full matrix on every push/PR. LightRAG `1.4.10` was dropped from the matrix due to a transient timing regression that produces non-deterministic test failures under integration load; fixed upstream in `1.4.11+`. Re-enablement tracked in [issue #6](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/6).

## Installation

```bash
pip install -e .

# With test dependencies
pip install -e ".[test]"
```

## Quick start

```python
from twindb_lightrag_memgraph import register

register()  # Call ONCE before instantiating LightRAG

from lightrag import LightRAG

rag = LightRAG(
    kv_storage="MemgraphKVStorage",
    vector_storage="MemgraphVectorDBStorage",
    doc_status_storage="MemgraphDocStatusStorage",
    graph_storage="MemgraphStorage",  # Built-in, not from this package
    # ...
)
```

## Configuration

All backends read their connection settings from environment variables (`os.environ`). Compatible with HashiCorp Vault agent injection, Kubernetes secrets, and systemd `EnvironmentFile`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEMGRAPH_URI` | Yes | `bolt://localhost:7687` | Bolt endpoint. `bolt+s://` for TLS (direct). `neo4j+s://` for TLS with routing protocol (Enterprise cluster). |
| `MEMGRAPH_USERNAME` | No | `""` | Auth username (empty = no auth) |
| `MEMGRAPH_PASSWORD` | No | `""` | Auth password |
| `MEMGRAPH_DATABASE` | No | `"memgraph"` | Database name passed to the Bolt driver. Enterprise supports multi-database. |
| `MEMGRAPH_WORKSPACE` | No | `"base"` | LightRAG/Memgraph workspace prefix in node labels for storage isolation (e.g., `KV_{workspace}_chunks`). This is not the Twin user-facing Folder. |
| `MEMGRAPH_WRITE_CONCURRENCY` | No | `10` | Max concurrent write operations (upsert/delete/drop). Prevents Bolt pool saturation during bulk uploads. |
| `MEMGRAPH_POOL_SIZE` | No | `50` | Write pool size (max Bolt connections for write operations) |
| `MEMGRAPH_READ_POOL_SIZE` | No | `20` | Read pool size (dedicated read-only Bolt connections, isolated from writes) |
| `MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT` | No | `5.0` | Seconds to wait for a free connection before failing (applies to both pools) |
| `TWIN_DEFAULT_FOLDER` | No | `default` | Default Twin Folder id. Used as a fallback for `MEMGRAPH_WORKSPACE` resolution when no LightRAG workspace is set. |
| `TWIN_DEFAULT_FOLDER_LABEL` | No | `Default folder` | Display label for the default Folder when no explicit Folder catalog is provided. |
| `TWIN_FOLDERS_JSON` | No | (empty) | JSON array defining the available Twin Folders. See [Twin Folders](#twin-folders). |
| `TWIN_MAX_FOLDERS` | No | `5` | Maximum number of configured runtime Folders. The implementation clamps this to 1..5. |
| `TWIN_FOLDERS_RUNTIME_FILE` | No | (empty) | Optional JSON file used to persist runtime-created Folders across process restarts. |
| `TWIN_API_BASE_URL` | No | `/twin/api` | Runtime API base injected into the React WebUI for Twin overlay routes. |
| `TWIN_LIGHTRAG_BASE_URL` | No | `""` | Runtime API base injected into the React WebUI for native LightRAG routes (`/documents`, `/health`, `/pipeline_status`, etc.). |
| `TWIN_MIP_LABEL_MAP` | No | (empty) | Path to a JSON file mapping Microsoft Information Protection label GUIDs to tenant classes (e.g. BNP `C1`/`C2`/`C3`/`C4`). See [Classification](#classification-microsoft-information-protection). |
| `TWIN_MIP_MAX_CLASSIFICATION` | No | `C2` | Maximum allowed class for ingested documents. Files outranking this are refused at the pre-insert hook. Unknown classes are treated as above the ceiling (fail-closed). |

## Twin Folders

The product concept is **Folder**. The preferred public contract is now:
`TWIN_DEFAULT_FOLDER`, `TWIN_FOLDERS_JSON`, `X-Twin-Folder`, runtime config
fields `defaultFolderId` / `folders` / `maxFolders`, and `/twin/api/folders`.

Legacy `space` names (`TWIN_DEFAULT_SPACE`, `TWIN_SPACES_JSON`,
`TWIN_MAX_SPACES`, `TWIN_SPACES_RUNTIME_FILE`, `X-Twin-Space`,
`/twin/api/spaces`) remain accepted for compatibility with existing BNP
deployments and older clients. They are aliases only; new code, docs, UI copy
and operator language should say Folder.

There are two different isolation concepts:

- **LightRAG workspace**: storage-level namespace used in Memgraph labels such
  as `KV_base_chunks`, `Vec_base_entities`, and `DocStatus_base`. It is resolved
  from `MEMGRAPH_WORKSPACE`, then `WORKSPACE`, then `TWIN_DEFAULT_FOLDER`, then
  `TWIN_DEFAULT_SPACE`, then `base`.
- **Twin Folder**: operator-facing subdivision exposed in the WebUI switcher
  and Twin overlay API. It scopes WebUI data, document metadata, tags, activity,
  notifications, and runtime catalog entries.

Minimal single-Folder deployment:

```bash
TWIN_DEFAULT_FOLDER=cib
TWIN_DEFAULT_FOLDER_LABEL="CIB Knowledge Folder"
```

Explicit multi-Folder catalog:

```bash
TWIN_DEFAULT_FOLDER=cib
TWIN_MAX_FOLDERS=5
TWIN_FOLDERS_JSON='[
  {"id":"cib","label":"CIB Knowledge Folder","kind":"primary","description":"Production KB"},
  {"id":"sandbox","label":"Sandbox Folder","kind":"sandbox","description":"Operator test area"}
]'
```

Folders are created by deployment configuration first. Runtime creation through
`POST /twin/api/folders` is available for admin users, and persists only when
`TWIN_FOLDERS_RUNTIME_FILE` points to a writable JSON file. In restricted BNP
deployments, prefer `TWIN_FOLDERS_JSON` for audited, reproducible provisioning.

Fresh runtime initialization is clean by default when the Twin overlay is
mounted with Memgraph stores: documents, tags, activity, notifications, and
graph projections start empty unless real LightRAG/Memgraph data or operator
mutations exist. Demo fixtures are still available only through explicit
`webui_stores="seed"` / in-memory settings for local demos and tests.

The browser sends the active Folder on every API call using
`X-Twin-Folder`. During the compatibility window it also sends `X-Twin-Space`
and `X-Twin-Workspace`. Backend code reads `X-Twin-Folder` first, then falls
back to the legacy headers.

Folder administration uses `/twin/api/folders`:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/twin/api/folders` | List configured Folders. |
| `POST` | `/twin/api/folders` | Create a runtime Folder. Requires admin scope. |
| `PATCH` | `/twin/api/folders/{folder_id}` | Update a runtime Folder label/kind/description. |
| `DELETE` | `/twin/api/folders/{folder_id}` | Delete an empty runtime Folder. Env-seeded Folders cannot be deleted through the API. |

`/twin/api/spaces` and `GET /twin/api/workspaces` are kept for older UI
compatibility and return the same catalog in the WebUI's historical shape.

## Twin WebUI and API routes

Calling `register(replace_ui=True, mount_server=True, shim_native_routes=True)`
extends a host LightRAG FastAPI app without patching LightRAG source files:

- replaces the bundled WebUI with the React Twin WebUI;
- mounts the Twin overlay under `/twin/api`;
- adds native-route shims so the React port can call stable document routes;
- captures the host `LightRAG` instance so Twin query endpoints use the same KB.

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

Core native/shimmed routes used by the WebUI:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Projected service health. |
| `GET` | `/pipeline_status` | LightRAG pipeline status in the React contract shape. |
| `GET` | `/documents` | List documents from DocStatus. |
| `GET` | `/documents/{doc_id}/chunks` | Read chunks for one document. |
| `POST` | `/documents/{doc_id}/scan` | Per-document scan compatibility endpoint. Currently an ack/no-op over LightRAG's global scan model. |
| `DELETE` | `/documents/{doc_id}` | Delete one document by id through LightRAG deletion. |
| `GET` | `/openapi` | Curated API groups for the WebUI tab. |

Native LightRAG routes remain available unless the host deployment disables
them. They are useful for integrators that want the upstream contract:

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/query` | Native non-streaming LightRAG query. |
| `POST` | `/query/stream` | Native LightRAG NDJSON stream. |
| `POST` | `/query/data` | Native structured retrieval data from `LightRAG.aquery_data()`. |
| `POST` | `/documents/upload` | Native multipart upload. |
| `POST` | `/documents/text` | Insert one text document. |
| `POST` | `/documents/texts` | Insert multiple text documents. |
| `POST` | `/documents/scan` | Native global input-directory scan. |
| `POST` | `/documents/reprocess_failed` | Requeue failed documents. |

### Restricted runtime smoke test

For BNP-style restricted containers, a stdlib-only smoke runner is available in
`tests/smoke`. It validates that `/webui`, local JWT authentication, native
LightRAG routes, and Twin overlay routes are wired to the expected service.
This is intended for developers, auditors, release engineers, and technical
reviewers who need a reproducible runtime check without browser automation or
external Python dependencies.

```bash
export TWIN_SMOKE_BASE_URL="https://your-runtime-host"
export ARTIFACTORY_USERNAME="..."
export ARTIFACTORY_PASSWORD="..."
python tests/smoke/run_smoke.py tests/smoke/bnp-runtime-smoke.json
```

The JSON manifest is the audit contract: it lists each expected route,
authentication transition, status code, cookie property, and response shape.
The runner only executes that contract against the deployed service.

The runner writes `/tmp/twin-smoke-report.json` and `/tmp/twin-smoke-http.log`
without logging credentials or bearer tokens. See `tests/smoke/README.md` for
the manifest contract, report format, and limitations. This smoke test proves
runtime routing and authentication wiring; it is not a replacement for unit
tests, Playwright WebUI flows, or end-to-end ingestion/query validation.

Twin overlay routes:

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/twin/api/query` | Structured non-streaming retrieval response: `{response, sources}`. |
| `POST` | `/twin/api/query/stream` | NDJSON streaming response. Emits token events and a final sources event. |
| `POST` | `/twin/api/query/data` | Structured retrieval data wrapper around `LightRAG.aquery_data()`. Supports the Twin `tag_filter` contract on returned data. |
| `GET` | `/twin/api/documents/{doc_id}/metadata` | Folder, tags, review, classification, and raw metadata for one document. |
| `POST` | `/twin/api/documents/bulk-delete` | Bulk document deletion with activity logging. |
| `POST` | `/twin/api/documents/_bulk-retag` | Add/remove tags on documents. |
| `POST` | `/twin/api/documents/{doc_id}/approve` | Approve a pending document in the governance flow. |
| `POST` | `/twin/api/documents/{doc_id}/reject` | Reject a pending document in the governance flow. |
| `GET` | `/twin/api/tags` | List governed tags. |
| `POST` | `/twin/api/tags` | Request/create a tag. |
| `PATCH` | `/twin/api/tags/{name}` | Edit tag definition/category/aliases. |
| `POST` | `/twin/api/tags/{name}/approve` | Approve a requested tag. |
| `POST` | `/twin/api/tags/{name}/reject` | Reject a requested tag. |
| `POST` | `/twin/api/tags/{name}/deprecate` | Deprecate a tag. |
| `POST` | `/twin/api/tags/{name}/synonyms` | Replace tag synonyms. |
| `DELETE` | `/twin/api/tags/{name}` | Delete or migrate a tag. |
| `GET` | `/twin/api/tags/categories` | List tag taxonomy categories. |
| `GET` | `/twin/api/tags/categories/template` | Download the canonical category template. |
| `POST` | `/twin/api/tags/categories/_import` | Import category taxonomy JSON. |
| `GET` | `/twin/api/graph/entities` | List projected knowledge-graph entities. |
| `POST` | `/twin/api/graph/entities` | Create a manual graph entity. |
| `PATCH` | `/twin/api/graph/entities/{entity_id}` | Edit a graph entity projection. |
| `DELETE` | `/twin/api/graph/entities/{entity_id}` | Delete a graph entity and its edges. |
| `GET` | `/twin/api/graph/relations` | List projected knowledge-graph relations. |
| `POST` | `/twin/api/graph/relations` | Create a manual graph relation. |
| `PATCH` | `/twin/api/graph/relations/{rel_id}` | Edit a graph relation projection. |
| `DELETE` | `/twin/api/graph/relations/{rel_id}` | Delete a graph relation. |
| `GET` | `/twin/api/activity` | Audit/activity feed. |
| `GET` | `/twin/api/notifications` | Operator notifications. |
| `POST` | `/twin/api/notifications/read-all` | Mark notifications as read. |
| `DELETE` | `/twin/api/notifications` | Clear notifications. |
| `GET` | `/twin/api/thesaurus` | Tag autocomplete/thesaurus entries. |
| `GET` | `/twin/api/health` | Twin overlay component health. |
| `POST` | `/twin/api/auth/logout` | Logout ack and future cookie clearing hook. |

## How it works

### 1. Registration (`__init__.py`)

`register()` patches three dicts in `lightrag.kg`:

| Dict | What it does | What we add |
|------|-------------|-------------|
| `STORAGE_IMPLEMENTATIONS` | Lists valid class names per storage type | `MemgraphKVStorage`, `MemgraphVectorDBStorage`, `MemgraphDocStatusStorage` |
| `STORAGE_ENV_REQUIREMENTS` | Env vars that must exist for each backend | `MEMGRAPH_URI` for all three |
| `STORAGES` | Maps class name to importable module path | Absolute paths like `twindb_lightrag_memgraph.kv_impl` |

The module paths **must be absolute** (not relative like `lightrag.storage.xxx`) because LightRAG's `lazy_external_import` calls `importlib.import_module(path, package="lightrag")` -- relative paths would resolve against the `lightrag` package and fail.

The function is idempotent (guarded by a `_registered` flag). Safe to call multiple times.

### 2. Dual connection pool (`_pool.py`)

Two independent `AsyncGraphDatabase` drivers (Bolt protocol) via module-level singletons: one **write pool** (`get_session()`) and one **read pool** (`get_read_session()`). All three backends share these pools.

**Why dual pools?** Under heavy indexing load (bulk file uploads), write operations can saturate the write pool's connections. A dedicated read pool guarantees that read endpoints (like `get_docs_paginated`) never compete with writes for connections, eliminating 502 errors during bulk ingestion.

**Event loop detection:** Both pools detect event loop changes by comparing `id(asyncio.get_running_loop())` to the loop ID at driver creation time. If the loop changed, the old driver is closed and a new one is created.

**Thread safety:** A shared `threading.Lock` with double-check locking protects concurrent driver creation.

**Connection acquire timeout:** Both pools apply `connection_acquisition_timeout` (default 5s, configurable via `MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT`). Sessions that cannot acquire a free connection within this timeout raise an error instead of hanging indefinitely.

**Protocol-aware database routing:** The pool detects the URI scheme and adapts how database selection is handled:

| Scheme | Protocol | `database=` in `session()` | `USE DATABASE` in session |
|--------|----------|:-:|:-:|
| `bolt://`, `bolt+s://`, `bolt+ssc://` | Direct | No (stripped) | Yes |
| `neo4j://`, `neo4j+s://`, `neo4j+ssc://` | Routing | Yes (native) | No |

On **Memgraph Community** (no Enterprise license), `USE DATABASE` fails — the pool detects this on the first attempt and silently skips it for all subsequent sessions.

**Write throttle:** `acquire_write_slot()` is an `asynccontextmanager` backed by an `asyncio.Semaphore` (default 10 slots, configurable via `MEMGRAPH_WRITE_CONCURRENCY`). All write operations (`upsert`, `delete`, `drop`) are wrapped with it. Read operations use `get_read_session()` from the dedicated read pool and are **never** gated.

**Note:** The built-in `MemgraphStorage` (graph backend from LightRAG itself) creates its own driver independently via `_SafeDriverWrapper`. In production, this means 3 Bolt connection pools total (write + read + graph). This is by design — the graph pool handles the heavy merge/query workload and benefits from its own isolation.

### 3. KV storage (`kv_impl.py`)

Stores arbitrary key-value data as Cypher nodes.

**Data model:**
```
(:KV_base_chunks {id: "chunk-001", data: '{"content": "...", "doc_id": "..."}', __created_at: "...", __updated_at: "..."})
```

- Label: `KV_{workspace}_{namespace}` (e.g., `KV_base_chunks`, `KV_base_full_documents`)
- The value dict is serialized to a single JSON string in the `data` property
- Index on `(id)` created at `initialize()`

**Key methods:**

| Method | Cypher pattern | Notes |
|--------|---------------|-------|
| `upsert(data)` | `UNWIND + MERGE` | Batch insert/update in a single query |
| `get_by_id(id)` | `MATCH ... RETURN n.data` | Deserializes JSON |
| `get_by_ids(ids)` | `UNWIND + OPTIONAL MATCH` | Preserves order, returns `None` for missing keys |
| `filter_keys(keys)` | `OPTIONAL MATCH ... WHERE n IS NULL` | Returns keys that do NOT exist |
| `delete(ids)` | `UNWIND + DETACH DELETE` | |
| `drop()` | `MATCH (n) DETACH DELETE n` | Drops all nodes for this namespace |

### 4. Vector storage (`vector_impl.py`)

Stores embeddings with metadata, supports cosine similarity search via Memgraph MAGE.

**Data model:**
```
(:Vec_base_entities {id: "e-paris", embedding: [0.12, 0.34, ...], entity_name: "Paris", content: "..."})
```

- Label: `Vec_{workspace}_{namespace}`
- Vector index: `CREATE VECTOR INDEX vec_{workspace}_{namespace} ON :Vec_...(embedding) WITH CONFIG {"dimension": N, "capacity": 100000, "metric": "cos"}`
- Both a label index on `(id)` and a vector index on `(embedding)` are created at `initialize()`

**Key methods:**

| Method | Cypher pattern | Notes |
|--------|---------------|-------|
| `upsert(data)` | `UNWIND + MERGE + SET embedding` | Batch. If no embedding provided, computes it from `content` via `embedding_func` |
| `query(query, top_k)` | `CALL vector_search.search(...)` | Filters by `cosine_better_than_threshold` (default 0.2). Returns `{id, similarity, distance, ...meta_fields}` |
| `delete_entity(name)` | `WHERE n.entity_name = $name` | Deletes all vectors for an entity |
| `delete_entity_relation(name)` | `WHERE n.src_id = $name OR n.tgt_id = $name` | Deletes relation vectors involving an entity |
| `get_vectors_by_ids(ids)` | `RETURN n.embedding` | Returns raw float lists |

**`cosine_better_than_threshold`:** Read from `global_config["vector_db_storage_cls_kwargs"]["cosine_better_than_threshold"]`. Defaults to `0.2` if not specified. Results below this similarity threshold are filtered out.

### 5. Doc status storage (`docstatus_impl.py`)

Tracks document processing state through the LightRAG pipeline.

**Data model:**
```
(:DocStatus_base {id: "doc1", status: "processed", content_summary: "...", content_length: 1234, file_path: "/data/doc.pdf", chunks_count: 42, track_id: "batch-001", metadata: '{"source": "upload"}', created_at: "...", updated_at: "..."})
```

- Label: `DocStatus_{workspace}` (no namespace suffix -- doc status is workspace-global)
- Indexes on `(id)`, `(status)`, `(file_path)`, `(track_id)`
- Complex fields (`metadata`, `chunks_list`) are JSON-serialized strings
- Unknown status values in the DB gracefully fall back to `PENDING` with a warning log

**Key methods:**

| Method | Cypher pattern | Notes |
|--------|---------------|-------|
| `upsert(data)` | `MERGE + SET` | Accepts both `DocProcessingStatus` objects and raw dicts |
| `get_status_counts()` | `RETURN n.status, count(n)` | Aggregate counts per status |
| `get_docs_by_status(status)` | `MATCH ... {status: $status}` | Returns `{doc_id: DocProcessingStatus}` |
| `get_docs_by_track_id(track_id)` | `MATCH ... {track_id: $track_id}` | Batch tracking |
| `get_docs_paginated(...)` | `ORDER BY ... SKIP ... LIMIT` | Pagination with sort (whitelist-protected against injection) |
| `get_doc_by_file_path(path)` | `MATCH ... {file_path: $path}` | Lookup by file path |

### 6. Buffered batch writes

During `merge_nodes_and_edges`, a `_BufferedGraphProxy` wraps the graph storage and intercepts `upsert_node`/`upsert_edge` calls, accumulating them in memory. On `flush()`, nodes are written first (UNWIND + MERGE), then edges (UNWIND + MATCH + MERGE). This reduces 130+ individual Bolt round-trips per document to 2-3 batch queries.

The proxy supports read-your-own-writes: `get_node`/`has_edge`/`get_edge` check the buffer before delegating to the real graph.

### 7. Batch read methods

The package patches `MemgraphGraphStorage` with batch methods that replace N sequential queries with single UNWIND queries:

| Method | Replaces | Description |
|--------|----------|-------------|
| `get_nodes_batch(ids)` | N × `get_node()` | Single UNWIND query for all node lookups |
| `node_degrees_batch(ids)` | N × `node_degree()` | Single UNWIND query for all degree counts |
| `get_edges_batch(pairs)` | N × `get_edge()` | Single UNWIND query for all edge lookups |
| `edge_degrees_batch(pairs)` | Derived from `node_degrees_batch` | Sum of endpoint degrees |
| `get_nodes_edges_batch(ids)` | N × `get_node_edges()` | Single UNWIND query |
| `get_nodes_with_degrees_batch(ids)` | Fused: nodes + degrees in 1 query | Eliminates a `gather()` |
| `get_edges_with_degrees_batch(pairs)` | Fused: edges + degrees in 1 session | 2 queries, 1 session |

## Node labels in Memgraph

When you connect to Memgraph with `mgconsole` or Memgraph Lab, you'll see labels like:

```
:KV_base_chunks              <- KV storage, workspace "base", namespace "chunks"
:KV_base_full_documents      <- KV storage, namespace "full_documents"
:Vec_base_entities           <- Vector storage, namespace "entities"
:Vec_base_relationships      <- Vector storage, namespace "relationships"
:DocStatus_base              <- Doc status, workspace "base"
```

With multiple LightRAG workspaces, a second workspace "prod" would create `KV_prod_chunks`, `Vec_prod_entities`, etc. They are fully isolated: `drop()` on one workspace does not affect another.

## Tests

```bash
# Unit tests only (no Memgraph needed)
pytest tests/test_register.py -v

# All integration tests (requires running Memgraph)
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/ --ignore=tests/test_bench.py -v

# Single test
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_kv.py::TestMemgraphKVStorage::test_upsert_and_get -v

# Benchmarks (latency, throughput, scaling at 100/1K/10K items)
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_bench.py -v -s
```

**Quick Memgraph for testing (Docker):**

```bash
docker run -d --name memgraph-test -p 7687:7687 memgraph/memgraph-mage:latest
```

Integration tests use the `@pytest.mark.integration` marker and are **auto-skipped** when `MEMGRAPH_URI` is not set (`conftest.py`).

## Debugging

### "Connection refused" or timeout on Memgraph

```bash
# Check Memgraph is running and reachable
docker logs memgraph-test 2>&1 | tail -5

# Test Bolt connectivity directly
python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687')
d.verify_connectivity()
print('OK')
d.close()
"
```

### Inspecting data in Memgraph

```bash
# Install mgconsole or use Memgraph Lab (http://localhost:3000 if Lab is running)

# List all labels
mgconsole --host localhost --port 7687 -c "CALL schema.node_type_properties() YIELD nodeLabels RETURN DISTINCT nodeLabels"

# Count entries per label
mgconsole --host localhost --port 7687 -c "MATCH (n:KV_base_chunks) RETURN count(n)"

# View a specific KV entry
mgconsole --host localhost --port 7687 -c "MATCH (n:KV_base_chunks {id: 'some-chunk-id'}) RETURN n.data"

# List vector indexes
mgconsole --host localhost --port 7687 -c "SHOW INDEX INFO"

# Manual vector search
mgconsole --host localhost --port 7687 -c "CALL vector_search.search('vec_base_entities', 5, [0.1, 0.2, ...]) YIELD node, similarity RETURN node.id, similarity"
```

### "Vector index not found" errors

Vector search requires Memgraph MAGE. The standard `memgraph/memgraph` Docker image does **not** include it. Use `memgraph/memgraph-mage`.

```bash
# Wrong -- no MAGE
docker run memgraph/memgraph

# Correct
docker run memgraph/memgraph-mage
```

### Backend not found by LightRAG

If LightRAG raises `ValueError: Unknown storage implementation: MemgraphKVStorage`, make sure `register()` was called **before** instantiating `LightRAG`:

```python
# Wrong
rag = LightRAG(kv_storage="MemgraphKVStorage", ...)  # Fails: not registered yet

# Correct
from twindb_lightrag_memgraph import register
register()  # Must be first
rag = LightRAG(kv_storage="MemgraphKVStorage", ...)
```

### Empty query results / low similarity scores

- Check `cosine_better_than_threshold`. Default is `0.2`. Set to `0.0` for debugging to see all results:
  ```python
  LightRAG(
      vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.0},
      ...
  )
  ```
- Verify embedding dimension matches the vector index dimension. A mismatch will silently return 0 results.

### Stale driver after event loop change

If you see `RuntimeError: Event loop is closed` in async code, the driver may be bound to a dead loop. The pool handles this automatically, but if you're managing event loops manually:

```python
from twindb_lightrag_memgraph._pool import close_driver
await close_driver()  # Force driver reset; next get_driver() creates a new one
```

## File map

```
src/twindb_lightrag_memgraph/
  __init__.py               register() -- monkey-patches lightrag.kg registry
  _pool.py                  Shared Bolt driver singleton (event-loop aware)
  _constants.py             Validators, defaults, env var names
  _buffered_graph.py        Buffered batch write proxy
  _hooks.py                 Post-indexation hooks
  kv_impl.py                MemgraphKVStorage -- key-value pairs as Cypher nodes
  vector_impl.py            MemgraphVectorDBStorage -- vector embeddings + cosine search
  docstatus_impl.py         MemgraphDocStatusStorage -- document processing status tracking
  classification.py         MSIP / sensitivity-label extractor (OOXML / OLE / PDF)
  _classification_hook.py   Pre-insert hook that classifies + gates documents

scripts/
  extract_msip.py    CLI: probe a file for its Microsoft sensitivity label

tests/
  conftest.py                  Auto-skip integration tests when MEMGRAPH_URI is unset
  test_register.py             Offline: registration logic
  test_kv.py                   Integration: KV CRUD
  test_vector.py               Integration: vector CRUD + search
  test_docstatus.py            Integration: doc status CRUD + queries
  test_prod_checklist.py       Integration: dim=1024, multi-workspace, full pipeline
  test_bench.py                Integration: performance benchmarks
  test_classification.py       Offline: MSIP extractor (OOXML / optional-dep paths)
  test_classification_hook.py  Offline: pre-insert hook gating + audit emission
```

## Classification (Microsoft Information Protection)

Optional pre-insert hook that reads the sensitivity label Microsoft 365 embeds in Office documents and refuses ingestion of files above a configured ceiling. Designed for regulated tenants (BNP, healthcare, defense) where letting a `C3 Strictement Confidentiel` document slip into a public retrieval index is a compliance incident.

### What it reads

- **OOXML** (`.docx` `.xlsx` `.pptx` and their `.docm`/`.xlsm`/`.pptm` macro-enabled siblings) — `MSIP_Label_<GUID>_*` properties in `docProps/custom.xml`. Pure stdlib, zero extra dependency.
- **Legacy OLE binary** (`.doc` `.xls` `.ppt`) — same `MSIP_Label_*` keys in the custom properties stream. Requires `olefile`.
- **PDF** — `MSIP_Label_*` blocks in the XMP metadata. Requires `pikepdf`.

Missing optional deps degrade gracefully — the affected formats return `ClassificationResult(class_id=None, reason='<pkg>-missing')` instead of raising.

### Tenant label map

MIP label GUIDs are tenant-specific (the GUID for "C2 Confidentiel" in the BNP tenant is different from another organization's). The mapping lives in a JSON file pointed to by `TWIN_MIP_LABEL_MAP`:

```json
{
  "11111111-2222-3333-4444-555555555555": "C1",
  "22222222-3333-4444-5555-666666666666": {"id": "C2", "name": "C2 Confidentiel"},
  "33333333-4444-5555-6666-777777777777": "C3",
  "44444444-5555-6666-7777-888888888888": "C4"
}
```

Long form (`{id, name}`) overrides the raw label name with a tenant-curated display string. Short form (just the id) is fine when the document already carries the right name.

### CLI

```bash
# Probe a single file
python scripts/extract_msip.py path/to/report.docx --label-map labels.json

# Fail the CI when any file outranks C2
python scripts/extract_msip.py --label-map labels.json --exit-code-on-above C2 docs/*.docx
```

### Programmatic use

```python
from twindb_lightrag_memgraph._classification_hook import install_classification_hook

# Build the hook once at server startup
hook = install_classification_hook(
    label_map_path="/etc/twin/labels.json",
    ceiling="C2",
    audit_emit=my_audit_callback,  # optional (kind, payload) callback
)

# Per document, before LightRAG.insert():
try:
    classification = hook(file_path)        # dict, ready for DocStatus.metadata
except ClassificationRejection as exc:
    log.warning("refused: %s", exc)
    # Skip the insert + surface the rejection in the operator UI
```

The returned `classification` dict is intended to be persisted on `DocStatus.metadata['classification']` — the WebUI's `DocDetailPanel` already gates the chunks tab and the "View raw" notice on `metadata.classification.class_id > 'C2'`.

### Behavior summary

| File state | `class_id` | `reason` | Default ceiling action |
|---|---|---|---|
| Labeled, GUID in map | `"C1".."C4"` | `None` | allow / reject per `is_above(class_id, ceiling)` |
| Labeled, GUID not in map | `"UNKNOWN"` | `"unknown-label-guid"` | reject (fail-closed) |
| No `docProps/custom.xml` | `None` | `"no-custom-props"` | reject (fail-closed) |
| `custom.xml` without MSIP property | `None` | `"no-msip-label"` | reject (fail-closed) |
| Malformed file | `None` | `"parse-error: <kind>"` | reject (fail-closed) |
| Unsupported extension | `None` | `"unsupported-extension: <ext>"` | reject (fail-closed) |
| Missing optional dep | `None` | `"olefile-missing"` / `"pikepdf-missing"` | reject (fail-closed); install the dep to enable detection |

Set `TWIN_MIP_MAX_CLASSIFICATION` to relax the ceiling per workspace. Per-workspace overrides (`install_classification_hook(ceiling="C3")`) take precedence over the env var.

## Known limitations

- **Three Bolt pools in production:** The built-in `MemgraphStorage` (graph) creates its own driver, separate from our write + read pools. ~120 max connections total (50 write + 20 read + 50 graph). This is by design — each pool is isolated from the others for stability under load.
- **DocStatus upserts are sequential:** Unlike KV and Vector (which use batch `UNWIND`), DocStatus upserts are one-by-one because each entry may be a `DocProcessingStatus` object or a raw dict, requiring per-item serialization logic.
