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
| **LightRAG 1.4.10** | OK | OK | OK |

CI runs this full matrix on every push/PR.

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
| `MEMGRAPH_WORKSPACE` | No | `"base"` | Workspace prefix in node labels for multi-tenancy (e.g., `KV_{workspace}_chunks`) |
| `MEMGRAPH_WRITE_CONCURRENCY` | No | `8` | Max concurrent write operations (upsert/delete/drop). Prevents Bolt pool saturation during bulk uploads. |
| `MEMGRAPH_POOL_SIZE` | No | `50` | Write pool size (max Bolt connections for write operations) |
| `MEMGRAPH_READ_POOL_SIZE` | No | `20` | Read pool size (dedicated read-only Bolt connections, isolated from writes) |
| `MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT` | No | `5.0` | Seconds to wait for a free connection before failing (applies to both pools) |
| `MEMGRAPH_MEMORY_LIMIT` | No | `0` (unlimited) | **Per-database** memory budget. Human-readable sizes: `2GiB`, `500MiB`. `0` = no limit. Uses per-database node/edge estimation (not instance-wide `SHOW STORAGE INFO`). |
| `MEMGRAPH_BUDGET_ENFORCE` | No | `off` | Pre-insert budget gate: `off` (no check), `warn` (log + proceed), `reject` (raise `MemoryBudgetExceeded`). |
| `MEMGRAPH_VECTOR_SCALAR_KIND` | No | `f16` | Vector quantization: `f32` (full precision), `f16` (50% memory savings), `i8` (75% savings). Requires Memgraph >= 3.8. |
| `MEMGRAPH_DELETE_BATCH_SIZE` | No | `10000` | Max nodes per transaction in `drop()` / batched delete. Prevents OOM on large datasets. |
| `MEMGRAPH_TTL_SECONDS` | No | — (disabled) | When set, KV upserts add `:TTL` label + `ttl` property (Unix epoch expiry). Requires Memgraph Enterprise `ENABLE TTL`. |
| `MEMGRAPH_TTL_LABELS` | No | `full_docs,text_chunks` | Comma-separated KV namespaces that get TTL. Only applies when `MEMGRAPH_TTL_SECONDS` is set. |
| `MEMGRAPH_PURGE_FULL_DOCS` | No | `off` | When `on`, purge `full_docs` content after PROCESSED status. Reconstruct from chunks on demand. Saves ~35-45% storage. |

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

With multi-workspace, a second workspace "prod" would create `KV_prod_chunks`, `Vec_prod_entities`, etc. They are fully isolated: `drop()` on one workspace does not affect another.

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
  __init__.py        register() -- monkey-patches lightrag.kg registry
  _pool.py           Shared Bolt driver singleton (event-loop aware)
  _constants.py      Validators, defaults, env var names
  _buffered_graph.py Buffered batch write proxy
  _hooks.py          Post-indexation hooks
  kv_impl.py         MemgraphKVStorage -- key-value pairs as Cypher nodes
  vector_impl.py     MemgraphVectorDBStorage -- vector embeddings + cosine search
  docstatus_impl.py  MemgraphDocStatusStorage -- document processing status tracking

tests/
  conftest.py              Auto-skip integration tests when MEMGRAPH_URI is unset
  test_register.py         Offline: registration logic
  test_kv.py               Integration: KV CRUD
  test_vector.py           Integration: vector CRUD + search
  test_docstatus.py        Integration: doc status CRUD + queries
  test_prod_checklist.py   Integration: dim=1024, multi-workspace, full pipeline
  test_bench.py            Integration: performance benchmarks
```

## Known limitations

- **Three Bolt pools in production:** The built-in `MemgraphStorage` (graph) creates its own driver, separate from our write + read pools. ~120 max connections total (50 write + 20 read + 50 graph). This is by design — each pool is isolated from the others for stability under load.
- **DocStatus upserts are sequential:** Unlike KV and Vector (which use batch `UNWIND`), DocStatus upserts are one-by-one because each entry may be a `DocProcessingStatus` object or a raw dict, requiring per-item serialization logic.
