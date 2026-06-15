# lightrag-memgraph

Memgraph storage backends (KV, Vector, DocStatus) for [LightRAG](https://github.com/HKUDS/LightRAG) **without modifying LightRAG's source code**.

LightRAG already ships with a built-in `MemgraphStorage` for the **graph** layer. This package fills the remaining 3 slots (KV, Vector, DocStatus) so that an entire LightRAG instance can run on a single Memgraph database.

## Why this exists

LightRAG has a plugin registry (`lightrag.kg`) that maps storage class names to module paths. The registry is hard-coded at import time and does not support third-party packages out of the box. This package works around that by monkey-patching the three registry dicts at runtime via a single `register()` call, before LightRAG is instantiated.

## Requirements

- Python `>= 3.10`
- Memgraph `>= 3.2` with [MAGE](https://memgraph.com/docs/mage) (for `vector_search.search()`)
- `lightrag-hku >= 1.4.9, < 2.0.0`
- `neo4j >= 5.0.0, < 7.0.0` (Bolt driver, compatible with Memgraph)

## Installation

```bash
pip install twindb-lightrag-memgraph
```

For local development:

```bash
pip install -e ".[test]"
```

## Usage

```python
from lightrag import LightRAG
from twindb_lightrag_memgraph import register

# 1. Patch LightRAG's storage registry ONCE, before any LightRAG(...).
register()

# 2. Instantiate LightRAG with the four Memgraph backends.
rag = LightRAG(
    working_dir="./rag_storage",
    kv_storage="MemgraphKVStorage",
    vector_storage="MemgraphVectorDBStorage",
    doc_status_storage="MemgraphDocStatusStorage",
    graph_storage="MemgraphStorage",  # already built into LightRAG
    # ... your llm_model_func, embedding_func, etc.
)
```

`register()` is idempotent — calling it more than once is a no-op.

## Environment

- `MEMGRAPH_URI` — Bolt URL, e.g. `bolt://localhost:7687`. Defaults to `bolt://localhost:7687`.
- `MEMGRAPH_USERNAME` / `MEMGRAPH_PASSWORD` — credentials when auth is enabled.
- `WORKSPACE` (or `MEMGRAPH_WORKSPACE`) — backtick-safe identifier used as the Memgraph label prefix (`KV_{workspace}`, `Vec_{workspace}`, `DocStatus_{workspace}`). Defaults to `base`.
- `MEMGRAPH_WRITE_CONCURRENCY` — max concurrent writes (default `8`).
- `MEMGRAPH_POOL_SIZE` / `MEMGRAPH_READ_POOL_SIZE` — Bolt pool sizing.
- `MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT` — seconds before a connection acquire fails (default `5.0`).

## What the backends do

- **`MemgraphKVStorage`** — label `KV_{workspace}_{namespace}`, dict value serialised to a single `data` JSON string. Batch via `UNWIND` + `MERGE`.
- **`MemgraphVectorDBStorage`** — label `Vec_{workspace}_{namespace}`, vector index `CREATE VECTOR INDEX ... WITH CONFIG {dimension, capacity, metric: "cos"}`. Auto-creates the index on first query if missing.
- **`MemgraphDocStatusStorage`** — label `DocStatus_{workspace}` (no namespace suffix). Indexes on `id`, `status`, `file_path`, `track_id`, `updated_at`, `created_at`. Paginated listing runs count + fetch in parallel.

The graph backend (built into LightRAG) is also patched at registration time with batch read methods (`get_nodes_batch`, `node_degrees_batch`, `get_edges_batch`, …) replacing N sequential queries with single `UNWIND` queries during bulk ingestion.

## Testing

Unit tests (no Memgraph required) — `tests/conftest.py` auto-skips `@pytest.mark.integration` when `MEMGRAPH_URI` is unset:

```bash
pytest tests/ --ignore=tests/test_bench.py -v
```

Integration tests (real Memgraph required):

```bash
docker compose up -d memgraph
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/ --ignore=tests/test_bench.py -v
```

Benchmarks (latency / throughput at 100 / 1k / 10k):

```bash
MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_bench.py -v -s
```

## License

MIT. See `LICENSE`.
