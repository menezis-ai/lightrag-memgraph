<a id="lightrag-memgraph"></a>

# lightrag-memgraph: Memgraph storage for LightRAG

Use **Memgraph with LightRAG** for native vector search, key-value storage and
document status tracking alongside the knowledge graph.

`lightrag-memgraph` is an open-source Python integration for graph-based
retrieval-augmented generation (GraphRAG). It adds the three storage backends
needed to complement LightRAG's built-in `MemgraphStorage`, so document text,
LLM response caches, embeddings, processing status and graph data can use one
Memgraph database.

Cosine vector search is the default. Full-text search and hybrid vector/text
retrieval are optional. Your application supplies the LLM and embedding models.

[Quick start](#installation-and-quick-start) ·
[Storage backends](#storage-backends-and-architecture) ·
[Runnable tutorial](docs/tutorial.md) ·
[Measured benchmarks](docs/benchmarks.md) ·
[Supported versions](#supported-lightrag-and-memgraph-versions) ·
[Releases](https://github.com/menezis-ai/lightrag-memgraph/releases) ·
[Report an issue](https://github.com/menezis-ai/lightrag-memgraph/issues)

## Contents

- [Why use Memgraph with LightRAG?](#why-use-memgraph-with-lightrag)
- [Installation and quick start](#installation-and-quick-start)
- [Supported LightRAG and Memgraph versions](#supported-lightrag-and-memgraph-versions)
- [Memgraph connection and workspace configuration](#memgraph-connection-and-workspace-configuration)
- [Storage backends and architecture](#storage-backends-and-architecture)
- [Vector, full-text and hybrid search](#vector-full-text-and-hybrid-search)
- [Frequently asked questions](#frequently-asked-questions)
- [Tests and performance qualification](#tests-and-performance-qualification)
- [Documentation and support](#documentation-and-support)

<a id="why-this-exists"></a>

## Why use Memgraph with LightRAG?

- **One storage engine:** keep graph data, embedding vectors, document records
  and processing status in Memgraph rather than configuring separate storage
  services for each backend.
- **Native retrieval:** use Memgraph cosine vector search and opt into full-text
  search or reciprocal-rank hybrid retrieval when the engine supports them.
- **Python integration:** install the package and call `register()` before
  constructing LightRAG; no LightRAG source fork is required.
- **Version-aware storage:** discover server procedures and validate existing
  index schemas before enabling optional retrieval features.

<a id="installation"></a>

## Installation and quick start

### Install the Python package

```bash
pip install "git+https://github.com/menezis-ai/lightrag-memgraph.git"
```

The distribution name is `lightrag-memgraph`; the Python import is
`lightrag_memgraph`. Install this repository with the GitHub command above:
the [same-named PyPI project](https://pypi.org/project/lightrag-memgraph/)
is a separate integration maintained in Memgraph's AI Toolkit. See
[supported versions](#supported-lightrag-and-memgraph-versions)
for the tested dependency profiles.

### Start Memgraph for local development

If you already run Memgraph with MAGE, set `MEMGRAPH_URI` to its Bolt endpoint.
For a local development instance using this repository's Compose file:

```bash
git clone https://github.com/menezis-ai/lightrag-memgraph.git
cd lightrag-memgraph
docker compose up -d memgraph
export MEMGRAPH_URI=bolt://localhost:7687
```

The Compose service defaults to Memgraph MAGE 3.14.0. It has no persistent data
volume; use a deployment with a persistent data mount for data you need to keep.

<a id="usage"></a>

### Configure the four LightRAG storage backends

Pass your LightRAG-compatible async LLM function and `EmbeddingFunc` instance
into this helper. Set `MEMGRAPH_URI` before constructing LightRAG.

```python
from lightrag import LightRAG
from lightrag_memgraph import register

register()

async def create_rag(llm_model_func, embedding_func):
    rag = LightRAG(
        working_dir="./rag_storage",
        workspace="my_project",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
    )
    await rag.initialize_storages()
    return rag
```

Use the returned instance with LightRAG's ingestion and query APIs and its
normal pipeline lifecycle. `register()` is idempotent and must run before the
instance is created. It installs process-wide patches; see
[storage architecture](#storage-backends-and-architecture).

<a id="requirements"></a>
<a id="lightrag-15-compatibility"></a>
<a id="release-review-and-feature-evidence"></a>
<a id="upgrading-an-existing-memgraph-deployment"></a>

## Supported LightRAG and Memgraph versions

| Component | Compatibility |
|---|---|
| Python | 3.10 or newer |
| LightRAG | `lightrag-hku >= 1.4.9, < 1.6.0`; latest qualified stable release: 1.5.7 |
| Memgraph with MAGE | 3.2+ storage APIs; 3.14.0 is the default development and CI target |
| Neo4j Python Bolt driver | 6.x in the current CI profile; published metadata allows `>= 5.0.0, < 7.0.0`, with minimum-version verification tracked in [issue #24](https://github.com/menezis-ai/lightrag-memgraph/issues/24) |

LightRAG 1.5 support includes bounded document scheduling, immutable creation
timestamps, content-hash lookup and partial status updates, while retaining
1.4.x compatibility. Optional text retrieval and quantization depend on the
server's version and procedure signatures.

Read the [LightRAG storage contracts, Memgraph compatibility matrix and upgrade guide](docs/memgraph-compatibility.md)
for exact CI profiles, engine changes and exclusions. Memgraph 3.2.1 has known
native ANN failures under deletion churn and is excluded from load qualification.

<a id="environment"></a>

## Memgraph connection and workspace configuration

- `MEMGRAPH_URI` — Bolt URL, e.g. `bolt://localhost:7687`. Defaults to `bolt://localhost:7687`.
- `MEMGRAPH_USERNAME` / `MEMGRAPH_PASSWORD` — credentials when auth is enabled.
- Workspace selection applies identically to graph, KV, vectors and document status:
  explicit `workspace=` argument → `global_config["workspace"]` →
  `MEMGRAPH_WORKSPACE` → `WORKSPACE` → `base`. Values must contain only letters,
  digits and underscores. Per-instance arguments take priority over deployment
  defaults, without changing process environment variables during construction.
- `MEMGRAPH_WRITE_CONCURRENCY` — max concurrent writes (default `8`).
- `MEMGRAPH_POOL_SIZE` / `MEMGRAPH_READ_POOL_SIZE` — Bolt pool sizing.
- `MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT` — seconds before a connection acquire fails (default `5.0`).
- `MEMGRAPH_ENCRYPTED` — `true` / `false` for plain Bolt URIs. TLS URI schemes accept `true`; contradictory settings are rejected before connecting.
- `MEMGRAPH_TRUST` — `TRUST_SYSTEM_CA` or `TRUST_ALL`, matching `+s` or `+ssc` when a TLS URI defines trust.

<a id="what-the-backends-do"></a>

## Storage backends and architecture

| LightRAG storage layer | Implementation | Data stored in Memgraph |
|---|---|---|
| Graph | Built-in `MemgraphStorage`, patched by this package | Entity and relationship graph |
| Key-value | `MemgraphKVStorage` | Documents, text chunks and LLM response cache as JSON |
| Vector | `MemgraphVectorDBStorage` | Embeddings and metadata with a native cosine vector index |
| Document status | `MemgraphDocStatusStorage` | Processing status, source path, timestamps and chunk references |

All four backends use distinct, versioned labels encoding both workspace and
namespace with component lengths. For example, KV `(a, b_c)` uses
`LRMG2_KV_w1_a_n3_b_c`, while `(a_b, c)` uses `LRMG2_KV_w3_a_b_n1_c`.
A persistent `LRMG2_REGISTRY` record verifies each physical store's ownership.
Writes use parameterized Cypher and batched `UNWIND` queries.

**Upgrading an existing database requires an explicit migration.** Initialization
refuses ambiguous legacy data/indexes and unclaimed new-format targets before
accessing application data. It never silently adopts, renames or deletes them.
Run `python -m lightrag_memgraph.migration --output inventory.json` for a read-only
inventory, then follow the [0.7 upgrade guide](docs/migration-0.7.0.md).
Direct storage instances must call `await initialize()` before data access.
Entity types beginning with `LRMG2_` (case-insensitive) are reserved; logical
workspace/namespace names with that prefix remain valid.

LightRAG's storage registry is populated at import time. `register()` adds this
package's three backends and patches the built-in graph backend for connection
handling and batch reads. It also patches two internal retrieval functions and
`LightRAG._insert_done` process-wide. The source files of LightRAG are unchanged,
but this integration depends on the supported upstream contracts.

The default environment variables remain supported when no workspace argument
is supplied. Applications that previously used `MEMGRAPH_WORKSPACE` to override
an explicit LightRAG workspace should pass their intended workspace explicitly.
The integration suite runs two real LightRAG instances concurrently with the
same document/chunk identifiers, then deletes one document and verifies all its
chunks, vectors and graph data are gone while the other tenant remains searchable.
Corrupt JSON in a document-status point read raises before deletion can proceed.
The [integrity validation notes](docs/data-integrity.md) describe checks for older
data and the limits of this qualification.

The shared KV/vector/status pools isolate connection acquisition, while each
initialized graph instance owns another driver. With the default sizes and `W`
graph instances, the configured upper bound is approximately **70 + 50 × W
connections per process**, rather than a fixed three-pool total. These settings
do not isolate Memgraph CPU, memory or locks and do not guarantee HTTP latency.
Read-pool sessions explicitly request Neo4j `READ` access mode; HA routing still
requires qualification on an actual Enterprise cluster.

The graph backend (built into LightRAG) is also patched at registration time with batch read methods (`get_nodes_batch`, `node_degrees_batch`, `get_edges_batch`, …) replacing N sequential queries with single `UNWIND` queries during bulk ingestion.

Graph merge writes use LightRAG's original path and complete before its keyed
entity/relationship locks are released. Document-level write buffering is
disabled because deferring persistence can overwrite another document's shared
descriptions and source IDs. The real-server concurrency regressions in
[`tests/test_merge_concurrency.py`](tests/test_merge_concurrency.py) exercise
overlapping document merges for both entities and relationships in every CI
dependency profile.

The latest published release is **0.6.1**. `main` carries the unreleased
**0.7.0 candidate**; see
[CHANGELOG.md](CHANGELOG.md) and the [upgrade notes](docs/migration-0.7.0.md).
Registration keeps the build version visible through LightRAG's `core_version`,
with the suffix `+memgraph-0.7.0` for this candidate.

<a id="memgraph-capabilities-and-optional-retrieval-features"></a>

## Vector, full-text and hybrid search

Cosine vector retrieval remains the default; optional retrieval features are
opt-in. Version 0.7 changes physical labels/index names and requires the explicit
[storage migration](docs/migration-0.7.0.md); logical document IDs and model
configuration can be retained when re-ingesting trusted source manifests. Initialization
reads `SHOW VERSION`, `mg.procedures()` signatures and vector index metadata
from the selected database. It does not change server settings, load modules,
recreate an existing vector index or migrate access permissions.

Metadata-only vector upserts preserve an existing embedding. Dropping a vector
namespace also removes its persisted text index, including after reopening the
store with text search disabled. Index ownership is checked before deleting data.

On single-store servers, vector retrieval filters deleted node references before
reading their properties and expands the candidate search if such references
hide live neighbours. Pre-3.8 servers retain the historical query projection;
pre-3.4 servers also check for an empty namespace before calling native search.

| Memgraph | Integration behavior |
|----------|----------------------|
| 3.2–3.3 | Original cosine retrieval and storage APIs; optional quantization/text retrieval fall back with a warning |
| 3.4+ | Configurable vector quantization (`scalar_kind`) |
| 3.6+ | Stable text indexes, plain-term, property-qualified and regex searches, if the corresponding procedures are loaded |
| 3.8+ | Single-store vectors supported; embeddings are read explicitly when retrieving stored vectors |
| 3.11+ | Native text relevance scores and search configuration, discovered through procedure signatures |
| 3.11+ | Fuzzy term, prefix and transposition settings when text procedures expose a `config` argument |
| 3.12+ | Fuzzy phrase search when `text_search.fuzzy_phrase_search` is available |
| 3.13+ | Explicit embedding reads support omitted vector properties. Enterprise permission filtering is enforced by the server and is not qualified by this Community test matrix. |
| 3.14.0 | Current pinned target; regression coverage for map-write rollback and failed vector-index creation preserving embeddings. Existing storage/retrieval APIs remain in use. |

Pass options in LightRAG's `vector_db_storage_cls_kwargs`:

```python
rag = LightRAG(
    kv_storage="MemgraphKVStorage",
    vector_storage="MemgraphVectorDBStorage",
    doc_status_storage="MemgraphDocStatusStorage",
    graph_storage="MemgraphStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2,
        "memgraph_query_mode": "hybrid",
        "memgraph_text_properties": ["content"],
        "memgraph_vector_index_config": {
            "capacity": 100_000,
            "resize_coefficient": 2,
            "scalar_kind": "f16",
        },
        "memgraph_feature_policy": "strict",
    },
    # ... embedding_func, llm_model_func, etc.
)
```

| Option | Default | Behavior |
|--------|---------|----------|
| `memgraph_query_mode` | `"vector"` | `"hybrid"` combines vector and text candidates in every `query()` call |
| `memgraph_text_search` | `False` | Create a workspace/namespace text index for explicit `query_text()` calls; hybrid mode also enables it |
| `memgraph_text_properties` | `["content"]` | Validated property names indexed for text retrieval |
| `memgraph_vector_index_config` | `{"capacity": 100000}` | New-index capacity, resize coefficient (>1), optional scalar kind; dimension is taken from the embedding function and metric remains cosine |
| `memgraph_feature_policy` | `"fallback"` | Warn and use legacy defaults for unavailable quantization or stable text search; `"strict"` refuses initialization instead |

Configuration is checked before DDL. Existing vector indexes must match the
namespace label, embedding property, embedding dimension, cosine metric and any
explicit scalar kind. A mismatch raises an actionable error without dropping
data. Capacity/resizing options apply when creating an index; they do not resize
an existing index. Changing embedding models or quantization requires an
explicit index rebuild or a new namespace. Initialization errors propagate.
Query-time recovery recreates a missing vector index and retries once, reusing
the query embedding. Failed recreation or a failed retry raises a `RuntimeError`
containing the index name, chained from the original exception in `__cause__`.
At the adapter level, vector and hybrid callers receive the failure rather than
empty hits. On LightRAG 1.5.7, `aquery_llm()` reports `status="failure"`, while
the legacy `aquery()` wrapper returns `None`. Older 1.4 query helpers may catch
backend failures and return empty context. See the
[migration notes](docs/migration-0.7.0.md#retrieval-behavior-changes).

On an initialized `MemgraphVectorDBStorage`:

```python
hits = await store.query_text("memgraph", top_k=10)
hits = await store.query_text(
    "data.content:big bd wo",
    top_k=10,
    mode="fuzzy_phrase_search",
    config={"fuzzy_distance": 1},
)
hits = await store.query_hybrid("memgraph", top_k=10)
```

Text modes are `search_all`, `search`, `regex_search` and `fuzzy_phrase_search`.
`search` and `fuzzy_phrase_search` use Tantivy expressions with `data.` property
prefixes. Configuration accepts `fuzzy_distance` (0–2), `fuzzy_prefix` and
`fuzzy_transpositions`, when the procedure accepts a config map. Regex mode
uses `top_k` for its limit and rejects fuzzy options before connecting. Unsupported
explicit search modes/configuration raise rather than changing search semantics.
Older text procedures without scores return a constant lexical score of 1.
When a legacy procedure exposes a positional integer limit, `top_k` is passed
in that position; newer procedures receive it in their config map. This avoids
the legacy server default silently truncating requests above 1,000 results.
Text results include `id`, `score`, `embedding` and requested metadata fields.

Hybrid retrieval fuses up to `4 * top_k` candidates from each search using
reciprocal rank fusion (`k=60`), deduplicates by ID and returns at most `top_k`.
`score` is the fused rank score and `text_score` is lexical relevance;
`similarity` and `distance` remain cosine similarity and `1 - similarity`.
Lexical-only hits have their cosine similarity calculated from the stored
embedding, and may be below the vector threshold. Hybrid retrieval falls back
to vector retrieval if text was unavailable during initialization under the
default feature policy. No text errors during an active search are hidden.

Discovery is also available independently and can be repeated after an upgrade:

```python
from lightrag_memgraph import inspect_memgraph

capabilities = await inspect_memgraph()
print(capabilities.version)
print(capabilities.has_procedure("text_search.fuzzy_phrase_search"))
```

These features use node indexes because LightRAG stores both entity and
relationship embeddings as nodes. Server-side edge indexes, imports,
parallel execution, FIPS and Enterprise administration remain server-level
features; the adapter does not enable them automatically.

## Frequently asked questions

### Does LightRAG already support Memgraph?

LightRAG includes `MemgraphStorage` for the graph layer. This package adds
Memgraph key-value, vector and document-status backends so all four storage
layers can use the same database.

### Do I need a Neo4j database to use this package?

No. The `neo4j` Python package is the Bolt client driver used to connect to
Memgraph. A separate Neo4j database server is not required.

### Can I use Memgraph as the vector database for LightRAG?

Yes. Select `MemgraphVectorDBStorage` and provide your embedding function.
Cosine vector retrieval is the default; full-text and hybrid search are opt-in
and require the corresponding Memgraph procedures. See
[vector and hybrid search configuration](#vector-full-text-and-hybrid-search).

### Can the GraphRAG pipeline run with local models?

The storage adapter does not select a model provider. Supply LightRAG-compatible
LLM and embedding functions backed by your local or remote models. Storing data
in Memgraph does not determine where model inference runs.

### What should I check before a multi-tenant or production deployment?

Workspace scoping does not enforce authorization. The 0.7 candidate addresses
[label collisions](https://github.com/menezis-ai/lightrag-memgraph/issues/21) with
versioned identities, ownership checks and refusal of ambiguous legacy data.
Read the [migration guide](docs/migration-0.7.0.md) before upgrading and retain
separate database access controls for independent tenants. Embedding-wrapper
compatibility (#22) and vector-index error reporting (#23) retain their fixes.
The Community CI matrix does not qualify Enterprise authorization or HA failover.

<a id="testing"></a>

## Tests and performance qualification

For local development and tests:

```bash
pip install -e ".[test]"
```

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

### Nightly scale qualification

The separate `Memgraph scale qualification` workflow runs daily at **03:17 UTC**
and supports manual dispatch. It also runs when a PR changes its harness,
workflow or dependency configuration, keeping ordinary PRs free of the large
datasets. Scheduled runs use the harness from the default branch and explicitly
check out both `main` and the published `v0.6.0` adapter, each with LightRAG 1.5.7.
The public repository does not contain the formerly referenced `stable/0.5.x` branch.
Each job builds/installs its target adapter as a wheel, then runs the same harness
against a real MAGE service on **3.8.1, 3.9.0, 3.11.0, 3.12.0, 3.13.2 and 3.14.0**.
Memgraph 3.2.1 remains excluded from load qualification for the native ANN issue
documented above. The released adapter is tested without modifying it.

| Workload | Entries | Concurrent calls | Samples | Maximum latency |
|---|---:|---:|---:|---:|
| Document pagination, including a deep page | 10,000 | 8 | 24 | < 2 seconds |
| Document pagination, including a deep page | 100,000 | 8 | 24 | < 10 seconds |
| Native vector search, top 10 | 10,000 | 8 | 24 | < 500 ms |

Seeding and a warm-up request precede measurements. Tests verify result counts
and retrieval of known vectors as well as latency. Any budget violation fails
the job. JUnit, JSON and Markdown reports retain p50/p95/max timings and adapter
commit provenance; GitHub's job summary shows the table even when budgets fail.
No Slack connection or comment-publishing token is required. This is a separate
workflow and does not change branch-protection requirements.

The scale suite is **opt-in**, even when `MEMGRAPH_URI` is configured:

```bash
MEMGRAPH_URI=bolt://localhost:7687 python -m pytest -q \
  tests/test_scale_e2e.py --run-scale --junitxml=reports/scale.xml
python scripts/summarize_scale.py reports/scale.xml \
  --json reports/scale.json --markdown reports/scale.md
```

Each test allocates a unique workspace and removes its own data and indexes.
Use an isolated test database: the 100K workload consumes database resources.

### PDF ingestion responsiveness

The current LightRAG profile also runs the real upstream PDF parser against
synthetic valid, duplicate-dictionary and invalid documents. HTTP probes must
finish while synchronous extraction is held in its worker thread. A negative
control verifies that the detector rejects inline blocking work. This qualifies
the upstream/storage combination; production validation of the separate BNP
pipeline follows [the ingestion validation procedure](docs/ingestion-validation.md).

## Documentation and support

- [LightRAG and Memgraph compatibility, CI matrix and engine upgrades](docs/memgraph-compatibility.md)
- [Storage integrity, workspace migration and deletion validation](docs/data-integrity.md)
- [Storage identity isolation qualification](docs/storage-identity-qualification.md)
- [Embedding scheduling, large-upsert reproduction and error boundaries](docs/embedding-qualification.md)
- [PDF ingestion responsiveness and deployment validation](docs/ingestion-validation.md)
- [Package changelog](CHANGELOG.md) and [GitHub releases](https://github.com/menezis-ai/lightrag-memgraph/releases)
- [Bug reports and feature requests](https://github.com/menezis-ai/lightrag-memgraph/issues)
- [Upstream LightRAG documentation](https://github.com/HKUDS/LightRAG)
- [Memgraph vector search documentation](https://memgraph.com/docs/querying/vector-search)

## License

MIT. See [LICENSE](LICENSE).
