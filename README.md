# lightrag-memgraph

Memgraph storage backends (KV, Vector, DocStatus) for [LightRAG](https://github.com/HKUDS/LightRAG) **without modifying LightRAG's source code**.

LightRAG already ships with a built-in `MemgraphStorage` for the **graph** layer. This package fills the remaining 3 slots (KV, Vector, DocStatus) so that an entire LightRAG instance can run on a single Memgraph database.

## Why this exists

LightRAG has a plugin registry (`lightrag.kg`) that maps storage class names to module paths. The registry is hard-coded at import time and does not support third-party packages out of the box. This package works around that by monkey-patching the three registry dicts at runtime via a single `register()` call, before LightRAG is instantiated.

## Requirements

- Python `>= 3.10`
- Memgraph `>= 3.2` with [MAGE](https://memgraph.com/docs/mage); **3.14.0 is the default development and CI target**. Legacy native ANN limitations are documented below.
- `lightrag-hku >= 1.4.9, < 1.6.0` (latest qualified stable release: **1.5.7**)
- `neo4j >= 5.0.0, < 7.0.0` (Bolt driver, compatible with Memgraph)

## Installation

```bash
pip install twindb-lightrag-memgraph
```

### LightRAG 1.5 compatibility

Version 0.6.0 supports the 1.5 document-status contracts while retaining 1.4.x
compatibility and existing data. Scheduling uses bounded `(created_at, id)`
keyset pages and a lightweight projection; full records are hydrated in batches.
Creation timestamps are immutable on updates. Strict reads propagate database
and decode failures, so failed reads cannot be mistaken for absent documents.
Native partial updates support pipeline stage metadata without loading chunk lists.

Filename/source resolution distinguishes absent, unique and conflicting primary
documents, ignoring duplicate markers. Content-hash lookup selects the earliest
holder and excludes both the requesting document and markers that point to it.
`content_hash`, parsing/analyzing statuses and arbitrary stage metadata persist;
multi-status listing uses the `status_filters=` keyword alongside the legacy
`status_filter` argument. New methods import 1.5 types lazily to preserve 1.4 installs.

The adapter writes immediately. Upstream deferred-index cleanup remains a no-op
for these backends; graph writes retain upstream keyed locks and recovery anchors.
LightRAG's optional source-conflict enumeration/repair tools are not implemented:
the upstream base methods raise `StorageCapabilityError`. Conflicts remain explicit
and are never resolved by choosing or deleting a primary automatically.

Reviewed the stable release notes from
[1.5.0](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.0) through
[1.5.7](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.7), and the tagged
[storage contracts](https://github.com/HKUDS/LightRAG/blob/v1.5.7/lightrag/base.py).
Parser/chunker improvements, role-specific LLMs, cancellation, UI branding and
the `/workspace` entry are upstream features; this package supplies storage and
does not bundle or qualify those application features.

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

## What the backends do

- **`MemgraphKVStorage`** — label `KV_{workspace}_{namespace}`, dict value serialised to a single `data` JSON string. Batch via `UNWIND` + `MERGE`.
- **`MemgraphVectorDBStorage`** — label `Vec_{workspace}_{namespace}`, vector index `CREATE VECTOR INDEX ... WITH CONFIG {dimension, capacity, metric: "cos"}`. Auto-creates the index on first query if missing.
- **`MemgraphDocStatusStorage`** — label `DocStatus_{workspace}` (no namespace suffix). Indexes on `id`, `status`, `file_path`, `track_id`, `updated_at`, `created_at`. Paginated listing runs count + fetch in parallel.

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

The target release is **0.6.1**; see [CHANGELOG.md](CHANGELOG.md). Registration
keeps the package version visible through LightRAG's `core_version`, with the
suffix `+memgraph-0.6.1`.

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

## Memgraph capabilities and optional retrieval features

Existing applications keep cosine vector retrieval, their labels, identifiers,
metadata and index names. New retrieval features are opt-in. Initialization
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
explicit index rebuild or a new namespace. DDL, authorization, inspection and
connectivity errors propagate; they are not treated as feature fallbacks.

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
from twindb_lightrag_memgraph import inspect_memgraph

capabilities = await inspect_memgraph()
print(capabilities.version)
print(capabilities.has_procedure("text_search.fuzzy_phrase_search"))
```

These features use node indexes because LightRAG stores both entity and
relationship embeddings as nodes. Server-side edge indexes, imports,
parallel execution, FIPS and Enterprise administration remain server-level
features; the adapter does not enable them automatically.

### Release review and feature evidence

Engine release notes from 3.2.1 through 3.14.0 were reviewed on 2026-10-28,
including intermediate minor releases and patch notes. The table records the
adapter's response to changes affecting its storage/retrieval contract.
Versions listed in the CI matrix are tested directly; a feature's introduction
version alone does not imply that exact engine version was run in CI.

| Engine notes | Storage impact and adapter response | Evidence or limit |
|--------------|-------------------------------------|-------------------|
| [3.2.1](https://github.com/memgraph/memgraph/releases/tag/v3.2.1) | Legacy vector storage and retrieval remain available. | Storage contracts run directly; native deletion churn is a known failure and legacy load is excluded. |
| [3.3](https://github.com/memgraph/memgraph/releases/tag/v3.3.0) | Nested property indexes and Bolt scheduling do not change the adapter's flat, label-scoped schema. | Nested-map indexing is not exposed; this engine is not a CI entry. |
| [3.4](https://github.com/memgraph/memgraph/releases/tag/v3.4.0) | Per-index `scalar_kind` quantization is exposed; vector metadata is read by named fields. | `test_advanced_options_fallback_or_strict` and index recovery validate f16 and reject mismatched schemas. Edge vector indexes are not exposed. |
| [3.5](https://github.com/memgraph/memgraph/releases/tag/v3.5.0) | Property-specific text indexes were still experimental; the adapter enables text only from the stable 3.6 API. | Flat metadata serialization remains unchanged. User profiles, multi-role administration, TTL recovery and STRICT_SYNC are not qualified here. |
| [3.6](https://github.com/memgraph/memgraph/releases/tag/v3.6.0) | Stable text indexing, changed index type names, edge text indexes and a native cosine helper. | Node text indexing, property-qualified/regex search, cleanup and hybrid cosine scores are tested. Hybrid computes cosine in Python for legacy portability. Edge text indexing is not exposed. |
| [3.7](https://github.com/memgraph/memgraph/releases/tag/v3.7.0), [3.7.1](https://github.com/memgraph/memgraph/releases/tag/v3.7.1), [3.7.2](https://github.com/memgraph/memgraph/releases/tag/v3.7.2) | LBAC syntax changes and vector/text recovery fixes require deployment review. | The adapter never rewrites grants or durability files. Licensed LBAC and HA recovery need separate qualification. |
| [3.8](https://github.com/memgraph/memgraph/releases/tag/v3.8.0), [3.8.1](https://github.com/memgraph/memgraph/releases/tag/v3.8.1) | Single-store vectors and stricter access to deleted objects. | Live-ID rebinding, deletion hygiene, candidate expansion and explicit vector reads are covered by deletion, reinitialization and storage/load tests. |
| [3.9](https://github.com/memgraph/memgraph/releases/tag/v3.9.0) | Text search changes to GIDs and pinned searchers; vector GC is fixed. Function-generated index configs, global float precision and global parameters are introduced. | Both dependency profiles run text/vector/storage tests. The adapter binds query parameters explicitly and keeps static index configs. Nondefault global float precision and coordinator recovery are not qualified. |
| [3.10](https://github.com/memgraph/memgraph/releases/tag/v3.10.0), [3.10.1](https://github.com/memgraph/memgraph/releases/tag/v3.10.1) | Parameterized configs, DESC indexes, new index syntax, edge single-store vectors and `DENY` semantics. | Literal config maps and label-scoped indexes remain valid. Metrics, edge indexes, remote embedding providers and permission migrations are not adapter APIs. This engine is not a CI entry. |
| [3.11](https://github.com/memgraph/memgraph/releases/tag/v3.11.0) | Text limits move to `config`; term fuzzy/prefix/transposition options and multi-label vector indexes are introduced. | Procedure signatures select argument/output shapes. Real tests cover typo, prefix and transposition semantics. The adapter retains one node label per namespace and exposes no multi-label/wildcard index creation API. |
| [3.12](https://github.com/memgraph/memgraph/releases/tag/v3.12.0) | Adds fuzzy phrase search, property permissions, lightweight edges and database suspension. | Phrase/hybrid retrieval is tested. Permission errors propagate. Property-level authorization, suspended tenants and nondefault lightweight-edge mode are not qualified by this matrix. |
| [3.13](https://github.com/memgraph/memgraph/releases/tag/v3.13.0), [3.13.1](https://github.com/memgraph/memgraph/releases/tag/v3.13.1), [3.13.2](https://github.com/memgraph/memgraph/releases/tag/v3.13.2) | Optional omission of vector properties, permission-filtered search, global/range indexes, Cypher/APOC changes, OpenMetrics default, FIPS and coordinator SSO. | Explicit embedding reads are also tested locally with omission enabled. Queries keep label-scoped indexes and avoid changed APOC/subquery APIs. Global index creation, FIPS, SSO, PBAC and HA are not qualified here. |
| [3.14.0](https://github.com/memgraph/memgraph/releases/tag/v3.14.0) | Changes bound-node pattern and null/comparison semantics; fixes vector recovery, map-write rollback and failed index creation. Adds Cypher subqueries and Python to the FIPS image. | CI retains earlier engines and adds 3.14.0 to all dependency profiles and scale jobs. New real-server cases check that rolled-back map writes and failed index creation preserve stored embeddings and ANN retrieval. The adapter does not use the changed administration queries or collection functions. Disk storage, durability recovery, FIPS and Enterprise administration remain outside this matrix. |

The retrieval cases in
[`tests/test_memgraph_compatibility.py`](tests/test_memgraph_compatibility.py)
cover property-qualified/regex search, limits above 1,000 results, fuzzy term
configuration, fuzzy phrases, hybrid unions, configured index recreation,
deleted-hit filtering and (on 3.14+) vector preservation after rollback or failed
index creation. These real-server checks require `MEMGRAPH_URI`; adding a CI
target does not substitute for a successful qualification run.

### Upgrading an existing Memgraph deployment

The Compose file is a development service. It does **not** define a persistent
data volume. Preserve your actual deployment's existing data mount before
recreating any container; changing an image tag alone does not preserve an
unmounted container data directory.

1. Record the current image/version and configuration. Create a snapshot and
   back up the durability directory, access configuration and custom modules.
2. Restore a copy into an isolated instance and rehearse the upgrade using the
   [official upgrade guide](https://memgraph.com/docs/database-management/upgrades)
   and [version-specific migrations](https://memgraph.com/docs/database-management/upgrades/specific-versions).
3. Stop the old instance gracefully. Start the target image with the existing
   data mount and updated flags. Use matching MAGE packages/images and rebuild
   any custom native query modules against the target server.
4. Start the application with the original workspace/namespace and vector mode.
   Verify stored documents, vectors, graph queries and document status before
   opting into quantization or hybrid retrieval. Keep an untouched pre-upgrade
   backup for rollback; an older binary is not a safe reader of newer durability
   files. A rollback must restore the old image **and** its pre-upgrade data.

Review these changes when crossing the corresponding versions:

| Version | Deployment action |
|---------|-------------------|
| 3.6 | Remove `--experimental-enabled=text-search`; it prevents startup once text search is stable |
| 3.7 | Recreate label/edge access rules following the official LBAC migration; previous rules are not retained |
| 3.8 | Remove deprecated auth/inactivity/index-recovery flags; use `--auth-module-mappings` and `--storage-parallel-schema-recovery` where appropriate. Text indexes cannot be created in analytical mode |
| 3.9 HA | Follow the coordinator migration guide; its coordinator-directory procedure does not apply to standalone data instances |
| 3.10+ | Recheck fine-grained `DENY` permissions and renamed storage-info fields; remove init-file bootstrap flags from coordinators/HA data instances |
| 3.11 | Text-search limits use a config map; update storage/schema metrics consumers, session tracing flags and TLS-reload privileges. Intra-cluster TLS is configured separately from Bolt TLS |
| 3.12 | Replace removed HA timeout flags with coordinator settings |
| 3.13 | Review coordinator SSO, removed `COORDINATOR` grants, APOC JSON aliases, subquery imports and the OpenMetrics default; verify that users can read both indexed labels and embedding properties |
| 3.14 | Rebuild custom C++ modules. Review bound-variable patterns, null/NaN and mixed-type comparisons, temporal ranges and uniqueness constraints; update positional consumers of transaction/user metadata and renamed edge-index metrics. Run profile changes outside explicit transactions. Qualify disk storage and the FIPS image separately before using them with vector/text retrieval |

Sources: [release notes](https://memgraph.com/docs/release-notes) and
[3.14.0 notes](https://github.com/memgraph/memgraph/releases/tag/v3.14.0)
(checked on 2026-10-28);
[vector search](https://memgraph.com/docs/querying/vector-search) and
[text search](https://memgraph.com/docs/querying/text-search) (checked on 2026-10-18).

To select an older development image explicitly:

```bash
MEMGRAPH_IMAGE=memgraph/memgraph-mage:3.8.1 docker compose up -d
```

Run integration tests only against an isolated test database: they create and
delete test namespaces. `MEMGRAPH_URI` enables them; without it, integration
tests are skipped. The CI matrix has 28 jobs: Memgraph **3.2.1, 3.8.1,
3.9.0, 3.11.0, 3.12.0, 3.13.2 and 3.14.0**, each with these pinned dependency profiles:

| Profile | Python | LightRAG | Neo4j driver |
|---------|--------|----------|--------------|
| Minimum | 3.10 | 1.4.9 | 5.x |
| Legacy | 3.12 | 1.4.16 | 6.x |
| First 1.5 | 3.10 | 1.5.0 | 5.x |
| Current | 3.12 | 1.5.7 | 6.x |

Each job checks source
with Ruff, builds the sdist and wheel, installs the wheel, verifies the public import surface,
and tests that installed package. JUnit reports are retained as CI artifacts.

The engine release is **3.12.0**; **3.12.1 is a Memgraph Lab release**, not
a `memgraph/memgraph-mage:3.12.1` engine image.

**Legacy native ANN limitation:** Memgraph 3.2.1 crashed with exit code 139 during
`TestVectorBenchmark.test_query_latency` after repeated index creation/deletion
and bulk inserts. The same sequence also crashes with the unmodified 0.2.5
adapter. Functional compatibility is tested on 3.2.1, but its native vector
module is not qualified for this load scenario. Repeated insert/search/delete
cycles also crashed during the functional suite. Direct Bolt queries without
the adapter returned incorrect empty results in 692 of 1,000 cycles despite a
live matching vector. The deletion-churn regression is therefore an explicit
known failure (`xfail`) on pre-3.4 engines, preventing a native crash from
interrupting the remaining contract tests. The CI runs load checks on 3.8.1,
3.9.0, 3.11.0, 3.12.0, 3.13.2 and 3.14.0; use a current server for production load.

All four profiles test installed public wheels in isolated Python environments.
The minimum profile runs storage contracts; the others run the full regression
suite. Load checks run on the current profile and modern engines.

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

## License

MIT. See `LICENSE`.
