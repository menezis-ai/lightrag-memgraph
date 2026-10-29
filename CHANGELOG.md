# Changelog

## 0.7.0 (2026-10-30)

0.7.0 supersedes the unpublished 0.6.2 preparation. Physical storage
identities change; follow the [upgrade guide](docs/migration-0.7.0.md).

- Encode storage kind, workspace and namespace with versioned length framing.
  Separate graph/KV/vector/status labels and vector/text index names, including
  underscores, reserved-looking workspaces and document-status namespaces (#21).
- Verify ownership in a persistent, uniquely keyed registry before data access.
  Refuse ambiguous legacy data/indexes or unclaimed new-format targets without
  changing them. Retain ownership records when dropping a store. Reserve the
  `LRMG2_` entity-type prefix to prevent cross-storage co-labeling.
- Require successful initialization for data I/O. Invalidate verification after
  finalization, pool reset or identity/connection changes. Reject connection
  configuration changes with active pools rather than verifying a different DB
  through a cached driver. This does not fix concurrent event-loop ownership.
- Provide a read-only JSON inventory (`python -m lightrag_memgraph.migration`)
  and an explicit backup/re-ingestion migration path; no automatic in-place
  adoption or destructive migration. Qualify identical-ID isolation, refusal
  without legacy data/index mutation and concurrent startup on real Memgraph.
- Call LightRAG's `EmbeddingFunc` wrapper for both query and document vectors,
  preserving dimension/token injection, concurrency limiting and asymmetric
  query/document context when supported upstream. Schedule query embeddings
  ahead of waiting ingestion. Reject invalid counts, dimensions and non-finite
  coordinates before database writes with diagnostic values (#22).
- Respect `embedding_batch_num` (default 10) and bound outstanding preparation
  batches by `embedding_func_max_async` (default 8). Prepare embeddings before
  acquiring the shared write slot; failed/cancelled preparation never writes
  partially prepared data (#25, #26).
- Propagate vector-index recovery failures instead of returning an empty result.
  Vector and hybrid callers can distinguish a failed search from no matches;
  a successful recovery still retries once and reuses the query embedding.
  Failed recreation/retry raises `RuntimeError` with the index name and chained
  cause, without a duplicate traceback. Cancel and await sibling hybrid searches
  on error or caller cancellation. On LightRAG 1.5.7, `aquery_llm()` exposes a
  failure status/message while legacy `aquery()` returns `None`; 1.4 query
  helpers may still swallow backend errors (#23).
- Use `lightrag-memgraph` as the distribution and project name and
  `lightrag_memgraph` as the Python namespace. Update imports, LightRAG storage
  registry paths, logging, tests and CI to the new namespace. The import rename
  itself preserves stored IDs; 0.7 physical identities require migration. Update
  application imports and restart running processes when upgrading. Install this repository from GitHub;
  the same-named PyPI distribution is a separate project.
- Add a runnable fixture/live-provider lifecycle tutorial, requalify the open
  isolation finding on Memgraph 3.14.0, and publish measured scale results with
  raw JUnit/JSON evidence and environment details.

## 0.6.1 (2026-10-28)

- Target Memgraph 3.14.0 in the development image and add it to the compatibility
  and nightly scale matrices while retaining previous engines. Add real-server
  regressions for embedding/ANN preservation after rolled-back map writes and
  failed vector-index creation. Document the Cypher and deployment changes;
  disk storage and FIPS remain outside the adapter's qualification scope.
- Compare nightly scale results against the published v0.6.0 adapter. The
  previous stable/0.5.x reference is not available in the public repository.

## 0.6.0 (2026-10-18)

- Honor the explicit LightRAG workspace consistently across all four backends,
  ahead of environment defaults, without mutating global environment state.
  Qualify concurrent ingestion and complete deletion across two tenants sharing
  document/chunk identifiers. Reject unreadable status JSON before deletion.
- Consume graph initialization/probe results and propagate deferred connection
  failures. Preserve upstream immediate graph writes and guard legacy entity-type
  label interpolation against backticks/NUL. Read sessions explicitly use `READ`.
- Add nightly/manual scale qualification for main and stable/0.5.x across modern
  Memgraph engines: 10K/100K document pagination and 10K-vector search, concurrent
  calls, enforced latency budgets and retained JUnit/JSON/Markdown evidence.
  Large datasets are opt-in during local/regular compatibility tests.
- Qualify the real LightRAG 1.5.7 PDF parser's HTTP responsiveness, including
  duplicate-dictionary input, parse failure and a watchdog-bounded blocking
  negative control. Document the separate production validation for ingestion
  502s; synthetic tests do not assert that the BNP incident is resolved.
- Qualify Memgraph 3.13.2 while retaining legacy storage contracts. CI covers
  3.2.1, 3.8.1, 3.9.0, 3.11.0, 3.12.0 and 3.13.2 with LightRAG 1.4.9,
  1.4.16, 1.5.0 and 1.5.7 / Neo4j 5.x and 6.x dependency profiles.
- Support LightRAG 1.5.7 and keep the 1.4.9 minimum. Implement bounded
  scheduling pages, strict batch hydration/point reads, source conflict
  resolution, deterministic content-hash deduplication, multi-status filters
  and native partial updates. Preserve immutable creation times, content hashes,
  stage metadata and legacy records. Optional source-conflict enumeration and
  repair tools remain unsupported and raise the upstream capability error.
- Discover server capabilities and procedure signatures before using vector
  quantization, text search, fuzzy search and hybrid retrieval. Preserve text
  query limits across the positional and config-map APIs.
- Preserve embeddings during metadata-only updates and when the server omits
  vector properties. Filter deleted search hits and validate index ownership
  before cleanup.
- Preserve LightRAG's keyed-lock guarantees for shared entities and relations.
  Remove document-level write buffering, which could overwrite descriptions and
  source IDs from concurrent document merges. Graph writes complete under the
  upstream locks; batch read optimizations remain available.
- Preserve existing namespaces, index names, TLS/database routing and the
  `register()` version suffix displayed by the LightRAG WebUI
  (`+memgraph-0.6.0`).

Memgraph 3.2.1 has a documented native ANN deletion-churn limitation. Enterprise
permissions, HA, FIPS and SSO are outside this release's qualification scope.
See the README's release review and feature evidence for detailed coverage.
