# Changelog

## 0.6.1 (unreleased)

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
