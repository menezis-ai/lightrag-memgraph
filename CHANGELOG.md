# Changelog

## 0.6.0 (unreleased)

- Qualify Memgraph 3.13.2 while retaining legacy storage contracts. CI covers
  3.2.1, 3.8.1, 3.9.0, 3.11.0, 3.12.0 and 3.13.2 with minimum and current
  LightRAG/Neo4j dependency profiles.
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
