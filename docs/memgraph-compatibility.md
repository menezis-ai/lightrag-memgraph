# LightRAG and Memgraph compatibility

This guide records the storage contracts, engine release review, upgrade steps
and CI coverage for `lightrag-memgraph`. For installation and configuration,
start with the [LightRAG Memgraph README](../README.md).

- [LightRAG 1.5 compatibility](#lightrag-15-compatibility)
- [Memgraph release review and feature evidence](#memgraph-release-review-and-feature-evidence)
- [Upgrade an existing Memgraph deployment](#upgrade-an-existing-memgraph-deployment)

## LightRAG 1.5 compatibility

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

## Memgraph release review and feature evidence

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
[`tests/test_memgraph_compatibility.py`](../tests/test_memgraph_compatibility.py)
cover property-qualified/regex search, limits above 1,000 results, fuzzy term
configuration, fuzzy phrases, hybrid unions, configured index recreation,
deleted-hit filtering and (on 3.14+) vector preservation after rollback or failed
index creation. These real-server checks require `MEMGRAPH_URI`; adding a CI
target does not substitute for a successful qualification run.

## Upgrade an existing Memgraph deployment

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

## Legacy native ANN limitation

 Memgraph 3.2.1 crashed with exit code 139 during
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

Return to [installation, configuration and troubleshooting](../README.md).
