# Storage identity qualification

Code/tests revision: [`88666dab63bf5b43f2307719cb0d576a7038a078`](https://github.com/menezis-ai/lightrag-memgraph/commit/88666dab63bf5b43f2307719cb0d576a7038a078).
This fixes the two reproduced collision families from issue #21; it also scopes
graph/document-status namespaces and checks existing-format ownership before I/O.
The release metadata/docs follow in a separate commit.

## Local evidence, 2026-10-29

The installed 0.7.0 candidate wheel was tested against disposable Memgraph MAGE
3.14.0 (2 CPUs, 2 GiB). The runner used Python 3.12.15, with LightRAG 1.5.7 /
Neo4j 6.4.1 for the current profile and LightRAG 1.4.9 / Neo4j 5.28.8 for minimum.
Exact versions, image digests and result counts are in the
[qualification manifest](qualification/storage-identity-2026-10-29/qualification.json).
CI separately covers Python 3.10 and the supported engine/dependency matrix.

| Run | Passed | Skipped | Failures/errors |
|---|---:|---:|---:|
| Complete current suite, including load tests | 515 | 3 opt-in scale cases | 0 |
| Final identity/migration suite | 59 | 0 | 0 |
| Minimum-dependency storage contract | 298 | 13 upstream capability gates | 0 |
| Opt-in 10K/100K status pagination + 10K-vector search | 3 | 0 | 0 |

Five migration/lifecycle cases were added after the complete current run; the
final 59-case identity run includes them. Counts from these overlapping runs
must not be added together. The [identity JUnit](qualification/storage-identity-2026-10-29/identity-current.xml)
and [scale JUnit](qualification/storage-identity-2026-10-29/scale-current.xml)
retain the individual cases. No production database, credentials or documents
were used. Existing benchmark documentation remains pinned to its older source.

## Guarantees exercised

- Same-ID reads, overwrites, deletes and `drop()` remain isolated across the
  underscore boundary and separate namespaces for KV, vector, status and graph.
- Graph workspaces equal to legacy/new KV/vector/status labels or the registry
  name cannot address those physical stores. Registered graph entity-type aliases
  are recognized; types starting with the reserved prefix reject before I/O.
- Legacy records and even empty property/vector/text indexes block initialization
  without mutation. Existing legacy ANN stays searchable. Unknown v2 targets and
  malformed registry entries cannot be silently adopted.
- Explicit fresh-workspace re-ingestion succeeds. The inventory leaves server
  records/schema unchanged. Finalization/pool reset requires re-verification.
- Twenty-four concurrent initializers obtain one ownership record. Dropping data
  retains that record, allowing a verified empty store to reopen.

Unit fixtures which mock all database I/O explicitly model a successfully
initialized store via `initialized_mock`; the identity/integration tests exercise
the actual guards and bootstrap without that shortcut.

## Limits

The format is a breaking persistence change. Follow the [migration guide](migration-0.7.0.md):
stop legacy writers, back up, resolve ownership from source manifests and re-ingest
into a verified fresh scope. Historical overwritten values cannot be recovered
from concatenated labels. Arbitrary Cypher clients/administrators can still alter
labels or ownership metadata; workspace scoping does not enforce authorization.
Live Enterprise permissions, HA and concurrent event-loop pool ownership (#27)
are outside this qualification. Local latency budgets do not establish production
HTTP latency or end-to-end answer quality.
