# Embedding scheduling and recovery qualification

The PR #33 review corrections are recorded in commit
`d4aab4dbf7c54102d685220b72eb6c222f300d27`. This qualification compares its
embedding/write path with the previous candidate
`86eda9bae372db10fdd9cc0ed465aec5b325418a` on a disposable Memgraph server.

## Reproduced large-upsert failure

Both runs use LightRAG 1.4.9, Neo4j driver 5.28.8 and Python 3.12.15 against
Memgraph MAGE 3.14.0. The real LightRAG priority scheduler wraps the provider
inside `EmbeddingFunc`, with 8 workers and `llm_timeout=30` (the 1.4.9 default).
Its effective worker execution timeout is 60 seconds. The provider is a local
fixture that sleeps `0.31 * len(texts)` seconds and returns 4-dimensional vectors;
there are no model API calls or answer-quality claims.

Each run upserts 201 chunk records with `embedding_batch_num=10` and
`embedding_func_max_async=8`. `MEMGRAPH_WRITE_CONCURRENCY=1` makes slot contention
observable. Once the provider starts, an independent KV write gets a 1-second
watchdog; another KV write after the upsert verifies that the slot is released.

| Observation | Previous candidate | Review correction |
|---|---:|---:|
| Provider batch sizes | One batch of 201 | Twenty batches of 10, one of 1 |
| Vector upsert | Worker timeout after 60 s | Success in 9.329 s |
| Independent KV probe during embedding | Timed out at 1.001 s | Completed in 0.001775 s |
| Persisted vector records | 0 | 201 |

This reproduces both the timeout and write-slot blockage under the stated
fixture. It does not establish that every provider or every large document
will time out. The corrected timings are observations from this controlled
sleeping fixture, not general ingestion throughput or a production latency SLA.

Server and Python runner each have a 2-CPU quota and 2 GiB limit, on a shared
Linux ARM64 development host. They share a network namespace and use loopback
Bolt. The [environment record](qualification/embedding-2026-10-29/environment.json)
retains source revisions, file hashes and image identities. Raw results are
retained for the [baseline](qualification/embedding-2026-10-29/large-upsert-baseline.json)
and [correction](qualification/embedding-2026-10-29/large-upsert-fixed.json).

## Reproduce the comparison

The [retained script](qualification/embedding-2026-10-29/qualify_large_upsert.py)
is the exact probe used for both runs. Use a disposable Memgraph 3.14.0 database,
Python 3.12 with `lightrag-hku==1.4.9`, `neo4j==5.28.8` and NumPy, and separate
checkouts of the two revisions above. Inside a Linux test container, copy the
script to a writable `/evidence` directory, set `MEMGRAPH_URI` to that server
and `MEMGRAPH_WRITE_CONCURRENCY=1`, then run each case with its source checkout:

```bash
PYTHONPATH=/baseline/src python /evidence/qualify_large_upsert.py baseline
PYTHONPATH=/fixed/src python /evidence/qualify_large_upsert.py fixed
```

The baseline takes about one minute by design. The script allocates random
workspace labels, checks its expected outcome, retains JSON in `/evidence` and
removes only its own data/indexes in `finally`. It selects the adapter source
with `PYTHONPATH`; do not install the unrelated same-named PyPI adapter.

## Error and cancellation boundaries

Regression tests use the real `EmbeddingFunc` plus
`priority_limit_async_func_call`, including an occupied worker and queued
ingestion to verify that a waiting query runs first. They cover bounded batches,
ID ordering, metadata/precomputed inputs, invalid settings, provider errors and
cancellation before any write slot is taken. Independent adapter-validation tests
cover bad counts/shapes, scalar/non-matrix results, NaN, infinities, booleans and
non-numeric coordinates even where newer upstream wrappers reject some earlier.

Recovery tests inject Neo4j `Forbidden`, `AuthError`, `ServiceUnavailable` and
module-related `ClientError`, both at `run()` and deferred `consume()`. They
check the named `RuntimeError`, exact `__cause__`, one retry, embedding reuse and
absence of a duplicate adapter traceback. Hybrid tests verify sibling task
cancellation and completion before an error is returned.

A real-storage naive-query regression on LightRAG 1.5.7 removes the test
namespace's index and injects a recreation denial. The storage call raises the
named/chained error; `aquery_llm()` serializes it as `status="failure"`; the
legacy `aquery()` wrapper returns `None`. This does not qualify Enterprise
authorization decisions or HTTP error mapping. See the
[migration notes](migration-0.6.2.md#retrieval-behavior-changes) for the 1.4 limits.
