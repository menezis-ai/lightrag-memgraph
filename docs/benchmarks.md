# LightRAG Memgraph storage benchmarks

These are observed adapter storage latencies from a local development run on
2026-10-29, not production HTTP or end-to-end GraphRAG timings. The source and
harness revision is `b881ab071767a3fbb976563ff9d180e3a5dd4256`, which contains
the embedding/error-reporting fixes prepared for 0.6.2. Installed metadata at
measurement time reported 0.6.1; `PYTHONPATH=/repo/src` selected that exact
revision's source. No comparison with another adapter or server is claimed.

These measurements remain a reference for that pinned revision. Subsequent PR
review corrections have separate [embedding qualification evidence](embedding-qualification.md)
and CI results; they do not change the provenance of this recorded run.

## Results

All three existing maximum-latency budgets passed. Each row contains 24 calls
in three groups of eight concurrent requests, after one unmeasured warm-up.

| Operation | Entries | p50 ms | p95 ms | Max ms | Budget ms |
|---|---:|---:|---:|---:|---:|
| Document pagination | 10,000 | 225.07 | 292.88 | 292.89 | 2,000 |
| Document pagination | 100,000 | 3,700.86 | 3,905.26 | 3,916.18 | 10,000 |
| Vector search, top 10 | 10,000 | 8.61 | 15.88 | 19.88 | 500 |

The 100,000-document pagination result is several seconds under this workload;
do not interpret passing its 10-second budget as an interactive UI latency
guarantee. The small sample count and shared development host limit percentile
precision. Host workloads, power policy and CPU model were not controlled or
available for this qualification. These measurements are not an SLA.

## Environment and protocol

- Linux ARM64 containers on OrbStack; Python 3.12.15, LightRAG 1.5.7, Neo4j driver
  6.4.1 and NumPy 2.5.4. The [dependency snapshot](benchmarks/2026-10-29/requirements.txt)
  records all installed versions.
- Server: Memgraph MAGE 3.14.0, digest
  `sha256:2ad342a011fb463c43af821f72ca627eee2b63a39dc663f5928355a9fd01d20a`.
  Server and Python runner each had a 2-CPU quota and 2 GiB memory limit.
- Runner: `python:3.12-slim`, digest
  `sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`.
  Both containers shared a network namespace; measured requests used loopback.
- The database was disposable. Every test allocated a random workspace, seeded
  its dataset outside the timer and cleaned only its own labels and indexes.
- Pagination returns 50 records and a total count. The harness includes pages
  1–7 and the last page, including offset/count work. Vectors are 128-dimensional,
  normalized seeded Gaussian data (`default_rng(42)`); search uses cosine and
  precomputed query vectors. The test verifies that the seeded target is found.
- Ingestion, model inference, embedding computation, index construction,
  cold-start latency, hybrid/text retrieval, HTTP, networking across hosts and
  Enterprise/HA behavior are outside these timings.

The [raw JUnit report](benchmarks/2026-10-29/scale.xml) contains per-test metrics;
the [JSON summary](benchmarks/2026-10-29/scale.json) retains full numeric precision,
budgets and source revisions. The [environment record](benchmarks/2026-10-29/environment.json)
retains image identities, limits and runtime details. The report is one complete
qualifying run; it is not the best of per-operation measurements.

## Reproduce

Use a disposable server rather than an application database. From a Git checkout
of the recorded revision, start Memgraph with the limits above, install the test
dependencies and run:

```bash
pip install -e ".[test,quality]" "lightrag-hku==1.5.7" "neo4j==6.4.1"
export MEMGRAPH_URI=bolt://localhost:7687
export MEMGRAPH_VERSION=3.14.0
export SCALE_ADAPTER_REF="$(git rev-parse --short HEAD)"
export SCALE_ADAPTER_SHA="$(git rev-parse HEAD)"
export GITHUB_SHA="$SCALE_ADAPTER_SHA"
python -m pytest -q tests/test_scale_e2e.py --run-scale --junitxml=reports/scale.xml
python scripts/summarize_scale.py reports/scale.xml \
  --json reports/scale.json --markdown reports/scale.md
```

Record your architecture, image digests, container limits and dependency snapshot
with your report. Different hardware, topology or competing work can change the
result. The [nightly workflow](../.github/workflows/scale-nightly.yml) exercises
multiple engines and a historical adapter; this local report qualifies one
engine/revision only. See the [compatibility guide](memgraph-compatibility.md)
for scope and server exclusions.
