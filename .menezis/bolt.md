## 2026-06-24 - [Upload duplicate lookup cache]
Target: find_existing_file_by_file_path (src: lightrag.api.routers.document_routes)
Before: 333.85ms / 11.35MB / 3.0 req/s
After: 0.023ms / 0.00MB / 42,387.5 req/s
Gain: 100% faster, +1,415,009.5% throughput, -11.35MB peak
Method: Cache
Compat: behavior parity locked against upstream via tests/test_upload_duplicate_lookup.py

## 2026-07-06 - [Optimization] Target: [query_data_filters._doc_ids_for_query_data_row] Before: [76.965 ms / 0.005 MB / 13.0 req/s] After: [11.325 ms / 0.026 MB / 88.3 req/s] Gain: [85.3% latency reduced, +579.6% throughput, +0.021 MB peak] Method: [Async/Parallelization]

## 2026-07-06 - [Optimization] Target: [response_sources._filter_sources_by_advanced_filters] Before: [220.289 ms / 0.010 MB / 4.5 req/s] After: [11.671 ms / 0.057 MB / 85.7 req/s] Gain: [94.7% latency reduced, +1787.3% throughput, +0.047 MB peak] Method: [Async/Parallelization]

## 2026-07-06 - [Optimization] Target: [response_sources._build_sources_legacy_fallback] Before: [225.101 ms / 0.026 MB / 4.4 req/s] After: [13.550 ms / 0.057 MB / 73.8 req/s] Gain: [94.0% latency reduced, +1560.2% throughput, +0.031 MB peak] Method: [Async/Parallelization]

## 2026-07-07 - [Optimization] Target: [AddSourceModal client upload readiness gate] Before: [8000.000 ms / 0 MB / 0.125 files/s; 24KiB apparent throughput 3.0 KiB/s] After: [0.000 ms / 0 MB / network-bound] Gain: [100.0% client-gate latency reduced; upload throughput no longer throttled before POST] Method: [Remove artificial timer gate]

## 2026-07-07 - [Optimization] Target: [webui/routes_documents.py::bulk_delete_documents] Before: [2268.771 ms mean / 2274.840 ms p95 / 0.44 req/s / 0.087 MB peak] After: [24.426 ms mean / 26.550 ms p95 / 40.53 req/s / 0.330 MB peak] Gain: [98.9% latency reduced, +9096.9% throughput, +0.243 MB peak] Method: [Async/Parallelization]
Bench: `python .venv/bin/python tests/benchmarks/bulk_delete_documents.py` (ITERATIONS=40, DOC_COUNT=100, LOOKUP_DELAY_SECONDS=0.004)

## 2026-07-07 - [Optimization] Target: [webui/router.py::_list_documents_from_doc_status + _filter_docs_to_active_folder + _doc_matches_query] Before: [50.019 ms mean / 46.658 ms p50 / 62.855 ms p95 / 63.403 ms p99 / 20.0 req/s / 3.044 MB peak] After: [40.593 ms mean / 38.300 ms p50 / 54.569 ms p95 / 55.132 ms p99 / 24.6 req/s / 2.484 MB peak] Gain: [18.8% latency reduced, +23.2% throughput, -0.560 MB peak] Method: [Filter query before membership lookups + narrow post-membership tagging path]
Bench: `python .venv/bin/python tests/benchmarks/list_documents_route.py` (ITERATIONS=60, DOC_COUNT=3000, MEMBERSHIP_LOOKUP_SECONDS=0.001, TAG_DELAY per call: 0.0008s base + 0.00002s/doc)

## 2026-07-07 - [Optimization] Target: [server/webui/routes_documents.py:_membership_locks] Before: [14.381ms mean / 15.827ms p95 / 69.5 req/s / 4.686MB peak in churn workload] After: [18.324ms mean / 19.926ms p95 / 54.6 req/s / 0.675MB peak in churn workload] Gain: [91.7% lock-map reduction, -21.5% throughput in synthetic churn micro-benchmark; memory stabilized] Method: [OrderedDict access-order + bounded cleanup (`_membership_locks` cache cap)]
Bench: `python .venv/bin/python tests/benchmarks/membership_lock_cache.py` (ITERATIONS=10, DOC_COUNT=3000, CLEANUP_EVERY=1024, MAX_LOCKS=2048)

## 2026-07-07 - [Optimization] Target: [vector_impl exact path + cosine projection] Before: [103.240ms mean / 102.854ms p50 / 105.674ms p95 / 119.528ms p99 / 9.7 req/s] After: [79.319ms mean / 78.732ms p50 / 83.301ms p95 / 94.992ms p99 / 12.6 req/s] Gain: [23.2% faster, +30.2% throughput] Method: [Precompute query vector norm once in `query()` and reuse in `_exact_cosine_projection`; prefilter `doc_any` in exact graph/chunk branches]
Bench: `python .venv/bin/python tests/benchmarks/vector_exact_similarity_projection.py` (ITERATIONS=80, EMBEDDING_DIM=384, CANDIDATE_COUNT=6000, TOP_K=25)

## 2026-07-08 - [Parallelize independent metadata reads in graph read paths]
Target:     server/graph_reader.py::read_graph_entities + read_graph_relations
Before:     62.862 ms mean / 65.092 ms p95 / 73.998 ms p99 / 15.9 req/s / 1.523 MB peak (RTT=4ms/query, folder-bound)
After:      57.938 ms mean / 59.857 ms p95 / 69.091 ms p99 / 17.3 req/s / 1.533 MB peak
Gain:       7.8% latency reduced, +8.5% throughput @ 4ms RTT (grows with DB latency: 5.1% @2ms, 10.3% @8ms, 12.3% @15ms)
Method:     Async/Parallelization — asyncio.gather the two independent post-fetch reads (_load_chunk_to_doc_index + _active_member_docs) instead of serializing one Memgraph round-trip
Verified:   Functional parity — graph suites 159 passed/17 skipped identical before(stash)/after; full server suite 954 passed/70 skipped; output entity count identical (400) every bench run
Load-test:  Gain scales with per-query RTT (idle→contended DB: 5.1%→12.3%), largest exactly under load where p95/p99 matter. Back-pressure preserved: reads use the unthrottled get_read_session pool; +1 concurrent read session/request (bounded, no fan-out), write semaphore untouched — no downstream saturation.
Bench:      `.venv/bin/python tests/benchmarks/graph_reader_metadata_gather.py` (RTT_MS env-tunable; ITERATIONS=200, ENTITY_COUNT=400, DOC_COUNT=300)
