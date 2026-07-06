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
