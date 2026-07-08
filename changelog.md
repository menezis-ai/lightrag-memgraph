# Changelog — twindb-lightrag-memgraph

> **v1.0.0 is the version officially deployed in BNP production since
> 2026-07-03** (GitHub delivery `export-1.0.0` @ `7132f6f`, built from `main`
> @ `89a446b`, frozen on the protected branch `stable/1.0.x`). The `main`
> development line is now **1.1.0**.

## v1.1.0 — 2026-07-08 (BNP delivery cut)

### TwinRAG quality and compliance hardening
- Tag governance: rejected and deleted tags are now excluded from catalog
  display, catalog export, and uniqueness checks, preventing obsolete entries
  from blocking approved tag creation.
- Retrieval parameters: numeric inputs now support standard value replacement
  without leading-zero artifacts, including decimal source-score values.
- Knowledge Graph extraction: expanded entity extraction guidance and graph
  type mapping for technology entities.
- Upload classification policy: operator-selected sensitivity is restricted to
  C1/C2, bulk upload assignment follows the same rule, and C3/C4 operator
  headers are rejected before ingestion.

### WebUI operator experience
- The active folder now persists across a page refresh instead of resetting to
  the default.
- Document search matches records whose `doc_id` is resolved from the projected
  fallback, so filtering no longer hides documents the backend actually returned.
- Bulk retag now applies to every selected document across paginated lists, not
  only the rows on the current page.
- Document upload is hardened: the readiness gate and busy-state feedback keep
  the operator from double-submitting or acting on a not-yet-ready upload.

### Performance and operational hardening
- Query source lookups are parallelized so the sources panel is assembled with
  fewer sequential round-trips, and earlier query-optimization regressions on
  that path are corrected.
- Exact vector search:
  - Prefilter `doc_any` directly when only explicit document-id filtering is
    needed in `_build_exact_chunks_search` and `_build_exact_graph_search` to
    avoid extra predicate passes and set-style filtering work.
  - Cache query vector norm once in Python and pass it as `query_norm` instead
    of recomputing `reduce(...)` per row in Cypher `_exact_cosine_projection`.
  - Benchmark result on 6000 candidates, 384 dimensions, 80 iterations:
    throughput +30.2%; p50 latency ~102.85ms to ~78.73ms; p99 latency
    ~119.53ms to ~94.99ms.
- Membership lock cache containment:
  - `_membership_locks` moved to a bounded `OrderedDict` cache with move-to-end
    on access and unlock-based eviction of stale entries to avoid unbounded
    memory growth.
  - Benchmark result on synthetic churn, 80 concurrent workers, 5 seconds:
    cache size reduced from 30001 to 2497 entries. Mean latency increased from
    14.38ms to 18.32ms and throughput decreased by 21.5%; production-like
    workload validation is required before broader rollout.

## v1.0.0 — 2026-06-11

First complete release: the package now ships the full Twin runtime on top of
an unmodified LightRAG — storage backends, server overlay and operator WebUI —
installable with a single `pip install` and activated with a single
`register()` call (the BNP container entrypoint is `python -m twindb_lightrag_memgraph.lightrag_server`; `twin_main.py` is the reference launcher).

### Storage (Memgraph)
- `MemgraphKVStorage`, `MemgraphVectorDBStorage`, `MemgraphDocStatusStorage`
  fill the three registry slots LightRAG leaves open; the graph layer uses
  LightRAG's built-in `MemgraphStorage`. One Memgraph database runs the whole
  instance.
- Dual connection pool (throttled writes / unthrottled reads), buffered batch
  writes during ingestion (~130 round-trips per document reduced to 2-3),
  batch read patches on the graph backend, auto-creation of the vector index
  with one retry on cold start.
- Workspace-scoped node labels (`KV_{ws}`, `Vec_{ws}`, `DocStatus_{ws}`)
  enable several knowledge bases on a shared Memgraph.

### Server overlay (FastAPI)
- `/twin/api/*`: folders (env-provisioned + admin CRUD, `X-Twin-Folder`
  scoping), tags governance (request/approve/reject/deprecate/migrate with
  audit trail), activity ledger, notifications, knowledge-graph read +
  entity/relation CRUD persisted to Memgraph, structured `POST /query` with
  sources.
- Native-route shims keep LightRAG's `/documents`, `/health`,
  `/pipeline_status`, auth routes behavior-identical for existing clients.
- Authentication: static API key, local JWT login, or corporate IdP via JWKS
  (`TWIN_IDP_JWKS_URL` activates RBAC with folder-scoped claims and an
  `admin:folders` gateway scope). Without any auth variable the server boots
  open-access with a loud warning — same default as LightRAG native.
- Optional document classification at ingestion: Microsoft Information
  Protection labels extracted from Office/PDF files, mapped to tenant classes,
  with a configurable rejection ceiling and audit events.

### Operator WebUI (embedded)
- React 19 + TypeScript console served at `/webui`, shipped pre-built inside
  the wheel — no Node toolchain needed on the target host.
- Documents (filters, upload, retag, review queue, bulk actions), Tags
  governance, Retrieval (threads, citations linked to sources, query
  parameters), live Knowledge Graph with tag/document filtering and entity
  editing, immutable Activity ledger with real polling, Settings (profile,
  API explorer, folder).
- Actions are attributed to the authenticated identity; open-access
  deployments display a neutral operator identity and an explicit
  "no authentication configured" notice.

### Quality
- Full CI matrix: LightRAG 1.4.9.11 / 1.4.11 / 1.4.12 × Memgraph MAGE 3.10.1,
  unit + integration + WebUI typecheck/lint/unit + Playwright end-to-end
  (106 scenarios covering every screen, modal and control).
- See `ENV_VARIABLES.txt` for the complete configuration reference and
  `Dockerfile.example` for the minimal container wiring.

---

## Earlier storage-only releases

- **v0.5.3** — patch version surfaced in the WebUI (`core_version`
  shows `v<lightrag>+memgraph-<version>`).
- **v0.5.2** — HTTP-level e2e regression suite; guarantees every endpoint
  returns JSON (never HTML) including 5xx and startup races; readiness probe
  pattern (`/ready`).
- **v0.5.1** — auto-create vector index on query; pagination performance
  (sortable-field indexes + parallel count/fetch) fixing gateway timeouts on
  `/documents/paginated`.
- **v0.3.2 (LTS)** — error propagation on bulk indexation
  (`await result.consume()` everywhere), write throttling,
  `query_embedding` forwarding for LightRAG >= 1.4.11.
