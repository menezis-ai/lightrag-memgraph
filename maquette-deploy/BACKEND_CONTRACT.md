# Backend Contract — TwinRAG WebUI ↔ FastAPI

**Status:** working draft (extracted from maquette `source/*.jsx` + `backend/twin_demo_api/`)
**Last update:** 2026-05-24
**Goal:** every user-visible interaction in the maquette has a backend endpoint. If a maquette button does something the backend can't persist or compute, the maquette is lying.

## How to read this

- ✅ = endpoint exists in `backend/twin_demo_api/main.py` today
- 🚧 = partial: exists but missing scope/fields/cascade/auth
- ❌ = does not exist yet — must be built

The current backend is a generic `(kind, id, data-as-json)` table with one generic `PATCH /{kind}/{id}` route that deep-merges. That's enough for the maquette's demo loop (mutate + audit-trail render), but it is **not** the contract that the real TwinRAG operator console needs against Memgraph + LightRAG. Most of the ✅ below are "the demo cheats with deep-merge" — every one needs domain-specific shape + side effects when wired to Memgraph.

---

## Cross-cutting concerns

### Auth + identity (MyAccess heritage)
Per `topbar.jsx::MyAccessPill` + project memory `project_twin_myaccess_rights_model`:
- Every request must carry a session bearer (OIDC / Keycloak). Backend resolves the bearer → BNP UID → ENTITY (via MyAccess bridge) → workspace.id mapping.
- Workspace scoping is **enforced server-side** based on ENTITY membership, not by trusting a client-sent `workspace` field.
- The `palier` (1/2/3) is Twin-local (not MyAccess). Backend reads it from a separate `palier` table keyed by UID + workspace. Affects RBAC on every write endpoint flagged below as "steward only" / "contributor+".
- All write endpoints must record `(uid, palier_at_action_time, workspace_id, ts)` into the audit trail.

### Pagination + filtering shape
- List endpoints adopt: `?cursor=&limit=&sort=&order=&q=` + per-resource filters.
- Response envelope: `{ items: [...], next_cursor: string|null, total_count: number }`.
- The maquette currently `GET /{kind}` returns a bare array (✅ today). Production must move to the envelope; `total_count` is needed for the pending counters, status pill counts (documents.jsx `counts`), etc.

### Error shape
- 4xx + 5xx: `{ error: { code: "E_FOO_BAR", message: "...", details?: {...} } }`.
- Already used codes the maquette displays verbatim: `E_UNSUPPORTED_FORMAT`, `E_PROVIDER_RATE_LIMIT`, `E_IP_NOT_ALLOWLISTED`. Treat as the start of the catalog, not the end.

### Idempotency
- `POST /documents/upload`, `POST /documents/scan`, ontology approve/reject, tag-request approve/reject: require an `Idempotency-Key` header. Backend stores `(uid, key) → response` for 24 h so a retried network op doesn't double-fire.

### Read-your-own-writes
- App.jsx pattern: optimistic UI update → `twinDb.patch(...)` fire-and-forget. The PATCH response is the merged row. **Every write endpoint must return the resulting entity (full shape)** so the client can reconcile if the optimistic shape diverges (especially for cascade fields: `chunks`, `last_edit`, computed counters).

### Audit trail
- `/api/mutations` (✅) is the spine. Every write path must call the same audit hook with a typed payload. Real wiring: a Memgraph `:AuditEvent` node linked to `:User` (UID) + the mutated entity, AND optionally an event-bus emit per `project_events_middleware`.
- Documents.jsx `DocDetailPanel`'s audit tab filters client-side `kind === "docs" && target_id === doc.id`. With real volume that must move server-side via `/api/mutations?target_kind=docs&target_id=d13`.

---

## Domain: Documents

Backing store: LightRAG DocStatus (label `DocStatus_{workspace}`) + KV (label `KV_{workspace}_{namespace}`) + Vector index (`Vec_{workspace}_{namespace}`). All under `src/twindb_lightrag_memgraph/`.

### Document shape (canonical, consumed by `documents.jsx`)

```ts
{
  id: string,                    // "d13" in fixture; UUID in prod
  type: "file"|"confluence"|"sharepoint"|"url",
  source: string,                // filename or path
  summary: string,
  tags: string[],                // tag names (canonical, lowercase)
  status: "completed"|"processing"|"pending"|"failed",
  chunks: number|null,           // null when not yet processed
  updated: string,               // human-readable "2h ago" — see note below
  updated_at: string,            // ISO — needed for sort, not yet in fixture
  visibility: "private"|"internal",
  workspace: string,
  review?: {                     // present only when in / past pending-review
    state: "pending-review"|"approved"|"rejected",
    requested_by: string,        // UID
    requested_at: string,        // ISO date
    justification?: string,
    reviewed_by?: string,
    reviewed_at?: string,
    reason?: string,             // on reject
    edited?: boolean             // on approve when steward used Edit & approve
  },
  // Fields the maquette uses from MOCK_LINEAGE but expects on the doc eventually:
  uploaded_by?: string,
  uploaded_at?: string,
  sha256?: string,
  bytes?: number,
  pipeline_version?: string,
  reingest_count?: number,
  last_ingest_duration_ms?: number
}
```

> **Note on `updated`:** the maquette displays the human string ("2h ago") that the backend currently returns verbatim. Production should return `updated_at` (ISO) and let the client format. Keep `updated` in the response for one release as a transition aid.

### `GET /api/documents` 🚧

**Today:** `GET /api/docs` returns the raw array (`backend/main.py::list_kind`).
**Needed:**
- Query params: `workspace`, `status` ∈ {all,completed,processing,pending,failed}, `pending_only=true`, `tag` (repeatable, AND semantics), `q` (substring on `source`), `sort` ∈ {updated_at,source,chunks}, `order`, `cursor`, `limit`.
- Response: `{ items: Document[], next_cursor, total_count, counts_by_status: { all, completed, processing, pending, failed } }`. The status-pill counts in `documents.jsx::counts` need a server-side aggregate to avoid loading every doc.
- MUST exclude `review.state ∈ {pending-review, rejected}` from `pending_only=false` (default) — the maquette filters this client-side today (documents.jsx L142–149).

**Consumed by:** `documents.jsx` (table, status pills, tag filters, sort).

### `GET /api/documents/pending` ❌

**Why separate:** the steward queue is rendered in a dedicated section above the grid (documents.jsx::pendingDocs) and the bell notifications also pull from this set (n_p01 / n_p02). Conceptually it is `GET /api/documents?pending_only=true&sort=requested_at`, but giving it its own endpoint lets ops cache it independently (notifications poll it every ~30s).
**Response:** `{ items: Document[] /* all with review.state == "pending-review" */, total_count }`.
**RBAC:** palier ≥ 2 sees the queue (own submissions visible read-only). Palier ≥ 3 sees + can act.
**Consumed by:** `documents.jsx::pendingDocs`, `topbar.jsx::NotificationsPopover` (indirectly via /notifications).

### `GET /api/documents/{id}` ✅

Already works generically. Must return the full Document shape above.
**Consumed by:** detail panel hydration if we ever switch from "list everything" to per-row fetch.

### `PATCH /api/documents/{id}` 🚧

**Today:** generic deep-merge.
**Needed (typed sub-paths):**
- `review.state` transitions are domain ops, not arbitrary patches. Reject any PATCH that mutates `review.state` directly — force callers through `/approve` and `/reject` so audit, notification, and downstream effects (re-index, exclude-from-retrieval, notify requester) fire deterministically.
- Allow PATCH on: `summary`, `tags`, `visibility`. Re-tagging via PATCH must trigger the same chunk-retag cascade as the bulk retag endpoint (see below).
- Response: full merged Document.

### `POST /api/documents/{id}/approve` ❌

**Body:** `{ reviewed_by: string /* UID, server may override from auth */, edited?: { summary?: string, tags?: string[] } }`.
**Side effects:**
- Set `review.state = "approved"`, `review.reviewed_at = now()`.
- If `edited` present: apply summary/tags BEFORE flipping state, record `review.edited = true`.
- Emit `doc-review` activity event with `from_state: "pending-review", to_state: "approved", review_duration_s`.
- Push notification to `review.requested_by`.
- If doc was excluded from retrieval set (because of pending-review tag), remove that overlay.
**RBAC:** palier ≥ 3.
**Consumed by:** `documents.jsx::approveDoc`, `documents.jsx::submitEditApprove`.

### `POST /api/documents/{id}/reject` ❌

**Body:** `{ reviewed_by: string, reason: string /* required, non-empty */ }`.
**Side effects:** as above, but `to_state: "rejected"`, `reason` stored in audit + notification body. Reject does NOT delete — doc stays in KB labeled rejected and is excluded from default retrieval.
**Consumed by:** `documents.jsx::submitReject`.

### `DELETE /api/documents/{id}` 🚧

**Today:** generic delete. Drops the row from SQLite only.
**Needed cascade:**
- Delete DocStatus node (label `DocStatus_{workspace}` where `id = {id}`).
- Delete all `KV_{workspace}_chunks` entries with `doc_id = {id}`.
- Delete all vectors for those chunk_ids in `Vec_{workspace}_chunks` (Memgraph `DROP VECTOR INDEX entry` per id, or rely on chunk-node delete cascading via `:MENTIONED_IN`).
- Decrement `sources_count` and `chunks_count` on every tag that pointed at this doc.
- Emit `doc.deleted` audit event with `{chunks_purged}`.
- Confirm via response: `{ ok: true, chunks_purged: number, vectors_purged: number, tags_decremented: string[] }` so the toast can show the real impact (currently fakes it with `${d.chunks ?? 0}`).
**RBAC:** palier ≥ 3.
**Consumed by:** `documents.jsx::confirmDelete` (single), `documents.jsx::bulkDeleteOpen` confirm flow.

### `POST /api/documents/bulk-delete` ❌

**Body:** `{ ids: string[] /* max 200 */ }`.
**Response:** `{ ok: true, deleted: [{id, chunks_purged}], failed: [{id, error}] }`.
**Side effects:** same cascade per id, batched into UNWIND queries against Memgraph. Single audit event per id (so each shows up in the timeline).
**Consumed by:** `documents.jsx` bulk-bar delete (currently loops `deleteDoc` client-side, which is correct as a fallback but doesn't scale).

### `POST /api/documents/upload` ❌

**Body:** multipart `files[]` + `tags[]` + `visibility` + `workspace` (server may override).
**Response:** `{ ok: true, queued: [{id, source, status: "pending"}], rejected: [{name, reason, error_code}] }`.
**Side effects:** writes binary to object store, creates a DocStatus node with `status="pending"`, enqueues the LightRAG ingestion job. The `/documents/pipeline_status` endpoint surfaces progress.
**Consumed by:** `modals.jsx::AddSourceModal::submit`. Today the modal animates a fake upload and emits a toast — no network call.

### `POST /api/documents/links` ❌

**Body:** `{ urls: [{url, type: "confluence"|"sharepoint"|"url"}], tags?, workspace? }`.
**Response:** same envelope as upload.
**Side effects:** linked sources are different from uploads — they get a Connection record (see Domain: Connections) instead of bytes. Worth keeping the endpoint separate because the validation (URL parsing, connection-pool check) diverges.
**Consumed by:** `modals.jsx::AddSourceModal` URL chips.

### `POST /api/documents/scan` ❌

**Query:** `?retry=failed` (default: full scan).
**Response:** `{ scan_id: string, queued: number, retrying: number }`.
**Side effects:** asks the pipeline to re-scan all sources for changes (or just failed ones). Mirrors the existing LightRAG `POST /documents/scan` route (referenced in `api.jsx::OPENAPI_GROUPS`).
**Consumed by:** `documents.jsx` Scan/Retry header button. Hardcoded toast text references this path already.

### `GET /api/documents/pipeline_status` ❌

**Response:**
```ts
{
  state: "busy"|"paused"|"idle",
  workers: [{ id, label, status: "ok"|"warn"|"error", throughput: string, note: string }],
  queue: [{ source, state: "processing"|"queued", progress: number, eta: string }],
  stats_24h: { processed: number, failed: number, queue_lag: string }
}
```
**Consumed by:** `documents.jsx::PipelineStatusPopover`. Today entirely mocked (`PIPELINE_WORKERS`, `PIPELINE_QUEUE` constants).

### `POST /api/documents/pipeline_status` ❌ (stop / resume)

**Body:** `{ action: "stop"|"resume" }`.
**Side effects:** drain workers (stop) or spin up (resume). Idempotent.
**Consumed by:** PipelineStatusPopover Stop / Resume buttons.

### `POST /api/documents/{id}/reprocess` ❌

**Response:** `{ ok: true, job_id: string }`.
**Side effects:** re-ingest the source through the LightRAG pipeline. Increments `reingest_count`.
**Consumed by:** `documents.jsx::DocDetailPanel::reprocess` and `documents.jsx` bulk-bar Re-process (which would need a `POST /api/documents/bulk-reprocess` variant accepting `ids[]`).

### `GET /api/documents/{id}/raw` ❌ + `GET /api/documents/{id}/original` ❌

- `raw` → post-extraction text (what the indexer sees). Used by `RawDocModal` "extracted" mode.
- `original` → binary stream of the source file with correct mime, for the PDF / Word / Confluence preview ("original" mode).

For Confluence / SharePoint linked sources, `original` could be a 302 redirect to the upstream URL the user is already authenticated for.
**Consumed by:** `documents.jsx::RawDocModal`. Today the modal renders `MOCK_RAW_TEXT_BY_DOC[d.id]` for d13/d14 and a generic fallback.

### `GET /api/documents/{id}/chunks` ❌

**Query:** `?cursor=&limit=&q=` (search within chunks).
**Response:** `{ items: [{id, pos, tokens, text, score?}], next_cursor }`.
**Consumed by:** `DocDetailPanel` "Chunks" tab. Today shows the same 3 hardcoded MOCK_CHUNKS for every doc.

### `GET /api/documents/{id}/lineage` ❌

**Response:** `{ uploaded_at, uploaded_by, ingest_history: [{at, duration_ms, pipeline_version, trigger}], sha256, bytes, workspace }`.
**Consumed by:** `DocDetailPanel` "Lineage" tab. Today hardcoded as `MOCK_LINEAGE`.

### `POST /api/llm/draft` ❌

**Body:** `{ kind: "summary"|"tags"|"reject_reason"|"tag_definition"|"tag_justification", entity_id: string, context?: {...} }`.
**Response:** `{ text: string, model: string, latency_ms: number }`.
**Consumed by:** `documents.jsx` Edit & approve modal, Reject modal, and `tags.jsx` Request / Edit / Reject modals via `window.AiAssistButton`. Comment in documents.jsx L1380 explicitly calls this out as the integration target.

---

## Domain: Tags / Thesaurus

Backing store: dedicated `KV_{workspace}_tags` namespace OR a typed `:Tag` node in Memgraph (recommend the latter — `project_twin_rag_tagging_roadmap.md` phase 2 already needs it).

### Tag shape

```ts
{
  id: string,                    // == tag name (lowercase, kebab)
  tag: string,                   // duplicated for convenience
  tier: 1|2|3|"requested",
  category: string,              // matches MOCK_TAG_CATEGORIES.id
  status: "active"|"pending-promotion"|"pending-review"|"deprecated"|"rejected",
  def: string,
  aliases: string[],
  deprecates: string[],
  sources_count: number,
  chunks_count: number,
  query_freq_30d: number,
  created: { by: string, at: string },
  last_edit: { by: string, at: string, action: string },
  related: [{ tag: string, strength: number /* 0..1 */ }],
  examples: string[],            // source names
  // Only when tier === "requested" / status === "pending-review":
  requested_by?: string,
  requested_at?: string,
  justification?: string,
  review?: { state, reviewed_by, reviewed_at, reason?, edited? }
}
```

### `GET /api/tags` 🚧

**Today:** generic `GET /api/tags` returns array.
**Needed:** `?category=&status=&tier=&q=&include_requested=true` + envelope with `total_count`. The maquette splits "requested" out into its own pending section (`tags.jsx::requested`) so the default response should NOT include them; add `?include_requested=true` for the pending queue.

### `GET /api/tags/pending` ❌

`GET /api/tags?include_requested=true&tier=requested` — convenience endpoint matching the same pattern as `/documents/pending` (notification fanout + counters). Returns the items rendered in `tags.jsx::requested.map(...)`.

### `GET /api/tags/{id}` ✅

Generic works. Real backend: also returns computed `related[]` (co-occurrence over chunks).

### `POST /api/tags` ❌ (request new tag)

**Body:** `{ name, def, category, aliases?, justification, requested_by /* server may override */ }`.
**Response:** the new Tag with `tier: "requested", status: "pending-review"`.
**Side effects:** push notification to all stewards in workspace; emit `tag.request_new` activity event.
**Consumed by:** `tags.jsx::TagActionModal::commit` for `action.kind === "request"`. Comment in tags.jsx L673 explicitly flags this as "would need a POST endpoint that the backend doesn't surface yet".
**RBAC:** palier ≥ 2.

### `PATCH /api/tags/{id}` 🚧

**Today:** generic deep-merge.
**Needed:** restrict editable fields to `def`, `aliases`, `last_edit`. Forbid status / tier changes through PATCH — those go through their own endpoints (approve, deprecate, …).

### `POST /api/tags/{id}/approve` ❌

**Body:** `{ reviewed_by, tier?: 1|2|3 /* default 2 */, edited?: { def?, aliases?, category? } }`.
**Side effects:** flip to `status="active"`, set tier, apply edits, emit `tag.approved`, notify `requested_by`.
**RBAC:** palier ≥ 3.
**Consumed by:** `tags.jsx::TagsTab` inline Approve button + `TagActionModal` `edit-approve`.

### `POST /api/tags/{id}/reject` ❌

**Body:** `{ reviewed_by, reason: string }`.
**Side effects:** `status="rejected"`, emit `tag.rejected`, notify.
**RBAC:** palier ≥ 3.
**Consumed by:** `tags.jsx::TagActionModal` `reject`.

### `POST /api/tags/{id}/deprecate` ❌

**Body:** `{ reason?: string }`.
**Side effects:** `status="deprecated"`. Docs that carry this tag get an overlay `excluded_from_default_retrieval`. Existing chunks keep the tag in the index (so `include_deprecated: true` queries still work, per tags.jsx L760).
**RBAC:** palier ≥ 3.
**Consumed by:** `tags.jsx::TagActionModal` `deprecate`.

### `POST /api/tags/{id}/synonyms` ❌

**Body:** `{ add?: string[], remove?: string[] }`.
**Side effects:** updates `aliases`. The query-rewriting gateway picks up the change; no chunk-level mutation.
**Consumed by:** `tags.jsx::TagActionModal` `synonyms`.

### `DELETE /api/tags/{id}` 🚧

**Today:** generic delete.
**Needed body:** `{ strategy: "migrate"|"untag", migrate_to?: string /* required when strategy=migrate */ }`.
**Side effects:**
- `migrate`: re-tag every chunk currently tagged `{id}` with `migrate_to`, then drop `{id}`. Atomic transaction (or compensating audit if not).
- `untag`: strip the tag from every chunk, drop `{id}`.
- Return `{ ok, chunks_affected, sources_affected }` for the toast (currently faked).
**RBAC:** palier ≥ 3.
**Consumed by:** `tags.jsx::TagActionModal` `delete`.

### `POST /api/tags/{id}/suggestions` ❌ (contributor edit suggestion)

Contributor (palier 2) can't directly edit a Tier-1/2 tag definition — they file a suggestion. Backend stores it as a pending review item (separate entity kind `tag_suggestion`?) for steward triage.
**Consumed by:** `tags.jsx::TagActionModal` `suggest`. Currently toast-only.

### `GET /api/tags/export` ❌

**Query:** `?format=json` (others later).
**Response:** the JSON payload `exportThesaurusJson()` builds client-side in `tags.jsx`. Moving it server-side lets the backend stamp signing metadata + skip the round-trip of 100+ tags through the client.
**Consumed by:** `tags.jsx` Export thesaurus header button.

---

## Domain: Cross-domain pending review queue

The bell badge (`topbar.jsx::unreadCount`) and the demo HUD need a single number for "outstanding review items across the workspace". Today, that's computed client-side from `notifications`. Server-side aggregate avoids polling 4 endpoints.

### `GET /api/review-queue/summary` ❌

**Response:**
```ts
{
  total: number,
  by_kind: {
    docs_pending: number,
    tags_pending: number,
    ontology_proposals_pending: number,
    tag_suggestions_pending: number
  }
}
```
**Consumed by:** topbar pill, future "Review center" tab.

---

## Domain: Ontology Studio

Backing store: TBD — see Open Questions. The maquette currently keeps everything in `window.__ontoLive` / `window.__ontoInbox` / `window.__ontoAudit` (in-memory, no persistence) and resets on tab unmount unless those globals are populated.

### Node shape

```ts
{
  id: string,                    // "e_router02"
  name: string,
  type: "PRODUCT"|"TECHNOLOGY"|"CONCEPT"|"ORG"|"PERSON"|"LOCATION"|"INFRA"|"APPLICATION",
  x: number, y: number,          // layout coords (steward-validated)
  status: "validated"|"pending"|"proposed",
  summary?: string,
  kind?: string                  // free-form sub-label, e.g. "Core router"
}
```

### Edge shape

```ts
{
  id: string,
  source: string, target: string,
  label: string,                 // e.g. "BACKS_UP", "ROUTES_VIA"
  strength: number,              // 0..1, used as confidence
  status: "validated"|"pending"|"proposed"
}
```

### Inbox proposal shape (3 kinds)

```ts
// entity
{ id, kind: "entity", name, type, confidence, evidence, rationale }
// relation
{ id, kind: "relation", source, target, label, confidence, evidence, rationale }
// entity-type-fix
{ id, kind: "entity-type-fix", subject, existing_type, proposed_type, note, confidence, evidence }
```

### `GET /api/ontology/graph` ❌

**Query:** `?workspace=` (server-enforced from auth).
**Response:** `{ nodes: Node[], edges: Edge[] }`.
**Source:** DSEP-validated graph in Memgraph (see `src/twindb_lightrag_memgraph/intelligence/ontology/`). Backed by the same nodes / edges that LightRAG built — the Studio is the steward's view of *validated* subset (status="validated") + the pending overlay.
**Consumed by:** `ontology-studio.jsx` on mount (seeds `nodes`, `edges` from `ONTOLOGY_SEED_*`). Also by `wargame.jsx::computeImpact` (which reads `window.__ontoLive` published by the studio).

### `GET /api/ontology/inbox` ❌

**Response:** `{ items: InboxProposal[], total_count }`.
**Source:** the DSEP pipeline `extract → cluster → enrich → validate` (`src/twindb_lightrag_memgraph/intelligence/ontology/pipeline.py`) — `pipeline.run()` is dry-run by default; the inbox surfaces those dry-run results until a steward approves.
**Consumed by:** `ontology-studio.jsx::inbox` state + `window.InboxPanel`.

### `POST /api/ontology/proposals/{id}/approve` ❌

**Body:** depends on kind:
- entity: `{ position?: {x, y} }` (steward drops it at a canvas position)
- relation: `{}` (just accept)
- entity-type-fix: `{}`

**Side effects:** writes the node / edge / type change to the validated graph in Memgraph (status="validated"). Removes the proposal from inbox. Emits ontology audit event with kind ∈ {create-node, create-edge, type-fix}.
Equivalent to `pipeline.approve(result, workspace)` per CLAUDE.md.
**RBAC:** palier ≥ 3 (steward-driven per UI design).
**Consumed by:** `ontology-studio.jsx::dropInboxItem`.

### `POST /api/ontology/proposals/{id}/reject` ❌

**Body:** `{ reason?: string }`.
**Side effects:** mark as rejected (audit + dropped from inbox; don't re-propose for N days).
**Consumed by:** `ontology-studio.jsx::rejectInboxItem`.

### `POST /api/ontology/nodes` ❌ (steward creates manually)

Not used in the current maquette (creation is always inbox-driven), but the canvas does have a drag-to-create-edge gesture (`ontology-canvas.jsx`). Defensible to add when needed.

### `POST /api/ontology/edges` ❌

**Body:** `{ source, target, label, status?: "validated"|"pending" }`.
**Response:** the new Edge with id.
**Side effects:** audit event `create-edge`.
**Consumed by:** `ontology-studio.jsx::createRelation`. Currently writes to in-memory state only.

### `PATCH /api/ontology/nodes/{id}` ❌

**Body:** `{ type?: string, name?: string, summary?: string, x?, y? }`.
**Side effects:** audit event `type-fix` if type changed, `move` if coords changed (or silent — design call).
**Consumed by:** `ontology-studio.jsx::onTypeChange` (NodeInspector type dropdown).

### `GET /api/ontology/audit` ❌

**Query:** `?since=&limit=`.
**Response:** `{ items: [{ts, who, action, target, kind}], next_cursor }`.
**Consumed by:** `ontology-studio.jsx::audit` + the InspectorPanel audit rail. Today seeded from `SEED_AUDIT`, appended to via `logAudit` (client-only).

---

## Domain: Wargame / Impact analysis

The wargame widget (`wargame.jsx::WargameImpactWidget`) and the Studio's wargame mode (`ontology-studio.jsx::wargameImpacted`) currently run a BFS in JavaScript over the in-memory graph. That's fine while the dataset is the 14-node demo, but a real CIB topology will be 10k+ nodes. Server-side traversal is non-negotiable.

### `POST /api/impact-analysis` ❌

**Body:** `{ origin_id: string, depth: number /* default 3, max 5 */, edge_filter?: { status?: ["validated","pending"] }, workspace? }`.
**Response:**
```ts
{
  origin: Node,
  impacted: [{
    node: Node,
    depth: 1|2|3,
    via: { from: string, edge: Edge }  // shortest path
  }],
  edge_stats: { total: number, validated: number, pending: number, confidence_pct: number },
  apps_impacted: number,
  pra_runbooks: [{ doc_id, source, section?, status: "approved"|"draft", approved_by?, approved_at? }]
}
```
The `confidence_pct` field IS what `wargame.jsx` computes from `validatedEdges / totalEdges`. Moving the BFS server-side keeps the math consistent regardless of which tab triggers it.
**Side effects:** none (read-only). Optional: log a `retrieval` activity event with `mode: "impact-analysis"` so the dashboard counts them.
**Consumed by:** `wargame.jsx::WargameImpactWidget` + `ontology-studio.jsx::wargameImpacted`.

### `GET /api/impact-analysis/pra-runbooks` ❌

**Query:** `?for_apps=app_id1,app_id2`.
**Response:** the `pra_runbooks` slice above, fetched on demand for the "Open DR procedures" toggle (`wargame.jsx::pra` state).

---

## Domain: Retrieval

Existing LightRAG endpoints (`/query`, `/query/stream`) are already documented in `api.jsx::OPENAPI_GROUPS`. The Twin gateway injects `tag_filter` and `visibility` scoping. The current maquette **does not call them yet** — `retrieval.jsx::send` streams `MOCK_ANSWER_TOKENS` with a `setInterval`. Banner at top of the tab (L315–323) is explicit about this.

### `POST /api/query/stream` ❌ (real wiring)

**Body (matches the existing LightRAG contract + Twin extensions):**
```ts
{
  query: string,
  mode: "naive"|"local"|"global"|"hybrid"|"mix"|"bypass",
  top_k: number,
  max_tokens: number,             // "Max tokens · text unit"
  history_turns: number,
  only_need_context?: boolean,
  only_need_prompt?: boolean,
  tag_filter?: { all: string[], any: string[] },
  workspace?: string,             // server-enforced
  conversation_id?: string        // for history + audit linkage
}
```

**Response:** SSE stream emitting tokens, ending with a final event carrying `{ sources: Source[], latency_ms, tokens_in, tokens_out }`.

**Source shape (consumed by `retrieval.jsx::Turn`):**
```ts
{
  n: number,                     // citation index
  type: "file"|"confluence"|"sharepoint"|"url",
  name: string,
  meta?: string,                  // "p.4" — page / section
  score: number,                  // 0..1
  doc_id?: string,                // so click → Documents tab works
  chunk_id?: string,
  url?: string                    // for "Open source" external link
}
```

**Side effects:** emit a `retrieval` activity event (the seeded Activity feed already has shape — see `data.js::MOCK_ACTIVITY` retrieval entries with `meta: { mode, top_k, tag_filter, latency_ms, tokens_in, tokens_out }`).
**Consumed by:** `retrieval.jsx::send`.

### Conversations / threads ❌

Conversations are persisted to **localStorage only** today (`retrieval.jsx::threads` state, `localStorage.setItem("twin-rag.threads", ...)`). For the BNP demo this is a footgun — clearing browser data drops everything, no cross-device.

#### `GET /api/conversations` ❌
**Response:** `{ items: [{id, title, created, updated, message_count}], next_cursor }`.

#### `POST /api/conversations` ❌
**Body:** `{ title?: string }` (auto-derive from first user message if absent).

#### `GET /api/conversations/{id}` ❌
**Response:** `{ id, title, created, updated, messages: [{role, text|tokens, sources?, answer_mode?}] }`.

#### `DELETE /api/conversations/{id}` ❌
**Consumed by:** `retrieval.jsx::deleteThread`.

#### `POST /api/conversations/{id}/messages` ❌
**Body:** `{ role, text }`. Append-only; the assistant message lands via the SSE stream and is upserted by the server.

---

## Domain: Activity

The seeded events live in `backend/seed-data/activity.json`. Generic GET works.

### Activity event shape

```ts
{
  id: string,                    // ULID e.g. "evt_01HX9ZRV1"
  ts: string,                    // ISO
  rel: string,                   // human "2m ago" — TBD: compute client-side from ts
  day: string,                   // "Today", "Yesterday", "2026-05-08"
  kind: "retrieval"|"tag-mutation"|"doc-review"|"source-uploaded"
       |"source-ready"|"source-failed"|"pipeline-warning"|"auth"
       |"settings"|"confluence"|"sharepoint"|"url",
  sev: "info"|"warning"|"error",
  actor: { user: string, role: string },
  target: { type: "source"|"query"|"session"|"bulk"|"workspace", label: string, id?: string },
  summary: string,
  meta: { /* per-kind, see seed for shapes */ }
}
```

### `GET /api/activity` 🚧

**Today:** generic GET returns the seed.
**Needed:**
- Query: `range` ∈ {24h,7d,30d,all}, `kind` (comma-separated), `sev`, `actor`, `q`, `cursor`, `limit`.
- Response: envelope + counters: `{ items, next_cursor, total_count, by_severity: {info, warning, error}, by_kind: {...}, by_bucket: {...} }`. The maquette computes those client-side today (`activity.jsx::activity-stats`).

### `GET /api/activity/{id}` ✅

Generic works.

### `GET /api/activity/poll` ❌

**Query:** `?since={ts}` (cursor on `ts` not `id`).
**Response:** `{ new_count: number, latest_ts: string }`.
**Why separate:** the activity tab polls every 15s today via `/api/mutations` (a different shape) and computes the pending-count locally. A dedicated lightweight endpoint avoids transferring full event bodies just to show "+3 new events".
**Consumed by:** `activity.jsx::pendingCount` (currently increments every 30s on a fake timer).

### `POST /api/activity/clear` ❌

**Body:** `{ confirm: "CLEAR" }`.
**Response:** `{ purged: number, retained: number }`.
**Side effects:** purge events past their retention window (see `activity.jsx::RETENTION_DAYS` + `data.js::MOCK_RETENTION`). Itself records an `admin.clear` event.
**RBAC:** palier ≥ 3.

### `POST /api/activity/{id}/replay` ❌

**Body:** `{}`.
**Side effects:** for `kind === "source-failed"` events only — re-enqueues the source for ingestion. Returns the new pipeline job id.
**Consumed by:** `activity.jsx::ActivityDetail` "Replay ingestion" button.

### `GET /api/activity/export` ❌

**Query:** `?format=csv&range=...&kind=...` (same filter set as `/activity`).
**Response:** streamed CSV with columns matching `activity.jsx::exportActivityCsv` (id, ts, kind, sev, actor.user, actor.role, target.type, target.label, summary, meta).
**Why server-side:** CSV of 50k events shouldn't flow through the browser. Currently exports the client-filtered view.

---

## Domain: Notifications

Notification shape (consumed by `topbar.jsx::NotificationsPopover`):

```ts
{
  id: string,
  kind: "doc-review"|"tag-request"|"tag-mutation"|"source-ready"|"source-failed"
       |"pipeline-warning"|"source-uploaded"|"retrieval"|"info",
  title: string,
  tagname?: string,              // for tag-related kinds, e.g. "argocd"
  suffix?: string,               // e.g. "awaiting steward approval"
  sub?: string,
  rel: string,                   // human relative time
  read: boolean,
  // Not in fixture but needed:
  ts?: string,                   // ISO for sorting
  target_id?: string,            // jump target (doc id, tag id, etc.)
  target_kind?: "doc"|"tag"|"source"
}
```

### `GET /api/notifications` 🚧

**Today:** generic GET.
**Needed:** `?unread_only=&kind=&since=&limit=` + counts envelope `{ items, unread_count, total_count }`.

### `POST /api/notifications/mark-all-read` ❌

**Body:** `{ /* optional: kinds? */ }`.
**Side effects:** flip `read=true` for the current user's notifications.
**Consumed by:** `topbar.jsx::onMarkAllRead`. Today mutates client state only.

### `POST /api/notifications/{id}/read` ❌

Single-item read marker. Useful when the user clicks a notification → it jumps to the target → should flip read.

### `DELETE /api/notifications` ❌

**Body:** `{ /* optional: kinds? or ids? */ }`. Default = clear all.
**Consumed by:** `topbar.jsx::onClearNotifications`.

### Server push (SSE) ❌

The bell needs new notifications to surface without poll. A long-lived `GET /api/notifications/stream` (SSE) per user-session that emits `{ event: "notification", data: Notification }` is the cheapest path; falls back to 30s polling if SSE is blocked.

---

## Domain: System status

The maquette's `system-status.jsx` runs the entire state machine client-side from a single `INITIAL_SYSTEM_STATUS` object mutated by the Tweaks panel. Backend needs to supply the truth.

### `GET /api/system/status` ❌

**Response:**
```ts
{
  gateway: "ok"|"degraded"|"down",
  gateway_last_success_at: string,
  llm_quota_percent: number,
  llm_quota_reset_at: string,
  embedder_status: "ok"|"rate-limited"|"degraded",
  reranker_status: "ok"|"rate-limited"|"degraded",
  memgraph: "ok"|"degraded"|"down",
  memgraph_lag_s?: number,
  indexer: "ok"|"degraded"|"down",
  indexer_throughput?: string,
  manual_read_only: boolean,
  session_expires_at: string|null
}
```
**Consumed by:** `app.jsx::systemStatus`, `system-status.jsx::computeSystemStatus` (banners + indicator). The `computeSystemStatus` logic stays client-side (it's pure derivation), but the inputs come from this endpoint.

### `POST /api/system/read-only` ❌

**Body:** `{ enabled: boolean }`.
**RBAC:** palier ≥ 3.
**Consumed by:** Tweaks panel today; real ops button later.

### `GET /api/system/status/stream` ❌

SSE stream emitting status diffs every ~5s OR on edge changes. Required so the gateway-down banner appears within 5s not 5min.

### `POST /api/system/retry-gateway` ❌

**Consumed by:** `system-status.jsx` "Retry now" button in the gateway-down banner.

---

## Domain: API tokens

### Token shape

```ts
{
  id: string,                    // "tok_a1b2"
  name: string,
  scopes: string[],              // ALL_SCOPES in settings.jsx
  last_used: string|"—",
  created: string,
  prefix: string                 // "tw_pat_a1b2" (never the full secret)
}
```

### `GET /api/tokens` ❌

**Response:** `{ items: Token[], total_count }`.

### `POST /api/tokens` ❌

**Body:** `{ name, scopes: string[] }`.
**Response:** `{ token: Token, secret: string }` — `secret` returned **once** (matches the UI's "Copy now" reveal pattern in settings.jsx::TokensSection L184).
**RBAC:** palier ≥ 2.

### `DELETE /api/tokens/{id}` ❌

Revokes. Returns `{ ok: true }`. Backend must invalidate the bearer at the gateway.
**Consumed by:** `settings.jsx::revoke`.

---

## Domain: Settings — Workspace

### `GET /api/workspaces/{id}` ❌ (or "current")

**Response:**
```ts
{
  id: string,
  display_name: string,
  visibility: "private"|"internal",
  region: string,
  default_tags: string[],
  retention: [{ area: string, ttl: string, note: string }]
}
```
Most fields are env-controlled (Helm) and read-only in the UI. `default_tags` is editable.

### `PATCH /api/workspaces/{id}` ❌

**Body:** `{ default_tags?: string[] }`.
**RBAC:** palier ≥ 3.
**Consumed by:** `settings.jsx::WorkspaceSection::addDefault / removeDefault`.

### `POST /api/workspaces/{id}/leave` ❌

**Body:** `{ confirm: string /* must equal workspace.id */ }`.
**Side effects:** revoke the caller's access to this workspace; flips `members` row to `status: "left"` (kept for audit). Caller is logged out of the workspace next request.
**Consumed by:** `settings.jsx::DangerSection` Leave button.

### `DELETE /api/workspaces/{id}` ❌

**Body:** `{ confirm: string }`.
**Side effects:** wipes Memgraph workspace labels, vector indices, KV namespace; leaves the Helm release orphaned. Two-step confirmation per UI hint (UI + email link).
**RBAC:** palier ≥ 3 + owner.
**Consumed by:** `settings.jsx::DangerSection` Delete button. Currently surfaces a hardcoded "not available in demo" toast.

---

## Domain: Settings — Members

### Member shape

```ts
{
  email: string,
  name: string,
  palier: 1|2|3,
  role: string,                  // free-form label
  joined: string,                // ISO
  last_seen: string|"—",
  status: "active"|"invited"|"left",
  uid?: string                   // MyAccess UID, needed for cross-system propagation
}
```

### `GET /api/workspaces/{id}/members` ❌

**Response:** `{ items: Member[], counts: { stewards, contributors, readers } }`.

### `POST /api/workspaces/{id}/members/invite` ❌

**Body:** `{ email, palier }`.
**Side effects:** create `Member` with `status="invited"`, send invite email, audit event.
**Consumed by:** `settings.jsx::MembersSection::invite`.
**RBAC:** palier ≥ 3.

### `PATCH /api/workspaces/{id}/members/{email}` ❌

**Body:** `{ palier?: 1|2|3 }`.
**Side effects:** update palier (Twin-local — does NOT touch MyAccess), audit. Refuse if `email == caller.email` (no self-demotion).
**Consumed by:** `settings.jsx::MembersSection::setPalier`.

### `DELETE /api/workspaces/{id}/members/{email}` ❌

Revokes access in Twin only. Refuses self-delete.
**Consumed by:** `settings.jsx::MembersSection::remove`.

---

## Domain: Settings — Providers

### `GET /api/providers` ❌

**Response:**
```ts
{
  llm: { provider, model, base_url, key_ref, rate_limit_rpm, monthly_quota_usd, monthly_spend_usd },
  embedder: { provider, model, base_url, key_ref, rate_limit_rpm, dims },
  reranker: { provider, model, base_url, enabled }
}
```
Matches `data.js::MOCK_PROVIDERS` shape. `key_ref` is `secret://...` indirection — never returns the actual key.

### `PATCH /api/providers/{kind}` ❌

`kind ∈ {llm, embedder, reranker}`.
**RBAC:** palier ≥ 3.
**Side effects:** triggers a config-reload on the inference path (or restart job).

### `POST /api/providers/{kind}/test` ❌

**Response:** `{ ok: true, latency_ms, model, sample?: string }`.
**Side effects:** sends a tiny probe request to the provider, records as a `settings` activity event.
**Consumed by:** `settings.jsx::ProvidersSection` "Test connection" button per provider (currently fakes via toast).

---

## Domain: Settings — Connections (external sync)

Backing store: `:Connection` node in Memgraph per workspace.

### Connection shape

```ts
{
  id: string,
  kind: "confluence"|"sharepoint"|"url",
  name: string,
  url: string,
  space_key?: string,            // confluence
  site_id?: string,              // sharepoint
  status: "ok"|"syncing"|"token-expired"|"sync-failed"|"disconnected",
  health: "ok"|"warn"|"error",
  sources_tracked: number,
  last_sync_at: string|null,
  last_sync_duration_ms: number|null,
  next_sync_at: string|null,
  schedule: string,              // "every 12h", "daily", "manual"
  oauth_account: string|null,
  scopes: string[],
  pages_added_7d: number,
  pages_changed_7d: number,
  pages_deleted_7d: number,
  default_tags: string[],
  visibility: "private"|"internal",
  connected_at: string,
  connected_by: string,
  error?: string                 // surface on non-ok status
}
```

### `GET /api/connections` ❌

**Response:** `{ items: Connection[], counts: { total, ok, warn, err } }`.
**Consumed by:** `connections.jsx::ConnectionsSection` (a Settings section, even though it lives in its own file).

### `POST /api/connections` ❌

**Body:** `{ kind, url, name, schedule, default_tags }`.
**Side effects:** for OAuth kinds, redirect into the OAuth flow first; on callback, persist the refresh token in the secret store and return the new Connection.

### `DELETE /api/connections/{id}` ❌

### `POST /api/connections/{id}/sync` ❌

**Response:** `{ ok: true, sync_id: string }`.
**Side effects:** trigger an out-of-schedule sync; emits a `confluence`/`sharepoint`/`url` activity event with outcome.

### `POST /api/connections/{id}/reconnect` ❌

Re-runs the OAuth dance, rotates the refresh token. Returns updated Connection.
**Consumed by:** `connections.jsx::ReconnectDialog`.

### `GET /api/connections/sync-history` ❌

**Query:** `?conn_id=&limit=`.
**Response:** `{ items: [{id, conn_id, at, outcome, summary, duration_ms}], next_cursor }`.
**Consumed by:** `connections.jsx` Recent syncs card.

---

## Domain: Knowledge Graph (KG tab — `graph.jsx`)

The KG tab is "view of LightRAG-extracted entities + relations", different from the Ontology Studio (which is the validated subset stewards curate). LightRAG already exposes `/graph/*` endpoints (see `api.jsx::OPENAPI_GROUPS::graph`); the maquette doesn't call them yet.

### Existing LightRAG endpoints to wire (no rewrite, just adopt the contract)

- `GET /graph/label/list` — list of entity labels for the type-filter rail.
- `GET /graph/label/popular?limit=` — for the "most mentioned" sidebar.
- `GET /graph/label/search?q=&limit=` — search box.
- `GET /graphs?label=&max_depth=&max_nodes=` — the subgraph fetch for the canvas.
- `GET /graph/entity/exists?name=`
- `POST /graph/entity/edit` — `{ entity_name, updated_data: { name?, type?, summary?, properties? } }`.
- `POST /graph/relation/edit` — analogous.
- `POST /graph/entity/create`
- `POST /graph/relation/create`

These are the contract the Twin gateway already proxies. The KG tab needs to call them with workspace scoping injected by the gateway.

### Twin-specific addition: tag co-occurrence

The KG tab filters entities by tag (`graph.jsx::activeTags`). The current LightRAG `/graphs` doesn't carry `tags` per entity. Either:
- Backend joins `:Entity` → `:Doc` (via `:MENTIONED_IN`) → tags, returns `tags[]` on each entity. Likely the right call given the tagging roadmap (`project_twin_rag_tagging_roadmap.md`).
- OR: separate `GET /api/graph/entities/{id}/tags` lazy fetch.

---

## Domain: Demo / dev affordances

The maquette has a "Reset demo data" button (✅ `POST /api/state/reset`) and a tweaks panel. None of these survive to production but they need to be air-gapped behind a feature flag, not removed.

### `POST /api/state/reset` ✅

Already wired. In prod: 404 unless `TWIN_DEMO_MODE=true`.

---

## Open questions to answer before sprint planning

1. **Ontology proposals storage.** Are they `:Proposal` nodes in the same Memgraph workspace? A separate `proposals.sqlite`? Output of a streaming pipeline (Kafka topic)? Today the maquette uses in-memory `window.__ontoInbox`. The DSEP pipeline runs dry by default — that result needs a home + a way for the UI to fetch it (and a TTL so stale proposals don't pile up).

2. **Désengagement (vector cascade on delete).** When a doc is deleted, who's responsible for the vector entries? Memgraph's `Vec_{workspace}_chunks` index doesn't auto-cascade from a node delete because vectors are property-attached. Spec needs to clarify: synchronous purge (slow at scale), async tombstone + sweep, or rebuild index nightly? The current bulk-delete UI promises "all chunks purged" — that promise needs an owner. Désengagement concern flagged at the 2026-05-22 exec demo.

3. **Audit trail location.** Is the audit a separate Memgraph store (`:AuditEvent` nodes), the existing SQLite `mutations` table promoted to prod, or piped into the future events middleware (`project_events_middleware.md`)? The UI assumes "queryable from the interface in <1s" (DocDetailPanel's audit tab). Decide before scaling.

4. **MyAccess ↔ workspace.id mapping discovery.** Per `project_twin_myaccess_rights_model`, ENTITY → workspace.id mapping is a Twin-owned referential that decouples from BNP org churn. Where does that table live? Who can edit it? The mapping is read on every request (auth path) so caching + invalidation is non-trivial.

5. **Conversation persistence target.** Per-user conversations need a store. Options: Memgraph (overkill, but consistent with the rest), PostgreSQL alongside (new infra), or LightRAG's own KV (`KV_{workspace}_conversations`). Recommend the latter — leverages the existing Memgraph pool, no new dependency.

6. **Notification push delivery.** SSE recommended above, but BNP infra reality (Caddy → FastAPI through gateway X-Forwarded-* + WAF buffering) may break long-lived connections. Verify before committing. Fallback: 30s poll on `/api/notifications?since=`.

7. **Pipeline status transport.** `GET /documents/pipeline_status` could be a snapshot endpoint OR a SSE stream. Snapshot is simpler; SSE matches the popover's "live state" feel. Given the popover is opened on demand and closed within seconds, snapshot with 2s client-side refresh is probably enough.

8. **Inbox proposal IDs.** The DSEP pipeline produces results without stable IDs. To support `POST /api/ontology/proposals/{id}/approve` we need persistent identity (so reload-and-approve doesn't reject because the id changed). Either hash the (extract-evidence + proposal-shape) or persist the dry-run results explicitly.

9. **Inconsistency: bell badge unread count.** `topbar.jsx::unreadCount` is computed locally from the notifications array, but the bell also surfaces governance items (doc-review, tag-request) that overlap with the per-tab pending sections. Two sources of truth (the notifications row count vs. the actual pending entities). Pick one: derive notifications from the pending entities server-side so they can't diverge.

10. **Inconsistency: `updated` vs `updated_at` on Document.** `documents.jsx` sorts/displays the human string `updated` ("2h ago"). Production needs ISO `updated_at`. Either add both server-side (breaks no one) or migrate the UI to format client-side. Choose now to avoid a flip mid-implementation.

11. **Inconsistency: `tag.id` vs `tag.tag` duplication.** Tag rows carry both `id` and `tag` (== same value) because the generic `(kind, id, data)` storage needs `id` but the UI code keys off `tag`. Picking a canonical field server-side (return only one) prevents the kind of `t.tag !== id && t.id !== id` defensive checks scattered through `app.jsx::mutateTag` / `deleteTag`.

12. **Inconsistency: ontology nodes carry layout coordinates (`x`, `y`).** Coords are steward-validated state and need persisting (else the canvas shuffles every load), BUT they're per-user-preference at the same time (different stewards may want different layouts). Decide: shared canonical layout (steward-edited, audited) OR per-user layout overlay on top of a default. The current maquette assumes shared.

13. **Inconsistency: documents.jsx pending workflow vs tags.jsx pending workflow.** Doc rejection writes `review.state = "rejected"` (doc retained); tag rejection writes `status = "rejected"` (and the row stays in the requested-tag list with that status). Two different shapes for the same concept. Align on one — recommend `review.state` since it cleanly separates lifecycle from content state.

14. **Workspace switching is wired in topbar but the data path doesn't exist.** `topbar.jsx::WorkspaceMenu` locks every non-current workspace (`is-locked`, tooltip "ships next sprint") with the comment "the exec sponsor would catch the lie immediately if a switch showed the same docs". Backend needs to actually serve scoped data per workspace before the switch is unlocked — coordinate with the MyAccess UID → workspace.id mapping discovery (open question 4).
