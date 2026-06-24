# Folder membership refactor — one document, many folders, stored once

> **Status:** spec / not started · **Opened:** 2026-06-24 · **Owner:** Julien
> **Intent:** make a folder a real *cloisonnement* (access/organization boundary) where a
> document can belong to **folder A AND folder B at the same time** without
> duplicating its bytes/chunks/vectors. Single physical copy, N logical memberships.

This file is **self-contained**: it carries the problem, the code evidence, the target
model, the behavioural contract, the migration, and the test bar. A cold reader (human or
agent) should be able to execute the chantier from this file alone.

---

## 0. Doctrine correction (this supersedes the repo's current framing)

`CLAUDE.md` currently says, in the **Folders** and **Vrai Graph** sections, that *"Twin
folders currently drive UX/document filtering, not separate graph labels"* and that the
graph/storage is *"scoped to the single LightRAG/Memgraph workspace"*. **That framing is
retired by this chantier.**

- **New doctrine:** a **folder is a cloisonnement** — an access/organization scope. A
  document is a **member of** one or more folders via a relation, not a single-valued
  property. The physical data (content, chunks, vectors, KV) is stored **once**,
  deduplicated by content hash; folders are membership + access, not copies.
- This is the direct application of the existing user doctrine
  **`feedback_relational_over_property`** ("on Memgraph, default to an auxiliary node +
  relation for provenance/RBAC/lineage, not a dict-property on hub nodes").
- It satisfies the Vihn requirement **"workspaces = cloisonnement vrai (≠ tags)"** while
  *not* paying for physical duplication.

When this lands, update `CLAUDE.md` (§ Folders, § Vrai Graph) and the memory
`project_workspace_as_memgraph_filter` accordingly (see §9).

---

## 0.5 Architect review & hardening (2026-06-24)

An architecture review confirmed the thesis is correct and the code evidence is
accurate, but flagged that the first draft **under-stated document identity** and
**under-scoped** the work. The following decisions HARDEN the plan and **override**
the looser phrasing further down where they conflict:

- **H1 — Do NOT change `doc_id` in this chantier.** The draft implied
  `doc_id = content_hash` as if it were LightRAG's natural state. It is not: for
  `/documents/upload`, LightRAG seeds `doc_id` from the **canonicalized file_path**
  and detects the *content* duplicate later (`existing_doc_id` lookup +
  `metadata.is_duplicate`). Making `doc_id = content_hash` the public identity is a
  deep, breaking change (ids, `/documents/{id}`, chunks, `source_docs`, frontend
  caches, deletes). **Keep LightRAG's existing `doc_id` as the public identity;
  treat `content_hash` only as a *dedup KEY* (an indexed property) used to find the
  doc to attach a membership to.** Migration of identity is explicitly out of scope.
- **H2 — DocStatus IS the logical Document (batch 1).** Today
  `DocStatus_{workspace}` already carries status + source + folder + chunks_list +
  hash. Attach `MEMBER_OF` to the **existing DocStatus node**; do **not** introduce a
  separate `:Document` label in the first batch (defer that decision). `(:Folder)`
  nodes are new; the doc node is the DocStatus node.
- **H3 — Scope is broader than docstatus + upload.** True cloisonnement also
  requires folder-scoping the **query/retrieval path** (sources/chunks/answers, not
  just the doc list), **tags**, the **graph `source_docs`** projection, and the
  **frontend caches** — and crucially the **native delete**: `rag.adelete_by_doc_id`
  (LightRAG `document_routes` ≈line 2252) physically deletes, so the ref-count MUST
  gate it (remove edge first; call `adelete_by_doc_id` only when refcount = 0).
- **H4 — Explicit membership endpoints are the PRIMARY contract** (`POST/DELETE
  /twin/api/documents/{doc_id}/folders`). The "re-upload same content → add
  membership" path is **secondary compat**, not the main contract — partly because
  the same-filename 409 fires *before* the content is read, so converting it to a
  membership-add would require hashing pre-check; the explicit endpoint is cleaner.

**Revised, de-risked execution order (supersedes §10):**
1. `MEMBER_OF` on existing DocStatus nodes, **without touching `doc_id`**; `content_hash` as an indexed dedup key.
2. Explicit membership endpoints **first**.
3. Delete ref-count (gate `adelete_by_doc_id`) **before** wiring the WebUI.
4. Folder-scope the query/retrieval + graph `source_docs` + tags + caches.
5. Upload duplicate→membership as secondary convenience.
6. WebUI "Add to folder" + MSW model (membership + refcount delete).
7. Update doctrine/memory (§9). Defer any `doc_id = content_hash` identity migration to a separate, later chantier.

---

## 1. Current behaviour (the problem) — confirmed in code

### 1.1 Folder is a single-valued property, not a relation

- Storage labels are **per-deployment**, derived from `resolve_workspace()` (env
  `MEMGRAPH_WORKSPACE` / `WORKSPACE` / `TWIN_DEFAULT_FOLDER`), **not** per-folder:
  - `kv_impl.py:35` → `KV_{workspace}_{namespace}`
  - `vector_impl.py` → `Vec_{workspace}_{namespace}`
  - `docstatus_impl.py:42` → `DocStatus_{workspace}`
- The folder is a **property** on the DocStatus node, resolved at write time from the
  `X-Twin-Folder` header via the ingestion context:
  - `docstatus_impl.py:_resolve_folder_for_props` (≈72-88): `props.folder` → `metadata.folder`
    → `get_active_storage_folder()`.
  - `docstatus_impl.py:193` → `SET n.folder = row.folder` (single value).
  - `_constants.py`: `storage_folder_context()` / `get_active_storage_folder()` carry the
    active folder; `registry.py` binds it from the header (`_install_storage_folder_capture`)
    and propagates it across the BackgroundTasks boundary
    (`_patch_background_tasks_folder_context`).
- Reads filter on that single property (`docstatus_impl.py` `get_docs_paginated`,
  `WHERE n.folder = …`).

**Consequence:** a document node carries exactly one `folder`. It cannot be in two.

### 1.2 Dedup currently *rejects* the second add (it does not share)

The LightRAG upload route (`lightrag.api.routers.document_routes`, patched by us) dedups
on two layers — confirmed from the route's own docstring:

1. **Filename duplicate (synchronous):** same canonical filename in the shared `input_dir`
   → **HTTP 409** (*"Input directory already contains a file with the same name"* /
   *"Document storage already contains '…'"*). The upload is rejected.
   - This is the path the cached `find_existing_file_by_file_path` lookup feeds
     (`registry.py:_patch_upload_duplicate_lookup`, applied at server boot). `input_dir` is
     **shared** across folders → the 409 is global.
2. **Content duplicate (asynchronous):** different filename, identical content. The
   upload seeds its `doc_id` from the canonicalized file_path, then during processing
   LightRAG finds an existing doc with the same **content_hash** (the `existing_doc_id`
   lookup) → marks the attempt `metadata.is_duplicate=true`,
   `error_msg="Content already exists. Original doc_id: …"`, **not indexed**.
   *(Nuance:* the original's indexed content is untouched and keeps its folder, but a
   **duplicate FAILED DocStatus record** can remain — with its *own* captured folder.
   "Dropped" is exact for the physical index, not for the DocStatus trace.)
   *Note (H1):* `doc_id` here is **file_path-seeded**, not content-hash — see §0.5 H1.

**Net:** putting "the same document" in a second folder today is impossible — it is either
409'd (same name) or flagged duplicate and dropped (same content). Dedup = **rejection**,
not sharing.

---

## 2. Target model

```
(:DocStatus_{workspace} {                  ← the logical Document IS the DocStatus node (H2)
     doc_id,                               ←   LightRAG's existing id, UNCHANGED (H1)
     content_hash,                         ←   dedup KEY only (indexed), NOT the identity
     content_summary, status, chunks_list, classification, … })   ← ONE node, stored once
      │
      ├─[:MEMBER_OF { added_at, added_by }]→ (:Folder { id: "A" })
      └─[:MEMBER_OF { added_at, added_by }]→ (:Folder { id: "B" })

(:Chunk)/(:Vec)/(:KV)  keyed by chunk_id/doc_id within the one workspace  ← stored once
```

> **Identity (H1):** the node keeps LightRAG's existing `doc_id`. `content_hash` is an
> indexed property used **only** to recognise "same content already ingested" and attach
> a new membership — never as the public id. No id rename in this chantier.

- **Physical layer unchanged in spirit:** content, chunks, vectors, KV remain
  deduplicated by hash and stored **once** in the single `{workspace}` namespace. We do
  **not** copy bytes/vectors per folder. (This is the "ne pas dupliquer les datas" goal.)
- **Membership layer is new:** `(:Document)-[:MEMBER_OF]->(:Folder)`, many-to-many.
  Adding a doc to a folder = `MERGE` one edge. Removing from a folder = delete one edge.
- **Folder nodes become first-class** (today folders live in env/`folder_store`, not as
  Memgraph nodes). A `(:Folder {id})` node per provisioned/runtime folder, in the
  workspace.
- **Cloisonnement = traversal + RBAC:** a folder query returns only docs reachable by
  `MEMBER_OF`; access is gated by which folders a user may traverse (MyAccess heritage).

### Why dedup stays (and is good)

Deduplicating content is *desirable* — it is exactly what avoids storing the same
document/vectors twice. The change is the **reaction** to a detected duplicate:
**dedup → share** (add a `MEMBER_OF` edge) instead of **dedup → reject** (409 / drop).

---

## 3. Behavioural contract (the spec to implement)

### 3.1 Upload / add-to-folder
- Upload of content C with `X-Twin-Folder: B`:
  - If no Document with `doc_id = hash(C)` exists → ingest normally, create the Document
    node, `MERGE (doc)-[:MEMBER_OF]->(Folder B)`.
  - If a Document with `doc_id = hash(C)` **already exists** (any folder) → **do NOT
    re-ingest** (keep dedup), instead **`MERGE (doc)-[:MEMBER_OF]->(Folder B)`** and return
    a success shape meaning *"added to folder"* (not 409, not `is_duplicate` failure).
  - Same canonical **filename** to a folder the doc is already a member of → idempotent
    no-op (or explicit "already in this folder").
- The synchronous filename-409 and the async content-duplicate paths in
  `document_routes` must be intercepted/translated by the Twin overlay so a duplicate
  becomes an **add-membership**, not a rejection. (Decide: shim the route vs. a Twin
  `/twin/api/documents/{id}/folders` add endpoint that the WebUI calls — see §6.)

### 3.2 Read / list / graph / **retrieval** (H3 — broader than the doc list)
- **Doc list:** `get_docs_paginated` + `get_status_counts` must filter by **membership
  traversal** (`MATCH (d)-[:MEMBER_OF]->(:Folder {id:X})`), not `n.folder = X`.
- **Retrieval/query (the load-bearing one):** real cloisonnement means a query issued
  *in folder X* must only ground on **sources/chunks/answers from docs that are
  `MEMBER_OF` X** — not just filter the doc list. LightRAG's retrieval is **not**
  naturally folder-scoped; the Twin query overlay (`server/twin_query_routes.py`, the
  intelligence L2/L3 path) must constrain candidate chunks/sources to the active
  folder's membership. **This is the part that makes folders a real boundary and the
  draft under-specified it.**
- **Tags:** doc-level tag propagation / tag views must respect membership (a tag view in
  folder X lists only member docs).
- **Graph (`source_docs`):** `server/graph_reader.py` builds `source_docs` from
  `DocStatus.chunks_list` and does **not** filter the graph by folder today. **Open
  decision (§8):** in a folder view, do we (a) hide entities whose `source_docs` are
  *not* members of the folder, or (b) keep the graph global and only filter the
  `source_docs` list per entity? (a) = stronger cloisonnement, heavier; (b) = cheaper,
  leaks entity existence across folders.
- A document `MEMBER_OF` A and B appears in both folder views — same node, one copy.

### 3.3 Delete (ref-counted) — **gate the native physical delete (H3)**
- "Delete from folder B" = remove the `(doc)-[:MEMBER_OF]->(B)` edge **only**.
- LightRAG's native delete (`rag.adelete_by_doc_id`, called from `document_routes`
  ≈line 2252) **physically removes** the doc + chunks + vectors. The Twin delete routes
  (`server/webui/routes_documents.py`, bulk-delete) MUST therefore: remove the membership
  edge first, and call `adelete_by_doc_id` **only when ref-count = 0** (no remaining
  `MEMBER_OF`). Calling it on a multi-folder doc would nuke the data out from under the
  other folders.
- The physical delete still cascades graph nodes/edges and invalidates the WebUI
  graph/query caches exactly like the current cascade (`server/graph_reader.py`, the
  MSW/e2e cascade contract — the four sensitive axes).
- A bulk "delete everywhere" remains available (drop all memberships → ref-count 0 →
  physical delete).

### 3.4 RBAC / cloisonnement
- A user may only see/add/remove memberships for folders in their MyAccess scope
  (existing `server/idp_jwt.py` + `server/folder.py:resolve_folder_for_request` two-tier
  model). Cross-folder sharing is itself a permissioned action (who may add a doc to a
  folder they can write).

### 3.5 Classification interaction (decide)
- Classification (MIP) is currently a **per-document** property set at ingestion (incl. the
  new operator `X-Twin-Classification` floor policy). With sharing, the classification
  lives on the shared Document node → **same class in every folder**. If a folder needs a
  *different* sensitivity for the same content, classification must move onto the
  `MEMBER_OF` edge (per-membership) — **open decision** (§8).

---

## 4. Code impact map (where the work lands)

| Area | File(s) | Change |
|---|---|---|
| Folder context | `_constants.py` (`storage_folder_context`, `get_active_storage_folder`) | keep — still carries the *target* folder for an add |
| Membership write | `docstatus_impl.py` (`_resolve_folder_for_props` ≈72-88, `_serialize_status` ≈200, `SET n.folder` ≈193) | replace single `folder` property with `MERGE (:Folder)` + `MERGE [:MEMBER_OF]` |
| Reads (doc list) | `docstatus_impl.py` (`get_docs_paginated`, `get_status_counts`, `WHERE n.folder …`) | filter by `MEMBER_OF` traversal |
| **Retrieval/query** | `server/twin_query_routes.py`, `intelligence/` L2/L3 path | constrain candidate chunks/sources to the folder's membership (the real boundary) |
| Tags | tag views / propagation | tag listings respect membership |
| Folder nodes | `_folders.py`, `server/folder.py`, `server/folder_store.py` | materialize provisioned/runtime folders as `(:Folder)` nodes; CRUD also manages nodes |
| Upload dedup → share | `patches/registry.py` (`_patch_upload_duplicate_lookup` and the upload route shim), LightRAG `document_routes` 409/duplicate paths | translate "duplicate" into "add membership" |
| Graph scoping | `server/graph_reader.py` | graph reads/cascade scoped by folder membership where relevant |
| Delete ref-count | `docstatus_impl.py` delete paths + `server/graph_reader.py` cascade + `server/webui/routes_documents.py` | remove edge; physical delete only at refcount 0 |
| WebUI | `lightrag_webui_twin/` (documents store, folder switch, an "add to folder" affordance), `src/mocks/handlers.ts` | membership in the contract; MSW must model add-membership + refcount delete |

---

## 5. Migration (existing data)

1. For every existing `DocStatus_{workspace}` node with a `folder` property:
   `MERGE (:Folder {id: n.folder})` and `MERGE (n)-[:MEMBER_OF]->(that folder)`.
2. Keep the `folder` property written for one release (dual-write) so a rollback is
   possible; reads switch to membership behind a flag; drop the property once verified.
3. Backfill `(:Folder)` nodes from the env catalog (`TWIN_FOLDERS_JSON`) +
   `folder_store` runtime folders, so every reachable folder exists as a node.
4. Idempotent, re-runnable migration script (per the global directive on idempotent
   install scripts). Dry-run + count report before commit.

---

## 6. Open API surface decision

Two ways to expose "this document is also in folder B":
- **(a) Implicit via upload** — re-uploading the same content with `X-Twin-Folder: B`
  adds membership (the dedup→share translation). Lowest UI change; reuses the upload flow.
- **(b) Explicit membership endpoints** — `POST /twin/api/documents/{doc_id}/folders`
  `{folder_id}` to add, `DELETE …/{folder_id}` to remove. Cleaner contract, lets the WebUI
  offer "Add to folder…" without a re-upload, and makes RBAC explicit.

**Recommendation:** ship (b) as the real contract, keep (a) working as a convenience
(upload of an already-known content just adds membership). Decide before implementing.

---

## 7. Test bar (non-negotiable — repo doctrine)

Follow `docs/test-doctrine-graph.md` and `docs/test-doctrine-lightrag-compat.md`:

- **Membership contract (e2e + backend):** same content uploaded to A then B → **one**
  physical Document, visible in both folder views; `MEMBER_OF` to both; vectors/chunks
  stored once (assert no second copy).
- **Read isolation:** folder A view lists the doc; folder C (no membership) does not.
- **Delete ref-count:** remove from B → still in A, physical data intact; remove from A
  (last) → physical + graph cascade + cache invalidation (the four sensitive axes of
  `test-doctrine-graph.md`: front cache keys, no inspector fallback, folder binding,
  `source_docs`).
- **LightRAG-compat:** with the overlay off / single folder, the native LightRAG path
  behaves identically (a duplicate still dedups; nothing regresses).
- **MSW fidelity:** the WebUI mock must model add-membership and refcount-delete, or e2e
  goes falsely green (same trap we just fixed for graph/api-key persistence).
- **RBAC:** a user without folder B cannot add/see the doc there.

---

## 8. Open decisions (resolve before coding)

1. **Per-membership metadata?** Does classification / tags / "approved" status need to be
   per-folder (on `MEMBER_OF`) or stay per-document (shared)? (§3.5). Default proposal:
   classification stays per-document (content-derived); approval/tags can be per-membership
   if the product needs folder-local lifecycle.
2. **API surface:** (a) implicit-upload vs (b) explicit membership endpoints (§6).
3. **Folder = workspace label, ever?** Do we keep a single physical `{workspace}` namespace
   forever (shared dedup) — yes, that is the whole point — or is there any case requiring a
   physically isolated folder (e.g. a regulatory hard wall) that needs a separate label?
   If yes, that is a *separate* "hard-isolated folder" tier, not this model.
4. **Cross-folder add = permissioned?** Who may add an existing doc to another folder
   (source-folder writer? target-folder writer? both?).
5. **Separate `:Document` label later?** Batch 1 keeps the DocStatus node as the logical
   document (H2). Do we eventually split a clean `:Document` (identity + membership) from
   `:DocStatus` (LightRAG ingestion-status compat), or keep them fused? Deferred.
6. **Graph folder-view semantics** (§3.2): hide non-member-sourced entities (strong
   cloisonnement, heavier) vs global graph with per-entity filtered `source_docs`
   (cheaper, leaks entity existence). Decide with the PO.

> Resolved by §0.5: API surface → explicit membership endpoints primary (H4);
> `doc_id = content_hash` identity migration → **deferred, out of scope** (H1);
> doc node = DocStatus node for batch 1 (H2).

---

## 9. Doctrine / memory to update when this lands

- `CLAUDE.md` § **Folders** and § **Vrai Graph**: replace "folders drive UX/filtering, not
  isolation / single workspace" with the membership model (folder = relational cloisonnement,
  data stored once, `MEMBER_OF`).
- Memory `project_workspace_as_memgraph_filter`: amend — workspace label stays the physical
  namespace; **folder = membership relation on top of it**, many-to-many.
- Cross-link `feedback_relational_over_property` (this chantier is its canonical application).

---

## 10. Suggested execution order

> **Superseded by the de-risked order in §0.5** (architect review). Kept here for the
> fuller breakdown; follow §0.5's sequencing on identity/delete/retrieval.

1. Resolve §8 open decisions with the PO.
2. Materialize `(:Folder)` nodes + the migration script (dual-write, idempotent, dry-run).
3. Write path: `MEMBER_OF` on ingestion + the dedup→share translation in the upload shim.
4. Read path: membership-scoped queries (`get_docs_paginated`, graph reads).
5. Explicit membership endpoints (§6b) + WebUI "Add to folder" + MSW model.
6. Delete ref-count + cascade + cache invalidation.
7. RBAC gating on membership operations.
8. Flip reads to membership behind a flag, verify, drop the legacy `folder` property.
9. Update doctrine + memory (§9).
