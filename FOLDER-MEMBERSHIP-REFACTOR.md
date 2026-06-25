# Folder membership refactor — one document, many folders, stored once

> **Status:** core landed · **Opened:** 2026-06-24 · **Owner:** Julien
> **Intent:** a folder is a real *cloisonnement* (access/organization boundary). A
> document can belong to **folder A AND folder B at the same time** without duplicating
> its bytes/chunks/vectors — single physical copy, N logical memberships.

This file is **non-accumulative**: it carries only the as-built baseline (compressed,
so a cold reader knows the membership model is the *current reality*, not a proposal)
and **what still remains**. Done work is removed, not archived here — read the commits.

---

## A. Landed baseline (as-built — do NOT re-do)

The membership model is implemented and merged. A document is the `DocStatus_{workspace}`
node (LightRAG's `doc_id` unchanged; `content_hash` is an indexed dedup key only). Folder
membership is a relation, stored once:

```
(:DocStatus_{workspace} { doc_id, content_hash, status, chunks_list, classification, … })
   ├─[:MEMBER_OF]→ (:Folder_{workspace} { id: "A" })
   └─[:MEMBER_OF]→ (:Folder_{workspace} { id: "B" })
```

What is live (commit refs are the source of truth, not this list):

- **Membership write + read.** Ingestion `MERGE`s `(:Folder)` + `[:MEMBER_OF]`; doc list,
  status counts, and folder-scoped reads traverse `MEMBER_OF`, not `n.folder = X`.
  Boot-time idempotent backfills (`docstatus_impl.py` `_backfill_missing_folders` →
  `_backfill_membership`) guarantee **every legacy doc ends up `MEMBER_OF` its folder**, so
  reads are membership-authoritative.
- **Explicit membership endpoints are the primary contract** (admin-gated):
  `GET/POST /twin/api/documents/{id}/folders`, `DELETE …/{id}/folders/{folder_id}`
  (`server/webui/routes_documents.py`). The API returns `{physically_deleted,
  remaining_folders}`.
- **Ref-counted delete.** `DELETE …/folders/{folder_id}` removes one edge; when it was the
  **last** membership it physically deletes (`_delete_doc_from_rag`), under a per-doc lock,
  with the edge still present so a failed delete is recoverable.
- **Folder-scoped retrieval & views** (the part that makes folders a *real* boundary):
  query/retrieval grounds only on member chunks (vector + KG expansion), graph
  entities/relations/`source_docs` are membership-filtered with mixed-provenance masking,
  tags + counters are membership-aware, and frontend caches are folder-scoped.
- **Dedup → share, not reject.** Re-uploading already-known content into a folder adds a
  membership instead of 409/`is_duplicate`-drop. Three read-shaped dedup hooks
  (`get_doc_by_file_path` / `_file_basename` / `_content_hash`) share into the active
  folder, gated by an **ingestion-only** context (`duplicate_share_folder_context`) so a
  query never mutates membership. Content-dup `upsert` (LightRAG 1.4.11/1.4.12
  `metadata.is_duplicate`) shares the original instead of writing a visible `dup-*` node;
  native behaviour preserved when no folder context is active.
- **WebUI "Add to folder"** (admin-only via `canManageFolders` → `admin:folders` scope):
  share dialog with explicit hard-delete confirmation on the last-folder case + MSW model.
  *(In CI on `feat/folder-add-to-folder-ui` at time of writing; treat as landing.)*

**Resolved decisions** (were open in the draft, now settled by the implementation):
API surface → explicit endpoints primary, upload-dedup secondary convenience ·
graph folder-view → hide non-member-sourced entities (strong cloisonnement) + mask mixed ·
tags → per-folder by label (`WebuiTag_{folder}`) · `doc_id` unchanged, `content_hash` =
dedup key only.

---

## B. What remains

### B1 — Doctrine / memory update *(only clean task left; doc-only, parallel-safe with CI)*
Apply the doctrine correction that this chantier earned:
- `CLAUDE.md` § **Folders** and § **Vrai Graph** — the line *"Twin folders currently drive
  UX/document filtering, not separate graph labels"* is now **false**. Folders scope
  retrieval (chunks + KG via `MEMBER_OF`), the graph view (`source_docs` membership-filtered
  + mixed masking), and tags/counters/caches. Replace with the membership model: folder =
  relational cloisonnement, data stored once, `MEMBER_OF`; the `{workspace}` label remains
  the single physical namespace, folders are a relation on top of it.
- Memory `project_workspace_as_memgraph_filter`: amend — workspace label = physical
  namespace; **folder = membership relation, many-to-many**.
- Cross-link `feedback_relational_over_property` (this is its canonical application).

### B2 — Drop the legacy `folder` dual-write *(deferred cleanup — gated, not urgent)*
Ingestion still dual-writes `SET n.folder = fid` next to `MEMBER_OF`
(`docstatus_impl.py` ≈461-464) as a migration safety net. **Do not remove it until** the
backfill is proven complete on every live deployment (a doc with no `MEMBER_OF` edge would
become invisible to tagging + uncounted). Removal is a deliberate batch: prove
completeness → flip any remaining `n.folder` readers to membership → drop the property.

---

## C. Still-open decisions (need PO / MyAccess, not codeable yet)

1. **Per-membership lifecycle?** Classification (MIP) and approve/reject status live on the
   **shared** DocStatus node → same value in every folder. Tags already went per-folder
   (by label). Open: does *approval/lifecycle* (and possibly a folder-local sensitivity
   floor) need to move onto the `MEMBER_OF` edge? Default: keep classification
   per-document (content-derived); revisit approval only if the product needs folder-local
   lifecycle.
2. **Cross-folder add = whose right?** Today it is **admin-gated interim** (`require_admin_user`,
   `admin:folders` scope) — explicitly *until per-user source-doc + target-folder RBAC lands
   in MyAccess* (the real model: source-folder writer vs target-folder writer). Wiring that
   per-user RBAC is the transmuted form of this decision; it belongs to the MyAccess
   integration, not this chantier.
3. **Hard-isolated folder tier?** The shared `{workspace}` namespace (shared dedup) is the
   whole point. If a regulatory hard-wall ever needs a *physically* isolated folder
   (separate label, no shared bytes), that is a **separate tier**, not this model.
4. **Split a clean `:Document` label later?** Batch 1 fused logical doc = DocStatus node.
   Whether to eventually separate `:Document` (identity + membership) from `:DocStatus`
   (LightRAG ingestion-status compat) is deferred.
5. **`doc_id = content_hash` identity migration** — explicitly **out of scope / separate
   later chantier** (changing the public id touches `/documents/{id}`, chunks,
   `source_docs`, caches, deletes). `content_hash` stays a dedup key only.

---

## D. Test bar (met — keep it met on any follow-up)

Per `docs/test-doctrine-graph.md` + `docs/test-doctrine-lightrag-compat.md`, already
covered by the landed batches and **required for B2 and any §C work**:

- Membership contract: same content into A then B → one physical node, visible in both,
  one set of chunks/vectors.
- Read isolation: member folder lists the doc; non-member folder does not.
- Delete ref-count: remove from B → still in A, data intact; remove last → physical +
  graph cascade + cache invalidation (the four sensitive axes).
- LightRAG-compat: overlay off / no folder context → native path identical (duplicate
  still dedups; `dup-*` node created natively). Verified on Memgraph **3.9.0 + 3.10.1**
  (the 3.10+ stale vector-index-on-delete footgun is a real CI gate).
- MSW fidelity: the mock models add-membership + refcount-delete, else e2e goes falsely
  green.
- RBAC: a non-admin (no `admin:folders`) cannot add/see/remove memberships.
