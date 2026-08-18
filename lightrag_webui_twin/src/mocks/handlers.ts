/**
 * MSW request handlers — back the API routes with the in-repo fixtures.
 *
 * Path layout mirrors the production split (Étape 0 sprint 2026-05-29):
 *   - LightRAG-native paths: `/documents`, `/documents/:id/chunks`,
 *     `/health`, `/openapi`, `/pipeline_status`.
 *   - Twin overlay paths: every endpoint under `${ANY}/twin/api/*`.
 *
 * Filtering mirrors what the tab implementations do locally on the fixtures so
 * tab-side state behaves identically whether data comes from MSW or from a
 * future real backend.
 */

import { http, HttpResponse } from 'msw';
import {
  ACTIVITY_FIXTURES,
  ACTIVITY_NOW_MS,
  API_VERSION,
  DOC_TO_GRAPH_ENTITIES,
  DOCUMENT_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_ENTITY_DOCS,
  GRAPH_RELATION_FIXTURES,
  NOTIFICATION_FIXTURES,
  OPENAPI_GROUPS,
  PROCEDURE_BUNDLE_FIXTURES,
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  FOLDER_FIXTURES,
} from '../fixtures';
import { ACTIVITY_RANGE_MS, type ActivityEvent } from '../types/activity';
import type { Document, DocumentStatus } from '../types/document';
import {
  bundleFolders,
  type ProcedureBundle,
  type ProcedureBundleSummary,
} from '../types/procedure';
import type { GraphEntity, GraphRelation } from '../types/graph';
import type { Notification } from '../types/topbar';
import type { TagCategory, TagEntry } from '../types/tag';

const ANY = '*';
const TWIN = '/twin/api';

/** BNP MIP C-code → business name. Mirrors the backend's operator-set
 *  classification payload so e2e/unit tests can assert the wiring end to end. */
const MIP_CLASS_NAMES: Readonly<Record<string, string>> = {
  C1: 'Public',
  C2: 'Internal',
  C3: 'Confidential',
  C4: 'Secret',
};
const OPERATOR_UPLOAD_CLASSES = new Set(['C1', 'C2']);

function uploadHeaderError(
  operatorClass: string | null,
  docType: string | null,
): string | null {
  if (operatorClass && !OPERATOR_UPLOAD_CLASSES.has(operatorClass)) {
    return 'X-Twin-Classification accepts only C1 or C2; C3/C4 uploads are rejected by policy.';
  }
  if (docType && docType !== 'procedure' && docType !== 'standard') {
    return "X-Twin-Doc-Type accepts only 'procedure' or 'standard' (omit for auto-detect).";
  }
  return null;
}

const E2E_DOCUMENTS_STORAGE_KEY = 'twin.e2e.documentsState.v1';
const E2E_TAG_CATEGORIES_STORAGE_KEY = 'twin.e2e.tagCategoriesState.v1';
const E2E_TAGS_STORAGE_KEY = 'twin.e2e.tagsState.v1';
const E2E_NOTIFICATIONS_STORAGE_KEY = 'twin.e2e.notificationsState.v1';
const E2E_ACTIVITY_STORAGE_KEY = 'twin.e2e.activityState.v1';
const E2E_SCENARIO_STORAGE_KEY = 'twin.e2e.scenario.v1';
const E2E_AUTH_USER_STORAGE_KEY = 'twin.e2e.localAuthUser.v1';
const E2E_GRAPH_ENTITIES_STORAGE_KEY = 'twin.e2e.graphEntitiesState.v1';
const E2E_GRAPH_RELATIONS_STORAGE_KEY = 'twin.e2e.graphRelationsState.v1';
const E2E_API_KEYS_STORAGE_KEY = 'twin.e2e.apiKeysState.v1';
const E2E_PROCEDURES_STORAGE_KEY = 'twin.e2e.proceduresState.v1';
const E2E_VISION_SETTINGS_STORAGE_KEY = 'twin.e2e.visionSettingsState.v1';

function cloneDocuments(docs: readonly Document[]): Document[] {
  return docs.map((doc) => ({
    ...doc,
    tags: [...doc.tags],
    metadata:
      doc.metadata && typeof doc.metadata === 'object'
        ? { ...doc.metadata }
        : doc.metadata,
    review:
      doc.review && typeof doc.review === 'object'
        ? { ...doc.review }
        : doc.review,
  }));
}

function trackStatusResponse(trackId: string, docId: string | undefined) {
  if (e2eScenario.trackStatusMode === 'processed' && docId) {
    const doc = documentsState.find((d) => d.doc_id === docId);
    return HttpResponse.json({
      track_id: trackId,
      documents: doc
        ? [{ id: doc.doc_id, status: 'processed', file_path: doc.file_path }]
        : [],
      total_count: doc ? 1 : 0,
      status_summary: doc ? { processed: 1 } : {},
    });
  }
  if (e2eScenario.trackStatusMode === 'timeout') {
    const doc = documentsState.find((d) => d.doc_id === docId);
    return HttpResponse.json({
      track_id: trackId,
      documents: docId
        ? [{ id: docId, status: 'processing', file_path: doc?.file_path ?? docId }]
        : [],
      total_count: docId ? 1 : 0,
      status_summary: docId ? { processing: 1 } : {},
    });
  }
  return HttpResponse.json({
    track_id: trackId,
    documents: [],
    total_count: 0,
    status_summary: {},
  });
}

function e2eStorage(): Storage | null {
  if (globalThis.window === undefined) return null;
  try {
    return globalThis.sessionStorage;
  } catch {
    return null;
  }
}

function loadDocumentsState(): Document[] {
  const raw = e2eStorage()?.getItem(E2E_DOCUMENTS_STORAGE_KEY);
  if (!raw) return cloneDocuments(DOCUMENT_FIXTURES);
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) return cloneDocuments(parsed as Document[]);
  } catch {
    // Ignore corrupt e2e state and fall back to fixtures.
  }
  return cloneDocuments(DOCUMENT_FIXTURES);
}

function persistDocumentsState(): void {
  e2eStorage()?.setItem(E2E_DOCUMENTS_STORAGE_KEY, JSON.stringify(documentsState));
}

function loadState<T>(
  key: string,
  fallback: readonly T[],
  clone: (items: readonly T[]) => T[],
): T[] {
  const raw = e2eStorage()?.getItem(key);
  if (!raw) return clone(fallback);
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) return clone(parsed as T[]);
  } catch {
    // Ignore corrupt e2e state and fall back to fixtures.
  }
  return clone(fallback);
}

function persistState(key: string, value: unknown): void {
  e2eStorage()?.setItem(key, JSON.stringify(value));
}

function cloneTagCategories(items: readonly TagCategory[]): TagCategory[] {
  return items.map((category) => ({ ...category }));
}

function cloneTags(items: readonly TagEntry[]): TagEntry[] {
  return items.map((tag) => ({
    ...tag,
    aliases: [...tag.aliases],
    deprecates: [...tag.deprecates],
    proposed_fields: tag.proposed_fields ? [...tag.proposed_fields] : [],
    related: tag.related.map((related) => ({ ...related })),
    examples: [...tag.examples],
    created: { ...tag.created },
    last_edit: { ...tag.last_edit },
  }));
}

function cloneNotifications(items: readonly Notification[]): Notification[] {
  return items.map((notification) => ({ ...notification }));
}

function cloneActivity(items: readonly ActivityEvent[]): ActivityEvent[] {
  return items.map((event) => ({
    ...event,
    actor: { ...event.actor },
    target: { ...event.target },
    meta: { ...event.meta },
  }));
}

function persistTagCategoriesState(): void {
  persistState(E2E_TAG_CATEGORIES_STORAGE_KEY, categoryState);
}

function persistTagState(): void {
  persistState(E2E_TAGS_STORAGE_KEY, tagState);
}

function persistNotificationState(): void {
  persistState(E2E_NOTIFICATIONS_STORAGE_KEY, notificationState);
}

function persistActivityState(): void {
  persistState(E2E_ACTIVITY_STORAGE_KEY, activityState);
}

function cloneGraphEntities(items: readonly GraphEntity[]): GraphEntity[] {
  return items.map((e) => ({
    ...e,
    tags: e.tags ? [...e.tags] : [],
    // Persisted entities already carry source_docs; fixtures get them
    // injected from the GRAPH_ENTITY_DOCS map.
    source_docs: e.source_docs
      ? [...e.source_docs]
      : [...(GRAPH_ENTITY_DOCS[e.id] ?? [])],
    properties: e.properties ? { ...e.properties } : {},
  }));
}

function cloneGraphRelations(items: readonly GraphRelation[]): GraphRelation[] {
  return items.map((r) => ({
    ...r,
    properties: r.properties ? { ...r.properties } : {},
  }));
}

/**
 * Graph mutations (rename/retype/tag, create, delete, doc-cascade) must
 * survive a page reload exactly like the documents/tags stores do — the real
 * backend writes them to Memgraph (durable, `SET n += $props`). Without this,
 * the graph mock alone re-seeded from fixtures on every reload, so the most
 * critical surface could never be tested for mutation persistence.
 */
function persistGraphState(): void {
  persistState(E2E_GRAPH_ENTITIES_STORAGE_KEY, graphEntityState);
  persistState(E2E_GRAPH_RELATIONS_STORAGE_KEY, graphRelationState);
}

/**
 * Mutable document store. Initialized from DOCUMENT_FIXTURES at module load,
 * then mutated by approve / reject / delete handlers so the UI sees state
 * changes when the host re-fetches via useDocuments() invalidation.
 *
 * Reset via `resetDocumentsState()` between tests so suites don't pollute
 * each other.
 */
let documentsState: Document[] = loadDocumentsState();
let documentMemberships: Record<string, string[]> = Object.fromEntries(
  documentsState.map((doc) => [doc.doc_id, [doc.folder || 'default']]),
);
let categoryState: TagCategory[] = loadState(
  E2E_TAG_CATEGORIES_STORAGE_KEY,
  TAG_CATEGORY_FIXTURES,
  cloneTagCategories,
);
let tagState: TagEntry[] = loadState(E2E_TAGS_STORAGE_KEY, TAG_FIXTURES, cloneTags);
let notificationState: Notification[] = loadState(
  E2E_NOTIFICATIONS_STORAGE_KEY,
  NOTIFICATION_FIXTURES,
  cloneNotifications,
);
let activityState: ActivityEvent[] = loadState(
  E2E_ACTIVITY_STORAGE_KEY,
  ACTIVITY_FIXTURES,
  cloneActivity,
);
let graphEntityState: GraphEntity[] = loadState(
  E2E_GRAPH_ENTITIES_STORAGE_KEY,
  GRAPH_ENTITY_FIXTURES,
  cloneGraphEntities,
);
let graphRelationState: GraphRelation[] = loadState(
  E2E_GRAPH_RELATIONS_STORAGE_KEY,
  GRAPH_RELATION_FIXTURES,
  cloneGraphRelations,
);

/**
 * Mirror the real backend cascade: an entity sourced *only* by docs in
 * the deletion set vanishes from `graphEntityState`, and any relation
 * touching such an entity disappears too. The reverse map is curated
 * in `fixtures/graph.ts` to avoid fragile file_path matching.
 * Shared between the unit `DELETE /documents/{id}` shim and the bulk
 * `POST /documents/bulk-delete` Twin route so both code paths trigger
 * the same Graph-tab refetch behaviour.
 */
function cascadeDocsFromGraph(deletedDocIds: Set<string>): void {
  if (deletedDocIds.size === 0) return;
  const orphanEntityIds = new Set<string>();
  for (const docId of deletedDocIds) {
    for (const entityId of DOC_TO_GRAPH_ENTITIES[docId] ?? []) {
      const stillSourced = Object.entries(DOC_TO_GRAPH_ENTITIES).some(
        ([otherDoc, ents]) =>
          !deletedDocIds.has(otherDoc) && ents.includes(entityId),
      );
      if (!stillSourced) orphanEntityIds.add(entityId);
    }
  }
  if (orphanEntityIds.size === 0) return;
  graphEntityState = graphEntityState.filter(
    (e) => !orphanEntityIds.has(e.id),
  );
  graphRelationState = graphRelationState.filter(
    (r) => !orphanEntityIds.has(r.source) && !orphanEntityIds.has(r.target),
  );
  persistGraphState();
}

let uploadSeq = 0;
const uploadedTrackDocs = new Map<string, string>();
const uploadedDocText = new Map<string, string>();

/**
 * Async-ingestion marker (audit 2026-07-02, DUP-2a). The real upload path is
 * asynchronous: `POST /documents/upload` only ENQUEUES (LightRAG
 * `document_routes.py` → `apipeline_enqueue_documents`, returning
 * `{status, message, track_id}`) and the DocStatus row starts PENDING, then
 * the pipeline flips it to PROCESSED. The mock mirrors that: an uploaded doc
 * carries this metadata counter and transitions PENDING → PROCESSED
 * deterministically on the SECOND `GET /documents` after the upload (no
 * timers, no randomness). Persisted inside doc.metadata so the state machine
 * survives an e2e page reload like the sessionStorage-backed stores do.
 */
const INGEST_POLLS_KEY = 'e2e_ingest_polls_remaining';

function advanceMockIngestion(): void {
  let changed = false;
  documentsState = documentsState.map((doc) => {
    const remaining = doc.metadata?.[INGEST_POLLS_KEY];
    if (typeof remaining !== 'number') return doc;
    changed = true;
    if (remaining > 0) {
      return {
        ...doc,
        metadata: { ...doc.metadata, [INGEST_POLLS_KEY]: remaining - 1 },
      };
    }
    const metadata = { ...doc.metadata };
    delete metadata[INGEST_POLLS_KEY];
    return {
      ...doc,
      status: 'PROCESSED',
      chunks_count: 1,
      metadata,
    };
  });
  if (changed) persistDocumentsState();
}

/**
 * Active-folder resolution for the delete surfaces. The real backend binds
 * the folder from the `X-Twin-Folder` header (`server/folder.py`
 * `resolve_folder_for_request`, falling back to the catalog default). The
 * mock falls back to the doc's own primary folder so raw `fetch()` tests
 * without the header keep the historical single-folder semantics.
 */
function activeFolderFor(request: Request, doc: Document): string {
  return request.headers.get('X-Twin-Folder') ?? doc.folder ?? 'default';
}

/**
 * Ref-counted delete shared by the three delete surfaces (audit 2026-07-02,
 * DUP-2b), mirroring the real backend: `native_shims._delete_or_unshare`
 * (native_shims.py:567-597) and `routes_documents._apply_membership_delete`
 * (routes_documents.py:306-329). Deleting from `activeFolder` only UN-SHARES
 * the doc there; the physical record disappears ONLY when that was its last
 * membership. Returns true on physical delete. The graph cascade is the
 * caller's job so bulk-delete can cascade the whole physically-deleted set in
 * one pass (shared-entity semantics).
 */
function removeDocFromFolderOrDelete(doc: Document, activeFolder: string): boolean {
  const folders = foldersForDoc(doc);
  if (folders.length === 1 && folders[0] === activeFolder) {
    documentsState = documentsState.filter((d) => d.doc_id !== doc.doc_id);
    delete documentMemberships[doc.doc_id];
    persistDocumentsState();
    return true;
  }
  documentMemberships[doc.doc_id] = folders.filter(
    (folder) => folder !== activeFolder,
  );
  return false;
}

/** One bulk-delete target that was actually processed (found + in-folder). */
interface BulkDeleteOutcome {
  doc: Document;
  physicallyDeleted: boolean;
  folder: string;
}

/**
 * Per-doc bulk-delete decision (mirrors
 * `routes_documents._delete_one_document`): resolve the doc, check the
 * active-folder membership, then apply the shared ref-counted delete.
 * Returns null when the target must be reported in `failed`.
 */
function applyBulkDeleteForDoc(
  request: Request,
  id: string,
): BulkDeleteOutcome | null {
  const doc = documentsState.find((d) => d.doc_id === id);
  if (!doc) return null;
  const active = activeFolderFor(request, doc);
  if (!foldersForDoc(doc).includes(active)) return null;
  const physicallyDeleted = removeDocFromFolderOrDelete(doc, active);
  return { doc, physicallyDeleted, folder: active };
}

/** Summary variants mirror routes_documents._emit_bulk_delete_activity. */
function bulkDeleteSummary(
  actor: string,
  affectedCount: number,
  physicalCount: number,
  unsharedCount: number,
  activeFolder: string,
  cascade: string,
): string {
  if (physicalCount && unsharedCount) {
    return (
      `Bulk delete by ${actor}: ${affectedCount} documents affected ` +
      `(${physicalCount} physically deleted with cascade, ` +
      `${unsharedCount} unshared from folder ${activeFolder})`
    );
  }
  if (physicalCount) {
    return `Bulk delete by ${actor}: ${affectedCount} documents physically deleted; ${cascade}`;
  }
  return (
    `Bulk delete by ${actor}: ${affectedCount} documents unshared ` +
    `from folder ${activeFolder}; no physical cascade`
  );
}

/** Activity-ledger shaping for a non-empty bulk-delete batch. */
function recordBulkDeleteActivity(
  rawActor: string | undefined,
  affected: readonly BulkDeleteOutcome[],
  failed: readonly string[],
): void {
  const actor = rawActor ?? 'operator.demo';
  const docIds = affected.map((entry) => entry.doc.doc_id);
  const physicalCount = affected.filter((e) => e.physicallyDeleted).length;
  const unsharedCount = affected.length - physicalCount;
  const activeFolder = affected[0].folder;
  const cascade =
    'physical delete cascades document data, chunks, vectors and graph links';
  recordActivity({
    id: `evt_doc_bulk_delete_${Date.now()}`,
    ts: new Date().toISOString(),
    rel: 'now',
    day: 'Today',
    kind: physicalCount ? 'doc-deleted' : 'doc-folder-removed',
    sev: 'info',
    actor: { user: actor, role: 'KB Steward' },
    target:
      affected.length === 1
        ? {
            type: 'document',
            label: affected[0].doc.file_path,
            id: affected[0].doc.doc_id,
          }
        : { type: 'bulk', label: `${affected.length} documents` },
    summary: bulkDeleteSummary(
      actor,
      affected.length,
      physicalCount,
      unsharedCount,
      activeFolder,
      cascade,
    ),
    meta: {
      operation: 'bulk-delete',
      folder: activeFolder,
      doc_count: affected.length,
      doc_ids: docIds,
      ...(affected.length === 1 ? { doc_id: docIds[0] } : {}),
      failed,
      failed_count: failed.length,
      physically_deleted_count: physicalCount,
      unshared_count: unsharedCount,
      cascade,
    },
  });
}

// Folders — admin CRUD mirror of the backend. The first folder fixture
// is treated as the SRE-provisioned default (env-seeded) and
// rejects mutations with 403; the remaining ones are seeded as
// runtime entries that an operator could realistically add. FOLDER_MAX
// matches the backend's clamp so the "at max" path is reachable in
// tests without re-seeding.
const FOLDER_MAX = 5;
let folderState = FOLDER_FIXTURES.slice(0, 1).map((w) => ({ ...w }));
const envSeededFolderIds = new Set(
  FOLDER_FIXTURES.slice(0, 1).map((w) => w.id),
);

// API keys — per-operator credentials minted via Settings → API keys.
// In-memory only; tests reset via resetDocumentsState() below.
interface ApiKeyMockEntry {
  id: string;
  name: string;
  prefix: string;
  full_value: string;
  created_at: number;
  created_by: string;
  last_used_at: number | null;
  revoked_at: number | null;
}
// API keys persist across reload like the real (Memgraph-backed, durable)
// api_key_store: a revoked key stays revoked in the audit trail.
let apiKeyState: ApiKeyMockEntry[] = loadState<ApiKeyMockEntry>(
  E2E_API_KEYS_STORAGE_KEY,
  [],
  (items) => items.map((e) => ({ ...e })),
);
let apiKeyCounter = apiKeyState.length;

function persistApiKeyState(): void {
  persistState(E2E_API_KEYS_STORAGE_KEY, apiKeyState);
}

function resetDocumentMemberships(): void {
  documentMemberships = Object.fromEntries(
    documentsState.map((doc) => [doc.doc_id, [doc.folder || 'default']]),
  );
}

function foldersForDoc(doc: Document): string[] {
  return documentMemberships[doc.doc_id] ?? [doc.folder || 'default'];
}

function isKnownMockFolder(folderId: string): boolean {
  return (
    folderState.some((folder) => folder.id === folderId) ||
    FOLDER_FIXTURES.some((folder) => folder.id === folderId)
  );
}

// Instance storage quota — in-memory snapshot tests inject via the
// scenario knob ``setMockQuotaState`` exported below. Defaults: the
// snapshot is "configured" with a 2 GiB limit and 100 MiB used (well
// within the OK band) so render-only tests don't see a banner unless
// they ask for one.
interface MockQuotaState {
  used_bytes: number | null;
  limit_bytes: number | null;
  used_pct: number | null;
  status: 'ok' | 'warning' | 'blocked';
  warn_threshold: number;
  configured: boolean;
}
function defaultQuotaState(): MockQuotaState {
  const used = 100 * 1024 * 1024;
  const limit = 2 * 1024 * 1024 * 1024;
  return {
    used_bytes: used,
    limit_bytes: limit,
    used_pct: used / limit,
    status: 'ok',
    warn_threshold: 0.85,
    configured: true,
  };
}
let quotaState: MockQuotaState = defaultQuotaState();
export function setMockQuotaState(patch: Partial<MockQuotaState>): void {
  quotaState = { ...quotaState, ...patch };
  if (
    (patch.used_bytes !== undefined || patch.limit_bytes !== undefined) &&
    patch.used_pct === undefined &&
    quotaState.limit_bytes
  ) {
      quotaState.used_pct =
      quotaState.used_bytes === null
        ? null
        : quotaState.used_bytes / quotaState.limit_bytes;
  }
}
// Vision/procedure ingestion settings — image curation plus the admin
// activation toggle. In-memory snapshot (like quota); defaults mirror the
// backend deployment defaults until a PUT lands a runtime override.
interface MockVisionSettingsState {
  min_ocr_chars: number;
  drop_classes: string[];
  procedure_enabled: boolean;
  procedure_available: boolean;
  source: 'runtime' | 'env-default';
  updated_at: number | null;
  updated_by: string | null;
}
function defaultVisionSettingsState(): MockVisionSettingsState {
  return {
    min_ocr_chars: 20,
    drop_classes: ['invalid', 'logo', 'signature'],
    procedure_enabled: false,
    procedure_available: true,
    source: 'env-default',
    updated_at: null,
    updated_by: null,
  };
}
function loadVisionSettingsState(): MockVisionSettingsState {
  const raw = e2eStorage()?.getItem(E2E_VISION_SETTINGS_STORAGE_KEY);
  if (!raw) return defaultVisionSettingsState();
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return {
        ...defaultVisionSettingsState(),
        ...(parsed as Partial<MockVisionSettingsState>),
      };
    }
  } catch {
    // Ignore corrupt e2e state and fall back to the defaults.
  }
  return defaultVisionSettingsState();
}

// Persisted across reloads (like procedures/api-keys): the real backend
// stores runtime vision settings server-side, so a reload-survival
// assertion must hold against the mock too.
let visionSettingsState: MockVisionSettingsState = loadVisionSettingsState();

function persistVisionSettingsState(): void {
  persistState(E2E_VISION_SETTINGS_STORAGE_KEY, visionSettingsState);
}
const VISION_DROP_CLASS_RE = /^[a-z0-9][a-z0-9 _-]{0,39}$/;

// ---------------------------------------------------------------------------
// Procedure approval bundles — STATEFUL mock of `server/procedure_routes.py`.
// Test doctrine: decisions must MUTATE the mock the way the real backend
// would (approve → bundle state flips AND the documents list gains the
// enqueued doc), otherwise the e2e journey goes falsely green.
// ---------------------------------------------------------------------------

function cloneProcedures(
  items: readonly ProcedureBundle[],
): ProcedureBundle[] {
  return structuredClone(items) as ProcedureBundle[];
}

function loadProceduresState(): ProcedureBundle[] {
  const raw = e2eStorage()?.getItem(E2E_PROCEDURES_STORAGE_KEY);
  if (!raw) return cloneProcedures(PROCEDURE_BUNDLE_FIXTURES);
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      return cloneProcedures(parsed as ProcedureBundle[]);
    }
  } catch {
    // Ignore corrupt e2e state and fall back to the fixtures.
  }
  return cloneProcedures(PROCEDURE_BUNDLE_FIXTURES);
}

let proceduresState: ProcedureBundle[] = loadProceduresState();
let procedureDocSeq = 0;

function persistProceduresState(): void {
  persistState(E2E_PROCEDURES_STORAGE_KEY, proceduresState);
}

/** Folder-bound list projection — mirrors `procedure_routes._summary`
 *  (NO paths, NO PNGs, NO full text in the list). */
function procedureSummary(bundle: ProcedureBundle): ProcedureBundleSummary {
  return {
    id: bundle.id,
    file_name: bundle.file_name,
    state: bundle.state,
    reason: bundle.reason,
    source: bundle.source,
    track_id: bundle.track_id ?? null,
    schematics_total: bundle.schematics_total ?? 0,
    schematics_described: (bundle.schematics ?? []).filter(
      (s) => s.informed !== null && s.informed !== undefined,
    ).length,
    classification: bundle.classification ?? null,
    operator_classification: bundle.operator_classification ?? null,
    created_at: bundle.created_at ?? null,
    updated_at: bundle.updated_at ?? null,
  };
}

/** Mirrors `_procedure.bundle_folders` — primary first, deduped,
 *  duplicate-request folders included, nulls skipped. */
function mockBundleFolders(bundle: ProcedureBundle): string[] {
  const folders: string[] = [];
  if (bundle.folder) folders.push(String(bundle.folder));
  for (const request of bundle.duplicate_requests ?? []) {
    const folder = request?.folder;
    if (folder && !folders.includes(folder)) folders.push(folder);
  }
  return folders;
}

function recordProcedureActivity(
  kind: string,
  bundle: ProcedureBundle,
  summary: string,
): void {
  recordActivity({
    id: `evt_proc_${kind}_${bundle.id}_${Date.now()}`,
    ts: new Date().toISOString(),
    rel: 'now',
    day: 'Today',
    kind: kind as ActivityEvent['kind'],
    sev: kind === 'procedure-approved' ? 'info' : 'warning',
    actor: { user: 'mock-operator', role: 'admin' },
    target: { type: 'document', label: bundle.file_name, id: bundle.id },
    summary,
    meta: { bundle_id: bundle.id },
  });
}

/** The document row an approved/rerouted bundle enqueues — lands PENDING
 *  and flips PROCESSED on a later poll like real ingestion. */
function enqueueProcedureDocument(
  bundle: ProcedureBundle,
  folder: string,
): void {
  procedureDocSeq += 1;
  documentsState = [
    {
      doc_id: `proc_doc_${bundle.id}_${procedureDocSeq}`,
      track_id: bundle.track_id ?? `track_proc_${procedureDocSeq}`,
      file_path: bundle.file_name,
      content_summary: `Approved procedure bundle ${bundle.id}`,
      content_length: bundle.full_text?.length ?? 0,
      status: 'PENDING',
      chunks_count: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      error_msg: null,
      metadata: {
        uploader: 'procedure-review',
        [INGEST_POLLS_KEY]: 1,
        ...(bundle.classification
          ? { classification: bundle.classification }
          : {}),
      },
      type: 'file',
      tags: [],
      folder,
      visibility: 'private',
    },
    ...documentsState,
  ];
  persistDocumentsState();
}

function nextApiKeyId(): string {
  apiKeyCounter += 1;
  return `mock-key-${apiKeyCounter}`;
}
function mintMockToken(): { full: string; preview: string } {
  apiKeyCounter += 1;
  const body = `mock${apiKeyCounter.toString().padStart(28, '0')}`;
  return { full: `twk_${body}`, preview: `twk_${body.slice(0, 8)}…` };
}
function publicApiKey(k: ApiKeyMockEntry): Omit<ApiKeyMockEntry, 'full_value'> {
  return {
    id: k.id,
    name: k.name,
    prefix: k.prefix,
    created_at: k.created_at,
    created_by: k.created_by,
    last_used_at: k.last_used_at,
    revoked_at: k.revoked_at,
  };
}

interface E2eScenario {
  bulkRetagStatus?: number;
  approveDelayMs?: number;
  tagApproveDelayMs?: number;
  trackStatusMode?: 'empty' | 'processed' | 'timeout';
  authGate?: boolean;
  uploadFailureNames?: string[];
  /** Delay (ms) before the bulk-delete handler responds. Used by the
   *  Playwright "DELETING badge" test to assert the optimistic state
   *  is visible while the round-trip is in flight. */
  bulkDeleteDelayMs?: number;
  /** Force selected ids to fail without mutating them, exercising HTTP 207. */
  bulkDeleteFailIds?: string[];
}

// Scenario + local auth survive a page reload (sessionStorage) so e2e specs
// can exercise full journeys that cross a navigation boundary (e.g. enable
// the auth gate, reload, land on the login screen). `/__e2e/reset` clears
// both along with the rest of the mock state.
function loadScenario(): E2eScenario {
  const raw = e2eStorage()?.getItem(E2E_SCENARIO_STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === 'object') return parsed;
  } catch {
    // Ignore corrupt e2e state and fall back to an empty scenario.
  }
  return {};
}

function persistScenario(): void {
  persistState(E2E_SCENARIO_STORAGE_KEY, e2eScenario);
}

function persistLocalAuthUser(): void {
  persistState(E2E_AUTH_USER_STORAGE_KEY, localAuthUser);
}

const e2eScenario: E2eScenario = loadScenario();
let localAuthUser: string | null = (() => {
  const raw = e2eStorage()?.getItem(E2E_AUTH_USER_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    return typeof parsed === 'string' ? parsed : null;
  } catch {
    return null;
  }
})();
const e2eStats = {
  approveCalls: {} as Record<string, number>,
  tagApproveCalls: {} as Record<string, number>,
  folderRequests: [] as Array<{
    path: string;
    folder: string | null;
  }>,
  queryRequests: [] as Array<{
    path: string;
    body: unknown;
  }>,
  uploadRequests: [] as Array<{
    name: string;
    docType: string | null;
    classification: string | null;
  }>,
};

function recordTwinFolderRequest(request: Request): void {
  const url = new URL(request.url);
  e2eStats.folderRequests.push({
    path: url.pathname,
    folder: request.headers.get('X-Twin-Folder'),
  });
}

export function mockCurrentScopes(): readonly string[] | null {
  if (globalThis.window === undefined) return null;
  const twinConfig = globalThis.window.__twinConfig;
  const config =
    globalThis.window.__twinE2eRuntimeConfig ??
    (twinConfig !== undefined && twinConfig !== null && typeof twinConfig === 'object'
      ? twinConfig
      : undefined);
  return config?.debugUser?.gateway_scopes ?? null;
}

function rejectFolderAdminMutationIfNeeded() {
  const scopes = mockCurrentScopes();
  if (scopes === null || scopes.includes('admin:folders')) return null;
  return HttpResponse.json(
    { detail: "Admin scope 'admin:folders' required" },
    { status: 403 },
  );
}

export function resetDocumentsState(): void {
  const storage = e2eStorage();
  storage?.removeItem(E2E_DOCUMENTS_STORAGE_KEY);
  storage?.removeItem(E2E_TAG_CATEGORIES_STORAGE_KEY);
  storage?.removeItem(E2E_TAGS_STORAGE_KEY);
  storage?.removeItem(E2E_NOTIFICATIONS_STORAGE_KEY);
  storage?.removeItem(E2E_ACTIVITY_STORAGE_KEY);
  storage?.removeItem(E2E_SCENARIO_STORAGE_KEY);
  storage?.removeItem(E2E_AUTH_USER_STORAGE_KEY);
  storage?.removeItem(E2E_GRAPH_ENTITIES_STORAGE_KEY);
  storage?.removeItem(E2E_GRAPH_RELATIONS_STORAGE_KEY);
  storage?.removeItem(E2E_API_KEYS_STORAGE_KEY);
  storage?.removeItem(E2E_PROCEDURES_STORAGE_KEY);
  storage?.removeItem(E2E_VISION_SETTINGS_STORAGE_KEY);
  documentsState = cloneDocuments(DOCUMENT_FIXTURES);
  proceduresState = cloneProcedures(PROCEDURE_BUNDLE_FIXTURES);
  procedureDocSeq = 0;
  resetDocumentMemberships();
  categoryState = cloneTagCategories(TAG_CATEGORY_FIXTURES);
  tagState = cloneTags(TAG_FIXTURES);
  notificationState = cloneNotifications(NOTIFICATION_FIXTURES);
  activityState = cloneActivity(ACTIVITY_FIXTURES);
  apiKeyState = [];
  apiKeyCounter = 0;
  quotaState = defaultQuotaState();
  visionSettingsState = defaultVisionSettingsState();
  graphEntityState = cloneGraphEntities(GRAPH_ENTITY_FIXTURES);
  graphRelationState = cloneGraphRelations(GRAPH_RELATION_FIXTURES);
  uploadedTrackDocs.clear();
  uploadedDocText.clear();
  uploadSeq = 0;
  folderState = FOLDER_FIXTURES.slice(0, 1).map((w) => ({ ...w }));
  e2eScenario.bulkRetagStatus = undefined;
  e2eScenario.approveDelayMs = undefined;
  e2eScenario.tagApproveDelayMs = undefined;
  e2eScenario.trackStatusMode = undefined;
  e2eScenario.authGate = undefined;
  e2eScenario.uploadFailureNames = undefined;
  e2eScenario.bulkDeleteDelayMs = undefined;
  e2eScenario.bulkDeleteFailIds = undefined;
  e2eStats.approveCalls = {};
  e2eStats.tagApproveCalls = {};
  e2eStats.folderRequests = [];
  e2eStats.queryRequests = [];
  e2eStats.uploadRequests = [];
  localAuthUser = null;
}

async function recordQueryRequest(request: Request): Promise<Record<string, unknown>> {
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  e2eStats.queryRequests.push({
    path: new URL(request.url).pathname,
    body,
  });
  return body;
}

const TWIN_QUERY_MODES = new Set(['naive', 'local', 'global', 'hybrid', 'mix']);

function rejectUnsafeTwinQuery(
  body: Record<string, unknown>,
): ReturnType<typeof HttpResponse.json> | undefined {
  if (typeof body.mode === 'string' && !TWIN_QUERY_MODES.has(body.mode)) {
    return HttpResponse.json(
      { detail: `Unsupported Twin query mode: ${body.mode}` },
      { status: 422 },
    );
  }
  if (body.only_need_prompt === true) {
    return HttpResponse.json(
      { detail: 'only_need_prompt is disabled on the external API' },
      { status: 422 },
    );
  }
  if (typeof body.user_prompt === 'string' && body.user_prompt.trim()) {
    return HttpResponse.json(
      { detail: 'raw user_prompt overrides are disabled' },
      { status: 422 },
    );
  }
  return undefined;
}

function updateDoc(id: string, patch: Partial<Document>): Document | null {
  const idx = documentsState.findIndex((d) => d.doc_id === id);
  if (idx < 0) return null;
  documentsState[idx] = { ...documentsState[idx], ...patch };
  persistDocumentsState();
  return documentsState[idx];
}

async function handleCreateFolder(request: Request): Promise<Response> {
  const forbidden = rejectFolderAdminMutationIfNeeded();
  if (forbidden) return forbidden;
  const body = (await request.json()) as {
    id: string;
    label: string;
    kind?: string;
    description?: string;
  };
  if (!/^[A-Za-z0-9_-]+$/.test(body.id)) {
    return HttpResponse.json(
      { detail: `Invalid folder id '${body.id}'` },
      { status: 422 },
    );
  }
  if (folderState.length >= FOLDER_MAX) {
    return HttpResponse.json(
      { detail: `Cannot create folder: catalog already at max (${FOLDER_MAX})` },
      { status: 422 },
    );
  }
  if (envSeededFolderIds.has(body.id)) {
    return HttpResponse.json(
      { detail: `Folder '${body.id}' is provisioned by the deploy env` },
      { status: 409 },
    );
  }
  if (folderState.some((s) => s.id === body.id)) {
    return HttpResponse.json(
      { detail: `Folder '${body.id}' already exists` },
      { status: 409 },
    );
  }
  const created = {
    id: body.id,
    kb: body.label,
    visibility: 'internal' as const,
    sources: 0,
    role: 'admin / steward' as const,
    current: false,
  };
  folderState = [...folderState, created];
  activityState = [
    {
      id: `evt_folder_${body.id}_${Date.now()}`,
      ts: new Date().toISOString(),
      rel: 'now',
      day: 'Today',
      kind: 'settings',
      sev: 'info',
      actor: { user: 'operator.demo', role: 'KB Steward' },
      target: { type: 'folder', label: body.label, id: body.id },
      summary: `Folder '${body.id}' created`,
      meta: { folder_id: body.id, operation: 'create' },
    },
    ...activityState,
  ];
  persistActivityState();
  return HttpResponse.json(created, { status: 201 });
}

async function handleUpdateFolder(
  id: string,
  request: Request,
): Promise<Response> {
  const forbidden = rejectFolderAdminMutationIfNeeded();
  if (forbidden) return forbidden;
  if (envSeededFolderIds.has(id)) {
    return HttpResponse.json(
      { detail: `Folder '${id}' is env-seeded and cannot be edited` },
      { status: 403 },
    );
  }
  const idx = folderState.findIndex((s) => s.id === id);
  if (idx < 0) {
    return HttpResponse.json(
      { detail: `Folder '${id}' not found` },
      { status: 404 },
    );
  }
  const patch = (await request.json()) as { label?: string };
  const next = {
    ...folderState[idx],
    kb: patch.label ?? folderState[idx].kb,
  };
  folderState = [
    ...folderState.slice(0, idx),
    next,
    ...folderState.slice(idx + 1),
  ];
  activityState = [
    {
      id: `evt_folder_${id}_${Date.now()}`,
      ts: new Date().toISOString(),
      rel: 'now',
      day: 'Today',
      kind: 'settings',
      sev: 'info',
      actor: { user: 'operator.demo', role: 'KB Steward' },
      target: { type: 'folder', label: next.kb, id },
      summary: `Folder '${id}' updated`,
      meta: { folder_id: id, operation: 'update' },
    },
    ...activityState,
  ];
  persistActivityState();
  return HttpResponse.json(next);
}

function handleDeleteFolder(id: string): Response {
  const forbidden = rejectFolderAdminMutationIfNeeded();
  if (forbidden) return forbidden;
  if (envSeededFolderIds.has(id)) {
    return HttpResponse.json(
      { detail: `Folder '${id}' is env-seeded and cannot be deleted` },
      { status: 403 },
    );
  }
  const idx = folderState.findIndex((s) => s.id === id);
  if (idx < 0) {
    return HttpResponse.json(
      { detail: `Folder '${id}' not found` },
      { status: 404 },
    );
  }
  folderState = folderState.filter((s) => s.id !== id);
  activityState = [
    {
      id: `evt_folder_${id}_${Date.now()}`,
      ts: new Date().toISOString(),
      rel: 'now',
      day: 'Today',
      kind: 'settings',
      sev: 'info',
      actor: { user: 'operator.demo', role: 'KB Steward' },
      target: { type: 'folder', label: id, id },
      summary: `Folder '${id}' deleted`,
      meta: { folder_id: id, operation: 'delete' },
    },
    ...activityState,
  ];
  persistActivityState();
  return new HttpResponse(null, { status: 204 });
}

function matchDocumentsQuery(d: Document, params: URLSearchParams): boolean {
  const status = params.get('status');
  if (status && status !== 'all' && d.status !== status) return false;
  const folder = params.get('folder');
  if (folder && !foldersForDoc(d).includes(folder)) return false;
  const q = params.get('q');
  if (q && !d.file_path.toLowerCase().includes(q.toLowerCase())) return false;
  const tag = params.get('tag');
  if (tag && !d.tags.includes(tag)) return false;
  return true;
}

function statusCountsFor(docs: readonly Document[]): Record<DocumentStatus, number> {
  return docs.reduce<Record<DocumentStatus, number>>(
    (acc, doc) => {
      acc[doc.status] = (acc[doc.status] ?? 0) + 1;
      return acc;
    },
    { PENDING: 0, PROCESSING: 0, PROCESSED: 0, FAILED: 0 },
  );
}

function matchActivityRange(e: ActivityEvent, range: string | null): boolean {
  if (range && range !== 'all') {
    const windowMs = ACTIVITY_RANGE_MS[range as keyof typeof ACTIVITY_RANGE_MS];
    if (windowMs !== undefined) {
      const ts = Date.parse(e.ts);
      if (Number.isNaN(ts) || ts < ACTIVITY_NOW_MS - windowMs) return false;
    }
  }
  return true;
}

function activityResourceIds(e: ActivityEvent): string[] {
  const metaDocIds = Array.isArray(e.meta?.doc_ids) ? e.meta.doc_ids.map(String) : [];
  const metaDocId =
    typeof e.meta?.doc_id === 'string' || typeof e.meta?.doc_id === 'number'
      ? [String(e.meta.doc_id)]
      : [];
  return [e.target.id, ...metaDocId, ...metaDocIds].filter(
    (id): id is string => typeof id === 'string' && id.length > 0,
  );
}

function matchActivityQuery(e: ActivityEvent, params: URLSearchParams): boolean {
  if (!matchActivityRange(e, params.get('range'))) return false;
  const kind = params.get('kind');
  if (kind) {
    const wanted = new Set(kind.split(','));
    if (!wanted.has(e.kind)) return false;
  }
  const sev = params.get('sev');
  if (sev && sev !== 'any' && e.sev !== sev) return false;
  const actor = params.get('actor');
  if (actor && actor !== 'any' && e.actor.user !== actor) return false;
  const resourceId = params.get('resource.id');
  if (resourceId && !activityResourceIds(e).includes(resourceId)) {
    return false;
  }
  const q = params.get('q');
  if (q) {
    const needle = q.toLowerCase();
    const hay = (
      e.summary +
      ' ' +
      e.target.label +
      ' ' +
      e.actor.user +
      ' ' +
      e.id
    ).toLowerCase();
    if (!hay.includes(needle)) return false;
  }
  return true;
}

function tagEntryStub(name: string, status: TagEntry['status'], action: string): TagEntry {
  return {
    tag: name,
    tier: status === 'pending-review' ? 'requested' : 3,
    category: 'infra',
    status,
    def: '',
    aliases: [],
    deprecates: [],
    sources_count: 0,
    chunks_count: 0,
    query_freq_30d: 0,
    created: { by: 'system', at: '2026-05-29' },
    last_edit: { by: 'system', at: '2026-05-29', action },
    related: [],
    examples: [],
  };
}

function upsertTag(tag: TagEntry): TagEntry {
  const idx = tagState.findIndex((t) => t.tag === tag.tag);
  if (idx >= 0) tagState[idx] = tag;
  else tagState = [tag, ...tagState];
  persistTagState();
  return tag;
}

function isArchivedTag(tag: TagEntry): boolean {
  return tag.status === 'rejected' || tag.status === 'deleted';
}

function cascadeDeletedTagFromDocuments(
  name: string,
  strategy: 'migrate' | 'untag' = 'untag',
  to?: string,
): number {
  let affected = 0;
  documentsState = documentsState.map((doc) => {
    if (!doc.tags.includes(name)) return doc;
    affected += 1;
    const nextTags = doc.tags.filter((tag) => tag !== name);
    if (strategy === 'migrate' && to && !nextTags.includes(to)) {
      nextTags.push(to);
    }
    return { ...doc, tags: nextTags };
  });
  if (affected > 0) persistDocumentsState();
  return affected;
}

function recordActivity(event: ActivityEvent): void {
  activityState = [event, ...activityState];
  persistActivityState();
}

function recordTagMutation(
  name: string,
  suffix: string,
  options: {
    actor?: string;
    sev?: ActivityEvent['sev'];
    meta?: Record<string, unknown>;
  } = {},
): void {
  const actor = options.actor ?? 'claire.benoit';
  notificationState = [
    {
      id: `n_tag_${name}_${Date.now()}`,
      kind: 'tag-mutation',
      title: 'Tag',
      tagname: name,
      suffix,
      sub: 'Tag catalog updated by e2e steward action',
      rel: 'now',
      read: false,
    },
    ...notificationState,
  ];
  recordActivity({
    id: `evt_tag_${name}_${Date.now()}`,
    ts: new Date().toISOString(),
    rel: 'now',
    day: 'Today',
    kind: 'tag-mutation',
    sev: options.sev ?? 'info',
    actor: { user: actor, role: 'KB Admin' },
    target: { type: 'tag', label: name, id: name },
    summary: `Tag ${name} ${suffix}`,
    meta: options.meta ?? { tag: name, action: suffix },
  });
  persistNotificationState();
}

function recordDocumentActivity(
  kind: Extract<ActivityEvent['kind'], 'doc-approved' | 'doc-rejected' | 'doc-deleted'>,
  doc: Document,
  summary: string,
  meta: Record<string, unknown>,
  options: {
    actor?: string;
    sev?: ActivityEvent['sev'];
  } = {},
): void {
  recordActivity({
    id: `evt_doc_${doc.doc_id}_${kind}_${Date.now()}`,
    ts: new Date().toISOString(),
    rel: 'now',
    day: 'Today',
    kind,
    sev: options.sev ?? 'info',
    actor: { user: options.actor ?? 'operator.demo', role: 'KB Steward' },
    target: { type: 'document', label: doc.file_path, id: doc.doc_id },
    summary,
    meta: { doc_id: doc.doc_id, ...meta },
  });
}

function authGateResponse(
  request: Request,
): ReturnType<typeof HttpResponse.json> | undefined {
  if (!e2eScenario.authGate) return undefined;
  if (request.headers.get('Authorization') || localAuthUser) return undefined;
  return HttpResponse.json(
    { detail: 'Unauthorized: login required' },
    {
      status: 401,
      headers: { 'WWW-Authenticate': 'Bearer' },
    },
  );
}

function makeE2eDocument(patch: Partial<Document>, index: number): Document {
  const id = patch.doc_id ?? `e2e_doc_${index}`;
  return {
    doc_id: id,
    track_id: patch.track_id ?? null,
    file_path: patch.file_path ?? `/cib/e2e/${id}.md`,
    content_summary: patch.content_summary ?? `${id} generated by e2e`,
    content_length: patch.content_length ?? 128,
    status: patch.status ?? 'PROCESSED',
    chunks_count: patch.chunks_count ?? 1,
    created_at: patch.created_at ?? '2026-06-02T00:00:00Z',
    updated_at: patch.updated_at ?? '2026-06-02T00:00:00Z',
    error_msg: patch.error_msg ?? null,
    metadata: patch.metadata ?? {},
    type: patch.type ?? 'file',
    tags: patch.tags ?? [],
    folder: patch.folder ?? 'default',
    visibility: patch.visibility ?? 'private',
    review: patch.review,
    extracted_text: patch.extracted_text,
  };
}

export const handlers = [
  // -------------------------------------------------------------------------
  // E2E-only controls. These endpoints are intercepted by MSW in dev/test and
  // let Playwright simulate backend failures without changing app code.
  // -------------------------------------------------------------------------
  http.post(`${ANY}/__e2e/reset`, () => {
    resetDocumentsState();
    return HttpResponse.json({ ok: true });
  }),
  http.post(`${ANY}/__e2e/scenario`, async ({ request }) => {
    const patch = (await request.json()) as E2eScenario;
    Object.assign(e2eScenario, patch);
    persistScenario();
    return HttpResponse.json({ ok: true, scenario: e2eScenario });
  }),
  http.post(`${ANY}/__e2e/quota`, async ({ request }) => {
    const patch = (await request.json()) as Partial<MockQuotaState>;
    setMockQuotaState(patch);
    return HttpResponse.json({ ok: true, quota: quotaState });
  }),
  http.get(`${ANY}/__e2e/stats`, () =>
    HttpResponse.json({
      approveCalls: e2eStats.approveCalls,
      tagApproveCalls: e2eStats.tagApproveCalls,
      folderRequests: e2eStats.folderRequests,
      queryRequests: e2eStats.queryRequests,
      uploadRequests: e2eStats.uploadRequests,
    }),
  ),
  http.post(`${ANY}/__e2e/documents`, async ({ request }) => {
    const body = (await request.json()) as
      | { documents?: Partial<Document>[] }
      | Partial<Document>[];
    const patches = Array.isArray(body) ? body : body.documents ?? [];
    const next = patches.map((patch, index) =>
      makeE2eDocument(patch, documentsState.length + index + 1),
    );
    documentsState = [...next, ...documentsState];
    persistDocumentsState();
    return HttpResponse.json({ ok: true, ids: next.map((doc) => doc.doc_id) });
  }),
  http.post(`${ANY}/__e2e/activity`, async ({ request }) => {
    const body = (await request.json()) as
      | { events?: ActivityEvent[] }
      | ActivityEvent[];
    const events = Array.isArray(body) ? body : body.events ?? [];
    activityState = [...cloneActivity(events), ...activityState];
    persistActivityState();
    return HttpResponse.json({ ok: true, ids: events.map((event) => event.id) });
  }),
  // Replace the parked-procedure queue (e.g. empty it): the pending section
  // aggregates documents AND procedures, so a doc-only journey that expects
  // the section to vanish must be able to clear the procedure seed too.
  http.post(`${ANY}/__e2e/procedures`, async ({ request }) => {
    const body = (await request.json()) as
      | { bundles?: ProcedureBundle[] }
      | ProcedureBundle[];
    const bundles = Array.isArray(body) ? body : body.bundles ?? [];
    proceduresState = cloneProcedures(bundles);
    persistProceduresState();
    return HttpResponse.json({ ok: true, count: proceduresState.length });
  }),

  // Local LightRAG-compatible auth endpoints. In production these are native
  // root routes, not Twin overlay routes.
  http.get(`${ANY}/auth-status`, () => {
    if (!e2eScenario.authGate) {
      return HttpResponse.json({
        auth_enabled: false,
        authenticated: true,
        user: null,
        expires_at: null,
        login_required: false,
      });
    }
    return HttpResponse.json({
      auth_enabled: true,
      authenticated: localAuthUser !== null,
      user: localAuthUser,
      expires_at: localAuthUser ? '2099-12-31T23:59:00Z' : null,
      login_required: localAuthUser === null,
    });
  }),
  http.post(`${ANY}/login`, async ({ request }) => {
    const body = (await request.json().catch(() => null)) as
      | { username?: string; password?: string }
      | null;
    const username = body?.username?.trim();
    // 'invalid-password' is the e2e knob for the failed-credentials path —
    // the mock otherwise accepts any non-empty pair like LightRAG demo auth.
    if (!username || !body?.password || body.password === 'invalid-password') {
      return HttpResponse.json(
        { detail: 'Invalid username or password' },
        { status: 401 },
      );
    }
    localAuthUser = username;
    persistLocalAuthUser();
    return HttpResponse.json({
      access_token: `mock-token-${username}`,
      token_type: 'bearer',
      expires_in: 3600,
    });
  }),
  http.post(`${ANY}/logout`, () => {
    localAuthUser = null;
    persistLocalAuthUser();
    return HttpResponse.json({ ok: true });
  }),

  // -------------------------------------------------------------------------
  // LightRAG-native endpoints
  // -------------------------------------------------------------------------
  http.get(`${ANY}/documents`, ({ request }) => {
    const url = new URL(request.url);
    // Skip Twin overlay paths so this generic /documents handler does not
    // shadow /twin/api/documents/* routes.
    if (url.pathname.startsWith(TWIN)) return undefined;
    // Mirror the real async pipeline: uploaded docs land PENDING and flip to
    // PROCESSED on the second poll (audit DUP-2a; see advanceMockIngestion).
    advanceMockIngestion();
    const filtered = documentsState.filter((d) =>
      matchDocumentsQuery(d, url.searchParams),
    );
    const page = Number(url.searchParams.get('cursor') || '1');
    const pageSize = 50;
    const safePage = Number.isFinite(page) && page > 0 ? Math.floor(page) : 1;
    const start = (safePage - 1) * pageSize;
    const items = filtered.slice(start, start + pageSize);
    const nextCursor =
      safePage * pageSize < filtered.length ? String(safePage + 1) : null;
    return HttpResponse.json({
      items,
      total: filtered.length,
      page: safePage,
      page_size: pageSize,
      status_counts: statusCountsFor(filtered),
      next_cursor: nextCursor,
    });
  }),
  http.get(`${ANY}/documents/:id/chunks`, ({ params, request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const id = String(params.id);
    const doc = documentsState.find((d) => d.doc_id === id);
    const storedText = uploadedDocText.get(id);
    if (storedText) {
      return HttpResponse.json([
        {
          chunk_id: `${id}_c0`,
          order: 0,
          text: storedText,
        },
      ]);
    }
    const chunks =
      doc && (doc.chunks_count ?? 0) > 0
        ? [
            {
              chunk_id: `${id}_c0`,
              order: 0,
              text: `Mock backend has no extracted text for ${doc.file_path}. Use the real backend to inspect LightRAG PDF/binary chunks.`,
            },
          ]
        : [];
    return HttpResponse.json(chunks);
  }),
  http.post(`${ANY}/documents/upload`, async ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const form = await request.formData();
    const file = form.get('file');
    const name =
      file instanceof File && file.name
        ? file.name
        : `uploaded-${Date.now()}.txt`;
    const operatorClass = request.headers.get('X-Twin-Classification');
    const docType = request.headers.get('X-Twin-Doc-Type');
    // Mirror the real upload contract: only operator C1/C2 and the two
    // explicit ingestion profiles are accepted before any state mutation.
    const headerError = uploadHeaderError(operatorClass, docType);
    if (headerError) {
      return HttpResponse.json({ detail: headerError }, { status: 400 });
    }
    e2eStats.uploadRequests.push({
      name,
      docType,
      classification: operatorClass,
    });
    const classificationMeta =
      operatorClass && operatorClass in MIP_CLASS_NAMES
        ? {
            classification: {
              class_id: operatorClass,
              class_name: MIP_CLASS_NAMES[operatorClass],
              source_format: 'operator',
              reason: 'operator-set',
            },
          }
        : {};
    if (e2eScenario.uploadFailureNames?.includes(name)) {
      return HttpResponse.json(
        { detail: `${name} upload failed by e2e scenario` },
        { status: 500 },
      );
    }
    uploadSeq += 1;
    const trackId = `track_${name.replaceAll(/[^a-z0-9]+/gi, '_').toLowerCase()}_${uploadSeq}`;

    // PARKED path — mirrors the backend seam: a FORCED procedure upload is
    // claimed by the profile, parks an approval bundle and creates NO
    // document until approval. The upload response is indistinguishable
    // from a normal enqueue (same receipt) — exactly why the UI reconciles
    // its optimistic row against the procedure queue by track_id.
    if (docType === 'procedure') {
      proceduresState = [
        {
          id: `proc_up_${uploadSeq}`,
          file_name: name,
          state: 'pending',
          reason: 'ok',
          source: 'forced',
          original_path: `/inputs/${name}`,
          track_id: trackId,
          folder: 'default',
          content_hash: `hash_${uploadSeq}`,
          full_text: 'Uploaded procedure body',
          schematics: [],
          schematics_total: 0,
          classification: null,
          operator_classification: operatorClass,
          duplicate_requests: [],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        ...proceduresState,
      ];
      persistProceduresState();
      return HttpResponse.json({
        status: 'success',
        message: `${name} queued for ingestion`,
        track_id: trackId,
      });
    }

    const docId = `uploaded_${uploadSeq}`;
    uploadedTrackDocs.set(trackId, docId);
    const text =
      file instanceof File && file.type.startsWith('text/')
        ? await file.text()
        : '';
    if (text.trim()) uploadedDocText.set(docId, text);
    // Real contract (audit DUP-2a): upload ENQUEUES. The response is the
    // enqueue receipt; the DocStatus row starts PENDING (chunks not counted
    // yet) and reaches PROCESSED only on a later documents poll — the UI
    // already polls (`refreshDocumentsUntilUploadsLand` + the 2s
    // useDocuments refetchInterval), so no app change is needed.
    documentsState = [
      {
        doc_id: docId,
        track_id: trackId,
        file_path: name,
        content_summary: `${name} uploaded by e2e`,
        content_length: file instanceof File ? file.size : 0,
        status: 'PENDING',
        chunks_count: null,
        created_at: '2026-06-02T00:00:00Z',
        updated_at: '2026-06-02T00:00:00Z',
        error_msg: null,
        metadata: {
          mime: file instanceof File ? file.type : 'text/plain',
          uploader: 'e2e',
          [INGEST_POLLS_KEY]: 1,
          ...classificationMeta,
        },
        type: 'file',
        tags: [],
        folder: 'default',
        visibility: 'private',
      },
      ...documentsState,
    ];
    persistDocumentsState();
    return HttpResponse.json({
      status: 'success',
      message: `${name} queued for ingestion`,
      track_id: trackId,
    });
  }),
  http.get(`${ANY}/documents/track_status/:trackId`, ({ params, request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const trackId = String(params.trackId);
    const docId = uploadedTrackDocs.get(trackId);
    return trackStatusResponse(trackId, docId);
  }),
  http.post(`${ANY}/documents/reprocess_failed`, ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const failedCount = documentsState.filter((d) => d.status === 'FAILED').length;
    return HttpResponse.json({
      status: 'queued',
      message: 'LightRAG is retrying all FAILED docs',
      failed_count: failedCount,
    });
  }),
  http.delete(`${ANY}/documents/:id`, ({ params, request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const id = String(params.id);
    // Real contract (audit DUP-2b): the native shim 404s unknown/invisible
    // docs and applies the ref-counted delete — unshare from the active
    // folder unless it was the doc's LAST membership
    // (`native_shims._delete_document_impl` + `_delete_or_unshare`,
    // native_shims.py:567-634).
    const doc = documentsState.find((d) => d.doc_id === id);
    if (!doc) {
      return HttpResponse.json(
        { detail: `Document ${id} not found` },
        { status: 404 },
      );
    }
    const active = activeFolderFor(request, doc);
    if (!foldersForDoc(doc).includes(active)) {
      return HttpResponse.json(
        { detail: `Document ${id} not found` },
        { status: 404 },
      );
    }
    const physicallyDeleted = removeDocFromFolderOrDelete(doc, active);
    if (physicallyDeleted) cascadeDocsFromGraph(new Set([id]));
    return HttpResponse.json({ ok: true });
  }),
  http.get(`${ANY}/health`, ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    return HttpResponse.json({ status: 'ok', version: API_VERSION });
  }),
  http.get(`${ANY}/pipeline_status`, () =>
    HttpResponse.json({
      busy: false,
      job_count: 0,
      job_name: null,
      latest_message: null,
      history_messages: [],
    }),
  ),
  http.get(`${ANY}/openapi`, ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    return HttpResponse.json({ groups: OPENAPI_GROUPS, version: API_VERSION });
  }),
  // Standard FastAPI auto OpenAPI 3.1 spec — the React Settings → API
  // tab hits this directly so it stays ISO with LightRAG by
  // construction (mock-kill F2). Synthesise a minimal but valid OpenAPI
  // 3.1 doc from the local OPENAPI_GROUPS fixture so the standalone demo
  // demo still renders the API tab.
  http.get(`${ANY}/openapi.json`, () => {
    // Mirror the real backend's per-operation security: public handshake
    // routes carry no requirement, everything else requires the bearer.
    const publicPaths = new Set([
      '/login',
      '/logout',
      '/auth-status',
      '/health',
      '/ready',
    ]);
    const paths: Record<string, Record<string, unknown>> = {};
    for (const g of OPENAPI_GROUPS) {
      for (const ep of g.endpoints) {
        paths[ep.p] ??= {};
        paths[ep.p][ep.m.toLowerCase()] = {
          tags: [g.id],
          summary: ep.s,
          responses: { '200': { description: 'OK' } },
          security: publicPaths.has(ep.p) ? [] : [{ HTTPBearer: [] }],
        };
      }
    }
    return HttpResponse.json({
      openapi: '3.1.0',
      info: { title: 'LightRAG Server API (mocked)', version: API_VERSION },
      tags: OPENAPI_GROUPS.map((g) => ({ name: g.id, description: g.desc })),
      paths,
    });
  }),
  http.post(`${ANY}/query`, async ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const body = (await request.json().catch(() => ({}))) as { query?: string };
    return HttpResponse.json({
      response: body.query
        ? `Mock retrieval response for: ${body.query}`
        : 'Mock retrieval response',
    });
  }),
  http.post(`${ANY}/query/data`, async ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const body = (await request.json().catch(() => ({}))) as { query?: string };
    return HttpResponse.json({
      status: 'success',
      message: 'Query executed successfully',
      data: {
        entities: [],
        relationships: [],
        chunks: [
          {
            chunk_id: 'mock-chunk-1',
            file_path: '/cib/runbooks/mock-source-1.pdf',
            content: body.query
              ? `Mock structured retrieval for: ${body.query}`
              : 'Mock structured retrieval',
            reference_id: '1',
          },
        ],
        references: [
          { reference_id: '1', file_path: '/cib/runbooks/mock-source-1.pdf' },
        ],
      },
      metadata: { query_mode: 'hybrid' },
    });
  }),
  // Twin overlay query — mirrors the structured `{response, sources}`
  // contract from the backend so dev / standalone parity is honest.
  http.post(`${ANY}${TWIN}/query`, async ({ request }) => {
    const body = await recordQueryRequest(request);
    const unsafe = rejectUnsafeTwinQuery(body);
    if (unsafe) return unsafe;
    const q = typeof body.query === 'string' ? body.query : '';
    const topK = typeof body.top_k === 'number' ? body.top_k : 3;
    const responseText = q
      ? `Mock retrieval response for: ${q}`
      : 'Mock retrieval response';
    const sources = Array.from({ length: Math.min(topK, 3) }).map((_, i) => ({
      n: i + 1,
      type: 'file',
      name: `/cib/runbooks/mock-source-${i + 1}.pdf`,
      meta: `chunk ${i + 1}`,
      score: Number((0.95 - i * 0.1).toFixed(2)),
      doc_id: `mock-doc-${i + 1}`,
      chunk_id: `mock-chunk-${i + 1}`,
      // First source carries a paragraph anchor so the passthrough
      // (sanitize → drill-down params) stays exercised in MSW runs.
      anchor:
        i === 0
          ? {
              start: 0,
              end: 24,
              paragraph_idx: 0,
              paragraph_count: 1,
              confidence: 0.62,
              method: 'lexical_overlap',
            }
          : null,
    }));
    return HttpResponse.json({ response: responseText, sources });
  }),
  http.post(`${ANY}${TWIN}/query/data`, async ({ request }) => {
    const body = await recordQueryRequest(request);
    const unsafe = rejectUnsafeTwinQuery(body);
    if (unsafe) return unsafe;
    const tagFilter = body.tag_filter as
      | { all?: string[]; any?: string[] }
      | undefined;
    const q = typeof body.query === 'string' ? body.query : '';
    return HttpResponse.json({
      status: 'success',
      message: 'Query executed successfully',
      data: {
        entities: [
          {
            entity_name: 'RMAN',
            entity_type: 'TECHNOLOGY',
            source_id: 'mock-chunk-1',
            reference_id: '1',
          },
        ],
        relationships: [],
        chunks: [
          {
            chunk_id: 'mock-chunk-1',
            full_doc_id: 'mock-doc-1',
            file_path: '/cib/runbooks/mock-source-1.pdf',
            content: q
              ? `Mock structured Twin retrieval for: ${q}`
              : 'Mock structured Twin retrieval',
            reference_id: '1',
          },
        ],
        references: [
          { reference_id: '1', file_path: '/cib/runbooks/mock-source-1.pdf' },
        ],
      },
      metadata: {
        query_mode: 'hybrid',
        ...(tagFilter ? { tag_filter: tagFilter } : {}),
      },
    });
  }),
  http.post(`${ANY}${TWIN}/query/stream`, async ({ request }) => {
    // Wire format matches the real backend: NDJSON, one event per line.
    //   {"type":"stage","value":"retrieval"}
    //   {"type":"stage","value":"generation"}
    //   {"type":"token","value":"<chunk>"}
    //   {"type":"stage","value":"sources"}
    //   {"type":"status","value":"grounded"}
    //   {"type":"sources","value":[<RetrievalSource>, ...]}
    // The client parses line-by-line and ignores anything else, so
    // returning plain text here used to silently produce an empty
    // assistant turn in MSW dev mode (no tokens, no sources).
    const body = await recordQueryRequest(request);
    const unsafe = rejectUnsafeTwinQuery(body);
    if (unsafe) return unsafe;
    const q = typeof body.query === 'string' ? body.query : '';
    const topK = Math.min(typeof body.top_k === 'number' ? body.top_k : 3, 3);
    const text = q
      ? `Mock retrieval response for: ${q}`
      : 'Mock retrieval response';
    const sources = Array.from({ length: topK }).map((_, i) => {
      // Source #1 maps to a real seeded document (DOCUMENT_FIXTURES[0]) so the
      // citation -> Documents drilldown e2e can assert a true linkage. The
      // remaining sources stay synthetic. This is the deliberate handler tune
      // the retrieval citation test's follow-up (2026-06-16) called for, scoped to
      // one source so it does not re-entrench broad fixture coupling.
      const real = i === 0 ? DOCUMENT_FIXTURES[0] : undefined;
      return {
        n: i + 1,
        type: 'file' as const,
        name: real ? real.file_path : `/cib/runbooks/mock-source-${i + 1}.pdf`,
        meta: `chunk ${i + 1}`,
        score: Number((0.95 - i * 0.1).toFixed(2)),
        doc_id: real ? real.doc_id : `mock-doc-${i + 1}`,
        chunk_id: `mock-chunk-${i + 1}`,
      };
    });
    const ndjson =
      JSON.stringify({ type: 'meta', value: { model: 'deepseek-chat' } }) +
      '\n' +
      JSON.stringify({ type: 'stage', value: 'retrieval' }) +
      '\n' +
      JSON.stringify({ type: 'stage', value: 'generation' }) +
      '\n' +
      JSON.stringify({ type: 'token', value: text }) +
      '\n' +
      JSON.stringify({ type: 'stage', value: 'sources' }) +
      '\n' +
      JSON.stringify({ type: 'status', value: 'grounded' }) +
      '\n' +
      JSON.stringify({ type: 'sources', value: sources }) +
      '\n';
    return new HttpResponse(ndjson, {
      status: 200,
      headers: { 'Content-Type': 'application/x-ndjson' },
    });
  }),

  // -------------------------------------------------------------------------
  // Twin overlay endpoints
  // -------------------------------------------------------------------------
  // Folders — admin CRUD on top of an env-seeded entry. The first
  // fixture is the env-seeded default and rejects mutations with the
  // same status codes as the backend (403). Any folder added through
  // the API is "runtime" and mutable.
  http.get(`${ANY}${TWIN}/folders`, ({ request }) => {
    recordTwinFolderRequest(request);
    return HttpResponse.json(folderState);
  }),
  http.post(`${ANY}${TWIN}/folders`, async ({ request }) =>
    handleCreateFolder(request),
  ),
  http.patch(`${ANY}${TWIN}/folders/:id`, async ({ params, request }) =>
    handleUpdateFolder(String(params.id), request),
  ),
  http.delete(`${ANY}${TWIN}/folders/:id`, ({ params }) =>
    handleDeleteFolder(String(params.id)),
  ),
  http.get(`${ANY}${TWIN}/quota`, () => HttpResponse.json(quotaState)),

  // About / system identity card. Fixture is the ADMIN shape (all blocks
  // present) so the panel renders fully in dev and e2e; the reduced
  // non-admin shape is covered by the AboutSection unit tests.
  http.get(`${ANY}${TWIN}/system/about`, () =>
    HttpResponse.json({
      twin: '1.1.0',
      lightrag: {
        native: '1.4.9.11',
        composite: '1.4.9.11+memgraph-1.1.0',
      },
      admin: true,
      memgraph: {
        // A base memgraph/memgraph image: reachable, core procedures
        // exposed, no MAGE. `procedures: 0` next to a resolved `mage` is
        // NOT a state the backend can emit — a failed probe reports both
        // as null. Keep the fixture on a shape the server can produce.
        reachable: true,
        version: '3.12.0',
        mage: false,
        procedures: 3,
        error: null,
      },
      runtime: {
        python: '3.12.13',
        implementation: 'CPython',
        platform: 'Linux-6.1.0-x86_64-with-glibc2.36',
      },
      storage: {
        kv: 'MemgraphKVStorage',
        vector: 'MemgraphVectorDBStorage',
        docstatus: 'MemgraphDocStatusStorage',
        graph: 'MemgraphGraphStorage',
      },
      overlay: {
        replace_ui: true,
        mount_server: true,
        shim_native_routes: true,
      },
    }),
  ),
  http.get(`${ANY}${TWIN}/settings/api-keys`, () =>
    HttpResponse.json(apiKeyState.map(publicApiKey)),
  ),
  http.post(`${ANY}${TWIN}/settings/api-keys`, async ({ request }) => {
    const body = (await request.json()) as { name?: string };
    const name = (body?.name ?? '').trim();
    if (!name) {
      return HttpResponse.json(
        { detail: 'Name is required.' },
        { status: 422 },
      );
    }
    const minted = mintMockToken();
    const entry: ApiKeyMockEntry = {
      id: nextApiKeyId(),
      name: name.slice(0, 120),
      prefix: minted.preview,
      full_value: minted.full,
      created_at: Date.now(),
      created_by: 'mock-operator',
      last_used_at: null,
      revoked_at: null,
    };
    apiKeyState = [entry, ...apiKeyState];
    persistApiKeyState();
    return HttpResponse.json(entry, { status: 201 });
  }),
  http.delete(`${ANY}${TWIN}/settings/api-keys/:id`, ({ params }) => {
    const id = String(params.id);
    const idx = apiKeyState.findIndex((k) => k.id === id);
    if (idx === -1) {
      return HttpResponse.json(
        { detail: `API key '${id}' not found` },
        { status: 404 },
      );
    }
    const current = apiKeyState[idx];
    if (current.revoked_at === null) {
      apiKeyState[idx] = { ...current, revoked_at: Date.now() };
      persistApiKeyState();
    }
    return HttpResponse.json(publicApiKey(apiKeyState[idx]));
  }),
  // Vision ingestion settings — GET is open to any authenticated user;
  // PUT mirrors the backend's admin gate (same `admin:folders` scope the
  // folder mutations check) and mutates the state so a refetch reflects it.
  http.get(`${ANY}${TWIN}/settings/vision`, () =>
    HttpResponse.json(visionSettingsState),
  ),
  http.put(`${ANY}${TWIN}/settings/vision`, async ({ request }) => {
    const denied = rejectFolderAdminMutationIfNeeded();
    if (denied) return denied;
    const body = (await request.json().catch(() => null)) as {
      min_ocr_chars?: unknown;
      drop_classes?: unknown;
      procedure_enabled?: unknown;
    } | null;
    const minOcr = body?.min_ocr_chars;
    if (
      typeof minOcr !== 'number' ||
      !Number.isInteger(minOcr) ||
      minOcr < 0 ||
      minOcr > 100_000
    ) {
      return HttpResponse.json(
        { detail: 'min_ocr_chars must be an integer in 0..100000' },
        { status: 422 },
      );
    }
    const rawClasses = body?.drop_classes;
    if (!Array.isArray(rawClasses) || rawClasses.length > 20) {
      return HttpResponse.json(
        { detail: 'drop_classes must be a list of at most 20 slugs' },
        { status: 422 },
      );
    }
    const cleaned: string[] = [];
    for (const value of rawClasses) {
      const slug = String(value).trim().toLowerCase();
      if (!slug) continue;
      if (!VISION_DROP_CLASS_RE.test(slug)) {
        return HttpResponse.json(
          { detail: `invalid drop class '${value}' (letters/digits/-_ only)` },
          { status: 422 },
        );
      }
      cleaned.push(slug);
    }
    if (
      body?.procedure_enabled !== undefined &&
      typeof body.procedure_enabled !== 'boolean'
    ) {
      return HttpResponse.json(
        { detail: 'procedure_enabled must be a boolean' },
        { status: 422 },
      );
    }
    // Match the backend's old-client compatibility contract: omitting the
    // newly added flag preserves the current workspace activation choice.
    const procedureEnabled =
      typeof body?.procedure_enabled === 'boolean'
        ? body.procedure_enabled
        : visionSettingsState.procedure_enabled;
    visionSettingsState = {
      min_ocr_chars: minOcr,
      drop_classes: [...new Set(cleaned)].sort((a, b) => a.localeCompare(b)),
      procedure_enabled: procedureEnabled,
      procedure_available: visionSettingsState.procedure_available,
      source: 'runtime',
      updated_at: Date.now(),
      updated_by: 'mock-operator',
    };
    persistVisionSettingsState();
    recordActivity({
      id: `evt_vision_settings_${Date.now()}`,
      ts: new Date().toISOString(),
      rel: 'now',
      day: 'Today',
      kind: 'vision-settings-updated',
      sev: 'info',
      actor: { user: 'mock-operator', role: 'admin' },
      target: { type: 'settings', label: 'vision' },
      summary: `Vision settings updated: min OCR chars ${visionSettingsState.min_ocr_chars}, drop classes ${visionSettingsState.drop_classes.join(', ') || '(none)'}, procedure ingestion ${visionSettingsState.procedure_enabled ? 'enabled' : 'disabled'}`,
      meta: {
        min_ocr_chars: visionSettingsState.min_ocr_chars,
        drop_classes: visionSettingsState.drop_classes,
        procedure_enabled: visionSettingsState.procedure_enabled,
      },
    });
    return HttpResponse.json(visionSettingsState);
  }),
  // Procedure approval bundles — STATEFUL and FOLDER-BOUND like the
  // backend: a bundle is visible when the X-Twin-Folder is its own folder
  // or one of its duplicate-request folders; an UNASSIGNED bundle (no
  // folder at all — scan-created) is visible in every folder, mirroring
  // server/procedure_routes._visible_in_folder (it would otherwise be
  // reviewable from nowhere). The rest mirrors the backend state machine:
  // decisions 409 outside their legal source states, folderless
  // approve/reroute 422 without an explicit folder, and approve/reroute
  // enqueue a real document row so the Documents tab reflects the release
  // (test doctrine: MSW mutates like the backend).
  http.get(`${ANY}${TWIN}/procedures`, ({ request }) => {
    recordTwinFolderRequest(request);
    const url = new URL(request.url);
    const state = url.searchParams.get('state');
    const activeFolder = request.headers.get('X-Twin-Folder');
    const items = proceduresState.filter((bundle) => {
      if (state && String(bundle.state) !== state) return false;
      if (!activeFolder) return true;
      const folders = bundleFolders(bundle);
      return folders.length === 0 || folders.includes(activeFolder);
    });
    return HttpResponse.json(items.map(procedureSummary));
  }),
  http.get(`${ANY}${TWIN}/procedures/:id`, ({ params }) => {
    const denied = rejectFolderAdminMutationIfNeeded();
    if (denied) return denied;
    const bundle = proceduresState.find((b) => b.id === String(params.id));
    if (!bundle) {
      return HttpResponse.json({ detail: 'unknown bundle' }, { status: 404 });
    }
    return HttpResponse.json(bundle);
  }),
  http.post(
    `${ANY}${TWIN}/procedures/:id/approve`,
    async ({ params, request }) => {
      const denied = rejectFolderAdminMutationIfNeeded();
      if (denied) return denied;
      const bundle = proceduresState.find((b) => b.id === String(params.id));
      if (!bundle) {
        return HttpResponse.json({ detail: 'unknown bundle' }, { status: 404 });
      }
      if (bundle.state !== 'pending') {
        return HttpResponse.json(
          {
            detail: `bundle is ${bundle.state}; only pending bundles can be approved`,
          },
          { status: 409 },
        );
      }
      const body = (await request.json().catch(() => null)) as {
        folder?: string | null;
      } | null;
      const folders = mockBundleFolders(bundle);
      const primary = body?.folder || folders[0];
      if (!primary) {
        return HttpResponse.json(
          {
            detail:
              "bundle has no requesting folder (scan-created): pass 'folder' in the request body to choose the target folder",
          },
          { status: 422 },
        );
      }
      bundle.state = 'approved';
      bundle.reason = 'approved';
      bundle.updated_at = new Date().toISOString();
      persistProceduresState();
      enqueueProcedureDocument(bundle, primary);
      recordProcedureActivity(
        'procedure-approved',
        bundle,
        `Procedure '${bundle.file_name}' approved and enqueued in folder '${primary}'`,
      );
      return HttpResponse.json(bundle);
    },
  ),
  http.post(
    `${ANY}${TWIN}/procedures/:id/reject`,
    async ({ params, request }) => {
      const denied = rejectFolderAdminMutationIfNeeded();
      if (denied) return denied;
      const bundle = proceduresState.find((b) => b.id === String(params.id));
      if (!bundle || !['pending', 'failed'].includes(String(bundle.state))) {
        return HttpResponse.json(
          { detail: 'bundle unknown or not in a rejectable state' },
          { status: 409 },
        );
      }
      const body = (await request.json().catch(() => null)) as {
        comment?: string | null;
      } | null;
      const comment = body?.comment ?? '';
      bundle.state = 'rejected';
      bundle.reason = comment ? `rejected: ${comment}` : 'rejected';
      bundle.updated_at = new Date().toISOString();
      persistProceduresState();
      recordProcedureActivity(
        'procedure-rejected',
        bundle,
        `Procedure '${bundle.file_name}' rejected`,
      );
      return HttpResponse.json(bundle);
    },
  ),
  http.post(`${ANY}${TWIN}/procedures/:id/retry`, ({ params }) => {
    const denied = rejectFolderAdminMutationIfNeeded();
    if (denied) return denied;
    if (!visionSettingsState.procedure_enabled) {
      return HttpResponse.json(
        {
          detail:
            'procedure ingestion is disabled; enable it in Settings > Vision before retrying',
        },
        { status: 409 },
      );
    }
    const bundle = proceduresState.find((b) => b.id === String(params.id));
    if (!bundle || !['failed', 'rejected'].includes(String(bundle.state))) {
      return HttpResponse.json(
        { detail: 'bundle unknown or not retryable (failed/rejected only)' },
        { status: 409 },
      );
    }
    // Mock rerun always succeeds → back to pending review.
    bundle.state = 'pending';
    bundle.reason = 're-processed after operator retry';
    bundle.updated_at = new Date().toISOString();
    persistProceduresState();
    recordProcedureActivity(
      'procedure-retried',
      bundle,
      `Procedure '${bundle.file_name}' re-processed (now pending)`,
    );
    return HttpResponse.json(bundle);
  }),
  http.post(
    `${ANY}${TWIN}/procedures/:id/reroute-standard`,
    async ({ params, request }) => {
      const denied = rejectFolderAdminMutationIfNeeded();
      if (denied) return denied;
      const bundle = proceduresState.find((b) => b.id === String(params.id));
      if (!bundle) {
        return HttpResponse.json({ detail: 'unknown bundle' }, { status: 404 });
      }
      if (!['pending', 'failed', 'rejected'].includes(String(bundle.state))) {
        return HttpResponse.json(
          { detail: `bundle is ${bundle.state}; cannot reroute` },
          { status: 409 },
        );
      }
      const body = (await request.json().catch(() => null)) as {
        folder?: string | null;
      } | null;
      const folders = mockBundleFolders(bundle);
      const primary = body?.folder || folders[0];
      if (!primary) {
        return HttpResponse.json(
          {
            detail:
              "bundle has no requesting folder (scan-created): pass 'folder' in the request body to choose the target folder",
          },
          { status: 422 },
        );
      }
      bundle.state = 'rerouted';
      bundle.reason = `rerouted-standard into folder '${primary}'`;
      bundle.updated_at = new Date().toISOString();
      persistProceduresState();
      enqueueProcedureDocument(bundle, primary);
      recordProcedureActivity(
        'procedure-rerouted',
        bundle,
        `Procedure '${bundle.file_name}' rerouted to the standard pipeline (folder '${primary}')`,
      );
      return HttpResponse.json(bundle);
    },
  ),
  http.get(`${ANY}${TWIN}/notifications`, ({ request }) => {
    recordTwinFolderRequest(request);
    return HttpResponse.json(notificationState);
  }),
  http.post(`${ANY}${TWIN}/notifications/read-all`, () => {
    notificationState = notificationState.map((n) => ({ ...n, read: true }));
    persistNotificationState();
    return HttpResponse.json({ ok: true });
  }),
  http.delete(`${ANY}${TWIN}/notifications`, () => {
    notificationState = [];
    persistNotificationState();
    return HttpResponse.json({ ok: true });
  }),

  http.get(`${ANY}${TWIN}/health`, ({ request }) => {
    recordTwinFolderRequest(request);
    return HttpResponse.json({ status: 'ok' });
  }),

  http.get(`${ANY}${TWIN}/thesaurus`, ({ request }) => {
    recordTwinFolderRequest(request);
    return HttpResponse.json(
      tagState
        .filter(
          (tag) =>
            tag.tier !== 'requested' &&
            tag.status !== 'deprecated' &&
            tag.status !== 'rejected',
        )
        .map((tag) => ({
          tag: tag.tag,
          category: tag.category,
          def: tag.def,
        })),
    );
  }),
  http.get(`${ANY}${TWIN}/tags`, ({ request }) => {
    recordTwinFolderRequest(request);
    const gate = authGateResponse(request);
    if (gate) return gate;
    return HttpResponse.json(tagState.filter((tag) => !isArchivedTag(tag)));
  }),
  http.get(`${ANY}${TWIN}/tags/categories`, ({ request }) => {
    recordTwinFolderRequest(request);
    return HttpResponse.json(categoryState);
  }),
  http.get(`${ANY}${TWIN}/tags/categories/template`, () =>
    HttpResponse.json(TAG_CATEGORY_FIXTURES),
  ),
  http.post(`${ANY}${TWIN}/tags/categories/_import`, async ({ request }) => {
    const body = (await request.json().catch(() => null)) as
      | { id?: string; label?: string; color?: string }[]
      | null;
    if (!Array.isArray(body)) {
      return HttpResponse.json(
        { detail: 'Root must be a JSON array of category objects.' },
        { status: 400 },
      );
    }
    const bad = body.findIndex((c) => !c.id || !c.label || !c.color);
    if (bad >= 0) {
      return HttpResponse.json(
        { detail: `Category[${bad}] missing required fields: id, label, color` },
        { status: 400 },
      );
    }
    const seen = new Set<string>();
    const duplicate = body.findIndex((c) => {
      if (!c.id) return false;
      if (seen.has(c.id)) return true;
      seen.add(c.id);
      return false;
    });
    if (duplicate >= 0) {
      return HttpResponse.json(
        { detail: `Category[${duplicate}] duplicate id: ${body[duplicate].id}` },
        { status: 400 },
      );
    }
    const badColor = body.findIndex(
      (c) => !/^#[0-9a-f]{6}$/i.test(String(c.color)),
    );
    if (badColor >= 0) {
      return HttpResponse.json(
        { detail: `Category[${badColor}] color must be a #RRGGBB hex value` },
        { status: 400 },
      );
    }
    categoryState = body.map((c) => ({
      id: c.id!,
      label: c.label!,
      color: c.color!,
    }));
    persistTagCategoriesState();
    return HttpResponse.json({ ok: true, count: categoryState.length });
  }),

  http.post(`${ANY}${TWIN}/tags`, async ({ request }) => {
    const body = (await request.json()) as {
      tag: string;
      def: string;
      category: string;
    };
    const existing = tagState.find((tag) => tag.tag === body.tag);
    if (existing && !isArchivedTag(existing)) {
      return HttpResponse.json(
        { detail: `Tag '${body.tag}' already exists` },
        { status: 409 },
      );
    }
    if (existing) {
      tagState = tagState.filter((tag) => tag.tag !== body.tag);
    }
    const next = upsertTag({
      ...tagEntryStub(body.tag, 'pending-review', 'requested'),
      def: body.def,
      category: body.category,
    });
    return HttpResponse.json(next, { status: 201 });
  }),
  http.post(`${ANY}${TWIN}/tags/:name/approve`, async ({ params }) => {
    const name = String(params.name);
    e2eStats.tagApproveCalls[name] =
      (e2eStats.tagApproveCalls[name] ?? 0) + 1;
    if (e2eScenario.tagApproveDelayMs) {
      await new Promise((resolve) =>
        setTimeout(resolve, e2eScenario.tagApproveDelayMs),
      );
    }
    const current = tagState.find((t) => t.tag === name);
    if (current?.proposal_kind === 'edit' && current.target_tag) {
      const target = tagState.find((t) => t.tag === current.target_tag);
      const fields = current.proposed_fields ?? [];
      const next = upsertTag({
        ...(target ?? tagEntryStub(current.target_tag, 'active', 'edit-approved')),
        def: fields.includes('def') ? current.def : (target?.def ?? current.def),
        long_description: fields.includes('long_description')
          ? current.long_description
          : target?.long_description,
        category: fields.includes('category')
          ? current.category
          : (target?.category ?? current.category),
        aliases: fields.includes('aliases')
          ? current.aliases
          : (target?.aliases ?? current.aliases),
        last_edit: { by: 'system', at: '2026-05-29', action: 'edit-approved' },
      });
      tagState = tagState.filter((t) => t.tag !== name);
      persistTagState();
      recordTagMutation(current.target_tag, 'edit approved');
      return HttpResponse.json(next);
    }
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'active', 'approved')),
      status: 'active',
      tier: 3,
      last_edit: { by: 'system', at: '2026-05-29', action: 'approved' },
    });
    recordTagMutation(name, 'approved');
    return HttpResponse.json(next);
  }),
  http.post(`${ANY}${TWIN}/tags/:name/reject`, async ({ params, request }) => {
    const name = String(params.name);
    const body = (await request.json().catch(() => ({}))) as {
      actor?: string;
      reason?: string;
    };
    const reason = body.reason?.trim() || 'rejected';
    const current = tagState.find((t) => t.tag === name);
    const next = {
      ...(current ?? tagEntryStub(name, 'rejected', 'rejected')),
      status: 'rejected',
      tier: 3,
      reject_reason: reason,
      last_edit: { by: body.actor ?? 'system', at: '2026-05-29', action: 'rejected' },
    };
    tagState = tagState.filter((tag) => tag.tag !== name);
    persistTagState();
    recordTagMutation(name, `rejected: ${reason}`, {
      actor: body.actor,
      sev: 'warning',
      meta: { tag: name, action: 'rejected', reason },
    });
    return HttpResponse.json(next);
  }),
  http.patch(`${ANY}${TWIN}/tags/:name`, async ({ params, request }) => {
    const name = String(params.name);
    const body = (await request.json()) as Record<string, unknown>;
    const current = tagState.find((t) => t.tag === name);
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'active', 'edited')),
      category: (body.category as string) ?? current?.category ?? 'infra',
      def: (body.def as string) ?? current?.def ?? '',
      aliases: (body.aliases as string[]) ?? current?.aliases ?? [],
      last_edit: { by: 'system', at: '2026-05-29', action: 'edited' },
    });
    return HttpResponse.json(next);
  }),
  http.post(`${ANY}${TWIN}/tags/:name/suggest-edit`, async ({ params, request }) => {
    const name = String(params.name);
    const current = tagState.find((t) => t.tag === name);
    if (!current) {
      return HttpResponse.json({ detail: `Tag '${name}' not found` }, { status: 404 });
    }
    const body = (await request.json()) as {
      actor?: string;
      def?: string;
      long_description?: string;
      category?: string;
      aliases?: string[];
      justification?: string;
    };
    const proposed = {
      def: body.def ?? current.def,
      long_description: body.long_description ?? current.long_description,
      category: body.category ?? current.category,
      aliases: body.aliases ?? current.aliases,
    };
    const proposedFields: string[] = [];
    if (proposed.def !== current.def) proposedFields.push('def');
    if (proposed.long_description !== current.long_description) {
      proposedFields.push('long_description');
    }
    if (proposed.category !== current.category) proposedFields.push('category');
    if (JSON.stringify(proposed.aliases) !== JSON.stringify(current.aliases)) {
      proposedFields.push('aliases');
    }
    if (proposedFields.length === 0) {
      return HttpResponse.json(
        { detail: 'Suggest edit requires at least one changed field' },
        { status: 400 },
      );
    }
    const proposalId = `${name}__edit__${body.actor ?? 'system'}-${Date.now()}`
      .replaceAll(/[^a-zA-Z0-9_-]+/g, '-');
    const next = upsertTag({
      ...tagEntryStub(proposalId, 'pending-review', 'edit-suggested'),
      ...proposed,
      deprecates: [...current.deprecates],
      sources_count: current.sources_count,
      chunks_count: current.chunks_count,
      query_freq_30d: current.query_freq_30d,
      requested_by: body.actor ?? 'system',
      requested_at: '2026-05-29',
      justification: body.justification ?? '',
      proposal_kind: 'edit',
      target_tag: name,
      proposed_fields: proposedFields,
    });
    recordTagMutation(name, 'edit suggested');
    return HttpResponse.json(next, { status: 201 });
  }),
  http.post(`${ANY}${TWIN}/tags/:name/deprecate`, ({ params }) => {
    const name = String(params.name);
    const current = tagState.find((t) => t.tag === name);
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'deprecated', 'deprecated')),
      status: 'deprecated',
      last_edit: { by: 'system', at: '2026-05-29', action: 'deprecated' },
    });
    return HttpResponse.json(next);
  }),
  http.post(`${ANY}${TWIN}/tags/:name/reactivate`, ({ params }) => {
    const name = String(params.name);
    const current = tagState.find((t) => t.tag === name);
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'active', 'reactivated')),
      status: 'active',
      last_edit: { by: 'system', at: '2026-05-29', action: 'reactivated' },
    });
    recordTagMutation(name, 'reactivated');
    return HttpResponse.json(next);
  }),
  http.post(
    `${ANY}${TWIN}/tags/:name/synonyms`,
    async ({ params, request }) => {
      const body = (await request.json()) as { aliases: string[] };
      const name = String(params.name);
      const current = tagState.find((t) => t.tag === name);
      const next = upsertTag({
        ...(current ?? tagEntryStub(name, 'active', 'synonyms updated')),
        aliases: body.aliases,
        last_edit: { by: 'system', at: '2026-05-29', action: 'synonyms updated' },
      });
      return HttpResponse.json(next);
    },
  ),
  http.delete(`${ANY}${TWIN}/tags/:name`, async ({ params, request }) => {
    const name = String(params.name);
    const body = (await request.json().catch(() => ({}))) as {
      strategy?: 'migrate' | 'untag';
      to?: string;
    };
    const strategy = body.strategy ?? 'untag';
    if (strategy === 'migrate' && body.to && !tagState.some((t) => t.tag === body.to)) {
      return HttpResponse.json(
        { detail: `Migration target tag '${body.to}' not found` },
        { status: 404 },
      );
    }
    tagState = tagState.filter((t) => t.tag !== name);
    persistTagState();
    const affected = cascadeDeletedTagFromDocuments(name, strategy, body.to);
    recordTagMutation(
      name,
      strategy === 'migrate'
        ? `migrated to ${body.to}`
        : `deleted (${affected} docs untagged)`,
    );
    return HttpResponse.json({ ok: true });
  }),

  http.get(`${ANY}${TWIN}/activity`, ({ request }) => {
    recordTwinFolderRequest(request);
    const url = new URL(request.url);
    const filtered = activityState.filter((e) =>
      matchActivityQuery(e, url.searchParams),
    );
    const limit = Math.max(
      1,
      Math.min(Number(url.searchParams.get('limit') ?? 200) || 200, 1000),
    );
    return HttpResponse.json({
      items: filtered.slice(0, limit),
      total: filtered.length,
      nowMs: ACTIVITY_NOW_MS,
    });
  }),

  // Document overlay
  http.post(`${ANY}${TWIN}/documents/resolve-upload`, async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as {
      file_name?: unknown;
    };
    if (typeof body.file_name !== 'string' || !body.file_name.trim()) {
      return HttpResponse.json({ detail: 'file_name is required' }, { status: 400 });
    }
    const fileName = body.file_name;
    const unsafeCharacter = Array.from(fileName).some((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return codePoint < 32 || codePoint === 127;
    });
    if (
      fileName !== fileName.trim() ||
      fileName.includes('/') ||
      fileName.includes('\\') ||
      fileName.includes(':') ||
      unsafeCharacter
    ) {
      return HttpResponse.json({ detail: 'Unsafe filename detected' }, { status: 400 });
    }
    const existing = documentsState.find(
      (doc) => doc.file_path.split(/[/\\]/u).at(-1) === fileName,
    );
    if (!existing) return HttpResponse.json({ action: 'upload' });

    const activeFolder = request.headers.get('X-Twin-Folder') ?? 'default';
    const memberships = new Set(foldersForDoc(existing));
    const alreadyPresent = memberships.has(activeFolder);
    memberships.add(activeFolder);
    documentMemberships[existing.doc_id] = Array.from(memberships);
    return HttpResponse.json({
      action: alreadyPresent ? 'already_present' : 'shared',
      doc_id: existing.doc_id,
      track_id: existing.track_id ?? '',
      message: alreadyPresent
        ? `'${fileName}' is already present in folder '${activeFolder}'.`
        : `'${fileName}' was added to folder '${activeFolder}'.`,
    });
  }),
  http.get(
    `${ANY}${TWIN}/documents/:id/metadata`,
    ({ params }) => {
      const id = String(params.id);
      const doc = documentsState.find((d) => d.doc_id === id);
      return HttpResponse.json({
        tags: doc?.tags ?? [],
        tags_source: 'tagged_with',
        tags_status: 'ok',
        folder: doc?.folder ?? 'cib',
        review: doc?.review,
      });
    },
  ),
  http.post(
    `${ANY}${TWIN}/documents/:id/approve`,
    async ({ params, request }) => {
      const id = String(params.id);
      e2eStats.approveCalls[id] = (e2eStats.approveCalls[id] ?? 0) + 1;
      if (e2eScenario.approveDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, e2eScenario.approveDelayMs));
      }
      const body = (await request.json().catch(() => ({}))) as {
        actor?: string;
        edits?: Partial<Document>;
      };
      const doc = documentsState.find((d) => d.doc_id === id);
      if (!doc) return HttpResponse.json({ error: 'not found' }, { status: 404 });
      // Real contract (audit DUP-2d): approve ONLY writes
      // `metadata.review = {state:'approved', actor, at, edits?}` — it does
      // NOT flip `status` and does NOT merge `edits` into the document
      // fields, and it returns `{doc_id, review}` (webui/router.py
      // approve_document, router.py:716-774). Operator edits are recorded
      // inside the review audit payload only.
      const actor = body.actor ?? 'operator.demo';
      const edits = body.edits ?? {};
      const review: Document['review'] = {
        ...doc.review,
        state: 'approved' as const,
        actor,
        at: new Date().toISOString(),
        ...(Object.keys(edits).length > 0 ? { edits } : {}),
      };
      const updated = updateDoc(id, { review });
      if (updated) {
        recordDocumentActivity(
          'doc-approved',
          updated,
          `approved by ${actor}${body.edits ? ' with edits' : ''}`,
          { edits },
          { actor: body.actor },
        );
      }
      return HttpResponse.json({ doc_id: id, review });
    },
  ),
  http.post(
    `${ANY}${TWIN}/documents/:id/reject`,
    async ({ params, request }) => {
      const id = String(params.id);
      const body = (await request.json()) as { actor?: string; reason: string };
      const doc = documentsState.find((d) => d.doc_id === id);
      if (!doc) return HttpResponse.json({ error: 'not found' }, { status: 404 });
      const reason = body.reason?.trim() || 'rejected';
      // Real contract (audit DUP-2d): reject writes `metadata.review =
      // {state:'rejected', actor, at, justification}` and returns
      // `{doc_id, review}` — the doc row itself is untouched
      // (webui/router.py reject_document, router.py:777-838).
      const actor = body.actor ?? 'operator.demo';
      const review: Document['review'] = {
        ...doc.review,
        state: 'rejected' as const,
        actor,
        at: new Date().toISOString(),
        justification: reason,
      };
      const updated = updateDoc(id, { review });
      if (updated) {
        recordDocumentActivity(
          'doc-rejected',
          updated,
          `rejected: ${reason}`,
          { reason },
          { actor: body.actor, sev: 'warning' },
        );
      }
      return HttpResponse.json({ doc_id: id, review });
    },
  ),
  http.get(`${ANY}${TWIN}/documents/:id/folders`, ({ params }) => {
    const forbidden = rejectFolderAdminMutationIfNeeded();
    if (forbidden) return forbidden;
    const id = String(params.id);
    const doc = documentsState.find((d) => d.doc_id === id);
    if (!doc) return HttpResponse.json({ detail: 'not found' }, { status: 404 });
    return HttpResponse.json({ doc_id: id, folders: foldersForDoc(doc) });
  }),
  http.post(`${ANY}${TWIN}/documents/:id/folders`, async ({ params, request }) => {
    const forbidden = rejectFolderAdminMutationIfNeeded();
    if (forbidden) return forbidden;
    const id = String(params.id);
    const doc = documentsState.find((d) => d.doc_id === id);
    if (!doc) return HttpResponse.json({ detail: 'not found' }, { status: 404 });
    const body = (await request.json()) as { folder_id?: string };
    const folderId = body.folder_id ?? '';
    if (!isKnownMockFolder(folderId)) {
      return HttpResponse.json({ detail: 'folder not found' }, { status: 404 });
    }
    const folders = new Set(foldersForDoc(doc));
    folders.add(folderId);
    documentMemberships[id] = Array.from(folders);
    return HttpResponse.json({ doc_id: id, folders: documentMemberships[id] });
  }),
  http.delete(
    `${ANY}${TWIN}/documents/:id/folders/:folderId`,
    ({ params }) => {
      const forbidden = rejectFolderAdminMutationIfNeeded();
      if (forbidden) return forbidden;
      const id = String(params.id);
      const folderId = String(params.folderId);
      const doc = documentsState.find((d) => d.doc_id === id);
      if (!doc) return HttpResponse.json({ detail: 'not found' }, { status: 404 });
      const folders = foldersForDoc(doc);
      if (!folders.includes(folderId)) {
        return HttpResponse.json({ detail: 'folder not found' }, { status: 404 });
      }
      // Same ref-counted semantics as the two delete surfaces above —
      // one shared helper so the three paths cannot diverge (audit DUP-2b;
      // real route: routes_documents.remove_document_from_folder).
      const physicallyDeleted = removeDocFromFolderOrDelete(doc, folderId);
      if (physicallyDeleted) cascadeDocsFromGraph(new Set([id]));
      return HttpResponse.json({
        ok: true,
        doc_id: id,
        removed_folder: folderId,
        physically_deleted: physicallyDeleted,
        remaining_folders: documentMemberships[id] ?? [],
      });
    },
  ),
  http.post(`${ANY}${TWIN}/documents/bulk-delete`, async ({ request }) => {
    // Real contract (audit DUP-2b/2c): per-doc ref-counted delete
    // (`routes_documents._delete_one_document` + `_apply_membership_delete`),
    // `{deleted, failed}` payload, and HTTP **207** when any target failed
    // (`routes_documents.bulk_delete_documents`, routes_documents.py:441-468).
    const body = (await request.json()) as { actor?: string; doc_ids: string[] };
    const affected: BulkDeleteOutcome[] = [];
    const failed: string[] = [];
    const physicallyDeletedIds = new Set<string>();
    for (const rawId of body.doc_ids) {
      const id = String(rawId);
      if (e2eScenario.bulkDeleteFailIds?.includes(id)) {
        failed.push(id);
        continue;
      }
      const outcome = applyBulkDeleteForDoc(request, id);
      if (!outcome) {
        failed.push(id);
        continue;
      }
      if (outcome.physicallyDeleted) physicallyDeletedIds.add(id);
      affected.push(outcome);
    }
    // Cascade the whole physically-deleted set at once — an entity shared by
    // two docs deleted in the same batch must orphan (graph doctrine).
    cascadeDocsFromGraph(physicallyDeletedIds);
    if (e2eScenario.bulkDeleteDelayMs && e2eScenario.bulkDeleteDelayMs > 0) {
      await new Promise((res) =>
        setTimeout(res, e2eScenario.bulkDeleteDelayMs),
      );
    }
    if (affected.length > 0) {
      recordBulkDeleteActivity(body.actor, affected, failed);
    }
    return HttpResponse.json(
      { deleted: affected.length, failed },
      failed.length > 0 ? { status: 207 } : undefined,
    );
  }),
  http.post(`${ANY}${TWIN}/documents/_bulk-retag`, async ({ request }) => {
    if (e2eScenario.bulkRetagStatus) {
      return HttpResponse.json(
        { detail: `Forced bulk-retag failure ${e2eScenario.bulkRetagStatus}` },
        { status: e2eScenario.bulkRetagStatus },
      );
    }
    const body = (await request.json()) as {
      targets: string[];
      adds?: string[];
      removes?: string[];
    };
    const activeTags = new Set(
      tagState
        .filter((tag) => tag.status === 'active' && tag.tier !== 'requested')
        .map((tag) => tag.tag),
    );
    const unapproved = Array.from(
      new Set((body.adds ?? []).filter((tag) => !activeTags.has(tag))),
    ).sort((a, b) => a.localeCompare(b));
    if (unapproved.length > 0) {
      // Match the real route: unknown and pending tags are neither auto-created
      // nor attached. Returning before document mutation keeps this atomic.
      return HttpResponse.json(
        {
          detail: {
            message: 'Only active, approved tags may be attached',
            unapproved_tags: unapproved,
          },
        },
        { status: 422 },
      );
    }
    const targetIds = new Set(body.targets);
    const failed: string[] = [];
    body.targets.forEach((id) => {
      if (!documentsState.some((d) => d.doc_id === id)) failed.push(id);
    });
    documentsState = documentsState.map((doc) => {
      if (!targetIds.has(doc.doc_id)) return doc;
      const tags = new Set(doc.tags);
      (body.adds ?? []).forEach((tag) => tags.add(tag));
      (body.removes ?? []).forEach((tag) => tags.delete(tag));
      return { ...doc, tags: Array.from(tags) };
    });
    persistDocumentsState();
    return HttpResponse.json({
      updated: body.targets.length - failed.length,
      failed,
    });
  }),
  http.post(`${ANY}${TWIN}/documents/uploads/activity`, async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as {
      source?: string;
      track_id?: string;
      status?: string;
      actor?: string;
    };
    const actor = body.actor || 'system';
    activityState = [{
      id: `evt_upload_${Date.now()}`,
      ts: new Date().toISOString(),
      rel: 'now',
      day: 'Today',
      kind: 'source-uploaded',
      sev: 'info',
      actor: { user: actor, role: 'operator' },
      target: { type: 'source', label: body.source || 'uploaded source' },
      summary: `uploaded by ${actor}`,
      meta: {
        source: body.source,
        track_id: body.track_id,
        status: body.status || 'accepted',
      },
    }, ...activityState];
    persistActivityState();
    return HttpResponse.json({ ok: true });
  }),

  // Auth
  http.post(`${ANY}${TWIN}/auth/logout`, () =>
    HttpResponse.json({ ok: true }),
  ),

  // Graph — entities + relations with mutable state so PATCH survives.
  http.get(`${ANY}${TWIN}/graph/entities`, () =>
    HttpResponse.json(graphEntityState),
  ),
  http.get(`${ANY}${TWIN}/graph/relations`, () =>
    HttpResponse.json(graphRelationState),
  ),
  http.patch(`${ANY}${TWIN}/graph/entities/:id`, async ({ params, request }) => {
    const id = String(params.id);
    const patch = (await request.json()) as Partial<GraphEntity>;
    const idx = graphEntityState.findIndex((e) => e.id === id);
    if (idx < 0) {
      return HttpResponse.json({ detail: 'Unknown entity' }, { status: 404 });
    }
    const next = { ...graphEntityState[idx], ...patch };
    graphEntityState = [
      ...graphEntityState.slice(0, idx),
      next,
      ...graphEntityState.slice(idx + 1),
    ];
    persistGraphState();
    activityState = [
      {
        id: `evt_graph_entity_${id}_${Date.now()}`,
        ts: new Date().toISOString(),
        rel: 'now',
        day: 'Today',
        kind: 'graph-entity-edited',
        sev: 'info',
        actor: { user: 'operator.demo', role: 'KB Steward' },
        target: { type: 'entity', label: next.name, id },
        summary: `Graph entity ${next.name} updated`,
        meta: { entity_id: id, patch: Object.keys(patch) },
      },
      ...activityState,
    ];
    persistActivityState();
    return HttpResponse.json(next);
  }),
  http.patch(`${ANY}${TWIN}/graph/relations/:id`, async ({ params, request }) => {
    const id = String(params.id);
    const patch = (await request.json()) as Partial<GraphRelation>;
    const idx = graphRelationState.findIndex((r) => r.id === id);
    if (idx < 0) {
      return HttpResponse.json({ detail: 'Unknown relation' }, { status: 404 });
    }
    const next = { ...graphRelationState[idx], ...patch };
    graphRelationState = [
      ...graphRelationState.slice(0, idx),
      next,
      ...graphRelationState.slice(idx + 1),
    ];
    persistGraphState();
    activityState = [
      {
        id: `evt_graph_relation_${id}_${Date.now()}`,
        ts: new Date().toISOString(),
        rel: 'now',
        day: 'Today',
        kind: 'graph-relation-edited',
        sev: 'info',
        actor: { user: 'operator.demo', role: 'KB Steward' },
        target: { type: 'relation', label: next.label, id },
        summary: `Graph relation ${next.label} updated`,
        meta: { relation_id: id, patch: Object.keys(patch) },
      },
      ...activityState,
    ];
    persistActivityState();
    return HttpResponse.json(next);
  }),

  // M12 batch 3 — lifecycle (POST + DELETE)
  http.post(`${ANY}${TWIN}/graph/entities`, async ({ request }) => {
    const body = (await request.json()) as {
      name: string;
      type: GraphEntity['type'];
      summary?: string;
      tags?: readonly string[];
    };
    if (graphEntityState.some((e) => e.name === body.name)) {
      return HttpResponse.json(
        { detail: `Graph entity '${body.name}' already exists` },
        { status: 409 },
      );
    }
    const id = `kg_${body.name}`;
    const entity: GraphEntity = {
      id,
      name: body.name,
      type: body.type,
      x: 480,
      y: 310,
      mentions: 0,
      sources: 0,
      summary: body.summary ?? '',
      tags: body.tags,
    };
    graphEntityState = [...graphEntityState, entity];
    persistGraphState();
    activityState = [
      {
        id: `evt_graph_entity_${id}_${Date.now()}`,
        ts: new Date().toISOString(),
        rel: 'now',
        day: 'Today',
        kind: 'graph-entity-edited',
        sev: 'info',
        actor: { user: 'operator.demo', role: 'KB Steward' },
        target: { type: 'entity', label: entity.name, id },
        summary: `Graph entity ${entity.name} created`,
        meta: { entity_id: id, operation: 'create' },
      },
      ...activityState,
    ];
    persistActivityState();
    return HttpResponse.json(entity, { status: 201 });
  }),

  http.delete(`${ANY}${TWIN}/graph/entities/:id`, async ({ params }) => {
    const id = String(params.id);
    const idx = graphEntityState.findIndex((e) => e.id === id);
    if (idx < 0) {
      return HttpResponse.json({ detail: 'Unknown entity' }, { status: 404 });
    }
    const removed = graphEntityState[idx];
    graphEntityState = graphEntityState.filter((e) => e.id !== id);
    // Cascade — drop edges that touch the removed node so the UI stays
    // consistent with the backend's DETACH DELETE semantics.
    graphRelationState = graphRelationState.filter(
      (r) => r.source !== id && r.target !== id,
    );
    persistGraphState();
    activityState = [
      {
        id: `evt_graph_entity_${id}_${Date.now()}`,
        ts: new Date().toISOString(),
        rel: 'now',
        day: 'Today',
        kind: 'graph-entity-edited',
        sev: 'info',
        actor: { user: 'operator.demo', role: 'KB Steward' },
        target: { type: 'entity', label: removed.name, id },
        summary: `Graph entity ${removed.name} deleted`,
        meta: { entity_id: id, operation: 'delete' },
      },
      ...activityState,
    ];
    persistActivityState();
    return new HttpResponse(null, { status: 204 });
  }),

  http.post(`${ANY}${TWIN}/graph/relations`, async ({ request }) => {
    const body = (await request.json()) as {
      source: string;
      target: string;
      label: string;
      strength?: number;
    };
    const sourceExists = graphEntityState.some((e) => e.id === body.source);
    const targetExists = graphEntityState.some((e) => e.id === body.target);
    if (!sourceExists || !targetExists) {
      return HttpResponse.json(
        { detail: 'One or both endpoints are missing' },
        { status: 422 },
      );
    }
    const id = `kr_${body.source}_${body.target}_${Date.now().toString(16)}`;
    const relation: GraphRelation = {
      id,
      source: body.source,
      target: body.target,
      label: body.label.toUpperCase().replaceAll(/\s+/g, '_'),
      strength: body.strength ?? 0.5,
    };
    graphRelationState = [...graphRelationState, relation];
    persistGraphState();
    activityState = [
      {
        id: `evt_graph_relation_${id}_${Date.now()}`,
        ts: new Date().toISOString(),
        rel: 'now',
        day: 'Today',
        kind: 'graph-relation-edited',
        sev: 'info',
        actor: { user: 'operator.demo', role: 'KB Steward' },
        target: { type: 'relation', label: relation.label, id },
        summary: `Graph relation ${relation.label} created`,
        meta: { rel_id: id, operation: 'create' },
      },
      ...activityState,
    ];
    persistActivityState();
    return HttpResponse.json(relation, { status: 201 });
  }),

  http.delete(`${ANY}${TWIN}/graph/relations/:id`, async ({ params }) => {
    const id = String(params.id);
    const idx = graphRelationState.findIndex((r) => r.id === id);
    if (idx < 0) {
      return HttpResponse.json({ detail: 'Unknown relation' }, { status: 404 });
    }
    const removed = graphRelationState[idx];
    graphRelationState = graphRelationState.filter((r) => r.id !== id);
    persistGraphState();
    activityState = [
      {
        id: `evt_graph_relation_${id}_${Date.now()}`,
        ts: new Date().toISOString(),
        rel: 'now',
        day: 'Today',
        kind: 'graph-relation-edited',
        sev: 'info',
        actor: { user: 'operator.demo', role: 'KB Steward' },
        target: { type: 'relation', label: removed.label, id },
        summary: `Graph relation ${removed.label} deleted`,
        meta: { rel_id: id, operation: 'delete' },
      },
      ...activityState,
    ];
    persistActivityState();
    return new HttpResponse(null, { status: 204 });
  }),
];
