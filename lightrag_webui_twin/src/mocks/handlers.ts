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
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  FOLDER_FIXTURES,
} from '../fixtures';
import type { ActivityEvent } from '../types/activity';
import type { Document, DocumentStatus } from '../types/document';
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
  documentsState = cloneDocuments(DOCUMENT_FIXTURES);
  resetDocumentMemberships();
  categoryState = cloneTagCategories(TAG_CATEGORY_FIXTURES);
  tagState = cloneTags(TAG_FIXTURES);
  notificationState = cloneNotifications(NOTIFICATION_FIXTURES);
  activityState = cloneActivity(ACTIVITY_FIXTURES);
  apiKeyState = [];
  apiKeyCounter = 0;
  quotaState = defaultQuotaState();
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
  e2eStats.approveCalls = {};
  e2eStats.tagApproveCalls = {};
  e2eStats.folderRequests = [];
  e2eStats.queryRequests = [];
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

function matchActivityQuery(e: ActivityEvent, params: URLSearchParams): boolean {
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
  if (resourceId && e.target.id !== resourceId) return false;
  const q = params.get('q');
  if (q) {
    const needle = q.toLowerCase();
    const hay = (e.summary + ' ' + e.target.label + ' ' + e.actor.user).toLowerCase();
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
  http.get(`${ANY}/__e2e/stats`, () =>
    HttpResponse.json({
      approveCalls: e2eStats.approveCalls,
      tagApproveCalls: e2eStats.tagApproveCalls,
      folderRequests: e2eStats.folderRequests,
      queryRequests: e2eStats.queryRequests,
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
    // Operator MIP classification rides as an HTTP header. Mirror the real
    // backend: when present, the operator value lands on the doc as a
    // structured classification payload (source_format 'operator').
    const operatorClass = request.headers.get('X-Twin-Classification');
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
    const docId = `uploaded_${uploadSeq}`;
    uploadedTrackDocs.set(trackId, docId);
    const text =
      file instanceof File && file.type.startsWith('text/')
        ? await file.text()
        : '';
    if (text.trim()) uploadedDocText.set(docId, text);
    documentsState = [
      {
        doc_id: docId,
        track_id: trackId,
        file_path: name,
        content_summary: `${name} uploaded by e2e`,
        content_length: file instanceof File ? file.size : 0,
        status: 'PROCESSED',
        chunks_count: 1,
        created_at: '2026-06-02T00:00:00Z',
        updated_at: '2026-06-02T00:00:00Z',
        error_msg: null,
        metadata: {
          mime: file instanceof File ? file.type : 'text/plain',
          uploader: 'e2e',
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
    documentsState = documentsState.filter((d) => d.doc_id !== id);
    persistDocumentsState();
    cascadeDocsFromGraph(new Set([id]));
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
    const paths: Record<string, Record<string, unknown>> = {};
    for (const g of OPENAPI_GROUPS) {
      for (const ep of g.endpoints) {
        paths[ep.p] ??= {};
        paths[ep.p][ep.m.toLowerCase()] = {
          tags: [g.id],
          summary: ep.s,
          responses: { '200': { description: 'OK' } },
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
    const body = (await recordQueryRequest(request)) as {
      query?: string;
      top_k?: number;
      chunk_top_k?: number;
      user_prompt?: string;
      enable_rerank?: boolean;
    };
    const q = body.query ?? '';
    const topK = body.top_k ?? 3;
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
    }));
    return HttpResponse.json({ response: responseText, sources });
  }),
  http.post(`${ANY}${TWIN}/query/data`, async ({ request }) => {
    const body = (await recordQueryRequest(request)) as {
      query?: string;
      tag_filter?: { all?: string[]; any?: string[] };
    };
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
            content: body.query
              ? `Mock structured Twin retrieval for: ${body.query}`
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
        ...(body.tag_filter ? { tag_filter: body.tag_filter } : {}),
      },
    });
  }),
  http.post(`${ANY}${TWIN}/query/stream`, async ({ request }) => {
    // Wire format matches the real backend: NDJSON, one event per line.
    //   {"type":"token","value":"<chunk>"}
    //   {"type":"sources","value":[<RetrievalSource>, ...]}
    // The client parses line-by-line and ignores anything else, so
    // returning plain text here used to silently produce an empty
    // assistant turn in MSW dev mode (no tokens, no sources).
    const body = (await recordQueryRequest(request)) as {
      query?: string;
      top_k?: number;
    };
    const q = body.query ?? '';
    const topK = Math.min(body.top_k ?? 3, 3);
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
      JSON.stringify({ type: 'token', value: text }) +
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
    return HttpResponse.json(tagState);
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
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'rejected', 'rejected')),
      status: 'rejected',
      tier: 3,
      reject_reason: reason,
      last_edit: { by: body.actor ?? 'system', at: '2026-05-29', action: 'rejected' },
    });
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
    return HttpResponse.json({
      items: filtered,
      total: filtered.length,
      nowMs: ACTIVITY_NOW_MS,
    });
  }),

  // Document overlay
  http.get(
    `${ANY}${TWIN}/documents/:id/metadata`,
    ({ params }) => {
      const id = String(params.id);
      const doc = documentsState.find((d) => d.doc_id === id);
      return HttpResponse.json({
        tags: doc?.tags ?? [],
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
      const updated = updateDoc(id, {
        ...body.edits,
        status: 'PROCESSED',
        review: { ...doc.review!, state: 'approved' as const },
      });
      if (updated) {
        recordDocumentActivity(
          'doc-approved',
          updated,
          `approved by ${body.actor ?? 'operator.demo'}${body.edits ? ' with edits' : ''}`,
          { edits: body.edits ?? {} },
          { actor: body.actor },
        );
      }
      return HttpResponse.json(updated);
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
      const updated = updateDoc(id, {
        review: {
          ...doc.review!,
          state: 'rejected' as const,
          justification: reason,
        },
      });
      if (updated) {
        recordDocumentActivity(
          'doc-rejected',
          updated,
          `rejected: ${reason}`,
          { reason },
          { actor: body.actor, sev: 'warning' },
        );
      }
      return HttpResponse.json(updated);
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
      if (folders.length === 1 && folders[0] === folderId) {
        documentsState = documentsState.filter((d) => d.doc_id !== id);
        delete documentMemberships[id];
        persistDocumentsState();
        cascadeDocsFromGraph(new Set([id]));
        return HttpResponse.json({
          ok: true,
          doc_id: id,
          removed_folder: folderId,
          physically_deleted: true,
          remaining_folders: [],
        });
      }
      documentMemberships[id] = folders.filter((folder) => folder !== folderId);
      return HttpResponse.json({
        ok: true,
        doc_id: id,
        removed_folder: folderId,
        physically_deleted: false,
        remaining_folders: documentMemberships[id],
      });
    },
  ),
  http.post(`${ANY}${TWIN}/documents/bulk-delete`, async ({ request }) => {
    const body = (await request.json()) as { actor?: string; doc_ids: string[] };
    const ids = new Set(body.doc_ids);
    const deletedDocs = documentsState.filter((d) => ids.has(d.doc_id));
    documentsState = documentsState.filter((d) => !ids.has(d.doc_id));
    deletedDocs.forEach((doc) => {
      delete documentMemberships[doc.doc_id];
      recordDocumentActivity(
        'doc-deleted',
        doc,
        `deleted by ${body.actor ?? 'operator.demo'}`,
        {
          folder: doc.folder,
          operation: 'bulk-delete',
          physically_deleted: true,
        },
        { actor: body.actor },
      );
    });
    ids.forEach((id) => {
      delete documentMemberships[id];
    });
    persistDocumentsState();
    cascadeDocsFromGraph(ids);
    if (e2eScenario.bulkDeleteDelayMs && e2eScenario.bulkDeleteDelayMs > 0) {
      await new Promise((res) =>
        setTimeout(res, e2eScenario.bulkDeleteDelayMs),
      );
    }
    const failed = body.doc_ids.filter(
      (id) => !deletedDocs.some((doc) => doc.doc_id === id),
    );
    return HttpResponse.json({ deleted: deletedDocs.length, failed });
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
