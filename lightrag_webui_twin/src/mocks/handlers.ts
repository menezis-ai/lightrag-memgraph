/**
 * MSW request handlers — back the API routes with the in-repo fixtures.
 *
 * Path layout mirrors the production split (Étape 0 sprint 2026-05-29):
 *   - LightRAG-native paths: `/documents`, `/documents/:id/chunks`,
 *     `/documents/:id/scan`, `/health`, `/openapi`, `/pipeline_status`.
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
  DOCUMENT_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  NOTIFICATION_FIXTURES,
  OPENAPI_GROUPS,
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  THESAURUS_FIXTURES,
  WORKSPACE_FIXTURES,
} from '../fixtures';
import type { ActivityEvent } from '../types/activity';
import type { Document } from '../types/document';
import type { TagCategory, TagEntry } from '../types/tag';

const ANY = '*';
const TWIN = '/twin/api';

/**
 * Mutable document store. Initialized from DOCUMENT_FIXTURES at module load,
 * then mutated by approve / reject / delete handlers so the UI sees state
 * changes when the host re-fetches via useDocuments() invalidation.
 *
 * Reset via `resetDocumentsState()` between tests so suites don't pollute
 * each other.
 */
let documentsState: Document[] = DOCUMENT_FIXTURES.map((d) => ({ ...d }));
let categoryState: TagCategory[] = TAG_CATEGORY_FIXTURES.map((c) => ({ ...c }));
let tagState: TagEntry[] = TAG_FIXTURES.map((t) => ({ ...t, aliases: [...t.aliases] }));
let uploadSeq = 0;
const uploadedTrackDocs = new Map<string, string>();

interface E2eScenario {
  bulkRetagStatus?: number;
  approveDelayMs?: number;
  trackStatusMode?: 'empty' | 'processed' | 'timeout';
  authGate?: boolean;
}

const e2eScenario: E2eScenario = {};
const e2eStats = {
  approveCalls: {} as Record<string, number>,
};

export function resetDocumentsState(): void {
  documentsState = DOCUMENT_FIXTURES.map((d) => ({ ...d }));
  categoryState = TAG_CATEGORY_FIXTURES.map((c) => ({ ...c }));
  tagState = TAG_FIXTURES.map((t) => ({ ...t, aliases: [...t.aliases] }));
  uploadedTrackDocs.clear();
  uploadSeq = 0;
  e2eScenario.bulkRetagStatus = undefined;
  e2eScenario.approveDelayMs = undefined;
  e2eScenario.trackStatusMode = undefined;
  e2eScenario.authGate = undefined;
  e2eStats.approveCalls = {};
}

function updateDoc(id: string, patch: Partial<Document>): Document | null {
  const idx = documentsState.findIndex((d) => d.doc_id === id);
  if (idx < 0) return null;
  documentsState[idx] = { ...documentsState[idx], ...patch };
  return documentsState[idx];
}

function matchDocumentsQuery(d: Document, params: URLSearchParams): boolean {
  const status = params.get('status');
  if (status && status !== 'all' && d.status !== status) return false;
  const q = params.get('q');
  if (q && !d.file_path.toLowerCase().includes(q.toLowerCase())) return false;
  const tag = params.get('tag');
  if (tag && !d.tags.includes(tag)) return false;
  return true;
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
  return tag;
}

function authGateResponse(
  request: Request,
): ReturnType<typeof HttpResponse.json> | undefined {
  if (!e2eScenario.authGate) return undefined;
  if (request.headers.get('Authorization')) return undefined;
  return HttpResponse.json(
    { detail: 'Unauthorized: Basic Auth required' },
    {
      status: 401,
      headers: { 'WWW-Authenticate': 'Basic realm="Twin RAG"' },
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
    metadata: patch.metadata ?? { classification: 'internal' },
    type: patch.type ?? 'file',
    tags: patch.tags ?? [],
    workspace: patch.workspace ?? 'cib',
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
    return HttpResponse.json({ ok: true, scenario: e2eScenario });
  }),
  http.get(`${ANY}/__e2e/stats`, () =>
    HttpResponse.json({ approveCalls: e2eStats.approveCalls }),
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
    return HttpResponse.json({ ok: true, ids: next.map((doc) => doc.doc_id) });
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
    return HttpResponse.json({ items: filtered, total: filtered.length });
  }),
  http.get(`${ANY}/documents/:id/chunks`, ({ params, request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    const id = String(params.id);
    const doc = documentsState.find((d) => d.doc_id === id);
    const count = doc?.chunks_count ?? 0;
    const chunks = Array.from({ length: Math.min(count, 6) }, (_, i) => ({
      chunk_id: `${id}_c${i}`,
      order: i,
      text: `Chunk ${i + 1} of ${doc?.file_path ?? id} — placeholder content.`,
    }));
    return HttpResponse.json(chunks);
  }),
  http.post(`${ANY}/documents/:id/scan`, ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    return HttpResponse.json({ ok: true });
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
    uploadSeq += 1;
    const trackId = `track_${name.replace(/[^a-z0-9]+/gi, '_').toLowerCase()}_${uploadSeq}`;
    const docId = `uploaded_${uploadSeq}`;
    uploadedTrackDocs.set(trackId, docId);
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
        metadata: { mime: file instanceof File ? file.type : 'text/plain', uploader: 'e2e' },
        type: 'file',
        tags: [],
        workspace: 'cib',
        visibility: 'private',
      },
      ...documentsState,
    ];
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
      return HttpResponse.json({
        track_id: trackId,
        documents: docId
          ? [{ id: docId, status: 'processing', file_path: documentsState.find((d) => d.doc_id === docId)?.file_path ?? docId }]
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
    return HttpResponse.json({ ok: true });
  }),
  http.get(`${ANY}/health`, ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    return HttpResponse.json({ status: 'ok', version: API_VERSION });
  }),
  http.get(`${ANY}/pipeline_status`, () =>
    HttpResponse.json({ busy: false, job_count: 0, latest_message: null }),
  ),
  http.get(`${ANY}/openapi`, ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith(TWIN)) return undefined;
    return HttpResponse.json({ groups: OPENAPI_GROUPS, version: API_VERSION });
  }),

  // -------------------------------------------------------------------------
  // Twin overlay endpoints
  // -------------------------------------------------------------------------
  http.get(`${ANY}${TWIN}/workspaces`, () =>
    HttpResponse.json(WORKSPACE_FIXTURES),
  ),
  http.get(`${ANY}${TWIN}/notifications`, () =>
    HttpResponse.json(NOTIFICATION_FIXTURES),
  ),
  http.post(`${ANY}${TWIN}/notifications/read-all`, () =>
    HttpResponse.json({ ok: true }),
  ),
  http.delete(`${ANY}${TWIN}/notifications`, () =>
    HttpResponse.json({ ok: true }),
  ),

  http.get(`${ANY}${TWIN}/health`, () => HttpResponse.json({ status: 'ok' })),

  http.get(`${ANY}${TWIN}/thesaurus`, () =>
    HttpResponse.json(THESAURUS_FIXTURES),
  ),
  http.get(`${ANY}${TWIN}/tags`, ({ request }) => {
    const gate = authGateResponse(request);
    if (gate) return gate;
    return HttpResponse.json(tagState);
  }),
  http.get(`${ANY}${TWIN}/tags/categories`, () => HttpResponse.json(categoryState)),
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
    categoryState = body.map((c) => ({
      id: c.id!,
      label: c.label!,
      color: c.color!,
    }));
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
  http.post(`${ANY}${TWIN}/tags/:name/approve`, ({ params }) => {
    const name = String(params.name);
    const current = tagState.find((t) => t.tag === name);
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'active', 'approved')),
      status: 'active',
      tier: 3,
      last_edit: { by: 'system', at: '2026-05-29', action: 'approved' },
    });
    return HttpResponse.json(next);
  }),
  http.post(`${ANY}${TWIN}/tags/:name/reject`, ({ params }) => {
    const name = String(params.name);
    const current = tagState.find((t) => t.tag === name);
    const next = upsertTag({
      ...(current ?? tagEntryStub(name, 'rejected', 'rejected')),
      status: 'rejected',
      tier: 'requested',
      last_edit: { by: 'system', at: '2026-05-29', action: 'rejected' },
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
  http.delete(`${ANY}${TWIN}/tags/:name`, ({ params }) => {
    tagState = tagState.filter((t) => t.tag !== String(params.name));
    return HttpResponse.json({ ok: true });
  }),

  http.get(`${ANY}${TWIN}/activity`, ({ request }) => {
    const url = new URL(request.url);
    const filtered = ACTIVITY_FIXTURES.filter((e) =>
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
        workspace: doc?.workspace ?? 'cib',
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
        edits?: Partial<Document>;
      };
      const doc = documentsState.find((d) => d.doc_id === id);
      if (!doc) return HttpResponse.json({ error: 'not found' }, { status: 404 });
      const updated = updateDoc(id, {
        ...(body.edits ?? {}),
        status: 'PROCESSED',
        review: { ...doc.review!, state: 'approved' as const },
      });
      return HttpResponse.json(updated);
    },
  ),
  http.post(
    `${ANY}${TWIN}/documents/:id/reject`,
    async ({ params, request }) => {
      const id = String(params.id);
      const body = (await request.json()) as { reason: string };
      const doc = documentsState.find((d) => d.doc_id === id);
      if (!doc) return HttpResponse.json({ error: 'not found' }, { status: 404 });
      const updated = updateDoc(id, {
        review: {
          ...doc.review!,
          state: 'rejected' as const,
          justification: body.reason,
        },
      });
      return HttpResponse.json(updated);
    },
  ),
  http.post(`${ANY}${TWIN}/documents/bulk-delete`, async ({ request }) => {
    const body = (await request.json()) as { doc_ids: string[] };
    const ids = new Set(body.doc_ids);
    documentsState = documentsState.filter((d) => !ids.has(d.doc_id));
    return HttpResponse.json({ deleted: body.doc_ids.length });
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
    return HttpResponse.json({
      updated: body.targets.length - failed.length,
      failed,
    });
  }),

  // Auth
  http.post(`${ANY}${TWIN}/auth/logout`, () =>
    HttpResponse.json({ ok: true }),
  ),

  // Graph
  http.get(`${ANY}${TWIN}/graph/entities`, () =>
    HttpResponse.json(GRAPH_ENTITY_FIXTURES),
  ),
  http.get(`${ANY}${TWIN}/graph/relations`, () =>
    HttpResponse.json(GRAPH_RELATION_FIXTURES),
  ),
];
