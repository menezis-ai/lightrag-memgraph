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

export function resetDocumentsState(): void {
  documentsState = DOCUMENT_FIXTURES.map((d) => ({ ...d }));
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

function tagEntryStub(name: string, status: string, action: string) {
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

export const handlers = [
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
  http.get(`${ANY}${TWIN}/tags`, () => HttpResponse.json(TAG_FIXTURES)),
  http.get(`${ANY}${TWIN}/tags/categories`, () =>
    HttpResponse.json(TAG_CATEGORY_FIXTURES),
  ),

  http.post(`${ANY}${TWIN}/tags`, async ({ request }) => {
    const body = (await request.json()) as {
      tag: string;
      def: string;
      category: string;
    };
    return HttpResponse.json(
      {
        ...tagEntryStub(body.tag, 'pending-review', 'requested'),
        def: body.def,
        category: body.category,
      },
      { status: 201 },
    );
  }),
  http.post(`${ANY}${TWIN}/tags/:name/approve`, ({ params }) =>
    HttpResponse.json(tagEntryStub(String(params.name), 'active', 'approved')),
  ),
  http.post(`${ANY}${TWIN}/tags/:name/reject`, ({ params }) =>
    HttpResponse.json(
      tagEntryStub(String(params.name), 'rejected', 'rejected'),
    ),
  ),
  http.patch(`${ANY}${TWIN}/tags/:name`, async ({ params, request }) => {
    const body = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      ...tagEntryStub(String(params.name), 'active', 'edited'),
      category: (body.category as string) ?? 'infra',
      def: (body.def as string) ?? '',
      aliases: (body.aliases as string[]) ?? [],
    });
  }),
  http.post(`${ANY}${TWIN}/tags/:name/deprecate`, ({ params }) =>
    HttpResponse.json(
      tagEntryStub(String(params.name), 'deprecated', 'deprecated'),
    ),
  ),
  http.post(
    `${ANY}${TWIN}/tags/:name/synonyms`,
    async ({ params, request }) => {
      const body = (await request.json()) as { aliases: string[] };
      return HttpResponse.json({
        ...tagEntryStub(
          String(params.name),
          'active',
          'synonyms updated',
        ),
        aliases: body.aliases,
      });
    },
  ),
  http.delete(`${ANY}${TWIN}/tags/:name`, () =>
    HttpResponse.json({ ok: true }),
  ),

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
