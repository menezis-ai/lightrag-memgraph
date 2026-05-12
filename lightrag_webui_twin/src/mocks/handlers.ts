/**
 * MSW request handlers — back the API routes with the in-repo fixtures.
 *
 * Each handler honors the shape declared in `src/api/resources.ts`. Filtering
 * mirrors what the tab implementations do locally on the fixture arrays, so
 * tab-side state behaves identically whether data comes from MSW or from a
 * future real backend.
 *
 * Path prefix: handlers match path-only (relative) so they intercept calls
 * regardless of `VITE_API_BASE_URL`. The fetch wrapper builds `${BASE}${path}`,
 * and msw matches against the full URL — using `/foo` with `*://*` patterns
 * keeps us origin-agnostic.
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

function matchDocumentsQuery(d: Document, params: URLSearchParams): boolean {
  const status = params.get('status');
  if (status && status !== 'all' && d.status !== status) return false;
  const q = params.get('q');
  if (q && !d.source.toLowerCase().includes(q.toLowerCase())) return false;
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
  const q = params.get('q');
  if (q) {
    const needle = q.toLowerCase();
    const hay = (e.summary + ' ' + e.target.label + ' ' + e.actor.user).toLowerCase();
    if (!hay.includes(needle)) return false;
  }
  return true;
}

export const handlers = [
  // Documents
  http.get(`${ANY}/documents`, ({ request }) => {
    const url = new URL(request.url);
    const filtered = DOCUMENT_FIXTURES.filter((d) =>
      matchDocumentsQuery(d, url.searchParams),
    );
    return HttpResponse.json({ items: filtered, total: filtered.length });
  }),

  // Workspaces
  http.get(`${ANY}/workspaces`, () => HttpResponse.json(WORKSPACE_FIXTURES)),

  // Notifications
  http.get(`${ANY}/notifications`, () => HttpResponse.json(NOTIFICATION_FIXTURES)),
  http.post(`${ANY}/notifications/read-all`, () => HttpResponse.json({ ok: true })),
  http.delete(`${ANY}/notifications`, () => HttpResponse.json({ ok: true })),

  // Thesaurus + tags
  http.get(`${ANY}/thesaurus`, () => HttpResponse.json(THESAURUS_FIXTURES)),
  http.get(`${ANY}/tags`, () => HttpResponse.json(TAG_FIXTURES)),
  http.get(`${ANY}/tags/categories`, () => HttpResponse.json(TAG_CATEGORY_FIXTURES)),

  // Activity
  http.get(`${ANY}/activity`, ({ request }) => {
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

  // OpenAPI
  http.get(`${ANY}/openapi`, () =>
    HttpResponse.json({ groups: OPENAPI_GROUPS, version: API_VERSION }),
  ),

  // Graph
  http.get(`${ANY}/graph/entities`, () => HttpResponse.json(GRAPH_ENTITY_FIXTURES)),
  http.get(`${ANY}/graph/relations`, () => HttpResponse.json(GRAPH_RELATION_FIXTURES)),
];
