/**
 * Smoke tests for the MSW handler set.
 *
 * Aligned on the sprint Étape 0 path split:
 *   - LightRAG-native endpoints stay un-prefixed (`/documents`, `/openapi`,
 *     `/health`).
 *   - Twin overlay endpoints sit under `/twin/api/*`.
 */

import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers, resetDocumentsState } from './handlers';
import {
  ACTIVITY_FIXTURES,
  ACTIVITY_NOW_MS,
  API_VERSION,
  DOCUMENT_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  FOLDER_FIXTURES,
  PORTABILITY_REPORT_HASH,
} from '../fixtures';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => {
  server.resetHandlers();
  resetDocumentsState();
});
afterAll(() => server.close());

const BASE = 'http://localhost';
const TWIN = '/twin/api';

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  expect(r.ok).toBe(true);
  return (await r.json()) as T;
}

async function startPortabilityImport(): Promise<string> {
  const form = new FormData();
  form.append(
    'bundle',
    new File(['canonical bundle'], 'staging.tar.gz', {
      type: 'application/gzip',
    }),
  );
  form.append('workspace', 'base');
  form.append('folder_map', '{}');
  const response = await fetch(`${BASE}${TWIN}/admin/portability/imports`, {
    method: 'POST',
    body: form,
  });
  expect(response.status).toBe(202);
  const job = (await response.json()) as { id: string };
  return job.id;
}

describe('MSW handlers — LightRAG-native endpoints', () => {
  it('GET /documents returns the full fixture envelope', async () => {
    const data = await getJson<{
      items: unknown[];
      total: number;
      page: number;
      page_size: number;
      status_counts: Record<string, number>;
      next_cursor: string | null;
    }>('/documents');
    expect(data.total).toBe(DOCUMENT_FIXTURES.length);
    expect(data.items).toHaveLength(DOCUMENT_FIXTURES.length);
    expect(data.page).toBe(1);
    expect(data.page_size).toBe(50);
    expect(data.status_counts.PROCESSED).toBeGreaterThan(0);
    expect(data.next_cursor).toBeNull();
  });

  it('GET /documents?status=FAILED filters the envelope', async () => {
    const data = await getJson<{ items: { status: string }[]; total: number }>(
      '/documents?status=FAILED',
    );
    expect(data.total).toBeGreaterThan(0);
    data.items.forEach((d) => expect(d.status).toBe('FAILED'));
  });

  it('GET /documents?q=oracle filters by file_path substring', async () => {
    const data = await getJson<{ items: { file_path: string }[] }>(
      '/documents?q=oracle',
    );
    expect(data.items.length).toBeGreaterThan(0);
    data.items.forEach((d) =>
      expect(d.file_path.toLowerCase()).toMatch(/oracle/),
    );
  });

  it('GET /documents/:id/chunks returns a chunk array', async () => {
    const data = await getJson<unknown[]>('/documents/d1/chunks');
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBeGreaterThan(0);
  });

  it('GET /documents/:id/chunks never returns placeholder content for uploaded text files', async () => {
    const form = new FormData();
    form.append(
      'file',
      new File(['real uploaded chunk text'], 'real-upload.md', {
        type: 'text/markdown',
      }),
    );
    const upload = await fetch(BASE + '/documents/upload', {
      method: 'POST',
      body: form,
    });
    expect(upload.ok).toBe(true);

    const docs = await getJson<{
      items: Array<{ doc_id: string; file_path: string }>;
    }>('/documents?q=real-upload');
    const doc = docs.items[0];
    const chunks = await getJson<Array<{ text: string }>>(
      `/documents/${doc.doc_id}/chunks`,
    );

    expect(chunks[0].text).toBe('real uploaded chunk text');
    expect(chunks[0].text).not.toMatch(/placeholder content/i);
  });

  it('POST /documents/upload rejects operator C3/C4 sensitivity headers', async () => {
    const form = new FormData();
    form.append(
      'file',
      new File(['classified'], 'classified.md', {
        type: 'text/markdown',
      }),
    );
    const upload = await fetch(BASE + '/documents/upload', {
      method: 'POST',
      headers: { 'X-Twin-Classification': 'C4' },
      body: form,
    });

    expect(upload.status).toBe(400);
    await expect(upload.json()).resolves.toMatchObject({
      detail: expect.stringContaining('accepts only C1 or C2'),
    });
  });

  it('GET /health returns ok', async () => {
    const data = await getJson<{ status: string }>('/health');
    expect(data.status).toBe('ok');
  });

  it('GET /openapi returns { groups, version }', async () => {
    const data = await getJson<{ groups: unknown[]; version: string }>(
      '/openapi',
    );
    expect(data.version).toBe(API_VERSION);
    expect(data.groups).toHaveLength(5);
  });

  it('POST /query returns a response envelope', async () => {
    const r = await fetch(BASE + '/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'oracle rman status' }),
    });
    expect(r.ok).toBe(true);
    expect(await r.json()).toEqual({
      response: 'Mock retrieval response for: oracle rman status',
    });
  });
});

describe('MSW handlers — Twin overlay endpoints', () => {
  it(`mirrors the ${TWIN}/linked-sources preview and optimistic lifecycle`, async () => {
    const headers = {
      'Content-Type': 'application/json',
      'X-Twin-Folder': 'sandbox',
    };
    const payload = {
      url: 'https://tenant.sharepoint.com/sites/pf/Shared/guide.pdf',
      doc_type: 'di',
      public: false,
      status: 'active',
    };

    const preview = await fetch(`${BASE}${TWIN}/linked-sources/preview`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ operation: 'create', body: payload }),
    });
    expect(preview.ok).toBe(true);
    const before = await fetch(`${BASE}${TWIN}/linked-sources`, { headers });
    await expect(before.json()).resolves.toMatchObject({ links: [] });

    const created = await fetch(`${BASE}${TWIN}/linked-sources`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
    expect(created.status).toBe(201);
    const createdBody = (await created.json()) as {
      link: { id: string; row_version: number; source_type: string };
    };
    expect(createdBody.link).toMatchObject({
      row_version: 1,
      source_type: 'sharepoint',
    });

    const patched = await fetch(
      `${BASE}${TWIN}/linked-sources/${createdBody.link.id}`,
      {
        method: 'PATCH',
        headers,
        body: JSON.stringify({ public: true, expected_version: 1 }),
      },
    );
    expect(patched.ok).toBe(true);
    await expect(patched.json()).resolves.toMatchObject({
      link: { public: true, row_version: 2 },
    });

    const disabled = await fetch(
      `${BASE}${TWIN}/linked-sources/${createdBody.link.id}/disable`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify({ expected_version: 2 }),
      },
    );
    expect(disabled.ok).toBe(true);
    await expect(disabled.json()).resolves.toMatchObject({
      link: { status: 'disabled', row_version: 3 },
    });
  });

  it('keeps linked-source optimistic conflicts mutation-free and retryable', async () => {
    await fetch(`${BASE}/__e2e/scenario`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ linkedSourceDisableConflictOnce: true }),
    });
    const linkId = '11111111-1111-4111-8111-111111111111';
    const disable = () =>
      fetch(`${BASE}${TWIN}/linked-sources/${linkId}/disable`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Twin-Folder': 'default',
        },
        body: JSON.stringify({ expected_version: 1 }),
      });

    const conflict = await disable();
    expect(conflict.status).toBe(409);
    await expect(conflict.json()).resolves.toEqual({
      detail: 'row_version mismatch — reload and retry',
    });
    const unchanged = await getJson<{
      links: Array<{ id: string; status: string; row_version: number }>;
    }>(`${TWIN}/linked-sources`);
    expect(unchanged.links.find((link) => link.id === linkId)).toMatchObject({
      status: 'active',
      row_version: 1,
    });

    const retry = await disable();
    expect(retry.ok).toBe(true);
    const stats = await getJson<{
      linkedSourceDisableCalls: number;
      linkedSourceDisableTransitions: number;
    }>('/__e2e/stats');
    expect(stats).toMatchObject({
      linkedSourceDisableCalls: 2,
      linkedSourceDisableTransitions: 1,
    });
  });

  it('blocks portability approval when dry-run findings are present', async () => {
    await fetch(`${BASE}/__e2e/scenario`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ portabilityBlockingDryRun: true }),
    });
    const jobId = await startPortabilityImport();
    const job = await getJson<{
      status: string;
      report: { blocking: Array<{ code: string }> };
    }>(`${TWIN}/admin/portability/imports/${jobId}`);
    expect(job.status).toBe('awaiting-approval');
    expect(job.report.blocking).toEqual([
      expect.objectContaining({ code: 'CLASSIFICATION_CEILING' }),
    ]);

    const approval = await fetch(
      `${BASE}${TWIN}/admin/portability/imports/${jobId}/approve`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_hash: PORTABILITY_REPORT_HASH }),
      },
    );
    expect(approval.status).toBe(409);
    const stats = await getJson<{
      portabilityApproveCalls: number;
      portabilityApproveTransitions: number;
    }>('/__e2e/stats');
    expect(stats).toMatchObject({
      portabilityApproveCalls: 1,
      portabilityApproveTransitions: 0,
    });
  });

  it('allows exactly one portability transition for concurrent approve and apply', async () => {
    await fetch(`${BASE}/__e2e/scenario`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        portabilityApproveDelayMs: 10,
        portabilityApplyDelayMs: 10,
      }),
    });
    const jobId = await startPortabilityImport();
    await getJson(`${TWIN}/admin/portability/imports/${jobId}`);
    const approve = () =>
      fetch(`${BASE}${TWIN}/admin/portability/imports/${jobId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_hash: PORTABILITY_REPORT_HASH }),
      });
    const approvals = await Promise.all([approve(), approve()]);
    expect(approvals.map((response) => response.status).sort()).toEqual([
      200, 409,
    ]);

    const apply = () =>
      fetch(`${BASE}${TWIN}/admin/portability/imports/${jobId}/apply`, {
        method: 'POST',
      });
    const applications = await Promise.all([apply(), apply()]);
    expect(applications.map((response) => response.status).sort()).toEqual([
      202, 409,
    ]);
    const stats = await getJson<{
      portabilityApproveCalls: number;
      portabilityApproveTransitions: number;
      portabilityApplyCalls: number;
      portabilityApplyTransitions: number;
    }>('/__e2e/stats');
    expect(stats).toMatchObject({
      portabilityApproveCalls: 2,
      portabilityApproveTransitions: 1,
      portabilityApplyCalls: 2,
      portabilityApplyTransitions: 1,
    });
  });

  it(`PUT ${TWIN}/settings/vision preserves procedure activation when omitted`, async () => {
    const put = (body: object) =>
      fetch(BASE + `${TWIN}/settings/vision`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

    const enable = await put({
      min_ocr_chars: 20,
      drop_classes: ['invalid'],
      procedure_enabled: true,
    });
    expect(enable.ok).toBe(true);

    const legacyUpdate = await put({
      min_ocr_chars: 40,
      drop_classes: ['logo'],
    });
    expect(legacyUpdate.ok).toBe(true);
    await expect(legacyUpdate.json()).resolves.toMatchObject({
      min_ocr_chars: 40,
      drop_classes: ['logo'],
      procedure_enabled: true,
    });
  });

  it(`GET ${TWIN}/folders returns the provisioned default folder`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/folders`);
    expect(data).toHaveLength(1);
    expect(data[0]).toMatchObject({ id: FOLDER_FIXTURES[0].id, current: true });
  });

  it(`GET ${TWIN}/tags returns the fixture entries`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/tags`);
    expect(data).toHaveLength(TAG_FIXTURES.length);
  });

  it(`POST ${TWIN}/tags/:name/suggest-edit queues an edit proposal`, async () => {
    const r = await fetch(BASE + `${TWIN}/tags/rman/suggest-edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        def: 'Updated RMAN wording',
        aliases: ['rmgr'],
        justification: 'clarify',
        actor: 'demo.qa',
      }),
    });
    expect(r.status).toBe(201);
    const proposal = (await r.json()) as {
      tag: string;
      proposal_kind: string;
      target_tag: string;
      proposed_fields: string[];
    };
    expect(proposal.proposal_kind).toBe('edit');
    expect(proposal.target_tag).toBe('rman');
    expect(proposal.proposed_fields).toEqual(
      expect.arrayContaining(['def', 'aliases']),
    );

    const tags = await getJson<Array<{ tag: string }>>(`${TWIN}/tags`);
    expect(tags.some((tag) => tag.tag === proposal.tag)).toBe(true);
  });

  it(`GET ${TWIN}/thesaurus returns the legacy projection of active tags`, async () => {
    const data = await getJson<Array<{ tag: string }>>(`${TWIN}/thesaurus`);
    const expected = TAG_FIXTURES.filter(
      (tag) =>
        tag.tier !== 'requested' &&
        tag.status !== 'deprecated' &&
        tag.status !== 'rejected',
    );
    expect(data).toHaveLength(expected.length);
    expect(data.map((entry) => entry.tag).sort()).toEqual(
      expected.map((entry) => entry.tag).sort(),
    );
  });

  it(`GET ${TWIN}/tags/categories returns the fixture array`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/tags/categories`);
    expect(data).toHaveLength(TAG_CATEGORY_FIXTURES.length);
  });

  it(`GET ${TWIN}/activity returns items + pinned nowMs`, async () => {
    const data = await getJson<{ items: unknown[]; nowMs: number }>(
      `${TWIN}/activity`,
    );
    expect(data.items.length).toBeGreaterThan(0);
    expect(data.nowMs).toBe(ACTIVITY_NOW_MS);
  });

  it(`GET ${TWIN}/activity?sev=error narrows the audit feed`, async () => {
    const data = await getJson<{ items: { sev: string }[] }>(
      `${TWIN}/activity?sev=error`,
    );
    expect(data.items.length).toBeGreaterThan(0);
    data.items.forEach((e) => expect(e.sev).toBe('error'));
  });

  it(`GET ${TWIN}/activity?q=event-id matches event ids`, async () => {
    const data = await getJson<{ items: { id: string }[]; total: number }>(
      `${TWIN}/activity?q=${ACTIVITY_FIXTURES[0].id}`,
    );
    expect(data.total).toBe(1);
    expect(data.items.map((event) => event.id)).toEqual([
      ACTIVITY_FIXTURES[0].id,
    ]);
  });

  it(`GET ${TWIN}/activity applies range and limit while total stays pre-limit`, async () => {
    const data = await getJson<{ items: { ts: string }[]; total: number }>(
      `${TWIN}/activity?range=24h&limit=1`,
    );
    const cutoff = ACTIVITY_NOW_MS - 24 * 60 * 60 * 1000;
    expect(data.items).toHaveLength(1);
    expect(data.total).toBeGreaterThan(data.items.length);
    expect(data.total).toBe(
      ACTIVITY_FIXTURES.filter((event) => Date.parse(event.ts) >= cutoff).length,
    );
  });

  it(`GET ${TWIN}/activity?resource.id matches meta doc_id`, async () => {
    await fetch(BASE + '/__e2e/activity', {
      method: 'POST',
      body: JSON.stringify({
        events: [
          {
            id: 'evt_meta_doc',
            ts: '2026-05-11T09:58:00Z',
            rel: 'now',
            day: 'Today',
            kind: 'doc-approved',
            sev: 'info',
            actor: { user: 'system', role: 'operator' },
            target: { type: 'document', label: 'doc via meta' },
            summary: 'Document approved',
            meta: { doc_id: 'doc-456' },
          },
          {
            id: 'evt_without_doc_id',
            ts: '2026-05-11T09:59:00Z',
            rel: 'now',
            day: 'Today',
            kind: 'doc-approved',
            sev: 'info',
            actor: { user: 'system', role: 'operator' },
            target: { type: 'document', label: 'no document id' },
            summary: 'Document-less event',
            meta: {},
          },
        ],
      }),
    });

    const data = await getJson<{ items: { id: string }[]; total: number }>(
      `${TWIN}/activity?resource.id=doc-456`,
    );
    expect(data.total).toBe(1);
    expect(data.items.map((event) => event.id)).toEqual(['evt_meta_doc']);
  });

  it(`GET ${TWIN}/graph/entities returns the fixture array`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/graph/entities`);
    expect(data).toHaveLength(GRAPH_ENTITY_FIXTURES.length);
  });

  it(`GET ${TWIN}/graph/relations returns the fixture array`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/graph/relations`);
    expect(data).toHaveLength(GRAPH_RELATION_FIXTURES.length);
  });

  it(`POST ${TWIN}/notifications/read-all returns ok envelope`, async () => {
    const r = await fetch(BASE + `${TWIN}/notifications/read-all`, {
      method: 'POST',
    });
    expect(r.ok).toBe(true);
    expect(await r.json()).toEqual({ ok: true });
  });

  it(`POST ${TWIN}/auth/logout returns ok envelope`, async () => {
    const r = await fetch(BASE + `${TWIN}/auth/logout`, { method: 'POST' });
    expect(r.ok).toBe(true);
    expect(await r.json()).toEqual({ ok: true });
  });

  it(`GET ${TWIN}/documents/d1/metadata returns overlay fields`, async () => {
    const data = await getJson<{ tags: string[]; folder: string }>(
      `${TWIN}/documents/d1/metadata`,
    );
    expect(Array.isArray(data.tags)).toBe(true);
    expect(typeof data.folder).toBe('string');
  });
});

describe('MSW handlers — delete cascade parity (unit + bulk)', () => {
  it(`POST ${TWIN}/documents/bulk-delete records one bulk activity event`, async () => {
    const del = await fetch(`${BASE}${TWIN}/documents/bulk-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_ids: ['d6', 'd7'], actor: 'demo.steward' }),
    });
    expect(del.ok).toBe(true);
    expect(await del.json()).toEqual({ deleted: 2, failed: [] });

    const activity = await getJson<{
      items: Array<{
        id: string;
        actor: { user: string };
        meta: Record<string, unknown>;
        summary: string;
        target: { type: string; label: string; id?: string };
      }>;
    }>(`${TWIN}/activity?kind=doc-deleted`);
    const deletes = activity.items.filter((event) =>
      Array.isArray(event.meta.doc_ids)
        ? event.meta.doc_ids.includes('d6') && event.meta.doc_ids.includes('d7')
        : false,
    );
    expect(deletes).toHaveLength(1);
    const event = deletes[0];
    expect(event.actor.user).toBe('demo.steward');
    expect(event.target).toMatchObject({ type: 'bulk', label: '2 documents' });
    expect(event.summary).toContain('2 documents physically deleted');
    expect(event.summary).toContain('cascade');
    expect(event.meta.operation).toBe('bulk-delete');
    expect(event.meta.doc_count).toBe(2);
    expect(event.meta.doc_ids).toEqual(['d6', 'd7']);
    expect(event.meta.physically_deleted_count).toBe(2);

    const d6Activity = await getJson<{ total: number; items: Array<{ id: string }> }>(
      `${TWIN}/activity?resource.id=d6`,
    );
    const d7Activity = await getJson<{ total: number; items: Array<{ id: string }> }>(
      `${TWIN}/activity?resource.id=d7`,
    );
    const missingActivity = await getJson<{ total: number }>(
      `${TWIN}/activity?resource.id=missing`,
    );
    expect(d6Activity.total).toBe(1);
    expect(d6Activity.items[0].id).toBe(event.id);
    expect(d7Activity.total).toBe(1);
    expect(d7Activity.items[0].id).toBe(event.id);
    expect(missingActivity.total).toBe(0);
  });

  it(`DELETE ${TWIN}/tags/cft untags documents in the native /documents feed`, async () => {
    const before = await getJson<{
      items: Array<{ file_path: string; tags: string[] }>;
    }>('/documents');
    const draftBefore = before.items.find(
      (doc) => doc.file_path === 'cft-vendor-api-spec-draft.pdf',
    );
    expect(draftBefore?.tags).toEqual(['cft', 'network']);

    const del = await fetch(`${BASE}${TWIN}/tags/cft`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strategy: 'untag' }),
    });
    expect(del.ok).toBe(true);

    const after = await getJson<{
      items: Array<{ file_path: string; tags: string[] }>;
    }>('/documents');
    const draftAfter = after.items.find(
      (doc) => doc.file_path === 'cft-vendor-api-spec-draft.pdf',
    );
    expect(draftAfter?.tags).toEqual(['network']);
  });

  it(`POST ${TWIN}/tags/argocd/reject records warning activity with reason`, async () => {
    const reject = await fetch(`${BASE}${TWIN}/tags/argocd/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor: 'demo.steward', reason: 'too broad' }),
    });
    expect(reject.ok).toBe(true);

    const activity = await getJson<{
      items: Array<{
        meta: Record<string, unknown>;
        sev: string;
        summary: string;
      }>;
    }>(`${TWIN}/activity?kind=tag-mutation&q=too%20broad`);
    expect(activity.items[0]).toMatchObject({
      sev: 'warning',
      summary: 'Tag argocd rejected: too broad',
    });
    expect(activity.items[0].meta.reason).toBe('too broad');

    const tagsAfterReject = await getJson<Array<{ tag: string }>>(`${TWIN}/tags`);
    expect(tagsAfterReject.some((tag) => tag.tag === 'argocd')).toBe(false);

    const recreate = await fetch(`${BASE}${TWIN}/tags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tag: 'argocd',
        def: 'GitOps controller',
        category: 'infra',
      }),
    });
    expect(recreate.status).toBe(201);
  });

  it(`POST ${TWIN}/documents/d6/reject records doc-rejected activity with reason`, async () => {
    const reject = await fetch(`${BASE}${TWIN}/documents/d6/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        actor: 'demo.steward',
        reason: 'policy mismatch',
      }),
    });
    expect(reject.ok).toBe(true);

    const activity = await getJson<{
      items: Array<{
        meta: Record<string, unknown>;
        sev: string;
        summary: string;
      }>;
    }>(`${TWIN}/activity?kind=doc-rejected&q=policy%20mismatch`);
    expect(activity.items[0]).toMatchObject({
      sev: 'warning',
      summary: 'rejected: policy mismatch',
    });
    expect(activity.items[0].meta.reason).toBe('policy mismatch');
  });

  it('DELETE /documents/d4 cascades to graph entities orphaned by d4', async () => {
    // d4 is the only fixture doc sourcing e_memgraph / e_mage / e_lightrag / e_cypher.
    const before = await getJson<unknown[]>(`${TWIN}/graph/entities`);
    const beforeIds = new Set((before as Array<{ id: string }>).map((e) => e.id));
    expect(beforeIds.has('e_memgraph')).toBe(true);

    const del = await fetch(`${BASE}/documents/d4`, { method: 'DELETE' });
    expect(del.ok).toBe(true);

    const after = await getJson<unknown[]>(`${TWIN}/graph/entities`);
    const afterIds = new Set((after as Array<{ id: string }>).map((e) => e.id));
    for (const id of ['e_memgraph', 'e_mage', 'e_lightrag', 'e_cypher']) {
      expect(afterIds.has(id)).toBe(false);
    }
  });

  it(`POST ${TWIN}/query/stream mirrors the ordered backend NDJSON contract`, async () => {
    const res = await fetch(`${BASE}${TWIN}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'hello', top_k: 2 }),
    });
    expect(res.ok).toBe(true);
    expect(res.headers.get('content-type')).toMatch(/application\/x-ndjson/);
    const text = await res.text();
    const events = text
      .split('\n')
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as { type: string; value: unknown });
    const tokens = events.filter((e) => e.type === 'token');
    const sources = events.filter((e) => e.type === 'sources');
    expect(events.map((event) => event.type)).toEqual([
      'meta',
      'stage',
      'stage',
      'token',
      'stage',
      'status',
      'sources',
    ]);
    expect(events.filter((event) => event.type === 'stage').map((event) => event.value)).toEqual([
      'retrieval',
      'generation',
      'sources',
    ]);
    expect(tokens.length).toBeGreaterThan(0);
    expect(sources.length).toBe(1);
    expect(Array.isArray(sources[0].value)).toBe(true);
    expect((sources[0].value as unknown[]).length).toBe(2);
  });
});

// ───────────────────────────────────────────────────────────────────────────
// Real-backend state machine parity (audit 2026-07-02, DUP-2). These pin the
// mock to the REAL contracts so e2e can no longer go green against an
// imaginary backend:
//   (a) upload is async  — native LightRAG upload only ENQUEUES; the row
//       lands PENDING and flips PROCESSED on a later poll;
//   (b) delete is ref-counted — native_shims._delete_or_unshare /
//       routes_documents._apply_membership_delete;
//   (c) bulk-delete returns 207 on partial failure —
//       routes_documents.bulk_delete_documents;
//   (d) approve only writes metadata.review — webui/router.approve_document.
// ───────────────────────────────────────────────────────────────────────────
describe('MSW handlers — real-backend state machine parity (audit DUP-2)', () => {
  async function uploadFile(name: string): Promise<string> {
    const form = new FormData();
    form.append('file', new File(['# body'], name, { type: 'text/markdown' }));
    const res = await fetch(`${BASE}/documents/upload`, {
      method: 'POST',
      body: form,
    });
    expect(res.ok).toBe(true);
    const body = (await res.json()) as { status: string; track_id: string };
    expect(body.status).toBe('success');
    return body.track_id;
  }

  it('upload is asynchronous: PENDING on the first poll, PROCESSED on the second', async () => {
    await uploadFile('async-upload.md');

    type Envelope = {
      items: Array<{
        doc_id: string;
        status: string;
        chunks_count: number | null;
      }>;
    };
    const first = await getJson<Envelope>('/documents?q=async-upload');
    expect(first.items).toHaveLength(1);
    expect(first.items[0].status).toBe('PENDING');
    expect(first.items[0].chunks_count).toBeNull();

    const second = await getJson<Envelope>('/documents?q=async-upload');
    expect(second.items[0].status).toBe('PROCESSED');
    expect(second.items[0].chunks_count).toBe(1);

    // Terminal state is stable on subsequent polls.
    const third = await getJson<Envelope>('/documents?q=async-upload');
    expect(third.items[0].status).toBe('PROCESSED');
  });

  it('single DELETE un-shares a multi-folder doc and only physically deletes on the last membership', async () => {
    // Share d1 into a second folder first (memberships: default + demo).
    const share = await fetch(`${BASE}${TWIN}/documents/d1/folders`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_id: 'demo' }),
    });
    expect(share.ok).toBe(true);

    // Delete from the active folder 'default' → un-share only.
    const unshare = await fetch(`${BASE}/documents/d1`, {
      method: 'DELETE',
      headers: { 'X-Twin-Folder': 'default' },
    });
    expect(unshare.status).toBe(200);
    const folders = await getJson<{ folders: string[] }>(
      `${TWIN}/documents/d1/folders`,
    );
    expect(folders.folders).toEqual(['demo']);
    const docs = await getJson<{ items: Array<{ doc_id: string }> }>(
      '/documents',
    );
    expect(docs.items.some((d) => d.doc_id === 'd1')).toBe(true);

    // Doc is no longer visible in 'default' → real shim 404s there.
    const notVisible = await fetch(`${BASE}/documents/d1`, {
      method: 'DELETE',
      headers: { 'X-Twin-Folder': 'default' },
    });
    expect(notVisible.status).toBe(404);

    // Delete from its LAST folder → physical delete.
    const last = await fetch(`${BASE}/documents/d1`, {
      method: 'DELETE',
      headers: { 'X-Twin-Folder': 'demo' },
    });
    expect(last.status).toBe(200);
    const after = await getJson<{ items: Array<{ doc_id: string }> }>(
      '/documents',
    );
    expect(after.items.some((d) => d.doc_id === 'd1')).toBe(false);
  });

  it(`POST ${TWIN}/documents/bulk-delete returns 207 with {deleted, failed} on partial failure`, async () => {
    const res = await fetch(`${BASE}${TWIN}/documents/bulk-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_ids: ['d1', 'does-not-exist'] }),
    });
    expect(res.status).toBe(207);
    expect(await res.json()).toEqual({
      deleted: 1,
      failed: ['does-not-exist'],
    });
  });

  it(`POST ${TWIN}/documents/bulk-delete un-shares (not deletes) a doc shared into another folder`, async () => {
    await fetch(`${BASE}${TWIN}/documents/d2/folders`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_id: 'demo' }),
    });
    const res = await fetch(`${BASE}${TWIN}/documents/bulk-delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Twin-Folder': 'default',
      },
      body: JSON.stringify({ doc_ids: ['d2'] }),
    });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ deleted: 1, failed: [] });
    // The physical record survives — it was only un-shared from 'default'.
    const docs = await getJson<{ items: Array<{ doc_id: string }> }>(
      '/documents',
    );
    expect(docs.items.some((d) => d.doc_id === 'd2')).toBe(true);
    const folders = await getJson<{ folders: string[] }>(
      `${TWIN}/documents/d2/folders`,
    );
    expect(folders.folders).toEqual(['demo']);
  });

  it(`POST ${TWIN}/documents/:id/approve only writes review — no status flip, no edits merge`, async () => {
    const res = await fetch(`${BASE}${TWIN}/documents/d6/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        actor: 'demo.steward',
        edits: { content_summary: 'edited summary that must NOT be merged' },
      }),
    });
    expect(res.ok).toBe(true);
    const receipt = (await res.json()) as {
      doc_id: string;
      review: { state: string; actor: string; edits?: Record<string, unknown> };
    };
    expect(receipt.doc_id).toBe('d6');
    expect(receipt.review.state).toBe('approved');
    expect(receipt.review.actor).toBe('demo.steward');
    expect(receipt.review.edits).toEqual({
      content_summary: 'edited summary that must NOT be merged',
    });

    const docs = await getJson<{
      items: Array<{
        doc_id: string;
        status: string;
        content_summary: string;
        review?: { state: string };
      }>;
    }>('/documents');
    const d6 = docs.items.find((d) => d.doc_id === 'd6');
    // Status untouched (fixture value), summary untouched, review persisted.
    expect(d6?.status).toBe('PROCESSING');
    expect(d6?.content_summary).toBe(
      'Vendor-provided spec — needs sign-off by a reviewer before retrieval. Confidence sourcing uncertain.',
    );
    expect(d6?.review?.state).toBe('approved');
  });
});
