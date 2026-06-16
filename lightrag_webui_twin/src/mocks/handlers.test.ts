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
import { handlers } from './handlers';
import {
  ACTIVITY_NOW_MS,
  API_VERSION,
  DOCUMENT_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  FOLDER_FIXTURES,
} from '../fixtures';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const BASE = 'http://localhost';
const TWIN = '/twin/api';

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  expect(r.ok).toBe(true);
  return (await r.json()) as T;
}

describe('MSW handlers — LightRAG-native endpoints', () => {
  it('GET /documents returns the full fixture envelope', async () => {
    const data = await getJson<{ items: unknown[]; total: number }>('/documents');
    expect(data.total).toBe(DOCUMENT_FIXTURES.length);
    expect(data.items).toHaveLength(DOCUMENT_FIXTURES.length);
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
  it(`GET ${TWIN}/folders returns the provisioned default folder`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/folders`);
    expect(data).toHaveLength(1);
    expect(data[0]).toMatchObject({ id: FOLDER_FIXTURES[0].id, current: true });
  });

  it(`GET ${TWIN}/tags returns the fixture entries`, async () => {
    const data = await getJson<unknown[]>(`${TWIN}/tags`);
    expect(data).toHaveLength(TAG_FIXTURES.length);
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

  it(`POST ${TWIN}/query/stream emits NDJSON token + sources events`, async () => {
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
    expect(tokens.length).toBeGreaterThan(0);
    expect(sources.length).toBe(1);
    expect(Array.isArray(sources[0].value)).toBe(true);
    expect((sources[0].value as unknown[]).length).toBe(2);
  });
});
