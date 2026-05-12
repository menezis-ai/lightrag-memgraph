/**
 * Smoke tests for the MSW handler set.
 *
 * We spin up the Node msw server (so handlers intercept fetch at the global
 * `fetch` level) and verify each route returns the expected fixture shape.
 * This catches handler-shape regressions independently of the App wiring.
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
  WORKSPACE_FIXTURES,
} from '../fixtures';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const BASE = 'http://localhost';

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  expect(r.ok).toBe(true);
  return (await r.json()) as T;
}

describe('MSW handlers — happy paths', () => {
  it('GET /documents returns the full fixture envelope', async () => {
    const data = await getJson<{ items: unknown[]; total: number }>('/documents');
    expect(data.total).toBe(DOCUMENT_FIXTURES.length);
    expect(data.items).toHaveLength(DOCUMENT_FIXTURES.length);
  });

  it('GET /documents?status=failed filters the envelope', async () => {
    const data = await getJson<{ items: { status: string }[]; total: number }>(
      '/documents?status=failed',
    );
    expect(data.total).toBeGreaterThan(0);
    data.items.forEach((d) => expect(d.status).toBe('failed'));
  });

  it('GET /documents?q=oracle filters by source substring', async () => {
    const data = await getJson<{ items: { source: string }[] }>(
      '/documents?q=oracle',
    );
    expect(data.items.length).toBeGreaterThan(0);
    data.items.forEach((d) =>
      expect(d.source.toLowerCase()).toMatch(/oracle/),
    );
  });

  it('GET /workspaces returns the fixture array', async () => {
    const data = await getJson<unknown[]>('/workspaces');
    expect(data).toHaveLength(WORKSPACE_FIXTURES.length);
  });

  it('GET /tags returns 21 entries (Tier 1-3 + requested)', async () => {
    const data = await getJson<unknown[]>('/tags');
    expect(data).toHaveLength(TAG_FIXTURES.length);
  });

  it('GET /tags/categories returns the fixture array', async () => {
    const data = await getJson<unknown[]>('/tags/categories');
    expect(data).toHaveLength(TAG_CATEGORY_FIXTURES.length);
  });

  it('GET /activity returns items + pinned nowMs', async () => {
    const data = await getJson<{ items: unknown[]; nowMs: number }>('/activity');
    expect(data.items.length).toBeGreaterThan(0);
    expect(data.nowMs).toBe(ACTIVITY_NOW_MS);
  });

  it('GET /activity?sev=error narrows the audit feed', async () => {
    const data = await getJson<{ items: { sev: string }[] }>(
      '/activity?sev=error',
    );
    expect(data.items.length).toBeGreaterThan(0);
    data.items.forEach((e) => expect(e.sev).toBe('error'));
  });

  it('GET /openapi returns { groups, version }', async () => {
    const data = await getJson<{ groups: unknown[]; version: string }>(
      '/openapi',
    );
    expect(data.version).toBe(API_VERSION);
    expect(data.groups).toHaveLength(5);
  });

  it('GET /graph/entities returns the fixture array', async () => {
    const data = await getJson<unknown[]>('/graph/entities');
    expect(data).toHaveLength(GRAPH_ENTITY_FIXTURES.length);
  });

  it('GET /graph/relations returns the fixture array', async () => {
    const data = await getJson<unknown[]>('/graph/relations');
    expect(data).toHaveLength(GRAPH_RELATION_FIXTURES.length);
  });

  it('POST /notifications/read-all returns ok envelope', async () => {
    const r = await fetch(BASE + '/notifications/read-all', { method: 'POST' });
    expect(r.ok).toBe(true);
    expect(await r.json()).toEqual({ ok: true });
  });
});
