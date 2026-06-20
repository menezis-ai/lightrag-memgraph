/**
 * Tests for the resources NDJSON stream parser (TR-RET-02).
 *
 * The ``queryStream`` helper in ``resources.ts`` consumes the
 * ``POST /twin/api/query/stream`` body: token / status / sources
 * events. We verify here that the new ``status`` event is parsed and
 * surfaced as ``answer_status`` on the resolved response so the
 * RetrievalTab can suppress the Sources panel honestly.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from './resources';

const originalFetch = globalThis.fetch;

function ndjsonResponse(lines: readonly string[]): Response {
  return new Response(lines.map((l) => `${l}\n`).join(''), {
    status: 200,
    headers: { 'Content-Type': 'application/x-ndjson' },
  });
}

beforeEach(() => {
  // Default stub — each test re-stubs with the body it wants.
  globalThis.fetch = vi.fn();
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('queryStream parser — answer_status propagation', () => {
  it('returns answer_status=insufficient_information when the stream carries the status event', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({
          type: 'token',
          value: "Sorry, I'm not able to provide an answer.",
        }),
        JSON.stringify({
          type: 'status',
          value: 'insufficient_information',
        }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );

    const chunks: string[] = [];
    const res = await api.queryStream(
      { query: 'unanswerable' },
      (c) => chunks.push(c),
    );

    expect(res.answer_status).toBe('insufficient_information');
    expect(res.sources).toEqual([]);
    // Token still streams through onChunk for the live UI.
    expect(chunks.join('')).toBe(
      "Sorry, I'm not able to provide an answer.",
    );
  });

  it('returns answer_status=grounded when the stream signals it explicitly', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: 'A real answer.' }),
        JSON.stringify({ type: 'status', value: 'grounded' }),
        JSON.stringify({
          type: 'sources',
          value: [{ n: 1, type: 'file', name: 'runbook.pdf', score: 0.9 }],
        }),
      ]),
    );

    const res = await api.queryStream({ query: 'real' }, () => undefined);
    expect(res.answer_status).toBe('grounded');
    expect(res.sources).toHaveLength(1);
  });

  it('defaults answer_status to grounded when the stream has no status event (back-compat)', async () => {
    // A legacy backend not yet shipping the status event must still
    // produce a sensible client default — the panel renders as before.
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: 'Legacy.' }),
        JSON.stringify({
          type: 'sources',
          value: [{ n: 1, type: 'file', name: 'legacy.pdf', score: 0.7 }],
        }),
      ]),
    );

    const res = await api.queryStream({ query: 'legacy' }, () => undefined);
    expect(res.answer_status).toBe('grounded');
    expect(res.sources).toHaveLength(1);
  });

  it('ignores an unknown status value defensively', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: 'oddly classified.' }),
        JSON.stringify({ type: 'status', value: 'something_new' }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );

    const res = await api.queryStream({ query: 'x' }, () => undefined);
    // Falls back to the safe default rather than propagating garbage.
    expect(res.answer_status).toBe('grounded');
  });
});

describe('listDocumentChunks', () => {
  it('normalizes native chunk content fields to the UI shape', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify([
          {
            chunk_id: 'chunk-a',
            chunk_order_index: 7,
            content: 'Native content field from chunk route.',
          },
        ]),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );

    const chunks = await api.listDocumentChunks('doc-a');

    expect(chunks).toEqual([
      {
        chunk_id: 'chunk-a',
        order: 7,
        text: 'Native content field from chunk route.',
      },
    ]);
  });
});

describe('uploadDocument', () => {
  it('sends only the file field accepted by the native upload route', async () => {
    const bodies: FormData[] = [];
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementationOnce(
      async (_url: string | URL | Request, init?: RequestInit) => {
        if (init?.body instanceof FormData) bodies.push(init.body);
        return new Response(
          JSON.stringify({
            status: 'success',
            message: 'ok',
            track_id: 'track-upload',
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      },
    );

    await api.uploadDocument(new File(['payload'], 'plain.md'));

    const body = bodies[0];
    expect((body.get('file') as File | null)?.name).toBe('plain.md');
    expect(body.has('classification')).toBe(false);
    expect(body.has('rag_engine')).toBe(false);
  });
});
