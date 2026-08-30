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
import { ApiError, onUnauthorized, setSessionAuthToken } from './client';
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
  setSessionAuthToken(null);
  globalThis.fetch = originalFetch;
});

describe('queryStream parser — answer_status propagation', () => {
  it('reports a stream 401 and preserves the parsed ApiError', async () => {
    const unauthorized = vi.fn();
    const unsubscribe = onUnauthorized(unauthorized);
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Token expired' }), {
        status: 401,
        statusText: 'Unauthorized',
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    let error: unknown;

    try {
      await api.queryStream({ query: 'expired session' }, () => undefined);
    } catch (caught) {
      error = caught;
    } finally {
      unsubscribe();
    }

    expect(unauthorized).toHaveBeenCalledOnce();
    expect(unauthorized).toHaveBeenCalledWith({
      path: '/twin/api/query/stream',
      status: 401,
    });
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 401,
      body: { detail: 'Token expired' },
    });
  });

  it('forwards retrieval progress stage events without polluting answer text', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'stage', value: 'retrieval' }),
        JSON.stringify({ type: 'stage', value: 'generation' }),
        JSON.stringify({ type: 'token', value: 'Answer.' }),
        JSON.stringify({ type: 'stage', value: 'sources' }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );
    const onStage = vi.fn();

    const result = await api.queryStream(
      { query: 'slow?' },
      () => undefined,
      undefined,
      onStage,
    );

    expect(onStage.mock.calls.flat()).toEqual(['retrieval', 'generation', 'sources']);
    expect(result.response).toBe('Answer.');
  });

  it('preserves deployment model metadata from the stream', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'meta', value: { model: 'deepseek-chat' } }),
        JSON.stringify({ type: 'token', value: 'Answer.' }),
        JSON.stringify({ type: 'status', value: 'grounded' }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );

    const res = await api.queryStream({ query: 'model?' }, () => undefined);
    expect(res.model).toBe('deepseek-chat');
  });

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

  it('preserves null, absent, zero, and numeric source scores', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: 'A real answer.' }),
        JSON.stringify({ type: 'status', value: 'grounded' }),
        JSON.stringify({
          type: 'sources',
          value: [
            { n: 1, type: 'file', name: 'null.pdf', score: null },
            { n: 2, type: 'file', name: 'absent.pdf' },
            { n: 3, type: 'file', name: 'zero.pdf', score: 0 },
            { n: 4, type: 'file', name: 'ranked.pdf', score: 0.82 },
          ],
        }),
      ]),
    );

    const res = await api.queryStream({ query: 'real' }, () => undefined);
    expect(res.sources.map((source) => source.score)).toEqual([
      null,
      undefined,
      0,
      0.82,
    ]);
  });

  it('returns answer_status=source_projection_failed when the stream signals it', async () => {
    // The grounded answer streamed, but its references could not be projected:
    // the parser must propagate the explicit status (NOT default to grounded,
    // which would hide the failure) and surface empty sources.
    const chunks: string[] = [];
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: 'A grounded answer.' }),
        JSON.stringify({ type: 'status', value: 'source_projection_failed' }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );

    const res = await api.queryStream(
      { query: 'real' },
      (c) => chunks.push(c),
    );
    expect(res.answer_status).toBe('source_projection_failed');
    expect(res.sources).toEqual([]);
    // The answer itself still reaches the live UI.
    expect(chunks.join('')).toBe('A grounded answer.');
  });

  it('returns answer_status=no_retrieval when the stream signals a sourceless mode', async () => {
    // only_need_context streams a context body but no sourced final answer:
    // the parser must propagate no_retrieval (NOT default to grounded) and
    // surface empty sources.
    const chunks: string[] = [];
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: 'Direct answer.' }),
        JSON.stringify({ type: 'status', value: 'no_retrieval' }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );

    const res = await api.queryStream(
      { query: 'anything', mode: 'mix', only_need_context: true },
      (c) => chunks.push(c),
    );
    expect(res.answer_status).toBe('no_retrieval');
    expect(res.sources).toEqual([]);
    expect(chunks.join('')).toBe('Direct answer.');
  });

  it('returns answer_status=query_failed when the stream signals a backend error', async () => {
    // A mid-stream backend error (aquery_llm raised, or a status=failure
    // envelope) is reported via a [query failed: …] token + query_failed
    // status. The parser must propagate query_failed (NOT default to grounded)
    // so the UI can suppress the panel and render an error cue.
    const chunks: string[] = [];
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      ndjsonResponse([
        JSON.stringify({ type: 'token', value: '\n[query failed: LLM down]' }),
        JSON.stringify({ type: 'status', value: 'query_failed' }),
        JSON.stringify({ type: 'sources', value: [] }),
      ]),
    );

    const res = await api.queryStream(
      { query: 'anything' },
      (c) => chunks.push(c),
    );
    expect(res.answer_status).toBe('query_failed');
    expect(res.sources).toEqual([]);
    expect(chunks.join('')).toContain('[query failed: LLM down]');
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

describe('portability resources', () => {
  const job = {
    id: 'imp_test',
    kind: 'import',
    workspace: 'base',
    status: 'dry-running',
    created_at: '2026-08-26T12:00:00Z',
    updated_at: '2026-08-26T12:00:00Z',
    actor: 'operator',
    options: {},
    result: null,
    report: null,
    validation: null,
    error: null,
    download_available: false,
  };

  it('sends the import as multipart without overriding its content type', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response(JSON.stringify(job), {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      }),
    );

    await api.createPortabilityImport({
      bundle: new File(['bundle'], 'staging.tar.gz', {
        type: 'application/gzip',
      }),
      folderMap: { staging: 'production' },
      allowUnverified: false,
    });

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(String(url)).toContain('/twin/api/admin/portability/imports');
    expect(init?.method).toBe('POST');
    expect(new Headers(init?.headers).has('Content-Type')).toBe(false);
    expect(init?.body).toBeInstanceOf(FormData);
    const form = init?.body as FormData;
    expect((form.get('bundle') as File).name).toBe('staging.tar.gz');
    expect(form.get('folder_map')).toBe('{"staging":"production"}');
    expect(form.get('allow_unverified')).toBe('false');
  });

  it('downloads the completed archive as a blob', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response('canonical bundle', {
        status: 200,
        headers: { 'Content-Type': 'application/gzip' },
      }),
    );

    const blob = await api.downloadPortabilityExport('exp_test');

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(String(url)).toContain(
      '/twin/api/admin/portability/exports/exp_test?download=true',
    );
    expect(new Headers(init?.headers).get('Accept')).toBe('application/gzip');
    expect(await blob.text()).toBe('canonical bundle');
  });

  it.each([
    {
      label: 'binary export download',
      request: () => api.downloadPortabilityExport('exp_test'),
      path: '/twin/api/admin/portability/exports/exp_test',
    },
    {
      label: 'multipart import upload',
      request: () =>
        api.createPortabilityImport({
          bundle: new File(['bundle'], 'staging.tar.gz', {
            type: 'application/gzip',
          }),
        }),
      path: '/twin/api/admin/portability/imports',
    },
  ])('reports a 401 from the $label path', async ({ request, path }) => {
    const unauthorized = vi.fn();
    const unsubscribe = onUnauthorized(unauthorized);
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Token expired' }), {
        status: 401,
        statusText: 'Unauthorized',
        headers: { 'Content-Type': 'application/json' },
      }),
    );

    try {
      await expect(request()).rejects.toMatchObject({ status: 401 });
    } finally {
      unsubscribe();
    }
    expect(unauthorized).toHaveBeenCalledOnce();
    expect(unauthorized).toHaveBeenCalledWith({ path, status: 401 });
  });
});

describe('uploadDocument', () => {
  /** Capture both the multipart body and the request headers of the upload. */
  function mockUploadOnce(): {
    bodies: FormData[];
    headers: Headers[];
  } {
    const bodies: FormData[] = [];
    const headers: Headers[] = [];
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation(
      async (_url: string | URL | Request, init?: RequestInit) => {
        if (String(_url).includes('/documents/resolve-upload')) {
          return new Response(JSON.stringify({ action: 'upload' }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        if (init?.body instanceof FormData) bodies.push(init.body);
        headers.push(new Headers(init?.headers));
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
    return { bodies, headers };
  }

  it('does not send the X-Twin-Classification header by default', async () => {
    const { bodies, headers } = mockUploadOnce();

    await api.uploadDocument(new File(['payload'], 'plain.md'));

    expect((bodies[0].get('file') as File | null)?.name).toBe('plain.md');
    // Classification is a header, never a multipart field.
    expect(bodies[0].has('classification')).toBe(false);
    expect(bodies[0].has('rag_engine')).toBe(false);
    expect(headers[0].has('X-Twin-Classification')).toBe(false);
  });

  it('preserves long NFD basenames byte-for-byte for ordinary uploads', async () => {
    const { bodies, headers } = mockUploadOnce();
    const originalName = `${'e\u0301'.repeat(75)}-rapport.pdf`;

    await api.uploadDocument(new File(['payload'], originalName));

    const resolveCall = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const resolveBody = JSON.parse(String(resolveCall[1]?.body)) as {
      file_name: string;
      relative_path?: string;
    };
    expect(resolveBody).toEqual({ file_name: originalName });
    expect((bodies[0].get('file') as File | null)?.name).toBe(originalName);
    expect(headers[0].has('X-Twin-Relative-Path')).toBe(false);
  });

  it('sends the operator classification as the X-Twin-Classification header', async () => {
    const { bodies, headers } = mockUploadOnce();

    await api.uploadDocument(new File(['payload'], 'classified.md'), {
      classification: 'C2',
    });

    expect((bodies[0].get('file') as File | null)?.name).toBe('classified.md');
    // Still a header, not a multipart field.
    expect(bodies[0].has('classification')).toBe(false);
    expect(headers[0].get('X-Twin-Classification')).toBe('C2');
  });

  it('does not send X-Twin-Doc-Type by default (auto-detect)', async () => {
    const { headers } = mockUploadOnce();

    await api.uploadDocument(new File(['payload'], 'plain.md'));

    expect(headers[0].has('X-Twin-Doc-Type')).toBe(false);
  });

  it('sends the forced doc type as the X-Twin-Doc-Type header', async () => {
    const { bodies, headers } = mockUploadOnce();

    await api.uploadDocument(new File(['payload'], 'runbook.pdf'), {
      docType: 'procedure',
    });

    expect((bodies[0].get('file') as File | null)?.name).toBe('runbook.pdf');
    // Profile is a header, never a multipart field.
    expect(bodies[0].has('doc_type')).toBe(false);
    expect(headers[0].get('X-Twin-Doc-Type')).toBe('procedure');
  });

  it('sends the active token as X-API-Key only for the native upload route', async () => {
    setSessionAuthToken('native-upload-token');
    const { headers } = mockUploadOnce();

    await api.uploadDocument(new File(['payload'], 'plain.md'));

    expect(headers[0].has('Authorization')).toBe(false);
    expect(headers[0].get('X-API-Key')).toBe('native-upload-token');
  });

  it('keeps JWT bearer auth on the native upload route', async () => {
    setSessionAuthToken('eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvcGVyYXRvciJ9.sig');
    const { headers } = mockUploadOnce();

    await api.uploadDocument(new File(['payload'], 'plain.md'));

    expect(headers[0].get('Authorization')).toBe(
      'Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvcGVyYXRvciJ9.sig',
    );
    expect(headers[0].has('X-API-Key')).toBe(false);
  });

  it('reports a native upload 401 so the shell returns to login', async () => {
    const unauthorized = vi.fn();
    const unsubscribe = onUnauthorized(unauthorized);
    (globalThis.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ action: 'upload' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Token expired' }), {
          status: 401,
          statusText: 'Unauthorized',
          headers: { 'Content-Type': 'application/json' },
        }),
      );

    try {
      await expect(
        api.uploadDocument(new File(['payload'], 'expired-session.pdf')),
      ).rejects.toMatchObject({ status: 401 });
      expect(unauthorized).toHaveBeenCalledOnce();
      expect(unauthorized).toHaveBeenCalledWith({
        path: '/documents/upload',
        status: 401,
      });
    } finally {
      unsubscribe();
    }
  });

  it('shares an existing document without sending multipart content again', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          action: 'shared',
          message: "'known.pdf' was added to folder 'sandbox'.",
          doc_id: 'doc-known',
          track_id: 'track-known',
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );

    await expect(
      api.uploadDocument(new File(['same bytes'], 'known.pdf')),
    ).resolves.toEqual({
      status: 'shared',
      message: "'known.pdf' was added to folder 'sandbox'.",
      doc_id: 'doc-known',
      track_id: 'track-known',
    });
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(
      (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0][1]?.body,
    ).not.toBeInstanceOf(FormData);
  });

  it('re-resolves a native name collision to close the preflight race', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ action: 'upload' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            detail: "Document storage already contains 'known.pdf'",
          }),
          {
            status: 409,
            statusText: 'Conflict',
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            action: 'shared',
            message: 'added',
            doc_id: 'doc-known',
            track_id: 'track-known',
          }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      );

    await expect(
      api.uploadDocument(new File(['payload'], 'known.pdf')),
    ).resolves.toMatchObject({ status: 'shared', doc_id: 'doc-known' });
    expect(globalThis.fetch).toHaveBeenCalledTimes(3);
  });

  it('does not mask a pipeline-busy 409 with an unrelated preflight retry', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ action: 'upload' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Pipeline is busy' }), {
          status: 409,
          statusText: 'Conflict',
          headers: { 'Content-Type': 'application/json' },
        }),
      );

    await expect(
      api.uploadDocument(new File(['payload'], 'new.pdf')),
    ).rejects.toMatchObject({ status: 409, body: { detail: 'Pipeline is busy' } });
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
  });
});

describe('deleteDocument — per-doc failure must not read as success', () => {
  it('resolves { ok: true } when the bulk endpoint deleted exactly the doc', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ deleted: 1, failed: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await expect(api.deleteDocument('doc-x')).resolves.toEqual({ ok: true });
  });

  it('throws when bulk-delete returns HTTP 200 but {deleted:0, failed:[doc]}', async () => {
    // Regression: bulk-delete reports per-doc failures in `failed` with a 200,
    // so a naive .then(() => ok) hid the failure (no rollback, false toast).
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify({ deleted: 0, failed: ['doc-x'] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await expect(api.deleteDocument('doc-x')).rejects.toThrow(/Delete failed for doc-x/);
  });
});
