/**
 * Unit tests for `apiFetch` — the thin typed fetch wrapper.
 *
 * Covers: GET path with query params, POST JSON body, bearer auth header
 * injection, ApiError on 4xx/5xx including non-JSON 502 (the nginx pattern
 * that crashed the BNP front in v0.5.2).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, apiFetch } from './client';

type FetchMock = ReturnType<typeof vi.fn>;
let originalFetch: typeof fetch;
let fetchMock: FetchMock;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  originalFetch = globalThis.fetch;
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('apiFetch', () => {
  it('GET request returns parsed JSON', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true, items: [1, 2] }));
    const data = await apiFetch<{ ok: boolean; items: number[] }>('/health');
    expect(data).toEqual({ ok: true, items: [1, 2] });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/health');
    expect((init as RequestInit).method).toBe('GET');
  });

  it('serializes query params via URLSearchParams (skipping null/undefined)', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    await apiFetch('/documents', {
      query: { status: 'completed', q: 'oracle', tag: null, cursor: undefined },
    });
    const [url] = fetchMock.mock.calls[0];
    const u = new URL(String(url), 'http://localhost');
    expect(u.searchParams.get('status')).toBe('completed');
    expect(u.searchParams.get('q')).toBe('oracle');
    expect(u.searchParams.has('tag')).toBe(false);
    expect(u.searchParams.has('cursor')).toBe(false);
  });

  it('POST request sends JSON body and Content-Type header', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'doc_1' }));
    await apiFetch('/documents/text', {
      method: 'POST',
      body: { text: 'hello', tags: ['twin'] },
    });
    const [, init] = fetchMock.mock.calls[0];
    expect((init as RequestInit).method).toBe('POST');
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers['Content-Type']).toBe('application/json');
    expect((init as RequestInit).body).toBe(
      JSON.stringify({ text: 'hello', tags: ['twin'] }),
    );
  });

  it('attaches bearer token from per-call override', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    await apiFetch('/secure', { token: 'eyJtest' });
    const [, init] = fetchMock.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer eyJtest');
  });

  it('throws ApiError with parsed JSON body on 4xx', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: 'forbidden' }, 403),
    );
    await expect(apiFetch('/secure')).rejects.toMatchObject({
      name: 'ApiError',
      status: 403,
      body: { detail: 'forbidden' },
    });
  });

  it('throws ApiError with raw text body on non-JSON 502 (nginx pattern)', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response('<html><body>502 Bad Gateway</body></html>', {
        status: 502,
        statusText: 'Bad Gateway',
      }),
    );
    let err: ApiError | null = null;
    try {
      await apiFetch('/documents/paginated');
    } catch (e) {
      err = e as ApiError;
    }
    expect(err).not.toBeNull();
    expect(err!.status).toBe(502);
    expect(typeof err!.body).toBe('string');
    expect(err!.body).toMatch(/502/);
  });
});
