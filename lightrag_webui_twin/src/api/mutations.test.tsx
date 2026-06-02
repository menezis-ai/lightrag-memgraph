/**
 * Unit tests for the tag mutation hooks (S4c slice 2).
 *
 * We mock global `fetch` and verify each `useXxxTag` hook builds the right
 * HTTP request (method, path, body). Cache invalidation is implicit (the
 * underlying useMutation calls invalidateQueries on success) — we test the
 * call surface, not TanStack internals.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import {
  useApproveTag,
  useBulkDeleteDocuments,
  useDeleteTag,
  useDeprecateTag,
  useEditTag,
  useRejectTag,
  useRequestTag,
  useUpdateTagSynonyms,
} from './queries';

type FetchMock = ReturnType<typeof vi.fn>;
let originalFetch: typeof fetch;
let fetchMock: FetchMock;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function wrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return Wrapper;
}

beforeEach(() => {
  originalFetch = globalThis.fetch;
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('useRequestTag', () => {
  it('POSTs /tags with the proposed entry', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ tag: 'newtag', tier: 'requested' }, 201),
    );
    const { result } = renderHook(() => useRequestTag(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({
        tag: 'newtag',
        def: 'A new tag',
        category: 'infra',
        actor: 'claire.benoit',
      });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/tags');
    expect((init as RequestInit).method).toBe('POST');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.tag).toBe('newtag');
    expect(body.def).toBe('A new tag');
  });
});

describe('useApproveTag', () => {
  it('POSTs /tags/{name}/approve', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ tag: 'argocd', tier: 3, status: 'active' }),
    );
    const { result } = renderHook(() => useApproveTag(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({ name: 'argocd', actor: 'claire.benoit' });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/tags/argocd/approve');
    expect((init as RequestInit).method).toBe('POST');
  });
});

describe('useRejectTag', () => {
  it('POSTs /tags/{name}/reject with reason', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ tag: 'argocd', status: 'rejected' }),
    );
    const { result } = renderHook(() => useRejectTag(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({ name: 'argocd', reason: 'duplicate of k8s' });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/tags/argocd/reject');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.reason).toBe('duplicate of k8s');
  });
});

describe('useEditTag', () => {
  it('PATCHes /tags/{name} with diff body', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ tag: 'rman', def: 'new' }));
    const { result } = renderHook(() => useEditTag(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({ name: 'rman', def: 'new' });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/tags/rman');
    expect((init as RequestInit).method).toBe('PATCH');
  });
});

describe('useDeprecateTag', () => {
  it('POSTs /tags/{name}/deprecate', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ tag: 'rman', status: 'deprecated' }),
    );
    const { result } = renderHook(() => useDeprecateTag(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({ name: 'rman', reason: 'old' });
    });
    expect(String(fetchMock.mock.calls[0][0])).toContain('/tags/rman/deprecate');
  });
});

describe('useUpdateTagSynonyms', () => {
  it('POSTs /tags/{name}/synonyms with aliases', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ tag: 'rman', aliases: ['rmgr'] }));
    const { result } = renderHook(() => useUpdateTagSynonyms(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      await result.current.mutateAsync({ name: 'rman', aliases: ['rmgr'] });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/tags/rman/synonyms');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.aliases).toEqual(['rmgr']);
  });
});

describe('useDeleteTag', () => {
  it('DELETEs /tags/{name} with strategy + to', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const { result } = renderHook(() => useDeleteTag(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({
        name: 'rman',
        strategy: 'migrate',
        to: 'oracle',
      });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/tags/rman');
    expect((init as RequestInit).method).toBe('DELETE');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.strategy).toBe('migrate');
    expect(body.to).toBe('oracle');
  });
});

describe('useBulkDeleteDocuments', () => {
  it('POSTs /documents/bulk-delete once with all selected ids', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ deleted: 2 }));
    const { result } = renderHook(() => useBulkDeleteDocuments(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      await result.current.mutateAsync({
        doc_ids: ['doc-a', 'doc-b'],
        actor: 'claire.benoit',
      });
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/documents/bulk-delete');
    expect((init as RequestInit).method).toBe('POST');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.doc_ids).toEqual(['doc-a', 'doc-b']);
    expect(body.actor).toBe('claire.benoit');
  });
});
