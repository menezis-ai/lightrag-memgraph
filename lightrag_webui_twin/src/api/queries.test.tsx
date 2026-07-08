/**
 * Unit tests for the TanStack Query hooks in queries.ts.
 *
 * Strategy mirrors mutations.test.tsx: mock `globalThis.fetch`, drive each
 * hook through `renderHook` with a fresh `QueryClient` (retry off), and assert
 * the call surface + cache side-effects. Plain helpers (`unwrap`,
 * `asDocuments`) are exercised directly.
 *
 * Coverage focus: the query-key / folder-scope branches, the optimistic
 * mutate / rollback paths on the graph + document mutations, the OpenAPI
 * fetch error branch, and the upload-batch concurrency/rejection paths.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

import { setActiveFolder } from './client';
import {
  asDocuments,
  unwrap,
  useActivity,
  useApiKeys,
  useBulkRetagDocuments,
  useClearNotifications,
  useCreateApiKey,
  useCreateFolder,
  useCreateGraphEntity,
  useCreateGraphRelation,
  useDeleteDocument,
  useDeleteFolder,
  useDocumentFolders,
  useDeleteGraphEntity,
  useDeleteGraphRelation,
  useAddDocumentToFolder,
  useDocuments,
  useFolders,
  useGraphEntities,
  useGraphRelations,
  useImportCategories,
  useInstanceQuota,
  useMarkAllNotificationsRead,
  useNotifications,
  useOpenApi,
  usePipelineStatus,
  useRevokeApiKey,
  useRemoveDocumentFromFolder,
  useTagCategories,
  useTags,
  useThesaurus,
  useUpdateFolder,
  useUpdateGraphEntity,
  useUpdateGraphRelation,
  useUploadDocument,
  useUploadDocumentsBatch,
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

function newClient(): QueryClient {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
}

function wrapperForClient(client: QueryClient) {
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return Wrapper;
}

function wrapper() {
  return wrapperForClient(newClient());
}

beforeEach(() => {
  originalFetch = globalThis.fetch;
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  setActiveFolder(null);
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  setActiveFolder(null);
});

// ── Plain helpers ──────────────────────────────────────────────────────────

describe('unwrap / asDocuments', () => {
  it('unwrap returns the items array when present', () => {
    expect(unwrap({ items: [1, 2] })).toEqual([1, 2]);
  });
  it('unwrap returns [] when data is undefined', () => {
    expect(unwrap(undefined)).toEqual([]);
  });
  it('asDocuments returns the items array when present', () => {
    const docs = [{ doc_id: 'a' }] as never;
    expect(asDocuments({ items: docs })).toBe(docs);
  });
  it('asDocuments returns [] when data is undefined', () => {
    expect(asDocuments(undefined)).toEqual([]);
  });
});

// ── Query hooks: happy paths + folder scoping branches ──────────────────────

describe('useDocuments', () => {
  it('fetches documents and scopes the query key to the requested folder', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ items: [], total: 0 }));
    const client = newClient();
    const { result } = renderHook(
      () => useDocuments({ status: 'PROCESSED', folder: 'fin' }),
      { wrapper: wrapperForClient(client) },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(String(fetchMock.mock.calls[0][0])).toContain('/documents');
    // folder=fin must drive the scoped query key
    expect(client.getQueryData(['documents', 'fin', { status: 'PROCESSED', folder: 'fin' }]))
      .toBeDefined();
  });

  it('falls back to getActiveFolder() then default for the scope', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    setActiveFolder('active-fld');
    const client = newClient();
    renderHook(() => useDocuments({}), { wrapper: wrapperForClient(client) });
    await waitFor(() =>
      expect(client.getQueryData(['documents', 'active-fld', {}])).toBeDefined(),
    );
  });

  it('respects the enabled:false gate (no fetch)', async () => {
    const { result } = renderHook(
      () => useDocuments({}, { enabled: false }),
      { wrapper: wrapper() },
    );
    expect(result.current.fetchStatus).toBe('idle');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('honours an explicit folderKey override over the active folder', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    setActiveFolder('active-fld');
    const client = newClient();
    renderHook(() => useDocuments({}, { folderKey: 'override' }), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() =>
      expect(client.getQueryData(['documents', 'override', {}])).toBeDefined(),
    );
  });
});

describe('simple list query hooks', () => {
  const cases: Array<[string, () => unknown, string]> = [
    ['usePipelineStatus', () => usePipelineStatus(), '/pipeline_status'],
    ['useFolders', () => useFolders(), '/folders'],
    ['useInstanceQuota', () => useInstanceQuota(), '/quota'],
    ['useApiKeys', () => useApiKeys(), '/settings/api-keys'],
    ['useNotifications', () => useNotifications(), '/notifications'],
    ['useThesaurus', () => useThesaurus(), '/thesaurus'],
    ['useTags', () => useTags(), '/tags'],
    ['useTagCategories', () => useTagCategories(), '/tags/categories'],
    ['useGraphEntities', () => useGraphEntities(), '/graph/entities'],
    ['useGraphRelations', () => useGraphRelations(), '/graph/relations'],
  ];
  it.each(cases)('%s fetches its endpoint', async (_name, hook, path) => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));
    const { result } = renderHook(() => hook(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current).toBeTruthy());
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(String(fetchMock.mock.calls[0][0])).toContain(path);
  });
});

describe('useActivity', () => {
  it('passes the activity query params and scopes to default folder', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ items: [], total: 0 }));
    const client = newClient();
    const query = {
      range: '30d',
      kind: 'doc-retagged',
      sev: 'warning',
      actor: 'claire.benoit',
      q: 'oracle',
      resourceId: 'doc-123',
      limit: 5,
    };
    renderHook(() => useActivity(query), { wrapper: wrapperForClient(client) });
    await waitFor(() =>
      expect(
        client.getQueryData([
          'activity',
          'default',
          query,
        ]),
      ).toBeDefined(),
    );
    const url = new URL(String(fetchMock.mock.calls[0][0]), 'http://test');
    expect(url.pathname).toContain('/activity');
    expect(url.searchParams.get('range')).toBe('30d');
    expect(url.searchParams.get('kind')).toBe('doc-retagged');
    expect(url.searchParams.get('sev')).toBe('warning');
    expect(url.searchParams.get('actor')).toBe('claire.benoit');
    expect(url.searchParams.get('q')).toBe('oracle');
    expect(url.searchParams.get('resource.id')).toBe('doc-123');
    expect(url.searchParams.get('limit')).toBe('5');
  });
});

describe('useOpenApi', () => {
  it('parses /openapi.json on a 200', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ info: { version: '9.9' }, paths: {}, tags: [] }),
    );
    const { result } = renderHook(() => useOpenApi(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.version).toBe('9.9');
    expect(String(fetchMock.mock.calls[0][0])).toContain('/openapi.json');
  });

  it('throws on a non-ok response (error branch)', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response('nope', { status: 500, statusText: 'Server Error' }),
    );
    const { result } = renderHook(() => useOpenApi(), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect((result.current.error as Error).message).toContain('OpenAPI fetch failed');
  });
});

// ── Folder / API-key / notification mutations: invalidation surface ─────────

describe('folder mutations', () => {
  it('useCreateFolder POSTs and invalidates folders + activity', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'f1', label: 'F1' }));
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useCreateFolder(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ id: 'f1', label: 'F1' });
    });
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe('POST');
    const keys = spy.mock.calls.map(([o]) => (o as { queryKey: unknown[] }).queryKey[0]);
    expect(keys).toEqual(expect.arrayContaining(['folders', 'activity']));
  });

  it('useUpdateFolder PATCHes the folder id', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'f1', label: 'New' }));
    const { result } = renderHook(() => useUpdateFolder(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync({ id: 'f1', patch: { label: 'New' } });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/folders/f1');
    expect((init as RequestInit).method).toBe('PATCH');
  });

  it('useDeleteFolder DELETEs the folder id', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(null));
    const { result } = renderHook(() => useDeleteFolder(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync('f1');
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/folders/f1');
    expect((init as RequestInit).method).toBe('DELETE');
  });
});

describe('document folder membership hooks', () => {
  it('useDocumentFolders fetches memberships for a document', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ doc_id: 'd1', folders: ['default'] }));
    const { result } = renderHook(() => useDocumentFolders('d1'), {
      wrapper: wrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(String(fetchMock.mock.calls[0][0])).toContain('/documents/d1/folders');
    expect(result.current.data?.folders).toEqual(['default']);
  });

  it('useAddDocumentToFolder POSTs the target folder', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ doc_id: 'd1', folders: ['default', 'ops'] }));
    const { result } = renderHook(() => useAddDocumentToFolder(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      await result.current.mutateAsync({ docId: 'd1', folderId: 'ops' });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/documents/d1/folders');
    expect((init as RequestInit).method).toBe('POST');
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      folder_id: 'ops',
    });
  });

  it('useRemoveDocumentFromFolder DELETEs the selected folder', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        doc_id: 'd1',
        removed_folder: 'default',
        physically_deleted: true,
        remaining_folders: [],
      }),
    );
    const { result } = renderHook(() => useRemoveDocumentFromFolder(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      await result.current.mutateAsync({ docId: 'd1', folderId: 'default' });
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/documents/d1/folders/default');
    expect((init as RequestInit).method).toBe('DELETE');
  });

  it('useAddDocumentToFolder invalidates source + destination document queries', async () => {
    setActiveFolder('default');
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        doc_id: 'd1',
        folders: ['default', 'sandbox'],
      }),
    );
    const { result } = renderHook(() => useAddDocumentToFolder(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ docId: 'd1', folderId: 'sandbox' });
    });
    const keys = spy.mock.calls.map(
      ([o]) => (o as { queryKey: unknown[] }).queryKey,
    );
    expect(keys).toEqual(
      expect.arrayContaining([
        ['document-folders', 'd1'],
        ['documents', 'default'],
        ['documents', 'sandbox'],
        ['documents'],
        ['activity'],
      ]),
    );
  });

  it('useRemoveDocumentFromFolder invalidates source + destination document queries', async () => {
    setActiveFolder('default');
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        doc_id: 'd1',
        removed_folder: 'default',
        physically_deleted: false,
        remaining_folders: ['sandbox'],
      }),
    );
    const { result } = renderHook(() => useRemoveDocumentFromFolder(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ docId: 'd1', folderId: 'default' });
    });
    const keys = spy.mock.calls.map(
      ([o]) => (o as { queryKey: unknown[] }).queryKey,
    );
    expect(keys).toEqual(
      expect.arrayContaining([
        ['document-folders', 'd1'],
        ['documents', 'default'],
        ['documents'],
        ['activity'],
        ['graph-entities'],
        ['graph-relations'],
      ]),
    );
  });
});

describe('api-key mutations', () => {
  it('useCreateApiKey POSTs and invalidates api-keys + activity', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'k1', token: 'sk-x' }));
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useCreateApiKey(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ name: 'ci' });
    });
    const keys = spy.mock.calls.map(([o]) => (o as { queryKey: unknown[] }).queryKey[0]);
    expect(keys).toEqual(expect.arrayContaining(['api-keys', 'activity']));
  });

  it('useRevokeApiKey DELETEs the key id', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'k1' }));
    const { result } = renderHook(() => useRevokeApiKey(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync('k1');
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/settings/api-keys/k1');
    expect((init as RequestInit).method).toBe('DELETE');
  });
});

describe('notification mutations', () => {
  it('useMarkAllNotificationsRead POSTs and invalidates notifications', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useMarkAllNotificationsRead(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync();
    });
    expect(String(fetchMock.mock.calls[0][0])).toContain('/notifications/read-all');
    expect(spy).toHaveBeenCalledWith({ queryKey: ['notifications'] });
  });

  it('useClearNotifications DELETEs notifications', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const { result } = renderHook(() => useClearNotifications(), { wrapper: wrapper() });
    await act(async () => {
      await result.current.mutateAsync();
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/notifications');
    expect((init as RequestInit).method).toBe('DELETE');
  });
});

describe('useImportCategories', () => {
  it('POSTs the taxonomy and invalidates tag-categories + tags + activity', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useImportCategories(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync([{ id: 'c1', label: 'C1', color: '#fff' }]);
    });
    expect(String(fetchMock.mock.calls[0][0])).toContain('/tags/categories/_import');
    const keys = spy.mock.calls.map(([o]) => (o as { queryKey: unknown[] }).queryKey[0]);
    expect(keys).toEqual(
      expect.arrayContaining(['tag-categories', 'tags', 'activity']),
    );
  });
});

// ── Graph entity mutations: optimistic patch + rollback ─────────────────────

const ENTITIES_KEY = ['graph-entities', 'default'] as const;
const RELATIONS_KEY = ['graph-relations', 'default'] as const;

function seedGraph(client: QueryClient) {
  client.setQueryData(ENTITIES_KEY, [
    { id: 'e1', name: 'Alpha', type: 'system' },
    { id: 'e2', name: 'Beta', type: 'system' },
  ]);
  client.setQueryData(RELATIONS_KEY, [
    { id: 'r1', source: 'e1', target: 'e2', label: 'links' },
    { id: 'r2', source: 'e2', target: 'e1', label: 'back' },
  ]);
}

describe('useUpdateGraphEntity', () => {
  it('optimistically patches the entity in the cache, then settles', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'e1', name: 'Renamed' }));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useUpdateGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ id: 'e1', patch: { name: 'Renamed' } });
    });
    const entities = client.getQueryData<Array<{ id: string; name: string }>>(ENTITIES_KEY);
    expect(entities?.find((e) => e.id === 'e1')?.name).toBe('Renamed');
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe('PATCH');
  });

  it('rolls back the optimistic patch on error', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useUpdateGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current
        .mutateAsync({ id: 'e1', patch: { name: 'Renamed' } })
        .catch(() => undefined);
    });
    const entities = client.getQueryData<Array<{ id: string; name: string }>>(ENTITIES_KEY);
    expect(entities?.find((e) => e.id === 'e1')?.name).toBe('Alpha');
  });

  it('handles an empty cache (no prev) without throwing', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'e1', name: 'X' }));
    const client = newClient();
    const { result } = renderHook(() => useUpdateGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ id: 'e1', patch: { name: 'X' } });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

describe('useUpdateGraphRelation', () => {
  it('optimistically patches the relation then settles', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'r1', label: 'depends' }));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useUpdateGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ id: 'r1', patch: { label: 'depends' } });
    });
    const rels = client.getQueryData<Array<{ id: string; label: string }>>(RELATIONS_KEY);
    expect(rels?.find((r) => r.id === 'r1')?.label).toBe('depends');
  });

  it('rolls back on error', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useUpdateGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current
        .mutateAsync({ id: 'r1', patch: { label: 'depends' } })
        .catch(() => undefined);
    });
    const rels = client.getQueryData<Array<{ id: string; label: string }>>(RELATIONS_KEY);
    expect(rels?.find((r) => r.id === 'r1')?.label).toBe('links');
  });

  it('handles an empty cache (no prev) without throwing', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'r1', label: 'X' }));
    const client = newClient();
    const { result } = renderHook(() => useUpdateGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ id: 'r1', patch: { label: 'X' } });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

describe('useCreateGraphEntity / useCreateGraphRelation', () => {
  it('useCreateGraphEntity POSTs and invalidates on settle', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'e9', name: 'New' }));
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useCreateGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ name: 'New', type: 'TECHNOLOGY' });
    });
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe('POST');
    const keys = spy.mock.calls.map(([o]) => (o as { queryKey: unknown[] }).queryKey[0]);
    expect(keys).toEqual(expect.arrayContaining(['graph-entities', 'activity']));
  });

  it('useCreateGraphRelation POSTs and invalidates on settle', async () => {
    const relationPayload = {
      source: 'e1',
      target: 'e2',
      label: 'rel',
      strength: 0.72,
    };
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'r9', label: 'rel' }));
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useCreateGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync(relationPayload);
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/graph/relations');
    expect((init as RequestInit).method).toBe('POST');
    expect(JSON.parse((init as RequestInit).body as string)).toEqual(
      relationPayload,
    );
    const keys = spy.mock.calls.map(([o]) => (o as { queryKey: unknown[] }).queryKey[0]);
    expect(keys).toEqual(expect.arrayContaining(['graph-relations', 'activity']));
  });
});

describe('useDeleteGraphEntity', () => {
  it('optimistically prunes the entity + incident relations, settles', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(null));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useDeleteGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('e1');
    });
    const entities = client.getQueryData<Array<{ id: string }>>(ENTITIES_KEY);
    const rels = client.getQueryData<Array<{ id: string }>>(RELATIONS_KEY);
    expect(entities?.some((e) => e.id === 'e1')).toBe(false);
    // both relations were incident on e1
    expect(rels).toEqual([]);
  });

  it('rolls back both entity and relation caches on error', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useDeleteGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('e1').catch(() => undefined);
    });
    expect(client.getQueryData<Array<{ id: string }>>(ENTITIES_KEY)).toHaveLength(2);
    expect(client.getQueryData<Array<{ id: string }>>(RELATIONS_KEY)).toHaveLength(2);
  });

  it('handles an empty cache (no prev entities/relations)', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(null));
    const client = newClient();
    const { result } = renderHook(() => useDeleteGraphEntity(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('e1');
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

describe('useDeleteGraphRelation', () => {
  it('optimistically prunes the relation then settles', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(null));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useDeleteGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('r1');
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/graph/relations/r1');
    expect((init as RequestInit).method).toBe('DELETE');
    const rels = client.getQueryData<Array<{ id: string }>>(RELATIONS_KEY);
    expect(rels?.some((r) => r.id === 'r1')).toBe(false);
    expect(rels).toHaveLength(1);
  });

  it('rolls back on error', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500));
    const client = newClient();
    seedGraph(client);
    const { result } = renderHook(() => useDeleteGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('r1').catch(() => undefined);
    });
    expect(client.getQueryData<Array<{ id: string }>>(RELATIONS_KEY)).toHaveLength(2);
  });

  it('handles an empty cache (no prev)', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(null));
    const client = newClient();
    const { result } = renderHook(() => useDeleteGraphRelation(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('r1');
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

// ── Document delete: single + rollback (mapDocumentsEnvelope branches) ───────

describe('useDeleteDocument', () => {
  it('flags the single doc as _deleting then settles', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ deleted: 1 }));
    const client = newClient();
    client.setQueryData(['documents', 'default', {}], {
      items: [
        { doc_id: 'doc-a', status: 'PROCESSED', file_path: 'a.pdf', tags: [] },
        { doc_id: 'doc-b', status: 'PROCESSED', file_path: 'b.pdf', tags: [] },
      ],
      total: 2,
    });
    const { result } = renderHook(() => useDeleteDocument(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('doc-a');
    });
    // bulk-delete cascade is the underlying call
    expect(String(fetchMock.mock.calls[0][0])).toContain('/documents/bulk-delete');
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });

  it('rolls back the _deleting flag on error', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500));
    const client = newClient();
    client.setQueryData(['documents', 'default', {}], {
      items: [{ doc_id: 'doc-a', status: 'PROCESSED', file_path: 'a.pdf', tags: [] }],
      total: 1,
    });
    const { result } = renderHook(() => useDeleteDocument(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('doc-a').catch(() => undefined);
    });
    const data = client.getQueryData<{ items: Array<{ _deleting?: boolean }> }>([
      'documents',
      'default',
      {},
    ]);
    expect(data?.items[0]._deleting).toBeUndefined();
  });

  it('handles an empty documents cache (mapDocumentsEnvelope undefined branch)', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ deleted: 1 }));
    const client = newClient();
    // Seed a documents entry with no items so mapDocumentsEnvelope hits !old?.items
    client.setQueryData(['documents', 'default', {}], { total: 0 });
    const { result } = renderHook(() => useDeleteDocument(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync('doc-a');
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

// ── Bulk retag: applyDocumentTags target hit/miss + rollback ────────────────

describe('useBulkRetagDocuments', () => {
  it('optimistically applies adds/removes to targeted docs only', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ updated: 1, failed: [] }));
    const client = newClient();
    client.setQueryData(['documents', 'default', {}], {
      items: [
        { doc_id: 'doc-a', status: 'PROCESSED', file_path: 'a.pdf', tags: ['old', 'keep'] },
        { doc_id: 'doc-b', status: 'PROCESSED', file_path: 'b.pdf', tags: ['untouched'] },
      ],
      total: 2,
    });
    const { result } = renderHook(() => useBulkRetagDocuments(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({
        targets: ['doc-a'],
        adds: ['new'],
        removes: ['old'],
      });
    });
    const data = client.getQueryData<{
      items: Array<{ doc_id: string; tags: string[] }>;
    }>(['documents', 'default', {}]);
    const a = data?.items.find((d) => d.doc_id === 'doc-a');
    const b = data?.items.find((d) => d.doc_id === 'doc-b');
    expect(a?.tags).toEqual(expect.arrayContaining(['keep', 'new']));
    expect(a?.tags).not.toContain('old');
    // non-target doc is unchanged
    expect(b?.tags).toEqual(['untouched']);
    expect(String(fetchMock.mock.calls[0][0])).toContain('/documents/_bulk-retag');
  });

  it('does not optimistically retag the same doc id in another folder cache', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ updated: 1, failed: [] }));
    setActiveFolder('default');
    const client = newClient();
    client.setQueryData(['documents', 'default', {}], {
      items: [
        { doc_id: 'shared-doc', status: 'PROCESSED', file_path: 'a.pdf', tags: ['a'] },
      ],
      total: 1,
    });
    client.setQueryData(['documents', 'sandbox', {}], {
      items: [
        { doc_id: 'shared-doc', status: 'PROCESSED', file_path: 'a.pdf', tags: ['b'] },
      ],
      total: 1,
    });
    const { result } = renderHook(() => useBulkRetagDocuments(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({
        targets: ['shared-doc'],
        adds: ['folder-a-tag'],
        removes: ['a'],
      });
    });
    const defaultData = client.getQueryData<{
      items: Array<{ tags: string[] }>;
    }>(['documents', 'default', {}]);
    const sandboxData = client.getQueryData<{
      items: Array<{ tags: string[] }>;
    }>(['documents', 'sandbox', {}]);
    expect(defaultData?.items[0].tags).toEqual(['folder-a-tag']);
    expect(sandboxData?.items[0].tags).toEqual(['b']);
  });

  it('rolls back the optimistic tags on error', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500));
    const client = newClient();
    client.setQueryData(['documents', 'default', {}], {
      items: [{ doc_id: 'doc-a', status: 'PROCESSED', file_path: 'a.pdf', tags: ['old'] }],
      total: 1,
    });
    const { result } = renderHook(() => useBulkRetagDocuments(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current
        .mutateAsync({ targets: ['doc-a'], adds: ['new'], removes: ['old'] })
        .catch(() => undefined);
    });
    const data = client.getQueryData<{ items: Array<{ tags: string[] }> }>([
      'documents',
      'default',
      {},
    ]);
    expect(data?.items[0].tags).toEqual(['old']);
  });

  it('handles a documents cache without items (envelope undefined branch)', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ updated: 0, failed: [] }));
    const client = newClient();
    client.setQueryData(['documents', 'default', {}], { total: 0 });
    const { result } = renderHook(() => useBulkRetagDocuments(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync({ targets: ['doc-a'], adds: [], removes: [] });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

// ── Upload hooks: single + batch concurrency / rejection ────────────────────

describe('useUploadDocument', () => {
  it('uploads a single file and invalidates documents + pipeline', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: 'success', message: 'ok', track_id: 't1' }),
    );
    const client = newClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useUploadDocument(), {
      wrapper: wrapperForClient(client),
    });
    await act(async () => {
      await result.current.mutateAsync(new File(['x'], 'a.txt'));
    });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['documents'] });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['pipeline_status'] });
  });
});

describe('useUploadDocumentsBatch', () => {
  it('normalizes both File and UploadDocumentInput entries', async () => {
    // Fresh Response per call — a Response body can only be read once, so a
    // shared mockResolvedValue would reject the second upload's .text().
    fetchMock.mockImplementation(async () =>
      jsonResponse({ status: 'success', message: 'ok', track_id: 't' }),
    );
    const { result } = renderHook(() => useUploadDocumentsBatch(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      const res = await result.current.mutateAsync([
        new File(['a'], 'a.txt'),
        { file: new File(['b'], 'b.txt') },
      ]);
      expect(res.every((r) => r.status === 'fulfilled')).toBe(true);
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('uploads every item in a batch larger than ten', async () => {
    fetchMock.mockImplementation(async () =>
      jsonResponse({ status: 'success', message: 'ok', track_id: 't' }),
    );
    const { result } = renderHook(() => useUploadDocumentsBatch(), {
      wrapper: wrapper(),
    });
    const files = Array.from(
      { length: 12 },
      (_, i) => new File([`payload-${i}`], `batch-${i + 1}.txt`),
    );

    await act(async () => {
      const res = await result.current.mutateAsync(files);
      expect(res.every((r) => r.status === 'fulfilled')).toBe(true);
    });

    expect(fetchMock).toHaveBeenCalledTimes(12);
  });

  it('captures per-file failures as rejected settled results', async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({ status: 'success', message: 'ok', track_id: 't' }),
      )
      .mockResolvedValueOnce(jsonResponse({ detail: 'bad file' }, 400));
    const { result } = renderHook(() => useUploadDocumentsBatch(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      const res = await result.current.mutateAsync([
        new File(['a'], 'a.txt'),
        new File(['b'], 'b.txt'),
      ]);
      const statuses = res.map((r) => r.status).sort();
      expect(statuses).toEqual(['fulfilled', 'rejected']);
    });
  });

  it('handles an empty upload list (workerCount floor branch)', async () => {
    const { result } = renderHook(() => useUploadDocumentsBatch(), {
      wrapper: wrapper(),
    });
    await act(async () => {
      const res = await result.current.mutateAsync([]);
      expect(res).toEqual([]);
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
