/**
 * Graph cache is folder-scoped — `docs/test-doctrine-graph.md` axis 1.
 *
 * The doctrine's contract chain is Cypher -> API response shape -> cache state.
 * The backend half is pinned against a live Memgraph in
 * `tests/test_server/test_graph_native_contract_e2e.py`; this file pins the
 * cache half, which no Python test can reach: switching the active folder must
 * land on a DIFFERENT TanStack cache entry and refetch, never re-serve the
 * previous folder's graph.
 *
 * That matters here specifically because the backend read is folder-scoped by
 * document membership: an entity is visible in folder A only when one of its
 * source chunks belongs to a doc MEMBER_OF A. If the cache key were not
 * folder-scoped, a correct, cloisonné backend would still show folder A's
 * entities to folder B for as long as the entry stayed fresh.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

import { setActiveFolder } from './client';
import { useGraphEntities, useGraphRelations } from './queries';

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

const ENTITY_A = [{ id: 'ent-a', name: 'Entity A', type: 'CONCEPT' }];
const ENTITY_B = [{ id: 'ent-b', name: 'Entity B', type: 'CONCEPT' }];

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

describe('graph cache is scoped per folder', () => {
  it('stores entities under a folder-scoped key', async () => {
    const client = newClient();
    setActiveFolder('A');
    fetchMock.mockResolvedValueOnce(jsonResponse(ENTITY_A));

    const { result } = renderHook(() => useGraphEntities(), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(client.getQueryData(['graph-entities', 'A'])).toEqual(ENTITY_A);
    // The unscoped key must stay empty — that is the entry a folder-blind
    // cache would have written, and the one another folder would have read.
    expect(client.getQueryData(['graph-entities'])).toBeUndefined();
  });

  it('does not serve folder A entities to folder B', async () => {
    const client = newClient();

    setActiveFolder('A');
    fetchMock.mockResolvedValueOnce(jsonResponse(ENTITY_A));
    const first = renderHook(() => useGraphEntities(), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true));
    expect(first.result.current.data).toEqual(ENTITY_A);

    // Switch folder: a second request must go out, and its result must land
    // on its own cache entry.
    setActiveFolder('B');
    fetchMock.mockResolvedValueOnce(jsonResponse(ENTITY_B));
    const second = renderHook(() => useGraphEntities(), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() => expect(second.result.current.isSuccess).toBe(true));

    expect(second.result.current.data).toEqual(ENTITY_B);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(client.getQueryData(['graph-entities', 'A'])).toEqual(ENTITY_A);
    expect(client.getQueryData(['graph-entities', 'B'])).toEqual(ENTITY_B);
  });

  it('scopes relations per folder too', async () => {
    const client = newClient();
    const relA = [{ id: 'r1', source: 'ent-a', target: 'ent-a2', label: 'x' }];
    const relB = [{ id: 'r2', source: 'ent-b', target: 'ent-b2', label: 'y' }];

    setActiveFolder('A');
    fetchMock.mockResolvedValueOnce(jsonResponse(relA));
    const first = renderHook(() => useGraphRelations(), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() => expect(first.result.current.isSuccess).toBe(true));

    setActiveFolder('B');
    fetchMock.mockResolvedValueOnce(jsonResponse(relB));
    const second = renderHook(() => useGraphRelations(), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() => expect(second.result.current.isSuccess).toBe(true));

    expect(client.getQueryData(['graph-relations', 'A'])).toEqual(relA);
    expect(client.getQueryData(['graph-relations', 'B'])).toEqual(relB);
    expect(second.result.current.data).toEqual(relB);
  });

  it('honours an explicit folderKey over the active folder', async () => {
    const client = newClient();
    setActiveFolder('A');
    fetchMock.mockResolvedValueOnce(jsonResponse(ENTITY_B));

    const { result } = renderHook(() => useGraphEntities({ folderKey: 'B' }), {
      wrapper: wrapperForClient(client),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(client.getQueryData(['graph-entities', 'B'])).toEqual(ENTITY_B);
    expect(client.getQueryData(['graph-entities', 'A'])).toBeUndefined();
  });
});
