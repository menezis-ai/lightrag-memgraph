/**
 * Unit tests for useAppNavigation.
 *
 * The hook is normally mocked in App-level tests, so it never executes there.
 * These tests run it for real: both branches of onNavigate (the
 * documents+doc/source detail-request branch and the clearing else branch),
 * the URLSearchParams reset+set logic, and all of onSwitchFolder (every setter,
 * setActiveFolder + writeUiPreference, and the 8 invalidateQueries keys).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useAppNavigation } from './useAppNavigation';
import { setActiveFolder } from '../api/client';
import { queryClient } from './queryClient';
import { writeUiPreference, FOLDER_STORAGE_KEY } from './uiPreferences';

vi.mock('../api/client', () => ({
  setActiveFolder: vi.fn(),
}));

vi.mock('./queryClient', () => ({
  queryClient: {
    invalidateQueries: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock('./uiPreferences', () => ({
  writeUiPreference: vi.fn(),
  FOLDER_STORAGE_KEY: 'twin.ui.folder.v1',
}));

function makeOptions() {
  return {
    setClearedNotificationIds: vi.fn(),
    setDetailChunkId: vi.fn(),
    setDetailDoc: vi.fn(),
    setDetailRequest: vi.fn(),
    setDocumentsSearch: vi.fn(),
    setDocumentsSourceFilters: vi.fn(),
    setDocumentsStatusFilter: vi.fn(),
    setDocumentsTagFilters: vi.fn(),
    setFolderState: vi.fn(),
    setReadNotificationIds: vi.fn(),
    setReadSourceDoc: vi.fn(),
    setRetagBulk: vi.fn(),
    setRetagDoc: vi.fn(),
    setTab: vi.fn(),
  };
}

const originalLocation = globalThis.location;
const originalHistory = globalThis.history;
let replaceStateSpy: ReturnType<typeof vi.fn>;

function stubLocation(search: string, pathname: string) {
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { ...originalLocation, search, pathname },
    writable: true,
  });
}

beforeEach(() => {
  replaceStateSpy = vi.fn();
  Object.defineProperty(globalThis, 'history', {
    configurable: true,
    value: { ...originalHistory, replaceState: replaceStateSpy },
    writable: true,
  });
  stubLocation('', '/');
});

afterEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: originalLocation,
    writable: true,
  });
  Object.defineProperty(globalThis, 'history', {
    configurable: true,
    value: originalHistory,
    writable: true,
  });
});

describe('useAppNavigation — onNavigate', () => {
  it('sets a detail request on the documents tab with a doc param and clears existing query string', () => {
    stubLocation('?old=1&stale=2', '/app');
    const opts = makeOptions();
    const { result } = renderHook(() => useAppNavigation(opts));

    result.current.onNavigate('documents', { doc: 'doc-9', chunk: 'c-3' });

    expect(opts.setDetailDoc).toHaveBeenCalledWith(null);
    expect(opts.setDetailChunkId).toHaveBeenCalledWith(null);
    expect(opts.setDetailRequest).toHaveBeenCalledWith({
      doc: 'doc-9',
      source: undefined,
      chunk: 'c-3',
    });
    expect(opts.setTab).toHaveBeenCalledWith('documents');

    // Old keys cleared, new params written into the URL.
    const url = replaceStateSpy.mock.calls[0][2] as string;
    expect(url).not.toContain('old=');
    expect(url).not.toContain('stale=');
    expect(url).toContain('/app?');
    expect(url).toContain('doc=doc-9');
    expect(url).toContain('chunk=c-3');
  });

  it('sets a detail request when only a source param is present', () => {
    const opts = makeOptions();
    const { result } = renderHook(() => useAppNavigation(opts));

    result.current.onNavigate('documents', { source: 'src-7' });

    expect(opts.setDetailRequest).toHaveBeenCalledWith({
      doc: undefined,
      source: 'src-7',
      chunk: undefined,
    });
  });

  it('clears the detail request on a non-documents tab', () => {
    stubLocation('?keep=me', '/');
    const opts = makeOptions();
    const { result } = renderHook(() => useAppNavigation(opts));

    result.current.onNavigate('graph');

    expect(opts.setDetailRequest).toHaveBeenCalledWith(null);
    expect(opts.setDetailDoc).not.toHaveBeenCalled();
    expect(opts.setDetailChunkId).not.toHaveBeenCalled();
    expect(opts.setTab).toHaveBeenCalledWith('graph');

    // No params provided → query string fully cleared → bare pathname.
    expect(replaceStateSpy).toHaveBeenCalledWith(null, '', '/');
  });

  it('clears the detail request on the documents tab when no doc/source param is given', () => {
    const opts = makeOptions();
    const { result } = renderHook(() => useAppNavigation(opts));

    result.current.onNavigate('documents', { foo: 'bar' });

    expect(opts.setDetailRequest).toHaveBeenCalledWith(null);
    const url = replaceStateSpy.mock.calls[0][2] as string;
    expect(url).toContain('foo=bar');
  });

  it('keeps a bare pathname (no ?) when params is undefined and search is empty', () => {
    stubLocation('', '/dash');
    const opts = makeOptions();
    const { result } = renderHook(() => useAppNavigation(opts));

    result.current.onNavigate('tags');

    expect(replaceStateSpy).toHaveBeenCalledWith(null, '', '/dash');
  });
});

describe('useAppNavigation — onSwitchFolder', () => {
  it('resets all navigation state, persists the folder and invalidates every cache key', async () => {
    const opts = makeOptions();
    const { result } = renderHook(() => useAppNavigation(opts));

    result.current.onSwitchFolder('cib');

    expect(replaceStateSpy).toHaveBeenCalledWith(null, '', '/');
    expect(setActiveFolder).toHaveBeenCalledWith('cib');
    expect(writeUiPreference).toHaveBeenCalledWith(FOLDER_STORAGE_KEY, 'cib');
    expect(opts.setFolderState).toHaveBeenCalledWith('cib');

    // Notification + detail state cleared.
    expect(opts.setReadNotificationIds).toHaveBeenCalledTimes(1);
    expect(opts.setClearedNotificationIds).toHaveBeenCalledTimes(1);
    expect(opts.setDetailDoc).toHaveBeenCalledWith(null);
    expect(opts.setDetailChunkId).toHaveBeenCalledWith(null);
    expect(opts.setDetailRequest).toHaveBeenCalledWith(null);
    expect(opts.setReadSourceDoc).toHaveBeenCalledWith(null);
    expect(opts.setRetagDoc).toHaveBeenCalledWith(null);
    expect(opts.setRetagBulk).toHaveBeenCalledWith(null);
    expect(opts.setDocumentsStatusFilter).toHaveBeenCalledWith('all');
    expect(opts.setDocumentsSearch).toHaveBeenCalledWith('');
    expect(opts.setDocumentsTagFilters).toHaveBeenCalledWith([]);
    expect(opts.setDocumentsSourceFilters).toHaveBeenCalledWith([]);

    // The two notification setters receive fresh empty sets.
    const readArg = (opts.setReadNotificationIds.mock.calls[0][0]) as Set<string>;
    const clearedArg = (opts.setClearedNotificationIds.mock.calls[0][0]) as Set<string>;
    expect(readArg).toBeInstanceOf(Set);
    expect(readArg.size).toBe(0);
    expect(clearedArg).toBeInstanceOf(Set);
    expect(clearedArg.size).toBe(0);

    // All 8 cache keys invalidated.
    const invalidate = queryClient.invalidateQueries as ReturnType<typeof vi.fn>;
    expect(invalidate).toHaveBeenCalledTimes(8);
    const keys = invalidate.mock.calls.map((c) => (c[0] as { queryKey: string[] }).queryKey[0]);
    expect(keys).toEqual([
      'documents',
      'pipeline_status',
      'tags',
      'tag-categories',
      'activity',
      'notifications',
      'graph-entities',
      'graph-relations',
    ]);

    // Let the void Promise.all settle so no unhandled rejection leaks.
    await Promise.resolve();
  });
});
