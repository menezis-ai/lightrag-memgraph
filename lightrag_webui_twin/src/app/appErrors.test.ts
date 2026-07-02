/**
 * Unit tests for the backend-error projection helpers in appErrors.ts.
 *
 * `resourceError` decides whether a tab should surface a banner (only when a
 * query has genuinely errored with no data and is not loading), and
 * `formatBackendError` reduces any thrown value to a human string. Both are
 * pure functions — no React, no fetch.
 */

import { describe, expect, it } from 'vitest';

import { ApiError } from '../api/client';
import {
  formatBackendError,
  resourceError,
  type QueryLike,
} from './appErrors';

function query<T>(overrides: Partial<QueryLike<T>>): QueryLike<T> {
  return {
    data: undefined,
    isError: false,
    isLoading: false,
    error: undefined,
    ...overrides,
  };
}

describe('resourceError', () => {
  it('returns a BackendResourceError when the query errored with no data', () => {
    const out = resourceError('Documents', query({
      isError: true,
      error: new Error('boom'),
    }));
    expect(out).toEqual({ label: 'Documents', message: 'boom' });
  });

  it('returns null when the query has data (even if isError flickers true)', () => {
    const out = resourceError('Tags', query({
      data: ['a'],
      isError: true,
      error: new Error('stale'),
    }));
    expect(out).toBeNull();
  });

  it('returns null while loading', () => {
    const out = resourceError('Activity', query({
      isLoading: true,
      isError: true,
      error: new Error('mid-flight'),
    }));
    expect(out).toBeNull();
  });

  it('returns null when there is no error', () => {
    const out = resourceError('Folders', query({ isError: false }));
    expect(out).toBeNull();
  });

  it('formats an ApiError into the projected message', () => {
    const out = resourceError('Quota', query({
      isError: true,
      error: new ApiError('Service Unavailable', 503, { detail: 'down' }),
    }));
    expect(out).toEqual({
      label: 'Quota',
      message: '503 Service Unavailable',
    });
  });
});

describe('formatBackendError', () => {
  it('prefixes the status for an ApiError', () => {
    expect(
      formatBackendError(new ApiError('Not Found', 404, null)),
    ).toBe('404 Not Found');
  });

  it('returns the message for a plain Error', () => {
    expect(formatBackendError(new Error('network down'))).toBe('network down');
  });

  it('falls back to a generic string for non-Error throws', () => {
    expect(formatBackendError('a bare string')).toBe('Backend request failed');
    expect(formatBackendError(undefined)).toBe('Backend request failed');
    expect(formatBackendError({ weird: true })).toBe('Backend request failed');
  });
});
