/**
 * Unit tests for the backend-error projection helpers in appErrors.ts.
 *
 * `resourceError` decides whether a tab should surface a banner (only when a
 * query has genuinely errored with no data and is not loading), and
 * `formatBackendError` reduces any thrown value to operator-facing copy via
 * the shared error-mapping layer (error-UX pass 2026-07-03) — raw statuses
 * and transport strings never reach the banner. Both are pure functions.
 */

import { describe, expect, it } from 'vitest';

import { ApiError } from '../api/client';
import {
  formatBackendError,
  resourceError,
  type QueryLike,
} from './appErrors';

const GENERIC_LOAD_MESSAGE =
  'Something went wrong while loading data from the backend. Please retry. If the problem continues, contact your platform administrator.';

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
    expect(out).toEqual({ label: 'Documents', message: GENERIC_LOAD_MESSAGE });
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

  it('formats an ApiError into mapped operator copy (no raw status)', () => {
    const out = resourceError('Quota', query({
      isError: true,
      error: new ApiError('Service Unavailable', 503, { detail: 'down' }),
    }));
    expect(out?.label).toBe('Quota');
    expect(out?.message).toBe(
      'The backend is temporarily unavailable. Retry in a moment. If the problem continues, contact your platform administrator.',
    );
    expect(out?.message).not.toContain('503');
  });
});

describe('formatBackendError', () => {
  it('maps an ApiError to operator copy without the raw status', () => {
    const out = formatBackendError(new ApiError('Not Found', 404, null));
    expect(out).toBe(
      'The requested item could not be found. It may have been removed.',
    );
    expect(out).not.toContain('404');
  });

  it('maps a plain technical Error to the generic copy', () => {
    expect(formatBackendError(new Error('network down'))).toBe(
      GENERIC_LOAD_MESSAGE,
    );
  });

  it('maps fetch network failures to the connectivity copy', () => {
    expect(formatBackendError(new TypeError('Failed to fetch'))).toBe(
      'Cannot reach the Twin backend. Check your connection and retry.',
    );
  });

  it('falls back to the generic copy for non-Error throws', () => {
    expect(formatBackendError('a bare string')).toBe(GENERIC_LOAD_MESSAGE);
    expect(formatBackendError(undefined)).toBe(GENERIC_LOAD_MESSAGE);
    expect(formatBackendError({ weird: true })).toBe(GENERIC_LOAD_MESSAGE);
  });
});
