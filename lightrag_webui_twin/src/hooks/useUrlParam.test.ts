/**
 * Unit tests for useUrlParam / useUrlArrayParam / useUrlNumberParam.
 *
 * Behaviors under test:
 *   - reads initial value from `?key=...`
 *   - uses default when key absent or invalid
 *   - rewrites the URL when value changes
 *   - removes the key when value matches the default
 *   - array variant joins/splits comma-separated values
 *   - number variant parses finite floats; falls back on NaN
 */

import { describe, expect, it, beforeEach, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import {
  useUrlParam,
  useUrlArrayParam,
  useUrlNumberParam,
} from './useUrlParam';

function resetURL() {
  window.history.replaceState(null, '', '/');
}

function getQuery(): string {
  return window.location.search.replace(/^\?/, '');
}

describe('useUrlParam', () => {
  beforeEach(resetURL);
  afterEach(resetURL);

  it('returns the default when key absent', () => {
    const { result } = renderHook(() => useUrlParam('q', 'hello'));
    expect(result.current[0]).toBe('hello');
  });

  it('reads from existing URL', () => {
    window.history.replaceState(null, '', '/?status=processing');
    const { result } = renderHook(() => useUrlParam('status', 'all'));
    expect(result.current[0]).toBe('processing');
  });

  it('writes the URL when the value changes', () => {
    const { result } = renderHook(() => useUrlParam('q', ''));
    act(() => result.current[1]('oracle'));
    expect(getQuery()).toBe('q=oracle');
  });

  it('removes the key when value matches default', () => {
    window.history.replaceState(null, '', '/?status=processing');
    const { result } = renderHook(() => useUrlParam('status', 'all'));
    act(() => result.current[1]('all'));
    expect(getQuery()).toBe('');
  });

  it('falls back to default when validate returns false', () => {
    window.history.replaceState(null, '', '/?status=garbage');
    const { result } = renderHook(() =>
      useUrlParam('status', 'all', {
        validate: (v) => ['all', 'completed', 'failed'].includes(v as string),
      }),
    );
    expect(result.current[0]).toBe('all');
  });

  it('syncs two hook instances bound to the same URL key', () => {
    const first = renderHook(() => useUrlParam('q', ''));
    const second = renderHook(() => useUrlParam('q', ''));

    act(() => first.result.current[1]('oracle'));

    expect(second.result.current[0]).toBe('oracle');
    expect(getQuery()).toBe('q=oracle');
  });
});

describe('useUrlArrayParam', () => {
  beforeEach(resetURL);
  afterEach(resetURL);

  it('parses a comma-separated value', () => {
    window.history.replaceState(null, '', '/?tag=rman,oracle');
    const { result } = renderHook(() => useUrlArrayParam('tag', []));
    expect(result.current[0]).toEqual(['rman', 'oracle']);
  });

  it('writes comma-separated value', () => {
    const { result } = renderHook(() => useUrlArrayParam('tag', []));
    act(() => result.current[1](['rman', 'memgraph']));
    expect(getQuery()).toBe('tag=rman%2Cmemgraph');
  });

  it('trims whitespace and drops empty parts', () => {
    window.history.replaceState(null, '', '/?tag=rman,%20,oracle');
    const { result } = renderHook(() => useUrlArrayParam('tag', []));
    expect(result.current[0]).toEqual(['rman', 'oracle']);
  });
});

describe('useUrlNumberParam', () => {
  beforeEach(resetURL);
  afterEach(resetURL);

  it('parses a finite integer', () => {
    window.history.replaceState(null, '', '/?top_k=42');
    const { result } = renderHook(() => useUrlNumberParam('top_k', 10));
    expect(result.current[0]).toBe(42);
  });

  it('parses a finite float', () => {
    window.history.replaceState(null, '', '/?threshold=0.42');
    const { result } = renderHook(() =>
      useUrlNumberParam('threshold', 0.2),
    );
    expect(result.current[0]).toBe(0.42);
  });

  it('falls back to default when value is not a finite number', () => {
    window.history.replaceState(null, '', '/?top_k=garbage');
    const { result } = renderHook(() => useUrlNumberParam('top_k', 10));
    expect(result.current[0]).toBe(10);
  });
});
