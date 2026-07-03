/**
 * Tests for ``mapCreateEntityError`` — pins the contract mirror between
 * the server (TR-KG-01) and the front. If the backend introduces a new
 * status, the unknown fallback fires and these tests stay green; if the
 * meaning of an existing status changes, the matching case test fails
 * and forces the host UX to be revisited.
 */

import { describe, expect, it } from 'vitest';

import { ApiError } from './client';
import { mapCreateEntityError } from './errors';

describe('mapCreateEntityError', () => {
  it('maps a 409 to the duplicate kind with name in the copy', () => {
    const err = new ApiError('Conflict', 409, {
      detail: "Graph entity 'Memgraph' already exists",
    });
    const out = mapCreateEntityError(err, 'Memgraph');
    expect(out.kind).toBe('duplicate');
    expect(out.message).toContain('Memgraph');
    expect(out.message.toLowerCase()).toContain('already exists');
  });

  it('maps a 422 to the validation kind', () => {
    const err = new ApiError('Unprocessable', 422, {
      detail: [{ loc: ['body', 'name'], msg: 'empty' }],
    });
    const out = mapCreateEntityError(err, '');
    expect(out.kind).toBe('validation');
    expect(out.message.toLowerCase()).toContain('invalid');
  });

  it('maps a 503 to the backend kind without leaking driver detail', () => {
    const err = new ApiError(
      'Service Unavailable',
      503,
      { detail: 'Bolt driver: session closed' },
    );
    const out = mapCreateEntityError(err, 'FreshOne');
    expect(out.kind).toBe('backend');
    // The driver-level detail belongs in logs, not in the user copy.
    expect(out.message.toLowerCase()).toContain('memgraph backend');
    expect(out.message).not.toContain('Bolt driver');
  });

  it('maps a 500 to the projection kind (half-success language)', () => {
    const err = new ApiError(
      'Internal Server Error',
      500,
      { detail: 'projection failed' },
    );
    const out = mapCreateEntityError(err, 'WroteButCantProject');
    expect(out.kind).toBe('projection');
    expect(out.message).toContain('WroteButCantProject');
    // The copy must NOT read like an outright failure — the entity
    // exists server-side and the next refetch will surface it.
    expect(out.message.toLowerCase()).toContain('created server-side');
  });

  it('falls back to the unknown kind on an unexpected status', () => {
    const err = new ApiError('Bad Gateway', 502, { detail: 'upstream' });
    const out = mapCreateEntityError(err, 'X');
    expect(out.kind).toBe('unknown');
    expect(out.message).toBeTruthy();
  });

  it('falls back to the unknown kind with mapped copy for non-ApiError throws', () => {
    const err = new Error('network down');
    const out = mapCreateEntityError(err, 'X');
    expect(out.kind).toBe('unknown');
    // Error-UX pass 2026-07-03: raw technical messages no longer leak.
    expect(out.message).toBe(
      'Something went wrong while creating the entity. Please retry or contact Twincore Team.',
    );
  });

  it('falls back to a default message when the throw has no message', () => {
    const out = mapCreateEntityError({}, 'X');
    expect(out.kind).toBe('unknown');
    expect(out.message).toBeTruthy();
  });
});
