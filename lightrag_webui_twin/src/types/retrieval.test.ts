/**
 * Pure-function tests for parseAnswer + relTime. No DOM, no fixtures.
 */

import { describe, expect, it } from 'vitest';
import { parseAnswer, relTime } from './retrieval';

describe('parseAnswer', () => {
  it('returns empty array for an empty token list', () => {
    expect(parseAnswer([])).toEqual([]);
  });

  it('emits a plain text part for a marker-free token', () => {
    expect(parseAnswer(['Hello world'])).toEqual([
      { type: 'text', value: 'Hello world' },
    ]);
  });

  it('extracts a citation', () => {
    expect(parseAnswer(['See ', '{cite:3}', ' for details'])).toEqual([
      { type: 'text', value: 'See ' },
      { type: 'cite', value: 3 },
      { type: 'text', value: ' for details' },
    ]);
  });

  it('extracts a code span', () => {
    expect(parseAnswer(['Run `shutdown immediate` now'])).toEqual([
      { type: 'text', value: 'Run ' },
      { type: 'code', value: 'shutdown immediate' },
      { type: 'text', value: ' now' },
    ]);
  });

  it('handles cite + code in the same token', () => {
    expect(
      parseAnswer(['Use `lsnrctl` then check {cite:7} for kernel notes']),
    ).toEqual([
      { type: 'text', value: 'Use ' },
      { type: 'code', value: 'lsnrctl' },
      { type: 'text', value: ' then check ' },
      { type: 'cite', value: 7 },
      { type: 'text', value: ' for kernel notes' },
    ]);
  });
});

describe('relTime', () => {
  it('returns empty for null/undefined/empty', () => {
    expect(relTime(null)).toBe('');
    expect(relTime(undefined)).toBe('');
    expect(relTime('')).toBe('');
  });

  it('returns "" for unparseable strings', () => {
    expect(relTime('garbage')).toBe('');
  });

  it('returns "now" for fresh timestamps', () => {
    expect(relTime(Date.now())).toBe('now');
    expect(relTime(Date.now() - 10_000)).toBe('now');
  });

  it('returns Xm minutes', () => {
    expect(relTime(Date.now() - 5 * 60_000)).toBe('5m');
  });

  it('returns Xh hours', () => {
    expect(relTime(Date.now() - 4 * 3_600_000)).toBe('4h');
  });

  it('returns Xd days', () => {
    expect(relTime(Date.now() - 3 * 86_400_000)).toBe('3d');
  });
});
