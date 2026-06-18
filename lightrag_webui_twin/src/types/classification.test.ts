/**
 * Unit tests for the classification helpers (Couche 2).
 *
 * Covers both data shapes:
 *   - Legacy string ("internal" / "restricted" / "public")
 *   - Structured ClassificationResult (from PR #157 MIP extractor)
 */

import { describe, expect, it } from 'vitest';
import {
  getClassId,
  getClassName,
  getMipDisplayName,
  getMipTone,
  isAbove,
  isAboveInternal,
  isStructured,
  type ClassificationResult,
} from './classification';

function mkCls(overrides: Partial<ClassificationResult> = {}): ClassificationResult {
  return {
    class_id: 'C2',
    class_name: 'C2 Confidentiel',
    label_guid: '22222222-2222-2222-2222-222222222222',
    raw_name: 'C2 Confidentiel',
    set_date: '2026-03-12T14:22:01Z',
    method: 'Standard',
    source_format: 'ooxml',
    reason: null,
    meta: {},
    ...overrides,
  };
}

describe('isStructured', () => {
  it('returns true for a ClassificationResult', () => {
    expect(isStructured(mkCls())).toBe(true);
  });
  it('returns false for a string', () => {
    expect(isStructured('internal')).toBe(false);
  });
  it('returns false for undefined', () => {
    expect(isStructured(undefined)).toBe(false);
  });
});

describe('getClassId', () => {
  it('returns the class_id when given a ClassificationResult', () => {
    expect(getClassId(mkCls({ class_id: 'C3' }))).toBe('C3');
  });
  it('passes through a legacy string', () => {
    expect(getClassId('internal')).toBe('internal');
  });
  it('returns UNCLASSIFIED for undefined', () => {
    expect(getClassId(undefined)).toBe('UNCLASSIFIED');
  });
  it('returns UNCLASSIFIED for a structured payload with null class_id', () => {
    expect(getClassId(mkCls({ class_id: null }))).toBe('UNCLASSIFIED');
  });
});

describe('getClassName', () => {
  it('returns class_name from a ClassificationResult', () => {
    expect(getClassName(mkCls({ class_name: 'Class C2' }))).toBe('Class C2');
  });
  it('falls back to raw_name when class_name is missing', () => {
    expect(
      getClassName(mkCls({ class_name: null, raw_name: 'fallback name' })),
    ).toBe('fallback name');
  });
  it('falls back to class_id when both names are missing', () => {
    expect(getClassName(mkCls({ class_name: null, raw_name: null }))).toBe('C2');
  });
  it('falls back to unclassified when structured class_id is null', () => {
    expect(
      getClassName(mkCls({ class_id: null, class_name: null, raw_name: null })),
    ).toBe('unclassified');
  });
  it('returns the legacy string as-is', () => {
    expect(getClassName('restricted')).toBe('restricted');
  });
  it('returns "unclassified" for undefined', () => {
    expect(getClassName(undefined)).toBe('unclassified');
  });
});

describe('MIP display mapping', () => {
  it('maps legacy C ids to the new business names', () => {
    expect(getMipTone('C1')).toBe('public');
    expect(getMipTone('C2')).toBe('internal');
    expect(getMipTone('C3')).toBe('confidential');
    expect(getMipTone('C4')).toBe('secret');
    expect(getMipDisplayName('C2')).toBe('Internal');
  });

  it('keeps the new MIP names as first-class ids', () => {
    expect(getMipTone('Private')).toBe('private');
    expect(getMipDisplayName('Secret')).toBe('Secret');
  });
});

describe('isAbove (MIP ladder)', () => {
  it('Confidential outranks Internal', () => {
    expect(isAbove('Confidential', 'Internal')).toBe(true);
    expect(isAbove('C3', 'C2')).toBe(true);
  });
  it('Internal does not outrank Internal', () => {
    expect(isAbove('Internal', 'Internal')).toBe(false);
    expect(isAbove('C2', 'C2')).toBe(false);
  });
  it('Public does not outrank Internal', () => {
    expect(isAbove('Public', 'Internal')).toBe(false);
    expect(isAbove('C1', 'C2')).toBe(false);
  });
  it('Private outranks Internal', () => {
    expect(isAbove('Private', 'Internal')).toBe(true);
  });
  it('unknown class is treated as above (fail-closed)', () => {
    expect(isAbove('UNKNOWN', 'C2')).toBe(true);
    expect(isAbove('lol', 'C2')).toBe(true);
  });
  it('missing classification is not treated as a MIP class', () => {
    expect(isAbove(undefined, 'Internal')).toBe(false);
  });
  it('throws when threshold is not on the ladder', () => {
    expect(() => isAbove('C2', 'BogusClass')).toThrow(/threshold/);
  });
});

describe('isAboveInternal', () => {
  it('returns true for Confidential/Secret/Private structured', () => {
    expect(isAboveInternal(mkCls({ class_id: 'Confidential' }))).toBe(true);
    expect(isAboveInternal(mkCls({ class_id: 'Secret' }))).toBe(true);
    expect(isAboveInternal(mkCls({ class_id: 'Private' }))).toBe(true);
    expect(isAboveInternal(mkCls({ class_id: 'C3' }))).toBe(true);
    expect(isAboveInternal(mkCls({ class_id: 'C4' }))).toBe(true);
  });
  it('returns false for Public/Internal structured', () => {
    expect(isAboveInternal(mkCls({ class_id: 'Public' }))).toBe(false);
    expect(isAboveInternal(mkCls({ class_id: 'Internal' }))).toBe(false);
    expect(isAboveInternal(mkCls({ class_id: 'C1' }))).toBe(false);
    expect(isAboveInternal(mkCls({ class_id: 'C2' }))).toBe(false);
  });
  it('returns false for legacy "internal" / "public"', () => {
    expect(isAboveInternal('internal')).toBe(false);
    expect(isAboveInternal('public')).toBe(false);
    expect(isAboveInternal('Public')).toBe(false); // case-insensitive
  });
  it('returns true for legacy "restricted" / "confidential"', () => {
    expect(isAboveInternal('restricted')).toBe(true);
    expect(isAboveInternal('confidential')).toBe(true);
  });
  it('returns false for undefined because no MIP classification is applied', () => {
    expect(isAboveInternal(undefined)).toBe(false);
  });
  it('returns true for UNKNOWN structured (fail-closed)', () => {
    expect(isAboveInternal(mkCls({ class_id: 'UNKNOWN' }))).toBe(true);
  });
});
