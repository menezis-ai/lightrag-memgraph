/**
 * Pins the canonical status vocabulary helpers (audit 2026-07-02, DUP-1).
 * These behaviours are wire-contract: `normalizeDocumentStatus` must keep the
 * exact semantics of the ingress normalizer it replaced in `api/resources.ts`,
 * and `statusCountFor` the exact dual-read order of the inline
 * `counts.processed ?? counts.PROCESSED ?? 0` it replaced in `DocumentsTab`.
 */

import { describe, expect, it } from 'vitest';
import {
  DOC_STATUSES,
  LIGHTRAG_15X_STATUSES,
  normalizeDocumentStatus,
  statusCountFor,
} from './docStatus';

describe('normalizeDocumentStatus', () => {
  it('uppercases the four LightRAG-native lowercase statuses', () => {
    expect(normalizeDocumentStatus('pending')).toBe('PENDING');
    expect(normalizeDocumentStatus('processing')).toBe('PROCESSING');
    expect(normalizeDocumentStatus('processed')).toBe('PROCESSED');
    expect(normalizeDocumentStatus('failed')).toBe('FAILED');
  });

  it('passes canonical UPPERCASE values through', () => {
    for (const status of DOC_STATUSES) {
      expect(normalizeDocumentStatus(status)).toBe(status);
    }
  });

  it('coerces the LightRAG 1.5.x statuses to PENDING (documented coercion)', () => {
    for (const status of LIGHTRAG_15X_STATUSES) {
      expect(normalizeDocumentStatus(status)).toBe('PENDING');
      expect(normalizeDocumentStatus(status.toLowerCase())).toBe('PENDING');
    }
  });

  it('falls back to PENDING on unknown or non-string input', () => {
    expect(normalizeDocumentStatus('weird')).toBe('PENDING');
    expect(normalizeDocumentStatus('')).toBe('PENDING');
    expect(normalizeDocumentStatus(undefined)).toBe('PENDING');
    expect(normalizeDocumentStatus(42)).toBe('PENDING');
  });
});

describe('statusCountFor', () => {
  it('reads the native lowercase bucket first', () => {
    expect(statusCountFor({ processed: 3, PROCESSED: 9 }, 'PROCESSED')).toBe(3);
  });

  it('falls back to the twin UPPERCASE bucket', () => {
    expect(statusCountFor({ PROCESSED: 9 }, 'PROCESSED')).toBe(9);
  });

  it('defaults to 0 when neither casing is present', () => {
    expect(statusCountFor({}, 'FAILED')).toBe(0);
  });
});
