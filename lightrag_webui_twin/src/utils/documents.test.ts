import { describe, expect, it } from 'vitest';
import type { Document } from '../types/document';
import { dedupeDocumentsBySource, documentContentHash } from './documents';

function doc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'doc-1',
    track_id: null,
    file_path: 'source.pdf',
    content_summary: 'summary',
    content_length: 100,
    status: 'PENDING',
    chunks_count: null,
    created_at: '2026-06-10T10:00:00Z',
    updated_at: '2026-06-10T10:00:00Z',
    error_msg: null,
    metadata: {},
    type: 'file',
    tags: [],
    folder: 'demo',
    visibility: 'internal',
    ...overrides,
  };
}

describe('dedupeDocumentsBySource', () => {
  it('keeps a newer non-failed attempt over an older failed extraction', () => {
    const failed = doc({
      doc_id: 'error-old',
      status: 'FAILED',
      updated_at: '2026-06-10T10:00:00Z',
    });
    const retry = doc({
      doc_id: 'doc-new',
      status: 'PENDING',
      updated_at: '2026-06-10T10:10:00Z',
    });

    expect(dedupeDocumentsBySource([failed, retry])).toEqual([retry]);
  });

  it('prefers processed documents over processing attempts with chunks', () => {
    const processing = doc({
      doc_id: 'doc-processing',
      status: 'PROCESSING',
      chunks_count: 120,
      updated_at: '2026-06-10T10:15:00Z',
    });
    const processed = doc({
      doc_id: 'doc-processed',
      status: 'PROCESSED',
      chunks_count: 90,
      updated_at: '2026-06-10T10:05:00Z',
    });

    expect(dedupeDocumentsBySource([processing, processed])).toEqual([
      processed,
    ]);
  });

  it('deduplicates renamed sources when their content hash matches', () => {
    const failedCopy = doc({
      doc_id: 'error-renamed',
      file_path: 'renamed-copy.pdf',
      status: 'FAILED',
      metadata: { sha256: 'ABC123' },
      updated_at: '2026-06-10T10:15:00Z',
    });
    const processedOriginal = doc({
      doc_id: 'doc-original',
      file_path: 'original.pdf',
      status: 'PROCESSED',
      chunks_count: 12,
      metadata: { content_hash: 'abc123' },
      updated_at: '2026-06-10T10:00:00Z',
    });

    expect(dedupeDocumentsBySource([failedCopy, processedOriginal])).toEqual([
      processedOriginal,
    ]);
  });

  it('keeps an optimistic upload over a backend row for the same source', () => {
    const optimistic = doc({
      doc_id: 'upload-tmp',
      file_path: 'new-source.pdf',
      status: 'PENDING',
      _optimisticUpload: true,
    });
    const backend = doc({
      doc_id: 'backend-row',
      file_path: 'new-source.pdf',
      status: 'PROCESSED',
      chunks_count: 12,
    });

    expect(dedupeDocumentsBySource([optimistic, backend])).toEqual([optimistic]);
  });

  it('breaks a same-rank tie by chunk count', () => {
    const fewChunks = doc({
      doc_id: 'few',
      status: 'PROCESSING',
      chunks_count: 5,
      updated_at: '2026-06-10T11:00:00Z',
    });
    const manyChunks = doc({
      doc_id: 'many',
      status: 'PROCESSING',
      chunks_count: 40,
      updated_at: '2026-06-10T09:00:00Z',
    });

    expect(dedupeDocumentsBySource([fewChunks, manyChunks])).toEqual([
      manyChunks,
    ]);
  });

  it('breaks a same-rank same-chunk tie by updated_at recency', () => {
    const older = doc({
      doc_id: 'older',
      status: 'PROCESSED',
      chunks_count: 10,
      updated_at: '2026-06-10T08:00:00Z',
    });
    const newer = doc({
      doc_id: 'newer',
      status: 'PROCESSED',
      chunks_count: 10,
      updated_at: '2026-06-10T12:00:00Z',
    });

    expect(dedupeDocumentsBySource([older, newer])).toEqual([newer]);
  });

  it('treats an unparseable updated_at as epoch zero (loses the tiebreak)', () => {
    const bad = doc({
      doc_id: 'bad-date',
      status: 'PROCESSED',
      chunks_count: 10,
      updated_at: 'not-a-date',
    });
    const good = doc({
      doc_id: 'good-date',
      status: 'PROCESSED',
      chunks_count: 10,
      updated_at: '2026-06-10T00:00:01Z',
    });

    expect(dedupeDocumentsBySource([bad, good])).toEqual([good]);
  });

  it('distinguishes PROCESSING rows with indexed chunks from source-order ties', () => {
    const pending = doc({
      doc_id: 'p',
      file_path: 'a.pdf',
      status: 'PENDING',
    });
    const processingNoChunks = doc({
      doc_id: 'pn',
      file_path: 'a.pdf',
      status: 'PROCESSING',
      chunks_count: 0,
    });

    // Both rank 2 when PROCESSING has no chunks, so the original server order
    // is stable. A positive chunk count elevates PROCESSING above PENDING.
    expect(dedupeDocumentsBySource([pending, processingNoChunks])).toEqual([
      pending,
    ]);
    const processingWithChunks = doc({
      doc_id: 'pc',
      file_path: 'a.pdf',
      status: 'PROCESSING',
      chunks_count: 1,
    });
    expect(dedupeDocumentsBySource([pending, processingWithChunks])).toEqual([
      processingWithChunks,
    ]);
  });

  it('ranks PENDING above FAILED and retains the first exact tie', () => {
    const pending = doc({ doc_id: 'pending', file_path: 'same.pdf' });
    const failed = doc({
      doc_id: 'failed',
      file_path: 'same.pdf',
      status: 'FAILED',
    });
    const sameAttempt = doc({ doc_id: 'same', file_path: 'same.pdf' });

    expect(dedupeDocumentsBySource([failed, pending])).toEqual([pending]);
    expect(dedupeDocumentsBySource([pending, sameAttempt])).toEqual([pending]);
  });

  it('keeps the better attempt when candidates arrive in reverse rank order', () => {
    const processed = doc({
      doc_id: 'processed',
      file_path: 'rank.pdf',
      status: 'PROCESSED',
      chunks_count: 1,
    });
    const pending = doc({ doc_id: 'pending', file_path: 'rank.pdf' });
    const manyChunks = doc({
      doc_id: 'many',
      file_path: 'chunks.pdf',
      status: 'PROCESSING',
      chunks_count: 10,
    });
    const fewChunks = doc({
      doc_id: 'few',
      file_path: 'chunks.pdf',
      status: 'PROCESSING',
      chunks_count: 1,
    });
    const newer = doc({
      doc_id: 'newer',
      file_path: 'date.pdf',
      status: 'PROCESSED',
      chunks_count: 1,
      updated_at: '2026-06-10T12:00:00Z',
    });
    const older = doc({
      doc_id: 'older',
      file_path: 'date.pdf',
      status: 'PROCESSED',
      chunks_count: 1,
      updated_at: '2026-06-10T08:00:00Z',
    });

    expect(dedupeDocumentsBySource([processed, pending])).toEqual([processed]);
    expect(dedupeDocumentsBySource([manyChunks, fewChunks])).toEqual([
      manyChunks,
    ]);
    expect(dedupeDocumentsBySource([newer, older])).toEqual([newer]);
  });

  it('normalizes fallback paths without collapsing distinct Unicode sources', () => {
    const padded = doc({
      doc_id: 'padded',
      file_path: '  REPORT.PDF  ',
      status: 'FAILED',
    });
    const canonical = doc({
      doc_id: 'canonical',
      file_path: 'report.pdf',
      status: 'PENDING',
    });
    const sharpS = doc({ doc_id: 'sharp-s', file_path: 'straße.pdf' });
    const doubleS = doc({ doc_id: 'double-s', file_path: 'strasse.pdf' });

    expect(dedupeDocumentsBySource([padded, canonical])).toEqual([canonical]);
    expect(dedupeDocumentsBySource([sharpS, doubleS])).toEqual([
      sharpS,
      doubleS,
    ]);
  });

  it('keeps an unknown status (rank 0) only when nothing better shares the source', () => {
    const unknown = doc({
      doc_id: 'u',
      file_path: 'z.pdf',
      status: 'UNKNOWN' as Document['status'],
    });
    expect(dedupeDocumentsBySource([unknown])).toEqual([unknown]);
  });
});

describe('documentContentHash', () => {
  it('returns null when no hash metadata is present', () => {
    expect(documentContentHash(doc())).toBeNull();
  });

  it('skips blank hash values and falls through to the next key', () => {
    const d = doc({
      metadata: { content_hash: '   ', sha256: 'DEADBEEF' },
    });
    expect(documentContentHash(d)).toEqual({
      label: 'SHA256',
      value: 'deadbeef',
    });
  });

  it('trims a chosen hash value before lowercasing it', () => {
    expect(documentContentHash(doc({ metadata: { sha256: '  DEADBEEF  ' } }))).toEqual({
      label: 'SHA256',
      value: 'deadbeef',
    });
  });

  it('ignores non-string hash values', () => {
    const d = doc({
      metadata: { content_hash: 12345 as unknown as string },
    });
    expect(documentContentHash(d)).toBeNull();
  });

  it.each([
    ['sha1', 'SHA1'],
    ['sha256', 'SHA256'],
    ['md5', 'MD5'],
    ['contentHash', 'content_hash'],
    ['fileHash', 'file_hash'],
    ['content_hash', 'content_hash'],
    ['file_hash', 'file_hash'],
    ['checksum', 'checksum'],
  ])('labels the %s metadata key as %s', (key, label) => {
    const d = doc({ metadata: { [key]: 'AbC123' } });
    expect(documentContentHash(d)).toEqual({ label, value: 'abc123' });
  });

  it('honors the key priority order (content_hash before sha256)', () => {
    const d = doc({
      metadata: { content_hash: 'FIRST', sha256: 'SECOND' },
    });
    expect(documentContentHash(d)).toEqual({
      label: 'content_hash',
      value: 'first',
    });
  });

  it('does not merge different content hashes', () => {
    const first = doc({
      doc_id: 'hash-first',
      file_path: 'first.pdf',
      metadata: { sha256: 'aaaa' },
    });
    const second = doc({
      doc_id: 'hash-second',
      file_path: 'second.pdf',
      metadata: { sha256: 'bbbb' },
    });

    expect(dedupeDocumentsBySource([first, second])).toEqual([first, second]);
  });
});
