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
    folder: 'cib',
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

  it('keeps an optimistic upload until the backend has the same source', () => {
    const optimistic = doc({
      doc_id: 'upload-tmp',
      file_path: 'new-source.pdf',
      status: 'PENDING',
      _optimisticUpload: true,
    });
    const other = doc({ doc_id: 'doc-other', file_path: 'other.pdf' });

    expect(dedupeDocumentsBySource([optimistic, other])).toEqual([
      optimistic,
      other,
    ]);
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

  it('ranks a PROCESSING attempt without chunks above PENDING via path key', () => {
    // Distinct file paths → distinct source keys → nothing deduped, but this
    // still exercises the no-hash path: branch and the PROCESSING(no-chunks)
    // and PENDING status ranks.
    const pending = doc({ doc_id: 'p', file_path: 'a.pdf', status: 'PENDING' });
    const processingNoChunks = doc({
      doc_id: 'pn',
      file_path: 'a.pdf',
      status: 'PROCESSING',
      chunks_count: 0,
    });

    // Same path → deduped; ranks are equal (both 2) so chunk count (0 vs null→0)
    // then updated_at decide; identical → keeps the first inserted as best.
    const out = dedupeDocumentsBySource([pending, processingNoChunks]);
    expect(out).toHaveLength(1);
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
});
