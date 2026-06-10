import { describe, expect, it } from 'vitest';
import type { Document } from '../types/document';
import { dedupeDocumentsBySource } from './documents';

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
});
