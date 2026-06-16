import type { Document } from '../types/document';

const HASH_METADATA_KEYS = [
  'content_hash',
  'contentHash',
  'file_hash',
  'fileHash',
  'sha256',
  'sha1',
  'md5',
  'checksum',
] as const;

export interface DocumentContentHash {
  label: string;
  value: string;
}

export function documentContentHash(doc: Document): DocumentContentHash | null {
  for (const key of HASH_METADATA_KEYS) {
    const value = doc.metadata[key];
    if (typeof value === 'string' && value.trim() !== '') {
      return { label: hashLabel(key), value: value.trim().toLocaleLowerCase() };
    }
  }
  return null;
}

function hashLabel(key: (typeof HASH_METADATA_KEYS)[number]): string {
  if (key === 'sha1') return 'SHA1';
  if (key === 'sha256') return 'SHA256';
  if (key === 'md5') return 'MD5';
  if (key === 'contentHash') return 'content_hash';
  if (key === 'fileHash') return 'file_hash';
  return key;
}

function sourceKey(doc: Document): string {
  const hash = documentContentHash(doc);
  if (hash) return `hash:${hash.value}`;
  return `path:${doc.file_path.trim().toLocaleLowerCase()}`;
}

function statusRank(doc: Document): number {
  if (doc._optimisticUpload) return 5;
  switch (doc.status) {
    case 'PROCESSED':
      return 4;
    case 'PROCESSING':
      return (doc.chunks_count ?? 0) > 0 ? 3 : 2;
    case 'PENDING':
      return 2;
    case 'FAILED':
      return 1;
    default:
      return 0;
  }
}

function updatedMs(doc: Document): number {
  const parsed = Date.parse(doc.updated_at);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function compareAttempts(a: Document, b: Document): number {
  const rankDelta = statusRank(a) - statusRank(b);
  if (rankDelta !== 0) return rankDelta;
  const chunkDelta = (a.chunks_count ?? 0) - (b.chunks_count ?? 0);
  if (chunkDelta !== 0) return chunkDelta;
  return updatedMs(a) - updatedMs(b);
}

export function dedupeDocumentsBySource(
  docs: readonly Document[],
): readonly Document[] {
  const bestBySource = new Map<string, Document>();
  for (const doc of docs) {
    const key = sourceKey(doc);
    const current = bestBySource.get(key);
    if (!current || compareAttempts(doc, current) > 0) {
      bestBySource.set(key, doc);
    }
  }

  const kept = new Set(bestBySource.values());
  return docs.filter((doc) => kept.has(doc));
}
