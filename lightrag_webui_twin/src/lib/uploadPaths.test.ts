import { describe, expect, it } from 'vitest';
import {
  canonicalUploadFileName,
  MAX_FOLDER_DEPTH,
  MAX_FOLDER_TOTAL_BYTES,
  MAX_RELATIVE_PATH_BYTES,
  normalizeRelativeUploadPath,
} from './uploadPaths';

describe('folder upload path contract', () => {
  it('keeps duplicate basenames in different folders distinct', () => {
    const first = canonicalUploadFileName('team-a/report.pdf');
    const second = canonicalUploadFileName('team-b/report.pdf');

    expect(first).not.toBe(second);
    expect(first).toMatch(/^twinrel_.*\.pdf$/);
    expect(second).toMatch(/^twinrel_.*\.pdf$/);
  });

  it.each([
    ['../secret.txt', 'Relative path contains an empty or traversal segment'],
    ['root/../secret.txt', 'Relative path contains an empty or traversal segment'],
    ['/absolute.txt', 'Relative path must be a non-empty POSIX path'],
    ['root\\windows.txt', 'Relative path must be a non-empty POSIX path'],
    ['root//empty.txt', 'Relative path contains an empty or traversal segment'],
    ['root/./dot.txt', 'Relative path contains an empty or traversal segment'],
  ])('rejects unsafe relative path %s', (path, error) => {
    expect(() => normalizeRelativeUploadPath(path)).toThrow(error);
  });

  it('trims and NFC-normalizes a valid path while rejecting control characters', () => {
    expect(normalizeRelativeUploadPath('  équipe/cafe\u0301.txt  ')).toBe(
      'équipe/café.txt',
    );
    expect(() => normalizeRelativeUploadPath('team/report\u0000.txt')).toThrow(
      'Relative path contains control characters',
    );
  });

  it('enforces exact depth and UTF-8 byte boundaries', () => {
    const deepestAllowed = Array.from(
      { length: MAX_FOLDER_DEPTH },
      () => 'folder',
    ).join('/');
    const oneLevelTooDeep = `${deepestAllowed}/file.txt`;
    const maxBytes = 'a'.repeat(MAX_RELATIVE_PATH_BYTES);

    expect(normalizeRelativeUploadPath(deepestAllowed)).toBe(deepestAllowed);
    expect(() => normalizeRelativeUploadPath(oneLevelTooDeep)).toThrow(
      `Relative path exceeds ${MAX_FOLDER_DEPTH} levels`,
    );
    expect(normalizeRelativeUploadPath(maxBytes)).toBe(maxBytes);
    expect(() => normalizeRelativeUploadPath(`${maxBytes}a`)).toThrow(
      `Relative path exceeds ${MAX_RELATIVE_PATH_BYTES} UTF-8 bytes`,
    );
  });

  it('keeps the 1 GiB total-folder budget exact', () => {
    expect(MAX_FOLDER_TOTAL_BYTES).toBe(1024 * 1024 * 1024);
  });

  it('preserves the old basename identity for ordinary file selection', () => {
    expect(canonicalUploadFileName('report.pdf')).toBe('report.pdf');
  });

  it('uses a lossless URL-safe UTF-8 identity and preserves hidden-file suffixes', () => {
    expect(canonicalUploadFileName('folder/Ͽ.txt')).toBe(
      'twinrel_Zm9sZGVyL8-_LnR4dA.txt',
    );
    expect(canonicalUploadFileName('folder/.env')).toBe(
      'twinrel_Zm9sZGVyLy5lbnY.env',
    );
  });
});
