export const MAX_FOLDER_FILES = 500;
export const MAX_FOLDER_DEPTH = 20;
export const MAX_FOLDER_TOTAL_BYTES = 1024 * 1024 * 1024;
export const MAX_RELATIVE_PATH_BYTES = 140;

const STORAGE_PREFIX = 'twinrel_';

/** Validate the server-owned POSIX/NFC relative-path contract. */
export function normalizeRelativeUploadPath(value: string): string {
  const normalized = value.trim().normalize('NFC');
  if (!normalized || normalized.startsWith('/') || normalized.includes('\\')) {
    throw new Error('Relative path must be a non-empty POSIX path');
  }
  if ([...normalized].some((char) => {
    const code = char.codePointAt(0) ?? 0;
    return code < 32 || code === 127;
  })) {
    throw new Error('Relative path contains control characters');
  }
  const parts = normalized.split('/');
  if (parts.some((part) => part === '' || part === '.' || part === '..')) {
    throw new Error('Relative path contains an empty or traversal segment');
  }
  if (parts.length > MAX_FOLDER_DEPTH) {
    throw new Error(`Relative path exceeds ${MAX_FOLDER_DEPTH} levels`);
  }
  if (new TextEncoder().encode(normalized).byteLength > MAX_RELATIVE_PATH_BYTES) {
    throw new Error(
      `Relative path exceeds ${MAX_RELATIVE_PATH_BYTES} UTF-8 bytes`,
    );
  }
  return parts.join('/');
}

function base64UrlUtf8(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary)
    .replaceAll('+', '-')
    .replaceAll('/', '_')
    .replace(/=+$/, '');
}

/** Lossless, traversal-safe LightRAG basename used as source identity. */
export function canonicalUploadFileName(relativePath: string): string {
  const normalized = normalizeRelativeUploadPath(relativePath);
  if (!normalized.includes('/')) return normalized;
  const basename = normalized.slice(normalized.lastIndexOf('/') + 1);
  const dot = basename.lastIndexOf('.');
  const suffix = dot >= 0 ? basename.slice(dot) : '';
  return `${STORAGE_PREFIX}${base64UrlUtf8(normalized)}${suffix}`;
}
