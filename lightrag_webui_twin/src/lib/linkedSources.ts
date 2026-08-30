import type { RagDocType } from '../types/linkedSource';

export const DISABLE_REASON = 'disabled from the Twin Sources RAG grid';

export function inferRagSourceType(
  url: string,
): 'sharepoint' | 'confluence' | null {
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname) {
      return null;
    }
    const host = parsed.hostname.toLowerCase();
    return host.includes('sharepoint') ? 'sharepoint' : 'confluence';
  } catch {
    return null;
  }
}

export function describeRagScope(
  url: string,
  docType: RagDocType,
): string | null {
  const sourceType = inferRagSourceType(url);
  if (sourceType === null) return null;
  if (docType !== 'general') {
    return sourceType === 'sharepoint'
      ? 'This document only'
      : 'This page only';
  }
  const value = url.toLowerCase();
  return value.includes('spacekey=') || /\/spaces\/[^/]+\/?$/.test(value)
    ? 'Entire space, recursive'
    : 'Root page and descendants, recursive';
}
