import type { GraphEntity } from '../../types/graph';

export const PINNED_STORAGE_KEY = 'twin.kg.pinned.v1';

export const tagsOf = (entity: GraphEntity): readonly string[] => entity.tags ?? [];

export const readPinnedEntityIds = (): readonly string[] => {
  if (globalThis.window === undefined) return [];
  try {
    const parsed = JSON.parse(
      globalThis.localStorage.getItem(PINNED_STORAGE_KEY) ?? '[]',
    );
    return Array.isArray(parsed)
      ? parsed.filter((id): id is string => typeof id === 'string')
      : [];
  } catch {
    return [];
  }
};
