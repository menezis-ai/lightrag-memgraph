import type { TagEntry } from '../types/tag';

export function tagCatalogForSuggestions(
  tags: readonly TagEntry[],
): readonly TagEntry[] {
  return tags.filter(
    (tag) =>
      tag.tier !== 'requested' &&
      tag.status !== 'rejected' &&
      tag.status !== 'deprecated',
  );
}

function rankTag(tag: TagEntry, query: string): number {
  if (!query) return 0;
  const name = tag.tag.toLowerCase();
  const aliases = tag.aliases.map((alias) => alias.toLowerCase());
  const def = tag.def.toLowerCase();
  if (name === query) return 0;
  if (name.startsWith(query)) return 1;
  if (aliases.some((alias) => alias === query)) return 2;
  if (aliases.some((alias) => alias.startsWith(query))) return 3;
  if (name.includes(query)) return 4;
  if (aliases.some((alias) => alias.includes(query))) return 5;
  if (def.includes(query)) return 6;
  return Number.POSITIVE_INFINITY;
}

export function tagMatchesQuery(tag: TagEntry, query: string): boolean {
  return rankTag(tag, query.toLowerCase().trim()) < Number.POSITIVE_INFINITY;
}

export function tagSuggestionComparator(query: string) {
  const needle = query.toLowerCase().trim();
  return (a: TagEntry, b: TagEntry): number =>
    rankTag(a, needle) - rankTag(b, needle) || a.tag.localeCompare(b.tag);
}
