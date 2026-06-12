import { describe, expect, it } from 'vitest';
import { shouldUseFixtureFallback } from './App';
import { TAG_FIXTURES } from './fixtures';
import { tagCatalogForSuggestions } from './utils/tags';

describe('shouldUseFixtureFallback', () => {
  it('allows fixture fallbacks in dev when MSW is active', () => {
    expect(shouldUseFixtureFallback({ dev: true })).toBe(true);
  });

  it('allows fixture fallbacks for the explicit standalone MSW demo build', () => {
    expect(
      shouldUseFixtureFallback({
        dev: false,
        forceMsw: 'true',
        useMsw: 'false',
      }),
    ).toBe(true);
  });

  it('disables fixture fallbacks in real-backend mode', () => {
    expect(shouldUseFixtureFallback({ dev: false })).toBe(false);
    expect(shouldUseFixtureFallback({ dev: true, useMsw: 'false' })).toBe(false);
  });
});

describe('tagCatalogForSuggestions', () => {
  it('uses governance tags as the single runtime suggestion catalog', () => {
    const catalog = tagCatalogForSuggestions(TAG_FIXTURES);
    const names = catalog.map((tag) => tag.tag);

    expect(names).toContain('rman');
    expect(names).toContain('memgraph');
    expect(names).not.toContain('argocd');
    expect(catalog.every((tag) => tag.status !== 'deprecated')).toBe(true);
  });
});
