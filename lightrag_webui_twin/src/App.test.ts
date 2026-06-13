import { describe, expect, it } from 'vitest';
import {
  mapTwinQueryResponseForRetrievalTab,
  shouldUseFixtureFallback,
} from './App';
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

describe('mapTwinQueryResponseForRetrievalTab — answer_status wiring', () => {
  // Codex review on fix/tag-filter-honesty: the App.tsx callbacks
  // used to drop ``res.answer_status`` between the api.query response
  // and the RetrievalTab. The host then defaulted the field to
  // ``grounded`` and never suppressed the Sources panel on real
  // insufficient_information answers — exactly the TR-RET-02 step 1
  // contract that the RetrievalTab unit tests verified in isolation
  // but the wiring did not honour in production.

  it('propagates answer_status verbatim when set to insufficient_information', () => {
    const out = mapTwinQueryResponseForRetrievalTab({
      response: "Sorry, I'm not able to answer.",
      sources: [],
      answer_status: 'insufficient_information',
    });
    expect(out.answer_status).toBe('insufficient_information');
    expect(out.response).toBe("Sorry, I'm not able to answer.");
    expect(out.sources).toEqual([]);
  });

  it('propagates answer_status verbatim when grounded', () => {
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'A real answer.',
      sources: [
        { n: 1, type: 'file', name: '/a/runbook.pdf', score: 0.9 },
      ],
      answer_status: 'grounded',
    });
    expect(out.answer_status).toBe('grounded');
    expect(out.sources).toHaveLength(1);
    expect(out.sources[0].n).toBe(1);
    expect(out.sources[0].type).toBe('file');
  });

  it('passes answer_status through as undefined when absent (legacy backend)', () => {
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'Legacy answer.',
      sources: [],
    });
    // RetrievalTab treats undefined as grounded for back-compat;
    // the helper itself stays honest and forwards undefined.
    expect(out.answer_status).toBeUndefined();
  });

  it('coerces unknown source types to file', () => {
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'x',
      sources: [
        { n: 1, type: 'unknown-future-thing', name: 'mystery', score: 0.5 },
      ],
      answer_status: 'grounded',
    });
    expect(out.sources[0].type).toBe('file');
  });

  it('passes through file / url / confluence / sharepoint source types', () => {
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'x',
      sources: [
        { n: 1, type: 'file', name: 'a.pdf', score: 0.9 },
        { n: 2, type: 'url', name: 'https://example.test', score: 0.7 },
        { n: 3, type: 'confluence', name: 'KB-12', score: 0.6 },
        { n: 4, type: 'sharepoint', name: 'spfile', score: 0.5 },
      ],
      answer_status: 'grounded',
    });
    expect(out.sources.map((s) => s.type)).toEqual([
      'file',
      'url',
      'confluence',
      'sharepoint',
    ]);
  });
});
