import { describe, expect, it } from 'vitest';
import { shouldUseFixtureFallback } from './App';
import { mapTwinQueryResponseForRetrievalTab } from './api/twinQueryResponse';
import { TAG_FIXTURES } from './fixtures';
import { isActiveCatalogTag, tagCatalogForSuggestions } from './utils/tags';

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
  it('centralizes the steward-approved active-tag predicate', () => {
    expect(isActiveCatalogTag({ status: 'active', tier: 2 })).toBe(true);
    expect(isActiveCatalogTag({ status: 'active', tier: 'requested' })).toBe(false);
    expect(isActiveCatalogTag({ status: 'deprecated', tier: 2 })).toBe(false);
  });

  it('uses governance tags as the single runtime suggestion catalog', () => {
    const catalog = tagCatalogForSuggestions(TAG_FIXTURES);
    const names = catalog.map((tag) => tag.tag);

    expect(names).toContain('rman');
    expect(names).toContain('memgraph');
    expect(names).not.toContain('argocd');
    expect(names).not.toContain('graphrag');
    expect(catalog.every((tag) => tag.status === 'active')).toBe(true);
    expect(catalog.every((tag) => tag.tier !== 'requested')).toBe(true);
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
        {
          n: 1,
          type: 'file',
          name: '/a/runbook.pdf',
          score: 0.9,
          doc_id: 'doc-runbook',
          chunk_id: 'chunk-runbook-2',
        },
      ],
      answer_status: 'grounded',
    });
    expect(out.answer_status).toBe('grounded');
    expect(out.sources).toHaveLength(1);
    expect(out.sources[0].n).toBe(1);
    expect(out.sources[0].type).toBe('file');
    expect(out.sources[0].doc_id).toBe('doc-runbook');
    expect(out.sources[0].chunk_id).toBe('chunk-runbook-2');
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

  it('preserves unavailable and real retrieval score values', () => {
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'x',
      sources: [
        {
          n: 1,
          type: 'file',
          name: 'null.pdf',
          score: null,
          retrieval_origin: 'graph',
        },
        { n: 2, type: 'file', name: 'absent.pdf' },
        { n: 3, type: 'file', name: 'zero.pdf', score: 0 },
        { n: 4, type: 'file', name: 'ranked.pdf', score: 0.82 },
      ],
      answer_status: 'grounded',
    });

    expect(out.sources.map((source) => source.score)).toEqual([
      null,
      undefined,
      0,
      0.82,
    ]);
    expect(out.sources[0].retrieval_origin).toBe('graph');
  });

  it('passes a structurally valid paragraph anchor through', () => {
    const anchor = {
      start: 216,
      end: 640,
      paragraph_idx: 1,
      paragraph_count: 4,
      confidence: 0.62,
      method: 'lexical_overlap',
    };
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'x',
      sources: [
        { n: 1, type: 'file', name: 'a.pdf', score: 0.9, anchor },
        { n: 2, type: 'file', name: 'b.pdf', score: 0.8, anchor: null },
        { n: 3, type: 'file', name: 'c.pdf', score: 0.7 },
      ],
      answer_status: 'grounded',
    });
    expect(out.sources[0].anchor).toEqual(anchor);
    expect(out.sources[1].anchor).toBeUndefined();
    expect(out.sources[2].anchor).toBeUndefined();
  });

  it('drops malformed anchors instead of letting bad offsets reach the UI', () => {
    const base = {
      paragraph_idx: 0,
      paragraph_count: 1,
      confidence: 0.5,
      method: 'lexical_overlap',
    };
    const out = mapTwinQueryResponseForRetrievalTab({
      response: 'x',
      sources: [
        // end <= start
        { n: 1, type: 'file', name: 'a.pdf', anchor: { ...base, start: 10, end: 10 } },
        // negative start
        { n: 2, type: 'file', name: 'b.pdf', anchor: { ...base, start: -1, end: 5 } },
        // non-integer offsets
        { n: 3, type: 'file', name: 'c.pdf', anchor: { ...base, start: 0.5, end: 5 } },
        // non-finite confidence
        {
          n: 4,
          type: 'file',
          name: 'd.pdf',
          anchor: { ...base, start: 0, end: 5, confidence: Number.NaN },
        },
      ],
      answer_status: 'grounded',
    });
    expect(out.sources.map((source) => source.anchor)).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
    ]);
  });
});
