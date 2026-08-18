/**
 * Unit tests for DocDetailPanel (#105).
 *
 * Behaviors under test:
 *   - returns null when no doc selected
 *   - 3 tabs: Chunks, Lineage, Audit; default = Chunks
 *   - Chunks fetches from /documents/{id}/chunks
 *   - Lineage shows uploader, dates, tags
 *   - Audit fetches /twin/api/activity?resource.id={id}
 *   - View raw notice explains the future raw download endpoint
 *   - Footer buttons fire onRetag / onReprocess / onDelete
 *   - Escape closes
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { DocDetailPanel } from './DocDetailPanel';
import type { Document } from '../types/document';
import { __resetAuthConfigCacheForTests } from '../hooks/useAuth';

function makeDoc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'd-test-1',
    track_id: 'tk_1',
    type: 'file',
    file_path: 'sample.pdf',
    content_summary: 'A doc',
    content_length: 200,
    tags: ['rman'],
    status: 'PROCESSED',
    chunks_count: 3,
    created_at: '2026-05-29T14:00:00Z',
    updated_at: '2026-05-29T14:00:00Z',
    error_msg: null,
    metadata: { uploader: 'claire.benoit', classification: 'internal' },
    visibility: 'private',
    folder: 'default',
    ...overrides,
  };
}

const auditRequests: string[] = [];

const server = setupServer(
  http.get('*/documents/:id/chunks', ({ params }) =>
    HttpResponse.json([
      { chunk_id: `${String(params.id)}_c0`, order: 0, text: 'Chunk 1 text content for the document.' },
      { chunk_id: `${String(params.id)}_c1`, order: 1, text: 'Chunk 2 text content with extra details.' },
    ]),
  ),
  http.get('*/twin/api/activity', ({ request }) => {
    auditRequests.push(request.url);
    return HttpResponse.json({
      items: [
        {
          id: 'a1',
          ts: '2026-05-29T13:00:00Z',
          rel: '99d ago',
          day: '2026-05-29',
          kind: 'tag-mutation',
          sev: 'info',
          actor: { user: 'claire.benoit', role: 'steward' },
          target: { type: 'source', label: 'sample.pdf', id: 'd-test-1' },
          summary: 'Tag applied: rman',
          meta: {},
        },
      ],
      total: 1,
      nowMs: Date.parse('2026-05-29T16:00:00Z'),
    });
  }),
);

function wrap(qc: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

beforeEach(() => {
  __resetAuthConfigCacheForTests();
  auditRequests.length = 0;
  (window as Window & typeof globalThis).__twinConfig = undefined;
  server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
  server.resetHandlers();
  server.close();
});

describe('DocDetailPanel — visibility', () => {
  it('returns null when doc is null', () => {
    const Wrap = wrap(new QueryClient());
    const { container } = render(
      <Wrap>
        <DocDetailPanel doc={null} onClose={() => {}} />
      </Wrap>,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders the header with the file_path and the 3 tabs', () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} />
      </Wrap>,
    );
    expect(screen.getByTestId('doc-detail-panel')).toBeInTheDocument();
    expect(screen.getByTestId('doc-detail-tab-chunks')).toBeInTheDocument();
    expect(screen.getByTestId('doc-detail-tab-lineage')).toBeInTheDocument();
    expect(screen.getByTestId('doc-detail-tab-audit')).toBeInTheDocument();
  });
});

describe('DocDetailPanel — tabs', () => {
  it('can open directly on the Lineage tab', () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          initialTab="lineage"
          onClose={() => {}}
        />
      </Wrap>,
    );

    expect(screen.getByTestId('doc-detail-tab-lineage')).toHaveAttribute(
      'aria-current',
      'true',
    );
    expect(screen.getByTestId('doc-detail-lineage')).toBeInTheDocument();
  });

  it('Chunks tab fetches and renders chunk list', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} />
      </Wrap>,
    );
    await waitFor(() =>
      expect(screen.getByTestId('doc-detail-chunks-list')).toBeInTheDocument(),
    );
    expect(screen.getByText('Chunk 1 text content for the document.')).toBeInTheDocument();
    expect(screen.getByText('Chunk 2 text content with extra details.')).toBeInTheDocument();
  });

  it('shows two-line chunk previews with a full-text toggle', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json([
          {
            chunk_id: 'chunk-long',
            order: 0,
            text: 'Line one of real content.\nLine two of real content.\nLine three visible only after expanding.',
          },
        ]),
      ),
    );
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} />
      </Wrap>,
    );

    await waitFor(() =>
      expect(screen.getByTestId('doc-detail-chunks-list')).toBeInTheDocument(),
    );
    const text = screen.getByTestId('doc-detail-chunk-text');
    expect(text).toHaveClass('doc-chunk-text');
    expect(text).not.toHaveClass('expanded');

    await userEvent.click(screen.getByTestId('doc-detail-chunk-toggle-chunk-long'));
    expect(text).toHaveClass('expanded');
    expect(
      screen.getByText(/Line three visible only after expanding/),
    ).toBeInTheDocument();
  });

  it('auto-expands the cited chunk when opened from a retrieval source', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json([
          {
            chunk_id: 'chunk-first',
            order: 0,
            text: 'First chunk preview.',
          },
          {
            chunk_id: 'chunk-cited',
            order: 1,
            text: 'Cited chunk line one.\nCited chunk line two.\nCited chunk line three.',
          },
        ]),
      ),
    );
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          initialExpandedChunkId="chunk-cited"
          onClose={() => {}}
        />
      </Wrap>,
    );

    await waitFor(() =>
      expect(screen.getByTestId('doc-detail-chunk-chunk-cited')).toBeInTheDocument(),
    );
    const citedChunk = screen.getByTestId('doc-detail-chunk-chunk-cited');
    expect(within(citedChunk).getByTestId('doc-detail-chunk-text')).toHaveClass(
      'expanded',
    );
    expect(within(citedChunk).getByTestId('doc-detail-chunk-toggle-chunk-cited')).toHaveAttribute(
      'aria-expanded',
      'true',
    );
  });

  it('Lineage tab shows the uploader and the tags', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('doc-detail-tab-lineage'));
    const panel = screen.getByTestId('doc-detail-lineage');
    expect(panel.textContent).toContain('claire.benoit');
    expect(panel.textContent).toContain('rman');
  });

  it('Lineage tab shows the document hash when metadata carries one', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc({
            metadata: {
              uploader: 'claire.benoit',
              classification: 'internal',
              sha1: 'ABCDEF012345',
            },
          })}
          onClose={() => {}}
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('doc-detail-tab-lineage'));
    expect(screen.getByTestId('doc-detail-hash')).toHaveTextContent(
      'SHA1: abcdef012345',
    );
  });

  it('Audit tab fetches scoped to the doc id', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('doc-detail-tab-audit'));
    await waitFor(() =>
      expect(screen.getByTestId('doc-detail-audit-list')).toBeInTheDocument(),
    );
    const url = new URL(auditRequests[0]);
    expect(url.searchParams.get('resource.id')).toBe('d-test-1');
    expect(url.searchParams.get('limit')).toBe('200');
    expect(screen.getByText(/claire\.benoit/)).toHaveTextContent('3h ago');
    expect(screen.queryByText(/99d ago/)).toBeNull();
  });
});

describe('DocDetailPanel — chunks content', () => {
  it('renders full chunk text even when metadata carries a restricted classification', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc({
            metadata: { uploader: 'x', classification: 'restricted' },
          })}
          onClose={() => {}}
        />
      </Wrap>,
    );
    await waitFor(() =>
      expect(screen.getByTestId('doc-detail-chunks-list')).toBeInTheDocument(),
    );
    expect(screen.getByText('Chunk 1 text content for the document.')).toBeInTheDocument();
    expect(screen.queryByTestId('doc-detail-chunks-redacted')).toBeNull();
  });
});

describe('DocDetailPanel — footer actions', () => {
  it('Delete button requires confirmation, then invokes onDelete with the doc', async () => {
    const onDelete = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} onDelete={onDelete} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('doc-detail-delete'));
    expect(onDelete).not.toHaveBeenCalled();
    await userEvent.click(screen.getByTestId('doc-detail-delete'));
    expect(onDelete).toHaveBeenCalled();
    expect(onDelete.mock.calls[0][0].doc_id).toBe('d-test-1');
  });

  it('Retag button invokes onRetag with the doc', async () => {
    const onRetag = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={() => {}} onRetag={onRetag} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('doc-detail-retag'));
    expect(onRetag).toHaveBeenCalled();
  });

  it('Re-process button invokes onReprocess with the doc', async () => {
    const onReprocess = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          onClose={() => {}}
          onReprocess={onReprocess}
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('doc-detail-reprocess'));
    expect(onReprocess).toHaveBeenCalled();
  });

  it('Escape calls onClose', () => {
    const onClose = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={makeDoc()} onClose={onClose} />
      </Wrap>,
    );
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(onClose).toHaveBeenCalled();
  });

  describe('TR-ING-01 — failed doc error_msg surface', () => {
    async function openLineage() {
      // Tabs are: Chunks (default), Lineage, Audit. Click into Lineage
      // where the Status/Chunks/Error fields live. The tab is keyed by
      // data-testid (no ARIA role="tab"), see DocDetailPanel.tsx:132.
      await userEvent.click(screen.getByTestId('doc-detail-tab-lineage'));
    }

    it('renders an Error section on the Lineage tab when FAILED + error_msg', async () => {
      const Wrap = wrap(new QueryClient());
      render(
        <Wrap>
          <DocDetailPanel
            doc={makeDoc({
              status: 'FAILED',
              chunks_count: 327,
              error_msg: 'LLM extractor: invalid JSON on chunk 14',
            })}
            onClose={() => {}}
          />
        </Wrap>,
      );
      await openLineage();
      const errDd = screen.getByTestId('doc-detail-error-msg');
      expect(errDd.textContent).toMatch(/indexing failed/i);
      expect(errDd.textContent).toContain(
        'LLM extractor: invalid JSON on chunk 14',
      );
    });

    it('explains an excluded logo classification in the error section', async () => {
      const Wrap = wrap(new QueryClient());
      render(
        <Wrap>
          <DocDetailPanel
            doc={makeDoc({
              status: 'FAILED',
              error_msg: "image-dropped: classification 'Logo'",
            })}
            onClose={() => {}}
          />
        </Wrap>,
      );
      await openLineage();
      const errDd = screen.getByTestId('doc-detail-error-msg');
      expect(errDd).toHaveTextContent('classified as “logo”');
      expect(errDd).toHaveTextContent('excluded by the active Vision settings');
      expect(errDd).not.toHaveTextContent('image-dropped');
    });

    it('labels chunks "(created before failure)" when FAILED with chunks > 0', async () => {
      const Wrap = wrap(new QueryClient());
      render(
        <Wrap>
          <DocDetailPanel
            doc={makeDoc({
              status: 'FAILED',
              chunks_count: 327,
              error_msg: 'pipeline aborted',
            })}
            onClose={() => {}}
          />
        </Wrap>,
      );
      await openLineage();
      // The Chunks <dd> shows the number AND the partial-state hint.
      const lineage = screen.getByTestId('doc-detail-lineage');
      expect(lineage.textContent).toContain('327');
      expect(lineage.textContent).toContain('(created before failure)');
    });

    it('omits the Error section when status is FAILED but error_msg is null', async () => {
      const Wrap = wrap(new QueryClient());
      render(
        <Wrap>
          <DocDetailPanel
            doc={makeDoc({
              status: 'FAILED',
              chunks_count: 0,
              error_msg: null,
            })}
            onClose={() => {}}
          />
        </Wrap>,
      );
      await openLineage();
      expect(screen.queryByTestId('doc-detail-error-msg')).toBeNull();
    });

    it('omits the Error section on a PROCESSED doc even if error_msg slipped in', async () => {
      // Defensive: status takes priority — a processed doc must not
      // surface a stale error_msg from a previous failed attempt.
      const Wrap = wrap(new QueryClient());
      render(
        <Wrap>
          <DocDetailPanel
            doc={makeDoc({
              status: 'PROCESSED',
              chunks_count: 42,
              error_msg: 'stale message from a previous retry',
            })}
            onClose={() => {}}
          />
        </Wrap>,
      );
      await openLineage();
      expect(screen.queryByTestId('doc-detail-error-msg')).toBeNull();
    });
  });
});

describe('policy-rejected docs (OVH audit 2026-07-28 — logo-starbucks case)', () => {
  const rejectedDoc = () =>
    makeDoc({
      status: 'FAILED',
      chunks_count: 0,
      content_summary: 'Image ingestion refused',
      error_msg:
        'vision-prefilter: image rejected before vision analysis; ' +
        'OCR detected 0 text characters, below configured minimum 20',
    });

  it('shows the verdict guidance and disables Re-process', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel doc={rejectedDoc()} onClose={() => {}} />
      </Wrap>,
    );
    expect(screen.getByTestId('doc-detail-reprocess')).toBeDisabled();
    await userEvent.click(screen.getByTestId('doc-detail-tab-lineage'));
    const verdict = screen.getByTestId('doc-detail-policy-rejection');
    expect(verdict.textContent).toMatch(/will not change the verdict/i);
    // The raw reason stays visible, un-mangled by ingestionFailureMessage.
    expect(screen.getByTestId('doc-detail-error-msg').textContent).toContain(
      'vision-prefilter',
    );
  });

  it('keeps a transient vision failure fully retryable', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc({
            status: 'FAILED',
            chunks_count: 0,
            content_summary: 'Image ingestion refused',
            error_msg: 'vision-llm-error: APIConnectionError: endpoint down',
          })}
          onClose={() => {}}
        />
      </Wrap>,
    );
    expect(screen.getByTestId('doc-detail-reprocess')).toBeEnabled();
    await userEvent.click(screen.getByTestId('doc-detail-tab-lineage'));
    expect(screen.queryByTestId('doc-detail-policy-rejection')).toBeNull();
  });
});

describe('paragraph-anchor highlight (PARAGRAPH-CITATION phase A)', () => {
  const CHUNK_TEXT = 'Chunk 1 text content for the document.';

  it('highlights the anchored range inside the drilled-down chunk only', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          initialExpandedChunkId="d-test-1_c0"
          initialAnchor={{ start: 8, end: 12 }}
          onClose={() => {}}
        />
      </Wrap>,
    );

    await waitFor(() =>
      expect(screen.getByTestId('chunk-anchor-highlight')).toBeInTheDocument(),
    );
    const marks = screen.getAllByTestId('chunk-anchor-highlight');
    expect(marks).toHaveLength(1);
    // Offsets are verifiable against the loaded text: slice(8, 12).
    expect(marks[0].textContent).toBe(CHUNK_TEXT.slice(8, 12));
    const anchoredChunk = screen.getByTestId('doc-detail-chunk-d-test-1_c0');
    expect(within(anchoredChunk).getByTestId('chunk-anchor-highlight')).toBe(
      marks[0],
    );
    // The full text is still rendered around the mark.
    expect(
      within(anchoredChunk).getByTestId('doc-detail-chunk-text').textContent,
    ).toBe(CHUNK_TEXT);
  });

  it('degrades to plain text when the anchor does not fit the loaded text', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          initialExpandedChunkId="d-test-1_c0"
          initialAnchor={{ start: 8, end: 9999 }}
          onClose={() => {}}
        />
      </Wrap>,
    );

    await waitFor(() =>
      expect(
        screen.getByTestId('doc-detail-chunk-d-test-1_c0'),
      ).toBeInTheDocument(),
    );
    // Stale offsets (re-indexed chunk, hand-edited URL) must never
    // mis-slice the render: no mark, text intact.
    expect(screen.queryByTestId('chunk-anchor-highlight')).toBeNull();
    const chunk = screen.getByTestId('doc-detail-chunk-d-test-1_c0');
    expect(
      within(chunk).getByTestId('doc-detail-chunk-text').textContent,
    ).toBe(CHUNK_TEXT);
  });

  it('slices on code points — an astral char before the paragraph must not shift the range', async () => {
    // Backend offsets are Unicode code points (Python indices). '😀' is one
    // code point but two UTF-16 units: a String.slice implementation would
    // start the mark one unit late and drop the final period (PR #418
    // review, finding 2).
    const emojiText =
      '😀 intro\n\nThe approval process requires two signatures.';
    const paragraph = 'The approval process requires two signatures.';
    const start = 9; // code points: '😀 intro' (7) + '\n\n' (2)
    const end = start + paragraph.length; // 54 == Array.from(emojiText).length
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json([
          { chunk_id: 'd-test-1_c0', order: 0, text: emojiText },
        ]),
      ),
    );
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          initialExpandedChunkId="d-test-1_c0"
          initialAnchor={{ start, end }}
          onClose={() => {}}
        />
      </Wrap>,
    );

    await waitFor(() =>
      expect(screen.getByTestId('chunk-anchor-highlight')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('chunk-anchor-highlight').textContent).toBe(
      paragraph,
    );
    const chunk = screen.getByTestId('doc-detail-chunk-d-test-1_c0');
    expect(
      within(chunk).getByTestId('doc-detail-chunk-text').textContent,
    ).toBe(emojiText);
  });

  it('renders no highlight when no anchor is provided', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <DocDetailPanel
          doc={makeDoc()}
          initialExpandedChunkId="d-test-1_c0"
          onClose={() => {}}
        />
      </Wrap>,
    );
    await waitFor(() =>
      expect(
        screen.getByTestId('doc-detail-chunk-d-test-1_c0'),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('chunk-anchor-highlight')).toBeNull();
  });
});
