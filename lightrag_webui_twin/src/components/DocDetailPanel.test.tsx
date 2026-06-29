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
          rel: '1h ago',
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
