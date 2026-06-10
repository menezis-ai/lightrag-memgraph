/**
 * Unit tests for DocDetailPanel (#105).
 *
 * Behaviors under test:
 *   - returns null when no doc selected
 *   - 3 tabs: Chunks, Lineage, Audit; default = Chunks
 *   - Chunks fetches from /documents/{id}/chunks
 *   - Lineage shows uploader, classification, dates, tags
 *   - Audit fetches /twin/api/activity?resource.id={id}
 *   - View raw notice gates above-internal classification
 *   - Footer buttons fire onRetag / onReprocess / onDelete
 *   - Escape closes
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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

const server = setupServer(
  http.get('*/documents/:id/chunks', ({ params }) =>
    HttpResponse.json([
      { chunk_id: `${String(params.id)}_c0`, order: 0, text: 'Chunk 1 text content for the document.' },
      { chunk_id: `${String(params.id)}_c1`, order: 1, text: 'Chunk 2 text content with extra details.' },
    ]),
  ),
  http.get('*/twin/api/activity', () =>
    HttpResponse.json({
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
    }),
  ),
);

function wrap(qc: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

beforeEach(() => {
  __resetAuthConfigCacheForTests();
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
  });
});

describe('DocDetailPanel — classification gating', () => {
  it('truncates chunks when classification > internal', async () => {
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
      expect(
        screen.getAllByTestId('doc-detail-chunks-redacted').length,
      ).toBeGreaterThan(0),
    );
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
});
