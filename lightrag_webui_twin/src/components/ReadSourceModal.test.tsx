/**
 * Unit tests for ReadSourceModal post-audit-C6 rewrite (Bucket B2).
 *
 * The modal now consumes ``/documents/{id}/chunks`` via TanStack
 * Query instead of rendering ``doc.extracted_text``. Tests use MSW
 * to stub the backend so the loading / error / empty / populated
 * branches are exercised end-to-end through the query layer.
 */

import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

import { ReadSourceModal } from './ReadSourceModal';
import type { Document } from '../types/document';

function makeDoc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'rs-1',
    track_id: null,
    type: 'file',
    file_path: 'sample.pdf',
    content_summary: 'A sample document',
    content_length: 2048,
    status: 'PROCESSED',
    chunks_count: 3,
    created_at: '2026-05-29T14:00:00Z',
    updated_at: '2026-05-29T14:00:00Z',
    error_msg: null,
    metadata: {},
    tags: ['rman'],
    folder: 'default',
    visibility: 'private',
    ...overrides,
  };
}

const server = setupServer();
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function renderWith(node: React.ReactElement) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={qc}>{node}</QueryClientProvider>,
  );
}

describe('ReadSourceModal — visibility', () => {
  it('returns null when no doc is selected', () => {
    const { container } = renderWith(
      <ReadSourceModal doc={null} onClose={() => {}} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders the file path, chunks count and extracted KB on the sub-line', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);
    expect(screen.getByTestId('read-source-modal')).toBeInTheDocument();
    expect(screen.getByText('sample.pdf')).toBeInTheDocument();
    expect(screen.getByText('3 chunks indexed')).toBeInTheDocument();
    expect(screen.getByText('2.0 KB extracted')).toBeInTheDocument();
  });
});

describe('ReadSourceModal — chunks loading + states (audit C6)', () => {
  it('shows the loading cue while the chunks query is in flight', () => {
    server.use(
      http.get('*/documents/:id/chunks', async () => {
        // Block the response so we can assert the loading state.
        await new Promise(() => undefined);
        return HttpResponse.json([]);
      }),
    );
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);
    expect(screen.getByTestId('rs-chunks-loading')).toBeInTheDocument();
  });

  it('shows an error banner when the chunks query fails', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json({ detail: 'boom' }, { status: 503 }),
      ),
    );
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);
    await waitFor(() =>
      expect(screen.getByTestId('rs-chunks-error')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('rs-chunks-error').textContent).toMatch(
      /Could not load chunks/i,
    );
  });

  it('shows the empty-state cue when the document has no indexed chunks', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);
    await waitFor(() =>
      expect(screen.getByTestId('rs-chunks-empty')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('rs-chunks-empty').textContent).toMatch(
      /No chunks indexed/i,
    );
  });

  it('renders chunks as separate blocks with chunk_id + order visible', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json([
          {
            chunk_id: 'chunk-aa',
            order: 0,
            text: 'First chunk content.',
          },
          {
            chunk_id: 'chunk-bb',
            order: 1,
            text: 'Second chunk content.',
          },
        ]),
      ),
    );
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);

    await waitFor(() =>
      expect(screen.getByTestId('rs-chunks-list')).toBeInTheDocument(),
    );

    // Each chunk is its own inspectable block — no merged text blob.
    expect(screen.getByTestId('rs-chunk-chunk-aa')).toBeInTheDocument();
    expect(screen.getByTestId('rs-chunk-chunk-bb')).toBeInTheDocument();

    // chunk_id + order visible per block.
    expect(screen.getByText('chunk-aa')).toBeInTheDocument();
    expect(screen.getByText('#0')).toBeInTheDocument();
    expect(screen.getByText('chunk-bb')).toBeInTheDocument();
    expect(screen.getByText('#1')).toBeInTheDocument();

    // Chunk text appears verbatim inside its own <pre> — not joined.
    expect(screen.getByText('First chunk content.')).toBeInTheDocument();
    expect(screen.getByText('Second chunk content.')).toBeInTheDocument();
  });

  it('renders chunk text without redaction badges', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json([
          {
            chunk_id: 'chunk-clear',
            order: 0,
            text: 'Open clear chunk.',
          },
          {
            chunk_id: 'chunk-conf',
            order: 1,
            text: 'Previously flagged content now shown to the admin user.',
            redacted: true,
          },
        ]),
      ),
    );
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);

    await waitFor(() =>
      expect(screen.getByTestId('rs-chunks-list')).toBeInTheDocument(),
    );

    expect(screen.getByText('Open clear chunk.')).toBeInTheDocument();
    expect(
      screen.getByText('Previously flagged content now shown to the admin user.'),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId('rs-chunk-redacted-chunk-clear'),
    ).toBeNull();
    expect(
      screen.queryByTestId('rs-chunk-redacted-chunk-conf'),
    ).toBeNull();
  });

  it('never renders the pre-C6 demo placeholder copy', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    renderWith(
      <ReadSourceModal
        doc={makeDoc({ extracted_text: undefined })}
        onClose={() => {}}
      />,
    );
    await waitFor(() =>
      expect(screen.getByTestId('rs-chunks-empty')).toBeInTheDocument(),
    );
    // The previous build leaked
    // "(Extracted text preview is not available for this source in the
    // demo build.)" to the operator on any doc without a fixture.
    expect(
      screen.queryByText(/Extracted text preview is not available/i),
    ).toBeNull();
    expect(screen.queryByText(/demo build/i)).toBeNull();
  });

  it('does not render doc.extracted_text even when it is set on the fixture', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () =>
        HttpResponse.json([
          {
            chunk_id: 'chunk-only',
            order: 0,
            text: 'Chunk-derived content.',
          },
        ]),
      ),
    );
    renderWith(
      <ReadSourceModal
        doc={makeDoc({
          extracted_text: 'Legacy extracted_text from a fixture',
        })}
        onClose={() => {}}
      />,
    );
    await waitFor(() =>
      expect(screen.getByText('Chunk-derived content.')).toBeInTheDocument(),
    );
    // The legacy field stays on the TS contract but is intentionally
    // ignored by the rewrite — the chunks endpoint is the only
    // truth source the modal trusts.
    expect(
      screen.queryByText(/Legacy extracted_text from a fixture/),
    ).toBeNull();
  });
});

describe('ReadSourceModal — status pills', () => {
  it('shows "awaiting reviewer sign-off" for pending-review', () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    renderWith(
      <ReadSourceModal
        doc={makeDoc({
          review: {
            state: 'pending-review',
            requested_by: 'x',
            requested_at: '2026-05-20',
            justification: 'because',
          },
        })}
        onClose={() => {}}
      />,
    );
    expect(screen.getByTestId('rs-pill-pending')).toBeInTheDocument();
  });

  it('shows "modified — awaiting re-validation" for modified', () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    renderWith(
      <ReadSourceModal
        doc={makeDoc({
          review: {
            state: 'modified',
            update: {
              requested_by: 'x',
              edited_rel: '1h ago',
              detected_at: '2026-05-26',
              chunks_indexed: 12,
              summary_diff: 'changed',
            },
          },
        })}
        onClose={() => {}}
      />,
    );
    expect(screen.getByTestId('rs-pill-modified')).toBeInTheDocument();
  });
});

describe('ReadSourceModal — interactions', () => {
  it('calls onClose when the close button is clicked', async () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    const onClose = vi.fn();
    const user = (await import('@testing-library/user-event')).default;
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={onClose} />);
    await user.setup().click(screen.getByLabelText('Close'));
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when Escape is pressed', () => {
    server.use(
      http.get('*/documents/:id/chunks', () => HttpResponse.json([])),
    );
    const onClose = vi.fn();
    renderWith(<ReadSourceModal doc={makeDoc()} onClose={onClose} />);
    document.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(onClose).toHaveBeenCalled();
  });
});
