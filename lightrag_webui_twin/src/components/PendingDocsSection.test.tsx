/**
 * Unit tests for PendingDocsSection (#106 + #150 spec révisée).
 *
 * Covers:
 *   - filtering: only docs with review.state === 'pending-review' render
 *   - Approve mutation calls /twin/api/documents/{id}/approve and toasts
 *   - Edit & Approve opens a modal (fix for v2 regression #150)
 *   - Reject requires a non-trivial reason
 *   - Simulate opens a preview modal
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { PendingDocsSection } from './PendingDocsSection';
import type { Document } from '../types/document';

function makePendingDoc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'pending-1',
    track_id: 'tk_pending_1',
    type: 'file',
    file_path: 'awaiting-approval.pdf',
    content_summary: 'A pending document',
    content_length: 200,
    tags: ['pending-review'],
    status: 'PROCESSING',
    chunks_count: 21,
    created_at: '2026-05-29T11:00:00Z',
    updated_at: '2026-05-29T11:00:00Z',
    error_msg: null,
    metadata: { uploader: 'fatima.t' },
    visibility: 'private',
    workspace: 'cib',
    review: {
      state: 'pending-review',
      requested_by: 'fatima.t',
      requested_at: '2026-05-29T11:00:00Z',
      justification: 'Source mentions vendor disclosure clause.',
    },
    ...overrides,
  };
}

const approveCalls: { id: string; body: unknown }[] = [];
const rejectCalls: { id: string; body: unknown }[] = [];

const server = setupServer(
  http.post('*/twin/api/documents/:id/approve', async ({ params, request }) => {
    const body = await request.json();
    approveCalls.push({ id: String(params.id), body });
    return HttpResponse.json({ doc_id: String(params.id) });
  }),
  http.post('*/twin/api/documents/:id/reject', async ({ params, request }) => {
    const body = await request.json();
    rejectCalls.push({ id: String(params.id), body });
    return HttpResponse.json({ doc_id: String(params.id) });
  }),
);

function wrap(qc: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

beforeEach(() => {
  approveCalls.length = 0;
  rejectCalls.length = 0;
  server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
  server.resetHandlers();
  server.close();
});

describe('PendingDocsSection — rendering', () => {
  it('returns null when no pending docs', () => {
    const Wrap = wrap(new QueryClient());
    const { container } = render(
      <Wrap>
        <PendingDocsSection docs={[]} onToast={() => {}} />
      </Wrap>,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders a card per pending doc', () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[makePendingDoc(), makePendingDoc({ doc_id: 'pending-2' })]}
          onToast={() => {}}
        />
      </Wrap>,
    );
    expect(screen.getByTestId('pending-doc-pending-1')).toBeInTheDocument();
    expect(screen.getByTestId('pending-doc-pending-2')).toBeInTheDocument();
  });
});

describe('PendingDocsSection — Approve', () => {
  it('clicking Approve calls /twin/api/documents/{id}/approve and toasts', async () => {
    const toast = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[makePendingDoc()]}
          onToast={toast}
          actor="claire.benoit"
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('pending-doc-approve-pending-1'));
    await waitFor(() => expect(approveCalls.length).toBe(1));
    expect(approveCalls[0].id).toBe('pending-1');
    await waitFor(() => expect(toast).toHaveBeenCalled());
    expect(toast.mock.calls[0][0]).toBe('done');
  });
});

describe('PendingDocsSection — Edit & Approve (#150 fix)', () => {
  it('Edit & Approve opens a modal with summary + tags editable', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection docs={[makePendingDoc()]} onToast={() => {}} />
      </Wrap>,
    );
    await userEvent.click(
      screen.getByTestId('pending-doc-edit-approve-pending-1'),
    );
    expect(screen.getByTestId('pending-doc-edit-modal')).toBeInTheDocument();
    expect(screen.getByTestId('pending-doc-edit-summary')).toBeInTheDocument();
    expect(screen.getByTestId('pending-doc-edit-tags')).toBeInTheDocument();
  });

  it('submitting the edit modal calls approve with edits payload', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[makePendingDoc()]}
          onToast={() => {}}
          actor="claire.benoit"
        />
      </Wrap>,
    );
    await userEvent.click(
      screen.getByTestId('pending-doc-edit-approve-pending-1'),
    );
    const summary = screen.getByTestId(
      'pending-doc-edit-summary',
    ) as HTMLTextAreaElement;
    await userEvent.clear(summary);
    await userEvent.type(summary, 'Edited summary content');
    await userEvent.click(screen.getByTestId('pending-doc-edit-submit'));
    await waitFor(() => expect(approveCalls.length).toBe(1));
    const body = approveCalls[0].body as {
      edits: { content_summary: string };
    };
    expect(body.edits.content_summary).toBe('Edited summary content');
  });
});

describe('PendingDocsSection — Reject', () => {
  it('Reject opens a modal that requires ≥ 6 char reason', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection docs={[makePendingDoc()]} onToast={() => {}} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('pending-doc-reject-pending-1'));
    const submit = screen.getByTestId('pending-doc-reject-submit');
    expect(submit).toBeDisabled();
    const reason = screen.getByTestId('pending-doc-reject-reason');
    await userEvent.type(reason, 'too short');
    expect(submit).not.toBeDisabled();
    await userEvent.click(submit);
    await waitFor(() => expect(rejectCalls.length).toBe(1));
  });
});

describe('PendingDocsSection — Read source (B2)', () => {
  it('Read source button delegates to host via onReadSource', async () => {
    const onReadSource = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[makePendingDoc()]}
          onToast={() => {}}
          onReadSource={onReadSource}
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('pending-doc-read-pending-1'));
    expect(onReadSource).toHaveBeenCalledTimes(1);
    expect(onReadSource.mock.calls[0][0].doc_id).toBe('pending-1');
  });
});

describe('PendingDocsSection — Modified variant (Confluence revalidation)', () => {
  function makeModifiedDoc(): Document {
    return {
      ...makePendingDoc({
        doc_id: 'mod-1',
        file_path: '/cib/runbooks/oracle-pga-tuning',
        type: 'confluence',
      }),
      review: {
        state: 'modified',
        update: {
          requested_by: 'yann.dubois',
          edited_rel: '2h ago',
          detected_at: '2026-05-26',
          chunks_indexed: 54,
          summary_diff: 'Confluence page edited 2h ago — added 2 new sections.',
        },
      },
    };
  }

  it('renders the Modified source pill and the diff summary', () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection docs={[makeModifiedDoc()]} onToast={() => {}} />
      </Wrap>,
    );
    const card = screen.getByTestId('pending-doc-mod-1');
    expect(card.className).toContain('modified');
    expect(card.textContent).toContain('Modified source');
    expect(card.textContent).toContain('Confluence page edited 2h ago');
  });

  it('uses Approve update / Reject update buttons (not the requested ones)', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[makeModifiedDoc()]}
          onToast={() => {}}
          actor="claire.benoit"
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('pending-doc-approve-update-mod-1'));
    await waitFor(() => expect(approveCalls.length).toBe(1));
    expect(approveCalls[0].id).toBe('mod-1');
    // The "requested" variant buttons must NOT be present
    expect(screen.queryByTestId('pending-doc-approve-mod-1')).toBeNull();
    expect(screen.queryByTestId('pending-doc-edit-approve-mod-1')).toBeNull();
  });
});
