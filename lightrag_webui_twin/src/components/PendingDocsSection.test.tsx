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
    folder: 'default',
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

async function openPendingDocsSection() {
  await userEvent.click(
    screen.getByRole('button', {
      name: /To be validated by your reviewer/i,
    }),
  );
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

  it('is collapsed by default and renders cards when opened', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[makePendingDoc(), makePendingDoc({ doc_id: 'pending-2' })]}
          onToast={() => {}}
        />
      </Wrap>,
    );
    expect(screen.queryByTestId('pending-doc-pending-1')).toBeNull();
    expect(
      screen.getByRole('button', {
        name: /To be validated by your reviewer/i,
      }),
    ).toHaveAttribute('aria-expanded', 'false');
    await openPendingDocsSection();
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
    await openPendingDocsSection();
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
    await openPendingDocsSection();
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
    await openPendingDocsSection();
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
    await openPendingDocsSection();
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
    await openPendingDocsSection();
    await userEvent.click(screen.getByTestId('pending-doc-read-pending-1'));
    expect(onReadSource).toHaveBeenCalledTimes(1);
    expect(onReadSource.mock.calls[0][0].doc_id).toBe('pending-1');
  });
});

describe('PendingDocsSection — procedure bundles (third variant)', () => {
  const PROC_SUMMARIES = [
    {
      id: 'pb-1',
      file_name: 'failover-procedure.pdf',
      state: 'pending',
      reason: 'procedure detected: schematic-heavy layout',
      source: 'detected',
      schematics_total: 2,
      schematics_described: 2,
      classification: {
        class_id: 'C2',
        class_name: 'C2 Confidentiel',
        reason: null,
      },
      operator_classification: 'C2',
      created_at: '2026-07-18T09:00:00Z',
      updated_at: null,
    },
    {
      id: 'pb-2',
      file_name: 'segmentation-procedure.pdf',
      state: 'failed',
      reason: 'vision pass failed on page 2',
      source: 'forced',
      schematics_total: 2,
      schematics_described: 1,
      classification: null,
      operator_classification: null,
      created_at: '2026-07-17T15:00:00Z',
      updated_at: null,
    },
  ];

  const procRejectCalls: { id: string; body: unknown }[] = [];

  function useProcedureHandlers() {
    procRejectCalls.length = 0;
    server.use(
      http.get('*/twin/api/procedures', () =>
        HttpResponse.json(PROC_SUMMARIES),
      ),
      http.post(
        '*/twin/api/procedures/:id/reject',
        async ({ params, request }) => {
          const body = await request.json().catch(() => null);
          procRejectCalls.push({ id: String(params.id), body });
          return HttpResponse.json({
            ...PROC_SUMMARIES[0],
            id: String(params.id),
            state: 'rejected',
          });
        },
      ),
    );
  }

  function queryClientNoRetry() {
    return new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  }

  it('renders the section from bundles alone (no pending docs) with state pills + schematic counts', async () => {
    useProcedureHandlers();
    const Wrap = wrap(queryClientNoRetry());
    render(
      <Wrap>
        <PendingDocsSection docs={[]} onToast={() => {}} defaultOpen />
      </Wrap>,
    );

    const pendingCard = await screen.findByTestId('pending-proc-pb-1');
    expect(pendingCard.textContent).toContain('failover-procedure.pdf');
    expect(pendingCard.textContent).toContain('procedure detected');
    expect(pendingCard.textContent).toContain('2/2 schematics described');
    expect(screen.getByTestId('pending-proc-state-pb-1').textContent).toContain(
      'Procedure review',
    );
    // MIP pill from the summary's partial classification payload.
    expect(pendingCard.querySelector('[data-testid="class-pill-pb-1"]')).not.toBeNull();

    const failedCard = screen.getByTestId('pending-proc-pb-2');
    expect(screen.getByTestId('pending-proc-state-pb-2').textContent).toContain(
      'Procedure failed',
    );
    expect(failedCard.textContent).toContain('1/2 schematics described');
    // No classification on pb-2 → the pill stays silent.
    expect(failedCard.querySelector('[data-testid="class-pill-pb-2"]')).toBeNull();
  });

  it('quick Reject posts to /procedures/{id}/reject and toasts', async () => {
    useProcedureHandlers();
    const toast = vi.fn();
    const Wrap = wrap(queryClientNoRetry());
    render(
      <Wrap>
        <PendingDocsSection docs={[]} onToast={toast} defaultOpen />
      </Wrap>,
    );

    await userEvent.click(await screen.findByTestId('pending-proc-reject-pb-1'));
    await waitFor(() => expect(procRejectCalls.length).toBe(1));
    expect(procRejectCalls[0].id).toBe('pb-1');
    await waitFor(() => expect(toast).toHaveBeenCalled());
    expect(toast.mock.calls[0][0]).toBe('done');
    expect(toast.mock.calls[0][1]).toBe('Procedure rejected');
  });

  it('hides Review/Reject when canReviewProcedures is false (cards stay visible)', async () => {
    useProcedureHandlers();
    const Wrap = wrap(queryClientNoRetry());
    render(
      <Wrap>
        <PendingDocsSection
          docs={[]}
          onToast={() => {}}
          defaultOpen
          canReviewProcedures={false}
        />
      </Wrap>,
    );

    await screen.findByTestId('pending-proc-pb-1');
    expect(screen.queryByTestId('pending-proc-review-pb-1')).toBeNull();
    expect(screen.queryByTestId('pending-proc-reject-pb-1')).toBeNull();
  });

  it('Review opens the procedure review modal (detail fetch)', async () => {
    useProcedureHandlers();
    server.use(
      http.get('*/twin/api/procedures/:id', ({ params }) =>
        HttpResponse.json({
          ...PROC_SUMMARIES[0],
          id: String(params.id),
          original_path: '/inputs/failover-procedure.pdf',
          track_id: null,
          folder: 'default',
          content_hash: null,
          full_text: 'text',
          duplicate_requests: [],
          schematics: [],
        }),
      ),
    );
    const Wrap = wrap(queryClientNoRetry());
    render(
      <Wrap>
        <PendingDocsSection docs={[]} onToast={() => {}} defaultOpen />
      </Wrap>,
    );

    await userEvent.click(await screen.findByTestId('pending-proc-review-pb-1'));
    expect(await screen.findByTestId('procedure-review-modal')).toBeInTheDocument();
  });

  it('keeps rejected bundles visible and recoverable (reject → retry path)', async () => {
    // Stateful test handlers: reject mutates the list the next refetch sees —
    // the rejected bundle must NOT disappear (the review modal is the only
    // surface offering retry/reroute recovery).
    const bundles = PROC_SUMMARIES.map((b) => ({ ...b }));
    server.use(
      http.get('*/twin/api/procedures', () => HttpResponse.json(bundles)),
      http.post('*/twin/api/procedures/:id/reject', ({ params }) => {
        const bundle = bundles.find((b) => b.id === String(params.id));
        if (bundle) bundle.state = 'rejected';
        return HttpResponse.json({ ...bundle, state: 'rejected' });
      }),
      http.get('*/twin/api/procedures/:id', ({ params }) => {
        const bundle = bundles.find((b) => b.id === String(params.id));
        return HttpResponse.json({
          ...bundle,
          original_path: '/inputs/x.pdf',
          track_id: null,
          folder: 'default',
          content_hash: null,
          full_text: 'text',
          duplicate_requests: [],
          schematics: [],
        });
      }),
    );
    const Wrap = wrap(queryClientNoRetry());
    render(
      <Wrap>
        <PendingDocsSection docs={[]} onToast={() => {}} defaultOpen />
      </Wrap>,
    );

    await userEvent.click(await screen.findByTestId('pending-proc-reject-pb-1'));

    // Still visible, now with the rejected pill and NO quick-reject.
    const pill = await screen.findByTestId('pending-proc-state-pb-1');
    await waitFor(() =>
      expect(pill.textContent).toContain('Procedure rejected'),
    );
    expect(
      screen.queryByTestId('pending-proc-reject-pb-1'),
    ).not.toBeInTheDocument();

    // Recovery path stays reachable: Review opens the modal with Retry.
    await userEvent.click(screen.getByTestId('pending-proc-review-pb-1'));
    expect(await screen.findByTestId('procedure-review-modal')).toBeInTheDocument();
    expect(
      await screen.findByTestId('procedure-review-retry'),
    ).toBeInTheDocument();
  });

  it('surfaces a store error with a retry instead of an empty queue', async () => {
    // A degraded store answers 503 precisely so parked work is never
    // presented as an empty list.
    let failures = 1;
    server.use(
      http.get('*/twin/api/procedures', () => {
        if (failures > 0) {
          failures -= 1;
          return HttpResponse.json(
            { detail: 'procedure store degraded: … store/recover' },
            { status: 503 },
          );
        }
        return HttpResponse.json(PROC_SUMMARIES);
      }),
    );
    const Wrap = wrap(queryClientNoRetry());
    render(
      <Wrap>
        <PendingDocsSection docs={[]} onToast={() => {}} defaultOpen />
      </Wrap>,
    );

    const errorCard = await screen.findByTestId('pending-procedures-error');
    expect(errorCard.textContent).toContain('NOT an empty queue');

    await userEvent.click(screen.getByTestId('pending-procedures-retry'));
    expect(await screen.findByTestId('pending-proc-pb-1')).toBeInTheDocument();
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

  it('renders the Modified source pill and the diff summary', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <PendingDocsSection docs={[makeModifiedDoc()]} onToast={() => {}} />
      </Wrap>,
    );
    await openPendingDocsSection();
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
    await openPendingDocsSection();
    await userEvent.click(screen.getByTestId('pending-doc-approve-update-mod-1'));
    await waitFor(() => expect(approveCalls.length).toBe(1));
    expect(approveCalls[0].id).toBe('mod-1');
    // The "requested" variant buttons must NOT be present
    expect(screen.queryByTestId('pending-doc-approve-mod-1')).toBeNull();
    expect(screen.queryByTestId('pending-doc-edit-approve-mod-1')).toBeNull();
  });
});
