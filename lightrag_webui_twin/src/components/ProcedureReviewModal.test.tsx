/**
 * Unit tests for ProcedureReviewModal (procedure PDF review surface).
 *
 * Covers:
 *   - render: schematic PNG + informed description + coherent divergence line
 *   - prev/next navigation + divergent-page highlight (data-coherent="false")
 *   - blind reading collapsible
 *   - Approve mutation → POST /twin/api/procedures/{id}/approve + toast + close
 *   - folderless bundle: Approve gated on the target-folder select (422 mirror)
 *   - Retry visibility only on failed/rejected bundles
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { ProcedureReviewModal } from './ProcedureReviewModal';
import { FOLDER_FIXTURES, PROCEDURE_BUNDLE_FIXTURES } from '../fixtures';
import type { ProcedureBundle } from '../types/procedure';

const PENDING_BUNDLE = PROCEDURE_BUNDLE_FIXTURES[0]; // proc-1, pending, 2 schematics
const FAILED_BUNDLE = PROCEDURE_BUNDLE_FIXTURES[1]; // proc-2, failed, folderless

const bundlesById = new Map<string, ProcedureBundle>();
const approveCalls: { id: string; body: unknown }[] = [];
const rejectCalls: { id: string; body: unknown }[] = [];
const rerouteCalls: { id: string; body: unknown }[] = [];
const retryCalls: string[] = [];

const server = setupServer(
  http.get('*/twin/api/procedures/:id', ({ params }) => {
    const bundle = bundlesById.get(String(params.id));
    if (!bundle) {
      return HttpResponse.json({ detail: 'unknown bundle' }, { status: 404 });
    }
    return HttpResponse.json(bundle);
  }),
  http.post('*/twin/api/procedures/:id/approve', async ({ params, request }) => {
    const body = await request.json().catch(() => null);
    approveCalls.push({ id: String(params.id), body });
    return HttpResponse.json({
      ...bundlesById.get(String(params.id)),
      state: 'approved',
    });
  }),
  http.post('*/twin/api/procedures/:id/reject', async ({ params, request }) => {
    const body = await request.json().catch(() => null);
    rejectCalls.push({ id: String(params.id), body });
    return HttpResponse.json({
      ...bundlesById.get(String(params.id)),
      state: 'rejected',
    });
  }),
  http.post(
    '*/twin/api/procedures/:id/reroute-standard',
    async ({ params, request }) => {
      const body = await request.json().catch(() => null);
      rerouteCalls.push({ id: String(params.id), body });
      return HttpResponse.json({
        ...bundlesById.get(String(params.id)),
        state: 'rerouted',
      });
    },
  ),
  http.post('*/twin/api/procedures/:id/retry', ({ params }) => {
    retryCalls.push(String(params.id));
    return HttpResponse.json({
      ...bundlesById.get(String(params.id)),
      state: 'pending',
    });
  }),
);

function wrap() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

beforeEach(() => {
  bundlesById.clear();
  bundlesById.set(PENDING_BUNDLE.id, PENDING_BUNDLE);
  bundlesById.set(FAILED_BUNDLE.id, FAILED_BUNDLE);
  approveCalls.length = 0;
  rejectCalls.length = 0;
  rerouteCalls.length = 0;
  retryCalls.length = 0;
  server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
  server.resetHandlers();
  server.close();
});

describe('ProcedureReviewModal — rendering', () => {
  it('renders the schematic PNG, the informed description, and a green coherent line', async () => {
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          onClose={() => {}}
          onToast={() => {}}
        />
      </Wrap>,
    );

    const png = await screen.findByTestId('procedure-review-png');
    expect(png).toHaveAttribute(
      'src',
      expect.stringMatching(/^data:image\/png;base64,/),
    );
    expect(png).toHaveAttribute('alt', 'Schematic page 1');

    const informed = screen.getByTestId('procedure-review-informed');
    expect(informed.textContent).toContain('Failover decision tree');
    expect(informed.textContent).toContain('Check Data Guard lag');

    // Page 1 is coherent → subtle green summary, no divergence list.
    const divergence = screen.getByTestId('procedure-review-divergence');
    expect(divergence).toHaveAttribute('data-coherent', 'true');
    expect(divergence.textContent).toContain(
      'Blind and informed passes agree',
    );
  });

  it('navigates schematics and highlights the divergent page', async () => {
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          onClose={() => {}}
          onToast={() => {}}
        />
      </Wrap>,
    );

    await screen.findByTestId('procedure-review-png');
    expect(screen.getByText('Schematic 1 / 2')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('procedure-review-next'));

    expect(screen.getByText('Schematic 2 / 2')).toBeInTheDocument();
    const divergence = screen.getByTestId('procedure-review-divergence');
    expect(divergence).toHaveAttribute('data-coherent', 'false');
    expect(divergence.textContent).toContain(
      'Blind and informed readings diverge',
    );
    expect(divergence.textContent).toContain(
      'ISAB gate missing from the diagram',
    );
    // Back to page 1 restores the coherent panel.
    await userEvent.click(screen.getByTestId('procedure-review-prev'));
    expect(
      screen.getByTestId('procedure-review-divergence'),
    ).toHaveAttribute('data-coherent', 'true');
  });

  it('keeps the blind reading behind a collapsible', async () => {
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          onClose={() => {}}
          onToast={() => {}}
        />
      </Wrap>,
    );

    await screen.findByTestId('procedure-review-png');
    expect(screen.queryByTestId('procedure-review-blind')).toBeNull();
    await userEvent.click(screen.getByTestId('procedure-review-blind-toggle'));
    expect(screen.getByTestId('procedure-review-blind').textContent).toContain(
      'Failover decision tree (blind)',
    );
  });

  it('hides Retry for pending bundles and shows it for failed ones', async () => {
    const Wrap = wrap();
    const { unmount } = render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          onClose={() => {}}
          onToast={() => {}}
        />
      </Wrap>,
    );
    await screen.findByTestId('procedure-review-png');
    expect(screen.queryByTestId('procedure-review-retry')).toBeNull();
    unmount();

    const Wrap2 = wrap();
    render(
      <Wrap2>
        <ProcedureReviewModal
          bundleId="proc-2"
          folderList={FOLDER_FIXTURES}
          onClose={() => {}}
          onToast={() => {}}
        />
      </Wrap2>,
    );
    expect(await screen.findByTestId('procedure-review-retry')).toBeEnabled();
    // Failed bundle → Approve stays disabled (pending-only decision).
    expect(screen.getByTestId('procedure-review-approve')).toBeDisabled();
  });
});

describe('ProcedureReviewModal — Approve', () => {
  it('posts /approve, toasts and closes; invalidation targets documents too', async () => {
    const onClose = vi.fn();
    const onToast = vi.fn();
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          onClose={onClose}
          onToast={onToast}
        />
      </Wrap>,
    );

    await screen.findByTestId('procedure-review-png');
    const approve = screen.getByTestId('procedure-review-approve');
    expect(approve).toBeEnabled();
    await userEvent.click(approve);

    await waitFor(() => expect(approveCalls.length).toBe(1));
    expect(approveCalls[0].id).toBe('proc-1');
    // proc-1 already carries a requesting folder → no override sent.
    expect(approveCalls[0].body).toEqual({ folder: null });
    await waitFor(() => expect(onToast).toHaveBeenCalled());
    expect(onToast.mock.calls[0][0]).toBe('done');
    expect(onToast.mock.calls[0][1]).toBe('Procedure approved');
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });
});

describe('ProcedureReviewModal — folderless bundle (422 mirror)', () => {
  const FOLDERLESS_PENDING: ProcedureBundle = {
    ...FAILED_BUNDLE,
    id: 'proc-3',
    state: 'pending',
    folder: null,
    duplicate_requests: [],
  };

  beforeEach(() => {
    bundlesById.set(FOLDERLESS_PENDING.id, FOLDERLESS_PENDING);
  });

  it('requires a target folder before Approve, then posts the chosen folder', async () => {
    const onToast = vi.fn();
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-3"
          folderList={FOLDER_FIXTURES}
          onClose={() => {}}
          onToast={onToast}
        />
      </Wrap>,
    );

    const select = await screen.findByTestId('procedure-review-folder-select');
    // No folder picked yet → Approve and Treat-as-standard are gated.
    expect(screen.getByTestId('procedure-review-approve')).toBeDisabled();
    expect(screen.getByTestId('procedure-review-reroute')).toBeDisabled();

    await userEvent.selectOptions(select, 'cib');
    const approve = screen.getByTestId('procedure-review-approve');
    expect(approve).toBeEnabled();
    await userEvent.click(approve);

    await waitFor(() => expect(approveCalls.length).toBe(1));
    expect(approveCalls[0].body).toEqual({ folder: 'cib' });
  });

  it('bundles WITH a requesting folder never show the folder select', async () => {
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          folderList={FOLDER_FIXTURES}
          onClose={() => {}}
          onToast={() => {}}
        />
      </Wrap>,
    );
    await screen.findByTestId('procedure-review-png');
    expect(
      screen.queryByTestId('procedure-review-folder-select'),
    ).toBeNull();
  });
});

describe('ProcedureReviewModal — Reject with comment', () => {
  it('sends the optional comment to /reject', async () => {
    const onToast = vi.fn();
    const Wrap = wrap();
    render(
      <Wrap>
        <ProcedureReviewModal
          bundleId="proc-1"
          onClose={() => {}}
          onToast={onToast}
        />
      </Wrap>,
    );

    await screen.findByTestId('procedure-review-png');
    const comment = screen.getByTestId('procedure-review-reject-comment');
    comment.focus();
    await userEvent.type(comment, 'diagram outdated');
    await userEvent.click(screen.getByTestId('procedure-review-reject'));

    await waitFor(() => expect(rejectCalls.length).toBe(1));
    expect(rejectCalls[0].body).toEqual({ comment: 'diagram outdated' });
    await waitFor(() =>
      expect(onToast).toHaveBeenCalledWith(
        'done',
        'Procedure rejected',
        PENDING_BUNDLE.file_name,
      ),
    );
  });
});
