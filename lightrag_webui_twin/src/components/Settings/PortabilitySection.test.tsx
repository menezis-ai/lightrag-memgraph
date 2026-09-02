import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { PortabilitySection } from './PortabilitySection';
import { handlers, resetDocumentsState } from '../../mocks/handlers';
import { PORTABILITY_REPORT_HASH } from '../../fixtures/portability';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(() => sessionStorage.clear());
afterEach(() => {
  server.resetHandlers();
  resetDocumentsState();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function renderSection() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <PortabilitySection />
    </QueryClientProvider>,
  );
}

describe('PortabilitySection', () => {
  it('runs dry-run → approval → apply → validation against stateful MSW', async () => {
    const user = userEvent.setup();
    renderSection();

    const file = new File(['canonical bundle'], 'staging.tar.gz', {
      type: 'application/gzip',
    });
    await user.upload(screen.getByTestId('portability-import-file'), file);
    fireEvent.change(screen.getByTestId('portability-folder-map'), {
      target: { value: '{"staging":"production"}' },
    });
    await user.click(screen.getByTestId('portability-import-start'));

    const report = await screen.findByTestId('portability-report', {}, { timeout: 3000 });
    expect(report).toHaveTextContent('ready for approval');
    expect(report).toHaveTextContent('all three probe cosines');
    expect(report).toHaveTextContent('C2');
    expect(report).toHaveTextContent('312');
    expect(report).toHaveTextContent('staging');
    expect(report).toHaveTextContent('production');

    await user.click(screen.getByTestId('portability-approve'));
    expect(await screen.findByTestId('portability-apply')).toBeEnabled();

    await user.click(screen.getByTestId('portability-apply'));
    const validate = await screen.findByTestId(
      'portability-validate',
      {},
      { timeout: 3000 },
    );
    await user.click(validate);

    expect(
      await screen.findByText(/Validation passed/, {}, { timeout: 3000 }),
    ).toBeInTheDocument();
  }, 10_000);

  it('keeps approval disabled when the dry-run has a blocker', async () => {
    const blockingJob = {
      id: 'imp_blocking',
      kind: 'import',
      workspace: 'base',
      status: 'awaiting-approval',
      created_at: '2026-08-26T13:30:00Z',
      updated_at: '2026-08-26T13:30:01Z',
      actor: 'operator.demo',
      options: {},
      result: null,
      report: {
        report_hash: PORTABILITY_REPORT_HASH,
        blocking: [
          {
            code: 'target_not_empty',
            message: 'target workspace contains portable state',
          },
        ],
        compat: [],
        classification: {
          source_max: 'C2',
          target_ceiling: 'C2',
          unknown_present: false,
        },
      },
      validation: null,
      error: null,
      download_available: false,
    };
    server.use(
      http.post('*/twin/api/admin/portability/imports', () =>
        HttpResponse.json(blockingJob, { status: 202 }),
      ),
      http.get('*/twin/api/admin/portability/imports/:id', () =>
        HttpResponse.json(blockingJob),
      ),
    );
    const user = userEvent.setup();
    renderSection();
    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'blocked.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByTestId('portability-import-start'));

    expect(await screen.findByText('target_not_empty')).toBeInTheDocument();
    expect(screen.getByTestId('portability-approve')).toBeDisabled();
  });

  it('resumes the persisted import job after a remount', async () => {
    const user = userEvent.setup();
    const first = renderSection();
    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'resume.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByTestId('portability-import-start'));
    expect(await screen.findByTestId('portability-report')).toHaveTextContent(
      'ready for approval',
    );

    first.unmount();
    renderSection();

    expect(await screen.findByTestId('portability-report')).toHaveTextContent(
      'ready for approval',
    );
    expect(screen.getByTestId('portability-approve')).toBeEnabled();
  });

  it('mounts the export anchor before clicking and revokes it asynchronously', async () => {
    const createObjectURL = vi.fn(() => 'blob:portability-export');
    const revokeObjectURL = vi.fn();
    const NativeURL = URL;
    class TestURL extends NativeURL {}
    TestURL.createObjectURL = createObjectURL;
    TestURL.revokeObjectURL = revokeObjectURL;
    vi.stubGlobal('URL', TestURL);
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(function clickMountedAnchor(this: HTMLAnchorElement) {
        expect(document.body.contains(this)).toBe(true);
      });
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId('portability-export-start'));
    await user.click(
      await screen.findByTestId('portability-export-download', {}, { timeout: 3000 }),
    );

    await waitFor(() => expect(click).toHaveBeenCalledOnce());
    await waitFor(() => expect(revokeObjectURL).toHaveBeenCalledWith('blob:portability-export'));
    expect(document.querySelector('a[download^="twin-kb-"]')).toBeNull();
  });

  it('advances the workflow rail one stage at a time through the import journey', async () => {
    const railState = (stage: string) =>
      screen.getByTestId(`portability-rail-${stage}`).dataset.state;
    const user = userEvent.setup();
    renderSection();

    // Nothing started: every stage is pending, and the rail is already on
    // screen so the operator can see the workflow before committing to it.
    expect(screen.getByTestId('portability-rail')).toBeInTheDocument();
    for (const stage of ['export', 'dry-run', 'approve', 'apply', 'validate']) {
      expect(railState(stage)).toBe('pending');
    }

    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'staging.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByTestId('portability-import-start'));
    await screen.findByTestId('portability-report', {}, { timeout: 3000 });

    // Dry-run landed on awaiting-approval: the approval is what is in flight.
    await waitFor(() => expect(railState('dry-run')).toBe('done'));
    expect(railState('approve')).toBe('active');
    expect(railState('apply')).toBe('pending');
    expect(railState('validate')).toBe('pending');

    // Each hop is anchored on the button the NEXT status unlocks, not on a
    // bare timeout: the job advances on the status poll, so a 1 s waitFor
    // races the poll interval instead of observing the transition.
    await user.click(screen.getByTestId('portability-approve'));
    const apply = await screen.findByTestId('portability-apply');
    expect(railState('approve')).toBe('done');
    expect(railState('apply')).toBe('active');

    await user.click(apply);
    const validate = await screen.findByTestId(
      'portability-validate',
      {},
      { timeout: 3000 },
    );
    expect(railState('apply')).toBe('done');
    expect(railState('validate')).toBe('active');

    await user.click(validate);
    await screen.findByText(/Validation passed/, {}, { timeout: 3000 });
    expect(railState('validate')).toBe('done');
    // Export runs on the SOURCE instance: importing a bundle must never claim
    // this instance produced it.
    expect(railState('export')).toBe('pending');
  }, 10_000);

  it('marks the export stage done once its archive is downloadable', async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId('portability-export-start'));
    await screen.findByTestId('portability-export-download', {}, { timeout: 3000 });
    expect(screen.getByTestId('portability-rail-export').dataset.state).toBe('done');
  });

  it('names the picked bundle itself rather than leaving the browser file control visible', async () => {
    const user = userEvent.setup();
    renderSection();

    const input = screen.getByTestId('portability-import-file');
    // Hidden: the native control renders its own chrome in the BROWSER's
    // locale, which is how French strings appeared in this English UI.
    expect(input).not.toBeVisible();
    expect(screen.getByTestId('portability-import-dropzone')).toHaveTextContent(
      'Drop a bundle here or click to browse',
    );

    await user.upload(
      input,
      new File(['x'.repeat(2048)], 'production-kb.tar.gz', {
        type: 'application/gzip',
      }),
    );

    const dropzone = screen.getByTestId('portability-import-dropzone');
    expect(dropzone).toHaveTextContent('production-kb.tar.gz');
    expect(screen.getByTestId('portability-import-filesize')).toHaveTextContent(
      '2 KiB',
    );
  });

  it('treats a blank folder mapping as "rename nothing" instead of rejecting it', async () => {
    const user = userEvent.setup();
    renderSection();

    expect(screen.getByTestId('portability-folder-map')).toHaveValue('');
    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'kb.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByTestId('portability-import-start'));

    expect(
      await screen.findByTestId('portability-report', {}, { timeout: 3000 }),
    ).toHaveTextContent('ready for approval');
    expect(
      screen.queryByText(/Folder mapping must be a JSON object/),
    ).not.toBeInTheDocument();
  });

  it('exposes the unverified-bundle opt-in as an off-by-default switch', async () => {
    const user = userEvent.setup();
    renderSection();

    const toggle = screen.getByTestId('portability-allow-unverified');
    expect(toggle).toHaveAttribute('role', 'switch');
    expect(toggle).toHaveAttribute('aria-checked', 'false');

    await user.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'true');
  });

  it('announces every rail stage state to assistive tech, not only through colour', async () => {
    // axe cannot flag a state that is simply never announced: colour, the
    // bullet glyph and data-state are all invisible to a screen reader, and
    // aria-current marks the active step only. This is the regression guard.
    const user = userEvent.setup();
    renderSection();

    const stageText = (stage: string) =>
      screen.getByTestId(`portability-rail-${stage}`).textContent;

    for (const stage of ['export', 'dry-run', 'approve', 'apply', 'validate']) {
      expect(stageText(stage)).toContain('not started');
    }

    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'kb.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByTestId('portability-import-start'));
    await screen.findByTestId('portability-report', {}, { timeout: 3000 });

    await waitFor(() => expect(stageText('dry-run')).toContain('completed'));
    expect(stageText('approve')).toContain('in progress');
    expect(stageText('apply')).toContain('not started');
  }, 10_000);

  it('points an oversized bundle at the module entry point that actually exists', async () => {
    // `twin-kb-bundle` is the bundle FORMAT (manifest.format), not a console
    // script — the root pyproject declares no [project.scripts] at all.
    const user = userEvent.setup();
    renderSection();

    const huge = new File(['x'], 'huge.tar.gz', { type: 'application/gzip' });
    Object.defineProperty(huge, 'size', { value: 200 * 1024 * 1024 });
    await user.upload(screen.getByTestId('portability-import-file'), huge);

    const notice = await screen.findByTestId('portability-import-oversized');
    expect(notice).toHaveTextContent(
      'python -m twindb_lightrag_memgraph.portability',
    );
    expect(notice).not.toHaveTextContent('twin-kb-bundle CLI');
    expect(screen.getByTestId('portability-import-start')).toBeDisabled();
  });

  it('describes the unverified opt-in as a consistency waiver, not a skipped signature', () => {
    // compat.py:321 gates on `manifest.consistency.status == "verified"`; the
    // per-file sha256 + size verification in bundle.py runs either way. There
    // is no signature anywhere in the format.
    renderSection();

    const zone = screen.getByTestId('portability-danger-zone');
    expect(zone).toHaveTextContent('consistency check was unverified');
    expect(zone).toHaveTextContent('Integrity checks still apply');
    expect(zone).not.toHaveTextContent(/signature/i);
  });

  /** A stub import job carrying the durable fields the rail reads. */
  function stubImportJob(overrides: Record<string, unknown>) {
    return {
      id: 'imp_stub',
      kind: 'import',
      workspace: 'base',
      status: 'awaiting-approval',
      created_at: '2026-08-26T13:30:00Z',
      updated_at: '2026-08-26T13:30:01Z',
      actor: 'operator.demo',
      approved_report_hash: null,
      applied_by: null,
      validated_by: null,
      cancelled_by: null,
      options: {},
      result: null,
      report: {
        report_hash: PORTABILITY_REPORT_HASH,
        blocking: [],
        compat: [],
        classification: {
          source_max: 'C2',
          target_ceiling: 'C2',
          unknown_present: false,
        },
      },
      validation: null,
      error: null,
      download_available: false,
      ...overrides,
    };
  }

  async function renderStubbedJob(overrides: Record<string, unknown>) {
    const job = stubImportJob(overrides);
    server.use(
      http.post('*/twin/api/admin/portability/imports', () =>
        HttpResponse.json(job, { status: 202 }),
      ),
      http.get('*/twin/api/admin/portability/imports/:id', () =>
        HttpResponse.json(job),
      ),
    );
    const user = userEvent.setup();
    renderSection();
    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'kb.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByTestId('portability-import-start'));
    // A job that failed before producing a report renders no report card, so
    // anchor on the status row, which every started job renders.
    await screen.findByTestId('portability-import-status', {}, { timeout: 3000 });
    if (job.report) {
      await screen.findByTestId('portability-report', {}, { timeout: 3000 });
    }
  }

  const railState = (stage: string) =>
    screen.getByTestId(`portability-rail-${stage}`).dataset.state;

  it('keeps the dry-run stage completed after the run is cancelled', async () => {
    // The report stays on screen once cancelled; a rail that resets to "not
    // started" would contradict it. cancel() refuses {applying, applied,
    // validating}, so a cancelled job never carries applied_by.
    await renderStubbedJob({ status: 'cancelled', cancelled_by: 'operator.demo' });

    expect(railState('dry-run')).toBe('done');
    expect(railState('approve')).toBe('pending');
    expect(railState('apply')).toBe('pending');
    expect(railState('validate')).toBe('pending');
    expect(screen.getByTestId('portability-rail-halted')).toHaveTextContent(
      'run cancelled',
    );
    expect(screen.getByTestId('portability-report')).toBeInTheDocument();
  });

  it('marks apply failed and keeps the earlier stages when apply breaks', async () => {
    // applied_by is written ENTERING `applying`, so its presence on a failed
    // job identifies apply as the stage that broke.
    await renderStubbedJob({
      status: 'failed',
      approved_report_hash: PORTABILITY_REPORT_HASH,
      applied_by: 'operator.demo',
      error: 'target store rejected the write',
    });

    expect(railState('dry-run')).toBe('done');
    expect(railState('approve')).toBe('done');
    expect(railState('apply')).toBe('failed');
    expect(railState('validate')).toBe('pending');
    expect(
      screen.getByTestId(`portability-rail-apply`).textContent,
    ).toContain('failed');
  });

  it('does not blame apply for a failure that happened during the dry-run', async () => {
    await renderStubbedJob({ status: 'failed', report: null, applied_by: null });

    expect(screen.queryByTestId('portability-report')).not.toBeInTheDocument();
    expect(railState('dry-run')).toBe('failed');
    expect(railState('apply')).toBe('pending');
  });

  it('links every switch to its visible explanation', () => {
    renderSection();

    for (const testId of [
      'portability-include-activity',
      'portability-include-procedures',
      'portability-allow-unverified',
    ]) {
      const control = screen.getByTestId(testId);
      const describedBy = control.getAttribute('aria-describedby');
      expect(describedBy).toBeTruthy();
      expect(document.getElementById(describedBy as string)).not.toBeNull();
    }

    // The destructive one must carry its warning, not just its label.
    const help = document.getElementById(
      screen
        .getByTestId('portability-allow-unverified')
        .getAttribute('aria-describedby') as string,
    );
    expect(help).toHaveTextContent('Never for production');
  });

  it('validates the folder map before uploading', async () => {
    const user = userEvent.setup();
    renderSection();
    await user.upload(
      screen.getByTestId('portability-import-file'),
      new File(['bundle'], 'kb.tar.gz', { type: 'application/gzip' }),
    );
    fireEvent.change(screen.getByTestId('portability-folder-map'), {
      target: { value: '[]' },
    });
    await user.click(screen.getByTestId('portability-import-start'));

    expect(
      await screen.findByText(/Folder mapping must be a JSON object/),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByTestId('portability-report')).not.toBeInTheDocument(),
    );
  });
});
