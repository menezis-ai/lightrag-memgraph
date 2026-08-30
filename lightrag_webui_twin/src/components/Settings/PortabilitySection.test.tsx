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
