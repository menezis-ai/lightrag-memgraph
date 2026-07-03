/**
 * Unit tests for DocumentsTab.
 *
 * Behaviors under test:
 *   - renders all docs by default
 *   - status filter pill click filters the table
 *   - search input filters by source name
 *   - tag filter chip: click a row tag adds it as a filter
 *   - multi-select toggles + bulk Retag invokes onOpenBulkRetag
 *   - Add source button calls onOpenAdd
 *   - source filename opens the document detail panel callback
 *   - row Retag button calls onOpenRetag(doc)
 *   - empty state appears when filters match no doc
 */

import type { ReactElement } from 'react';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DocumentsTab } from './DocumentsTab';
import {
  DOCUMENT_FIXTURES,
  FOLDER_FIXTURES,
  TAG_FIXTURES,
} from '../fixtures';

function renderTab(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function defaultProps() {
  return {
    docs: DOCUMENT_FIXTURES,
    tagCatalog: TAG_FIXTURES,
    onOpenAdd: vi.fn(),
    onOpenRetag: vi.fn(),
    onOpenBulkRetag: vi.fn(),
    onAddToast: vi.fn(),
  };
}

beforeEach(() => {
  window.history.replaceState(null, '', '/');
});
afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('DocumentsTab — rendering', () => {
  it('renders all docs by default', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    expect(screen.getByText('Document management')).toBeInTheDocument();
    expect(screen.getByText('Indexed preview')).toBeInTheDocument();
    expect(screen.queryByText('Summary')).toBeNull();
    DOCUMENT_FIXTURES.forEach((d) => {
      expect(screen.getByText(d.file_path)).toBeInTheDocument();
    });
  });

  it('does not render the legacy Twin suffix beside tag controls', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);

    expect(screen.getByText('Filter by tag')).toBeInTheDocument();
    expect(screen.queryByText(/Twin/)).toBeNull();
  });

  it('shows status counts in the filter pills', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    // DOCUMENT_FIXTURES = 7 docs: 4 PROCESSED, 1 FAILED, 2 PROCESSING
    expect(screen.getByRole('button', { name: /^All \(7\)/ })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Completed \(4\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Failed \(1\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Processing \(2\)/ }),
    ).toBeInTheDocument();
  });

  it('renders optimistic upload rows as pending but not actionable', () => {
    const optimisticDoc = {
      ...DOCUMENT_FIXTURES[0],
      doc_id: 'upload_tmp_1',
      track_id: 'track_tmp_1',
      file_path: 'new-runbook.pdf',
      content_summary: 'Upload accepted, waiting for ingestion worker.',
      status: 'PENDING' as const,
      chunks_count: null,
      _optimisticUpload: true,
    };

    renderTab(
      <DocumentsTab
        {...defaultProps()}
        docs={[optimisticDoc]}
        onDeleteDoc={vi.fn()}
      />,
    );

    expect(screen.getByText('new-runbook.pdf')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^All \(1\)/ })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Pending \(1\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText('new-runbook.pdf is waiting for ingestion'),
    ).toBeDisabled();
    expect(screen.queryByLabelText('Retag new-runbook.pdf')).toBeNull();
    expect(screen.queryByLabelText('Delete new-runbook.pdf')).toBeNull();
  });
});

describe('DocumentsTab — filters', () => {
  it('status filter narrows the visible rows', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Failed \(1\)/ }));
    // Only the failed doc visible
    expect(screen.getByTestId('docs-row-d3')).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d1')).toBeNull();
  });

  it('?source= filter shows a removable pill; removing it restores the table', async () => {
    const target = DOCUMENT_FIXTURES[0].file_path;
    window.history.replaceState(
      null,
      '',
      `/?source=${encodeURIComponent(target)}`,
    );
    renderTab(<DocumentsTab {...defaultProps()} />);

    // Filter applied AND visible — it must never be an invisible filter.
    expect(screen.getByTestId('source-filter-row')).toBeInTheDocument();
    expect(screen.getByTestId(`source-filter-${target}`)).toBeInTheDocument();
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();

    await userEvent.click(
      screen.getByLabelText(`Remove source filter ${target}`),
    );
    expect(screen.queryByTestId('source-filter-row')).toBeNull();
    expect(screen.getByTestId('docs-row-d4')).toBeInTheDocument();
  });

  it('?source= also filters by doc_id for Graph source_docs drilldown', () => {
    const target = DOCUMENT_FIXTURES[0].doc_id;
    window.history.replaceState(
      null,
      '',
      `/?source=${encodeURIComponent(target)}`,
    );
    renderTab(<DocumentsTab {...defaultProps()} />);

    expect(screen.getByTestId('source-filter-row')).toBeInTheDocument();
    expect(screen.getByTestId(`source-filter-${target}`)).toBeInTheDocument();
    expect(screen.getByTestId(`docs-row-${target}`)).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();
  });

  it('search filters by source name', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'oracle');
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();
  });

  it('keeps All global while non-all status counts follow the active search filter', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'oracle');

    expect(screen.getByRole('button', { name: /^All \(7\)/ })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Completed \(2\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Failed \(0\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Processing \(0\)/ }),
    ).toBeInTheDocument();
  });

  it('clicking a tag on a row adds it as a filter', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    // d1 has tags rman + oracle; click the chip in the row
    const tagSpan = screen.getByTestId('row-tag-d1-rman');
    await userEvent.click(tagSpan);
    // Now the tag filter row should contain "rman" as a removable chip;
    // the row should still be visible since it has rman.
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    // d4 has no rman tag → filtered out
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();
  });

  it('keeps All global while non-all status counts follow the active tag filter', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('row-tag-d1-rman'));

    expect(screen.getByRole('button', { name: /^All \(7\)/ })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Completed \(2\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Failed \(0\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /^Processing \(0\)/ }),
    ).toBeInTheDocument();
  });

  it('shows the document count for the active tag filter selection', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('row-tag-d1-rman'));

    const count = screen.getByTestId('tag-filter-count');
    expect(count).toHaveTextContent(
      '2 documents match',
    );
    expect(count).not.toHaveClass('pill');
  });

  it('clears search, status and tag filters from the tag-filter row', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const clear = screen.getByRole('button', {
      name: 'Clear all tags/filters',
    });
    expect(clear).toBeDisabled();

    await userEvent.click(screen.getByTestId('row-tag-d1-rman'));
    await userEvent.type(screen.getByLabelText('Search source'), 'oracle');
    await userEvent.click(screen.getByRole('button', { name: /^Failed/ }));

    expect(clear).toBeEnabled();
    expect(screen.getByTestId('tag-filter-count')).toHaveTextContent(
      '0 documents match',
    );

    await userEvent.click(clear);

    expect(screen.getByLabelText('Search source')).toHaveValue('');
    expect(screen.queryByTestId('tag-filter-count')).toBeNull();
    expect(screen.getByRole('button', { name: /^All \(7\)/ })).toBeInTheDocument();
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    expect(screen.getByTestId('docs-row-d4')).toBeInTheDocument();
    expect(clear).toBeDisabled();
  });

  it('exposes hidden row tags from the +N overflow chip without opening a modal', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const row = screen.getByTestId('docs-row-d5');
    const overflow = within(row).getByRole('button', {
      name: /Show 1 more tag/i,
    });
    expect(overflow).toHaveTextContent('+1');
    expect(within(overflow).getByText('production')).toBeInTheDocument();
  });

  it('navigates Add-tag suggestions with arrow keys and Enter', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: '+ Add tag' }));

    const input = screen.getByLabelText('Add tag filter');
    expect(screen.getByRole('listbox', { name: 'Tag suggestions' })).toBeInTheDocument();
    expect(input).toHaveAttribute('aria-controls', 'documents-tag-suggestions');

    await userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}');

    expect(
      String(input.getAttribute('aria-activedescendant')),
    ).toMatch(/^documents-tag-suggestions-option-\d+$/);
    expect(screen.getAllByRole('button', { name: /^Remove (?!source)/ })).toHaveLength(1);
  });

  it('status and tag filters are URL backed', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);

    await userEvent.click(screen.getByRole('button', { name: /^Failed \(1\)/ }));
    expect(window.location.search).toContain('status=failed');

    await userEvent.click(screen.getByRole('button', { name: '+ Add tag' }));
    await userEvent.type(screen.getByLabelText('Add tag filter'), 'rman');
    await userEvent.click(screen.getByTestId('docs-tag-sugg-rman'));
    expect(window.location.search).toContain('tag=rman');
  });

  it('empty state appears when filters match nothing', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'nope-no-match-zzz');
    expect(screen.getByTestId('docs-empty')).toBeInTheDocument();
  });
});

describe('DocumentsTab — selection + bulk', () => {
  it('toggling rows builds a selection and reveals the bulk bar', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByLabelText('Select oracle-restart-procedure.pdf'));
    const bulkBar = screen.getByRole('region', { name: 'Bulk actions' });
    expect(bulkBar).toBeInTheDocument();
    // "1 selected" copy inside the bulk bar (not the pipeline badge)
    expect(bulkBar.textContent).toMatch(/\b1\b/);
    expect(bulkBar.textContent).toMatch(/selected/);
  });

  it('Bulk Retag invokes onOpenBulkRetag with selected docs', async () => {
    const p = defaultProps();
    renderTab(<DocumentsTab {...p} />);
    await userEvent.click(screen.getByLabelText('Select oracle-restart-procedure.pdf'));
    await userEvent.click(
      screen.getByRole('button', { name: /Retag 1 sources/ }),
    );
    expect(p.onOpenBulkRetag).toHaveBeenCalledTimes(1);
    const arg = p.onOpenBulkRetag.mock.calls[0][0];
    expect(arg).toHaveLength(1);
    expect(arg[0].doc_id).toBe('d1');
  });
});

describe('DocumentsTab — header actions', () => {
  it('Add source button calls onOpenAdd', async () => {
    const p = defaultProps();
    renderTab(<DocumentsTab {...p} />);
    await userEvent.click(screen.getByRole('button', { name: /Add source/ }));
    expect(p.onOpenAdd).toHaveBeenCalled();
  });

  it('"Re-process failed sources" invokes the retry callback with the failed count', async () => {
    const p = { ...defaultProps(), onScanRetry: vi.fn() };
    renderTab(<DocumentsTab {...p} />);
    // DOCUMENT_FIXTURES carries 1 FAILED doc (huge-archive.zip).
    const btn = screen.getByRole('button', {
      name: /Re-process failed/,
    });
    expect(btn).toBeEnabled();
    await userEvent.click(btn);
    expect(p.onScanRetry).toHaveBeenCalledWith(1);
    expect(p.onAddToast).not.toHaveBeenCalled();
  });

  it('disables the re-process button + shows tooltip when no source is failed (audit C7)', () => {
    const p = {
      ...defaultProps(),
      docs: defaultProps().docs.filter(
        (d) => String(d.status).toUpperCase() !== 'FAILED',
      ),
      onScanRetry: vi.fn(),
    };
    renderTab(<DocumentsTab {...p} />);
    const btn = screen.getByRole('button', {
      name: /Re-process failed/,
    });
    expect(btn).toBeDisabled();
    expect(btn.getAttribute('title')).toMatch(
      /No failed sources to re-process/i,
    );
    // The badge counter is gone when there is nothing to re-process.
    expect(btn.textContent).not.toMatch(/\d+/);
  });

  it('shows the failed count and POST /documents/reprocess_failed in the button tooltip (audit C7)', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const btn = screen.getByRole('button', {
      name: /Re-process failed/,
    });
    const title = btn.getAttribute('title') ?? '';
    expect(title).toContain('1 failed source');
    expect(title).toContain('POST /documents/reprocess_failed');
  });

  it('shows real pipeline status messages from props without fixture rows', async () => {
    const p = {
      ...defaultProps(),
      onTogglePipeline: vi.fn(),
      onRefreshPipeline: vi.fn(),
      pipelineOpen: true,
      pipelineStatus: {
        busy: true,
        job_count: 2,
        job_name: 'document indexing',
        latest_message: 'Memgraph merge complete',
        history_messages: [
          'Dequeued BNP incident note',
          'Embedding batch 1/1 complete',
          'Memgraph merge complete',
        ],
      },
    };
    renderTab(<DocumentsTab {...p} />);

    const dialog = screen.getByRole('dialog', { name: 'Pipeline logs' });
    expect(dialog).toHaveTextContent('document indexing');
    expect(dialog).toHaveTextContent('Dequeued BNP incident note');
    expect(dialog).toHaveTextContent('Embedding batch 1/1 complete');
    expect(dialog).toHaveTextContent('Memgraph merge complete');
    expect(dialog).not.toHaveTextContent(/worker|queued at pipeline/i);

    await userEvent.click(screen.getByRole('button', { name: /Refresh/ }));
    expect(p.onRefreshPipeline).toHaveBeenCalledTimes(1);
  });

  it('does not render any of the legacy "Scan" wording (audit C7)', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const legacyRetryLabel = new RegExp(['Scan', 'Retry'].join(' \\/ '));
    const legacyPipelineToast = new RegExp(['Pipeline', 'scan'].join(' '));
    const legacyCompletedToast = new RegExp(['Scan', 'completed'].join(' '));
    expect(screen.queryByText(legacyRetryLabel)).toBeNull();
    expect(screen.queryByText(legacyPipelineToast)).toBeNull();
    expect(screen.queryByText(legacyCompletedToast)).toBeNull();
    // The bare "Scan" word should also not appear on its own as a
    // button label — `getAllByRole` is enough to make this explicit.
    const buttons = screen.getAllByRole('button');
    const justScan = buttons.find((b) => b.textContent?.trim() === 'Scan');
    expect(justScan).toBeUndefined();
  });

  it('row Retag button calls onOpenRetag with the doc', async () => {
    const p = defaultProps();
    renderTab(<DocumentsTab {...p} />);
    await userEvent.click(
      screen.getByLabelText('Retag oracle-restart-procedure.pdf'),
    );
    expect(p.onOpenRetag).toHaveBeenCalled();
    expect(p.onOpenRetag.mock.calls[0][0].doc_id).toBe('d1');
  });

  it('exposes row tag/folder actions as a separate action group', () => {
    renderTab(
      <DocumentsTab
        {...defaultProps()}
        activeFolder="default"
        folderList={FOLDER_FIXTURES}
        canManageFolders
      />,
    );

    const actions = screen.getByRole('group', {
      name: 'Actions for oracle-restart-procedure.pdf',
    });
    expect(actions).toHaveAttribute('data-testid', 'docs-row-actions-d1');
    expect(
      within(actions).getByLabelText('Retag oracle-restart-procedure.pdf'),
    ).toBeInTheDocument();
    expect(
      within(actions).getByLabelText('Manage folders for oracle-restart-procedure.pdf'),
    ).toBeInTheDocument();
  });

  it('source filename opens the document detail callback', async () => {
    const p = { ...defaultProps(), onOpenDetail: vi.fn() };
    renderTab(<DocumentsTab {...p} />);
    await userEvent.click(screen.getByTestId('docs-row-filename-d1'));
    expect(p.onOpenDetail).toHaveBeenCalledTimes(1);
    expect(p.onOpenDetail.mock.calls[0][0].doc_id).toBe('d1');
  });

});

describe('DocumentsTab — folder membership admin actions', () => {
  let originalFetch: typeof fetch;
  let fetchMock: ReturnType<typeof vi.fn>;
  const folderList = [
    {
      ...FOLDER_FIXTURES[0],
      id: 'default',
      kb: 'Default',
      current: true,
    },
    {
      ...FOLDER_FIXTURES[0],
      id: 'sandbox',
      kb: 'Sandbox',
      current: false,
    },
  ];

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('hides folder membership controls for non-admin users', () => {
    renderTab(
      <DocumentsTab
        {...defaultProps()}
        activeFolder="default"
        folderList={folderList}
        canManageFolders={false}
      />,
    );
    expect(
      screen.queryByLabelText('Manage folders for oracle-restart-procedure.pdf'),
    ).toBeNull();
  });

  it('bulk copy adds every selected document to the target folder', async () => {
    const props = defaultProps();
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/quota')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              used_bytes: 1,
              limit_bytes: 10,
              used_pct: 0.1,
              status: 'ok',
              warn_threshold: 0.85,
              configured: true,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/') && url.includes('/folders') && init?.method === 'POST') {
        const docId = url.split('/documents/')[1]?.split('/folders')[0];
        return Promise.resolve(
          new Response(
            JSON.stringify({ doc_id: docId, folders: ['default', 'sandbox'] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    });

    renderTab(
      <DocumentsTab
        {...props}
        activeFolder="default"
        folderList={folderList}
        canManageFolders
      />,
    );

    await userEvent.click(screen.getByLabelText(`Select ${DOCUMENT_FIXTURES[0].file_path}`));
    await userEvent.click(screen.getByLabelText(`Select ${DOCUMENT_FIXTURES[1].file_path}`));

    const folderActions = screen.getByRole('group', { name: 'Folder actions' });
    expect(folderActions).toHaveAttribute('data-testid', 'docs-bulk-folder-actions');
    expect(within(folderActions).getByTestId('docs-bulk-copy')).toBeInTheDocument();
    expect(within(folderActions).getByTestId('docs-bulk-move')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('docs-bulk-copy'));
    await userEvent.selectOptions(screen.getByLabelText('Bulk target folder'), 'sandbox');
    await userEvent.click(screen.getByTestId('bulk-folder-copy'));

    await waitFor(() =>
      expect(props.onAddToast).toHaveBeenCalledWith(
        'Sources copied to folder',
        expect.stringContaining('2 sources copied'),
      ),
    );
    const postCalls = fetchMock.mock.calls.filter(
      ([url, init]) =>
        String(url).includes('/documents/') &&
        String(url).includes('/folders') &&
        (init as RequestInit | undefined)?.method === 'POST',
    );
    expect(postCalls).toHaveLength(2);
    expect(
      postCalls.map(([url]) => String(url)).sort(),
    ).toEqual(
      expect.arrayContaining([
        expect.stringContaining('/documents/d1/folders'),
        expect.stringContaining('/documents/d2/folders'),
      ]),
    );
    postCalls.forEach(([, init]) => {
      expect(JSON.parse((init as RequestInit).body as string)).toEqual({
        folder_id: 'sandbox',
      });
    });
  });

  it('bulk move adds target membership before removing the active folder', async () => {
    const props = defaultProps();
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/quota')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              used_bytes: 1,
              limit_bytes: 10,
              used_pct: 0.1,
              status: 'ok',
              warn_threshold: 0.85,
              configured: true,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/') && url.includes('/folders') && init?.method === 'POST') {
        const docId = url.split('/documents/')[1]?.split('/folders')[0];
        return Promise.resolve(
          new Response(
            JSON.stringify({ doc_id: docId, folders: ['default', 'sandbox'] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/') && url.includes('/folders/default') && init?.method === 'DELETE') {
        const docId = url.split('/documents/')[1]?.split('/folders')[0];
        return Promise.resolve(
          new Response(
            JSON.stringify({
              doc_id: docId,
              removed_folder: 'default',
              physically_deleted: false,
              remaining_folders: ['sandbox'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    });

    renderTab(
      <DocumentsTab
        {...props}
        activeFolder="default"
        folderList={folderList}
        canManageFolders
      />,
    );

    await userEvent.click(screen.getByLabelText(`Select ${DOCUMENT_FIXTURES[0].file_path}`));
    await userEvent.click(screen.getByLabelText(`Select ${DOCUMENT_FIXTURES[1].file_path}`));
    await userEvent.click(screen.getByTestId('docs-bulk-move'));
    await userEvent.selectOptions(screen.getByLabelText('Bulk target folder'), 'sandbox');
    await userEvent.click(screen.getByTestId('bulk-folder-move'));

    await waitFor(() =>
      expect(props.onAddToast).toHaveBeenCalledWith(
        'Sources moved to folder',
        expect.stringContaining('2 sources moved'),
      ),
    );
    for (const docId of ['d1', 'd2']) {
      const postIdx = fetchMock.mock.calls.findIndex(
        ([url, init]) =>
          String(url).includes(`/documents/${docId}/folders`) &&
          (init as RequestInit | undefined)?.method === 'POST',
      );
      const deleteIdx = fetchMock.mock.calls.findIndex(
        ([url, init]) =>
          String(url).includes(`/documents/${docId}/folders/default`) &&
          (init as RequestInit | undefined)?.method === 'DELETE',
      );
      expect(postIdx).toBeGreaterThanOrEqual(0);
      expect(deleteIdx).toBeGreaterThan(postIdx);
    }
  });

  it('adds a document to another folder from the admin modal', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/quota')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              used_bytes: 1,
              limit_bytes: 10,
              used_pct: 0.1,
              status: 'ok',
              warn_threshold: 0.85,
              configured: true,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders') && init?.method === 'POST') {
        return Promise.resolve(
          new Response(
            JSON.stringify({ doc_id: 'd1', folders: ['default', 'sandbox'] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders')) {
        return Promise.resolve(
          new Response(JSON.stringify({ doc_id: 'd1', folders: ['default'] }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    });
    renderTab(
      <DocumentsTab
        {...defaultProps()}
        activeFolder="default"
        folderList={folderList}
        canManageFolders
      />,
    );

    await userEvent.click(
      screen.getByLabelText('Manage folders for oracle-restart-procedure.pdf'),
    );
    expect(await screen.findByTestId('document-folders-modal')).toHaveTextContent(
      'Active folder: Default (default)',
    );
    await screen.findByRole('option', { name: 'Sandbox (sandbox)' });
    await userEvent.selectOptions(screen.getByLabelText('Target folder'), 'sandbox');
    await userEvent.click(screen.getByTestId('document-folder-copy'));

    const postCall = fetchMock.mock.calls.find(
      ([url, init]) =>
        String(url).includes('/documents/d1/folders') &&
        (init as RequestInit | undefined)?.method === 'POST',
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse((postCall?.[1] as RequestInit).body as string)).toEqual({
      folder_id: 'sandbox',
    });
  });

  it('move = add to the target folder then remove from the active folder', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/quota')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              used_bytes: 1,
              limit_bytes: 10,
              used_pct: 0.1,
              status: 'ok',
              warn_threshold: 0.85,
              configured: true,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders') && init?.method === 'POST') {
        return Promise.resolve(
          new Response(
            JSON.stringify({ doc_id: 'd1', folders: ['default', 'sandbox'] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders') && init?.method === 'DELETE') {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              doc_id: 'd1',
              removed_folder: 'default',
              physically_deleted: false,
              remaining_folders: ['sandbox'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders')) {
        return Promise.resolve(
          new Response(JSON.stringify({ doc_id: 'd1', folders: ['default'] }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    });

    renderTab(
      <DocumentsTab
        {...defaultProps()}
        activeFolder="default"
        folderList={folderList}
        canManageFolders
      />,
    );

    await userEvent.click(
      screen.getByLabelText('Manage folders for oracle-restart-procedure.pdf'),
    );
    await screen.findByTestId('document-folders-modal');
    await screen.findByRole('option', { name: 'Sandbox (sandbox)' });
    await userEvent.selectOptions(screen.getByLabelText('Target folder'), 'sandbox');
    await userEvent.click(screen.getByTestId('document-folder-move'));

    // Move MUST be POST(target) THEN DELETE(active) — order is load-bearing:
    // adding first guarantees the doc is never folderless mid-move. Assert the
    // call indices, not just presence (a DELETE-then-POST impl must fail here).
    const postIdx = fetchMock.mock.calls.findIndex(
      ([url, init]) =>
        String(url).includes('/documents/d1/folders') &&
        (init as RequestInit | undefined)?.method === 'POST',
    );
    const deleteIdx = fetchMock.mock.calls.findIndex(
      ([url, init]) =>
        String(url).includes('/documents/d1/folders/default') &&
        (init as RequestInit | undefined)?.method === 'DELETE',
    );
    expect(postIdx).toBeGreaterThanOrEqual(0);
    expect(deleteIdx).toBeGreaterThan(postIdx);
    expect(
      JSON.parse(
        (fetchMock.mock.calls[postIdx][1] as RequestInit).body as string,
      ),
    ).toEqual({ folder_id: 'sandbox' });
  });

  it('move that copies but fails to remove warns the doc is now in both folders', async () => {
    const props = defaultProps();
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/quota')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              used_bytes: 1,
              limit_bytes: 10,
              used_pct: 0.1,
              status: 'ok',
              warn_threshold: 0.85,
              configured: true,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders') && init?.method === 'POST') {
        return Promise.resolve(
          new Response(
            JSON.stringify({ doc_id: 'd1', folders: ['default', 'sandbox'] }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders') && init?.method === 'DELETE') {
        // The removal half fails — the doc stays in BOTH folders.
        return Promise.resolve(new Response('boom', { status: 500 }));
      }
      if (url.includes('/documents/d1/folders')) {
        return Promise.resolve(
          new Response(JSON.stringify({ doc_id: 'd1', folders: ['default'] }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    });

    renderTab(
      <DocumentsTab
        {...props}
        activeFolder="default"
        folderList={folderList}
        canManageFolders
      />,
    );

    await userEvent.click(
      screen.getByLabelText('Manage folders for oracle-restart-procedure.pdf'),
    );
    await screen.findByTestId('document-folders-modal');
    await screen.findByRole('option', { name: 'Sandbox (sandbox)' });
    await userEvent.selectOptions(screen.getByLabelText('Target folder'), 'sandbox');
    await userEvent.click(screen.getByTestId('document-folder-move'));

    await waitFor(() =>
      expect(props.onAddToast).toHaveBeenCalledWith(
        'Move incomplete — copied, not removed',
        expect.stringContaining('now in both folders'),
      ),
    );
  });

  it('requires explicit confirmation before removing the last folder membership', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/quota')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              used_bytes: 1,
              limit_bytes: 10,
              used_pct: 0.1,
              status: 'ok',
              warn_threshold: 0.85,
              configured: true,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders/default')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              ok: true,
              doc_id: 'd1',
              removed_folder: 'default',
              physically_deleted: true,
              remaining_folders: [],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        );
      }
      if (url.includes('/documents/d1/folders')) {
        return Promise.resolve(
          new Response(JSON.stringify({ doc_id: 'd1', folders: ['default'] }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    });
    renderTab(
      <DocumentsTab
        {...defaultProps()}
        activeFolder="default"
        folderList={folderList}
        canManageFolders
      />,
    );

    await userEvent.click(
      screen.getByLabelText('Manage folders for oracle-restart-procedure.pdf'),
    );
    const modal = await screen.findByTestId('document-folders-modal');
    expect(modal).toHaveTextContent('oracle-restart-procedure.pdf');
    expect(modal).toHaveTextContent('Default (default)');
    expect(modal).toHaveTextContent('permanently delete the document');

    await userEvent.click(screen.getByTestId('document-folder-remove-active'));
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).includes('/documents/d1/folders/default') &&
          (init as RequestInit | undefined)?.method === 'DELETE',
      ),
    ).toBe(false);
    expect(
      screen.getByRole('button', { name: 'Confirm permanent delete' }),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('document-folder-remove-active'));
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).includes('/documents/d1/folders/default') &&
          (init as RequestInit | undefined)?.method === 'DELETE',
      ),
    ).toBe(true);
  });
});

describe('DocumentsTab — failed row surfaces error_msg (TR-ING-01)', () => {
  it('renders one non-duplicated failure reason inline on a FAILED row', () => {
    // d3 = the FAILED fixture with error_msg='Unsupported MIME type: …'.
    // Before this PR the row went red without ever exposing the reason —
    // the operator's exact complaint from QA.
    renderTab(<DocumentsTab {...defaultProps()} />);
    const err = screen.getByTestId('docs-row-error-d3');
    expect(err.textContent).toBe(
      'Failed ingest — Unsupported MIME type: application/zip',
    );
    expect(err.textContent?.match(/unsupported MIME/gi)).toHaveLength(1);
  });

  it('omits the error line on a row that has no error_msg', () => {
    // d1 = PROCESSED, no error_msg → the slot must not render at all.
    renderTab(<DocumentsTab {...defaultProps()} />);
    expect(screen.queryByTestId('docs-row-error-d1')).toBeNull();
  });

  it('labels chunks "created before failure" when FAILED with chunks > 0', () => {
    // QA reported case: 327 chunks indexed before a downstream failure.
    // The chunks count stays a number (no relabel in the table) but the
    // cell carries a tooltip that names the partial state honestly.
    const failedWithChunks = {
      ...DOCUMENT_FIXTURES[2], // d3, the FAILED fixture
      doc_id: 'd3-with-chunks',
      chunks_count: 327,
    };
    const props = {
      ...defaultProps(),
      docs: [failedWithChunks],
    };
    renderTab(<DocumentsTab {...props} />);
    const cell = screen.getByTestId('docs-row-chunks-d3-with-chunks');
    expect(cell.textContent).toBe('327');
    expect(cell.getAttribute('title')).toBe(
      '327 chunks created before failure',
    );
  });

  it('omits the chunks tooltip on a FAILED row with zero chunks', () => {
    // d3 has chunks_count = 0 — no "created before failure" claim to make.
    renderTab(<DocumentsTab {...defaultProps()} />);
    const cell = screen.getByTestId('docs-row-chunks-d3');
    expect(cell.getAttribute('title')).toBeNull();
  });
});

describe('DocumentsTab — page pagination', () => {
  it('shows no pagination controls without backend totals or more pages', () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    expect(screen.queryByTestId('docs-pagination')).toBeNull();
  });

  it('renders page totals and calls next page', async () => {
    const onNextPage = vi.fn();
    renderTab(
      <DocumentsTab
        {...defaultProps()}
        currentPage={2}
        totalCount={125}
        hasNextPage
        onNextPage={onNextPage}
        onPreviousPage={vi.fn()}
      />,
    );
    expect(screen.getByTestId('docs-pagination')).toHaveTextContent('Page 2');
    expect(screen.getByTestId('docs-pagination')).toHaveTextContent('125 total');
    await userEvent.click(screen.getByTestId('docs-page-next'));
    expect(onNextPage).toHaveBeenCalledTimes(1);
  });

  it('disables page buttons while fetching', () => {
    renderTab(
      <DocumentsTab
        {...defaultProps()}
        currentPage={2}
        totalCount={125}
        hasNextPage
        isPageFetching
        onNextPage={vi.fn()}
        onPreviousPage={vi.fn()}
      />,
    );
    expect(screen.getByTestId('docs-page-prev')).toBeDisabled();
    expect(screen.getByTestId('docs-page-next')).toBeDisabled();
    expect(screen.getByTestId('docs-page-next')).toHaveTextContent('Loading');
  });
});
