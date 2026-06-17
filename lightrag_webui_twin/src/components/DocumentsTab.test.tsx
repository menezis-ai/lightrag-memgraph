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
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DocumentsTab } from './DocumentsTab';
import {
  DOCUMENT_FIXTURES,
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

  it('search filters by source name', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'oracle');
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();
  });

  it('status counts are scoped by the active search filter', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'oracle');

    expect(screen.getByRole('button', { name: /^All \(2\)/ })).toBeInTheDocument();
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

  it('status counts are scoped by the active tag filter', async () => {
    renderTab(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('row-tag-d1-rman'));

    expect(screen.getByRole('button', { name: /^All \(2\)/ })).toBeInTheDocument();
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
      name: /Re-process failed sources/,
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
      name: /Re-process failed sources/,
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
      name: /Re-process failed sources/,
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

  it('source filename opens the document detail callback', async () => {
    const p = { ...defaultProps(), onOpenDetail: vi.fn() };
    renderTab(<DocumentsTab {...p} />);
    await userEvent.click(screen.getByTestId('docs-row-filename-d1'));
    expect(p.onOpenDetail).toHaveBeenCalledTimes(1);
    expect(p.onOpenDetail.mock.calls[0][0].doc_id).toBe('d1');
  });

});

describe('DocumentsTab — failed row surfaces error_msg (TR-ING-01)', () => {
  it('renders the indexing failure reason inline on a FAILED row', () => {
    // d3 = the FAILED fixture with error_msg='Unsupported MIME type: …'.
    // Before this PR the row went red without ever exposing the reason —
    // the operator's exact complaint from QA.
    renderTab(<DocumentsTab {...defaultProps()} />);
    const err = screen.getByTestId('docs-row-error-d3');
    expect(err.textContent).toMatch(/indexing failed/i);
    expect(err.textContent).toContain('Unsupported MIME type: application/zip');
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
