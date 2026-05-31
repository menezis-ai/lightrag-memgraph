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
 *   - row Retag button calls onOpenRetag(doc)
 *   - Clear button resets filters and emits a "Filters cleared" toast
 *   - empty state appears when filters match no doc
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DocumentsTab } from './DocumentsTab';
import {
  DOCUMENT_FIXTURES,
  THESAURUS_FIXTURES,
} from '../fixtures';

function defaultProps() {
  return {
    docs: DOCUMENT_FIXTURES,
    thesaurus: THESAURUS_FIXTURES,
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
    render(<DocumentsTab {...defaultProps()} />);
    expect(screen.getByText('Document management')).toBeInTheDocument();
    DOCUMENT_FIXTURES.forEach((d) => {
      expect(screen.getByText(d.file_path)).toBeInTheDocument();
    });
  });

  it('shows status counts in the filter pills', () => {
    render(<DocumentsTab {...defaultProps()} />);
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
});

describe('DocumentsTab — filters', () => {
  it('status filter narrows the visible rows', async () => {
    render(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: /^Failed \(1\)/ }));
    // Only the failed doc visible
    expect(screen.getByTestId('docs-row-d3')).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d1')).toBeNull();
  });

  it('search filters by source name', async () => {
    render(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'oracle');
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();
  });

  it('clicking a tag on a row adds it as a filter', async () => {
    render(<DocumentsTab {...defaultProps()} />);
    // d1 has tags rman + oracle; click the chip in the row
    const tagSpan = screen.getByTestId('row-tag-d1-rman');
    await userEvent.click(tagSpan);
    // Now the tag filter row should contain "rman" as a removable chip;
    // the row should still be visible since it has rman.
    expect(screen.getByTestId('docs-row-d1')).toBeInTheDocument();
    // d4 has no rman tag → filtered out
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();
  });

  it('empty state appears when filters match nothing', async () => {
    render(<DocumentsTab {...defaultProps()} />);
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'nope-no-match-zzz');
    expect(screen.getByTestId('docs-empty')).toBeInTheDocument();
  });
});

describe('DocumentsTab — selection + bulk', () => {
  it('toggling rows builds a selection and reveals the bulk bar', async () => {
    render(<DocumentsTab {...defaultProps()} />);
    await userEvent.click(screen.getByLabelText('Select oracle-restart-procedure.pdf'));
    const bulkBar = screen.getByRole('region', { name: 'Bulk actions' });
    expect(bulkBar).toBeInTheDocument();
    // "1 selected" copy inside the bulk bar (not the pipeline badge)
    expect(bulkBar.textContent).toMatch(/\b1\b/);
    expect(bulkBar.textContent).toMatch(/selected/);
  });

  it('Bulk Retag invokes onOpenBulkRetag with selected docs', async () => {
    const p = defaultProps();
    render(<DocumentsTab {...p} />);
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
    render(<DocumentsTab {...p} />);
    await userEvent.click(screen.getByRole('button', { name: /Add source/ }));
    expect(p.onOpenAdd).toHaveBeenCalled();
  });

  it('row Retag button calls onOpenRetag with the doc', async () => {
    const p = defaultProps();
    render(<DocumentsTab {...p} />);
    await userEvent.click(
      screen.getByLabelText('Retag oracle-restart-procedure.pdf'),
    );
    expect(p.onOpenRetag).toHaveBeenCalled();
    expect(p.onOpenRetag.mock.calls[0][0].doc_id).toBe('d1');
  });

  it('Clear button resets filters and emits a toast', async () => {
    const p = defaultProps();
    render(<DocumentsTab {...p} />);
    // Activate a filter first
    const searchBox = screen.getByLabelText('Search source');
    await userEvent.type(searchBox, 'oracle');
    await userEvent.click(screen.getByRole('button', { name: /Clear$/i }));
    expect(p.onAddToast).toHaveBeenCalled();
    expect(p.onAddToast.mock.calls[0][0]).toBe('Filters cleared');
  });
});
