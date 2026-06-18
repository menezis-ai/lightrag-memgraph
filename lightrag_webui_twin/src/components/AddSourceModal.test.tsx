/**
 * Unit tests for AddSourceModal.
 *
 * Behaviors under test:
 *   - returns null when not open
 *   - renders title + dropzone
 *   - initialFiles / initialUrls render in the file & URL lists
 *   - URL input + Enter adds a linked source with correct type detection
 *   - remove URL & remove file remove the corresponding entry
 *   - tag autocomplete: input filters; Enter adds first match
 *   - Cancel calls onClose
 *   - Submit emits AddSourceAction with readyCount
 *   - Submit button disabled when nothing ready
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  AddSourceModal,
  formatFileSize,
  type AddSourceAction,
  type FileUpload,
  type LinkedSource,
} from './AddSourceModal';
import { TAG_FIXTURES, FORMAT_CATEGORY_FIXTURES } from '../fixtures';

const sampleUploaded: FileUpload = {
  name: 'oracle-config-guide.pdf',
  size: 2.3,
  state: 'uploaded',
};
const sampleError: FileUpload = {
  name: 'huge-archive.zip',
  size: 68,
  state: 'error',
  error: 'Exceeds 50 MB · unsupported type',
};
const sampleConfluence: LinkedSource = {
  url: 'confluence.corp/cib/runbooks',
  type: 'confluence',
};

function defaultProps() {
  return {
    open: true,
    tagCatalog: TAG_FIXTURES,
    formatCategories: FORMAT_CATEGORY_FIXTURES,
    onClose: vi.fn(),
    onSubmit: vi.fn<(a: AddSourceAction) => void>(),
  };
}

describe('AddSourceModal — basic rendering', () => {
  it('returns null when not open', () => {
    const { container } = render(
      <AddSourceModal {...defaultProps()} open={false} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders the title + dropzone', () => {
    render(<AddSourceModal {...defaultProps()} />);
    expect(screen.getByText('Add source')).toBeInTheDocument();
    expect(
      screen.getByText('Drop files here or click to browse'),
    ).toBeInTheDocument();
  });

  it('renders the initial files (URL chips are gated — see Coming soon)', () => {
    render(
      <AddSourceModal
        {...defaultProps()}
        initialFiles={[sampleUploaded, sampleError]}
        initialUrls={[sampleConfluence]}
      />,
    );
    expect(screen.getByText('oracle-config-guide.pdf')).toBeInTheDocument();
    expect(screen.getByText('huge-archive.zip')).toBeInTheDocument();
    // initialUrls is preserved in state and submitted, but its UI chip
    // doesn't render while the linked-sources block is gated.
    expect(
      screen.queryByText('confluence.corp/cib/runbooks'),
    ).toBeNull();
  });
});

describe('AddSourceModal — Linked sources (Coming soon)', () => {
  // The linked-sources block is gated until the RAG 1.5 connector
  // (upstream RAG team) ships its API. Until then, the input is
  // disabled and a "Coming soon" pill is rendered next to the label.
  it('renders the Coming soon pill and a disabled input', () => {
    render(<AddSourceModal {...defaultProps()} />);
    expect(screen.getByText('Coming soon')).toBeInTheDocument();
    const input = screen.getByLabelText(
      'URL input (disabled — coming soon)',
    ) as HTMLInputElement;
    expect(input.disabled).toBe(true);
  });

  it('still forwards initialUrls to onSubmit (state preserved while UI is gated)', async () => {
    const onSubmit = vi.fn();
    render(
      <AddSourceModal
        {...defaultProps()}
        initialUrls={[sampleConfluence]}
        initialFiles={[sampleUploaded]}
        onSubmit={onSubmit}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: /add 2/i }));
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ urls: [sampleConfluence] }),
    );
  });
});

describe('AddSourceModal — files', () => {
  it('renders per-file MIP and RAG engine controls with no MIP by default', () => {
    render(
      <AddSourceModal {...defaultProps()} initialFiles={[sampleUploaded]} />,
    );

    expect(
      screen.getByLabelText('Classification for oracle-config-guide.pdf'),
    ).toHaveValue('');
    expect(
      screen.getByTestId('addsource-rag-lightrag-oracle-config-guide.pdf'),
    ).toHaveClass('active');
    expect(
      screen.getByTestId('addsource-rag15-oracle-config-guide.pdf'),
    ).toBeDisabled();
  });

  it('removes a file when its X is clicked', async () => {
    render(
      <AddSourceModal {...defaultProps()} initialFiles={[sampleUploaded]} />,
    );
    const x = screen.getByRole('button', {
      name: 'Remove oracle-config-guide.pdf',
    });
    await userEvent.click(x);
    expect(screen.queryByText('oracle-config-guide.pdf')).toBeNull();
  });

  it('marks unsupported files as errors and excludes them from the ready count', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    await userEvent.upload(
      input,
      new File(['zip payload'], 'unsupported.zip', { type: 'application/zip' }),
    );
    expect(screen.getByText('unsupported.zip')).toBeInTheDocument();
    expect(screen.getByText('unsupported type')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add 0 sources' })).toBeDisabled();
  });

  it('marks oversized files as errors and excludes them from the ready count', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    const oversized = new File(['pdf payload'], 'oversized.pdf', {
      type: 'application/pdf',
    });
    Object.defineProperty(oversized, 'size', {
      value: 51 * 1024 * 1024,
    });

    await userEvent.upload(input, oversized);
    expect(screen.getByText('oversized.pdf')).toBeInTheDocument();
    expect(screen.getByText('Exceeds 50 MB')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add 0 sources' })).toBeDisabled();
  });
});

describe('AddSourceModal — tag autocomplete', () => {
  it('does not show autocomplete with empty input', () => {
    render(<AddSourceModal {...defaultProps()} />);
    expect(
      document.querySelector('.autocomplete-row'),
    ).toBeNull();
  });

  it('shows autocomplete rows filtered by input', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByLabelText('Tag input');
    input.focus();
    await userEvent.type(input, 'rman');
    await waitFor(() =>
      expect(screen.getByTestId('tag-sugg-rman')).toBeInTheDocument(),
    );
  });

  it('Enter on tag input adds the first suggestion', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByLabelText('Tag input');
    input.focus();
    await userEvent.type(input, 'rman');
    await userEvent.keyboard('{Enter}');
    await waitFor(() => {
      const chips = document.querySelectorAll('.tag-chip');
      expect(
        Array.from(chips).some((c) => c.textContent?.includes('rman')),
      ).toBe(true);
    });
  });

  it('Escape in tag input clears autocomplete without closing the modal', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} />);
    const input = screen.getByLabelText('Tag input') as HTMLInputElement;
    input.focus();
    await userEvent.type(input, 'rman');
    await waitFor(() =>
      expect(screen.getByTestId('tag-sugg-rman')).toBeInTheDocument(),
    );
    await userEvent.keyboard('{Escape}');
    expect(input.value).toBe('');
    expect(p.onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});

describe('AddSourceModal — submit & close', () => {
  it('Add button is disabled when nothing ready', () => {
    render(<AddSourceModal {...defaultProps()} />);
    expect(screen.getByRole('button', { name: /Add 0/ })).toBeDisabled();
  });

  it('Cancel calls onClose', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(p.onClose).toHaveBeenCalled();
  });

  it('Submit emits the AddSourceAction and closes', async () => {
    const p = defaultProps();
    render(
      <AddSourceModal
        {...p}
        initialFiles={[sampleUploaded]}
        initialUrls={[sampleConfluence]}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: /Add 2 sources/ }));
    expect(p.onSubmit).toHaveBeenCalledTimes(1);
    const action = p.onSubmit.mock.calls[0][0];
    expect(action.readyCount).toBe(2);
    expect(action.files).toHaveLength(1);
    expect(action.urls).toHaveLength(1);
    expect(action.tags).toEqual([]);
    expect(action.fileOptions).toEqual([
      {
        name: 'oracle-config-guide.pdf',
        ragEngine: 'lightrag',
      },
    ]);
    expect(p.onClose).toHaveBeenCalled();
  });

  it('submits the selected per-file MIP classification', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} initialFiles={[sampleUploaded]} />);

    await userEvent.selectOptions(
      screen.getByLabelText('Classification for oracle-config-guide.pdf'),
      'confidential',
    );
    await userEvent.click(screen.getByRole('button', { name: /Add 1 source/ }));

    expect(p.onSubmit.mock.calls[0][0].fileOptions).toEqual([
      {
        name: 'oracle-config-guide.pdf',
        classification: 'confidential',
        ragEngine: 'lightrag',
      },
    ]);
  });
});

describe('formatFileSize', () => {
  it('renders bytes for tiny payloads', () => {
    expect(formatFileSize(512)).toBe('512 B');
  });

  it('renders KB (no decimals) for sub-MB payloads — no more "0 MB"', () => {
    // 47 KB JSON file: was displayed as "0 MB" before the fix.
    expect(formatFileSize(47_000)).toBe('46 KB');
  });

  it('renders MB (1 decimal) above one MB', () => {
    expect(formatFileSize(2_400_000)).toBe('2.3 MB');
  });

  it('handles zero + negative + NaN safely', () => {
    expect(formatFileSize(0)).toBe('0 B');
    expect(formatFileSize(-1)).toBe('0 B');
    expect(formatFileSize(NaN)).toBe('0 B');
  });
});
