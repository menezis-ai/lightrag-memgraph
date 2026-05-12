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
  type AddSourceAction,
  type FileUpload,
  type LinkedSource,
} from './AddSourceModal';
import { THESAURUS_FIXTURES, FORMAT_CATEGORY_FIXTURES } from '../fixtures';

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
  url: 'confluence.bnp/cib/runbooks',
  type: 'confluence',
};

function defaultProps() {
  return {
    open: true,
    thesaurus: THESAURUS_FIXTURES,
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

  it('renders the initial files and URLs', () => {
    render(
      <AddSourceModal
        {...defaultProps()}
        initialFiles={[sampleUploaded, sampleError]}
        initialUrls={[sampleConfluence]}
      />,
    );
    expect(screen.getByText('oracle-config-guide.pdf')).toBeInTheDocument();
    expect(screen.getByText('huge-archive.zip')).toBeInTheDocument();
    expect(
      screen.getByText('confluence.bnp/cib/runbooks'),
    ).toBeInTheDocument();
  });
});

describe('AddSourceModal — URL handling', () => {
  it('adds a URL on Enter, auto-detecting type', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByLabelText('URL input');
    await userEvent.type(input, 'sharepoint.bnp/cib/incidents{Enter}');
    expect(
      screen.getByText('sharepoint.bnp/cib/incidents'),
    ).toBeInTheDocument();
    // SourceIcon for sharepoint = `cloud`
    expect(document.querySelector('svg[data-icon="cloud"]')).not.toBeNull();
  });

  it('removes a URL on chip X click', async () => {
    render(
      <AddSourceModal {...defaultProps()} initialUrls={[sampleConfluence]} />,
    );
    const x = screen.getByRole('button', {
      name: 'Remove confluence.bnp/cib/runbooks',
    });
    await userEvent.click(x);
    expect(
      screen.queryByText('confluence.bnp/cib/runbooks'),
    ).toBeNull();
  });
});

describe('AddSourceModal — files', () => {
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
    // useModalA11y schedules a setTimeout(30ms) that auto-focuses the first
    // input in the modal (the URL input, which is positioned before the tag
    // input in the DOM). On slow CI runners, that timer fires mid-`type` and
    // steals our keystrokes. Wait past it before we type into the tag input.
    await new Promise((r) => setTimeout(r, 60));
    const input = screen.getByLabelText('Tag input');
    input.focus();
    await userEvent.type(input, 'rman');
    await waitFor(() =>
      expect(screen.getByTestId('tag-sugg-rman')).toBeInTheDocument(),
    );
  });

  it('Enter on tag input adds the first suggestion', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    await new Promise((r) => setTimeout(r, 60));
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
    expect(p.onClose).toHaveBeenCalled();
  });
});
