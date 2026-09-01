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
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
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
  error: 'Exceeds 50 MB · ZIP format is not supported',
};
const sampleConfluence: LinkedSource = {
  url: 'knowledge.example.com/demo/runbooks',
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
      screen.queryByText('knowledge.example.com/demo/runbooks'),
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

  it('keeps the legacy DOM identical when catalogEnabled is false or omitted', () => {
    const omitted = render(<AddSourceModal {...defaultProps()} />);
    const legacyHtml = omitted.container.innerHTML;
    omitted.unmount();

    const explicit = render(
      <AddSourceModal {...defaultProps()} catalogEnabled={false} />,
    );
    expect(explicit.container.innerHTML).toBe(legacyHtml);
  });

  it('links to the Sources RAG grid when the catalogue is enabled', async () => {
    const onOpenLinkedSources = vi.fn();
    render(
      <AddSourceModal
        {...defaultProps()}
        catalogEnabled
        onOpenLinkedSources={onOpenLinkedSources}
      />,
    );

    expect(screen.queryByText('Coming soon')).toBeNull();
    expect(screen.queryByLabelText(/URL input \(disabled/i)).toBeNull();
    await userEvent.click(
      screen.getByRole('button', { name: 'Manage Sources RAG' }),
    );
    expect(onOpenLinkedSources).toHaveBeenCalledOnce();
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
  it('renders the per-file MIP classification select (no MIP by default), and no engine toggle', () => {
    render(
      <AddSourceModal {...defaultProps()} initialFiles={[sampleUploaded]} />,
    );

    const select = screen.getByLabelText(
      'Classification for oracle-config-guide.pdf',
    );
    expect(select).toBeInTheDocument();
    // Default = "no MIP" (empty value).
    expect(select).toHaveValue('');
    expect(
      screen.getByTestId('addsource-classification-oracle-config-guide.pdf'),
    ).toBe(select);
    const selectScope = within(select as HTMLElement);
    // C1/C2 only: C3 is query-restricted and C4 is rejected by policy.
    expect(
      selectScope.getByRole('option', { name: 'C1 · Public' }),
    ).toBeInTheDocument();
    expect(
      selectScope.getByRole('option', { name: 'C2 · Internal' }),
    ).toBeInTheDocument();
    expect(
      selectScope.queryByRole('option', { name: 'C3 · Confidential' }),
    ).not.toBeInTheDocument();
    expect(
      selectScope.queryByRole('option', { name: 'C4 · Secret' }),
    ).not.toBeInTheDocument();
    // The LightRAG/RAG1.5 engine toggle is NOT restored.
    expect(screen.queryByText('RAG 1.5')).not.toBeInTheDocument();
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
    expect(screen.getByText('ZIP format is not supported')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add 0 sources' })).toBeDisabled();
  });

  it('format tooltip stops claiming "coming soon" when images are live', async () => {
    const { buildFormatCategories } = await import(
      '../constants/formatCategories'
    );
    const floor = buildFormatCategories(undefined);
    expect(floor.find((c) => c.cat === 'Images')?.future).toBe(true);

    const live = buildFormatCategories(['png', 'jpg', 'jpeg']);
    const images = live.find((c) => c.cat === 'Images');
    expect(images?.future).toBeUndefined();
    expect(images?.fmts).toBe('PNG JPG JPEG');
    // Other rows untouched.
    expect(live.find((c) => c.cat === 'Documents')).toEqual(
      floor.find((c) => c.cat === 'Documents'),
    );
  });

  it('rejects images unless the backend advertises them (BNP 2026-07-20 report)', async () => {
    // Without extraUploadExtensions the deployment has no vision endpoint:
    // the modal must keep rejecting images honestly.
    const { unmount } = render(<AddSourceModal {...defaultProps()} />);
    let input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    await userEvent.upload(
      input,
      new File(['png payload'], 'diagram.png', { type: 'image/png' }),
    );
    expect(screen.getByText('PNG format is not supported')).toBeInTheDocument();
    unmount();

    // A vision-enabled backend advertises the image extensions via runtime
    // config — the SAME file becomes uploadable.
    render(
      <AddSourceModal
        {...defaultProps()}
        extraUploadExtensions={['png', 'jpg', 'jpeg']}
        extraUploadMaxBytes={{
          jpeg: 20 * 1024 * 1024,
          jpg: 20 * 1024 * 1024,
          png: 20 * 1024 * 1024,
        }}
      />,
    );
    input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    await userEvent.upload(
      input,
      new File(['png payload'], 'diagram.png', { type: 'image/png' }),
    );
    expect(screen.getByText('diagram.png')).toBeInTheDocument();
    expect(screen.queryByText('PNG format is not supported')).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Add 1 source' }),
    ).not.toBeDisabled();
  });

  it('rejects an image above the advertised vision cap before upload', async () => {
    render(
      <AddSourceModal
        {...defaultProps()}
        extraUploadExtensions={['png', 'jpg', 'jpeg']}
        extraUploadMaxBytes={{ png: 20 * 1024 * 1024 }}
      />,
    );
    const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    const oversized = new File(['png payload'], 'oversized.png', {
      type: 'image/png',
    });
    Object.defineProperty(oversized, 'size', {
      value: 21 * 1024 * 1024,
    });

    await userEvent.upload(input, oversized);

    expect(screen.getByText('Exceeds 20 MB')).toBeInTheDocument();
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

  it('submits every file from a large batch upload', async () => {
    const onSubmit = vi.fn<(a: AddSourceAction) => void>();
    render(<AddSourceModal {...defaultProps()} onSubmit={onSubmit} />);
    const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    const batch = Array.from(
      { length: 12 },
      (_, i) =>
        new File([`payload-${i}`], `batch-${i + 1}.txt`, {
          type: 'text/plain',
          lastModified: i + 1,
        }),
    );

    fireEvent.change(input, { target: { files: batch } });
    expect(screen.getByText('(12 added)')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Add 12 sources' }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const action = onSubmit.mock.calls[0][0];
    expect(action.rawFiles).toHaveLength(12);
    expect(action.fileOptions).toHaveLength(12);
    expect(action.rawFiles.map((file) => file.name)).toEqual(
      batch.map((file) => file.name),
    );
  });

  it('preserves folder-relative paths for colliding basenames', async () => {
    const onSubmit = vi.fn<(a: AddSourceAction) => void>();
    render(<AddSourceModal {...defaultProps()} onSubmit={onSubmit} />);
    const input = screen.getByTestId('addsource-folder-input') as HTMLInputElement;
    const first = new File(['a'], 'report.pdf', { type: 'application/pdf' });
    const second = new File(['b'], 'report.pdf', { type: 'application/pdf' });
    Object.defineProperty(first, 'webkitRelativePath', {
      value: 'root/team-a/report.pdf',
    });
    Object.defineProperty(second, 'webkitRelativePath', {
      value: 'root/team-b/report.pdf',
    });

    await userEvent.upload(input, [first, second]);

    expect(screen.getByText('root/team-a/report.pdf')).toBeInTheDocument();
    expect(screen.getByText('root/team-b/report.pdf')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Add 2 sources' }));
    expect(onSubmit.mock.calls[0][0].rawFiles).toHaveLength(2);
    expect(onSubmit.mock.calls[0][0].fileOptions.map((item) => item.relativePath))
      .toEqual(['root/team-a/report.pdf', 'root/team-b/report.pdf']);
  });

  it('offers explicit cancellation while a batch is uploading', async () => {
    const onCancel = vi.fn();
    render(
      <AddSourceModal
        {...defaultProps()}
        submitting
        onCancel={onCancel}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Cancel upload' }));
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it('keeps same-name files as separate upload payloads', async () => {
    const onSubmit = vi.fn<(a: AddSourceAction) => void>();
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      render(<AddSourceModal {...defaultProps()} onSubmit={onSubmit} />);
      const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
      const batch = [
        new File(['first'], 'runbook.md', {
          type: 'text/markdown',
          lastModified: 1,
        }),
        new File(['second'], 'runbook.md', {
          type: 'text/markdown',
          lastModified: 2,
        }),
      ];

      fireEvent.change(input, { target: { files: batch } });
      expect(screen.getByText('(2 added)')).toBeInTheDocument();
      expect(consoleError).not.toHaveBeenCalledWith(
        expect.stringContaining('Encountered two children with the same key'),
        expect.anything(),
        expect.anything(),
      );
      const classificationSelects = screen.getAllByTestId(
        'addsource-classification-runbook.md',
      );
      fireEvent.change(classificationSelects[0], { target: { value: 'C2' } });

      fireEvent.click(screen.getByRole('button', { name: 'Add 2 sources' }));

      expect(onSubmit).toHaveBeenCalledTimes(1);
      const action = onSubmit.mock.calls[0][0];
      expect(action.rawFiles).toHaveLength(2);
      expect(action.rawFiles).toEqual(batch);
      expect(action.fileOptions).toEqual([
        { name: 'runbook.md', classification: 'C2' },
        { name: 'runbook.md' },
      ]);
    } finally {
      consoleError.mockRestore();
    }
  });

  it('removes only the selected same-name file row without leaking its options', async () => {
    const onSubmit = vi.fn<(a: AddSourceAction) => void>();
    render(<AddSourceModal {...defaultProps()} onSubmit={onSubmit} />);
    const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    const first = new File(['first'], 'runbook.md', {
      type: 'text/markdown',
      lastModified: 1,
    });
    const second = new File(['second'], 'runbook.md', {
      type: 'text/markdown',
      lastModified: 2,
    });

    fireEvent.change(input, { target: { files: [first, second] } });
    const classificationSelects = screen.getAllByTestId(
      'addsource-classification-runbook.md',
    );
    fireEvent.change(classificationSelects[0], { target: { value: 'C2' } });
    fireEvent.click(screen.getAllByRole('button', { name: 'Remove runbook.md' })[0]);

    expect(screen.getByText('(1 added)')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Add 1 source' }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const action = onSubmit.mock.calls[0][0];
    expect(action.rawFiles).toEqual([second]);
    expect(action.fileOptions).toEqual([{ name: 'runbook.md' }]);
  });
});

describe('AddSourceModal — tag autocomplete', () => {
  it('does not show autocomplete with empty input', () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByRole('combobox', { name: 'Tag input' });
    expect(document.querySelector('.autocomplete-row')).toBeNull();
    expect(input).toHaveAttribute('aria-expanded', 'false');
    expect(input).not.toHaveAttribute('aria-controls');
    expect(input).not.toHaveAttribute('aria-activedescendant');
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

  it('Arrow keys navigate tag suggestions and Enter picks the focused item', async () => {
    render(<AddSourceModal {...defaultProps()} />);
    const input = screen.getByLabelText('Tag input');
    input.focus();
    await userEvent.type(input, 'r');

    await waitFor(() => {
      const options = screen.getAllByRole('option');
      expect(options.length).toBeGreaterThan(1);
    });
    const options = screen.getAllByRole('option');
    const targetTag = options[1].getAttribute('data-testid')?.replace('tag-sugg-', '');
    if (!targetTag) {
      throw new Error('Expected a second tag suggestion option with test id');
    }

    await userEvent.keyboard('{ArrowDown}');
    expect(options[1]).toHaveClass('autocomplete-row focus');

    await userEvent.keyboard('{Enter}');
    await waitFor(() => {
      expect(screen.getByText(targetTag)).toBeInTheDocument();
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

  it('Submit emits the AddSourceAction; the host owns closing (modal stays open during upload)', async () => {
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
    // Fixture-only rows carry no browser File and therefore cannot create a
    // shifted fileOptions/rawFiles pair.
    expect(action.rawFiles).toEqual([]);
    expect(action.fileOptions).toEqual([]);
    // submit no longer self-closes — the host keeps the modal open during the
    // upload and closes it when the mutation settles.
    expect(p.onClose).not.toHaveBeenCalled();
  });

  it('renders host-reported state and error for each submitted file', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} />);
    const input = screen.getByTestId('addsource-file-input') as HTMLInputElement;
    await userEvent.upload(
      input,
      new File(['payload'], 'retry.md', { type: 'text/markdown' }),
    );
    await userEvent.click(screen.getByRole('button', { name: /Add 1 source/ }));
    const action = p.onSubmit.mock.calls[0][0];

    act(() => action.onFileStateChange?.(0, 'uploading'));
    expect(screen.getByText(/0%/)).toBeInTheDocument();

    act(() => action.onFileStateChange?.(0, 'error', 'Network interrupted'));
    expect(screen.getByText('Network interrupted')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Add 1 source/ })).toBeEnabled();
  });

  it('derives raw files, options and callbacks from one aligned record list', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} initialFiles={[sampleUploaded]} />);
    const input = screen.getByTestId('addsource-folder-input') as HTMLInputElement;
    const first = new File(['a'], 'first.pdf', { type: 'application/pdf' });
    const second = new File(['b'], 'second.pdf', { type: 'application/pdf' });
    Object.defineProperty(first, 'webkitRelativePath', {
      value: 'root/a/first.pdf',
    });
    Object.defineProperty(second, 'webkitRelativePath', {
      value: 'root/b/second.pdf',
    });
    await userEvent.upload(input, [first, second]);

    await userEvent.click(screen.getByRole('button', { name: /Add 3 sources/ }));
    const action = p.onSubmit.mock.calls[0][0];
    expect(action.rawFiles).toEqual([first, second]);
    expect(action.fileOptions.map((item) => item.relativePath)).toEqual([
      'root/a/first.pdf',
      'root/b/second.pdf',
    ]);

    act(() => action.onFileStateChange?.(0, 'error', 'first failed'));
    const firstRow = screen.getByText('root/a/first.pdf').closest('.file-row');
    const secondRow = screen.getByText('root/b/second.pdf').closest('.file-row');
    expect(firstRow).toHaveClass('error');
    expect(secondRow).not.toHaveClass('error');
  });

  it('blocks close (X / Cancel / backdrop) and disables submit while submitting', async () => {
    const p = defaultProps();
    render(
      <AddSourceModal
        {...p}
        initialFiles={[sampleUploaded]}
        submitting
      />,
    );
    const x = screen.getByRole('button', { name: 'Close dialog' });
    const cancel = screen.getByRole('button', { name: 'Cancel' });
    expect(x).toBeDisabled();
    expect(cancel).toBeDisabled();
    expect(screen.getByRole('button', { name: /Uploading/ })).toBeDisabled();
    // Clicking the (disabled) X must not fire onClose.
    await userEvent.click(x);
    expect(p.onClose).not.toHaveBeenCalled();
    // Backdrop click is also neutralised while uploading.
    await userEvent.click(screen.getByTestId('addsource-backdrop'));
    expect(p.onClose).not.toHaveBeenCalled();
  });

  it('flows the selected per-file MIP classification into the emitted fileOptions', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} />);
    await userEvent.upload(
      screen.getByTestId('addsource-file-input'),
      new File(['payload'], 'oracle-config-guide.pdf', {
        type: 'application/pdf',
      }),
    );

    await userEvent.selectOptions(
      screen.getByLabelText('Classification for oracle-config-guide.pdf'),
      'C2',
    );
    await userEvent.click(screen.getByRole('button', { name: /Add 1 source/ }));

    expect(p.onSubmit.mock.calls[0][0].fileOptions).toEqual([
      {
        name: 'oracle-config-guide.pdf',
        classification: 'C2',
      },
    ]);
  });

  it('applies a C1/C2 bulk sensitivity only to uploadable files', async () => {
    const p = defaultProps();
    const errorFile = {
      ...sampleUploaded,
      name: 'huge-archive.zip',
      state: 'error' as const,
      error: 'Exceeds 50 MB',
    };
    render(
      <AddSourceModal
        {...p}
        initialFiles={[errorFile]}
      />,
    );
    await userEvent.upload(screen.getByTestId('addsource-file-input'), [
      new File(['pdf'], 'oracle-config-guide.pdf', {
        type: 'application/pdf',
      }),
      new File(['text'], 'unix-notes.txt', { type: 'text/plain' }),
    ]);

    await userEvent.selectOptions(
      screen.getByLabelText('Sensitivity for all files'),
      'C2',
    );

    expect(
      screen.getByLabelText('Classification for oracle-config-guide.pdf'),
    ).toHaveValue('C2');
    expect(screen.getByLabelText('Classification for unix-notes.txt')).toHaveValue(
      'C2',
    );
    expect(
      screen.queryByLabelText('Classification for huge-archive.zip'),
    ).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Add 2 source/ }));

    expect(p.onSubmit.mock.calls[0][0].fileOptions).toEqual([
      { name: 'oracle-config-guide.pdf', classification: 'C2' },
      { name: 'unix-notes.txt', classification: 'C2' },
    ]);
  });
});

describe('AddSourceModal — document type selector', () => {
  it('defaults to auto-detect and omits docType from the emitted action', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} initialFiles={[sampleUploaded]} />);

    expect(screen.getByTestId('addsource-doc-type')).toHaveValue('');

    await userEvent.click(screen.getByRole('button', { name: /Add 1 source/ }));

    const action = p.onSubmit.mock.calls[0][0];
    // Auto-detect must OMIT the key entirely — the host only sends the
    // X-Twin-Doc-Type header when docType is present.
    expect('docType' in action).toBe(false);
  });

  it('emits docType "procedure" when the operator forces the procedure profile', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} initialFiles={[sampleUploaded]} />);

    await userEvent.selectOptions(
      screen.getByLabelText('Document type'),
      'procedure',
    );
    await userEvent.click(screen.getByRole('button', { name: /Add 1 source/ }));

    expect(p.onSubmit.mock.calls[0][0].docType).toBe('procedure');
  });

  it('emits docType "standard" when procedure detection is bypassed', async () => {
    const p = defaultProps();
    render(<AddSourceModal {...p} initialFiles={[sampleUploaded]} />);

    await userEvent.selectOptions(
      screen.getByTestId('addsource-doc-type'),
      'standard',
    );
    await userEvent.click(screen.getByRole('button', { name: /Add 1 source/ }));

    expect(p.onSubmit.mock.calls[0][0].docType).toBe('standard');
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
