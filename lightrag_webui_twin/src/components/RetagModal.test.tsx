/**
 * Unit tests for RetagModal.
 *
 * Behaviors under test:
 *   - returns null when not open
 *   - single-doc title, bulk-mode title
 *   - shows currently applied tags
 *   - shared/partial split in bulk mode
 *   - autocomplete suggestions filter by input
 *   - clicking a suggestion adds it to pendingAdd
 *   - clicking remove on a current tag stages a pendingRemove (line-through)
 *   - Apply button disabled when no changes
 *   - submit invokes onSubmit with the right RetagAction and closes
 *   - Cancel calls onClose
 *   - Esc inside the dialog calls onClose
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RetagModal, type RetagAction } from './RetagModal';
import type { Document } from '../types/document';
import { THESAURUS_FIXTURES } from '../fixtures';

function makeDoc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'd1',
    track_id: null,
    type: 'file',
    file_path: 'oracle-restart-procedure.pdf',
    content_summary: 'A doc',
    content_length: 100,
    tags: ['rman'],
    status: 'PROCESSED',
    chunks_count: 418,
    created_at: '2026-05-29T14:00:00Z',
    updated_at: '2026-05-29T14:00:00Z',
    error_msg: null,
    metadata: {},
    visibility: 'private',
    folder: 'default',
    ...overrides,
  };
}

describe('RetagModal — basic rendering', () => {
  it('returns null when not open', () => {
    const { container } = render(
      <RetagModal
        open={false}
        doc={makeDoc()}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('returns null when open but no doc/docs', () => {
    const { container } = render(
      <RetagModal
        open
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders single-doc title', () => {
    render(
      <RetagModal
        open
        doc={makeDoc()}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(screen.getByText('Retag document')).toBeInTheDocument();
  });

  it('renders bulk title when docs.length > 1', () => {
    const docs = [
      makeDoc({ doc_id: 'a' }),
      makeDoc({ doc_id: 'b' }),
      makeDoc({ doc_id: 'c' }),
    ];
    render(
      <RetagModal
        open
        docs={docs}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(screen.getByText('Retag 3 sources')).toBeInTheDocument();
  });
});

describe('RetagModal — tag list', () => {
  it('renders the currently-applied tags', () => {
    render(
      <RetagModal
        open
        doc={makeDoc({ tags: ['rman', 'oracle'] })}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(screen.getByText('Currently applied')).toBeInTheDocument();
    expect(screen.getByText('rman')).toBeInTheDocument();
    expect(screen.getByText('oracle')).toBeInTheDocument();
  });

  it('partials are shown separately in bulk mode', () => {
    const docs = [
      makeDoc({ doc_id: 'a', tags: ['rman', 'oracle'] }),
      makeDoc({ doc_id: 'b', tags: ['rman', 'incident'] }),
    ];
    render(
      <RetagModal
        open
        docs={docs}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(screen.getByText('Tags on all selected')).toBeInTheDocument();
    expect(screen.getByText(/On some selected/)).toBeInTheDocument();
    // rman = shared, oracle / incident = partial
    const partials = document.querySelectorAll('.tag-chips')[1];
    expect(partials.textContent).toContain('oracle');
    expect(partials.textContent).toContain('incident');
  });
});

describe('RetagModal — autocomplete & interactions', () => {
  it('shows initial suggestions (top 4) when input empty', () => {
    render(
      <RetagModal
        open
        doc={makeDoc({ tags: [] })}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    const sugRows = document.querySelectorAll('.autocomplete-row');
    expect(sugRows.length).toBe(4);
  });

  it('filters suggestions by input', async () => {
    render(
      <RetagModal
        open
        doc={makeDoc({ tags: [] })}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    const input = screen.getByLabelText('Tag input');
    await userEvent.type(input, 'oracle');
    // "oracle" is itself a tag + "rman" mentions Oracle in def — at least the
    // direct tag must appear; just check we have ≥1 row and it includes oracle.
    const sugRows = Array.from(document.querySelectorAll('.autocomplete-row'));
    expect(sugRows.length).toBeGreaterThan(0);
    expect(sugRows.some((r) => r.textContent?.includes('oracle'))).toBe(true);
  });

  it('clicking a suggestion adds it to pendingAdd', async () => {
    render(
      <RetagModal
        open
        doc={makeDoc({ tags: [] })}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    const sugg = screen.getByTestId('sugg-rman');
    await userEvent.click(sugg);
    // pendingAdd zone is the first .tag-chips above the input
    const chips = document.querySelectorAll('.tag-chip');
    expect(Array.from(chips).some((c) => c.textContent?.includes('rman'))).toBe(
      true,
    );
  });

  it('removing a current tag stages it (line-through opacity 0.45)', async () => {
    render(
      <RetagModal
        open
        doc={makeDoc({ tags: ['rman'] })}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    const removeBtn = screen.getByRole('button', { name: 'Remove rman' });
    await userEvent.click(removeBtn);
    const rmanSpan = removeBtn.closest('span[style]') as HTMLElement | null;
    // The styling is applied on the outer span wrapping the chip.
    expect(rmanSpan?.style.textDecoration).toBe('line-through');
  });
});

describe('RetagModal — submit & close', () => {
  it('Apply button is disabled with no changes', () => {
    render(
      <RetagModal
        open
        doc={makeDoc()}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {}}
        onSubmit={() => {}}
      />,
    );
    expect(screen.getByRole('button', { name: /Apply tag/ })).toBeDisabled();
  });

  it('submit invokes onSubmit with the RetagAction and onClose', async () => {
    const onSubmit = vi.fn<(a: RetagAction) => void>();
    const onClose = vi.fn();
    const d = makeDoc({ tags: [] });
    render(
      <RetagModal
        open
        doc={d}
        thesaurus={THESAURUS_FIXTURES}
        onClose={onClose}
        onSubmit={onSubmit}
      />,
    );
    await userEvent.click(screen.getByTestId('sugg-rman'));
    await userEvent.click(screen.getByRole('button', { name: /Apply tag/ }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const action = onSubmit.mock.calls[0][0];
    expect(action.bulk).toBe(false);
    expect(action.adds).toEqual(['rman']);
    expect(action.removes).toEqual([]);
    expect(action.primary.doc_id).toBe('d1');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Cancel calls onClose', async () => {
    const onClose = vi.fn();
    render(
      <RetagModal
        open
        doc={makeDoc()}
        thesaurus={THESAURUS_FIXTURES}
        onClose={onClose}
        onSubmit={() => {}}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalled();
  });

  it('Escape inside the dialog calls onClose', async () => {
    const onClose = vi.fn();
    render(
      <RetagModal
        open
        doc={makeDoc()}
        thesaurus={THESAURUS_FIXTURES}
        onClose={onClose}
        onSubmit={() => {}}
      />,
    );
    const dialog = screen.getByRole('dialog');
    dialog.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(onClose).toHaveBeenCalled();
  });

  it('Escape in tag input clears autocomplete without closing the dialog', async () => {
    const onClose = vi.fn();
    render(
      <RetagModal
        open
        doc={makeDoc()}
        thesaurus={THESAURUS_FIXTURES}
        onClose={onClose}
        onSubmit={() => {}}
      />,
    );
    const input = screen.getByLabelText('Tag input') as HTMLInputElement;
    input.focus();
    await userEvent.type(input, 'ora');
    expect(input.value).toBe('ora');
    await userEvent.keyboard('{Escape}');
    expect(input.value).toBe('');
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});
