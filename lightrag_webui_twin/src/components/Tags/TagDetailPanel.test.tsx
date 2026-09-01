import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { TAG_CATEGORY_FIXTURES, TAG_FIXTURES } from '../../fixtures';
import { TagDetailPanel } from './TagDetailPanel';

function renderPanel(overrides: Partial<Parameters<typeof TagDetailPanel>[0]> = {}) {
  const tag = TAG_FIXTURES.find((candidate) => candidate.status !== 'pending-review')!;
  const props = {
    t: tag,
    allTags: TAG_FIXTURES,
    categories: TAG_CATEGORY_FIXTURES,
    onSelect: vi.fn(),
    onAction: vi.fn(),
    onCommit: vi.fn(),
    onNavigate: vi.fn(),
    canEdit: true,
    canSuggest: true,
    ...overrides,
  };
  render(<TagDetailPanel {...props} />);
  return props;
}

describe('TagDetailPanel', () => {
  it('owns document navigation for the selected tag', async () => {
    const props = renderPanel();
    await userEvent.click(
      screen.getByRole('button', { name: /View \d+ documents tagged/ }),
    );
    if (!props.t) throw new Error('fixture tag is required');
    expect(props.onNavigate).toHaveBeenCalledWith('documents', {
      tag: props.t.tag,
    });
  });

  it('keeps governance actions behind edit capability', async () => {
    const props = renderPanel();
    await userEvent.click(screen.getByRole('button', { name: 'Edit' }));
    expect(props.onAction).toHaveBeenCalledWith({ kind: 'edit', tag: props.t });

    renderPanel({ canEdit: false, canSuggest: false });
    expect(screen.getByText(/Palier 1 — read-only/)).toBeInTheDocument();
  });
});
