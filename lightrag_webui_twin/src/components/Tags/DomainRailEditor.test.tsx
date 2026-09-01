import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DomainRailEditor, type DomainDraft } from './DomainRailEditor';

const DRAFT: readonly DomainDraft[] = [
  { key: 'database', id: 'database', label: 'Database', color: '#336699', existing: true },
  { key: 'draft-1', id: 'new-domain', label: 'New domain', color: '#5A7FB4', existing: false },
];

function renderEditor(overrides: Partial<Parameters<typeof DomainRailEditor>[0]> = {}) {
  const props = {
    draft: DRAFT,
    error: null,
    tagCounts: { database: 4 },
    removedDomainsWithTags: [],
    isSaving: false,
    onAdd: vi.fn(),
    onUpdate: vi.fn(),
    onRemove: vi.fn(),
    onCancel: vi.fn(),
    onSave: vi.fn(),
    ...overrides,
  };
  render(<DomainRailEditor {...props} />);
  return props;
}

describe('DomainRailEditor', () => {
  it('normalizes ids owned by a new draft before reporting the update', () => {
    const props = renderEditor();
    const id = screen.getByLabelText('New domain domain id');
    fireEvent.change(id, { target: { value: 'Risk Controls' } });
    expect(props.onUpdate).toHaveBeenLastCalledWith('draft-1', {
      id: 'risk-controls',
    });
  });

  it('surfaces destructive impact and disables actions while saving', () => {
    renderEditor({
      isSaving: true,
      removedDomainsWithTags: [
        { id: 'legacy', label: 'Legacy', color: '#333333', count: 3 },
      ],
    });
    expect(screen.getByText(/Legacy \(3\)/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Saving…' })).toBeDisabled();
  });
});
