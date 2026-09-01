import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { TAG_CATEGORY_FIXTURES, TAG_FIXTURES } from '../../fixtures';
import { TagsGridEmpty } from './TagEmptyStates';

describe('TagsGridEmpty', () => {
  it('distinguishes an empty catalog from a filtered result', async () => {
    const onRequest = vi.fn();
    const { rerender } = render(
      <TagsGridEmpty
        totalActive={0}
        q=""
        selectedCat="all"
        selectedStatus="all"
        categories={TAG_CATEGORY_FIXTURES}
        suggestions={[]}
        canSuggest
        onClear={() => {}}
        onPickTag={() => {}}
        onRequest={onRequest}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: /Request the first tag/ }));
    expect(onRequest).toHaveBeenCalledOnce();

    rerender(
      <TagsGridEmpty
        totalActive={4}
        q="missing"
        selectedCat="all"
        selectedStatus="all"
        categories={TAG_CATEGORY_FIXTURES}
        suggestions={[TAG_FIXTURES[0]]}
        canSuggest
        onClear={() => {}}
        onPickTag={() => {}}
        onRequest={() => {}}
      />,
    );
    expect(screen.getByTestId('tags-empty-filtered')).toBeInTheDocument();
    expect(screen.getByText('search: "missing"')).toBeInTheDocument();
  });
});
