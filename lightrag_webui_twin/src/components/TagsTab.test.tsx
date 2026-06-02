/**
 * Unit tests for TagsTab + TagActionModal.
 *
 * Covers: header counts, pending section, category rail, search/status
 * filters, empty-zero + empty-filtered states, card selection, related
 * chip navigation, palier-gated detail actions, modal dispatch (8 kinds),
 * commit payloads, exportThesaurusJson helper.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import {
  render as rtlRender,
  screen,
  within,
  type RenderOptions,
} from '@testing-library/react';
import type { ReactElement, ReactNode } from 'react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TagsTab, exportThesaurusJson } from './TagsTab';
import { TAG_CATEGORY_FIXTURES, TAG_FIXTURES } from '../fixtures';
import type { TagCurrentUser } from '../types/tag';

// TagsTab now consumes useImportCategories() which calls useQueryClient().
// Every render must therefore sit inside a QueryClientProvider. We shadow
// the bare render() so the existing test bodies stay identical.
function render(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
): ReturnType<typeof rtlRender> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return rtlRender(ui, { wrapper: Wrapper, ...options });
}

const PALIER3: TagCurrentUser = {
  name: 'claire.benoit',
  palier: 3,
  role: 'admin / steward',
};
const PALIER2: TagCurrentUser = { name: 'yann.dubois', palier: 2, role: 'steward' };
const PALIER1: TagCurrentUser = { name: 'marc.berthier', palier: 1, role: 'reader' };

function defaultProps(user: TagCurrentUser = PALIER3) {
  return {
    tags: TAG_FIXTURES,
    categories: TAG_CATEGORY_FIXTURES,
    currentUser: user,
    onApprove: vi.fn(),
    onCommit: vi.fn(),
    onNavigate: vi.fn(),
  };
}

beforeEach(() => {
  window.history.replaceState(null, '', '/');
});
afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('TagsTab — rendering', () => {
  it('renders header with active + pending counts; no palier pill', () => {
    render(<TagsTab {...defaultProps()} />);
    expect(screen.getByRole('heading', { name: 'Tags' })).toBeInTheDocument();
    // 21 fixtures - 2 requested = 19 active
    const sub = document.querySelector('.tags-sub') as HTMLElement;
    expect(sub.textContent).toMatch(/19 active tags · 2 pending requests/);
    // palier-pill killed per 30/05 cleanup — role lives in JWT, not in chrome
    expect(sub.textContent).not.toMatch(/palier 3/);
    expect(document.querySelector('.palier-pill')).toBeNull();
  });

  it('renders the pending requests section with Approve/Edit-approve/Reject for palier 3', () => {
    render(<TagsTab {...defaultProps()} />);
    expect(screen.getByText('Tag requests')).toBeInTheDocument();
    expect(document.querySelector('.pending-counts')?.textContent).toMatch(
      /2 tag requests awaiting review/,
    );
    expect(screen.getByTestId('pending-argocd')).toBeInTheDocument();
    expect(screen.getByTestId('pending-pacs008')).toBeInTheDocument();
    const argocd = screen.getByTestId('pending-argocd');
    expect(within(argocd).getByRole('button', { name: 'Approve' })).toBeInTheDocument();
    expect(
      within(argocd).getByRole('button', { name: 'Edit & approve' }),
    ).toBeInTheDocument();
    expect(within(argocd).getByRole('button', { name: 'Reject' })).toBeInTheDocument();
  });

  it('palier 2 sees pending section but only "Awaiting reviewer approval" caption', () => {
    render(<TagsTab {...defaultProps(PALIER2)} />);
    const argocd = screen.getByTestId('pending-argocd');
    expect(within(argocd).queryByRole('button', { name: 'Approve' })).toBeNull();
    expect(within(argocd).getByText('Awaiting reviewer approval')).toBeInTheDocument();
  });

  it('palier 1 does NOT see the pending section at all', () => {
    render(<TagsTab {...defaultProps(PALIER1)} />);
    expect(screen.queryByTestId('pending-argocd')).toBeNull();
  });

  it('renders all category rail rows + counts', () => {
    render(<TagsTab {...defaultProps()} />);
    expect(screen.getByTestId('rail-all')).toBeInTheDocument();
    TAG_CATEGORY_FIXTURES.forEach((c) => {
      expect(screen.getByTestId(`rail-${c.id}`)).toBeInTheDocument();
    });
  });
});

describe('TagsTab — filters', () => {
  it('clicking a category narrows the visible cards', async () => {
    render(<TagsTab {...defaultProps()} />);
    expect(screen.getByTestId('tag-card-rman')).toBeInTheDocument();
    expect(screen.getByTestId('tag-card-swift')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('rail-payment'));
    expect(screen.queryByTestId('tag-card-rman')).toBeNull();
    expect(screen.getByTestId('tag-card-swift')).toBeInTheDocument();
  });

  it('search filter narrows by tag name + def + aliases', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search tags'), 'rman');
    expect(screen.getByTestId('tag-card-rman')).toBeInTheDocument();
    expect(screen.queryByTestId('tag-card-swift')).toBeNull();
  });

  it('search match on alias (e.g. "recovery-manager" → rman)', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search tags'), 'recovery-manager');
    expect(screen.getByTestId('tag-card-rman')).toBeInTheDocument();
    expect(screen.queryByTestId('tag-card-oracle')).toBeNull();
  });

  it('status filter narrows to deprecated / pending-promotion only', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.selectOptions(
      screen.getByLabelText('Status filter'),
      'pending-promotion',
    );
    expect(screen.getByTestId('tag-card-graphrag')).toBeInTheDocument();
    expect(screen.queryByTestId('tag-card-rman')).toBeNull();
  });

  it('empty filtered state appears when no tag matches the query', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search tags'), 'zzz-no-tag');
    expect(screen.getByTestId('tags-empty-filtered')).toBeInTheDocument();
    // Suggestion chips visible
    expect(screen.getAllByText(/docs$/).length).toBeGreaterThan(0);
  });

  it('Clear filters CTA from the empty state restores the grid', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search tags'), 'zzz-no-tag');
    await userEvent.click(
      within(screen.getByTestId('tags-empty-filtered')).getByRole('button', {
        name: 'Clear filters',
      }),
    );
    expect(screen.getByTestId('tag-card-rman')).toBeInTheDocument();
  });

  it('empty-zero state appears when the workspace has 0 active tags', () => {
    render(
      <TagsTab
        {...defaultProps()}
        tags={TAG_FIXTURES.filter((t) => t.tier === 'requested')}
      />,
    );
    expect(screen.getByTestId('tags-empty-zero')).toBeInTheDocument();
  });
});

describe('TagsTab — selection + detail', () => {
  it('clicking a card updates the detail panel', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('tag-card-swift'));
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/SWIFT messaging/);
    expect(detail.textContent).toMatch(/542/); // chunks_count = 542
  });

  it('clicking a related chip in the detail navigates selection', async () => {
    render(<TagsTab {...defaultProps()} />);
    // Detail starts on rman (first fixture). Related contains oracle.
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByTestId('related-oracle'));
    expect(detail.textContent).toMatch(/Oracle Database engine/);
  });

  it('palier 1 detail panel is read-only (shows muted hint)', () => {
    render(<TagsTab {...defaultProps(PALIER1)} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/Palier 1 — read-only/);
    expect(within(detail).queryByRole('button', { name: 'Edit' })).toBeNull();
  });

  it('palier 2 detail panel offers Suggest edit but no destructive actions', () => {
    render(<TagsTab {...defaultProps(PALIER2)} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    expect(within(detail).getByRole('button', { name: 'Suggest edit' })).toBeInTheDocument();
    expect(within(detail).queryByRole('button', { name: 'Delete' })).toBeNull();
  });

  it('palier 3 detail panel exposes Edit / Synonyms / Deprecate / Delete', () => {
    render(<TagsTab {...defaultProps()} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    ['Edit', 'Manage synonyms', 'Deprecate', 'Delete'].forEach((label) => {
      expect(
        within(detail).getByRole('button', { name: label }),
      ).toBeInTheDocument();
    });
  });
});

describe('TagsTab — approve direct (palier 3)', () => {
  it('Approve button on a pending request calls onApprove with the tag', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const argocd = screen.getByTestId('pending-argocd');
    await userEvent.click(within(argocd).getByRole('button', { name: 'Approve' }));
    expect(p.onApprove).toHaveBeenCalledTimes(1);
    expect(p.onApprove.mock.calls[0][0].tag.tag).toBe('argocd');
  });
});

describe('TagsTab — modal dispatch', () => {
  it('Edit on the detail panel opens the edit modal', async () => {
    render(<TagsTab {...defaultProps()} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Edit' }));
    expect(
      await screen.findByRole('dialog', { name: 'Edit tag' }),
    ).toBeInTheDocument();
  });

  it('Edit modal Save commits a TagActionCommit with kind=edit', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Edit' }));
    const dialog = await screen.findByRole('dialog', { name: 'Edit tag' });
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    expect(p.onCommit).toHaveBeenCalledTimes(1);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({ kind: 'edit' });
  });

  it('Delete modal — migrate strategy requires replacement selection', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Delete' }));
    const dialog = await screen.findByRole('dialog', { name: 'Delete tag' });
    // Migrate is the default strategy → Submit button disabled until target picked.
    const submit = within(dialog).getByRole('button', {
      name: 'Migrate and delete',
    });
    expect(submit).toBeDisabled();
    await userEvent.selectOptions(
      within(dialog).getByLabelText('Replacement tag'),
      'oracle',
    );
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({
      kind: 'delete',
      migrate: { strategy: 'migrate', to: 'oracle' },
    });
  });

  it('Delete modal — untag strategy is enabled immediately', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Delete' }));
    const dialog = await screen.findByRole('dialog', { name: 'Delete tag' });
    await userEvent.click(
      within(dialog).getByRole('radio', { name: /Untag and delete/ }),
    );
    const submit = within(dialog).getByRole('button', {
      name: 'Untag and delete',
    });
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({
      kind: 'delete',
      migrate: { strategy: 'untag' },
    });
  });

  it('Reject modal requires a non-empty reason', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const argocd = screen.getByTestId('pending-argocd');
    await userEvent.click(within(argocd).getByRole('button', { name: 'Reject' }));
    const dialog = await screen.findByRole('dialog', { name: 'Reject request' });
    const submit = within(dialog).getByRole('button', { name: 'Reject request' });
    expect(submit).toBeDisabled();
    const reasonInput = within(dialog).getByLabelText('Reason');
    await new Promise((r) => setTimeout(r, 60));
    (reasonInput as HTMLTextAreaElement).focus();
    await userEvent.type(reasonInput, 'duplicate of k8s');
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({
      kind: 'reject',
      reason: 'duplicate of k8s',
    });
  });

  it('Request modal requires a non-empty proposed name', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    await userEvent.click(
      screen.getByRole('button', { name: /Request new tag/ }),
    );
    const dialog = await screen.findByRole('dialog', { name: 'Request new tag' });
    const submit = within(dialog).getByRole('button', { name: 'Submit request' });
    expect(submit).toBeDisabled();
    const nameInput = within(dialog).getByLabelText(/Proposed name/);
    await new Promise((r) => setTimeout(r, 60));
    (nameInput as HTMLInputElement).focus();
    await userEvent.type(nameInput, 'argocd');
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({
      kind: 'request',
      name: 'argocd',
    });
  });

  it('clicking the modal backdrop closes the modal without committing', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Edit' }));
    await screen.findByRole('dialog', { name: 'Edit tag' });
    await userEvent.click(screen.getByTestId('tagaction-backdrop'));
    expect(screen.queryByRole('dialog', { name: 'Edit tag' })).toBeNull();
    expect(p.onCommit).not.toHaveBeenCalled();
  });
});

describe('exportThesaurusJson helper', () => {
  it('produces a JSON blob with workspace + tags + categories', () => {
    const calls: { blob: Blob }[] = [];
    const origCreate = URL.createObjectURL;
    const origRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn((b: Blob) => {
      calls.push({ blob: b });
      return 'blob:mock';
    }) as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn() as unknown as typeof URL.revokeObjectURL;
    try {
      exportThesaurusJson(TAG_FIXTURES, TAG_CATEGORY_FIXTURES, 'tester');
      expect(calls).toHaveLength(1);
      expect(calls[0].blob.type).toMatch(/application\/json/);
    } finally {
      URL.createObjectURL = origCreate;
      URL.revokeObjectURL = origRevoke;
    }
  });
});
