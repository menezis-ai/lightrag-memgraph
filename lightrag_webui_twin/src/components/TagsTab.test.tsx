/**
 * Unit tests for TagsTab + TagActionModal.
 *
 * Covers: header counts, pending section, category rail, search/status
 * filters, empty-zero + empty-filtered states, card selection, related
 * chip navigation, palier-gated detail actions, modal dispatch (8 kinds),
 * commit payloads, exportTagCatalogJson helper.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import {
  fireEvent,
  render as rtlRender,
  screen,
  waitFor,
  within,
  type RenderOptions,
} from '@testing-library/react';
import type { ReactElement, ReactNode } from 'react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TagsTab, exportTagCatalogJson } from './TagsTab';
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

async function openTagRequests() {
  await userEvent.click(
    screen.getByRole('button', { name: /Tag requests/i }),
  );
}

describe('TagsTab — rendering', () => {
  it('renders header with active + pending counts; no palier pill', () => {
    render(<TagsTab {...defaultProps()} folderLabel="sandbox" />);
    expect(screen.getByRole('heading', { name: 'Tags' })).toBeInTheDocument();
    // 21 fixtures - 2 requested = 19 active
    const sub = document.querySelector('.tags-sub') as HTMLElement;
    expect(sub.textContent).toMatch(/19 active tags · 2 pending requests/);
    expect(sub.textContent).toMatch(/folder sandbox/);
    // palier-pill killed per 30/05 cleanup — role lives in JWT, not in chrome
    expect(sub.textContent).not.toMatch(/palier 3/);
    expect(document.querySelector('.palier-pill')).toBeNull();
  });

  it('renders the pending requests section collapsed by default', () => {
    render(<TagsTab {...defaultProps()} />);
    expect(screen.getByText('Tag requests')).toBeInTheDocument();
    expect(document.querySelector('.pending-counts')?.textContent).toMatch(
      /2 tag requests awaiting review/,
    );
    expect(screen.queryByTestId('pending-argocd')).toBeNull();
    expect(screen.getByRole('button', { name: /Tag requests/i })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
  });

  it('does not keep rejected requested tags in the approval queue', () => {
    const props = defaultProps();
    render(
      <TagsTab
        {...props}
        tags={props.tags.map((tag) =>
          tag.tag === 'argocd' ? { ...tag, status: 'rejected' } : tag,
        )}
      />,
    );
    const sub = document.querySelector('.tags-sub') as HTMLElement;
    expect(sub.textContent).toMatch(/19 active tags · 1 pending requests/);
    expect(document.querySelector('.pending-counts')?.textContent).toMatch(
      /1 tag request awaiting review/,
    );
  });

  it('opens pending requests with Approve/Edit-approve/Reject for palier 3', async () => {
    render(<TagsTab {...defaultProps()} />);
    await openTagRequests();
    expect(screen.getByTestId('pending-argocd')).toBeInTheDocument();
    expect(screen.getByTestId('pending-pacs008')).toBeInTheDocument();
    const argocd = screen.getByTestId('pending-argocd');
    expect(within(argocd).getByRole('button', { name: 'Approve' })).toBeInTheDocument();
    expect(
      within(argocd).getByRole('button', { name: 'Edit & approve' }),
    ).toBeInTheDocument();
    expect(within(argocd).getByRole('button', { name: 'Reject' })).toBeInTheDocument();
  });

  it('palier 2 sees pending section but only "Awaiting reviewer approval" caption', async () => {
    render(<TagsTab {...defaultProps(PALIER2)} />);
    await openTagRequests();
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
    await userEvent.click(screen.getByTestId('rail-messaging'));
    expect(screen.queryByTestId('tag-card-rman')).toBeNull();
    expect(screen.getByTestId('tag-card-swift')).toBeInTheDocument();
  });

  it('uncategorized rail shows active tags outside known categories', async () => {
    const props = defaultProps();
    render(
      <TagsTab
        {...props}
        tags={[
          ...props.tags,
          {
            ...props.tags[0],
            tag: 'orphan-tag',
            category: 'legacy-domain',
            def: 'Imported without a mapped domain',
          },
        ]}
      />,
    );
    await userEvent.click(screen.getByTestId('rail-uncategorized'));
    expect(screen.getByTestId('tag-card-orphan-tag')).toBeInTheDocument();
    expect(screen.queryByTestId('tag-card-rman')).toBeNull();
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

  it('empty-zero state appears when the folder has 0 active tags', () => {
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

  it('palier 2 detail panel shows Suggest edit disabled (no backend endpoint) and no destructive actions', () => {
    render(<TagsTab {...defaultProps(PALIER2)} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    expect(
      within(detail).getByRole('button', { name: 'Suggest edit' }),
    ).toBeDisabled();
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
    await openTagRequests();
    const argocd = screen.getByTestId('pending-argocd');
    await userEvent.click(within(argocd).getByRole('button', { name: 'Approve' }));
    expect(p.onApprove).toHaveBeenCalledTimes(1);
    expect(p.onApprove.mock.calls[0][0].tag.tag).toBe('argocd');
  });
});

describe('TagsTab — domain taxonomy editor', () => {
  it('saves direct domain edits through the taxonomy import endpoint', async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    try {
      render(<TagsTab {...defaultProps()} />);
      await userEvent.click(
        screen.getByRole('button', { name: 'Edit domains' }),
      );
      const dialog = await screen.findByRole('dialog', { name: 'Edit domains' });

      const messagingLabel = within(dialog).getByLabelText(
        'messaging domain label',
      );
      fireEvent.change(messagingLabel, { target: { value: 'Communication' } });

      await userEvent.click(
        within(dialog).getByRole('button', { name: 'Add domain' }),
      );
      const newId = within(dialog).getByLabelText('New domain domain id');
      fireEvent.change(newId, { target: { value: 'collaboration' } });
      const newLabel = within(dialog).getByLabelText('collaboration domain label');
      fireEvent.change(newLabel, { target: { value: 'Collaboration' } });

      await userEvent.click(
        within(dialog).getByRole('button', { name: 'Save domains' }),
      );

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
      const [url, init] = fetchMock.mock.calls[0];
      expect(String(url)).toContain('/tags/categories/_import');
      expect(init?.method).toBe('POST');
      const payload = JSON.parse(String(init?.body));
      expect(payload).toEqual(
        expect.arrayContaining([
          { id: 'messaging', label: 'Communication', color: '#7B5BB8' },
          { id: 'collaboration', label: 'Collaboration', color: '#5A7FB4' },
        ]),
      );
      expect(
        await screen.findByTestId('taxonomy-import-status'),
      ).toHaveTextContent(/domains applied/);
    } finally {
      globalThis.fetch = originalFetch;
    }
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
    await openTagRequests();
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

  it('Request modal commits long description, synonyms and justification', async () => {
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
    expect(submit).toBeDisabled();
    await userEvent.type(
      within(dialog).getByLabelText(/Definition/),
      'Argo CD deployment automation',
    );
    await userEvent.type(
      within(dialog).getByLabelText(/Long description/),
      'Use for GitOps deployment workflows.',
    );
    await userEvent.type(
      within(dialog).getByLabelText(/Synonyms/),
      'argo-cd, gitops',
    );
    await userEvent.type(
      within(dialog).getByLabelText(/Justification/),
      'Existing infra tag is too broad.',
    );
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({
      kind: 'request',
      name: 'argocd',
      def: 'Argo CD deployment automation',
      longDescription: 'Use for GitOps deployment workflows.',
      aliases: ['argo-cd', 'gitops'],
      justification: 'Existing infra tag is too broad.',
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

describe('exportTagCatalogJson helper', () => {
  it('produces a JSON blob with folder + tags + categories', () => {
    const calls: { blob: Blob }[] = [];
    const origCreate = URL.createObjectURL;
    const origRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn((b: Blob) => {
      calls.push({ blob: b });
      return 'blob:mock';
    }) as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn() as unknown as typeof URL.revokeObjectURL;
    try {
      exportTagCatalogJson(TAG_FIXTURES, TAG_CATEGORY_FIXTURES, 'tester');
      expect(calls).toHaveLength(1);
      expect(calls[0].blob.type).toMatch(/application\/json/);
    } finally {
      URL.createObjectURL = origCreate;
      URL.revokeObjectURL = origRevoke;
    }
  });
});
