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
  name: 'demo.steward',
  palier: 3,
  role: 'admin / steward',
};
const PALIER2: TagCurrentUser = { name: 'demo.reviewer', palier: 2, role: 'steward' };
const PALIER1: TagCurrentUser = { name: 'demo.operator', palier: 1, role: 'reader' };

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
    screen.getByRole('button', { name: /Tag review queue/i }),
  );
}

describe('TagsTab — rendering', () => {
  it('renders header with active + pending counts; no palier pill', () => {
    render(<TagsTab {...defaultProps()} folderLabel="sandbox" />);
    expect(screen.getByRole('heading', { name: 'Tags' })).toBeInTheDocument();
    // 21 fixtures - 2 requested = 19 active
    const sub = document.querySelector('.tags-sub') as HTMLElement;
    expect(sub.textContent).toMatch(/19 active tags · 2 pending items/);
    expect(sub.textContent).toMatch(/folder sandbox/);
    // palier-pill killed per 30/05 cleanup — role lives in JWT, not in chrome
    expect(sub.textContent).not.toMatch(/palier 3/);
    expect(document.querySelector('.palier-pill')).toBeNull();
  });

  it('renders the pending requests section collapsed by default', () => {
    render(<TagsTab {...defaultProps()} />);
    expect(screen.getByText('Tag review queue')).toBeInTheDocument();
    expect(document.querySelector('.pending-counts')?.textContent).toMatch(
      /2 governance items awaiting review/,
    );
    expect(screen.queryByTestId('pending-argocd')).toBeNull();
    expect(screen.getByRole('button', { name: /Tag review queue/i })).toHaveAttribute(
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
    expect(sub.textContent).toMatch(/19 active tags · 1 pending item/);
    expect(document.querySelector('.pending-counts')?.textContent).toMatch(
      /1 governance item awaiting review/,
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

  it('renders pending edit proposals with target tag and approve-edit action', async () => {
    const props = defaultProps();
    const proposal = {
      ...TAG_FIXTURES[0],
      tag: 'rman__edit__demo.qa-20260629',
      tier: 'requested' as const,
      status: 'pending-review' as const,
      def: 'Updated RMAN definition',
      requested_by: 'demo.qa',
      requested_at: '2026-06-29',
      justification: 'Clarify wording',
      proposal_kind: 'edit' as const,
      target_tag: 'rman',
      proposed_fields: ['def', 'aliases'],
      last_edit: { by: 'demo.qa', at: '2026-06-29', action: 'edit-suggested' },
    };
    render(<TagsTab {...props} tags={[...props.tags, proposal]} />);
    await openTagRequests();
    const card = screen.getByTestId('pending-rman__edit__demo.qa-20260629');
    expect(within(card).getByText('Edit suggestion')).toBeInTheDocument();
    expect(within(card).getByText('rman')).toBeInTheDocument();
    expect(within(card).getByRole('button', { name: 'Approve edit' })).toBeInTheDocument();
    expect(within(card).queryByRole('button', { name: 'Edit & approve' })).toBeNull();
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

  it('status filter omits the redundant Pending option', async () => {
    const props = defaultProps();
    render(
      <TagsTab
        {...props}
        tags={props.tags.map((tag) =>
          tag.tag === 'graphrag' ? { ...tag, status: 'deprecated' } : tag,
        )}
      />,
    );
    expect(
      within(screen.getByLabelText('Status filter')).queryByRole('option', {
        name: 'Pending',
      }),
    ).toBeNull();
    await userEvent.selectOptions(
      screen.getByLabelText('Status filter'),
      'deprecated',
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

  it('detail CTA opens Documents filtered by the tag (QA TAG-V7-001)', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(
      within(detail).getByRole('button', {
        name: /See documents containing this tag/,
      }),
    );
    expect(p.onNavigate).toHaveBeenCalledWith('documents', { tag: 'rman' });
  });

  it('palier 1 detail panel is read-only (shows muted hint)', () => {
    render(<TagsTab {...defaultProps(PALIER1)} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/Palier 1 — read-only/);
    expect(within(detail).queryByRole('button', { name: 'Edit' })).toBeNull();
  });

  it('palier 2 detail panel opens Suggest edit and emits proposed fields', async () => {
    const props = defaultProps(PALIER2);
    render(<TagsTab {...props} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Suggest edit' }));
    const dialog = await screen.findByRole('dialog', { name: 'Suggest tag edit' });
    expect(
      within(dialog).getByRole('button', { name: 'Submit suggestion' }),
    ).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText(/Short definition/), {
      target: { value: 'Updated recovery manager definition' },
    });
    fireEvent.change(within(dialog).getByLabelText(/Proposed synonyms/), {
      target: { value: 'rmgr, recovery-manager' },
    });
    fireEvent.change(within(dialog).getByLabelText(/Justification/), {
      target: { value: 'Clarify Demo QA recette wording' },
    });
    await userEvent.click(
      within(dialog).getByRole('button', { name: 'Submit suggestion' }),
    );

    expect(props.onCommit).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'suggest',
        tag: expect.objectContaining({ tag: 'rman' }),
        def: 'Updated recovery manager definition',
        aliases: ['rmgr', 'recovery-manager'],
        justification: 'Clarify Demo QA recette wording',
      }),
    );
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

  it('usage docs count navigates directly to Documents with the tag filter', async () => {
    const props = defaultProps();
    render(<TagsTab {...props} />);

    await userEvent.click(
      screen.getByRole('button', {
        name: /View \d+ documents tagged rman/,
      }),
    );

    expect(props.onNavigate).toHaveBeenCalledWith('documents', { tag: 'rman' });
  });

  it('deprecated tags expose a real Reactivate action instead of another Deprecate action', async () => {
    const props = defaultProps();
    render(
      <TagsTab
        {...props}
        tags={props.tags.map((tag) =>
          tag.tag === 'rman' ? { ...tag, status: 'deprecated' } : tag,
        )}
      />,
    );
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    expect(within(detail).queryByRole('button', { name: 'Deprecate' })).toBeNull();
    await userEvent.click(within(detail).getByRole('button', { name: 'Reactivate' }));
    expect(props.onCommit).toHaveBeenCalledWith({
      kind: 'reactivate',
      tag: expect.objectContaining({ tag: 'rman', status: 'deprecated' }),
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
  it('rejects domain names duplicated by case, accents, or whitespace', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Manage domains' }));
    const editor = screen.getByTestId('domain-rail-editor');

    fireEvent.change(within(editor).getByLabelText('infra domain label'), {
      target: { value: 'Réseau' },
    });
    fireEvent.change(within(editor).getByLabelText('network domain label'), {
      target: { value: '  RESEAU  ' },
    });
    await userEvent.click(within(editor).getByRole('button', { name: 'Save' }));

    expect(within(editor).getByRole('alert')).toHaveTextContent(
      /duplicates "Réseau" after case, accent, and whitespace normalization/i,
    );
  });

  it('renders the backend detail when Unicode case folding rejects the draft', async () => {
    const originalFetch = globalThis.fetch;
    const detail =
      'Domain name "STRASSE" duplicates "Straße" after Unicode normalization.';
    const fetchMock = vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify({ detail }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    try {
      render(<TagsTab {...defaultProps()} />);
      await userEvent.click(
        screen.getByRole('button', { name: 'Manage domains' }),
      );
      const editor = screen.getByTestId('domain-rail-editor');

      fireEvent.change(within(editor).getByLabelText('infra domain label'), {
        target: { value: 'Straße' },
      });
      fireEvent.change(within(editor).getByLabelText('network domain label'), {
        target: { value: 'STRASSE' },
      });
      await userEvent.click(within(editor).getByRole('button', { name: 'Save' }));

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
      expect(await within(editor).findByRole('alert')).toHaveTextContent(detail);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('edits domains inline from the domain rail', async () => {
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
      expect(screen.queryByRole('button', { name: 'Edit domains' })).toBeNull();
      await userEvent.click(
        screen.getByRole('button', { name: 'Manage domains' }),
      );
      const editor = screen.getByTestId('domain-rail-editor');

      const existingLabel = within(editor).getByLabelText(
        'messaging domain label',
      );
      expect(existingLabel).toBeRequired();
      expect(existingLabel).toHaveAttribute('aria-required', 'true');

      fireEvent.change(within(editor).getByLabelText('messaging domain label'), {
        target: { value: 'Communication' },
      });

      await userEvent.click(
        within(editor).getByRole('button', { name: 'Add domain' }),
      );
      const newDomainId = within(editor).getByLabelText('New domain domain id');
      expect(newDomainId).toBeRequired();
      expect(newDomainId).toHaveAttribute('aria-required', 'true');
      fireEvent.change(within(editor).getByLabelText('New domain domain id'), {
        target: { value: 'collaboration' },
      });
      fireEvent.change(
        within(editor).getByLabelText('collaboration domain label'),
        { target: { value: 'Collaboration' } },
      );

      await userEvent.click(within(editor).getByRole('button', { name: 'Save' }));

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

  it('Edit modal blocks saving an emptied definition (QA TAG-V8-001)', async () => {
    const p = defaultProps();
    render(<TagsTab {...p} />);
    const detail = document.querySelector('.tag-detail') as HTMLElement;
    await userEvent.click(within(detail).getByRole('button', { name: 'Edit' }));
    const dialog = await screen.findByRole('dialog', { name: 'Edit tag' });
    const defInput = within(dialog).getByLabelText(/Short definition/);
    await userEvent.clear(defInput);
    const submit = within(dialog).getByRole('button', { name: 'Save' });
    expect(submit).toBeDisabled();
    expect(
      within(dialog).getByText(
        'Definition is required — a tag cannot be saved without one.',
      ),
    ).toBeInTheDocument();
    await userEvent.type(defInput, 'Restored definition');
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    expect(p.onCommit.mock.calls[0][0]).toMatchObject({
      kind: 'edit',
      def: 'Restored definition',
    });
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
    const reasonInput = within(dialog).getByLabelText(/^Reason/);
    expect(reasonInput).toBeRequired();
    expect(reasonInput).toHaveAttribute('aria-required', 'true');
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
    expect(
      within(dialog).getByText(/Proposed name is required/i),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText(/Definition is required/i),
    ).toBeInTheDocument();
    const nameInput = within(dialog).getByLabelText(/Proposed name/);
    await new Promise((r) => setTimeout(r, 60));
    (nameInput as HTMLInputElement).focus();
    await userEvent.type(nameInput, 'argocd');
    expect(submit).toBeDisabled();
    expect(within(dialog).queryByText(/Proposed name is required/i)).toBeNull();
    expect(
      within(dialog).getByText(/Definition is required/i),
    ).toBeInTheDocument();
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

  it('Request modal inherits the selected domain from the rail', async () => {
    render(<TagsTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('rail-governance'));
    await userEvent.click(
      screen.getByRole('button', { name: /Request new tag/ }),
    );
    const dialog = await screen.findByRole('dialog', { name: 'Request new tag' });
    expect(within(dialog).getByLabelText('Domain')).toHaveValue('governance');
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
