/**
 * Unit tests for useTagActions.
 *
 * The hook wraps the tag-governance TanStack mutations (request / approve /
 * reject / edit / deprecate / synonyms / delete) and turns each into a
 * structured toast on success / failure. We mock `../api/resources` so the
 * real `../api/queries` mutation hooks run their onSuccess/onError plumbing
 * against controllable resolvers, and assert the emitted toast payloads for
 * every `TagActionCommit['kind']` plus the direct approve path.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { useTagActions } from './useTagActions';
import type { TagActionCommit } from '../components/TagActionModal';
import type { TagEntry } from '../types/tag';
import type { Toast } from '../types/toast';

// ── Mock the resource layer the queries hooks call ─────────────────────────
const requestTagMock = vi.hoisted(() => vi.fn());
const approveTagMock = vi.hoisted(() => vi.fn());
const rejectTagMock = vi.hoisted(() => vi.fn());
const editTagMock = vi.hoisted(() => vi.fn());
const deprecateTagMock = vi.hoisted(() => vi.fn());
const updateTagSynonymsMock = vi.hoisted(() => vi.fn());
const deleteTagMock = vi.hoisted(() => vi.fn());

vi.mock('../api/resources', () => ({
  api: {
    requestTag: requestTagMock,
    approveTag: approveTagMock,
    rejectTag: rejectTagMock,
    editTag: editTagMock,
    deprecateTag: deprecateTagMock,
    updateTagSynonyms: updateTagSynonymsMock,
    deleteTag: deleteTagMock,
  },
}));

const ACTOR = 'claire.benoit';

function makeTag(overrides: Partial<TagEntry> = {}): TagEntry {
  return {
    tag: 'argocd',
    tier: 3,
    category: 'infra',
    status: 'active',
    def: 'GitOps CD',
    aliases: ['argo'],
    deprecates: [],
    sources_count: 0,
    chunks_count: 0,
    query_freq_30d: 0,
    created: { by: ACTOR, at: '2026-01-01' },
    last_edit: { by: ACTOR, at: '2026-01-01' },
    related: [],
    examples: [],
    ...overrides,
  };
}

function wrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return Wrapper;
}

function setup(pushToast: (toast: Omit<Toast, 'id'>) => void) {
  return renderHook(
    () => useTagActions({ currentActor: ACTOR, pushToast }),
    { wrapper: wrapper() },
  );
}

function resolveAll() {
  requestTagMock.mockResolvedValue({});
  approveTagMock.mockResolvedValue({});
  rejectTagMock.mockResolvedValue({});
  editTagMock.mockResolvedValue({});
  deprecateTagMock.mockResolvedValue({});
  updateTagSynonymsMock.mockResolvedValue({});
  deleteTagMock.mockResolvedValue({});
}

beforeEach(() => {
  vi.clearAllMocks();
  resolveAll();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('onTagApprove', () => {
  it('approves the tag and pushes a done toast', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    await result.current.onTagApprove({ tag: makeTag() });

    expect(approveTagMock).toHaveBeenCalledWith('argocd', ACTOR);
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'done',
        title: 'Tag',
        tagname: 'argocd',
        titleSuffix: 'approved',
      }),
    );
  });

  it('pushes an error toast with the Error message on rejection', async () => {
    approveTagMock.mockRejectedValueOnce(new Error('backend down'));
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    await result.current.onTagApprove({ tag: makeTag({ tag: 'k8s' }) });

    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Tag approval failed',
        tagname: 'k8s',
        sub: 'backend down',
      }),
    );
  });

  it('falls back to "Mutation rejected" when the error is not an Error', async () => {
    approveTagMock.mockRejectedValueOnce('boom-string');
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    await result.current.onTagApprove({ tag: makeTag() });

    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', sub: 'Mutation rejected' }),
    );
  });
});

describe('onTagCommit success paths', () => {
  it('edit → editTag + "updated" toast', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    const commit: TagActionCommit = {
      kind: 'edit',
      tag: makeTag(),
      name: 'argo-cd',
      def: 'new def',
      longDescription: 'long',
      category: 'platform',
      reason: 'cleanup',
    };
    result.current.onTagCommit(commit);

    await waitFor(() => expect(editTagMock).toHaveBeenCalled());
    expect(editTagMock).toHaveBeenCalledWith('argocd', {
      tag: 'argo-cd',
      def: 'new def',
      long_description: 'long',
      category: 'platform',
      actor: ACTOR,
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'done',
          titleSuffix: 'updated',
          sub: 'cleanup',
        }),
      ),
    );
  });

  it('suggest is gated off → fires no mutation (no backend edit-proposal endpoint)', () => {
    // Regression guard: "Suggest edit" used to route through requestTag with the
    // existing tag name, which the backend rejects with 409, and it dropped the
    // edited fields. The affordance is disabled until a real endpoint exists, so
    // onTagCommit('suggest') must be a no-op — with or without a tag.
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'suggest', tag: makeTag() });
    result.current.onTagCommit({ kind: 'suggest', tag: null });
    expect(requestTagMock).not.toHaveBeenCalled();
    expect(pushToast).not.toHaveBeenCalled();
  });

  it('synonyms → updateSynonyms using commit.aliases', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'synonyms',
      tag: makeTag(),
      aliases: ['a1', 'a2'],
    });

    await waitFor(() => expect(updateTagSynonymsMock).toHaveBeenCalled());
    expect(updateTagSynonymsMock).toHaveBeenCalledWith('argocd', {
      aliases: ['a1', 'a2'],
      actor: ACTOR,
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ titleSuffix: 'synonyms updated' }),
      ),
    );
  });

  it('synonyms → falls back to tag.aliases when commit.aliases is absent', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'synonyms', tag: makeTag() });

    await waitFor(() => expect(updateTagSynonymsMock).toHaveBeenCalled());
    expect(updateTagSynonymsMock).toHaveBeenCalledWith('argocd', {
      aliases: ['argo'],
      actor: ACTOR,
    });
  });

  it('synonyms with no tag → no mutation (guard branch)', () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'synonyms', tag: null });
    expect(updateTagSynonymsMock).not.toHaveBeenCalled();
  });

  it('deprecate → deprecateTag forwards the reason + "deprecated" toast', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'deprecate',
      tag: makeTag(),
      reason: 'Superseded by iso20022',
    });

    await waitFor(() => expect(deprecateTagMock).toHaveBeenCalled());
    expect(deprecateTagMock).toHaveBeenCalledWith('argocd', {
      actor: ACTOR,
      reason: 'Superseded by iso20022',
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ titleSuffix: 'deprecated' }),
      ),
    );
  });

  it('delete with untag strategy → "deleted (docs untagged)" toast', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'delete',
      tag: makeTag(),
      migrate: { strategy: 'untag' },
    });

    await waitFor(() => expect(deleteTagMock).toHaveBeenCalled());
    expect(deleteTagMock).toHaveBeenCalledWith('argocd', {
      strategy: 'untag',
      to: undefined,
      actor: ACTOR,
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ titleSuffix: 'deleted (docs untagged)' }),
      ),
    );
  });

  it('delete with migrate strategy → "migrated to X" toast + default strategy when migrate missing', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    // migrate present → migrate verb
    result.current.onTagCommit({
      kind: 'delete',
      tag: makeTag(),
      migrate: { strategy: 'migrate', to: 'flux' },
    });
    await waitFor(() =>
      expect(deleteTagMock).toHaveBeenCalledWith('argocd', {
        strategy: 'migrate',
        to: 'flux',
        actor: ACTOR,
      }),
    );
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ titleSuffix: 'migrated to flux' }),
      ),
    );
  });

  it('delete with no migrate → defaults strategy to untag', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'delete', tag: makeTag() });
    await waitFor(() =>
      expect(deleteTagMock).toHaveBeenCalledWith('argocd', {
        strategy: 'untag',
        to: undefined,
        actor: ACTOR,
      }),
    );
  });

  it('delete migrate with empty "to" → "migrated to " verb (nullish branch)', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'delete',
      tag: makeTag(),
      migrate: { strategy: 'migrate' },
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ titleSuffix: 'migrated to ' }),
      ),
    );
  });

  it('reject → rejectTag with provided reason', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'reject',
      tag: makeTag(),
      reason: 'duplicate of k8s',
    });
    await waitFor(() =>
      expect(rejectTagMock).toHaveBeenCalledWith('argocd', {
        reason: 'duplicate of k8s',
        actor: ACTOR,
      }),
    );
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ titleSuffix: 'rejected' }),
      ),
    );
  });

  it('reject → defaults reason to "rejected" when empty', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'reject', tag: makeTag(), reason: '' });
    await waitFor(() =>
      expect(rejectTagMock).toHaveBeenCalledWith('argocd', {
        reason: 'rejected',
        actor: ACTOR,
      }),
    );
  });

  it('request with name → requestTag with full body and defaults', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'request',
      tag: null,
      name: 'newtag',
      justification: 'we need it',
    });
    await waitFor(() =>
      expect(requestTagMock).toHaveBeenCalledWith({
        tag: 'newtag',
        def: '',
        long_description: undefined,
        category: 'infra',
        aliases: [],
        justification: 'we need it',
        actor: ACTOR,
      }),
    );
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          titleSuffix: 'requested for review',
          tagname: 'newtag',
        }),
      ),
    );
  });

  it('request with explicit def/category/aliases → forwards them', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'request',
      tag: null,
      name: 'newtag',
      def: 'the def',
      category: 'platform',
      aliases: ['nt'],
    });
    await waitFor(() =>
      expect(requestTagMock).toHaveBeenCalledWith(
        expect.objectContaining({
          def: 'the def',
          category: 'platform',
          aliases: ['nt'],
        }),
      ),
    );
  });

  it('request with no name → no mutation (guard branch)', () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'request', tag: null });
    expect(requestTagMock).not.toHaveBeenCalled();
  });

  it('uses commit.name as tagname when tag is null', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'edit', tag: null, name: 'fromname' });
    await waitFor(() =>
      expect(editTagMock).toHaveBeenCalledWith(
        'fromname',
        expect.objectContaining({ tag: 'fromname' }),
      ),
    );
  });

  it('falls back to empty tagname when both tag and name are absent', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'edit', tag: null });
    await waitFor(() => expect(editTagMock).toHaveBeenCalled());
    expect(editTagMock).toHaveBeenCalledWith(
      '',
      expect.objectContaining({ tag: undefined }),
    );
  });
});

describe('onTagCommit error paths (commitTagMutation onError)', () => {
  it('edit failure → "Tag edit failed" toast with Error message', async () => {
    editTagMock.mockRejectedValueOnce(new Error('write conflict'));
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'edit', tag: makeTag(), name: 'x' });

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Tag edit failed',
          tagname: 'argocd',
          sub: 'write conflict',
        }),
      ),
    );
  });

  it('reject failure with non-Error → "Mutation rejected" fallback', async () => {
    rejectTagMock.mockRejectedValueOnce('weird');
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'reject', tag: makeTag(), reason: 'r' });

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Tag reject failed',
          sub: 'Mutation rejected',
        }),
      ),
    );
  });
});

describe('onTagCommit edit-approve path', () => {
  it('renames via edit then approves the trimmed NEW name (not the deleted old one)', async () => {
    // Regression #2/#1: the rename deletes the old tag, so approving the old
    // name 404s; and the backend stores the TRIMMED rename target, so approve
    // must target commit.name.trim() — a raw "  argo-cd  " would 404.
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'edit-approve',
      tag: makeTag(),
      name: '  argo-cd  ',
    });

    await waitFor(() => expect(editTagMock).toHaveBeenCalled());
    await waitFor(() => expect(approveTagMock).toHaveBeenCalled());
    expect(approveTagMock).toHaveBeenCalledWith('argo-cd', ACTOR);
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'done',
          titleSuffix: 'approved (edited)',
        }),
      ),
    );
  });

  it('skips the edit when no edit fields are present, only approves', async () => {
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'edit-approve', tag: makeTag() });

    await waitFor(() => expect(approveTagMock).toHaveBeenCalled());
    expect(editTagMock).not.toHaveBeenCalled();
  });

  it('error during edit-approve → error toast', async () => {
    approveTagMock.mockRejectedValueOnce(new Error('approve boom'));
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({ kind: 'edit-approve', tag: makeTag() });

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Tag edit-approve failed',
          sub: 'approve boom',
        }),
      ),
    );
  });

  it('error during edit-approve with non-Error → "Mutation rejected"', async () => {
    editTagMock.mockRejectedValueOnce('str');
    const pushToast = vi.fn();
    const { result } = setup(pushToast);
    result.current.onTagCommit({
      kind: 'edit-approve',
      tag: makeTag(),
      def: 'changed',
    });

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'error', sub: 'Mutation rejected' }),
      ),
    );
  });
});
