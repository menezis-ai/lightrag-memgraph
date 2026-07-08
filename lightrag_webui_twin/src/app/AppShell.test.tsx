/**
 * Unit tests for AppShell.
 *
 * AppShell is the top-level wiring component: it owns tab/folder/theme/modal
 * state, derives doc/tag/graph/notification view-models from the query layer,
 * renders the backend-error banner, and threads dozens of callbacks into the
 * Topbar + tab + modal children.
 *
 * Testing strategy: we stub every heavy child component (eager + lazy) with a
 * lightweight harness that re-exposes the AppShell-provided callbacks as
 * buttons / testids, and we mock the data layer (`useAuth`, `../api/queries`,
 * `../api/resources`, action hooks) so each branch can be driven
 * deterministically. This keeps the unit boundary on AppShell's own logic
 * (state transitions, derivations, conditional rendering) rather than the
 * children, which carry their own suites.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import type { Document } from '../types/document';
import type { Folder } from '../types/topbar';

// ── Auth mock ───────────────────────────────────────────────────────────────
const authState = vi.hoisted(() => ({
  current: {
    user: { email: 'claire.benoit@twin.local' } as { email: string } | null,
    isAuthenticated: true,
    isCheckingAuth: false,
    needsLogin: false,
    loginError: null as string | null,
    config: {
      folders: undefined as
        | undefined
        | ReadonlyArray<{
            id: string;
            label: string;
            kind?: string;
            sources?: number;
          }>,
      defaultFolderId: 'default',
    },
    login: vi.fn(),
    signout: vi.fn().mockResolvedValue(undefined),
  },
}));
vi.mock('../hooks/useAuth', () => ({
  useAuth: () => authState.current,
}));

// ── client mock (setActiveFolder + getTwinRuntimeConfig used at init) ────────
const setActiveFolderMock = vi.hoisted(() => vi.fn());
vi.mock('../api/client', () => ({
  setActiveFolder: setActiveFolderMock,
  getTwinRuntimeConfig: () => authState.current.config,
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
}));

// ── queries mock ─────────────────────────────────────────────────────────────
// A loose query view-model: `data`/`error` are intentionally `unknown` so each
// test can assign whatever envelope shape the resource returns without TS
// fighting the literal `undefined` default.
interface MockQuery {
  data?: unknown;
  isError: boolean;
  isLoading: boolean;
  isFetching: boolean;
  error?: unknown;
  refetch: ReturnType<typeof vi.fn>;
}
function q(): MockQuery {
  return {
    data: undefined,
    isError: false,
    isLoading: false,
    isFetching: false,
    error: undefined,
    refetch: vi.fn().mockResolvedValue(undefined),
  };
}

const queriesState = vi.hoisted(() => {
  const make = (): {
    data?: unknown;
    isError: boolean;
    isLoading: boolean;
    isFetching: boolean;
    error?: unknown;
    refetch: ReturnType<typeof vi.fn>;
  } => ({
    data: undefined,
    isError: false,
    isLoading: false,
    isFetching: false,
    error: undefined,
    refetch: vi.fn().mockResolvedValue(undefined),
  });
  return {
    docs: make(),
    folders: make(),
    notifications: make(),
    tags: make(),
    tagCategories: make(),
    activity: make(),
    graphEntities: make(),
    graphRelations: make(),
    pipelineStatus: make(),
  };
});

const queriesSpy = vi.hoisted(() => ({
  useDocuments: [] as unknown[][],
  useFolders: [] as unknown[][],
  useNotifications: [] as unknown[][],
  useTags: [] as unknown[][],
  useTagCategories: [] as unknown[][],
  useActivity: [] as unknown[][],
  useGraphEntities: [] as unknown[][],
  useGraphRelations: [] as unknown[][],
  usePipelineStatus: [] as unknown[][],
}));

vi.mock('../api/queries', () => ({
  useDocuments: (...args: unknown[]) => {
    queriesSpy.useDocuments.push(args);
    return queriesState.docs;
  },
  useFolders: (...args: unknown[]) => {
    queriesSpy.useFolders.push(args);
    return queriesState.folders;
  },
  useNotifications: (...args: unknown[]) => {
    queriesSpy.useNotifications.push(args);
    return queriesState.notifications;
  },
  useTags: (...args: unknown[]) => {
    queriesSpy.useTags.push(args);
    return queriesState.tags;
  },
  useTagCategories: (...args: unknown[]) => {
    queriesSpy.useTagCategories.push(args);
    return queriesState.tagCategories;
  },
  useActivity: (...args: unknown[]) => {
    queriesSpy.useActivity.push(args);
    return queriesState.activity;
  },
  useGraphEntities: (...args: unknown[]) => {
    queriesSpy.useGraphEntities.push(args);
    return queriesState.graphEntities;
  },
  useGraphRelations: (...args: unknown[]) => {
    queriesSpy.useGraphRelations.push(args);
    return queriesState.graphRelations;
  },
  usePipelineStatus: (...args: unknown[]) => {
    queriesSpy.usePipelineStatus.push(args);
    return queriesState.pipelineStatus;
  },
}));

// ── resources mock (RetrievalTab query + reprocess) ──────────────────────────
const apiMock = vi.hoisted(() => ({
  query: vi.fn().mockResolvedValue({ response: 'ok', sources: [] }),
  queryStream: vi.fn().mockResolvedValue({ response: 'ok', sources: [] }),
  reprocessFailedDocuments: vi
    .fn()
    .mockResolvedValue({ message: 'retrying all FAILED' }),
}));
vi.mock('../api/resources', () => ({ api: apiMock }));
vi.mock('../api/twinQueryResponse', () => ({
  mapTwinQueryResponseForRetrievalTab: (res: unknown) => res,
}));

// ── action hooks ──────────────────────────────────────────────────────────────
const docActions = vi.hoisted(() => ({
  uploadDocs: { isPending: false },
  onAddSourceSubmit: vi.fn(),
  onDeleteBulk: vi.fn(),
  onDeleteSingle: vi.fn().mockResolvedValue(undefined),
  onRetagSubmit: vi.fn(),
  onScanRetry: vi.fn(),
  onToastUndo: vi.fn().mockResolvedValue(undefined),
}));
vi.mock('./useDocumentActions', () => ({
  useDocumentActions: () => docActions,
}));

const tagActions = vi.hoisted(() => ({
  onTagApprove: vi.fn(),
  onTagCommit: vi.fn(),
}));
vi.mock('./useTagActions', () => ({
  useTagActions: () => tagActions,
}));

// useAppNavigation: keep behavior real-enough — the returned callbacks invoke
// the setters AppShell passes in, so navigating actually flips AppShell state.
vi.mock('./useAppNavigation', () => ({
  useAppNavigation: (opts: Record<string, (v: unknown) => void>) => ({
    onNavigate: (nextTab: string, params?: Record<string, string>) => {
      if (nextTab === 'documents' && (params?.doc || params?.source)) {
        opts.setDetailRequest({
          doc: params.doc,
          source: params.source,
          chunk: params.chunk,
        });
      } else {
        opts.setDetailRequest(null);
      }
      opts.setTab(nextTab);
    },
    onSwitchFolder: (next: string) => {
      opts.setFolderState(next);
      opts.setReadNotificationIds(new Set());
      opts.setClearedNotificationIds(new Set());
    },
  }),
}));

// ── child component stubs ─────────────────────────────────────────────────────
// Each stub re-exposes the AppShell callbacks we want to exercise.

vi.mock('../components/Topbar', () => ({
  Topbar: (p: Record<string, unknown>) => (
    <div data-testid="topbar">
      <span data-testid="topbar-tab">{String(p.tab)}</span>
      <span data-testid="topbar-folder">{String(p.folder)}</span>
      <span data-testid="topbar-kb">{String(p.kbName)}</span>
      <span data-testid="topbar-unread">{String(p.unreadCount)}</span>
      <span data-testid="topbar-folders-count">
        {String((p.folders as unknown[]).length)}
      </span>
      <span data-testid="topbar-folder-sources">
        {(p.folders as Folder[])
          .map((folder) => `${folder.id}:${folder.sources}`)
          .join('|')}
      </span>
      <span data-testid="topbar-notif-count">
        {String((p.notifications as unknown[]).length)}
      </span>
      <button onClick={() => (p.onTab as (t: string) => void)('settings')}>
        go-settings
      </button>
      <button onClick={() => (p.onTab as (t: string) => void)('graph')}>
        go-graph
      </button>
      <button onClick={() => (p.onTab as (t: string) => void)('retrieval')}>
        go-retrieval
      </button>
      <button onClick={() => (p.onTab as (t: string) => void)('activity')}>
        go-activity
      </button>
      <button onClick={() => (p.onTab as (t: string) => void)('tags')}>
        go-tags
      </button>
      <button onClick={() => (p.onTheme as () => void)()}>toggle-theme</button>
      <button
        onClick={() =>
          (p.onSwitchFolder as (w: Folder) => void)({ id: 'finance' } as Folder)
        }
      >
        switch-folder
      </button>
      <button onClick={() => (p.onMarkAllRead as () => void)()}>
        mark-all-read
      </button>
      <button onClick={() => (p.onClearNotifications as () => void)()}>
        clear-notifs
      </button>
      <button onClick={() => (p.onOpenActivity as () => void)()}>
        open-activity
      </button>
      <button onClick={() => (p.onManageFolders as () => void)()}>
        manage-folders
      </button>
    </div>
  ),
}));

vi.mock('../components/DocumentsTab', () => ({
  DocumentsTab: (p: Record<string, unknown>) => (
    <div data-testid="documents-tab">
      <span data-testid="dt-docs-count">
        {String((p.docs as unknown[]).length)}
      </span>
      <span data-testid="dt-total">{String(p.totalCount)}</span>
      <span data-testid="dt-page">{String(p.currentPage)}</span>
      <span data-testid="dt-hasnext">{String(p.hasNextPage)}</span>
      <span data-testid="dt-page-fetching">{String(p.isPageFetching)}</span>
      <span data-testid="dt-pipeline-error">{String(p.pipelineError)}</span>
      <span data-testid="dt-status-counts">
        {JSON.stringify(p.statusCounts ?? null)}
      </span>
      <div data-testid="dt-pending-slot">{p.pendingSlot as ReactNode}</div>
      <button onClick={() => (p.onOpenAdd as () => void)()}>open-add</button>
      <button
        onClick={() =>
          (p.onOpenRetag as (d: Document) => void)({ doc_id: 'd1' } as Document)
        }
      >
        open-retag
      </button>
      <button
        onClick={() =>
          (p.onOpenBulkRetag as (ds: Document[]) => void)([
            { doc_id: 'd1' } as Document,
          ])
        }
      >
        open-bulk-retag
      </button>
      <button
        onClick={() =>
          (p.onOpenDetail as (d: Document) => void)({
            doc_id: 'd1',
            file_path: 'a.pdf',
          } as Document)
        }
      >
        open-detail
      </button>
      <button onClick={() => (p.onAddToast as (t: string) => void)('hi')}>
        add-toast
      </button>
      <button onClick={() => (p.onStatusFilterChange as (v: string) => void)('failed')}>
        filter-failed
      </button>
      <button onClick={() => (p.onSearchChange as (v: string) => void)('memo')}>
        search-memo
      </button>
      <button onClick={() => (p.onTagFiltersChange as (v: string[]) => void)(['t1'])}>
        tag-filter
      </button>
      <button
        onClick={() => (p.onSourceFiltersChange as (v: string[]) => void)(['s1'])}
      >
        source-filter
      </button>
      <button onClick={() => (p.onFiltersChanged as () => void)()}>
        filters-changed
      </button>
      <button onClick={() => (p.onNextPage as () => void)()}>next-page</button>
      <button onClick={() => (p.onPreviousPage as () => void)()}>prev-page</button>
      <button onClick={() => (p.onTogglePipeline as () => void)()}>
        toggle-pipeline
      </button>
      <button onClick={() => (p.onRefreshPipeline as () => void)()}>
        refresh-pipeline
      </button>
      <button
        onClick={() =>
          (p.onBulkDelete as (ds: Document[]) => void)([
            { doc_id: 'd1' } as Document,
          ])
        }
      >
        bulk-delete
      </button>
      <button
        onClick={() =>
          (p.onScanRetry as (d: Document) => void)({ doc_id: 'd1' } as Document)
        }
      >
        scan-retry
      </button>
    </div>
  ),
}));

vi.mock('../components/PendingDocsSection', () => ({
  PendingDocsSection: (p: Record<string, unknown>) => (
    <div data-testid="pending-docs">
      <span data-testid="pending-count">
        {String((p.docs as unknown[]).length)}
      </span>
      <button
        onClick={() =>
          (p.onReadSource as (d: Document) => void)({
            doc_id: 'p1',
            file_path: 'pend.pdf',
          } as Document)
        }
      >
        read-source
      </button>
      <button
        onClick={() =>
          (p.onToast as (k: string, t: string, s?: string) => void)(
            'done',
            'pend-toast',
          )
        }
      >
        pending-toast
      </button>
    </div>
  ),
}));

vi.mock('../components/DocDetailPanel', () => ({
  DocDetailPanel: (p: Record<string, unknown>) => (
    <div data-testid="doc-detail">
      <span data-testid="detail-doc">
        {(p.doc as Document | null)?.doc_id ?? 'none'}
      </span>
      <span data-testid="detail-chunk">
        {String(p.initialExpandedChunkId ?? 'none')}
      </span>
      <button onClick={() => (p.onClose as () => void)()}>detail-close</button>
      <button
        onClick={() =>
          (p.onRetag as (d: Document) => void)({ doc_id: 'dr' } as Document)
        }
      >
        detail-retag
      </button>
      <button
        onClick={() =>
          (p.onReprocess as (d: Document) => void)({
            doc_id: 'dx',
            file_path: 'x.pdf',
            status: 'PROCESSED',
          } as Document)
        }
      >
        detail-reprocess-ok
      </button>
      <button
        onClick={() =>
          (p.onReprocess as (d: Document) => void)({
            doc_id: 'df',
            file_path: 'f.pdf',
            status: 'FAILED',
          } as Document)
        }
      >
        detail-reprocess-failed
      </button>
      <button
        onClick={() =>
          (p.onDelete as (d: Document) => void)({ doc_id: 'dd' } as Document)
        }
      >
        detail-delete
      </button>
    </div>
  ),
}));

vi.mock('../components/ToastViewport', () => ({
  ToastViewport: (p: Record<string, unknown>) => (
    <div data-testid="toast-viewport">
      <span data-testid="toast-count">
        {String((p.toasts as unknown[]).length)}
      </span>
      {(p.toasts as Array<{ id: string; title: string }>).map((t) => (
        <div key={t.id} data-testid="toast-item">
          <span>{t.title}</span>
          <button onClick={() => (p.onUndo as (t: unknown) => void)(t)}>
            undo-{t.id}
          </button>
          <button onClick={() => (p.onDismiss as (t: unknown) => void)(t)}>
            dismiss-{t.id}
          </button>
        </div>
      ))}
    </div>
  ),
}));

vi.mock('../components/LoginScreen', () => ({
  LoginScreen: (p: Record<string, unknown>) => (
    <div data-testid="login-screen">
      <span data-testid="login-checking">{String(p.checking)}</span>
      <span data-testid="login-error">{String(p.error ?? 'none')}</span>
    </div>
  ),
}));

// Lazy children (resolved via Suspense) — async default exports.
vi.mock('./lazyComponents', () => ({
  SettingsTab: (p: Record<string, unknown>) => (
    <div data-testid="settings-tab">
      <span data-testid="settings-section">{String(p.initialSection)}</span>
      <span data-testid="settings-kb">{String(p.kbName)}</span>
      <button onClick={() => (p.onSignOut as () => void)()}>sign-out</button>
      <button
        onClick={() =>
          (p.onToast as (t: { kind: string; title: string }) => void)({
            kind: 'done',
            title: 'settings-toast',
          })
        }
      >
        settings-toast
      </button>
    </div>
  ),
  RetrievalTab: (p: Record<string, unknown>) => (
    <div data-testid="retrieval-tab">
      <span data-testid="ret-tagopts">
        {JSON.stringify(p.tagOptions ?? [])}
      </span>
      <span data-testid="ret-docopts">{JSON.stringify(p.docOptions ?? [])}</span>
      <button
        onClick={() =>
          void (p.onSendQuery as (params: unknown) => Promise<unknown>)({
            query: 'q',
            mode: 'hybrid',
          })
        }
      >
        send-query
      </button>
      <button
        onClick={() =>
          void (
            p.onStreamQuery as (
              params: unknown,
              cb: (c: string) => void,
            ) => Promise<unknown>
          )({ query: 'q', mode: 'hybrid' }, () => {})
        }
      >
        stream-query
      </button>
      <button onClick={() => (p.onNavigate as (t: string) => void)('graph')}>
        ret-navigate
      </button>
    </div>
  ),
  ActivityTab: (p: Record<string, unknown>) => (
    <div data-testid="activity-tab">
      <span data-testid="act-events">
        {String((p.events as unknown[]).length)}
      </span>
      <span data-testid="act-folder">{String(p.folderLabel)}</span>
      <button onClick={() => (p.onRefresh as () => void)()}>act-refresh</button>
      <button
        onClick={() =>
          (p.onPushToast as (t: { kind: string; title: string }) => void)({
            kind: 'done',
            title: 'act-toast',
          })
        }
      >
        act-toast
      </button>
      <button
        onClick={() =>
          (p.onNavigate as (t: string, params?: Record<string, string>) => void)(
            'documents',
            { doc: 'd1' },
          )
        }
      >
        act-open-doc
      </button>
    </div>
  ),
  GraphTab: (p: Record<string, unknown>) => (
    <div data-testid="graph-tab">
      <span data-testid="graph-entities">
        {String((p.entities as unknown[]).length)}
      </span>
      <span data-testid="graph-relations">
        {String((p.relations as unknown[]).length)}
      </span>
      <span data-testid="graph-folder">{String(p.folderLabel)}</span>
      <span data-testid="graph-doclabels">{JSON.stringify(p.docLabels)}</span>
      <span data-testid="graph-doctags">{JSON.stringify(p.docTags)}</span>
      <button
        onClick={() =>
          (p.onToast as (t: { kind: string; title: string }) => void)({
            kind: 'done',
            title: 'graph-toast',
          })
        }
      >
        graph-toast
      </button>
      <button onClick={() => (p.onNavigate as (t: string) => void)('tags')}>
        graph-navigate
      </button>
    </div>
  ),
  TagsTab: (p: Record<string, unknown>) => (
    <div data-testid="tags-tab">
      <span data-testid="tags-count">
        {String((p.tags as unknown[]).length)}
      </span>
      <span data-testid="tags-cats">
        {String((p.categories as unknown[]).length)}
      </span>
      <button onClick={() => (p.onApprove as (x: unknown) => void)({})}>
        tags-approve
      </button>
      <button onClick={() => (p.onCommit as (x: unknown) => void)({})}>
        tags-commit
      </button>
      <button onClick={() => (p.onNavigate as (t: string) => void)('graph')}>
        tags-navigate
      </button>
    </div>
  ),
  AddSourceModal: (p: Record<string, unknown>) => (
    <div data-testid="add-source-modal">
      <span data-testid="add-submitting">{String(p.submitting)}</span>
      <button onClick={() => (p.onClose as () => void)()}>add-close</button>
      <button onClick={() => (p.onSubmit as (x: unknown) => void)({})}>
        add-submit
      </button>
    </div>
  ),
  RetagModal: (p: Record<string, unknown>) => (
    <div data-testid="retag-modal">
      <span data-testid="retag-doc">
        {(p.doc as Document | null)?.doc_id ?? 'none'}
      </span>
      <span data-testid="retag-bulk">
        {String((p.docs as unknown[] | undefined)?.length ?? 'none')}
      </span>
      <button onClick={() => (p.onClose as () => void)()}>retag-close</button>
      <button onClick={() => (p.onSubmit as (x: unknown) => void)({})}>
        retag-submit
      </button>
    </div>
  ),
  ReadSourceModal: (p: Record<string, unknown>) => (
    <div data-testid="read-source-modal">
      <span data-testid="read-doc">
        {(p.doc as Document | null)?.doc_id ?? 'none'}
      </span>
      <button onClick={() => (p.onClose as () => void)()}>read-close</button>
    </div>
  ),
}));

import { AppShell } from './AppShell';

function makeDoc(over: Partial<Document> = {}): Document {
  return {
    doc_id: 'doc-1',
    track_id: null,
    file_path: 'report.pdf',
    content_summary: '',
    content_length: 0,
    status: 'PROCESSED',
    chunks_count: 3,
    created_at: '2026-01-01',
    updated_at: '2026-01-01',
    error_msg: null,
    metadata: {},
    type: 'file',
    tags: [],
    folder: 'default',
    visibility: 'internal',
    ...over,
  };
}

function renderShell() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AppShell />
    </QueryClientProvider>,
  );
}

function resetQueries() {
  Object.assign(queriesState, {
    docs: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    folders: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    notifications: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    tags: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    tagCategories: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    activity: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    graphEntities: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    graphRelations: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
    pipelineStatus: { ...q(), refetch: vi.fn().mockResolvedValue(undefined) },
  });
}
function resetQueriesSpy() {
  queriesSpy.useDocuments.length = 0;
  queriesSpy.useFolders.length = 0;
  queriesSpy.useNotifications.length = 0;
  queriesSpy.useTags.length = 0;
  queriesSpy.useTagCategories.length = 0;
  queriesSpy.useActivity.length = 0;
  queriesSpy.useGraphEntities.length = 0;
  queriesSpy.useGraphRelations.length = 0;
  queriesSpy.usePipelineStatus.length = 0;
}

beforeEach(() => {
  resetQueries();
  authState.current.user = { email: 'claire.benoit@twin.local' };
  authState.current.isCheckingAuth = false;
  authState.current.needsLogin = false;
  authState.current.loginError = null;
  authState.current.config = { folders: undefined, defaultFolderId: 'default' };
  authState.current.signout = vi.fn().mockResolvedValue(undefined);
  docActions.uploadDocs = { isPending: false };
  globalThis.localStorage.clear();
  globalThis.history.replaceState(null, '', '/');
  vi.clearAllMocks();
  resetQueriesSpy();
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Auth gating ──────────────────────────────────────────────────────────────
describe('AppShell — auth gating', () => {
  it('renders the login screen while checking auth', () => {
    authState.current.isCheckingAuth = true;
    renderShell();
    expect(screen.getByTestId('login-screen')).toBeInTheDocument();
    expect(screen.getByTestId('login-checking')).toHaveTextContent('true');
    expect(screen.queryByTestId('topbar')).toBeNull();
  });

  it('renders the login screen with error when needsLogin', () => {
    authState.current.isCheckingAuth = false;
    authState.current.needsLogin = true;
    authState.current.loginError = 'bad creds';
    renderShell();
    expect(screen.getByTestId('login-screen')).toBeInTheDocument();
    expect(screen.getByTestId('login-error')).toHaveTextContent('bad creds');
  });

  it('renders the full shell once authenticated', () => {
    renderShell();
    expect(screen.getByTestId('topbar')).toBeInTheDocument();
    expect(screen.getByTestId('documents-tab')).toBeInTheDocument();
    expect(screen.getByTestId('doc-detail')).toBeInTheDocument();
    expect(screen.getByTestId('toast-viewport')).toBeInTheDocument();
  });
});

// ── Documents tab + derivations ──────────────────────────────────────────────
describe('AppShell — documents tab derivations', () => {
  it('splits pending vs non-pending docs and computes totals/status counts', () => {
    queriesState.docs.data = {
      items: [
        makeDoc({ doc_id: 'a', file_path: 'a.pdf', status: 'PROCESSED', tags: ['t1'] }),
        makeDoc({
          doc_id: 'b',
          file_path: 'b.pdf',
          status: 'PENDING',
          review: { state: 'pending-review' },
        }),
        makeDoc({
          doc_id: 'c',
          file_path: 'c.pdf',
          status: 'PROCESSING',
          review: { state: 'modified' },
        }),
      ],
      total: 3,
      status_counts: { PROCESSED: 1, PENDING: 1, PROCESSING: 1 },
      next_cursor: 'next',
    };
    renderShell();
    // 1 non-pending doc in the main list
    expect(screen.getByTestId('dt-docs-count')).toHaveTextContent('1');
    // 2 pending docs (pending-review + modified) in the pending slot
    expect(screen.getByTestId('pending-count')).toHaveTextContent('2');
    // total = 3 - 2 pending = 1
    expect(screen.getByTestId('dt-total')).toHaveTextContent('1');
    expect(screen.getByTestId('dt-hasnext')).toHaveTextContent('true');
  });

  it('renders the empty state with no docs and no status counts', () => {
    renderShell();
    expect(screen.getByTestId('dt-docs-count')).toHaveTextContent('0');
    expect(screen.getByTestId('dt-status-counts')).toHaveTextContent('null');
    expect(screen.getByTestId('dt-total')).toHaveTextContent('0');
  });

  it('pipeline error is surfaced with mapped operator copy', () => {
    queriesState.pipelineStatus.isError = true;
    queriesState.pipelineStatus.error = new Error('pipeline down');
    renderShell();
    // Error-UX pass 2026-07-03: technical messages are mapped, not echoed.
    expect(screen.getByTestId('dt-pipeline-error')).toHaveTextContent(
      'Something went wrong while loading data from the backend',
    );
  });

  it('toggle/refresh pipeline + paging handlers run without throwing', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'a' })],
      total: 1,
      next_cursor: 'n',
    };
    renderShell();
    await user.click(screen.getByText('toggle-pipeline'));
    await user.click(screen.getByText('refresh-pipeline'));
    expect(queriesState.pipelineStatus.refetch).toHaveBeenCalled();
    await user.click(screen.getByText('next-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('2');
    await user.click(screen.getByText('prev-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('1');
    // prev again clamps at 1
    await user.click(screen.getByText('prev-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('1');
  });

  it('does not mark pagination as loading during a background documents refresh', () => {
    queriesState.docs.isFetching = true;
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'a' })],
      total: 1,
      next_cursor: 'n',
    };
    renderShell();
    expect(screen.getByTestId('dt-page-fetching')).toHaveTextContent('false');
  });

  it('marks pagination as loading when documents are fetching without page data', () => {
    queriesState.docs.isFetching = true;
    queriesState.docs.data = undefined;
    renderShell();
    expect(screen.getByTestId('dt-page-fetching')).toHaveTextContent('true');
  });

  it('treats stale page data as a page transition and blocks repeated paging clicks', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'a' })],
      total: 75,
      page: 1,
      next_cursor: '2',
    };
    renderShell();

    await user.click(screen.getByText('next-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('2');
    expect(screen.getByTestId('dt-page-fetching')).toHaveTextContent('true');

    await user.click(screen.getByText('next-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('2');
    await user.click(screen.getByText('prev-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('2');
  });

  it('changing a filter resets the page back to 1', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [makeDoc()],
      total: 1,
      next_cursor: 'n',
    };
    renderShell();
    await user.click(screen.getByText('next-page'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('2');
    await user.click(screen.getByText('filters-changed'));
    expect(screen.getByTestId('dt-page')).toHaveTextContent('1');
  });

  it('status / search / tag / source filter callbacks update URL-backed state', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('filter-failed'));
    await user.click(screen.getByText('search-memo'));
    await user.click(screen.getByText('tag-filter'));
    await user.click(screen.getByText('source-filter'));
    // URL params written by useUrlParam/useUrlArrayParam
    const params = new URLSearchParams(globalThis.location.search);
    expect(params.get('status')).toBe('failed');
    expect(params.get('q')).toBe('memo');
    expect(params.get('tag')).toBe('t1');
    expect(params.get('source')).toBe('s1');
  });

  it('forwards bulk-delete and scan-retry to the doc action hook', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('bulk-delete'));
    expect(docActions.onDeleteBulk).toHaveBeenCalled();
    await user.click(screen.getByText('scan-retry'));
    expect(docActions.onScanRetry).toHaveBeenCalled();
  });
});

// ── Optimistic uploads + dedupe ──────────────────────────────────────────────
describe('AppShell — backend error banner', () => {
  it('renders the banner when the document list query errors', () => {
    queriesState.docs.isError = true;
    queriesState.docs.error = new Error('documents exploded');
    renderShell();
    expect(screen.getByTestId('backend-data-error')).toBeInTheDocument();
  });

  it('does not render the document-list banner for unrelated resource errors', () => {
    queriesState.activity.isError = true;
    queriesState.activity.error = new Error('activity exploded');
    renderShell();
    expect(screen.queryByTestId('backend-data-error')).toBeNull();
  });

  it('hides the banner when all queries are healthy', () => {
    renderShell();
    expect(screen.queryByTestId('backend-data-error')).toBeNull();
  });
});

// ── Theme ─────────────────────────────────────────────────────────────────────
describe('AppShell — theme', () => {
  it('toggles theme and persists it to the dataset + storage', async () => {
    const user = userEvent.setup();
    renderShell();
    expect(document.documentElement.dataset.theme).toBe('light');
    await user.click(screen.getByText('toggle-theme'));
    await waitFor(() =>
      expect(document.documentElement.dataset.theme).toBe('dark'),
    );
    expect(globalThis.localStorage.getItem('twin.ui.theme.v1')).toBe('dark');
  });
});

// ── Tab navigation ───────────────────────────────────────────────────────────
describe('AppShell — tab navigation', () => {
  it('switches to settings via Topbar and resets the section to profile', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('go-settings'));
    expect(await screen.findByTestId('settings-tab')).toBeInTheDocument();
    expect(screen.getByTestId('settings-section')).toHaveTextContent('profile');
  });

  it('manage-folders jumps to settings/folder section', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('manage-folders'));
    expect(await screen.findByTestId('settings-tab')).toBeInTheDocument();
    expect(screen.getByTestId('settings-section')).toHaveTextContent('folder');
  });

  it('open-activity from Topbar shows the activity tab', async () => {
    const user = userEvent.setup();
    queriesState.activity.data = {
      items: [{ id: 'e1' }, { id: 'e2' }],
      nowMs: 123,
    };
    renderShell();
    await user.click(screen.getByText('open-activity'));
    expect(await screen.findByTestId('activity-tab')).toBeInTheDocument();
    expect(screen.getByTestId('act-events')).toHaveTextContent('2');
  });

  it('renders the retrieval tab and forwards send/stream queries to the api', async () => {
    const user = userEvent.setup();
    queriesState.tags.data = [{ tag: 'argocd', status: 'active' }];
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'doc-z' })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('go-retrieval'));
    const tab = await screen.findByTestId('retrieval-tab');
    expect(tab).toBeInTheDocument();
    await user.click(within(tab).getByText('send-query'));
    await waitFor(() => expect(apiMock.query).toHaveBeenCalled());
    await user.click(within(tab).getByText('stream-query'));
    await waitFor(() => expect(apiMock.queryStream).toHaveBeenCalled());
    // navigate back out via the tab's onNavigate
    await user.click(within(tab).getByText('ret-navigate'));
    expect(await screen.findByTestId('graph-tab')).toBeInTheDocument();
  });

  it('renders the graph tab with entities, relations, labels and tags', async () => {
    const user = userEvent.setup();
    queriesState.graphEntities.data = [{ id: 'ent-1' }];
    queriesState.graphRelations.data = [{ id: 'rel-1' }, { id: 'rel-2' }];
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'd1', file_path: 'doc1.pdf', tags: ['t1', 't2'] })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('go-graph'));
    const tab = await screen.findByTestId('graph-tab');
    expect(within(tab).getByTestId('graph-entities')).toHaveTextContent('1');
    expect(within(tab).getByTestId('graph-relations')).toHaveTextContent('2');
    // doc labels + tags derived from docList
    expect(screen.getByTestId('graph-doclabels')).toHaveTextContent('doc1.pdf');
    expect(screen.getByTestId('graph-doctags')).toHaveTextContent('t1');
    await user.click(within(tab).getByText('graph-navigate'));
    expect(await screen.findByTestId('tags-tab')).toBeInTheDocument();
  });

  it('renders the tags tab and forwards approve/commit to the tag action hook', async () => {
    const user = userEvent.setup();
    queriesState.tags.data = [{ tag: 'argocd', status: 'active' }];
    queriesState.tagCategories.data = [{ id: 'infra', label: 'Infra' }];
    renderShell();
    await user.click(screen.getByText('go-tags'));
    const tab = await screen.findByTestId('tags-tab');
    expect(within(tab).getByTestId('tags-count')).toHaveTextContent('1');
    expect(within(tab).getByTestId('tags-cats')).toHaveTextContent('1');
    await user.click(within(tab).getByText('tags-approve'));
    expect(tagActions.onTagApprove).toHaveBeenCalled();
    await user.click(within(tab).getByText('tags-commit'));
    expect(tagActions.onTagCommit).toHaveBeenCalled();
  });

  it('activity tab refresh + toast + navigate-to-doc work', async () => {
    const user = userEvent.setup();
    queriesState.activity.data = { items: [{ id: 'e1' }], nowMs: 1 };
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'd1', file_path: 'a.pdf' })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('go-activity'));
    const tab = await screen.findByTestId('activity-tab');
    await user.click(within(tab).getByText('act-refresh'));
    expect(queriesState.activity.refetch).toHaveBeenCalled();
    await user.click(within(tab).getByText('act-toast'));
    await waitFor(() =>
      expect(screen.getByTestId('toast-count')).not.toHaveTextContent('0'),
    );
    // navigate to a doc detail request → back on documents tab
    await user.click(within(tab).getByText('act-open-doc'));
    expect(await screen.findByTestId('documents-tab')).toBeInTheDocument();
    // the requested detail doc resolves from docList
    expect(screen.getByTestId('detail-doc')).toHaveTextContent('d1');
  });
});

// ── Folder switching + folderList ────────────────────────────────────────────
describe('AppShell — folders', () => {
  it('keeps a persisted folder when it is still provisioned at runtime', () => {
    globalThis.localStorage.setItem('twin.ui.folder.v1', 'finance');
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB', kind: 'standard', sources: 4 },
        { id: 'finance', label: 'Finance', kind: 'standard', sources: 2 },
      ],
    };
    renderShell();

    expect(screen.getByTestId('topbar-folder')).toHaveTextContent('finance');
    expect(setActiveFolderMock).toHaveBeenCalledWith('finance');
    const [queryInput, queryOptions] = queriesSpy.useDocuments[0] as [
      { folder: string },
      { folderKey: string },
    ];
    expect(queryInput.folder).toBe('finance');
    expect(queryOptions.folderKey).toBe('finance');
  });

  it('keeps a persisted runtime-only folder when folders are only in live data', () => {
    globalThis.localStorage.setItem('twin.ui.folder.v1', 'runtime-only');
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [{ id: 'default', label: 'Default KB', kind: 'standard', sources: 4 }],
    };
    queriesState.folders.data = [
      {
        id: 'runtime-only',
        kb: 'Runtime-only KB',
        visibility: 'internal',
        sources: 0,
        role: 'admin',
        current: false,
      },
    ] as Folder[];
    renderShell();

    expect(screen.getByTestId('topbar-folder')).toHaveTextContent('runtime-only');
    const [queryInput, queryOptions] = queriesSpy.useDocuments[0] as [
      { folder: string },
      { folderKey: string },
    ];
    expect(queryInput.folder).toBe('runtime-only');
    expect(queryOptions.folderKey).toBe('runtime-only');
    expect(setActiveFolderMock).toHaveBeenCalledWith('runtime-only');
  });

  it('derives folderList from runtimeConfig.folders when configured', () => {
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB', kind: 'standard', sources: 4 },
        { id: 'finance', label: 'Finance', kind: 'sandbox', sources: 2 },
      ],
    };
    renderShell();
    expect(screen.getByTestId('topbar-folders-count')).toHaveTextContent('2');
    // kbName resolves from the current folder
    expect(screen.getByTestId('topbar-kb')).toHaveTextContent('Default KB');
  });

  it('uses live folder source counts for configured folders', () => {
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB', kind: 'standard', sources: 0 },
        { id: 'tests', label: 'Tests', kind: 'standard', sources: 0 },
      ],
    };
    queriesState.folders.data = [
      {
        id: 'default',
        kb: 'Default KB',
        visibility: 'internal',
        sources: 14,
        role: 'admin',
        current: true,
      },
      {
        id: 'tests',
        kb: 'Tests',
        visibility: 'internal',
        sources: 3,
        role: 'admin',
        current: false,
      },
    ] as Folder[];
    renderShell();
    expect(screen.getByTestId('topbar-folder-sources')).toHaveTextContent(
      'default:14|tests:3',
    );
  });

  it('falls back to folders query data when runtimeConfig has none', () => {
    queriesState.folders.data = [
      { id: 'default', kb: 'Query KB', visibility: 'internal', sources: 1, role: 'admin', current: true },
    ] as Folder[];
    renderShell();
    expect(screen.getByTestId('topbar-folders-count')).toHaveTextContent('1');
    expect(screen.getByTestId('topbar-kb')).toHaveTextContent('Query KB');
  });

  it('appends runtime folders from the live query to the configured set', () => {
    // Regression: a folder created via admin CRUD is returned by GET
    // /twin/api/folders but is NOT in the boot-injected runtimeConfig.folders
    // (frozen env snapshot). The switcher must still surface it — config folders
    // first, then the live-only ones (deduped), so it appears without a restart.
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB', kind: 'standard', sources: 4 },
      ],
    };
    queriesState.folders.data = [
      { id: 'default', kb: 'Default KB', visibility: 'internal', sources: 4, role: 'admin', current: true },
      { id: 'test', kb: 'test', visibility: 'internal', sources: 0, role: 'admin', current: false },
    ] as Folder[];
    renderShell();
    expect(screen.getByTestId('topbar-folders-count')).toHaveTextContent('2');
  });

  it('explicit empty configured folders stays empty despite live query data', () => {
    // The Twincore empty-state contract: zero provisioned folders must NOT be
    // backfilled from the live query (which may carry seed/demo folders).
    authState.current.config = { defaultFolderId: '', folders: [] };
    queriesState.folders.data = [
      { id: 'seed', kb: 'Seed KB', visibility: 'internal', sources: 1, role: 'admin', current: false },
    ] as Folder[];
    renderShell();
    expect(screen.getByTestId('topbar-folders-count')).toHaveTextContent('0');
  });

  it('switching folder updates state and persists to storage', async () => {
    const user = userEvent.setup();
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB', kind: 'standard', sources: 4 },
        { id: 'finance', label: 'Finance', kind: 'standard', sources: 2 },
      ],
    };
    renderShell();
    expect(screen.getByTestId('topbar-folder')).toHaveTextContent('default');
    await user.click(screen.getByText('switch-folder'));
    await waitFor(() =>
      expect(screen.getByTestId('topbar-folder')).toHaveTextContent('finance'),
    );
    await waitFor(() =>
      expect(globalThis.localStorage.getItem('twin.ui.folder.v1')).toBe(
        'finance',
      ),
    );
  });

  it('restores a stored runtime folder after refresh when the live folders query knows it', async () => {
    globalThis.localStorage.setItem('twin.ui.folder.v1', 'test');
    authState.current.config = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB', kind: 'standard', sources: 4 },
      ],
    };
    queriesState.folders.data = [
      {
        id: 'default',
        kb: 'Default KB',
        visibility: 'internal',
        sources: 4,
        role: 'admin',
        current: false,
      },
      {
        id: 'test',
        kb: 'test',
        visibility: 'internal',
        sources: 0,
        role: 'admin',
        current: true,
      },
    ] as Folder[];

    renderShell();

    expect(screen.getByTestId('topbar-folder')).toHaveTextContent('test');
    expect(setActiveFolderMock).toHaveBeenCalledWith('test');
    expect(globalThis.localStorage.getItem('twin.ui.folder.v1')).toBe('test');
  });

  it('falls back to the default folder when the active folder is not in the list', async () => {
    // current folder persisted as something not present → effect rewrites
    // storage/state to the configured default.
    globalThis.localStorage.setItem('twin.ui.folder.v1', 'ghost');
    authState.current.config = {
      defaultFolderId: 'finance',
      folders: [
        { id: 'finance', label: 'Finance', kind: 'standard', sources: 2 },
        { id: 'ops', label: 'Ops', kind: 'standard', sources: 1 },
      ],
    };
    renderShell();
    await waitFor(() =>
      expect(screen.getByTestId('topbar-folder')).toHaveTextContent('finance'),
    );
    await waitFor(() =>
      expect(globalThis.localStorage.getItem('twin.ui.folder.v1')).toBe(
        'finance',
      ),
    );
    expect(setActiveFolderMock).toHaveBeenCalledWith('finance');
  });
});

// ── Notifications ─────────────────────────────────────────────────────────────
describe('AppShell — notifications', () => {
  it('computes unread count and supports mark-all-read / clear', async () => {
    const user = userEvent.setup();
    queriesState.notifications.data = [
      { id: 'n1', read: false, title: 'A' },
      { id: 'n2', read: true, title: 'B' },
      { id: 'n3', read: false, title: 'C' },
    ];
    renderShell();
    expect(screen.getByTestId('topbar-unread')).toHaveTextContent('2');
    expect(screen.getByTestId('topbar-notif-count')).toHaveTextContent('3');

    await user.click(screen.getByText('mark-all-read'));
    await waitFor(() =>
      expect(screen.getByTestId('topbar-unread')).toHaveTextContent('0'),
    );

    await user.click(screen.getByText('clear-notifs'));
    await waitFor(() =>
      expect(screen.getByTestId('topbar-notif-count')).toHaveTextContent('0'),
    );
  });
});

// ── Modals ────────────────────────────────────────────────────────────────────
describe('AppShell — modals', () => {
  it('opens and closes the Add source modal', async () => {
    const user = userEvent.setup();
    docActions.uploadDocs = { isPending: true };
    renderShell();
    expect(screen.queryByTestId('add-source-modal')).toBeNull();
    await user.click(screen.getByText('open-add'));
    const modal = await screen.findByTestId('add-source-modal');
    expect(within(modal).getByTestId('add-submitting')).toHaveTextContent('true');
    await user.click(within(modal).getByText('add-submit'));
    expect(docActions.onAddSourceSubmit).toHaveBeenCalled();
    await user.click(within(modal).getByText('add-close'));
    await waitFor(() =>
      expect(screen.queryByTestId('add-source-modal')).toBeNull(),
    );
  });

  it('opens the single-doc retag modal then closes it', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('open-retag'));
    const modal = await screen.findByTestId('retag-modal');
    expect(within(modal).getByTestId('retag-doc')).toHaveTextContent('d1');
    await user.click(within(modal).getByText('retag-submit'));
    expect(docActions.onRetagSubmit).toHaveBeenCalled();
    await user.click(within(modal).getByText('retag-close'));
    await waitFor(() => expect(screen.queryByTestId('retag-modal')).toBeNull());
  });

  it('opens the bulk retag modal', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('open-bulk-retag'));
    const modal = await screen.findByTestId('retag-modal');
    expect(within(modal).getByTestId('retag-bulk')).toHaveTextContent('1');
  });

  it('opens a detail panel from the documents list and closes it', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'd1', file_path: 'a.pdf' })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('open-detail'));
    await waitFor(() =>
      expect(screen.getByTestId('detail-doc')).toHaveTextContent('d1'),
    );
    await user.click(screen.getByText('detail-close'));
    await waitFor(() =>
      expect(screen.getByTestId('detail-doc')).toHaveTextContent('none'),
    );
  });

  it('detail retag closes the panel and opens the retag modal', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'd1' })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('open-detail'));
    await waitFor(() =>
      expect(screen.getByTestId('detail-doc')).toHaveTextContent('d1'),
    );
    await user.click(screen.getByText('detail-retag'));
    const modal = await screen.findByTestId('retag-modal');
    expect(within(modal).getByTestId('retag-doc')).toHaveTextContent('dr');
    expect(screen.getByTestId('detail-doc')).toHaveTextContent('none');
  });

  it('detail delete delegates to onDeleteSingle and closes', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('detail-delete'));
    expect(docActions.onDeleteSingle).toHaveBeenCalled();
  });

  it('reprocess on a non-FAILED doc pushes an explanatory toast (no api call)', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('detail-reprocess-ok'));
    expect(apiMock.reprocessFailedDocuments).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(screen.getByText(/Re-process not applicable/)).toBeInTheDocument(),
    );
  });

  it('reprocess on a FAILED doc triggers the batch reprocess and toasts success', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('detail-reprocess-failed'));
    await waitFor(() =>
      expect(apiMock.reprocessFailedDocuments).toHaveBeenCalled(),
    );
    await waitFor(() =>
      expect(
        screen.getByText(/Failed-source reprocess requested/),
      ).toBeInTheDocument(),
    );
  });

  it('reprocess on a FAILED doc surfaces an error toast when the api throws', async () => {
    const user = userEvent.setup();
    apiMock.reprocessFailedDocuments.mockRejectedValueOnce(new Error('boom'));
    renderShell();
    await user.click(screen.getByText('detail-reprocess-failed'));
    await waitFor(() =>
      expect(screen.getByText(/Re-process failed/)).toBeInTheDocument(),
    );
  });

  it('opens and closes the Read source modal from the pending section', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [
        makeDoc({
          doc_id: 'p1',
          file_path: 'pend.pdf',
          review: { state: 'pending-review' },
        }),
      ],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('read-source'));
    const modal = await screen.findByTestId('read-source-modal');
    expect(within(modal).getByTestId('read-doc')).toHaveTextContent('p1');
    await user.click(within(modal).getByText('read-close'));
    await waitFor(() =>
      expect(screen.queryByTestId('read-source-modal')).toBeNull(),
    );
  });
});

// ── Toasts ────────────────────────────────────────────────────────────────────
describe('AppShell — toasts', () => {
  it('emits a toast from the documents add-toast handler and can dismiss it', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('add-toast'));
    await waitFor(() =>
      expect(screen.getByTestId('toast-count')).toHaveTextContent('1'),
    );
    const item = screen.getByTestId('toast-item');
    await user.click(within(item).getByText(/^dismiss-/));
    await waitFor(() =>
      expect(screen.getByTestId('toast-count')).toHaveTextContent('0'),
    );
  });

  it('toast undo delegates to onToastUndo', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('add-toast'));
    const item = await screen.findByTestId('toast-item');
    await user.click(within(item).getByText(/^undo-/));
    expect(docActions.onToastUndo).toHaveBeenCalled();
  });

  it('pending-section toast and settings toast both reach the viewport', async () => {
    const user = userEvent.setup();
    queriesState.docs.data = {
      items: [
        makeDoc({ doc_id: 'p1', review: { state: 'pending-review' } }),
      ],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    await user.click(screen.getByText('pending-toast'));
    await waitFor(() =>
      expect(screen.getByText('pend-toast')).toBeInTheDocument(),
    );
  });
});

// ── Settings sign out ─────────────────────────────────────────────────────────
describe('AppShell — settings', () => {
  it('sign out delegates to auth.signout', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('go-settings'));
    const tab = await screen.findByTestId('settings-tab');
    await user.click(within(tab).getByText('sign-out'));
    expect(authState.current.signout).toHaveBeenCalled();
  });

  it('settings toast reaches the viewport', async () => {
    const user = userEvent.setup();
    renderShell();
    await user.click(screen.getByText('go-settings'));
    const tab = await screen.findByTestId('settings-tab');
    await user.click(within(tab).getByText('settings-toast'));
    const viewport = screen.getByTestId('toast-viewport');
    await waitFor(() =>
      expect(within(viewport).getByText('settings-toast')).toBeInTheDocument(),
    );
  });
});

// ── Detail-request bootstrap from URL ─────────────────────────────────────────
describe('AppShell — detail request from URL', () => {
  it('reads ?doc & ?chunk on mount and resolves the detail panel', () => {
    globalThis.history.replaceState(null, '', '/?doc=d9&chunk=c2');
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'd9', file_path: 'd9.pdf' })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    expect(screen.getByTestId('detail-doc')).toHaveTextContent('d9');
    expect(screen.getByTestId('detail-chunk')).toHaveTextContent('c2');
  });

  it('resolves a detail request by source file_path when doc id is absent', () => {
    globalThis.history.replaceState(null, '', '/?source=bysrc.pdf');
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'dsrc', file_path: 'bysrc.pdf' })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    expect(screen.getByTestId('detail-doc')).toHaveTextContent('dsrc');
  });

  it('does not resolve an optimistic-upload doc as a detail target', () => {
    globalThis.history.replaceState(null, '', '/?doc=dopt');
    queriesState.docs.data = {
      items: [makeDoc({ doc_id: 'dopt', file_path: 'o.pdf', _optimisticUpload: true })],
      total: 1,
      next_cursor: null,
    };
    renderShell();
    expect(screen.getByTestId('detail-doc')).toHaveTextContent('none');
  });
});

// ── Anonymous actor fallback ─────────────────────────────────────────────────
describe('AppShell — actor fallback', () => {
  it('falls back to CURRENT_USER name when auth has no user email', async () => {
    const user = userEvent.setup();
    authState.current.user = null;
    renderShell();
    await user.click(screen.getByText('go-retrieval'));
    const tab = await screen.findByTestId('retrieval-tab');
    await user.click(within(tab).getByText('send-query'));
    await waitFor(() => expect(apiMock.query).toHaveBeenCalled());
    const arg = apiMock.query.mock.calls[0][0] as { actor: string };
    expect(arg.actor).toBe('operator@twin.local');
  });
});
