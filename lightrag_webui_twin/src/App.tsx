/**
 * App shell — wires Topbar + tabs + modals against the TanStack Query layer.
 *
 * Data flow (S4a):
 *   - Each resource has a typed query hook (`useDocuments`, `useTags`, ...)
 *     that hits `/documents`, `/tags`, etc. via `apiFetch`.
 *   - In dev, MSW intercepts those fetches and answers from the fixtures —
 *     the contract template lives in `src/fixtures/`.
 *   - Each `useQuery` is seeded with `initialData = FIXTURE` so the first
 *     paint is instant. Fetched data replaces the fixture as soon as the
 *     query resolves (background revalidation pattern).
 *   - Components keep their prop-driven signature so unit tests pass arrays
 *     directly without a QueryClient wrapper.
 *
 * Env switches:
 *   - VITE_USE_MSW=false   → skip the MSW worker; fetches hit VITE_API_BASE_URL.
 *   - VITE_API_BASE_URL=…  → real backend origin.
 *   - VITE_AUTH_TOKEN=…    → bearer attached on every fetch.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { ActivityTab } from './components/ActivityTab';
import { AddSourceModal, type AddSourceAction } from './components/AddSourceModal';
import { DocDetailPanel } from './components/DocDetailPanel';
import { DocumentsTab } from './components/DocumentsTab';
import { GraphTab } from './components/GraphTab';
import { OnboardingWizard } from './components/OnboardingWizard';
import { PendingDocsSection } from './components/PendingDocsSection';
import { ReadSourceModal } from './components/ReadSourceModal';
import { RetagModal, type RetagAction } from './components/RetagModal';
import { RetrievalTab } from './components/RetrievalTab';
import { SettingsTab } from './components/SettingsTab';
import { TagsTab, type TagApproveAction } from './components/TagsTab';
import type { TagActionCommit } from './components/TagActionModal';
import { ToastViewport } from './components/ToastViewport';
import { Topbar } from './components/Topbar';
import { useAuth } from './hooks/useAuth';
import { useOnboarding } from './hooks/useOnboarding';
import {
  useActivity,
  useApproveTag,
  useBulkRetagDocuments,
  useDeleteDocument,
  useUploadDocument,
  useDeleteTag,
  useDeprecateTag,
  useDocuments,
  useEditTag,
  useGraphEntities,
  useGraphRelations,
  useNotifications,
  useRejectTag,
  useRequestTag,
  useTagCategories,
  useTags,
  useThesaurus,
  useUpdateTagSynonyms,
  useWorkspaces,
} from './api/queries';
import { api } from './api/resources';
import {
  ACTIVITY_FIXTURES,
  ACTIVITY_NOW_MS,
  ANSWER_TOKENS_FIXTURE,
  DOCUMENT_FIXTURES,
  FORMAT_CATEGORY_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  NOTIFICATION_FIXTURES,
  RETRIEVAL_SOURCES_FIXTURE,
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  THESAURUS_FIXTURES,
  WORKSPACE_FIXTURES,
  makeSampleThreads,
} from './fixtures';
import type { Document } from './types/document';
import type { TagCurrentUser } from './types/tag';
import type { Theme } from './types/topbar';
import type { Toast } from './types/toast';

const CURRENT_USER: TagCurrentUser = {
  name: 'claire.benoit',
  palier: 3,
  role: 'admin / steward',
};

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppShell() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>('light');
  const [workspace, setWorkspace] = useState('cib');
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Modal state
  const [addOpen, setAddOpen] = useState(false);
  const [retagDoc, setRetagDoc] = useState<Document | null>(null);
  const [retagBulk, setRetagBulk] = useState<readonly Document[] | null>(null);
  const [detailDoc, setDetailDoc] = useState<Document | null>(null);
  const [readSourceDoc, setReadSourceDoc] = useState<Document | null>(null);

  // Auth + onboarding
  const auth = useAuth();
  const onboarding = useOnboarding();
  const onboardingOpen = !onboarding.state.dismissed;

  // Data — every tab is backed by a query, seeded with the corresponding
  // fixture so first paint is instant even if the worker is still booting.
  const docs = useDocuments();
  const workspaces = useWorkspaces();
  const notificationsQ = useNotifications();
  const thesaurus = useThesaurus();
  const tags = useTags();
  const tagCategories = useTagCategories();
  const activity = useActivity();
  const graphEntities = useGraphEntities();
  const graphRelations = useGraphRelations();

  // Notifications carry mutable client state (read/cleared) on top of the
  // query data, so we mirror them locally and use the query result as the
  // source of truth on first load + refetch.
  const [notifications, setNotifications] = useState([...NOTIFICATION_FIXTURES]);
  useEffect(() => {
    if (notificationsQ.data) setNotifications([...notificationsQ.data]);
  }, [notificationsQ.data]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const unreadCount = notifications.filter((n) => !n.read).length;
  const workspaceList = workspaces.data ?? WORKSPACE_FIXTURES;
  const kbName = workspaceList.find((w) => w.id === workspace)?.kb ?? '';

  const pushToast = (t: Omit<Toast, 'id'>) => {
    const id = `tst_${Date.now()}_${Math.random().toString(16).slice(2, 6)}`;
    setToasts((ts) => [...ts, { id, ...t }]);
  };

  const onAddToast = (title: string, sub?: string) =>
    pushToast({ kind: 'done', title, sub });

  const bulkRetagDocs = useBulkRetagDocuments();

  const onRetagSubmit = async (action: RetagAction) => {
    const verb = action.adds.length > 0 ? 'applied' : 'removed';
    const sample = action.adds[0] ?? action.removes[0];
    try {
      const result = await bulkRetagDocs.mutateAsync({
        targets: action.targets.map((d) => d.doc_id),
        adds: action.adds,
        removes: action.removes,
        actor: CURRENT_USER.name,
      });
      const failedCount = result.failed.length;
      pushToast({
        kind: 'done',
        title: 'Tag',
        tagname: sample,
        titleSuffix: action.bulk
          ? `${verb} to ${result.updated} source${result.updated === 1 ? '' : 's'}${
              failedCount > 0 ? ` · ${failedCount} skipped` : ''
            }`
          : verb,
        sub: action.primary.file_path,
        undo: { adds: action.adds, removes: action.removes },
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Tag mutation failed',
        sub:
          err instanceof Error
            ? err.message
            : `Could not persist tags on ${action.targets.length} document${
                action.targets.length === 1 ? '' : 's'
              }`,
      });
    }
  };

  const uploadDoc = useUploadDocument();
  const deleteDoc = useDeleteDocument();

  const onDeleteSingle = async (doc: { doc_id: string; file_path: string }) => {
    try {
      await deleteDoc.mutateAsync(doc.doc_id);
      pushToast({
        kind: 'done',
        title: 'Document deleted',
        sub: `${doc.file_path} — removed from Memgraph (cascade: chunks + entities + relations)`,
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Delete failed',
        sub:
          err instanceof Error
            ? err.message
            : `Could not delete ${doc.file_path}`,
      });
    }
  };

  const onDeleteBulk = async (
    docs: readonly { doc_id: string; file_path: string }[],
  ) => {
    if (docs.length === 0) return;
    pushToast({
      kind: 'propagating',
      title: 'Deleting sources…',
      sub: `${docs.length} source${docs.length === 1 ? '' : 's'} → DELETE /documents/{id}`,
    });
    const results = await Promise.allSettled(
      docs.map((d) => deleteDoc.mutateAsync(d.doc_id)),
    );
    const ok = results.filter((r) => r.status === 'fulfilled').length;
    const ko = results.filter((r) => r.status === 'rejected').length;
    pushToast({
      kind: ko === 0 ? 'done' : 'error',
      title:
        ko === 0
          ? `${ok} source${ok === 1 ? '' : 's'} deleted`
          : `${ko} delete${ko === 1 ? '' : 's'} failed`,
      sub: `${ok} ok · ${ko} ko`,
    });
  };

  const onAddSourceSubmit = async (action: AddSourceAction) => {
    if (action.rawFiles.length === 0) {
      // No raw files (e.g. test path or all entries were URLs which
      // are gated coming-soon). Acknowledge without server round-trip.
      pushToast({
        kind: 'done',
        title: 'Sources queued',
        sub: `${action.readyCount} entr${action.readyCount === 1 ? 'y' : 'ies'}`,
      });
      return;
    }

    pushToast({
      kind: 'propagating',
      title: 'Uploading sources…',
      sub: `${action.rawFiles.length} file${action.rawFiles.length === 1 ? '' : 's'} → LightRAG /documents/upload`,
    });

    const results = await Promise.allSettled(
      action.rawFiles.map((f) => uploadDoc.mutateAsync(f)),
    );

    const ok = results.filter((r) => r.status === 'fulfilled').length;
    const ko = results.filter((r) => r.status === 'rejected').length;
    const dup = results.filter(
      (r) =>
        r.status === 'fulfilled' &&
        (r.value as { status?: string }).status === 'duplicated',
    ).length;

    if (ko === 0) {
      pushToast({
        kind: 'done',
        title: 'Sources queued for ingestion',
        sub: `${ok} uploaded${dup > 0 ? ` (${dup} already present)` : ''}${
          action.tags.length
            ? ' · initial tags will apply once docs land'
            : ''
        }`,
      });
    } else {
      pushToast({
        kind: 'error',
        title: `${ko} upload${ko === 1 ? '' : 's'} failed`,
        sub: `${ok} ok · ${ko} ko${
          dup > 0 ? ` · ${dup} already present` : ''
        }`,
      });
    }

    // Auto-tag-on-upload: when the operator typed tags in the modal,
    // poll /documents/track_status/{track_id} until each upload's doc
    // lands as 'processed', then fire a bulk-retag with the saved
    // tags. We don't block the main toast on this — it runs in the
    // background and pushes a follow-up toast on completion.
    if (action.tags.length > 0) {
      const successfulTrackIds = results
        .filter((r) => r.status === 'fulfilled')
        .map((r) => (r as PromiseFulfilledResult<{ track_id: string }>).value.track_id)
        .filter(Boolean);
      if (successfulTrackIds.length > 0) {
        void applyInitialTagsAfterIngestion(
          successfulTrackIds,
          action.tags,
        );
      }
    }
  };

  /**
   * Poll track_status for each track_id (up to ~60s with 2s intervals)
   * and, once the corresponding doc reaches 'processed' (or 'failed'),
   * collect its doc_id and fire a single bulk-retag with the operator's
   * initial tags. Survives across page refreshes only for the duration
   * of the modal interaction (lost on reload — intentional, the operator
   * can retag manually if the page is closed mid-ingestion).
   */
  const applyInitialTagsAfterIngestion = async (
    trackIds: readonly string[],
    tags: readonly string[],
  ): Promise<void> => {
    const POLL_INTERVAL_MS = 2000;
    const MAX_POLLS = 30;
    const TERMINAL_STATUSES = new Set([
      'processed',
      'PROCESSED',
      'failed',
      'FAILED',
    ]);
    const resolvedDocIds = new Set<string>();
    const pending = new Set(trackIds);
    for (let i = 0; i < MAX_POLLS && pending.size > 0; i++) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      for (const tid of Array.from(pending)) {
        try {
          const status = await api.trackStatus(tid);
          const terminalDocs = status.documents.filter((d) =>
            TERMINAL_STATUSES.has(d.status),
          );
          if (
            terminalDocs.length > 0 &&
            terminalDocs.length === status.documents.length
          ) {
            terminalDocs
              .filter((d) => d.status.toLowerCase() === 'processed')
              .forEach((d) => resolvedDocIds.add(d.id));
            pending.delete(tid);
          }
        } catch {
          // 404 from track_status is fine while LightRAG is still
          // booking the doc — keep polling until MAX_POLLS.
        }
      }
    }
    if (resolvedDocIds.size === 0) {
      if (trackIds.length > 0) {
        pushToast({
          kind: 'error',
          title: 'Initial tags not applied',
          sub: `Ingestion didn't reach a terminal state within ${(MAX_POLLS * POLL_INTERVAL_MS) / 1000}s. Retag manually once the docs land.`,
        });
      }
      return;
    }
    try {
      await bulkRetagDocs.mutateAsync({
        targets: Array.from(resolvedDocIds),
        adds: tags,
        removes: [],
        actor: CURRENT_USER.name,
      });
      pushToast({
        kind: 'done',
        title: 'Initial tags applied',
        sub: `${resolvedDocIds.size} doc${resolvedDocIds.size === 1 ? '' : 's'} · tags: ${tags.join(', ')}`,
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Initial tags failed',
        sub:
          err instanceof Error ? err.message : 'bulk-retag returned an error',
      });
    }
  };

  // Tag mutations — call the backend through TanStack Query. Each mutation
  // invalidates ['tags']+['activity']+['notifications'] on success so the
  // operator sees the new state + audit event + notification on refetch.
  // The host still pushes a local toast so the action feels instant; the
  // toast title mirrors what the backend would emit as `last_edit.action`.
  const requestTag = useRequestTag();
  const approveTag = useApproveTag();
  const rejectTag = useRejectTag();
  const editTag = useEditTag();
  const deprecateTag = useDeprecateTag();
  const updateSynonyms = useUpdateTagSynonyms();
  const deleteTag = useDeleteTag();

  const onTagApprove = (action: TagApproveAction) => {
    approveTag.mutate({ name: action.tag.tag, actor: CURRENT_USER.name });
    pushToast({
      kind: 'done',
      title: 'Tag',
      tagname: action.tag.tag,
      titleSuffix: 'approved',
      sub: 'Added to thesaurus · Tier 3',
    });
  };

  const onTagCommit = (commit: TagActionCommit) => {
    const tagname = commit.tag?.tag ?? commit.name ?? '';
    const actor = CURRENT_USER.name;
    switch (commit.kind) {
      case 'edit':
        editTag.mutate({ name: tagname, actor });
        break;
      case 'suggest':
        // No backend endpoint for "suggest" yet — surface as a request.
        if (commit.tag) {
          requestTag.mutate({
            tag: commit.tag.tag,
            def: commit.tag.def,
            category: commit.tag.category,
            actor,
            justification: 'suggested edit',
          });
        }
        break;
      case 'synonyms':
        if (commit.tag) {
          updateSynonyms.mutate({
            name: tagname,
            aliases: commit.newSynonym
              ? [...commit.tag.aliases, commit.newSynonym]
              : commit.tag.aliases,
            actor,
          });
        }
        break;
      case 'deprecate':
        deprecateTag.mutate({ name: tagname, actor });
        break;
      case 'delete':
        deleteTag.mutate({
          name: tagname,
          strategy: commit.migrate?.strategy ?? 'untag',
          to: commit.migrate?.to,
          actor,
        });
        break;
      case 'reject':
        rejectTag.mutate({
          name: tagname,
          reason: commit.reason || 'rejected',
          actor,
        });
        break;
      case 'edit-approve':
        approveTag.mutate({ name: tagname, actor });
        break;
      case 'request':
        if (commit.name) {
          requestTag.mutate({
            tag: commit.name,
            def: commit.tag?.def ?? '',
            category: commit.tag?.category ?? 'infra',
            actor,
          });
        }
        break;
    }
    // Local toast so the action feels instant; the synthesized backend event
    // also lands on /activity which the Activity tab will surface on refetch.
    const verbMap: Record<TagActionCommit['kind'], string> = {
      edit: 'definition updated',
      suggest: 'edit suggested',
      synonyms: 'synonyms updated',
      deprecate: 'deprecated',
      delete:
        commit.migrate?.strategy === 'migrate'
          ? `migrated to ${commit.migrate.to ?? ''}`
          : 'deleted (docs untagged)',
      reject: 'rejected',
      'edit-approve': 'approved (edited)',
      request: 'requested for review',
    };
    pushToast({
      kind: 'done',
      title: 'Tag',
      tagname,
      titleSuffix: verbMap[commit.kind],
      sub: commit.reason ?? '',
    });
  };

  const onNavigate = (nextTab: string, params?: Record<string, string>) => {
    const search = new URLSearchParams(window.location.search);
    Array.from(search.keys()).forEach((k) => search.delete(k));
    if (params) {
      Object.entries(params).forEach(([k, v]) => search.set(k, v));
    }
    const qs = search.toString();
    window.history.replaceState(
      null,
      '',
      window.location.pathname + (qs ? '?' + qs : ''),
    );
    setTab(nextTab);
  };

  // Resolved props — fall back to the local fixtures while the first fetch is
  // in flight so the UI never shows an empty shell on cold start.
  const docList = docs.data?.items ?? DOCUMENT_FIXTURES;
  // Pending = "needs reviewer attention", covers both first-time approval
  // (pending-review) AND Confluence/SharePoint upstream-edit re-validation
  // (modified — Fabrice 2026-05-26 spec). Sort so pending-review cards come
  // first, modified second, to keep the reviewer's eye on new arrivals.
  const isPendingReview = (d: Document) =>
    d.review?.state === 'pending-review' || d.review?.state === 'modified';
  const pendingDocs = docList
    .filter(isPendingReview)
    .slice()
    .sort(
      (a, b) =>
        (a.review!.state === 'modified' ? 1 : 0) -
        (b.review!.state === 'modified' ? 1 : 0),
    );
  const nonPendingDocs = docList.filter((d) => !isPendingReview(d));
  const thesaurusList = thesaurus.data ?? THESAURUS_FIXTURES;
  const tagList = tags.data ?? TAG_FIXTURES;
  const tagCategoryList = tagCategories.data ?? TAG_CATEGORY_FIXTURES;
  const activityEvents = activity.data?.items ?? ACTIVITY_FIXTURES;
  const activityNow = activity.data?.nowMs ?? ACTIVITY_NOW_MS;
  const graphEntityList = graphEntities.data ?? GRAPH_ENTITY_FIXTURES;
  const graphRelationList = graphRelations.data ?? GRAPH_RELATION_FIXTURES;

  return (
    <div className="app">
      <Topbar
        tab={tab}
        onTab={setTab}
        theme={theme}
        onTheme={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
        workspace={workspace}
        kbName={kbName}
        onSwitchWorkspace={(w) => setWorkspace(w.id)}
        workspaces={workspaceList}
        notifications={notifications}
        unreadCount={unreadCount}
        onMarkAllRead={() =>
          setNotifications((ns) => ns.map((n) => ({ ...n, read: true })))
        }
        onClearNotifications={() => setNotifications([])}
      />
      <main
        style={{
          flex: 1,
          overflow: 'hidden',
          display: 'flex',
          position: 'relative',
        }}
      >
        <div className="tab-pane" key={tab}>
          {tab === 'documents' && (
            <DocumentsTab
              docs={nonPendingDocs}
              thesaurus={thesaurusList}
              pendingSlot={
                <PendingDocsSection
                  docs={pendingDocs}
                  actor={auth.user?.email ?? 'anonymous'}
                  onReadSource={(d) => setReadSourceDoc(d)}
                  onToast={(kind, title, sub) =>
                    pushToast({ kind, title, sub })
                  }
                />
              }
              onOpenAdd={() => setAddOpen(true)}
              onOpenRetag={(d) => setRetagDoc(d)}
              onOpenBulkRetag={(ds) => setRetagBulk(ds)}
              onAddToast={onAddToast}
              onDeleteDoc={(d) => setDetailDoc(d)}
              onBulkDelete={onDeleteBulk}
            />
          )}
          {tab === 'settings' && (
            <SettingsTab
              activeWorkspace={workspace}
              kbName={kbName}
              onSignOut={() => {
                void (async () => {
                  try {
                    await api.logout();
                  } catch {
                    // Even if the endpoint hiccups, we want to clear
                    // the local state and force the operator back to
                    // the Basic Auth prompt — never block sign-out
                    // on a server error.
                  }
                  // Drop all cached React Query state (tags, docs,
                  // activity, …). The next request after reload will
                  // either re-prompt Basic Auth (current model) or
                  // hit the JWT/IdP login (future).
                  queryClient.clear();
                  window.location.reload();
                })();
              }}
              onRestartTutorial={() =>
                pushToast({
                  kind: 'done',
                  title: 'Tutorial restarted',
                  sub: 'Welcome modal will appear · 0 of 6 steps complete',
                })
              }
            />
          )}
          {tab === 'retrieval' && (
            <RetrievalTab
              thesaurus={thesaurusList}
              answerTokens={ANSWER_TOKENS_FIXTURE}
              answerSources={RETRIEVAL_SOURCES_FIXTURE}
              initialThreads={makeSampleThreads()}
            />
          )}
          {tab === 'activity' && (
            <ActivityTab
              events={activityEvents}
              nowMs={activityNow}
              density="comfortable"
              live={true}
              onPushToast={pushToast}
              onNavigate={onNavigate}
            />
          )}
          {tab === 'graph' && (
            <GraphTab
              entities={graphEntityList}
              relations={graphRelationList}
              onNavigate={onNavigate}
            />
          )}
          {tab === 'tags' && (
            <TagsTab
              tags={tagList}
              categories={tagCategoryList}
              currentUser={CURRENT_USER}
              onApprove={onTagApprove}
              onCommit={onTagCommit}
              onNavigate={onNavigate}
            />
          )}
        </div>
      </main>

      <AddSourceModal
        open={addOpen}
        thesaurus={thesaurusList}
        formatCategories={FORMAT_CATEGORY_FIXTURES}
        onClose={() => setAddOpen(false)}
        onSubmit={onAddSourceSubmit}
      />
      <RetagModal
        open={retagDoc !== null || retagBulk !== null}
        doc={retagDoc}
        docs={retagBulk ?? undefined}
        thesaurus={thesaurusList}
        onClose={() => {
          setRetagDoc(null);
          setRetagBulk(null);
        }}
        onSubmit={onRetagSubmit}
      />
      <DocDetailPanel
        doc={detailDoc}
        onClose={() => setDetailDoc(null)}
        onRetag={(d) => {
          setDetailDoc(null);
          setRetagDoc(d);
        }}
        onReprocess={(d) => {
          // LightRAG 1.4.9.11 has no per-doc-by-id reprocess (only a
          // global /documents/reprocess_failed batch). Surface the
          // honest semantics rather than fake a per-doc success:
          //   - status FAILED → trigger the batch (this doc gets in)
          //   - status anything else → no-op + explain
          const failed =
            String(d.status).toLowerCase() === 'failed' ||
            String(d.status).toUpperCase() === 'FAILED';
          if (!failed) {
            pushToast({
              kind: 'done',
              title: 'Re-process not applicable',
              sub: `${d.file_path} is "${d.status}". LightRAG re-process targets the FAILED batch only. To force a refresh: delete + re-upload.`,
            });
            return;
          }
          void (async () => {
            try {
              const r = await api.reprocessFailedDocuments();
              pushToast({
                kind: 'done',
                title: 'Re-process queued (failed batch)',
                sub: `${r.message ?? 'LightRAG is retrying all FAILED docs'} · ${d.file_path} included`,
              });
            } catch (err) {
              pushToast({
                kind: 'error',
                title: 'Re-process failed',
                sub: err instanceof Error ? err.message : String(err),
              });
            }
          })();
        }}
        onDelete={(d) => {
          setDetailDoc(null);
          void onDeleteSingle(d);
        }}
      />
      <ReadSourceModal
        doc={readSourceDoc}
        onClose={() => setReadSourceDoc(null)}
      />
      <OnboardingWizard
        open={onboardingOpen}
        onAddSource={() => setAddOpen(true)}
        onGoToRetrieval={() => setTab('retrieval')}
      />
      <ToastViewport
        toasts={toasts}
        onUndo={(t) =>
          setToasts((ts) => ts.filter((x) => x.id !== t.id))
        }
        onDismiss={(t) =>
          setToasts((ts) => ts.filter((x) => x.id !== t.id))
        }
      />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  );
}

export default App;
