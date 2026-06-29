/**
 * App shell — wires Topbar + tabs + modals against the TanStack Query layer.
 *
 * Data flow (S4a):
 *   - Each resource has a typed query hook (`useDocuments`, `useTags`, ...)
 *     that hits `/documents`, `/tags`, etc. via `apiFetch`.
 *   - Backend failures render an explicit error instead of local sample data.
 *   - Components keep their prop-driven signature so unit tests pass arrays
 *     directly without a QueryClient wrapper.
 *
 * Env switches:
 *   - globalThis.__twinConfig  → server-injected API bases + current identity.
 *   - VITE_USE_MSW=false   → skip the MSW worker.
 *   - VITE_API_BASE_URL=…  → optional dev/test backend origin fallback.
 *   - VITE_AUTH_TOKEN=…    → optional dev/test bearer fallback.
 */

import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { DocDetailPanel } from '../components/DocDetailPanel';
import { DocumentsTab } from '../components/DocumentsTab';
import { PendingDocsSection } from '../components/PendingDocsSection';
import { LoginScreen } from '../components/LoginScreen';
import { ToastViewport } from '../components/ToastViewport';
import { Topbar } from '../components/Topbar';
import type { SettingsSectionKey } from '../components/SettingsTab';
import { useAuth } from '../hooks/useAuth';
import { canManageFolders } from '../lib/permissions';
import { useUrlArrayParam, useUrlParam } from '../hooks/useUrlParam';
import {
  useActivity,
  useDocuments,
  useGraphEntities,
  useGraphRelations,
  useNotifications,
  usePipelineStatus,
  useTagCategories,
  useTags,
  useFolders,
} from '../api/queries';
import { setActiveFolder } from '../api/client';
import { api, type ActivityQuery } from '../api/resources';
import { mapTwinQueryResponseForRetrievalTab } from '../api/twinQueryResponse';
import { FORMAT_CATEGORIES } from '../constants/formatCategories';
import type { Document } from '../types/document';
import type { Theme, Folder } from '../types/topbar';
import { dedupeDocumentsBySource } from '../utils/documents';
import { tagCatalogForSuggestions } from '../utils/tags';
import { formatBackendError, resourceError } from './appErrors';
import {
  CURRENT_USER,
  DOCUMENTS_STATUS_FILTERS,
  DOCUMENTS_STATUS_TO_API,
  type DocumentsStatusFilterKey,
} from './appConstants';
import {
  ActivityTab,
  AddSourceModal,
  GraphTab,
  ReadSourceModal,
  RetagModal,
  RetrievalTab,
  SettingsTab,
  TagsTab,
} from './lazyComponents';
import {
  FOLDER_STORAGE_KEY,
  THEME_STORAGE_KEY,
  getInitialFolderId,
  getInitialTheme,
  writeUiPreference,
} from './uiPreferences';
import { type DetailRequest, useAppNavigation } from './useAppNavigation';
import { useDocumentActions } from './useDocumentActions';
import { useTagActions } from './useTagActions';
import { useToasts } from './useToasts';

const ACTIVITY_PAGE_LIMIT = 200;

declare global {
  interface Window {
    __TWIN_E2E_INITIAL_TAG_POLL?: {
      intervalMs?: number;
      maxPolls?: number;
    };
  }
}

export function AppShell() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme());
  const [settingsSection, setSettingsSection] =
    useState<SettingsSectionKey>('profile');
  const [folder, setFolder] = useState(() => {
    const initial = getInitialFolderId();
    setActiveFolder(initial);
    return initial;
  });
  const { toasts, setToasts, pushToast, dismissToast } = useToasts();

  // Modal state
  const [addOpen, setAddOpen] = useState(false);
  const [retagDoc, setRetagDoc] = useState<Document | null>(null);
  const [retagBulk, setRetagBulk] = useState<readonly Document[] | null>(null);
  const [detailDoc, setDetailDoc] = useState<Document | null>(null);
  const [detailChunkId, setDetailChunkId] = useState<string | null>(null);
  const [detailRequest, setDetailRequest] = useState<DetailRequest | null>(() => {
    const params = new URLSearchParams(globalThis.location.search);
    const doc = params.get('doc') ?? undefined;
    const source = params.get('source') ?? undefined;
    if (!doc && !source) return null;
    return {
      doc,
      source,
      chunk: params.get('chunk') ?? undefined,
    };
  });
  const [readSourceDoc, setReadSourceDoc] = useState<Document | null>(null);
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const [documentsStatusFilter, setDocumentsStatusFilter] =
    useUrlParam<DocumentsStatusFilterKey>('status', 'all', {
      validate: (v) =>
        (DOCUMENTS_STATUS_FILTERS as readonly string[]).includes(v),
    });
  const [documentsSearch, setDocumentsSearch] = useUrlParam<string>('q', '');
  const [documentsTagFilters, setDocumentsTagFilters] = useUrlArrayParam('tag', []);
  const [documentsSourceFilters, setDocumentsSourceFilters] =
    useUrlArrayParam('source', []);
  const [documentsPagination, setDocumentsPagination] = useState<{
    scope: string;
    page: number;
  }>(() => ({ scope: '', page: 1 }));
  const [optimisticUploadDocs, setOptimisticUploadDocs] = useState<
    readonly Document[]
  >([]);
  const documentsStatusParam =
    documentsStatusFilter === 'all'
      ? undefined
      : DOCUMENTS_STATUS_TO_API[documentsStatusFilter];
  const documentsSearchParam = documentsSearch.trim() || undefined;
  const documentsTagParam = documentsTagFilters[0] || undefined;
  const documentsPageScope = [
    folder,
    documentsStatusFilter,
    documentsSearchParam ?? '',
    documentsTagFilters.join(','),
    documentsSourceFilters.join(','),
  ].join(':');
  const documentsPage =
    documentsPagination.scope === documentsPageScope
      ? documentsPagination.page
      : 1;
  const setDocumentsPageForScope = (
    updater: number | ((page: number) => number),
  ) => {
    setDocumentsPagination((prev) => {
      const currentPage =
        prev.scope === documentsPageScope ? prev.page : 1;
      const nextPage =
        typeof updater === 'function' ? updater(currentPage) : updater;
      return {
        scope: documentsPageScope,
        page: Math.max(1, nextPage),
      };
    });
  };

  // Auth
  const auth = useAuth();
  const runtimeConfig = auth.config;
  const currentActor = auth.user?.email ?? CURRENT_USER.name;
  const authReady = !auth.isCheckingAuth && !auth.needsLogin;
  const retagOpen = retagDoc !== null || retagBulk !== null;

  // Data — every visible resource comes from the API query layer. No local
  // sample fallback is allowed on the operator surface.
  const docs = useDocuments(
    {
      folder,
      cursor: documentsPage > 1 ? String(documentsPage) : undefined,
      status: documentsStatusParam,
      q: documentsSearchParam,
      tag: documentsTagParam,
    },
    {
      folderKey: folder,
      enabled:
        authReady &&
        (tab === 'documents' || tab === 'retrieval' || tab === 'graph'),
    },
  );
  const folders = useFolders({ enabled: authReady });
  const notificationsQ = useNotifications({ enabled: authReady, folderKey: folder });
  // Twin overlay tag surfaces stay always-enabled (vs. tab-gated): both
  // are lightweight, the catalog is used cross-tab (badge counts, filter
  // pickers, retag modal), and the e2e contract on "switching folder
  // rescopes /twin/api/tags immediately" depends on the query existing
  // in the cache for `refetchQueries` to trigger. Gating heavy reads
  // (documents, graph) preserves the bulk of the perf win.
  const tags = useTags({ enabled: authReady, folderKey: folder });
  const tagCategories = useTagCategories({ enabled: authReady, folderKey: folder });
  const [activityQuery, setActivityQuery] = useState<ActivityQuery>({
    range: '7d',
    limit: ACTIVITY_PAGE_LIMIT,
  });
  const handleActivityQueryChange = useCallback((next: ActivityQuery) => {
    setActivityQuery((current) =>
      JSON.stringify(current) === JSON.stringify(next) ? current : next,
    );
  }, []);
  // Activity stays always-enabled (vs. tab-gated): the feed drives the
  // topbar unread counters cross-tab, and the e2e contract requires
  // `/twin/api/activity` to refire under the new folder header at switch
  // time. Lightweight read (bounded via `limit`), so the perf cost is
  // negligible compared to documents / graph which remain gated.
  const activity = useActivity(activityQuery, { enabled: authReady, folderKey: folder });
  const graphEntities = useGraphEntities({
    enabled: authReady && tab === 'graph',
    folderKey: folder,
  });
  const graphRelations = useGraphRelations({
    enabled: authReady && tab === 'graph',
    folderKey: folder,
  });
  const pipelineStatus = usePipelineStatus({
    enabled: authReady && tab === 'documents',
    folderKey: folder,
  });

  // Notifications carry mutable client state (read/cleared) on top of the
  // query data. Keep only local overrides in React state so refetches can
  // merge without an effect-driven mirror.
  const [readNotificationIds, setReadNotificationIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [clearedNotificationIds, setClearedNotificationIds] = useState<
    ReadonlySet<string>
  >(() => new Set());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writeUiPreference(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const notificationSource = notificationsQ.data ?? [];
  const notifications = notificationSource
    .filter((notification) => !clearedNotificationIds.has(notification.id))
    .map((notification) =>
      readNotificationIds.has(notification.id)
        ? { ...notification, read: true }
        : notification,
    );
  const unreadCount = notifications.filter((n) => !n.read).length;
  const configuredFolders = runtimeConfig.folders;
  const folderList = useMemo<readonly Folder[]>(() => {
    if (configuredFolders) {
      // Explicit empty config = no folder provisioned for this KB → the topbar
      // shows the Twincore empty-state guidance, never live folders.
      if (configuredFolders.length === 0) return [];
      const mapped: Folder[] = configuredFolders.map((item) => ({
        id: item.id,
        kb: item.label,
        visibility: item.kind === 'sandbox' ? 'private' : 'internal',
        sources: item.sources ?? 0,
        role: 'admin / steward',
        current: item.id === folder,
      }));
      // The boot-injected config is frozen at server start, so operator-created
      // runtime folders are missing from it. Append the ones the live
      // /twin/api/folders query knows about (deduped by id) so they reach the
      // switcher without a service restart.
      const known = new Set(configuredFolders.map((item) => item.id));
      const extra = (folders.data ?? [])
        .filter((item) => !known.has(item.id))
        .map((item) => ({ ...item, current: item.id === folder }));
      return [...mapped, ...extra];
    }
    return folders.data ?? [];
  }, [configuredFolders, folder, folders]);
  const kbName = folderList.find((w) => w.id === folder)?.kb ?? '';

  useEffect(() => {
    if (folderList.length === 0) return;
    if (folderList.some((item) => item.id === folder)) {
      writeUiPreference(FOLDER_STORAGE_KEY, folder);
      return;
    }
    const fallback =
      folderList.find((item) => item.id === runtimeConfig.defaultFolderId)?.id ??
      folderList[0].id;
    writeUiPreference(FOLDER_STORAGE_KEY, fallback);
  }, [folder, folderList, runtimeConfig.defaultFolderId]);

  const onAddToast = (title: string, sub?: string) =>
    pushToast({ kind: 'done', title, sub });

  const {
    uploadDocs,
    onAddSourceSubmit,
    onDeleteBulk,
    onDeleteSingle,
    onRetagSubmit,
    onScanRetry,
    onToastUndo,
  } = useDocumentActions({
    activity,
    currentActor,
    docs,
    folder,
    folderList,
    pushToast,
    setAddOpen,
    setOptimisticUploadDocs,
    setToasts,
  });

  const { onTagApprove, onTagCommit } = useTagActions({ currentActor, pushToast });

  const { onNavigate, onSwitchFolder } = useAppNavigation({
    setClearedNotificationIds,
    setDetailChunkId,
    setDetailDoc,
    setDetailRequest,
    setDocumentsSearch,
    setDocumentsSourceFilters,
    setDocumentsStatusFilter,
    setDocumentsTagFilters,
    setFolderState: setFolder,
    setReadNotificationIds,
    setReadSourceDoc,
    setRetagBulk,
    setRetagDoc,
    setTab,
  });

  const documentsError = resourceError('Documents', docs);

  // Resolved props. Do not silently fall back to local samples: empty arrays
  // + the document error banner make document-list failures visible instead
  // of showing stale sample data.
  const backendDocList = useMemo(
    () => docs.data?.items ?? [],
    [docs.data],
  );
  const docList = useMemo(() => {
    const backendTrackIds = new Set(
      backendDocList.map((doc) => doc.track_id).filter(Boolean),
    );
    const backendKeys = new Set(
      backendDocList.map((doc) => `${doc.folder}:${doc.file_path}`),
    );
    const pendingUploads = optimisticUploadDocs.filter(
      (doc) =>
        doc.folder === folder &&
        !(
          (doc.track_id && backendTrackIds.has(doc.track_id)) ||
          backendKeys.has(`${doc.folder}:${doc.file_path}`)
        ),
    );
    return dedupeDocumentsBySource([...pendingUploads, ...backendDocList]);
  }, [backendDocList, folder, optimisticUploadDocs]);
  const requestedDetailDoc = useMemo(() => {
    if (tab !== 'documents' || !detailRequest) return null;
    const target =
      docList.find((doc) => doc.doc_id === detailRequest.doc) ??
      docList.find((doc) => doc.file_path === detailRequest.source);
    if (!target || target._optimisticUpload) return null;
    return target;
  }, [detailRequest, docList, tab]);
  const activeDetailDoc = detailDoc ?? requestedDetailDoc;
  const activeDetailChunkId =
    detailDoc === null ? (detailRequest?.chunk ?? null) : detailChunkId;
  // Pending = "needs reviewer attention", covers both first-time approval
  // (pending-review) AND Confluence/SharePoint upstream-edit re-validation
  // (modified — upstream re-validation spec). Sort so pending-review cards come
  // first, modified second, to keep the reviewer's eye on new arrivals.
  const graphDocLabels = useMemo(() => {
    const labels: Record<string, string> = {};
    docList.forEach((d) => {
      if (d.file_path) labels[d.doc_id] = d.file_path;
    });
    return labels;
  }, [docList]);
  const graphDocTags = useMemo(() => {
    const tags: Record<string, readonly string[]> = {};
    docList.forEach((d) => {
      if (d.tags.length > 0) {
        tags[d.doc_id] = d.tags;
        if (d.file_path) tags[d.file_path] = d.tags;
      }
    });
    return tags;
  }, [docList]);
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
  const uploadedStatusCounts = useMemo(() => {
    const raw = docs.data?.status_counts;
    if (!raw) return null;
    const counts: Record<string, number> = { ...raw };
    pendingDocs.forEach((doc) => {
      const keys = [doc.status, doc.status.toLowerCase()];
      keys.forEach((key) => {
        if (typeof counts[key] === 'number') {
          counts[key] = Math.max(0, counts[key] - 1);
        }
      });
    });
    return counts;
  }, [docs.data?.status_counts, pendingDocs]);
  const uploadedTotalCount = Math.max(
    0,
    (docs.data?.total ?? backendDocList.length) - pendingDocs.length,
  );
  const tagList = tags.data ?? [];
  const tagCatalog = tagCatalogForSuggestions(tagList);
  const tagCategoryList = tagCategories.data ?? [];
  const activityEvents = activity.data?.items ?? [];
  const activityNow = activity.data?.nowMs;
  const activityTotal = activity.data?.total ?? activityEvents.length;
  const graphEntityList = graphEntities.data ?? [];
  const graphRelationList = graphRelations.data ?? [];

  if (auth.isCheckingAuth || auth.needsLogin) {
    return (
      <LoginScreen
        checking={auth.isCheckingAuth}
        error={auth.loginError}
        onLogin={auth.login}
      />
    );
  }

  return (
    <div className="app">
      <Topbar
        tab={tab}
        onTab={(nextTab) => {
          if (nextTab === 'settings') setSettingsSection('profile');
          setTab(nextTab);
        }}
        theme={theme}
        onTheme={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
        folder={folder}
        kbName={kbName}
        onSwitchFolder={(w) => onSwitchFolder(w.id)}
        folders={folderList}
        notifications={notifications}
        unreadCount={unreadCount}
        onMarkAllRead={() =>
          setReadNotificationIds(
            new Set(notificationSource.map((notification) => notification.id)),
          )
        }
        onClearNotifications={() =>
          setClearedNotificationIds(
            new Set(notificationSource.map((notification) => notification.id)),
          )
        }
        onOpenActivity={() => setTab('activity')}
        onManageFolders={() => {
          setSettingsSection('folder');
          setTab('settings');
        }}
      />
      {documentsError && (
        <div className="sys-banner-stack" role="status" aria-live="polite">
          <div className="sys-banner sys-info" data-testid="backend-data-error">
            <span className="sys-banner-ico" aria-hidden="true">
              i
            </span>
            <div className="sys-banner-body">
              <div className="sys-banner-line1">
                <span className="sys-banner-title">Data temporarily unavailable</span>
                <span className="sys-banner-sub">
                  Refresh the page or sign in again if the document list does not return.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
      <main
        tabIndex={-1}
        data-focus-fallback="app-main"
        style={{
          flex: 1,
          overflow: 'hidden',
          display: 'flex',
          position: 'relative',
        }}
      >
        <div className="tab-pane" key={`${tab}:${folder}`}>
          <Suspense fallback={<div className="tab-loading" aria-live="polite" />}>
          {tab === 'documents' && (
            <DocumentsTab
              docs={nonPendingDocs}
              currentPage={documentsPage}
              totalCount={uploadedTotalCount}
              statusCounts={uploadedStatusCounts}
              hasNextPage={Boolean(docs.data?.next_cursor)}
              isPageFetching={Boolean(docs.isFetching && !docs.data)}
              statusFilter={documentsStatusFilter}
              onStatusFilterChange={setDocumentsStatusFilter}
              search={documentsSearch}
              onSearchChange={setDocumentsSearch}
              tagFilters={documentsTagFilters}
              onTagFiltersChange={setDocumentsTagFilters}
              sourceFilters={documentsSourceFilters}
              onSourceFiltersChange={setDocumentsSourceFilters}
              onFiltersChanged={() => setDocumentsPageForScope(1)}
              onPreviousPage={() =>
                setDocumentsPageForScope((page) => page - 1)
              }
              onNextPage={() => setDocumentsPageForScope((page) => page + 1)}
              activeFolder={folder}
              canManageFolders={canManageFolders(auth.user)}
              folderList={folderList}
              tagCatalog={tagCatalog}
              pendingSlot={
                <PendingDocsSection
                  docs={pendingDocs}
                  actor={auth.user?.email ?? 'anonymous'}
                  defaultOpen
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
              onOpenDetail={(d) => {
                setDetailRequest(null);
                setDetailChunkId(null);
                setDetailDoc(d);
              }}
              onBulkDelete={onDeleteBulk}
              onScanRetry={onScanRetry}
              pipelineStatus={pipelineStatus.data ?? null}
              pipelineOpen={pipelineOpen}
              pipelineLoading={pipelineStatus.isFetching}
              pipelineError={
                pipelineStatus.isError
                  ? formatBackendError(pipelineStatus.error)
                  : null
              }
              onTogglePipeline={() => setPipelineOpen((open) => !open)}
              onRefreshPipeline={() => {
                void pipelineStatus.refetch();
              }}
            />
          )}
          {tab === 'settings' && (
            <SettingsTab
              activeFolder={folder}
              kbName={kbName}
              initialSection={settingsSection}
              onSignOut={() => {
                void auth.signout();
              }}
              onToast={pushToast}
            />
          )}
          {tab === 'retrieval' && (
            <RetrievalTab
              onSendQuery={async (params) => {
                const res = await api.query({
                  query: params.query,
                  actor: currentActor,
                  mode: params.mode,
                  top_k: params.topK,
                  chunk_top_k: params.chunkTopK,
                  max_total_tokens: params.maxTokens,
                  history_turns: params.historyTurns,
                  conversation_history: params.conversationHistory,
                  only_need_context: params.onlyContext,
                  only_need_prompt: params.onlyPrompt,
                  user_prompt: params.userPrompt,
                  enable_rerank: params.enableRerank,
                  min_score: params.minScore,
                  tag_filter: params.tagFilter,
                  doc_filter: params.docFilter,
                });
                return mapTwinQueryResponseForRetrievalTab(res);
              }}
              onStreamQuery={async (params, onChunk) => {
                const res = await api.queryStream(
                  {
                    query: params.query,
                    actor: currentActor,
                    mode: params.mode,
                    top_k: params.topK,
                    chunk_top_k: params.chunkTopK,
                    max_total_tokens: params.maxTokens,
                    history_turns: params.historyTurns,
                    conversation_history: params.conversationHistory,
                    only_need_context: params.onlyContext,
                    only_need_prompt: params.onlyPrompt,
                    user_prompt: params.userPrompt,
                    enable_rerank: params.enableRerank,
                    min_score: params.minScore,
                    tag_filter: params.tagFilter,
                    doc_filter: params.docFilter,
                  },
                  onChunk,
                );
                return mapTwinQueryResponseForRetrievalTab(res);
              }}
              initialThreads={[]}
              suggestions={[]}
              tagOptions={tagCatalog.map((tag) => tag.tag)}
              docOptions={docList.map((doc) => doc.doc_id)}
              docLabels={graphDocLabels}
              onNavigate={onNavigate}
              activeFolder={folder}
            />
          )}
          {tab === 'activity' && (
            <ActivityTab
              events={activityEvents}
              total={activityTotal}
              nowMs={activityNow}
              folderLabel={kbName || folder}
              density="comfortable"
              live={true}
              onPushToast={pushToast}
              onNavigate={onNavigate}
              onRefresh={() => activity.refetch()}
              onQueryChange={handleActivityQueryChange}
              limit={activityQuery.limit ?? ACTIVITY_PAGE_LIMIT}
            />
          )}
          {tab === 'graph' && (
            <GraphTab
              entities={graphEntityList}
              relations={graphRelationList}
              docLabels={graphDocLabels}
              docTags={graphDocTags}
              tagCatalog={tagCatalog.map((tag) => tag.tag)}
              folderLabel={kbName || folder}
              onNavigate={onNavigate}
              onToast={pushToast}
            />
          )}
          {tab === 'tags' && (
            <TagsTab
              tags={tagList}
              categories={tagCategoryList}
              currentUser={CURRENT_USER}
              folderLabel={kbName || folder}
              defaultPendingOpen
              onApprove={onTagApprove}
              onCommit={onTagCommit}
              onNavigate={onNavigate}
            />
          )}
          </Suspense>
        </div>
      </main>

      <Suspense fallback={null}>
        {addOpen && (
          <AddSourceModal
            open={addOpen}
            tagCatalog={tagCatalog}
            formatCategories={FORMAT_CATEGORIES}
            submitting={uploadDocs.isPending}
            onClose={() => setAddOpen(false)}
            onSubmit={onAddSourceSubmit}
          />
        )}
        {retagOpen && (
          <RetagModal
            open={retagOpen}
            doc={retagDoc}
            docs={retagBulk ?? undefined}
            tagCatalog={tagCatalog}
            onClose={() => {
              setRetagDoc(null);
              setRetagBulk(null);
            }}
            onSubmit={onRetagSubmit}
          />
        )}
      </Suspense>
      <DocDetailPanel
        doc={activeDetailDoc}
        initialExpandedChunkId={activeDetailChunkId}
        onClose={() => {
          setDetailDoc(null);
          setDetailChunkId(null);
          setDetailRequest(null);
        }}
        onRetag={(d) => {
          setDetailDoc(null);
          setDetailChunkId(null);
          setDetailRequest(null);
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
                title: 'Failed-source reprocess requested',
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
          setDetailChunkId(null);
          setDetailRequest(null);
          void onDeleteSingle(d);
        }}
      />
      <Suspense fallback={null}>
        {readSourceDoc && (
          <ReadSourceModal
            doc={readSourceDoc}
            onClose={() => setReadSourceDoc(null)}
          />
        )}
      </Suspense>
      <ToastViewport
        toasts={toasts}
        onUndo={(t) => {
          void onToastUndo(t);
        }}
        onDismiss={dismissToast}
      />
    </div>
  );
}
