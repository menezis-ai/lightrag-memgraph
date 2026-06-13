/**
 * App shell — wires Topbar + tabs + modals against the TanStack Query layer.
 *
 * Data flow (S4a):
 *   - Each resource has a typed query hook (`useDocuments`, `useTags`, ...)
 *     that hits `/documents`, `/tags`, etc. via `apiFetch`.
 *   - In dev/MSW demo mode, unresolved queries may display fixtures so first
 *     paint stays useful while the worker boots.
 *   - In production real-backend mode, local fixture fallbacks are disabled:
 *     backend failures render an explicit error instead of stale CIB data.
 *   - Components keep their prop-driven signature so unit tests pass arrays
 *     directly without a QueryClient wrapper.
 *
 * Env switches:
 *   - window.__twinConfig  → server-injected API bases + current identity.
 *   - VITE_USE_MSW=false   → skip the MSW worker.
 *   - VITE_API_BASE_URL=…  → optional dev/test backend origin fallback.
 *   - VITE_AUTH_TOKEN=…    → optional dev/test bearer fallback.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import type { AddSourceAction } from './components/AddSourceModal';
import { DocDetailPanel } from './components/DocDetailPanel';
import { DocumentsTab } from './components/DocumentsTab';
import { PendingDocsSection } from './components/PendingDocsSection';
import { LoginScreen } from './components/LoginScreen';
import type { RetagAction } from './components/RetagModal';
import type { TagApproveAction } from './components/TagsTab';
import type { TagActionCommit } from './components/TagActionModal';
import { ToastViewport } from './components/ToastViewport';
import { Topbar } from './components/Topbar';
import type { SettingsSectionKey } from './components/SettingsTab';
import { useAuth } from './hooks/useAuth';
import {
  useActivity,
  useApproveTag,
  useBulkDeleteDocuments,
  useBulkRetagDocuments,
  useDeleteDocument,
  useUploadDocumentsBatch,
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
  useUpdateTagSynonyms,
  useFolders,
} from './api/queries';
import { ApiError, getTwinRuntimeConfig, setActiveFolder } from './api/client';
import { api } from './api/resources';
import {
  ACTIVITY_FIXTURES,
  ACTIVITY_NOW_MS,
  DOCUMENT_FIXTURES,
  FORMAT_CATEGORY_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  NOTIFICATION_FIXTURES,
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  FOLDER_FIXTURES,
  makeSampleThreads,
} from './fixtures';
import type { Document } from './types/document';
import type { TagCurrentUser } from './types/tag';
import type { Theme, Folder } from './types/topbar';
import { TOAST_AUTO_DISMISS_MS, type Toast } from './types/toast';
import { dedupeDocumentsBySource } from './utils/documents';
import { tagCatalogForSuggestions } from './utils/tags';

// Fallback identity when no auth backend resolves a user (open-access /
// LightRAG-parity deployments). Matches the backend's anonymous actor
// label — never a fictional demo persona.
const CURRENT_USER: TagCurrentUser = {
  name: 'operator@twin.local',
  palier: 3,
  role: 'admin / steward',
};

// Keep the default Documents surface eager, but split secondary tabs and
// modal bodies out of the entry bundle so first paint does less JS work.
const ActivityTab = lazy(() =>
  import('./components/ActivityTab').then(({ ActivityTab }) => ({
    default: ActivityTab,
  })),
);
const AddSourceModal = lazy(() =>
  import('./components/AddSourceModal').then(({ AddSourceModal }) => ({
    default: AddSourceModal,
  })),
);
const GraphTab = lazy(() =>
  import('./components/GraphTab').then(({ GraphTab }) => ({ default: GraphTab })),
);
const ReadSourceModal = lazy(() =>
  import('./components/ReadSourceModal').then(({ ReadSourceModal }) => ({
    default: ReadSourceModal,
  })),
);
const RetagModal = lazy(() =>
  import('./components/RetagModal').then(({ RetagModal }) => ({
    default: RetagModal,
  })),
);
const RetrievalTab = lazy(() =>
  import('./components/RetrievalTab').then(({ RetrievalTab }) => ({
    default: RetrievalTab,
  })),
);
const SettingsTab = lazy(() =>
  import('./components/SettingsTab').then(({ SettingsTab }) => ({
    default: SettingsTab,
  })),
);
const TagsTab = lazy(() =>
  import('./components/TagsTab').then(({ TagsTab }) => ({ default: TagsTab })),
);

const FIXTURE_FALLBACK_ENABLED = shouldUseFixtureFallback({
  dev: Boolean(import.meta.env.DEV),
  forceMsw: import.meta.env.VITE_FORCE_MSW,
  useMsw: import.meta.env.VITE_USE_MSW,
});

// eslint-disable-next-line react-refresh/only-export-components -- pure helper used by App; moving it costs more than the HMR full-reload it triggers.
export function shouldUseFixtureFallback(env: {
  dev: boolean;
  forceMsw?: string;
  useMsw?: string;
}): boolean {
  if (env.forceMsw === 'true') return true;
  return env.dev && env.useMsw !== 'false';
}

type QueryLike<T> = {
  data?: T;
  isError: boolean;
  isLoading: boolean;
  error: unknown;
};

interface BackendResourceError {
  label: string;
  message: string;
}

function resolveQueryData<T>(query: QueryLike<T>, fixture: T): T | undefined {
  return query.data ?? (FIXTURE_FALLBACK_ENABLED ? fixture : undefined);
}

function resourceError<T>(
  label: string,
  query: QueryLike<T>,
): BackendResourceError | null {
  if (FIXTURE_FALLBACK_ENABLED || query.data || query.isLoading || !query.isError) {
    return null;
  }
  return { label, message: formatBackendError(query.error) };
}

function formatBackendError(error: unknown): string {
  if (error instanceof ApiError) return `${error.status} ${error.message}`;
  if (error instanceof Error) return error.message;
  return 'Backend request failed';
}

declare global {
  interface Window {
    __TWIN_E2E_INITIAL_TAG_POLL?: {
      intervalMs?: number;
      maxPolls?: number;
    };
    __TWIN_E2E_QUERY_CLIENT?: QueryClient;
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

if (typeof window !== 'undefined' && import.meta.env.DEV) {
  window.__TWIN_E2E_QUERY_CLIENT = queryClient;
}

function getInitialFolderId(): string {
  const cfg = getTwinRuntimeConfig();
  return (
    cfg.defaultFolderId ||
    cfg.folders?.[0]?.id ||
    'default'
  );
}

function AppShell() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>('light');
  const [settingsSection, setSettingsSection] =
    useState<SettingsSectionKey>('profile');
  const [folder, setFolderState] = useState(() => {
    const initial = getInitialFolderId();
    setActiveFolder(initial);
    return initial;
  });
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Modal state
  const [addOpen, setAddOpen] = useState(false);
  const [retagDoc, setRetagDoc] = useState<Document | null>(null);
  const [retagBulk, setRetagBulk] = useState<readonly Document[] | null>(null);
  const [detailDoc, setDetailDoc] = useState<Document | null>(null);
  const [readSourceDoc, setReadSourceDoc] = useState<Document | null>(null);
  const [optimisticUploadDocs, setOptimisticUploadDocs] = useState<
    readonly Document[]
  >([]);

  // Auth
  const auth = useAuth();
  const runtimeConfig = auth.config;
  const currentActor = auth.user?.email ?? CURRENT_USER.name;
  const authReady = !auth.isCheckingAuth && !auth.needsLogin;
  const retagOpen = retagDoc !== null || retagBulk !== null;

  // Data — every tab is backed by a query, seeded with the corresponding
  // fixture so first paint is instant even if the worker is still booting.
  const docs = useDocuments(
    { folder },
    { enabled: authReady && tab === 'documents' },
  );
  const folders = useFolders({ enabled: authReady });
  const notificationsQ = useNotifications({ enabled: authReady });
  // Twin overlay tag surfaces stay always-enabled (vs. tab-gated): both
  // are lightweight, the catalog is used cross-tab (badge counts, filter
  // pickers, retag modal), and the e2e contract on "switching folder
  // rescopes /twin/api/tags immediately" depends on the query existing
  // in the cache for `refetchQueries` to trigger. Gating heavy reads
  // (documents, graph) preserves the bulk of the perf win.
  const tags = useTags({ enabled: authReady });
  const tagCategories = useTagCategories({ enabled: authReady });
  // Activity stays always-enabled (vs. tab-gated): the feed drives the
  // topbar unread counters cross-tab, and the e2e contract requires
  // `/twin/api/activity` to refire under the new folder header at switch
  // time. Lightweight read (bounded via `limit`), so the perf cost is
  // negligible compared to documents / graph which remain gated.
  const activity = useActivity({}, { enabled: authReady });
  const graphEntities = useGraphEntities({ enabled: authReady && tab === 'graph' });
  const graphRelations = useGraphRelations({ enabled: authReady && tab === 'graph' });

  // Notifications carry mutable client state (read/cleared) on top of the
  // query data. Keep only local overrides in React state so refetches can
  // merge without an effect-driven mirror.
  const [readNotificationIds, setReadNotificationIds] = useState<ReadonlySet<string>>(
    () =>
      new Set(
        FIXTURE_FALLBACK_ENABLED
          ? NOTIFICATION_FIXTURES.filter((notification) => notification.read).map(
              (notification) => notification.id,
            )
          : [],
      ),
  );
  const [clearedNotificationIds, setClearedNotificationIds] = useState<
    ReadonlySet<string>
  >(() => new Set());

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const notificationSource =
    resolveQueryData(notificationsQ, NOTIFICATION_FIXTURES) ?? [];
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
      return configuredFolders.map((item) => ({
        id: item.id,
        kb: item.label,
        visibility: item.kind === 'sandbox' ? 'private' : 'internal',
        sources: item.sources ?? 0,
        role: 'admin / steward',
        current: item.id === folder,
      }));
    }
    return resolveQueryData(folders, FOLDER_FIXTURES) ?? [];
  }, [configuredFolders, folder, folders]);
  const kbName = folderList.find((w) => w.id === folder)?.kb ?? '';

  const pushToast = (t: Omit<Toast, 'id'>) => {
    const id = `tst_${Date.now()}_${Math.random().toString(16).slice(2, 6)}`;
    setToasts((ts) => [...ts, { id, ...t }]);
    window.setTimeout(() => {
      setToasts((ts) => ts.filter((x) => x.id !== id));
    }, TOAST_AUTO_DISMISS_MS);
  };

  const onAddToast = (title: string, sub?: string) =>
    pushToast({ kind: 'done', title, sub });

  const onScanRetry = (failedCount: number) => {
    pushToast({
      kind: 'propagating',
      title:
        failedCount > 0
          ? `Retrying ${failedCount} failed source${failedCount > 1 ? 's' : ''}`
          : 'Pipeline scan started',
      sub:
        failedCount > 0
          ? 'POST /documents/reprocess_failed'
          : 'POST /documents/reprocess_failed · no failed source visible',
    });
    void (async () => {
      try {
        const r = await api.reprocessFailedDocuments();
        pushToast({
          kind: 'done',
          title: failedCount > 0 ? 'Retry queued' : 'Scan completed',
          sub: r.message ?? `failed_count=${r.failed_count ?? failedCount}`,
        });
      } catch (err) {
        pushToast({
          kind: 'error',
          title: 'Scan / Retry failed',
          sub: err instanceof Error ? err.message : String(err),
        });
      } finally {
        void queryClient.invalidateQueries({ queryKey: ['documents'] });
        void queryClient.invalidateQueries({ queryKey: ['pipeline_status'] });
        void queryClient.invalidateQueries({ queryKey: ['activity'] });
      }
    })();
  };

  const bulkRetagDocs = useBulkRetagDocuments();

  const onRetagSubmit = async (action: RetagAction) => {
    const verb = action.adds.length > 0 ? 'applied' : 'removed';
    const sample = action.adds[0] ?? action.removes[0];
    try {
      const result = await bulkRetagDocs.mutateAsync({
        targets: action.targets.map((d) => d.doc_id),
        adds: action.adds,
        removes: action.removes,
        actor: currentActor,
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

  const uploadDocs = useUploadDocumentsBatch();
  const deleteDoc = useDeleteDocument();
  const bulkDeleteDocs = useBulkDeleteDocuments();

  const makeOptimisticUploadDocs = (
    files: readonly File[],
    tags: readonly string[],
  ): readonly Document[] => {
    const now = new Date().toISOString();
    const visibility =
      folderList.find((item) => item.id === folder)?.visibility ?? 'internal';
    return files.map((file, index) => ({
      doc_id: `upload_${Date.now()}_${index}`,
      track_id: null,
      file_path: file.name,
      content_summary: 'Upload queued, waiting for ingestion worker.',
      content_length: file.size,
      status: 'PENDING',
      _optimisticUpload: true,
      chunks_count: null,
      created_at: now,
      updated_at: now,
      error_msg: null,
      metadata: {
        size_bytes: file.size,
        upload_state: 'pending',
      },
      type: 'file',
      tags: [...tags],
      folder,
      visibility,
    }));
  };

  const refreshDocumentsUntilUploadsLand = async (
    trackIds: readonly string[],
  ): Promise<void> => {
    if (trackIds.length === 0) return;
    const pending = new Set(trackIds);
    for (let i = 0; i < 30 && pending.size > 0; i += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
      const result = await docs.refetch();
      for (const item of result.data?.items ?? []) {
        if (item.track_id) pending.delete(item.track_id);
      }
    }
  };

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
      sub: `${docs.length} source${docs.length === 1 ? '' : 's'} → DELETE /documents/bulk-delete`,
    });
    try {
      const result = await bulkDeleteDocs.mutateAsync({
        doc_ids: docs.map((d) => d.doc_id),
        actor: currentActor,
      });
      pushToast({
        kind: 'done',
        title: `${result.deleted} source${result.deleted === 1 ? '' : 's'} deleted`,
        sub: 'Cascade removal successful',
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Bulk delete failed',
        sub: err instanceof Error ? err.message : String(err),
      });
    }
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

    const optimisticDocs = makeOptimisticUploadDocs(
      action.rawFiles,
      action.tags,
    );
    setOptimisticUploadDocs((current) => [...optimisticDocs, ...current]);

    pushToast({
      kind: 'propagating',
      title: 'Uploading sources…',
      sub: `${action.rawFiles.length} file${action.rawFiles.length === 1 ? '' : 's'} → LightRAG /documents/upload`,
    });

    const results = await uploadDocs.mutateAsync(action.rawFiles);
    const failedOptimisticIds = new Set(
      optimisticDocs
        .filter((_, index) => results[index]?.status === 'rejected')
        .map((doc) => doc.doc_id),
    );
    const acceptedByOptimisticId = new Map(
      optimisticDocs.flatMap((doc, index) => {
        const result = results[index];
        return result?.status === 'fulfilled'
          ? [[doc.doc_id, result.value] as const]
          : [];
      }),
    );
    const acceptedTrackIds = Array.from(acceptedByOptimisticId.values()).map(
      (result) => result.track_id,
    );
    setOptimisticUploadDocs((current) =>
      current
        .map((doc) => {
          const result = acceptedByOptimisticId.get(doc.doc_id);
          if (!result) return doc;
          return {
            ...doc,
            track_id: result.track_id,
            content_summary:
              result.status === 'duplicated'
                ? 'Upload accepted as duplicate, waiting for source refresh.'
                : 'Upload accepted, waiting for ingestion worker.',
            updated_at: new Date().toISOString(),
            metadata: {
              ...doc.metadata,
              upload_state: result.status,
            },
          };
        })
        .filter((doc) => !failedOptimisticIds.has(doc.doc_id)),
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

    const uploadAuditWrites = results.map((result, index) => {
      if (result.status !== 'fulfilled') return null;
      return api.recordSourceUploaded({
        source: action.rawFiles[index]?.name ?? result.value.track_id,
        track_id: result.value.track_id,
        status: result.value.status,
        actor: currentActor,
      });
    });
    void Promise.allSettled(uploadAuditWrites.filter(Boolean)).then(() => {
      void activity.refetch();
    });

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
    void refreshDocumentsUntilUploadsLand(acceptedTrackIds);
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
    const e2ePoll = window.__TWIN_E2E_INITIAL_TAG_POLL;
    const POLL_INTERVAL_MS = e2ePoll?.intervalMs ?? 2000;
    const MAX_POLLS = e2ePoll?.maxPolls ?? 30;
    const TERMINAL_STATUSES = new Set([
      'processed',
      'PROCESSED',
      'failed',
      'FAILED',
    ]);
    const resolvedDocIds = new Set<string>();
    const pending = new Set(trackIds);
    const transientErrors = new Map<string, number>();
    for (let i = 0; i < MAX_POLLS && pending.size > 0; i++) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      for (const tid of Array.from(pending)) {
        try {
          const status = await api.trackStatus(tid);
          transientErrors.delete(tid);
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
        } catch (err) {
          // 404 from track_status is fine while LightRAG is still
          // booking the doc — keep polling until MAX_POLLS.
          if (err instanceof ApiError && err.status === 404) continue;
          const nextFailures = (transientErrors.get(tid) ?? 0) + 1;
          transientErrors.set(tid, nextFailures);
          if (nextFailures >= 3) {
            pending.delete(tid);
            pushToast({
              kind: 'error',
              title: 'Initial tag polling failed',
              sub:
                err instanceof Error
                  ? `${tid}: ${err.message}`
                  : `${tid}: track_status unavailable`,
            });
          }
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
        actor: currentActor,
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

  const onTagApprove = async (action: TagApproveAction) => {
    try {
      await approveTag.mutateAsync({
        name: action.tag.tag,
        actor: currentActor,
      });
      pushToast({
        kind: 'done',
        title: 'Tag',
        tagname: action.tag.tag,
        titleSuffix: 'approved',
        sub: 'Added to tag catalog · Tier 3',
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Tag approval failed',
        tagname: action.tag.tag,
        sub: err instanceof Error ? err.message : 'Mutation rejected',
      });
    }
  };

  const commitTagMutation = (
    run: (callbacks: {
      onSuccess: () => void;
      onError: (err: unknown) => void;
    }) => void,
    toast: Omit<Toast, 'id'>,
    failureTitle: string,
  ) => {
    run({
      onSuccess: () => pushToast(toast),
      onError: (err) =>
        pushToast({
          kind: 'error',
          title: failureTitle,
          tagname: toast.tagname,
          sub: err instanceof Error ? err.message : 'Mutation rejected',
        }),
    });
  };

  const onTagCommit = (commit: TagActionCommit) => {
    const tagname = commit.tag?.tag ?? commit.name ?? '';
    const actor = currentActor;
    const verbMap: Record<TagActionCommit['kind'], string> = {
      edit: 'updated',
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
    const successToast: Omit<Toast, 'id'> = {
      kind: 'done',
      title: 'Tag',
      tagname,
      titleSuffix: verbMap[commit.kind],
      sub: commit.reason ?? '',
    };
    const failureTitle = `Tag ${commit.kind} failed`;

    switch (commit.kind) {
      case 'edit':
        commitTagMutation(
          (cb) =>
            editTag.mutate(
              {
                name: tagname,
                tag: commit.name,
                def: commit.def,
                long_description: commit.longDescription,
                category: commit.category,
                actor,
              },
              cb,
            ),
          successToast,
          failureTitle,
        );
        break;
      case 'suggest':
        // No backend endpoint for "suggest" yet — surface as a request.
        if (commit.tag) {
          commitTagMutation(
            (cb) =>
              requestTag.mutate(
                {
                  tag: commit.tag!.tag,
                  def: commit.tag!.def,
                  category: commit.tag!.category,
                  actor,
                  justification: 'suggested edit',
                },
                cb,
              ),
            successToast,
            failureTitle,
          );
        }
        break;
      case 'synonyms':
        if (commit.tag) {
          commitTagMutation(
            (cb) =>
              updateSynonyms.mutate(
                {
                  name: tagname,
                  aliases: commit.aliases ?? commit.tag!.aliases,
                  actor,
                },
                cb,
              ),
            successToast,
            failureTitle,
          );
        }
        break;
      case 'deprecate':
        commitTagMutation(
          (cb) => deprecateTag.mutate({ name: tagname, actor }, cb),
          successToast,
          failureTitle,
        );
        break;
      case 'delete':
        commitTagMutation(
          (cb) =>
            deleteTag.mutate(
              {
                name: tagname,
                strategy: commit.migrate?.strategy ?? 'untag',
                to: commit.migrate?.to,
                actor,
              },
              cb,
            ),
          successToast,
          failureTitle,
        );
        break;
      case 'reject':
        commitTagMutation(
          (cb) =>
            rejectTag.mutate(
              {
                name: tagname,
                reason: commit.reason || 'rejected',
                actor,
              },
              cb,
            ),
          successToast,
          failureTitle,
        );
        break;
      case 'edit-approve':
        void (async () => {
          try {
            if (
              commit.name ||
              commit.def ||
              commit.longDescription ||
              commit.category
            ) {
              await editTag.mutateAsync({
                name: tagname,
                tag: commit.name,
                def: commit.def,
                long_description: commit.longDescription,
                category: commit.category,
                actor,
              });
            }
            await approveTag.mutateAsync({ name: tagname, actor });
            pushToast(successToast);
          } catch (err) {
            pushToast({
              kind: 'error',
              title: failureTitle,
              tagname: successToast.tagname,
              sub: err instanceof Error ? err.message : 'Mutation rejected',
            });
          }
        })();
        break;
      case 'request':
        if (commit.name) {
          commitTagMutation(
            (cb) =>
              requestTag.mutate(
                {
                  tag: commit.name!,
                  def: commit.def ?? '',
                  category: commit.category ?? 'infra',
                  actor,
                },
                cb,
              ),
            successToast,
            failureTitle,
          );
        }
        break;
    }
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

  const onSwitchFolder = (nextFolder: string) => {
    window.history.replaceState(null, '', window.location.pathname);
    setActiveFolder(nextFolder);
    setFolderState(nextFolder);
    setReadNotificationIds(new Set());
    setClearedNotificationIds(new Set());
    setDetailDoc(null);
    setReadSourceDoc(null);
    setRetagDoc(null);
    setRetagBulk(null);
    // Use `refetchQueries` with `type: 'all'` so disabled (tab-gated)
    // queries also fetch immediately on folder switch — otherwise the
    // invariant "switching folder rescopes every Twin overlay surface"
    // breaks for inactive tabs and the e2e contract on
    // `/twin/api/tags` (under sandbox header) fails. Preserves the
    // perf gain at boot/tab switch; only the user-initiated folder
    // switch pays the cost of refreshing all four resources.
    void Promise.all([
      queryClient.refetchQueries({ queryKey: ['documents'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['tags'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['activity'], type: 'all' }),
      queryClient.refetchQueries({ queryKey: ['notifications'], type: 'all' }),
    ]);
  };

  const backendErrors = [
    resourceError('Documents', docs),
    resourceError('Folders', folders),
    resourceError('Notifications', notificationsQ),
    resourceError('Tags', tags),
    resourceError('Tag categories', tagCategories),
    resourceError('Activity', activity),
    resourceError('Graph entities', graphEntities),
    resourceError('Graph relations', graphRelations),
  ].filter((err): err is BackendResourceError => err !== null);

  // Resolved props. In prod real-backend mode, do not silently fall back to
  // local fixtures: empty arrays + the backend error banner make the failure
  // visible instead of showing stale demo data.
  const backendDocList = useMemo(
    () =>
      docs.data?.items ??
      (FIXTURE_FALLBACK_ENABLED
        ? DOCUMENT_FIXTURES.filter((doc) => doc.folder === folder)
        : []),
    [docs.data?.items, folder],
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
  const tagList = resolveQueryData(tags, TAG_FIXTURES) ?? [];
  const tagCatalog = tagCatalogForSuggestions(tagList);
  const tagCategoryList = resolveQueryData(tagCategories, TAG_CATEGORY_FIXTURES) ?? [];
  const activityFallback = resolveQueryData(activity, {
    items: ACTIVITY_FIXTURES,
    total: ACTIVITY_FIXTURES.length,
    nowMs: ACTIVITY_NOW_MS,
  });
  const activityEvents = activity.data?.items ?? activityFallback?.items ?? [];
  const activityNow = activity.data?.nowMs ?? activityFallback?.nowMs;
  const graphEntityList = resolveQueryData(graphEntities, GRAPH_ENTITY_FIXTURES) ?? [];
  const graphRelationList =
    resolveQueryData(graphRelations, GRAPH_RELATION_FIXTURES) ?? [];

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
      {backendErrors.length > 0 && (
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
              tagCatalog={tagCatalog}
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
              onScanRetry={onScanRetry}
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
                // TR-RET-02 step 3 / audit C1: tag_filter is NOT sent
                // to /query because LightRAG 1.4.x does not apply it
                // to retrieval. The backend now 422s if it slips in.
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
                });
                // Map the backend SourceRow shape to the RetrievalSource
                // contract the chat panel consumes. `type` is the WebUI
                // SourceIcon key — backend currently always emits "file"
                // (LightRAG-stored chunks); fall back defensively for
                // any future expansion.
                const sources = (res.sources ?? []).map((s) => ({
                  n: s.n,
                  type:
                    s.type === 'file' ||
                    s.type === 'url' ||
                    s.type === 'confluence' ||
                    s.type === 'sharepoint'
                      ? (s.type as 'file' | 'url' | 'confluence' | 'sharepoint')
                      : ('file' as const),
                  name: s.name,
                  meta: s.meta ?? undefined,
                  score: s.score,
                }));
                return { response: res.response, sources };
              }}
              onStreamQuery={async (params, onChunk) => {
                // Same C1 honesty as the non-stream branch above:
                // no tag_filter forwarded to /query/stream.
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
                  },
                  onChunk,
                );
                const sources = (res.sources ?? []).map((s) => ({
                  n: s.n,
                  type:
                    s.type === 'file' ||
                    s.type === 'url' ||
                    s.type === 'confluence' ||
                    s.type === 'sharepoint'
                      ? (s.type as 'file' | 'url' | 'confluence' | 'sharepoint')
                      : ('file' as const),
                  name: s.name,
                  meta: s.meta ?? undefined,
                  score: s.score,
                }));
                return { response: res.response, sources };
              }}
              initialThreads={
                FIXTURE_FALLBACK_ENABLED ? makeSampleThreads() : []
              }
              suggestions={FIXTURE_FALLBACK_ENABLED ? undefined : []}
              onNavigate={onNavigate}
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
              onRefresh={() => activity.refetch()}
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
            formatCategories={FORMAT_CATEGORY_FIXTURES}
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
