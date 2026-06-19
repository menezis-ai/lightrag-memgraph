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
  usePipelineStatus,
  useRejectTag,
  useRequestTag,
  useTagCategories,
  useTags,
  useUpdateTagSynonyms,
  useFolders,
} from './api/queries';
import { ApiError, getTwinRuntimeConfig, setActiveFolder } from './api/client';
import { api, type UploadDocumentInput } from './api/resources';
import { mapTwinQueryResponseForRetrievalTab } from './api/twinQueryResponse';
import { FORMAT_CATEGORIES } from './constants/formatCategories';
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

interface RetagUndoPayload {
  targets: readonly string[];
  adds: readonly string[];
  removes: readonly string[];
}

interface BackendResourceError {
  label: string;
  message: string;
}

function resourceError<T>(
  label: string,
  query: QueryLike<T>,
): BackendResourceError | null {
  if (query.data || query.isLoading || !query.isError) {
    return null;
  }
  return { label, message: formatBackendError(query.error) };
}

function formatBackendError(error: unknown): string {
  if (error instanceof ApiError) return `${error.status} ${error.message}`;
  if (error instanceof Error) return error.message;
  return 'Backend request failed';
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function asRetagUndoPayload(value: unknown): RetagUndoPayload | null {
  if (!value || typeof value !== 'object') return null;
  const payload = value as Record<string, unknown>;
  if (
    !isStringArray(payload.targets) ||
    !isStringArray(payload.adds) ||
    !isStringArray(payload.removes)
  ) {
    return null;
  }
  return {
    targets: payload.targets,
    adds: payload.adds,
    removes: payload.removes,
  };
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

const THEME_STORAGE_KEY = 'twin.ui.theme.v1';
const FOLDER_STORAGE_KEY = 'twin.ui.folder.v1';

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

function readUiPreference(key: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeUiPreference(key: string, value: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Browsers can reject localStorage in private/restricted modes. The UI
    // still works for the current session; only refresh persistence is lost.
  }
}

function isTheme(value: string | null): value is Theme {
  return value === 'light' || value === 'dark';
}

function getInitialTheme(): Theme {
  const stored = readUiPreference(THEME_STORAGE_KEY);
  return isTheme(stored) ? stored : 'light';
}

function getConfiguredDefaultFolderId(): string {
  const cfg = getTwinRuntimeConfig();
  return (
    cfg.defaultFolderId ||
    cfg.folders?.[0]?.id ||
    'default'
  );
}

function getInitialFolderId(): string {
  const cfg = getTwinRuntimeConfig();
  const fallback = getConfiguredDefaultFolderId();
  const stored = readUiPreference(FOLDER_STORAGE_KEY);
  if (!stored) return fallback;
  if (cfg.folders && !cfg.folders.some((folder) => folder.id === stored)) {
    return fallback;
  }
  return stored;
}

function AppShell() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme());
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
  const [detailChunkId, setDetailChunkId] = useState<string | null>(null);
  const [detailRequest, setDetailRequest] = useState<{
    doc?: string;
    source?: string;
    chunk?: string;
  } | null>(() => {
    const params = new URLSearchParams(window.location.search);
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
  const [optimisticUploadDocs, setOptimisticUploadDocs] = useState<
    readonly Document[]
  >([]);

  // Auth
  const auth = useAuth();
  const runtimeConfig = auth.config;
  const currentActor = auth.user?.email ?? CURRENT_USER.name;
  const authReady = !auth.isCheckingAuth && !auth.needsLogin;
  const retagOpen = retagDoc !== null || retagBulk !== null;

  // Data — every visible resource comes from the API query layer. No local
  // sample fallback is allowed on the operator surface.
  const docs = useDocuments(
    { folder },
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
  // Activity stays always-enabled (vs. tab-gated): the feed drives the
  // topbar unread counters cross-tab, and the e2e contract requires
  // `/twin/api/activity` to refire under the new folder header at switch
  // time. Lightweight read (bounded via `limit`), so the perf cost is
  // negligible compared to documents / graph which remain gated.
  const activity = useActivity({}, { enabled: authReady, folderKey: folder });
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
    document.documentElement.setAttribute('data-theme', theme);
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
      return configuredFolders.map((item) => ({
        id: item.id,
        kb: item.label,
        visibility: item.kind === 'sandbox' ? 'private' : 'internal',
        sources: item.sources ?? 0,
        role: 'admin / steward',
        current: item.id === folder,
      }));
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
    // Audit C7: the button is disabled when ``failedCount === 0`` in
    // ``DocumentsTab``, so we only land here on the failed-batch
    // path. No "queued" wording — the backend doesn't expose an
    // observable queue; we honour the request, the operator hears
    // back with the failed_count summary.
    pushToast({
      kind: 'propagating',
      title: 'Re-processing failed sources',
      sub: `POST /documents/reprocess_failed · ${failedCount} failed source${
        failedCount > 1 ? 's' : ''
      }`,
    });
    void (async () => {
      try {
        const r = await api.reprocessFailedDocuments();
        pushToast({
          kind: 'done',
          title: 'Reprocess request sent',
          sub: r.message ?? `failed_count=${r.failed_count ?? failedCount}`,
        });
      } catch (err) {
        pushToast({
          kind: 'error',
          title: 'Re-process failed',
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
        undo: {
          targets: action.targets.map((target) => target.doc_id),
          adds: action.adds,
          removes: action.removes,
        },
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

  const onToastUndo = async (toast: Toast) => {
    setToasts((ts) => ts.filter((x) => x.id !== toast.id));
    const undo = asRetagUndoPayload(toast.undo);
    if (!undo) return;

    try {
      const result = await bulkRetagDocs.mutateAsync({
        targets: undo.targets,
        adds: undo.removes,
        removes: undo.adds,
        actor: currentActor,
      });
      pushToast({
        kind: 'done',
        title: 'Undo applied',
        tagname: toast.tagname,
        titleSuffix:
          result.failed.length > 0
            ? `${result.updated} source${result.updated === 1 ? '' : 's'} · ${result.failed.length} skipped`
            : `${result.updated} source${result.updated === 1 ? '' : 's'}`,
        sub: toast.sub,
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Undo failed',
        tagname: toast.tagname,
        sub: err instanceof Error ? err.message : 'Mutation rejected',
      });
    }
  };

  const uploadDocs = useUploadDocumentsBatch();
  const deleteDoc = useDeleteDocument();
  const bulkDeleteDocs = useBulkDeleteDocuments();

  const makeOptimisticUploadDocs = (
    uploads: readonly UploadDocumentInput[],
    tags: readonly string[],
  ): readonly Document[] => {
    const now = new Date().toISOString();
    const visibility =
      folderList.find((item) => item.id === folder)?.visibility ?? 'internal';
    return uploads.map((upload, index) => ({
      doc_id: `upload_${Date.now()}_${index}`,
      track_id: null,
      file_path: upload.file.name,
      content_summary: 'Upload queued, waiting for ingestion worker.',
      content_length: upload.file.size,
      status: 'PENDING',
      _optimisticUpload: true,
      chunks_count: null,
      created_at: now,
      updated_at: now,
      error_msg: null,
      metadata: {
        size_bytes: upload.file.size,
        upload_state: 'pending',
        ...(upload.classification ? { classification: upload.classification } : {}),
        rag_engine: upload.ragEngine ?? 'lightrag',
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
      for (const item of result.data?.pages.flatMap((p) => p.items) ?? []) {
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

    const uploadInputs: readonly UploadDocumentInput[] = action.rawFiles.map(
      (file, index) => {
        const opts = action.fileOptions[index];
        return {
          file,
          classification: opts?.classification,
          ragEngine: opts?.ragEngine ?? 'lightrag',
        };
      },
    );

    const optimisticDocs = makeOptimisticUploadDocs(uploadInputs, action.tags);
    setOptimisticUploadDocs((current) => [...optimisticDocs, ...current]);

    pushToast({
      kind: 'propagating',
      title: 'Uploading sources…',
      sub: `${uploadInputs.length} file${uploadInputs.length === 1 ? '' : 's'} → LightRAG /documents/upload`,
    });

    const results = await uploadDocs.mutateAsync(uploadInputs);
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
        source: uploadInputs[index]?.file.name ?? result.value.track_id,
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
                  long_description: commit.longDescription,
                  category: commit.category ?? 'infra',
                  aliases: commit.aliases ?? [],
                  justification: commit.justification,
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
    if (nextTab === 'documents' && (params?.doc || params?.source)) {
      setDetailDoc(null);
      setDetailChunkId(null);
      setDetailRequest({
        doc: params.doc,
        source: params.source,
        chunk: params.chunk,
      });
    } else {
      setDetailRequest(null);
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
    writeUiPreference(FOLDER_STORAGE_KEY, nextFolder);
    setFolderState(nextFolder);
    setReadNotificationIds(new Set());
    setClearedNotificationIds(new Set());
    setDetailDoc(null);
    setDetailChunkId(null);
    setDetailRequest(null);
    setReadSourceDoc(null);
    setRetagDoc(null);
    setRetagBulk(null);
    void Promise.all([
      queryClient.invalidateQueries({ queryKey: ['documents'] }),
      queryClient.invalidateQueries({ queryKey: ['pipeline_status'] }),
      queryClient.invalidateQueries({ queryKey: ['tags'] }),
      queryClient.invalidateQueries({ queryKey: ['tag-categories'] }),
      queryClient.invalidateQueries({ queryKey: ['activity'] }),
      queryClient.invalidateQueries({ queryKey: ['notifications'] }),
      queryClient.invalidateQueries({ queryKey: ['graph-entities'] }),
      queryClient.invalidateQueries({ queryKey: ['graph-relations'] }),
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

  // Resolved props. Do not silently fall back to local samples: empty arrays
  // + the backend error banner make failures visible instead of showing stale
  // sample data.
  const backendDocList = useMemo(
    () => docs.data?.pages.flatMap((p) => p.items) ?? [],
    [docs.data],
  );
  // Real DB total (from the first page envelope) — the loaded list may be a
  // subset until the operator pulls more pages.
  const docsTotal = docs.data?.pages[0]?.total ?? backendDocList.length;
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
    detailDoc !== null ? detailChunkId : (detailRequest?.chunk ?? null);
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
  const tagList = tags.data ?? [];
  const tagCatalog = tagCatalogForSuggestions(tagList);
  const tagCategoryList = tagCategories.data ?? [];
  const activityEvents = activity.data?.items ?? [];
  const activityNow = activity.data?.nowMs;
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
              loadedCount={backendDocList.length}
              totalCount={docsTotal}
              hasMore={docs.hasNextPage}
              isLoadingMore={docs.isFetchingNextPage}
              onLoadMore={() => void docs.fetchNextPage()}
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
            />
          )}
          {tab === 'activity' && (
            <ActivityTab
              events={activityEvents}
              nowMs={activityNow}
              folderLabel={kbName || folder}
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
