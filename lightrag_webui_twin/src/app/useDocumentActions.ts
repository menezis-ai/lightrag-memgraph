import type { Dispatch, SetStateAction } from 'react';
import { ApiError } from '../api/client';
import {
  useBulkDeleteDocuments,
  useBulkRetagDocuments,
  useDeleteDocument,
  useDocuments,
  useUploadDocumentsBatch,
} from '../api/queries';
import { api, type UploadDocumentInput } from '../api/resources';
import type { AddSourceAction } from '../components/AddSourceModal';
import type { RetagAction } from '../components/RetagModal';
import type { Document } from '../types/document';
import type { Folder } from '../types/topbar';
import type { Toast } from '../types/toast';
import { queryClient } from './queryClient';
import { asRetagUndoPayload } from './useToasts';

interface UseDocumentActionsOptions {
  activity: { refetch: () => unknown };
  currentActor: string;
  docs: ReturnType<typeof useDocuments>;
  folder: string;
  folderList: readonly Folder[];
  pushToast: (toast: Omit<Toast, 'id'>) => void;
  setAddOpen: Dispatch<SetStateAction<boolean>>;
  setOptimisticUploadDocs: Dispatch<SetStateAction<readonly Document[]>>;
  setToasts: Dispatch<SetStateAction<Toast[]>>;
}

export function useDocumentActions({
  activity,
  currentActor,
  docs,
  folder,
  folderList,
  pushToast,
  setAddOpen,
  setOptimisticUploadDocs,
  setToasts,
}: UseDocumentActionsOptions) {
  const bulkRetagDocs = useBulkRetagDocuments();
  const uploadDocs = useUploadDocumentsBatch();
  const deleteDoc = useDeleteDocument();
  const bulkDeleteDocs = useBulkDeleteDocuments();

  const onScanRetry = (failedCount: number) => {
    // Audit C7: the button is disabled when ``failedCount === 0`` in
    // ``DocumentsTab``, so we only land here on the failed-batch path.
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
    setToasts((current) => current.filter((item) => item.id !== toast.id));
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
    docsToDelete: readonly { doc_id: string; file_path: string }[],
  ) => {
    if (docsToDelete.length === 0) return;
    pushToast({
      kind: 'propagating',
      title: 'Deleting sources…',
      sub: `${docsToDelete.length} source${docsToDelete.length === 1 ? '' : 's'} → DELETE /documents/bulk-delete`,
    });
    try {
      const result = await bulkDeleteDocs.mutateAsync({
        doc_ids: docsToDelete.map((doc) => doc.doc_id),
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
      pushToast({
        kind: 'done',
        title: 'Sources queued',
        sub: `${action.readyCount} entr${action.readyCount === 1 ? 'y' : 'ies'}`,
      });
      setAddOpen(false);
      return;
    }

    const uploadInputs: readonly UploadDocumentInput[] = action.rawFiles.map(
      (file) => ({ file }),
    );

    const optimisticDocs = makeOptimisticUploadDocs(uploadInputs, action.tags);
    setOptimisticUploadDocs((current) => [...optimisticDocs, ...current]);

    pushToast({
      kind: 'propagating',
      title: 'Uploading sources…',
      sub: `${uploadInputs.length} file${uploadInputs.length === 1 ? '' : 's'} → LightRAG /documents/upload`,
    });

    const results = await uploadDocs.mutateAsync(uploadInputs);
    setAddOpen(false);
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

    const ok = results.filter((result) => result.status === 'fulfilled').length;
    const ko = results.filter((result) => result.status === 'rejected').length;
    const dup = results.filter(
      (result) =>
        result.status === 'fulfilled' &&
        (result.value as { status?: string }).status === 'duplicated',
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

    if (action.tags.length > 0) {
      const successfulTrackIds = results
        .filter((result) => result.status === 'fulfilled')
        .map(
          (result) =>
            (result as PromiseFulfilledResult<{ track_id: string }>).value
              .track_id,
        )
        .filter(Boolean);
      if (successfulTrackIds.length > 0) {
        void applyInitialTagsAfterIngestion(successfulTrackIds, action.tags);
      }
    }
    void refreshDocumentsUntilUploadsLand(acceptedTrackIds);
  };

  const applyInitialTagsAfterIngestion = async (
    trackIds: readonly string[],
    tags: readonly string[],
  ): Promise<void> => {
    const e2ePoll = window.__TWIN_E2E_INITIAL_TAG_POLL;
    const pollIntervalMs = e2ePoll?.intervalMs ?? 2000;
    const maxPolls = e2ePoll?.maxPolls ?? 30;
    const terminalStatuses = new Set([
      'processed',
      'PROCESSED',
      'failed',
      'FAILED',
    ]);
    const resolvedDocIds = new Set<string>();
    const pending = new Set(trackIds);
    const transientErrors = new Map<string, number>();
    for (let i = 0; i < maxPolls && pending.size > 0; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
      for (const trackId of Array.from(pending)) {
        try {
          const status = await api.trackStatus(trackId);
          transientErrors.delete(trackId);
          const terminalDocs = status.documents.filter((doc) =>
            terminalStatuses.has(doc.status),
          );
          if (
            terminalDocs.length > 0 &&
            terminalDocs.length === status.documents.length
          ) {
            terminalDocs
              .filter((doc) => doc.status.toLowerCase() === 'processed')
              .forEach((doc) => resolvedDocIds.add(doc.id));
            pending.delete(trackId);
          }
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) continue;
          const nextFailures = (transientErrors.get(trackId) ?? 0) + 1;
          transientErrors.set(trackId, nextFailures);
          if (nextFailures >= 3) {
            pending.delete(trackId);
            pushToast({
              kind: 'error',
              title: 'Initial tag polling failed',
              sub:
                err instanceof Error
                  ? `${trackId}: ${err.message}`
                  : `${trackId}: track_status unavailable`,
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
          sub: `Ingestion didn't reach a terminal state within ${(maxPolls * pollIntervalMs) / 1000}s. Retag manually once the docs land.`,
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

  return {
    uploadDocs,
    onAddSourceSubmit,
    onDeleteBulk,
    onDeleteSingle,
    onRetagSubmit,
    onScanRetry,
    onToastUndo,
  };
}
