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
import {
  isProcessedTrackStatus,
  isTerminalTrackStatus,
} from '../lib/docStatus';
import {
  logTechnicalError,
  uploadFailureMessage,
  userErrorMessage,
} from '../lib/errorMessages';
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

type UploadResponse = { status: string; message: string; track_id: string };
type UploadResult = PromiseSettledResult<UploadResponse>;
type TrackStatus = Awaited<ReturnType<typeof api.trackStatus>>;
type PushToast = (toast: Omit<Toast, 'id'>) => void;

function summarizeUploadResults(results: readonly UploadResult[]) {
  const ok = results.filter((result) => result.status === 'fulfilled').length;
  const ko = results.length - ok;
  const dup = results.filter(
    (result) =>
      result.status === 'fulfilled' && result.value.status === 'duplicated',
  ).length;
  return { ok, ko, dup };
}

function failedOptimisticUploadIds(
  optimisticDocs: readonly Document[],
  results: readonly UploadResult[],
): Set<string> {
  return new Set(
    optimisticDocs
      .filter((_, index) => results[index]?.status === 'rejected')
      .map((doc) => doc.doc_id),
  );
}

function acceptedUploadsByOptimisticId(
  optimisticDocs: readonly Document[],
  results: readonly UploadResult[],
): Map<string, UploadResponse> {
  return new Map(
    optimisticDocs.flatMap((doc, index) => {
      const result = results[index];
      return result?.status === 'fulfilled'
        ? [[doc.doc_id, result.value] as const]
        : [];
    }),
  );
}

function patchOptimisticUploadDocs(
  docs: readonly Document[],
  acceptedByOptimisticId: ReadonlyMap<string, UploadResponse>,
  failedIds: ReadonlySet<string>,
): readonly Document[] {
  return docs
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
    .filter((doc) => !failedIds.has(doc.doc_id));
}

function recordUploadAudit(
  results: readonly UploadResult[],
  uploadInputs: readonly UploadDocumentInput[],
  actor: string,
): Promise<unknown>[] {
  return results.flatMap((result, index) =>
    result.status === 'fulfilled'
      ? [
          api.recordSourceUploaded({
            source: uploadInputs[index]?.file.name ?? result.value.track_id,
            track_id: result.value.track_id,
            status: result.value.status,
            actor,
          }),
        ]
      : [],
  );
}

function uploadResultTrackIds(results: readonly UploadResult[]): string[] {
  return results.flatMap((result) =>
    result.status === 'fulfilled' ? [result.value.track_id] : [],
  );
}

function buildOptimisticUploadState(
  optimisticDocs: readonly Document[],
  results: readonly UploadResult[],
) {
  const failedOptimisticIds = failedOptimisticUploadIds(optimisticDocs, results);
  const acceptedByOptimisticId = acceptedUploadsByOptimisticId(
    optimisticDocs,
    results,
  );
  return {
    failedOptimisticIds,
    acceptedByOptimisticId,
  };
}

function dispatchUploadAudit(
  results: readonly UploadResult[],
  uploadInputs: readonly UploadDocumentInput[],
  actor: string,
  activity: { refetch: () => unknown },
) {
  const uploadAuditWrites = recordUploadAudit(results, uploadInputs, actor);
  Promise.allSettled(uploadAuditWrites)
    .then(() => activity.refetch())
    .catch(() => undefined);
}

function maybeApplyInitialTags(
  results: readonly UploadResult[],
  tags: readonly string[],
  applyInitialTagsAfterIngestion: (trackIds: readonly string[], tags: readonly string[]) => Promise<void>,
) {
  if (!tags.length) return;
  const trackIds = uploadResultTrackIds(results);
  if (trackIds.length === 0) return;
  void applyInitialTagsAfterIngestion(trackIds, tags);
}

/** Human reason for the first rejected upload — e.g. "archive.zip: ZIP
 *  format is not supported" — so the failure toast explains itself
 *  instead of only counting. Remaining failures are summarized. */
function firstUploadFailureReason(
  results: readonly UploadResult[],
  fileNames: readonly string[],
): string | null {
  const index = results.findIndex((result) => result.status === 'rejected');
  if (index === -1) return null;
  const rejected = results[index] as PromiseRejectedResult;
  logTechnicalError('upload', rejected.reason);
  const fileName = fileNames[index];
  const reason = uploadFailureMessage(rejected.reason, fileName);
  const prefix = fileName ? `${fileName}: ` : '';
  const moreFailures = results.filter((r) => r.status === 'rejected').length - 1;
  const moreSuffix = moreFailures > 0 ? ` (+${moreFailures} more)` : '';
  return `${prefix}${reason}${moreSuffix}`;
}

function pushUploadSummaryToast(
  results: readonly UploadResult[],
  fileNames: readonly string[],
  tags: readonly string[],
  pushToast: PushToast,
) {
  const { ok, ko, dup } = summarizeUploadResults(results);
  const duplicateSuffix = dup > 0 ? ` (${dup} already present)` : '';
  const initialTagsSuffix = tags.length
    ? ' · initial tags will apply once docs land'
    : '';
  if (ko === 0) {
    pushToast({
      kind: 'done',
      title: 'Sources queued for ingestion',
      sub: `${ok} uploaded${duplicateSuffix}${initialTagsSuffix}`,
    });
    return;
  }
  const duplicateErrorSuffix = dup > 0 ? ` · ${dup} already present` : '';
  const reason = firstUploadFailureReason(results, fileNames);
  const reasonSuffix = reason ? ` — ${reason}` : '';
  pushToast({
    kind: 'error',
    title: `${ko} upload${ko === 1 ? '' : 's'} failed`,
    sub: `${ok} uploaded · ${ko} failed${duplicateErrorSuffix}${reasonSuffix}`,
  });
}

function sourceCountLabel(count: number): string {
  return `${count} source${count === 1 ? '' : 's'}`;
}

function retagTitleSuffix(
  action: RetagAction,
  verb: string,
  updated: number,
  failedCount: number,
): string {
  if (!action.bulk) return verb;
  const skippedSuffix = failedCount > 0 ? ` · ${failedCount} skipped` : '';
  return `${verb} to ${sourceCountLabel(updated)}${skippedSuffix}`;
}

function retagErrorMessage(err: unknown, targetCount: number): string {
  logTechnicalError('retag', err);
  return userErrorMessage(err, {
    action: `updating tags on ${sourceCountLabel(targetCount)}`,
  });
}

function undoTitleSuffix(updated: number, failedCount: number): string {
  const skippedSuffix = failedCount > 0 ? ` · ${failedCount} skipped` : '';
  return `${sourceCountLabel(updated)}${skippedSuffix}`;
}

function trackStatusErrorMessage(err: unknown): string {
  logTechnicalError('track-status', err);
  return userErrorMessage(err, { action: 'checking ingestion status' });
}

function processedDocIdsIfTerminal(status: TrackStatus): string[] | null {
  // Dual-cased status reads are owned by the shared vocabulary module
  // (audit 2026-07-02, DUP-1).
  const terminalDocs = status.documents.filter((doc) =>
    isTerminalTrackStatus(doc.status),
  );
  if (terminalDocs.length === 0 || terminalDocs.length !== status.documents.length) {
    return null;
  }
  return terminalDocs
    .filter((doc) => isProcessedTrackStatus(doc.status))
    .map((doc) => doc.id);
}

function shouldRetryTrackStatus(
  trackId: string,
  err: unknown,
  transientErrors: Map<string, number>,
  pushToast: PushToast,
): boolean {
  if (err instanceof ApiError && err.status === 404) return true;
  const nextFailures = (transientErrors.get(trackId) ?? 0) + 1;
  transientErrors.set(trackId, nextFailures);
  if (nextFailures < 3) return true;
  pushToast({
    kind: 'error',
    title: 'Initial tag polling failed',
    sub: trackStatusErrorMessage(err),
  });
  return false;
}

interface InitialTagPollOptions {
  trackIds: readonly string[];
  pollIntervalMs: number;
  maxPolls: number;
  pushToast: PushToast;
}

async function pollProcessedDocIds({
  trackIds,
  pollIntervalMs,
  maxPolls,
  pushToast,
}: InitialTagPollOptions): Promise<Set<string>> {
  const resolvedDocIds = new Set<string>();
  const pending = new Set(trackIds);
  const transientErrors = new Map<string, number>();
  for (let i = 0; i < maxPolls && pending.size > 0; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    for (const trackId of Array.from(pending)) {
      try {
        const status = await api.trackStatus(trackId);
        transientErrors.delete(trackId);
        const processedDocIds = processedDocIdsIfTerminal(status);
        if (!processedDocIds) continue;
        processedDocIds.forEach((id) => resolvedDocIds.add(id));
        pending.delete(trackId);
      } catch (err) {
        if (shouldRetryTrackStatus(trackId, err, transientErrors, pushToast))
          continue;
        pending.delete(trackId);
      }
    }
  }
  return resolvedDocIds;
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
      sub: `${failedCount} failed source${failedCount > 1 ? 's' : ''} queued for retry`,
    });
    void (async () => {
      try {
        const r = await api.reprocessFailedDocuments();
        pushToast({
          kind: 'done',
          title: 'Reprocess request sent',
          sub:
            r.message ??
            `${r.failed_count ?? failedCount} failed source${
              (r.failed_count ?? failedCount) === 1 ? '' : 's'
            } queued for retry`,
        });
      } catch (err) {
        logTechnicalError('reprocess', err);
        pushToast({
          kind: 'error',
          title: 'Re-process failed',
          sub: userErrorMessage(err, { action: 're-processing failed sources' }),
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
        titleSuffix: retagTitleSuffix(
          action,
          verb,
          result.updated,
          failedCount,
        ),
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
        title: 'Tag update failed',
        sub: retagErrorMessage(err, action.targets.length),
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
        titleSuffix: undoTitleSuffix(result.updated, result.failed.length),
        sub: toast.sub,
      });
    } catch (err) {
      logTechnicalError('retag-undo', err);
      pushToast({
        kind: 'error',
        title: 'Undo failed',
        tagname: toast.tagname,
        sub: userErrorMessage(err, { action: 'undoing the tag change' }),
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
      await new Promise((resolve) => globalThis.setTimeout(resolve, 2000));
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
        title: 'Document removed',
        // Ref-counted (membership) delete: the doc is un-shared from the active
        // folder; its chunks/entities/relations cascade only when this was its
        // last folder. Don't claim a full cascade — a shared doc stays in its
        // other folders.
        sub: `${doc.file_path} — removed from the active folder (kept in any other folders it is shared into; fully deleted only if this was its last folder)`,
      });
    } catch (err) {
      logTechnicalError('delete-document', err);
      pushToast({
        kind: 'error',
        title: 'Delete failed',
        sub: userErrorMessage(err, { action: `deleting ${doc.file_path}` }),
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
      sub: `${docsToDelete.length} source${docsToDelete.length === 1 ? '' : 's'} being removed`,
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
      logTechnicalError('bulk-delete', err);
      pushToast({
        kind: 'error',
        title: 'Bulk delete failed',
        sub: userErrorMessage(err, { action: 'deleting the selected sources' }),
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
      (file, index) => {
        const opts = action.fileOptions[index];
        return {
          file,
          ...(opts?.classification
            ? { classification: opts.classification }
            : {}),
        };
      },
    );

    const optimisticDocs = makeOptimisticUploadDocs(uploadInputs, action.tags);
    setOptimisticUploadDocs((current) => [...optimisticDocs, ...current]);

    pushToast({
      kind: 'propagating',
      title: 'Uploading sources…',
      sub: `${uploadInputs.length} file${uploadInputs.length === 1 ? '' : 's'} being sent for ingestion`,
    });

    const results: readonly UploadResult[] = await uploadDocs.mutateAsync(uploadInputs);
    setAddOpen(false);

    const {
      failedOptimisticIds,
      acceptedByOptimisticId,
    } = buildOptimisticUploadState(optimisticDocs, results);
    const acceptedTrackIds = Array.from(acceptedByOptimisticId.values()).map(
      (result) => result.track_id,
    );
    setOptimisticUploadDocs((current) =>
      patchOptimisticUploadDocs(
        current,
        acceptedByOptimisticId,
        failedOptimisticIds,
      ),
    );

    pushUploadSummaryToast(
      results,
      uploadInputs.map((input) => input.file.name),
      action.tags,
      pushToast,
    );
    dispatchUploadAudit(results, uploadInputs, currentActor, activity);
    maybeApplyInitialTags(results, action.tags, applyInitialTagsAfterIngestion);
    void refreshDocumentsUntilUploadsLand(acceptedTrackIds);
  };

  const applyInitialTagsAfterIngestion = async (
    trackIds: readonly string[],
    tags: readonly string[],
  ): Promise<void> => {
    const e2ePoll = globalThis.window?.__TWIN_E2E_INITIAL_TAG_POLL;
    const pollIntervalMs = e2ePoll?.intervalMs ?? 2000;
    const maxPolls = e2ePoll?.maxPolls ?? 30;
    const resolvedDocIds = await pollProcessedDocIds({
      trackIds,
      pollIntervalMs,
      maxPolls,
      pushToast,
    });
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
      logTechnicalError('initial-tags', err);
      pushToast({
        kind: 'error',
        title: 'Initial tags failed',
        sub: userErrorMessage(err, { action: 'applying initial tags' }),
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
