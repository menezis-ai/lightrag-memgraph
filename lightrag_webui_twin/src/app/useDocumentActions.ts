import { useRef, type Dispatch, type SetStateAction } from 'react';
import { ApiError } from '../api/client';
import {
  useBulkDeleteDocuments,
  useBulkRetagDocuments,
  useDeleteDocument,
  useDocuments,
  useUploadDocumentsBatch,
} from '../api/queries';
import {
  api,
  type UploadDocumentInput,
  type UploadDocumentResponse,
} from '../api/resources';
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

type UploadResponse = UploadDocumentResponse;
type UploadResult = PromiseSettledResult<UploadResponse>;
type TrackStatus = Awaited<ReturnType<typeof api.trackStatus>>;
type PushToast = (toast: Omit<Toast, 'id'>) => void;

/** Retry cadence for deletions the busy ingestion pipeline defers (423 /
 *  `busy` ids in a 207). The tab-local queue re-issues the delete until the
 *  pipeline drains or the deadline passes — the backend guarantees deferred
 *  docs are untouched, so the retry is idempotent. */
export const BULK_DELETE_BUSY_RETRY_MS = 15_000;
export const BULK_DELETE_BUSY_DEADLINE_MS = 10 * 60_000;

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

const isPipelineBusyDeleteError = (err: unknown): boolean =>
  err instanceof ApiError && err.status === 423;

function recoveryRequiredDeleteProgress(
  err: unknown,
): {
  deleted: number;
  failed: readonly string[];
  busy: readonly string[];
  unattempted: readonly string[];
} | null {
  if (!(err instanceof ApiError) || err.status !== 503) return null;
  if (!err.body || typeof err.body !== 'object') return null;
  const body = err.body as Record<string, unknown>;
  if (body.recovery_required !== true) return null;
  const deleted =
    typeof body.deleted === 'number' && Number.isSafeInteger(body.deleted)
      ? Math.max(0, body.deleted)
      : 0;
  const failed = Array.isArray(body.failed)
    ? body.failed.filter((value): value is string => typeof value === 'string')
    : [];
  const busy = Array.isArray(body.busy)
    ? body.busy.filter((value): value is string => typeof value === 'string')
    : [];
  const unattempted = Array.isArray(body.unattempted)
    ? body.unattempted.filter(
        (value): value is string => typeof value === 'string',
      )
    : [];
  return { deleted, failed, busy, unattempted };
}

async function parkedProcedureTrackIds(
  pending: ReadonlySet<string>,
): Promise<readonly string[]> {
  try {
    const bundles = await api.listProcedures();
    return bundles.flatMap((bundle) =>
      bundle.track_id && pending.has(bundle.track_id) ? [bundle.track_id] : [],
    );
  } catch {
    // The pending section reports procedure-store errors. Document polling can
    // still make progress when only this reconciliation endpoint is down.
    return [];
  }
}

function registerParkedTrackIds(
  pending: Set<string>,
  parked: Set<string>,
  trackIds: readonly string[],
): void {
  for (const trackId of trackIds) {
    pending.delete(trackId);
    parked.add(trackId);
  }
}

function removeResolvedDocumentTrackIds(
  pending: Set<string>,
  items: readonly Document[],
): Set<string> {
  const resolved = new Set<string>();
  for (const item of items) {
    if (item.track_id && pending.delete(item.track_id)) {
      resolved.add(item.track_id);
    }
  }
  return resolved;
}

function summarizeUploadResults(results: readonly UploadResult[]) {
  const ok = results.filter((result) => result.status === 'fulfilled').length;
  const ko = results.length - ok;
  const shared = results.filter(
    (result) => result.status === 'fulfilled' && result.value.status === 'shared',
  ).length;
  const dup = results.filter(
    (result) =>
      result.status === 'fulfilled' && result.value.status === 'duplicated',
  ).length;
  return { ok, ko, dup, shared };
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
  return docs.flatMap((doc) => {
    if (failedIds.has(doc.doc_id)) return [];
    const result = acceptedByOptimisticId.get(doc.doc_id);
    if (!result) return [doc];
    // Sharing is synchronous: the authoritative document already exists and
    // the documents query is invalidated by the batch mutation. Keeping an
    // optimistic ingestion row would show a false pending duplicate.
    if (result.doc_id) return [];
    return [
      {
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
      },
    ];
  });
}

function uploadResultTrackIds(results: readonly UploadResult[]): string[] {
  return results.flatMap((result) =>
    result.status === 'fulfilled' && !result.value.doc_id && result.value.track_id
      ? [result.value.track_id]
      : [],
  );
}

function uploadResultExistingDocIds(results: readonly UploadResult[]): string[] {
  return results.flatMap((result) =>
    result.status === 'fulfilled' && result.value.doc_id
      ? [result.value.doc_id]
      : [],
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
  activity: { refetch: () => unknown },
) {
  // R-03a (security audit 2026-08-06): the authoritative source-uploaded
  // event is now emitted server-side by the ingestion pipeline; the client
  // no longer writes audit entries. Just refresh the feed.
  Promise.resolve()
    .then(() => activity.refetch())
    .catch(() => undefined);
}

function maybeApplyInitialTags(
  results: readonly UploadResult[],
  tags: readonly string[],
  applyInitialTagsAfterIngestion: (
    trackIds: readonly string[],
    tags: readonly string[],
    existingDocIds: readonly string[],
  ) => Promise<void>,
) {
  if (!tags.length) return;
  const trackIds = uploadResultTrackIds(results);
  const existingDocIds = uploadResultExistingDocIds(results);
  if (trackIds.length === 0 && existingDocIds.length === 0) return;
  void applyInitialTagsAfterIngestion(trackIds, tags, existingDocIds);
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
  const { ok, ko, dup, shared } = summarizeUploadResults(results);
  const duplicateSuffix = dup > 0 ? ` (${dup} already present)` : '';
  const initialTagsSuffix = tags.length
    ? ' · initial tags will apply once docs land'
    : '';
  if (shared > 0) {
    const queued = ok - shared - dup;
    const parts = [
      queued > 0 ? `${queued} uploaded` : '',
      `${shared} copied to this folder`,
      dup > 0 ? `${dup} already present` : '',
    ].filter(Boolean);
    if (ko === 0) {
      pushToast({
        kind: 'done',
        title: queued > 0 ? 'Sources accepted' : 'Sources added to folder',
        sub: `${parts.join(' · ')}${initialTagsSuffix}`,
      });
    } else {
      const reason = firstUploadFailureReason(results, fileNames);
      const reasonSuffix = reason ? ` — ${reason}` : '';
      pushToast({
        kind: 'error',
        title: `${ko} upload${ko === 1 ? '' : 's'} failed`,
        sub: `${parts.join(' · ')} · ${ko} failed${reasonSuffix}`,
      });
    }
    return;
  }
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

function retagGovernanceDetail(err: ApiError): string | null {
  if (err.status !== 422 || !err.body || typeof err.body !== 'object') {
    return null;
  }
  const detail = (err.body as Record<string, unknown>).detail;
  if (!detail || typeof detail !== 'object') return null;
  const payload = detail as Record<string, unknown>;
  const message =
    typeof payload.message === 'string' && payload.message.trim()
      ? payload.message.trim()
      : 'Only active, approved tags may be attached';
  const tags = Array.isArray(payload.unapproved_tags)
    ? payload.unapproved_tags.filter(
        (tag): tag is string => typeof tag === 'string' && Boolean(tag.trim()),
      )
    : [];
  return tags.length > 0 ? `${message}: ${tags.join(', ')}.` : `${message}.`;
}

function retagErrorMessage(
  err: unknown,
  targetCount: number,
  scope = 'retag',
  action = `updating tags on ${sourceCountLabel(targetCount)}`,
): string {
  logTechnicalError(scope, err);
  if (err instanceof ApiError) {
    const governanceDetail = retagGovernanceDetail(err);
    if (governanceDetail) return governanceDetail;
  }
  return userErrorMessage(err, { action });
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
  const uploadAbortRef = useRef<AbortController | null>(null);
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
      pushToast({
        kind: 'error',
        title: 'Undo failed',
        tagname: toast.tagname,
        sub: retagErrorMessage(
          err,
          undo.targets.length,
          'retag-undo',
          'undoing the tag change',
        ),
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
    const e2eRefresh = globalThis.window?.__TWIN_E2E_UPLOAD_REFRESH;
    const intervalMs = e2eRefresh?.intervalMs ?? 2000;
    // 60 × 2s: generous enough for the MarkItDown conversion tier, which
    // serializes conversions before the DocStatus row exists.
    const maxPolls = e2eRefresh?.maxPolls ?? 60;
    const pending = new Set(trackIds);
    const parked = new Set<string>();
    // QA DOC-V5-001: a resolved track must REMOVE its optimistic row from
    // state — the render-time mask alone (track_id / folder:file_path match)
    // left ghost "pending" rows behind on any projection mismatch, until a
    // manual page reload.
    const dropOptimisticRows = (resolved: ReadonlySet<string>) => {
      if (resolved.size === 0) return;
      setOptimisticUploadDocs((current) =>
        current.filter((doc) => !(doc.track_id && resolved.has(doc.track_id))),
      );
    };
    for (let i = 0; i < maxPolls && pending.size > 0; i += 1) {
      await new Promise((resolve) => globalThis.setTimeout(resolve, intervalMs));
      // A PARKED procedure (detected or forced) never lands in /documents —
      // the backend deliberately creates no document until approval. Its
      // optimistic row must resolve against the approval queue instead of
      // dangling forever, and the review card must appear without a manual
      // refresh. Reconciliation is exact: BundleSummary projects track_id.
      const newlyParked = await parkedProcedureTrackIds(pending);
      registerParkedTrackIds(pending, parked, newlyParked);
      if (newlyParked.length > 0) {
        setOptimisticUploadDocs((current) =>
          current.filter((doc) => !(doc.track_id && parked.has(doc.track_id))),
        );
        void queryClient.invalidateQueries({ queryKey: ['procedures'] });
      }
      if (pending.size === 0) break;
      const result = await docs.refetch();
      dropOptimisticRows(
        removeResolvedDocumentTrackIds(pending, result.data?.items ?? []),
      );
    }
    // Window exhausted: DocStatus rows are created at enqueue, so whatever is
    // still unresolved is a projection mismatch (e.g. an upload deduplicated
    // into an existing document that never surfaces this track_id). The
    // authoritative 2s document polling owns the list — drop the leftovers
    // instead of leaving ghost pending rows.
    dropOptimisticRows(pending);
    if (parked.size > 0) {
      pushToast({
        kind: 'done',
        title: 'Parked for review',
        sub: `${parked.size} procedure${parked.size === 1 ? '' : 's'} awaiting approval in the pending section`,
      });
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
    // Tab-local deletion queue: LightRAG refuses physical deletes while its
    // pipeline runs an ingestion job. The backend answers 423 (nothing done)
    // or 207 with the deferred ids in `busy` (untouched); both are transient,
    // so keep retrying the leftover until the pipeline drains instead of
    // surfacing a scary failure for an operation that just has to wait.
    let remaining: readonly string[] = docsToDelete.map((doc) => doc.doc_id);
    let deletedTotal = 0;
    const failedIds = new Set<string>();
    const progressSummary = () =>
      `${deletedTotal} deleted${
        failedIds.size > 0
          ? ` · ${failedIds.size} failed and still visible`
          : ''
      }`;
    let queuedToastShown = false;
    const deadline = Date.now() + BULK_DELETE_BUSY_DEADLINE_MS;
    for (;;) {
      let busy: readonly string[];
      try {
        const result = await bulkDeleteDocs.mutateAsync({
          doc_ids: remaining,
          actor: currentActor,
        });
        deletedTotal += result.deleted;
        for (const docId of result.failed ?? []) failedIds.add(docId);
        busy = result.busy ?? [];
      } catch (err) {
        if (!isPipelineBusyDeleteError(err)) {
          logTechnicalError('bulk-delete', err);
          const recovery = recoveryRequiredDeleteProgress(err);
          if (recovery) {
            deletedTotal += recovery.deleted;
            for (const docId of recovery.failed) failedIds.add(docId);
            const sourceNames = new Map(
              docsToDelete.map((doc) => [doc.doc_id, doc.file_path]),
            );
            const describeIds = (ids: readonly string[]) => {
              const sample = ids
                .slice(0, 3)
                .map((docId) => sourceNames.get(docId) ?? docId)
                .join(', ');
              const remainingCount = ids.length - 3;
              return remainingCount > 0
                ? `${sample}, +${remainingCount} more`
                : sample;
            };
            const outstanding = [
              recovery.busy.length > 0
                ? `${recovery.busy.length} deferred (${describeIds(recovery.busy)})`
                : '',
              recovery.unattempted.length > 0
                ? `${recovery.unattempted.length} not attempted (${describeIds(recovery.unattempted)})`
                : '',
            ].filter(Boolean);
            pushToast({
              kind: 'error',
              title: 'Workspace recovery required',
              sub: [progressSummary(), ...outstanding, userErrorMessage(err)].join(
                ' · ',
              ),
            });
            return;
          }
          const detail = userErrorMessage(err, {
            action: 'deleting the selected sources',
          });
          if (deletedTotal > 0 || failedIds.size > 0) {
            pushToast({
              kind: 'error',
              title: 'Bulk delete partially completed',
              sub: `${progressSummary()} · deletion stopped: ${detail}`,
            });
            return;
          }
          pushToast({
            kind: 'error',
            title: 'Bulk delete failed',
            sub: detail,
          });
          return;
        }
        busy = remaining;
      }
      if (busy.length === 0) {
        if (failedIds.size > 0) {
          pushToast({
            kind: 'error',
            title: 'Bulk delete partially completed',
            sub: `${progressSummary()} — retry the failed sources`,
          });
          return;
        }
        pushToast({
          kind: 'done',
          title: `${deletedTotal} source${deletedTotal === 1 ? '' : 's'} deleted`,
          sub: 'Cascade removal successful',
        });
        return;
      }
      if (Date.now() >= deadline) {
        pushToast({
          kind: 'error',
          title: 'Deletion still waiting on the pipeline',
          sub: `The ingestion pipeline stayed busy — ${busy.length} source${busy.length === 1 ? ' was' : 's were'} not deleted. ${progressSummary()}. Retry once document processing completes.`,
        });
        return;
      }
      remaining = [...busy];
      if (!queuedToastShown) {
        queuedToastShown = true;
        pushToast({
          kind: 'propagating',
          title: 'Deletion queued — pipeline busy',
          sub: `${busy.length} deletion${busy.length === 1 ? '' : 's'} will retry automatically once the current document processing finishes (keep this tab open).`,
        });
      }
      await sleep(BULK_DELETE_BUSY_RETRY_MS);
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

    const uploadAbort = new AbortController();
    uploadAbortRef.current = uploadAbort;
    const uploadInputs: readonly UploadDocumentInput[] = action.rawFiles.map(
      (file, index) => {
        const opts = action.fileOptions[index];
        return {
          file,
          signal: uploadAbort.signal,
          ...(action.onFileStateChange
            ? {
                onStateChange: (
                  state: 'uploading' | 'complete' | 'error',
                  error?: string,
                ) => action.onFileStateChange?.(index, state, error),
              }
            : {}),
          ...(opts?.relativePath ? { relativePath: opts.relativePath } : {}),
          ...(opts?.classification
            ? { classification: opts.classification }
            : {}),
          // Batch-level operator profile → per-upload X-Twin-Doc-Type header
          // (omitted for auto-detect, see api.uploadDocument).
          ...(action.docType ? { docType: action.docType } : {}),
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
    if (uploadAbortRef.current === uploadAbort) uploadAbortRef.current = null;
    if (results.every((result) => result.status === 'fulfilled')) {
      setAddOpen(false);
    }

    const {
      failedOptimisticIds,
      acceptedByOptimisticId,
    } = buildOptimisticUploadState(optimisticDocs, results);
    const acceptedTrackIds = Array.from(
      acceptedByOptimisticId.values(),
    ).flatMap((result) =>
      !result.doc_id && result.track_id ? [result.track_id] : [],
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
    dispatchUploadAudit(activity);
    maybeApplyInitialTags(results, action.tags, applyInitialTagsAfterIngestion);
    void refreshDocumentsUntilUploadsLand(acceptedTrackIds);
  };

  const cancelAddSourceUpload = () => {
    uploadAbortRef.current?.abort();
    pushToast({
      kind: 'done',
      title: 'Upload cancelled',
      sub: 'Requests not yet accepted by the server were stopped.',
    });
  };

  const applyInitialTagsAfterIngestion = async (
    trackIds: readonly string[],
    tags: readonly string[],
    existingDocIds: readonly string[] = [],
  ): Promise<void> => {
    const e2ePoll = globalThis.window?.__TWIN_E2E_INITIAL_TAG_POLL;
    const pollIntervalMs = e2ePoll?.intervalMs ?? 2000;
    const maxPolls = e2ePoll?.maxPolls ?? 30;
    const resolvedDocIds = new Set(existingDocIds);
    if (trackIds.length > 0) {
      const ingestedDocIds = await pollProcessedDocIds({
        trackIds,
        pollIntervalMs,
        maxPolls,
        pushToast,
      });
      for (const docId of ingestedDocIds) resolvedDocIds.add(docId);
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
        sub: retagErrorMessage(
          err,
          resolvedDocIds.size,
          'initial-tags',
          'applying initial tags',
        ),
      });
    }
  };

  return {
    uploadDocs,
    onAddSourceSubmit,
    cancelAddSourceUpload,
    onDeleteBulk,
    onDeleteSingle,
    onRetagSubmit,
    onScanRetry,
    onToastUndo,
  };
}
