/**
 * Unit tests for useDocumentActions.
 *
 * The hook orchestrates upload / retag / delete / undo / scan-retry flows on
 * top of the TanStack mutation hooks. We mock `../api/resources` so the real
 * `../api/queries` mutations run their plumbing against controllable
 * resolvers, and drive every callback the hook returns plus its private
 * helpers (optimistic upload patching, initial-tag polling, document
 * refresh loop) through the public surface.
 *
 * Polling/refresh loops use `globalThis.setTimeout`; we install Vitest fake
 * timers and the `__TWIN_E2E_INITIAL_TAG_POLL` window override so the
 * intervals collapse to near-zero without changing source behavior.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { useDocumentActions } from './useDocumentActions';
import { ApiError } from '../api/client';
import { queryClient } from './queryClient';
import type { RetagAction } from '../components/RetagModal';
import type { AddSourceAction } from '../components/AddSourceModal';
import type { Document } from '../types/document';
import type { Folder } from '../types/topbar';
import type { Toast } from '../types/toast';

// ── Mock the resource layer ────────────────────────────────────────────────
const bulkRetagDocumentsMock = vi.hoisted(() => vi.fn());
const uploadDocumentMock = vi.hoisted(() => vi.fn());
const bulkDeleteDocumentsMock = vi.hoisted(() => vi.fn());
const reprocessFailedDocumentsMock = vi.hoisted(() => vi.fn());
const recordSourceUploadedMock = vi.hoisted(() => vi.fn());
const trackStatusMock = vi.hoisted(() => vi.fn());
// api.deleteDocument delegates to bulkDeleteDocuments in resources.ts; the
// hook calls deleteDoc.mutateAsync(doc_id), so we expose a deleteDocument mock.
const deleteDocumentMock = vi.hoisted(() => vi.fn());

vi.mock('../api/resources', () => ({
  api: {
    bulkRetagDocuments: bulkRetagDocumentsMock,
    uploadDocument: uploadDocumentMock,
    bulkDeleteDocuments: bulkDeleteDocumentsMock,
    deleteDocument: deleteDocumentMock,
    reprocessFailedDocuments: reprocessFailedDocumentsMock,
    recordSourceUploaded: recordSourceUploadedMock,
    trackStatus: trackStatusMock,
  },
}));

const ACTOR = 'claire.benoit';

function makeDoc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'doc-1',
    track_id: null,
    file_path: 'rman.pdf',
    content_summary: '',
    content_length: 0,
    status: 'PROCESSED',
    chunks_count: 0,
    created_at: '2026-01-01',
    updated_at: '2026-01-01',
    error_msg: null,
    metadata: {},
    type: 'file',
    tags: [],
    folder: 'WORKSPACE',
    visibility: 'internal',
    ...overrides,
  };
}

const FOLDER_LIST: readonly Folder[] = [
  {
    id: 'WORKSPACE',
    kb: 'Main KB',
    visibility: 'public',
    sources: 3,
    role: 'steward',
    current: true,
  },
];

interface HarnessOptions {
  pushToast?: (toast: Omit<Toast, 'id'>) => void;
  setAddOpen?: (v: boolean | ((p: boolean) => boolean)) => void;
  setOptimisticUploadDocs?: (
    v:
      | readonly Document[]
      | ((p: readonly Document[]) => readonly Document[]),
  ) => void;
  setToasts?: (v: Toast[] | ((p: Toast[]) => Toast[])) => void;
  refetchActivity?: () => unknown;
  refetchDocs?: () => unknown;
  folder?: string;
}

function setup(opts: HarnessOptions = {}) {
  const pushToast = opts.pushToast ?? vi.fn();
  const setAddOpen = opts.setAddOpen ?? vi.fn();
  const setOptimisticUploadDocs = opts.setOptimisticUploadDocs ?? vi.fn();
  const setToasts = opts.setToasts ?? vi.fn();
  const refetchActivity = opts.refetchActivity ?? vi.fn();
  const refetchDocs =
    opts.refetchDocs ?? vi.fn().mockResolvedValue({ data: { items: [] } });

  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );

  const docs = { refetch: refetchDocs } as unknown as Parameters<
    typeof useDocumentActions
  >[0]['docs'];

  const rendered = renderHook(
    () =>
      useDocumentActions({
        activity: { refetch: refetchActivity },
        currentActor: ACTOR,
        docs,
        folder: opts.folder ?? 'WORKSPACE',
        folderList: FOLDER_LIST,
        pushToast,
        setAddOpen,
        setOptimisticUploadDocs,
        setToasts,
      }),
    { wrapper: Wrapper },
  );

  return {
    ...rendered,
    pushToast,
    setAddOpen,
    setOptimisticUploadDocs,
    setToasts,
    refetchActivity,
    refetchDocs,
    client,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  bulkRetagDocumentsMock.mockResolvedValue({ updated: 1, failed: [] });
  uploadDocumentMock.mockResolvedValue({
    status: 'success',
    message: 'queued',
    track_id: 'trk-1',
  });
  bulkDeleteDocumentsMock.mockResolvedValue({ deleted: 1 });
  deleteDocumentMock.mockResolvedValue({ ok: true });
  reprocessFailedDocumentsMock.mockResolvedValue({
    status: 'ok',
    message: 'queued',
    failed_count: 2,
  });
  recordSourceUploadedMock.mockResolvedValue({ ok: true });
  trackStatusMock.mockResolvedValue({
    track_id: 'trk-1',
    documents: [],
    total_count: 0,
    status_summary: {},
  });
  // Collapse the initial-tag poll loop so tests don't wait real seconds.
  (globalThis.window as unknown as Record<string, unknown>).__TWIN_E2E_INITIAL_TAG_POLL =
    { intervalMs: 1, maxPolls: 3 };
});

afterEach(() => {
  vi.useRealTimers();
  delete (globalThis.window as unknown as Record<string, unknown>)
    .__TWIN_E2E_INITIAL_TAG_POLL;
  vi.restoreAllMocks();
});

// ─────────────────────────────────────────────────────────────────────────
describe('onRetagSubmit', () => {
  function retagAction(overrides: Partial<RetagAction> = {}): RetagAction {
    const primary = makeDoc({ doc_id: 'doc-1', file_path: 'a.pdf' });
    return {
      primary,
      targets: [primary],
      bulk: false,
      adds: ['oracle'],
      removes: [],
      ...overrides,
    };
  }

  it('applies tags and pushes a done toast with undo payload', async () => {
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onRetagSubmit(retagAction());
    });
    expect(bulkRetagDocumentsMock).toHaveBeenCalledWith({
      targets: ['doc-1'],
      adds: ['oracle'],
      removes: [],
      actor: ACTOR,
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'done',
        title: 'Tag',
        tagname: 'oracle',
        titleSuffix: 'applied',
        undo: { targets: ['doc-1'], adds: ['oracle'], removes: [] },
      }),
    );
  });

  it('bulk with removes-only → "removed to N sources" suffix and skipped count', async () => {
    bulkRetagDocumentsMock.mockResolvedValueOnce({
      updated: 2,
      failed: ['doc-x'],
    });
    const a = makeDoc({ doc_id: 'd1', file_path: 'a' });
    const b = makeDoc({ doc_id: 'd2', file_path: 'b' });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onRetagSubmit({
        primary: a,
        targets: [a, b],
        bulk: true,
        adds: [],
        removes: ['legacy'],
      });
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        tagname: 'legacy',
        titleSuffix: 'removed to 2 sources · 1 skipped',
      }),
    );
  });

  it('error → "Tag update failed" toast with mapped copy (no raw message)', async () => {
    bulkRetagDocumentsMock.mockRejectedValueOnce(new Error('db down'));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onRetagSubmit(retagAction());
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Tag update failed',
        sub: 'Something went wrong while updating tags on 1 source. Please retry or contact Twincore Team.',
      }),
    );
  });

  it('error non-Error → generic mapped copy with the source count', async () => {
    bulkRetagDocumentsMock.mockRejectedValueOnce('weird');
    const a = makeDoc({ doc_id: 'd1' });
    const b = makeDoc({ doc_id: 'd2' });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onRetagSubmit({
        primary: a,
        targets: [a, b],
        bulk: true,
        adds: ['x'],
        removes: [],
      });
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        sub: 'Something went wrong while updating tags on 2 sources. Please retry or contact Twincore Team.',
      }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onToastUndo', () => {
  const undoToast: Toast = {
    id: 'tst-1',
    kind: 'done',
    title: 'Tag',
    tagname: 'oracle',
    sub: 'a.pdf',
    undo: { targets: ['doc-1'], adds: ['oracle'], removes: ['old'] },
  };

  it('removes the toast and applies the inverse retag', async () => {
    const setToasts = vi.fn();
    const { result, pushToast } = setup({ setToasts });
    await act(async () => {
      await result.current.onToastUndo(undoToast);
    });
    // toast filtered out
    expect(setToasts).toHaveBeenCalledWith(expect.any(Function));
    // inverse: adds<->removes swapped
    expect(bulkRetagDocumentsMock).toHaveBeenCalledWith({
      targets: ['doc-1'],
      adds: ['old'],
      removes: ['oracle'],
      actor: ACTOR,
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'done', title: 'Undo applied' }),
    );
  });

  it('returns early (no mutation) when the toast has no valid undo payload', async () => {
    const { result } = setup();
    await act(async () => {
      await result.current.onToastUndo({
        id: 't2',
        kind: 'done',
        title: 'X',
        undo: undefined,
      });
    });
    expect(bulkRetagDocumentsMock).not.toHaveBeenCalled();
  });

  it('error → "Undo failed" toast with mapped copy', async () => {
    bulkRetagDocumentsMock.mockRejectedValueOnce(new Error('nope'));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onToastUndo(undoToast);
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Undo failed',
        sub: 'Something went wrong while undoing the tag change. Please retry or contact Twincore Team.',
      }),
    );
  });

  it('error non-Error → same mapped fallback copy', async () => {
    bulkRetagDocumentsMock.mockRejectedValueOnce('x');
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onToastUndo(undoToast);
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        sub: 'Something went wrong while undoing the tag change. Please retry or contact Twincore Team.',
      }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onDeleteSingle', () => {
  it('deletes and pushes a done toast', async () => {
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteSingle({
        doc_id: 'doc-1',
        file_path: 'a.pdf',
      });
    });
    expect(deleteDocumentMock).toHaveBeenCalledWith('doc-1');
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'done', title: 'Document removed' }),
    );
  });

  it('error Error → mapped copy naming the file in sub', async () => {
    deleteDocumentMock.mockRejectedValueOnce(new Error('locked'));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteSingle({
        doc_id: 'doc-1',
        file_path: 'a.pdf',
      });
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Delete failed',
        sub: 'Something went wrong while deleting a.pdf. Please retry or contact Twincore Team.',
      }),
    );
  });

  it('error non-Error → same mapped fallback copy', async () => {
    deleteDocumentMock.mockRejectedValueOnce('x');
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteSingle({
        doc_id: 'doc-1',
        file_path: 'a.pdf',
      });
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        sub: 'Something went wrong while deleting a.pdf. Please retry or contact Twincore Team.',
      }),
    );
  });

  it('pipeline-busy 409 → explicit action-not-taken toast', async () => {
    deleteDocumentMock.mockRejectedValueOnce(
      new ApiError('DELETE /documents/doc-1 → 409 Conflict', 409, {
        detail: 'Pipeline is busy. Please try again later',
      }),
    );
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteSingle({
        doc_id: 'doc-1',
        file_path: 'a.pdf',
      });
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Delete failed',
        sub: 'Action not taken while deleting a.pdf: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.',
      }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onDeleteBulk', () => {
  it('returns early on empty selection', async () => {
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteBulk([]);
    });
    expect(pushToast).not.toHaveBeenCalled();
    expect(bulkDeleteDocumentsMock).not.toHaveBeenCalled();
  });

  it('deletes a batch → propagating then done toast', async () => {
    bulkDeleteDocumentsMock.mockResolvedValueOnce({ deleted: 2 });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteBulk([
        { doc_id: 'a', file_path: 'a' },
        { doc_id: 'b', file_path: 'b' },
      ]);
    });
    expect(bulkDeleteDocumentsMock).toHaveBeenCalledWith({
      doc_ids: ['a', 'b'],
      actor: ACTOR,
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'propagating', title: 'Deleting sources…' }),
    );
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'done', title: '2 sources deleted' }),
    );
  });

  it('single doc → singular grammar in propagating toast', async () => {
    bulkDeleteDocumentsMock.mockResolvedValueOnce({ deleted: 1 });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteBulk([{ doc_id: 'a', file_path: 'a' }]);
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'propagating',
        sub: expect.stringContaining('1 source being removed'),
      }),
    );
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: '1 source deleted' }),
    );
  });

  it('error → "Bulk delete failed" toast', async () => {
    bulkDeleteDocumentsMock.mockRejectedValueOnce(new Error('cascade boom'));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onDeleteBulk([{ doc_id: 'a', file_path: 'a' }]);
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Bulk delete failed',
        sub: 'Something went wrong while deleting the selected sources. Please retry or contact Twincore Team.',
      }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onScanRetry', () => {
  it('reprocesses failed docs → propagating then done, and invalidates queries', async () => {
    const invalidateSpy = vi
      .spyOn(queryClient, 'invalidateQueries')
      .mockResolvedValue(undefined);
    const { result, pushToast } = setup();
    await act(async () => {
      result.current.onScanRetry(3);
      await Promise.resolve();
      await Promise.resolve();
    });
    await waitFor(() => expect(reprocessFailedDocumentsMock).toHaveBeenCalled());
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'propagating',
        title: 'Re-processing failed sources',
      }),
    );
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'done', title: 'Reprocess request sent' }),
      ),
    );
    const keys = invalidateSpy.mock.calls.map(
      ([opts]) => (opts as { queryKey: unknown[] }).queryKey[0],
    );
    expect(keys).toEqual(
      expect.arrayContaining(['documents', 'pipeline_status', 'activity']),
    );
  });

  it('falls back to failed_count when message is absent', async () => {
    reprocessFailedDocumentsMock.mockResolvedValueOnce({
      status: 'ok',
      failed_count: 7,
    });
    vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue(undefined);
    const { result, pushToast } = setup();
    await act(async () => {
      result.current.onScanRetry(2);
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Reprocess request sent',
          sub: '7 failed sources queued for retry',
        }),
      ),
    );
  });

  it('error → "Re-process failed" toast and still invalidates', async () => {
    reprocessFailedDocumentsMock.mockRejectedValueOnce(new Error('endpoint 500'));
    const invalidateSpy = vi
      .spyOn(queryClient, 'invalidateQueries')
      .mockResolvedValue(undefined);
    const { result, pushToast } = setup();
    await act(async () => {
      result.current.onScanRetry(1);
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Re-process failed',
          sub: 'Something went wrong while re-processing failed sources. Please retry or contact Twincore Team.',
        }),
      ),
    );
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onAddSourceSubmit — no files (URL-only) path', () => {
  it('pushes a queued toast and closes the modal without uploading', async () => {
    const setAddOpen = vi.fn();
    const { result, pushToast } = setup({ setAddOpen });
    const action: AddSourceAction = {
      files: [],
      rawFiles: [],
      fileOptions: [],
      urls: [],
      tags: [],
      readyCount: 3,
    };
    await act(async () => {
      await result.current.onAddSourceSubmit(action);
    });
    expect(uploadDocumentMock).not.toHaveBeenCalled();
    expect(setAddOpen).toHaveBeenCalledWith(false);
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'done', title: 'Sources queued', sub: '3 entries' }),
    );
  });

  it('singular grammar for a single entry', async () => {
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit({
        files: [],
        rawFiles: [],
        fileOptions: [],
        urls: [],
        tags: [],
        readyCount: 1,
      });
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ sub: '1 entry' }),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onAddSourceSubmit — file upload path', () => {
  function fileAction(tags: readonly string[] = []): AddSourceAction {
    return {
      files: [],
      rawFiles: [new File(['hi'], 'a.txt', { type: 'text/plain' })],
      fileOptions: [],
      urls: [],
      tags,
      readyCount: 1,
    };
  }

  it('uploads, patches optimistic docs, pushes success toast, records audit', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-9',
    });
    const setOptimisticUploadDocs = vi.fn();
    const setAddOpen = vi.fn();
    const refetchActivity = vi.fn();
    const { result, pushToast } = setup({
      setOptimisticUploadDocs,
      setAddOpen,
      refetchActivity,
    });

    await act(async () => {
      await result.current.onAddSourceSubmit(fileAction());
      // flush the dispatched (fire-and-forget) audit + refresh microtasks
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(uploadDocumentMock).toHaveBeenCalledTimes(1);
    expect(setAddOpen).toHaveBeenCalledWith(false);
    // optimistic insert + patch (two setOptimisticUploadDocs calls)
    expect(setOptimisticUploadDocs.mock.calls.length).toBeGreaterThanOrEqual(2);
    // propagating then success toast
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'propagating', title: 'Uploading sources…' }),
    );
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'done',
        title: 'Sources queued for ingestion',
      }),
    );
    await waitFor(() => expect(recordSourceUploadedMock).toHaveBeenCalled());
  });

  it('patchOptimisticUploadDocs: stamps accepted docs with track_id + upload_state and drops failed ones', async () => {
    // Two files: first succeeds, second rejects → patch updater keeps #1
    // (stamped) and removes #2.
    uploadDocumentMock
      .mockResolvedValueOnce({
        status: 'success',
        message: 'queued',
        track_id: 'trk-ok',
      })
      .mockRejectedValueOnce(new ApiError('boom', 500, null));

    const setOptimisticUploadDocs = vi.fn();
    const { result } = setup({ setOptimisticUploadDocs });
    await act(async () => {
      await result.current.onAddSourceSubmit({
        files: [],
        rawFiles: [
          new File(['a'], 'a.txt', { type: 'text/plain' }),
          new File(['b'], 'b.txt', { type: 'text/plain' }),
        ],
        fileOptions: [],
        urls: [],
        tags: [],
        readyCount: 2,
      });
      await Promise.resolve();
    });

    // Reconstruct the optimistic insert (call 0) then feed it to the patch
    // updater (call 1) to exercise patchOptimisticUploadDocs.
    const insert = setOptimisticUploadDocs.mock.calls[0][0] as (
      p: readonly Document[],
    ) => readonly Document[];
    const optimistic = insert([]);
    expect(optimistic).toHaveLength(2);
    const patch = setOptimisticUploadDocs.mock.calls[1][0] as (
      p: readonly Document[],
    ) => readonly Document[];
    const patched = patch(optimistic);
    // Failed upload's optimistic doc removed; accepted one stamped.
    expect(patched).toHaveLength(1);
    expect(patched[0].track_id).toBe('trk-ok');
    expect((patched[0].metadata as Record<string, unknown>).upload_state).toBe(
      'success',
    );
    // A doc not in the accepted map passes through unchanged.
    const stranger = makeDoc({ doc_id: 'unrelated' });
    expect(patch([...optimistic, stranger])).toContainEqual(stranger);
  });

  it('patchOptimisticUploadDocs: duplicated upload stamps the duplicate summary copy', async () => {
    uploadDocumentMock.mockResolvedValueOnce({
      status: 'duplicated',
      message: 'dup',
      track_id: 'trk-dup',
    });
    const setOptimisticUploadDocs = vi.fn();
    const { result } = setup({ setOptimisticUploadDocs });
    await act(async () => {
      await result.current.onAddSourceSubmit(fileAction());
      await Promise.resolve();
    });
    const insert = setOptimisticUploadDocs.mock.calls[0][0] as (
      p: readonly Document[],
    ) => readonly Document[];
    const patch = setOptimisticUploadDocs.mock.calls[1][0] as (
      p: readonly Document[],
    ) => readonly Document[];
    const patched = patch(insert([]));
    expect(patched[0].content_summary).toContain('duplicate');
  });

  it('mixed failed + duplicate batch → error toast carries the "already present" suffix', async () => {
    uploadDocumentMock
      .mockResolvedValueOnce({
        status: 'duplicated',
        message: 'dup',
        track_id: 'trk-dup',
      })
      .mockRejectedValueOnce(new ApiError('boom', 500, null));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit({
        files: [],
        rawFiles: [
          new File(['a'], 'a.txt', { type: 'text/plain' }),
          new File(['b'], 'b.txt', { type: 'text/plain' }),
        ],
        fileOptions: [],
        urls: [],
        tags: [],
        readyCount: 2,
      });
      await Promise.resolve();
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        sub: expect.stringContaining('already present'),
      }),
    );
  });

  it('reports a failed upload via the error summary toast', async () => {
    uploadDocumentMock.mockRejectedValueOnce(
      new ApiError('boom', 500, { detail: 'x' }),
    );
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(fileAction());
      await Promise.resolve();
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: '1 upload failed',
      }),
    );
  });

  it('backend unsupported-type 400 → summary names the file and the format', async () => {
    // Error-UX pass 2026-07-03: the summary explains WHY the upload failed
    // ("ZIP format is not supported"), never LightRAG's raw detail or the
    // transport string.
    uploadDocumentMock.mockRejectedValueOnce(
      new ApiError('POST /documents/upload → 400 Bad Request', 400, {
        detail: "Unsupported file type. Supported types: ('.pdf', '.docx')",
      }),
    );
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit({
        files: [],
        rawFiles: [new File(['z'], 'archive.zip', { type: 'application/zip' })],
        fileOptions: [],
        urls: [],
        tags: [],
        readyCount: 1,
      });
      await Promise.resolve();
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: '1 upload failed',
        sub: '0 uploaded · 1 failed — archive.zip: ZIP format is not supported',
      }),
    );
  });

  it('pipeline-busy upload refusal is surfaced in the batch failure toast', async () => {
    uploadDocumentMock.mockRejectedValueOnce(
      new ApiError('POST /documents/upload → 409 Conflict', 409, {
        detail: 'Pipeline is busy. Please try again later',
      }),
    );
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(fileAction());
      await Promise.resolve();
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: '1 upload failed',
        sub: '0 uploaded · 1 failed — a.txt: Action not taken while uploading the file: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.',
      }),
    );
  });

  it('duplicated upload surfaces the "(N already present)" suffix', async () => {
    uploadDocumentMock.mockResolvedValueOnce({
      status: 'duplicated',
      message: 'dup',
      track_id: 'trk-d',
    });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(fileAction());
      await Promise.resolve();
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'done',
        sub: expect.stringContaining('already present'),
      }),
    );
  });

  it('uses the folder visibility fallback when folder is not in folderList', async () => {
    const setOptimisticUploadDocs = vi.fn();
    const { result } = setup({
      folder: 'UNKNOWN_FOLDER',
      setOptimisticUploadDocs,
    });
    await act(async () => {
      await result.current.onAddSourceSubmit(fileAction());
      await Promise.resolve();
    });
    // first call inserts the optimistic docs; visibility should default to 'internal'
    const inserter = setOptimisticUploadDocs.mock.calls[0][0] as (
      p: readonly Document[],
    ) => readonly Document[];
    const inserted = inserter([]);
    expect(inserted[0].visibility).toBe('internal');
    expect(inserted[0].status).toBe('PENDING');
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('onAddSourceSubmit — initial tags poll (applyInitialTagsAfterIngestion)', () => {
  function tagFileAction(): AddSourceAction {
    return {
      files: [],
      rawFiles: [new File(['hi'], 'a.txt', { type: 'text/plain' })],
      fileOptions: [],
      urls: [],
      tags: ['oracle'],
      readyCount: 1,
    };
  }

  it('applies initial tags once the track reaches a terminal processed state', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    // First poll returns processed terminal doc → resolves doc id.
    trackStatusMock.mockResolvedValue({
      track_id: 'trk-1',
      documents: [{ id: 'doc-x', status: 'processed', file_path: 'a.txt' }],
      total_count: 1,
      status_summary: { processed: 1 },
    });

    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      // let the fire-and-forget poll + bulk-retag settle
      await new Promise((r) => setTimeout(r, 20));
    });

    await waitFor(() =>
      expect(bulkRetagDocumentsMock).toHaveBeenCalledWith(
        expect.objectContaining({ targets: ['doc-x'], adds: ['oracle'] }),
      ),
    );
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'done', title: 'Initial tags applied' }),
      ),
    );
  });

  it('pushes "not applied" when polling never reaches a terminal state', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    // Always non-terminal → maxPolls exhausted, resolvedDocIds stays empty.
    trackStatusMock.mockResolvedValue({
      track_id: 'trk-1',
      documents: [{ id: 'doc-x', status: 'processing', file_path: 'a.txt' }],
      total_count: 1,
      status_summary: { processing: 1 },
    });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      await new Promise((r) => setTimeout(r, 30));
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Initial tags not applied',
        }),
      ),
    );
  });

  it('error from bulk-retag during initial tags → "Initial tags failed" toast', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    trackStatusMock.mockResolvedValue({
      track_id: 'trk-1',
      documents: [{ id: 'doc-x', status: 'processed', file_path: 'a.txt' }],
      total_count: 1,
      status_summary: { processed: 1 },
    });
    bulkRetagDocumentsMock.mockRejectedValue(new Error('retag boom'));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      await new Promise((r) => setTimeout(r, 20));
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Initial tags failed',
          sub: 'Something went wrong while applying initial tags. Please retry or contact Twincore Team.',
        }),
      ),
    );
  });

  it('success summary carries the "initial tags will apply" suffix when tags are present', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    // Never terminal so the retag side never fires; we only assert the
    // upload summary toast suffix here.
    trackStatusMock.mockResolvedValue({
      track_id: 'trk-1',
      documents: [{ id: 'doc-x', status: 'processing', file_path: 'a.txt' }],
      total_count: 1,
      status_summary: { processing: 1 },
    });
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      await new Promise((r) => setTimeout(r, 20));
    });
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'done',
        title: 'Sources queued for ingestion',
        sub: expect.stringContaining('initial tags will apply'),
      }),
    );
  });

  it('initial tags failed with a non-Error → mapped fallback copy', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    trackStatusMock.mockResolvedValue({
      track_id: 'trk-1',
      documents: [{ id: 'doc-x', status: 'processed', file_path: 'a.txt' }],
      total_count: 1,
      status_summary: { processed: 1 },
    });
    bulkRetagDocumentsMock.mockRejectedValue('weird-non-error');
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      await new Promise((r) => setTimeout(r, 20));
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Initial tags failed',
          sub: 'Something went wrong while applying initial tags. Please retry or contact Twincore Team.',
        }),
      ),
    );
  });

  it('retries on a transient 404 then resolves (shouldRetryTrackStatus 404 branch)', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    trackStatusMock
      .mockRejectedValueOnce(new ApiError('not yet', 404, null))
      .mockResolvedValue({
        track_id: 'trk-1',
        documents: [{ id: 'doc-x', status: 'processed', file_path: 'a.txt' }],
        total_count: 1,
        status_summary: { processed: 1 },
      });
    const { result } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      await new Promise((r) => setTimeout(r, 30));
    });
    await waitFor(() =>
      expect(bulkRetagDocumentsMock).toHaveBeenCalledWith(
        expect.objectContaining({ targets: ['doc-x'] }),
      ),
    );
  });

  it('gives up after 3 transient non-404 errors and warns (shouldRetryTrackStatus exhaustion)', async () => {
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-1',
    });
    trackStatusMock.mockRejectedValue(new Error('transient'));
    const { result, pushToast } = setup();
    await act(async () => {
      await result.current.onAddSourceSubmit(tagFileAction());
      await new Promise((r) => setTimeout(r, 40));
    });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Initial tag polling failed',
        }),
      ),
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────
describe('refreshDocumentsUntilUploadsLand (via successful upload)', () => {
  it('polls docs.refetch until the accepted track_ids land', async () => {
    // The refresh loop uses a hardcoded 2000ms globalThis.setTimeout that the
    // e2e poll override does not shorten. Drive it with fake timers so the
    // loop body (docs.refetch + track reconciliation) actually executes.
    vi.useFakeTimers();
    uploadDocumentMock.mockResolvedValue({
      status: 'success',
      message: 'queued',
      track_id: 'trk-land',
    });
    // First refetch: track not present. Second: present → loop exits.
    const refetchDocs = vi
      .fn()
      .mockResolvedValueOnce({ data: { items: [] } })
      .mockResolvedValue({
        data: { items: [{ track_id: 'trk-land' }] },
      });
    const { result } = setup({ refetchDocs });

    let submit!: Promise<void>;
    await act(async () => {
      submit = result.current.onAddSourceSubmit({
        files: [],
        rawFiles: [new File(['x'], 'a.txt')],
        fileOptions: [],
        urls: [],
        tags: [],
        readyCount: 1,
      });
    });
    // Let the awaited upload + synchronous post-upload work settle.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
      await submit;
    });
    // Now drive the detached 2000ms refresh loop through two iterations.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(refetchDocs).toHaveBeenCalled();
  });
});
