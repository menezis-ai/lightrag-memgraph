/**
 * ProcedureReviewModal — side-by-side review of a parked procedure bundle.
 *
 * Left: the schematic PNG (base64, page-numbered, prev/next when several).
 * Right: the INFORMED description (title, description, task list) — the pass
 * that read the full document text. The blind pass (image-only) sits behind
 * a collapsible so the reviewer can compare. Below: the divergence report —
 * highlighted when blind and informed disagree (the whole point of the
 * double-pass doctrine), a subtle green line when coherent.
 *
 * Footer decisions map 1:1 to `server/procedure_routes.py`:
 *   Approve            POST /procedures/{id}/approve   (pending only)
 *   Reject             POST /procedures/{id}/reject    (pending/failed)
 *   Treat as standard  POST /procedures/{id}/reroute-standard
 *   Retry              POST /procedures/{id}/retry     (failed/rejected only)
 *
 * A folderless bundle (scan-created, no operator duplicate request) requires
 * an explicit target folder before Approve / Treat as standard — mirrors the
 * backend 422.
 *
 * RC-1 invariant: TanStack useMutation, onSuccess → invalidateQueries →
 * toast. An approved/rerouted bundle becomes a document, so those two also
 * invalidate ['documents'].
 */

import { useRef, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useProcedureBundle } from '../api/queries';
import { api } from '../api/resources';
import { useModalA11y } from '../hooks/useModalA11y';
import { logTechnicalError, userErrorMessage } from '../lib/errorMessages';
import { Icon } from './Icon';
import {
  bundleFolders,
  type PassPayload,
  type ProcedureBundle,
  type SchematicEntry,
} from '../types/procedure';
import type { Folder } from '../types/topbar';

export interface ProcedureReviewModalProps {
  bundleId: string;
  folderList?: readonly Folder[];
  onClose: () => void;
  onToast: (kind: 'done' | 'error', title: string, sub?: string) => void;
}

function PassBlock({
  pass,
  testId,
}: Readonly<{ pass: PassPayload; testId: string }>) {
  return (
    <div data-testid={testId}>
      <h3 style={{ fontSize: 14, margin: '0 0 4px' }}>{pass.title}</h3>
      <p className="muted" style={{ fontSize: 12.5, margin: '0 0 8px' }}>
        {pass.description}
      </p>
      {pass.tasks.length > 0 && (
        <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12 }}>
          {pass.tasks.map((task) => (
            <li key={task.id} style={{ marginBottom: 4 }}>
              <b>
                {task.id} — {task.title}
              </b>
              {task.responsible && <> · responsible: {task.responsible}</>}
              {task.conditions && <> · when: {task.conditions}</>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function DivergencePanel({
  schematic,
}: Readonly<{ schematic: SchematicEntry }>) {
  const divergence = schematic.divergence;
  if (!divergence) return null;
  if (divergence.coherent) {
    return (
      <div
        data-testid="procedure-review-divergence"
        data-coherent="true"
        style={{
          marginTop: 10,
          padding: '7px 10px',
          borderRadius: 'var(--border-radius-md)',
          background: 'var(--twin-green-50, #EAF6EC)',
          color: 'var(--twin-green-700)',
          fontSize: 12,
        }}
      >
        <Icon name="circle-check" size={12} /> {divergence.summary}
      </div>
    );
  }
  return (
    <div
      data-testid="procedure-review-divergence"
      data-coherent="false"
      style={{
        marginTop: 10,
        padding: '9px 12px',
        borderRadius: 'var(--border-radius-md)',
        background: 'var(--twin-amber-50)',
        border: '1px solid var(--twin-red-vivid)',
        fontSize: 12.5,
      }}
    >
      <div style={{ fontWeight: 600, color: 'var(--twin-red-vivid)' }}>
        <Icon name="alert-triangle" size={13} /> Blind and informed readings
        diverge
      </div>
      <p style={{ margin: '4px 0' }}>{divergence.summary}</p>
      <ul style={{ margin: 0, paddingLeft: 16 }}>
        {divergence.divergences.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function SchematicPane({
  schematic,
  index,
  total,
  onPrev,
  onNext,
  showBlind,
  onToggleBlind,
}: Readonly<{
  schematic: SchematicEntry;
  index: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  showBlind: boolean;
  onToggleBlind: () => void;
}>) {
  return (
    <>
      {total > 1 && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 8,
          }}
        >
          <button
            type="button"
            className="btn"
            onClick={onPrev}
            disabled={index === 0}
            data-testid="procedure-review-prev"
            aria-label="Previous schematic"
          >
            Prev
          </button>
          <span className="muted" style={{ fontSize: 12 }}>
            Schematic {index + 1} / {total}
          </span>
          <button
            type="button"
            className="btn"
            onClick={onNext}
            disabled={index >= total - 1}
            data-testid="procedure-review-next"
            aria-label="Next schematic"
          >
            Next <Icon name="chevron-right" size={13} />
          </button>
        </div>
      )}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 14,
          alignItems: 'start',
        }}
      >
        <div>
          <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
            Page {schematic.page}
          </div>
          {schematic.png_base64 ? (
            <img
              src={`data:image/png;base64,${schematic.png_base64}`}
              alt={`Schematic page ${schematic.page}`}
              data-testid="procedure-review-png"
              style={{
                width: '100%',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--border-radius-md)',
                background: '#fff',
              }}
            />
          ) : (
            <div className="muted" style={{ fontSize: 12 }}>
              No rendering available for this page.
            </div>
          )}
        </div>
        <div>
          {schematic.error && (
            <div
              className="muted"
              style={{ color: 'var(--twin-red-vivid)', fontSize: 12.5 }}
              data-testid="procedure-review-schematic-error"
            >
              <Icon name="alert-triangle" size={13} /> {schematic.error}
            </div>
          )}
          {schematic.informed && (
            <PassBlock
              pass={schematic.informed}
              testId="procedure-review-informed"
            />
          )}
          {!schematic.informed && !schematic.error && (
            <div className="muted" style={{ fontSize: 12 }}>
              No informed description for this page.
            </div>
          )}
          {schematic.blind && (
            <div style={{ marginTop: 10 }}>
              <button
                type="button"
                className="btn"
                onClick={onToggleBlind}
                aria-expanded={showBlind}
                data-testid="procedure-review-blind-toggle"
              >
                <Icon name="eye" size={12} /> Blind reading
              </button>
              {showBlind && (
                <div style={{ marginTop: 8 }}>
                  <PassBlock
                    pass={schematic.blind}
                    testId="procedure-review-blind"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <DivergencePanel schematic={schematic} />
    </>
  );
}

export function ProcedureReviewModal({
  bundleId,
  folderList = [],
  onClose,
  onToast,
}: Readonly<ProcedureReviewModalProps>) {
  const queryClient = useQueryClient();
  const modalRef = useRef<HTMLDialogElement>(null);
  useModalA11y({ open: true, onClose, ref: modalRef });

  const bundleQuery = useProcedureBundle(bundleId);
  const bundle = bundleQuery.data;

  const [schematicIndex, setSchematicIndex] = useState(0);
  const [showBlind, setShowBlind] = useState(false);
  const [comment, setComment] = useState('');
  const [folderChoice, setFolderChoice] = useState('');

  const invalidateAfterDecision = async (becameDocument: boolean) => {
    const keys: (readonly string[])[] = [
      ['procedures'],
      ['procedure', bundleId],
      ['activity'],
      ['notifications'],
    ];
    // An approved / rerouted bundle is enqueued as a document — the
    // Documents tab must refetch or the operator never sees it land.
    if (becameDocument) keys.push(['documents']);
    await Promise.all(
      keys.map((queryKey) =>
        queryClient.invalidateQueries({ queryKey: [...queryKey] }),
      ),
    );
  };

  const decisionError =
    (action: string, title: string) => (err: Error) => {
      logTechnicalError(`procedure-${action}`, err);
      onToast(
        'error',
        title,
        `${bundle?.file_name ?? bundleId} · ${userErrorMessage(err, {
          action: `${action} the procedure`,
        })}`,
      );
    };

  const approveMut = useMutation({
    mutationFn: () =>
      api.approveProcedure(bundleId, {
        folder: folderChoice || null,
      }),
    onSuccess: async (updated) => {
      onToast(
        'done',
        String(updated.state) === 'failed'
          ? 'Procedure refused by the classification gate'
          : 'Procedure approved',
        bundle?.file_name,
      );
      await invalidateAfterDecision(true);
      onClose();
    },
    onError: decisionError('approving', 'Approve failed'),
  });

  const rejectMut = useMutation({
    mutationFn: () =>
      api.rejectProcedure(bundleId, {
        comment: comment.trim() || null,
      }),
    onSuccess: async () => {
      onToast('done', 'Procedure rejected', bundle?.file_name);
      await invalidateAfterDecision(false);
      onClose();
    },
    onError: decisionError('rejecting', 'Reject failed'),
  });

  const rerouteMut = useMutation({
    mutationFn: () =>
      api.rerouteProcedureStandard(bundleId, {
        folder: folderChoice || null,
      }),
    onSuccess: async () => {
      onToast(
        'done',
        'Rerouted to the standard pipeline',
        bundle?.file_name,
      );
      await invalidateAfterDecision(true);
      onClose();
    },
    onError: decisionError('rerouting', 'Reroute failed'),
  });

  const retryMut = useMutation({
    mutationFn: () => api.retryProcedure(bundleId),
    onSuccess: async (updated) => {
      onToast(
        'done',
        'Procedure re-processed',
        `${bundle?.file_name ?? bundleId} · now ${updated.state}`,
      );
      await invalidateAfterDecision(false);
      onClose();
    },
    onError: decisionError('retrying', 'Retry failed'),
  });

  const busy =
    approveMut.isPending ||
    rejectMut.isPending ||
    rerouteMut.isPending ||
    retryMut.isPending;

  const state = String(bundle?.state ?? '');
  // Mirrors the backend `_resolve_primary_folder` 422: a bundle with no
  // requesting folder (scan-created, no operator duplicate request) needs
  // an explicit target before Approve / Treat as standard.
  const needsFolder =
    bundle !== undefined && bundleFolders(bundle as ProcedureBundle).length === 0;
  const folderMissing = needsFolder && !folderChoice;
  const canApprove = state === 'pending' && !folderMissing && !busy;
  const canReject = (state === 'pending' || state === 'failed') && !busy;
  const canReroute =
    ['pending', 'failed', 'rejected'].includes(state) && !folderMissing && !busy;
  const canRetry = (state === 'failed' || state === 'rejected') && !busy;

  const schematics = bundle?.schematics ?? [];
  const safeIndex = Math.min(schematicIndex, Math.max(schematics.length - 1, 0));
  const schematic = schematics[safeIndex];

  return (
    <div className="modal-backdrop">
      <button
        type="button"
        className="modal-backdrop-dismiss"
        onClick={onClose}
        aria-label="Close procedure review dialog"
        data-testid="procedure-review-backdrop"
      />
      <dialog
        open
        ref={modalRef}
        className="modal modal-lg"
        aria-modal="true"
        aria-label="Review procedure"
        data-testid="procedure-review-modal"
        tabIndex={-1}
      >
        <div className="modal-header">
          <div>
            <h2>Review procedure</h2>
            {bundle && (
              <div className="muted mono breakable" style={{ fontSize: 12 }}>
                {bundle.file_name} · {state}
              </div>
            )}
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close"
          >
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="modal-body">
          {bundleQuery.isLoading && (
            <p className="muted" data-testid="procedure-review-loading">
              Loading bundle…
            </p>
          )}
          {bundleQuery.isError && (
            <p
              className="muted"
              style={{ color: 'var(--twin-red-vivid)' }}
              data-testid="procedure-review-error"
            >
              {userErrorMessage(bundleQuery.error, {
                action: 'loading the procedure bundle',
              })}
            </p>
          )}
          {bundle && bundle.reason && (
            <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
              {bundle.reason}
            </p>
          )}
          {bundle && schematics.length === 0 && (
            <p className="muted">This bundle carries no schematic pages.</p>
          )}
          {bundle && schematic && (
            <SchematicPane
              schematic={schematic}
              index={safeIndex}
              total={schematics.length}
              onPrev={() => {
                setShowBlind(false);
                setSchematicIndex((i) => Math.max(i - 1, 0));
              }}
              onNext={() => {
                setShowBlind(false);
                setSchematicIndex((i) =>
                  Math.min(i + 1, schematics.length - 1),
                );
              }}
              showBlind={showBlind}
              onToggleBlind={() => setShowBlind((v) => !v)}
            />
          )}
          {bundle && needsFolder && (
            <label
              className="bulk-classification-row"
              style={{ marginTop: 12 }}
            >
              <span>
                Target folder{' '}
                <span className="muted">
                  — required (bundle has no requesting folder)
                </span>
              </span>
              <select
                className="bulk-classification-control"
                value={folderChoice}
                onChange={(e) => setFolderChoice(e.target.value)}
                aria-label="Target folder"
                data-testid="procedure-review-folder-select"
              >
                <option value="">Select a folder…</option>
                {folderList.map((folder) => (
                  <option key={folder.id} value={folder.id}>
                    {folder.kb}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <div className="modal-footer" style={{ flexWrap: 'wrap', gap: 8 }}>
          <input
            type="text"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Rejection comment (optional)"
            aria-label="Rejection comment (optional)"
            data-testid="procedure-review-reject-comment"
            style={{ flex: 1, minWidth: 160 }}
          />
          {canRetry && (
            <button
              type="button"
              className="btn"
              disabled={!canRetry}
              onClick={() => retryMut.mutate()}
              data-testid="procedure-review-retry"
            >
              {retryMut.isPending ? 'Retrying…' : 'Retry'}
            </button>
          )}
          <button
            type="button"
            className="btn"
            disabled={!canReroute}
            onClick={() => rerouteMut.mutate()}
            data-testid="procedure-review-reroute"
          >
            {rerouteMut.isPending ? 'Rerouting…' : 'Treat as standard'}
          </button>
          <button
            type="button"
            className="btn danger"
            disabled={!canReject}
            onClick={() => rejectMut.mutate()}
            data-testid="procedure-review-reject"
          >
            {rejectMut.isPending ? 'Rejecting…' : 'Reject'}
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={!canApprove}
            onClick={() => approveMut.mutate()}
            data-testid="procedure-review-approve"
          >
            {approveMut.isPending ? 'Approving…' : 'Approve'}
          </button>
        </div>
      </dialog>
    </div>
  );
}
