/**
 * PendingDocsSection — top-of-DocumentsTab card list for docs the operator
 * must validate. Two card variants are rendered side-by-side:
 *
 *   - `requested`  : new source awaiting first sign-off (`review.state === 'pending-review'`)
 *   - `modified`   : Confluence/SharePoint source edited upstream after
 *                    first approval; needs re-validation (`review.state === 'modified'`).
 *                    The diff summary comes from `review.update.summary_diff`.
 *
 * Visual structure mirrors the design-prototype handoff (Bucket B1):
 *   section.pending-section[.is-open]
 *     button.pending-h           (collapsible: alert + .pending-title + .pending-counts)
 *     div.pending-grid           (auto-fill 2-col, equal-height cards)
 *       div.pending-card[.requested | .modified]
 *         div.pending-card-h     (SourceIcon + .pending-card-title + [.pending-pill.amber])
 *         div.pending-body       (summary OR diff)
 *         div.pending-meta       (Submitted by | Modification detected · date · chunks · tags)
 *         div.pending-actions.grid2
 *           button.pbtn.ghost    (Read source / Edit & approve)
 *           button.pbtn.approve  (Approve / Approve update — soft-filled)
 *           button.pbtn.danger   (Reject / Reject update — red outline)
 *
 * RC-1 invariant: TanStack useMutation, onSuccess → invalidateQueries → toast.
 * NO optimistic UI — `isPending` from the mutation drives a per-doc busy
 * state so the operator sees "Approving…" instead of a half-faked transition.
 */

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/resources';
import { Icon, SourceIcon } from './Icon';
import type { Document } from '../types/document';

export interface PendingDocsSectionProps {
  docs: readonly Document[];
  /** Open the Read-source modal at the App level (host owns modal state). */
  onReadSource?: (doc: Document) => void;
  /** Push a UI toast — host owns the queue. */
  onToast: (kind: 'done' | 'error', title: string, sub?: string) => void;
  /** Current operator's identifier for audit attribution. */
  actor?: string;
}

function fmtDate(iso: string | undefined): string {
  return iso ? String(iso).slice(0, 10) : '';
}

export function PendingDocsSection({
  docs,
  onReadSource,
  onToast,
  actor,
}: PendingDocsSectionProps) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState<Document | null>(null);
  const [rejecting, setRejecting] = useState<Document | null>(null);
  const [open, setOpen] = useState(true);

  const approveMut = useMutation({
    mutationFn: ({
      doc,
      edits,
    }: {
      doc: Document;
      edits?: Partial<Document>;
      update?: boolean;
    }) => api.approveDocument(doc.doc_id, { actor, edits }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['activity'] });
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      onToast(
        'done',
        variables.update ? 'Update approved' : 'Document approved',
        variables.doc.file_path,
      );
    },
    onError: (err: Error, variables) => {
      onToast(
        'error',
        'Approve failed',
        `${variables.doc.file_path} · ${err.message}`,
      );
    },
  });

  const rejectMut = useMutation({
    mutationFn: ({ doc, reason }: { doc: Document; reason: string }) =>
      api.rejectDocument(doc.doc_id, { reason, actor }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['activity'] });
      onToast('done', 'Document rejected', variables.doc.file_path);
    },
    onError: (err: Error, variables) => {
      onToast(
        'error',
        'Reject failed',
        `${variables.doc.file_path} · ${err.message}`,
      );
    },
  });

  if (docs.length === 0) return null;

  const isBusy = (docId: string) =>
    approveMut.isPending && approveMut.variables?.doc.doc_id === docId;

  return (
    <section
      className={`pending-section${open ? ' is-open' : ''}`}
      data-testid="pending-docs-section"
      aria-label="Documents to validate"
    >
      <button
        type="button"
        className="pending-h"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <Icon name="alert-triangle" size={14} color="var(--twin-amber-vivid)" />
        <span className="pending-title">To be validated by your reviewer</span>
        <span className="pending-counts">
          <b>{docs.length}</b> documents awaiting your sign-off
        </span>
        <span
          style={{
            marginLeft: 'auto',
            display: 'inline-flex',
            transform: open ? 'none' : 'rotate(-90deg)',
            transition: 'transform .15s',
          }}
        >
          <Icon
            name="chevron-down"
            size={14}
            color="var(--color-text-tertiary)"
          />
        </span>
      </button>
      {open && (
        <div className="pending-grid">
          {docs.map((doc) => {
            const modified = doc.review?.state === 'modified';
            const upd = modified ? doc.review?.update : undefined;
            const isMono = doc.type !== 'file';
            return (
              <div
                key={doc.doc_id}
                className={`pending-card ${modified ? 'modified' : 'requested'}`}
                data-testid={`pending-doc-${doc.doc_id}`}
              >
                <div className="pending-card-h">
                  <SourceIcon type={doc.type} size={14} />
                  <span
                    className={`pending-card-title${isMono ? ' mono' : ''}`}
                  >
                    {doc.file_path}
                  </span>
                  {modified && (
                    <span
                      className="pending-pill amber"
                      style={{ marginLeft: 'auto' }}
                    >
                      <Icon name="alert-triangle" size={11} /> Modified source
                    </span>
                  )}
                </div>
                <div className="pending-body">
                  {modified
                    ? (upd?.summary_diff ?? doc.content_summary)
                    : doc.content_summary}
                </div>
                <div className="pending-meta">
                  {modified ? (
                    <>
                      Modification detected · {fmtDate(upd?.detected_at)} ·{' '}
                      {doc.chunks_count ?? 0} chunks indexed · tags{' '}
                      {doc.tags.join(', ')}
                    </>
                  ) : (
                    <>
                      Submitted by <b>{doc.review?.requested_by}</b> ·{' '}
                      {fmtDate(doc.review?.requested_at)} ·{' '}
                      {doc.chunks_count ?? 0} chunks · tags{' '}
                      {doc.tags.join(', ')}
                    </>
                  )}
                </div>
                <div className="pending-actions grid2">
                  <button
                    type="button"
                    className="pbtn ghost"
                    onClick={() => onReadSource?.(doc)}
                    data-testid={`pending-doc-read-${doc.doc_id}`}
                  >
                    <Icon name="eye" size={13} /> Read source
                  </button>
                  {modified ? (
                    <>
                      <button
                        type="button"
                        className="pbtn approve"
                        disabled={isBusy(doc.doc_id)}
                        onClick={() =>
                          approveMut.mutate({ doc, update: true })
                        }
                        data-testid={`pending-doc-approve-update-${doc.doc_id}`}
                      >
                        {isBusy(doc.doc_id) ? 'Approving…' : 'Approve update'}
                      </button>
                      <button
                        type="button"
                        className="pbtn danger span2"
                        onClick={() => setRejecting(doc)}
                        data-testid={`pending-doc-reject-update-${doc.doc_id}`}
                      >
                        Reject update
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="pbtn ghost"
                        onClick={() => setEditing(doc)}
                        data-testid={`pending-doc-edit-approve-${doc.doc_id}`}
                      >
                        Edit &amp; approve
                      </button>
                      <button
                        type="button"
                        className="pbtn approve"
                        disabled={isBusy(doc.doc_id)}
                        onClick={() => approveMut.mutate({ doc })}
                        data-testid={`pending-doc-approve-${doc.doc_id}`}
                      >
                        {isBusy(doc.doc_id) ? 'Approving…' : 'Approve'}
                      </button>
                      <button
                        type="button"
                        className="pbtn danger"
                        onClick={() => setRejecting(doc)}
                        data-testid={`pending-doc-reject-${doc.doc_id}`}
                      >
                        Reject
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {editing && (
        <EditApproveModal
          doc={editing}
          onClose={() => setEditing(null)}
          onSubmit={(edits) => {
            approveMut.mutate({ doc: editing, edits });
            setEditing(null);
          }}
        />
      )}
      {rejecting && (
        <RejectModal
          doc={rejecting}
          onClose={() => setRejecting(null)}
          onSubmit={(reason) => {
            rejectMut.mutate({ doc: rejecting, reason });
            setRejecting(null);
          }}
        />
      )}
    </section>
  );
}

interface EditApproveModalProps {
  doc: Document;
  onClose: () => void;
  onSubmit: (edits: Partial<Document>) => void;
}

function EditApproveModal({ doc, onClose, onSubmit }: EditApproveModalProps) {
  const [summary, setSummary] = useState(doc.content_summary);
  const [tags, setTags] = useState(doc.tags.join(', '));
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid="pending-doc-edit-modal"
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Edit & Approve"
        style={{ width: 520 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Edit &amp; Approve</h2>
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
          <label>
            Summary
            <textarea
              value={summary}
              onChange={(e) => setSummary(e.target.value)}
              data-testid="pending-doc-edit-summary"
              rows={3}
            />
          </label>
          <label>
            Tags (comma-separated)
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              data-testid="pending-doc-edit-tags"
            />
          </label>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn primary"
            data-testid="pending-doc-edit-submit"
            onClick={() =>
              onSubmit({
                content_summary: summary,
                tags: tags
                  .split(',')
                  .map((t) => t.trim())
                  .filter(Boolean),
              })
            }
          >
            Approve with edits
          </button>
        </div>
      </div>
    </div>
  );
}

interface RejectModalProps {
  doc: Document;
  onClose: () => void;
  onSubmit: (reason: string) => void;
}

function RejectModal({ doc, onClose, onSubmit }: RejectModalProps) {
  const [reason, setReason] = useState('');
  const canSubmit = reason.trim().length >= 6;
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid="pending-doc-reject-modal"
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Reject document"
        style={{ width: 480 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Reject {doc.file_path}</h2>
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
          <p className="muted">
            A short reason is required (≥ 6 chars). It is logged in the audit
            trail and surfaced to the uploader.
          </p>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            data-testid="pending-doc-reject-reason"
            rows={4}
            aria-label="Rejection reason"
          />
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn danger"
            disabled={!canSubmit}
            data-testid="pending-doc-reject-submit"
            onClick={() => onSubmit(reason.trim())}
          >
            Reject
          </button>
        </div>
      </div>
    </div>
  );
}
