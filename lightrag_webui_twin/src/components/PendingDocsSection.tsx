/**
 * PendingDocsSection — top-of-DocumentsTab card list for docs in
 * `review.state === 'pending-review'`.
 *
 * Actions per card (#106 + #150 spec révisée):
 *   - Approve         : POST /twin/api/documents/{id}/approve (TanStack useMutation)
 *   - Edit & Approve  : opens EditApproveModal (régression v2 fix — was no-op);
 *                       submits the editable summary back to /approve as `edits`
 *   - Reject          : opens RejectModal asking for a mandatory reason
 *   - Simulate change : opens an in-place diff stub (preview impact)
 *
 * Every mutation follows the RC-1 pattern (TanStack useMutation, optimistic
 * UI is forbidden — toast + invalidate AFTER server ack).
 */

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/resources';
import { Icon, SourceIcon } from './Icon';
import { relativeTime } from '../utils/relativeTime';
import type { Document } from '../types/document';

export interface PendingDocsSectionProps {
  docs: readonly Document[];
  /** Push a UI toast — host owns the queue. */
  onToast: (
    kind: 'done' | 'error',
    title: string,
    sub?: string,
  ) => void;
  /** Current operator's identifier for audit attribution. */
  actor?: string;
  nowMs?: number;
}

export function PendingDocsSection({
  docs,
  onToast,
  actor,
  nowMs,
}: PendingDocsSectionProps) {
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState<Document | null>(null);
  const [rejecting, setRejecting] = useState<Document | null>(null);
  const [simulating, setSimulating] = useState<Document | null>(null);

  const approveMut = useMutation({
    mutationFn: ({
      doc,
      edits,
    }: {
      doc: Document;
      edits?: Partial<Document>;
    }) => api.approveDocument(doc.doc_id, { actor, edits }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['activity'] });
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      onToast(
        'done',
        `Document approved`,
        variables.doc.file_path,
      );
    },
    onError: (err: Error, variables) => {
      onToast(
        'error',
        `Approve failed`,
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
      onToast('done', `Document rejected`, variables.doc.file_path);
    },
    onError: (err: Error, variables) => {
      onToast(
        'error',
        `Reject failed`,
        `${variables.doc.file_path} · ${err.message}`,
      );
    },
  });

  if (docs.length === 0) return null;

  return (
    <section
      className="pending-docs"
      data-testid="pending-docs-section"
      aria-label="Pending documents"
    >
      <header className="pending-docs-header">
        <h2>
          Pending review{' '}
          <span className="muted">({docs.length})</span>
        </h2>
        <p className="muted">
          These sources are queued for Steward approval before they index into
          the workspace KB.
        </p>
      </header>
      <ul className="pending-docs-list">
        {docs.map((doc) => (
          <li
            key={doc.doc_id}
            className="pending-doc-card"
            data-testid={`pending-doc-${doc.doc_id}`}
          >
            <div className="pending-doc-head">
              <SourceIcon type={doc.type} size={14} />
              <strong className={doc.type !== 'file' ? 'mono' : ''}>
                {doc.file_path}
              </strong>
              <span className="muted">
                · {relativeTime(doc.created_at, nowMs)}
              </span>
            </div>
            <div className="pending-doc-summary">{doc.content_summary}</div>
            {doc.review?.justification && (
              <div className="pending-doc-just muted">
                Justification: {doc.review.justification}
              </div>
            )}
            <div className="pending-doc-actions">
              <button
                type="button"
                className="btn small primary"
                disabled={approveMut.isPending}
                onClick={() => approveMut.mutate({ doc })}
                data-testid={`pending-doc-approve-${doc.doc_id}`}
              >
                <Icon name="circle-check" size={12} />{' '}
                {approveMut.isPending ? 'Approving…' : 'Approve'}
              </button>
              <button
                type="button"
                className="btn small"
                onClick={() => setEditing(doc)}
                data-testid={`pending-doc-edit-approve-${doc.doc_id}`}
              >
                <Icon name="settings" size={12} /> Edit &amp; Approve
              </button>
              <button
                type="button"
                className="btn small"
                onClick={() => setSimulating(doc)}
                data-testid={`pending-doc-simulate-${doc.doc_id}`}
              >
                <Icon name="eye" size={12} /> Simulate change
              </button>
              <button
                type="button"
                className="btn small danger"
                onClick={() => setRejecting(doc)}
                data-testid={`pending-doc-reject-${doc.doc_id}`}
              >
                <Icon name="x" size={12} /> Reject
              </button>
            </div>
          </li>
        ))}
      </ul>

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
      {simulating && (
        <SimulateModal doc={simulating} onClose={() => setSimulating(null)} />
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

interface SimulateModalProps {
  doc: Document;
  onClose: () => void;
}

function SimulateModal({ doc, onClose }: SimulateModalProps) {
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid="pending-doc-simulate-modal"
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Simulate change"
        style={{ width: 460 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Simulate change</h2>
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
          <p>
            Approving this source will add{' '}
            <strong>{doc.chunks_count ?? '~'}</strong> chunks and{' '}
            <strong>{doc.tags.length}</strong> tag link(s) to workspace{' '}
            <code className="mono">{doc.workspace}</code>.
          </p>
          <p className="muted">
            Full dry-run diff (graph + retrieval impact) ships in backend
            phase 2.
          </p>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
