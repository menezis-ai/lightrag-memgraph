/**
 * ReadSourceModal — full-screen preview of the post-extraction text the
 * indexer ingested. Opened from PendingDocsSection's "Read source" button
 * (per design prototype Bucket B2).
 *
 * Distinct from `DocDetailPanel.viewRaw` notice: this modal shows the
 * **extracted text** (what LightRAG actually chunks + embeds), not the raw
 * binary. A reviewer needs this to audit "what did the indexer see?" before
 * approving — the binary is irrelevant once parsed.
 *
 * Visual: full-width pre block, monospaced, with a small header carrying
 *   - icon + title "Extracted text — what the indexer ingested"
 *   - sub-line: file_path · "N chunks indexed" · "X KB extracted" + status pill
 *   - footer notice: "Post-extraction text used for retrieval, not the
 *     original binary."
 *
 * Status pill (amber, top-right of sub-line):
 *   - "awaiting reviewer sign-off"  when review.state === 'pending-review'
 *   - "modified — awaiting re-validation"  when review.state === 'modified'
 */

import { useEffect } from 'react';
import { Icon } from './Icon';
import type { Document } from '../types/document';

export interface ReadSourceModalProps {
  doc: Document | null;
  onClose: () => void;
}

export function ReadSourceModal({ doc, onClose }: ReadSourceModalProps) {
  useEffect(() => {
    if (!doc) return undefined;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [doc, onClose]);

  if (!doc) return null;

  const kb = (doc.content_length / 1024).toFixed(1);
  const text =
    doc.extracted_text ??
    `=== ${doc.file_path} ===\n\n(Extracted text preview is not available for this source in the demo build.)`;
  const modified = doc.review?.state === 'modified';
  const pending = doc.review?.state === 'pending-review';

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid="read-source-modal"
    >
      <div
        className="modal read-source"
        role="dialog"
        aria-modal="true"
        aria-label="Extracted text"
        style={{ width: 760, maxWidth: '94vw' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header rs-header">
          <div className="rs-title">
            <Icon
              name="eye"
              size={16}
              color="var(--color-text-secondary)"
            />
            <h2>Extracted text — what the indexer ingested</h2>
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
        <div className="rs-sub">
          <code className={doc.type !== 'file' ? 'mono' : ''}>
            {doc.file_path}
          </code>
          <span className="dot-sep">·</span>
          <span>{doc.chunks_count ?? 0} chunks indexed</span>
          <span className="dot-sep">·</span>
          <span>{kb} KB extracted</span>
          {pending && (
            <span
              className="pending-pill amber"
              style={{ marginLeft: 'auto' }}
              data-testid="rs-pill-pending"
            >
              awaiting reviewer sign-off
            </span>
          )}
          {modified && (
            <span
              className="pending-pill amber"
              style={{ marginLeft: 'auto' }}
              data-testid="rs-pill-modified"
            >
              modified — awaiting re-validation
            </span>
          )}
        </div>
        <div className="modal-body rs-body">
          <pre className="rs-text">{text}</pre>
        </div>
        <div className="modal-footer rs-footer">
          <Icon
            name="info-circle"
            size={13}
            color="var(--color-text-tertiary)"
          />
          <span>
            Post-extraction text used for retrieval, not the original binary.
          </span>
        </div>
      </div>
    </div>
  );
}
