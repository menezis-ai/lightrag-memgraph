/**
 * ReadSourceModal — full-screen preview of the indexed chunks LightRAG
 * ingested. Opened from PendingDocsSection's "Read source" button.
 *
 * Audit C6 / 2026-06-13: the modal used to render ``doc.extracted_text``
 * with a placeholder fallback. The backend never populated that field
 * in production, so the placeholder was the operator's actual
 * experience on most documents. The audit's first recommendation is
 * to surface the real ``/documents/{id}/chunks`` projection, which
 * shows what LightRAG actually keeps and uses for retrieval.
 *
 * Doctrine for the rewrite:
 * - Chunks are NOT joined into a single text block. They are
 *   rendered as separate inspectable units with chunk_id + #order
 *   visible. We don't recreate a fake impression of continuous
 *   extracted text — the indexed content is chunked, the view says
 *   so.
 * - ``chunk.redacted === true`` is surfaced as a discrete
 *   indicator; we never pretend the chunk is complete when the
 *   backend marked it as truncated.
 * - No silent fallback to ``doc.extracted_text``. The field stays
 *   on the TS Document type (for back-compat with fixtures) but
 *   is no longer rendered — chunks are the source of truth.
 */

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Icon } from './Icon';
import { api } from '../api/resources';
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

  const docId = doc?.doc_id ?? '';
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['doc-chunks', docId] as const,
    queryFn: () => api.listDocumentChunks(docId),
    enabled: Boolean(docId),
  });

  if (!doc) return null;

  const kb = (doc.content_length / 1024).toFixed(1);
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
        aria-label="Indexed chunks"
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
            <h2>Indexed chunks — what the indexer ingested</h2>
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
          {isLoading && (
            <div className="muted" data-testid="rs-chunks-loading">
              Loading indexed chunks…
            </div>
          )}
          {isError && (
            <div
              className="error-banner"
              role="alert"
              data-testid="rs-chunks-error"
            >
              Could not load chunks
              {error instanceof Error ? ` — ${error.message}` : ''}.
            </div>
          )}
          {!isLoading && !isError && (!data || data.length === 0) && (
            <div className="muted" data-testid="rs-chunks-empty">
              No chunks indexed for this source yet.
            </div>
          )}
          {!isLoading && !isError && data && data.length > 0 && (
            <ul className="rs-chunks" data-testid="rs-chunks-list">
              {data.map((c) => (
                <li
                  key={c.chunk_id}
                  className="rs-chunk"
                  data-testid={`rs-chunk-${c.chunk_id}`}
                >
                  <div className="rs-chunk-meta">
                    <code className="mono">{c.chunk_id}</code>
                    <span className="muted">#{c.order}</span>
                    {c.redacted === true && (
                      <span
                        className="muted"
                        data-testid={`rs-chunk-redacted-${c.chunk_id}`}
                      >
                        · redacted (truncated)
                      </span>
                    )}
                  </div>
                  <pre className="rs-chunk-text">{c.text}</pre>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="modal-footer rs-footer">
          <Icon
            name="info-circle"
            size={13}
            color="var(--color-text-tertiary)"
          />
          <span>
            Post-extraction chunks used for retrieval, not the original
            binary.
          </span>
        </div>
      </div>
    </div>
  );
}
