/**
 * DocDetailPanel — right-side document inspector with 3 tabs: Chunks, Lineage,
 * Audit. Opened from a row click in DocumentsTab; closes via X or backdrop.
 *
 * Compliance gating (Louis HORVAT 2026-05-28):
 *   - Chunks content is truncated when the doc classification > internal.
 *     The metadata.classification field carries that hint.
 *   - "View raw" surfaces a prompt reminder, never actually exposes content
 *     above the operator's clearance.
 *
 * Footer actions (#105 + #149 spec révisée):
 *   - Retag: opens RetagModal via callback
 *   - View raw: gated by classification, surfaces a notice if above palier
 *   - Re-process: POST /documents/{id}/scan
 *   - Delete (cascade): DELETE /documents/{id}, both individual & multi
 */

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/resources';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
import { ClassPill } from './ClassPill';
import { relativeTime } from '../utils/relativeTime';
import type { Document } from '../types/document';
import {
  getClassName,
  isAboveInternal,
  type ClassificationValue,
} from '../types/classification';

export type DetailTab = 'chunks' | 'lineage' | 'audit';

export interface DocDetailPanelProps {
  doc: Document | null;
  onClose: () => void;
  onRetag?: (doc: Document) => void;
  onReprocess?: (doc: Document) => void;
  onDelete?: (doc: Document) => void;
  nowMs?: number;
}

/**
 * Get a single human-readable classification token from either shape
 * (legacy string maquette baseline OR structured ClassificationResult from
 * the PR #157 MIP extractor). Used in the header text + raw-notice modal.
 */
function classificationOf(doc: Document): string {
  const cls = doc.metadata?.classification as ClassificationValue;
  return getClassName(cls);
}

/**
 * "Is this above what an internal-cleared operator can read?" Drives the
 * chunks-tab truncation + the raw-bytes notice gate (doctrine Eric 28/05).
 * Handles both legacy `string` and structured `ClassificationResult` shapes
 * via `isAboveInternal()` in `types/classification.ts`.
 */
function isClassificationAboveInternal(doc: Document): boolean {
  return isAboveInternal(doc.metadata?.classification as ClassificationValue);
}

export function DocDetailPanel({
  doc,
  onClose,
  onRetag,
  onReprocess,
  onDelete,
  nowMs,
}: DocDetailPanelProps) {
  const [tabState, setTabState] = useState<{
    docId: string;
    tab: DetailTab;
  }>({ docId: '', tab: 'chunks' });
  const [rawNoticeOpen, setRawNoticeOpen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<{
    docId: string;
    armed: boolean;
  } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && doc) onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [doc, onClose]);

  if (!doc) return null;
  const deleteArmed =
    deleteConfirm?.docId === doc.doc_id && deleteConfirm.armed;
  const tab = tabState.docId === doc.doc_id ? tabState.tab : 'chunks';
  const setTab = (next: DetailTab) =>
    setTabState({ docId: doc.doc_id, tab: next });

  return (
    <aside
      className="doc-detail-panel"
      role="dialog"
      aria-modal="false"
      aria-label={`Detail: ${doc.file_path}`}
      data-testid="doc-detail-panel"
    >
      <header className="doc-detail-header">
        <div className="doc-detail-title">
          <SourceIcon type={doc.type} size={14} />
          <strong className={doc.type !== 'file' ? 'mono' : ''}>
            {doc.file_path}
          </strong>
          <ClassPill
            cls={doc.metadata?.classification as ClassificationValue}
            docId={doc.doc_id}
          />
          <span className="muted">· {classificationOf(doc)}</span>
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={onClose}
          aria-label="Close"
        >
          <Icon name="x" size={16} />
        </button>
      </header>

      <nav className="doc-detail-tabs">
        {(['chunks', 'lineage', 'audit'] as DetailTab[]).map((t) => (
          <button
            key={t}
            type="button"
            className={`tab small${tab === t ? ' active' : ''}`}
            data-testid={`doc-detail-tab-${t}`}
            onClick={() => setTab(t)}
            aria-current={tab === t}
          >
            {t}
          </button>
        ))}
      </nav>

      <section className="doc-detail-body">
        {tab === 'chunks' && <ChunksTab docId={doc.doc_id} doc={doc} />}
        {tab === 'lineage' && <LineageTab doc={doc} nowMs={nowMs} />}
        {tab === 'audit' && <AuditTab docId={doc.doc_id} nowMs={nowMs} />}
      </section>

      <footer className="doc-detail-footer">
        <button
          type="button"
          className="btn small"
          onClick={() => onRetag?.(doc)}
          data-testid="doc-detail-retag"
        >
          <Icon name="tags" size={12} /> Retag
        </button>
        <button
          type="button"
          className="btn small"
          onClick={() => setRawNoticeOpen(true)}
          data-testid="doc-detail-view-raw"
        >
          <Icon name="eye" size={12} /> View raw
        </button>
        <button
          type="button"
          className="btn small"
          onClick={() => onReprocess?.(doc)}
          data-testid="doc-detail-reprocess"
        >
          <Icon name="refresh" size={12} /> Re-process
        </button>
        <button
          type="button"
          className="btn small danger"
          onClick={() => {
            if (!deleteArmed) {
              setDeleteConfirm({ docId: doc.doc_id, armed: true });
              return;
            }
            onDelete?.(doc);
          }}
          data-testid="doc-detail-delete"
          aria-label={
            deleteArmed
              ? `Confirm delete ${doc.file_path}`
              : `Delete ${doc.file_path}`
          }
        >
          <Icon name="x" size={12} /> {deleteArmed ? 'Confirm delete' : 'Delete'}
        </button>
      </footer>

      {rawNoticeOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setRawNoticeOpen(false)}
          data-testid="doc-detail-raw-notice"
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="View raw notice"
            style={{ width: 420 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h2>View raw</h2>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setRawNoticeOpen(false)}
                aria-label="Close"
              >
                <Icon name="x" size={16} />
              </button>
            </div>
            <div className="modal-body">
              <p>
                Source classification: <code>{classificationOf(doc)}</code>.
              </p>
              {isClassificationAboveInternal(doc) ? (
                <p className="muted">
                  Raw bytes are not exposed in the Twin UI above{' '}
                  <code>internal</code>. To view this source, request access
                  through your Steward.
                </p>
              ) : (
                <p className="muted">
                  Raw content is large and downloads as a stream. This is a
                  notice gate — the actual download endpoint is wired in
                  backend phase 2.
                </p>
              )}
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn"
                onClick={() => setRawNoticeOpen(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

interface ChunksTabProps {
  docId: string;
  doc: Document;
}

function ChunksTab({ docId, doc }: ChunksTabProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['doc-chunks', docId] as const,
    queryFn: () => api.listDocumentChunks(docId),
  });
  const above = isClassificationAboveInternal(doc);
  if (isLoading) {
    return (
      <div className="muted" data-testid="doc-detail-chunks-loading">
        Loading chunks…
      </div>
    );
  }
  if (!data || data.length === 0) {
    return (
      <div className="muted" data-testid="doc-detail-chunks-empty">
        No chunks indexed yet.
      </div>
    );
  }
  return (
    <ul className="doc-chunks" data-testid="doc-detail-chunks-list">
      {data.map((c) => {
        const shown = above ? c.text.slice(0, Math.floor(c.text.length * 0.2)) : c.text;
        return (
          <li key={c.chunk_id} className="doc-chunk">
            <div className="doc-chunk-meta">
              <code className="mono">{c.chunk_id}</code>
              <span className="muted">#{c.order}</span>
            </div>
            <div className="doc-chunk-text">
              {shown}
              {above && (
                <span
                  className="muted"
                  data-testid="doc-detail-chunks-redacted"
                >
                  {' '}
                  · (truncated, classification &gt; internal)
                </span>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

interface LineageTabProps {
  doc: Document;
  nowMs?: number;
}

function LineageTab({ doc, nowMs }: LineageTabProps) {
  return (
    <div className="doc-lineage" data-testid="doc-detail-lineage">
      <dl className="settings-dl">
        <dt>Source path</dt>
        <dd className="mono">{doc.file_path}</dd>
        <dt>Space</dt>
        <dd className="mono">{doc.workspace}</dd>
        <dt>Uploader</dt>
        <dd>{String(doc.metadata?.uploader ?? 'unknown')}</dd>
        <dt>Created</dt>
        <dd>{relativeTime(doc.created_at, nowMs)}</dd>
        <dt>Updated</dt>
        <dd>{relativeTime(doc.updated_at, nowMs)}</dd>
        <dt>Track id</dt>
        <dd className="mono">{doc.track_id ?? 'n/a'}</dd>
        <dt>Status</dt>
        <dd>{doc.status}</dd>
        <dt>Chunks</dt>
        <dd>{doc.chunks_count ?? 0}</dd>
        <dt>Tags</dt>
        <dd>
          {doc.tags.length === 0 ? (
            <span className="muted">none</span>
          ) : (
            <div className="tag-chips">
              {doc.tags.map((t) => (
                <TagChip key={t} tag={t} />
              ))}
            </div>
          )}
        </dd>
      </dl>
    </div>
  );
}

interface AuditTabProps {
  docId: string;
  nowMs?: number;
}

function AuditTab({ docId, nowMs }: AuditTabProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['doc-audit', docId] as const,
    queryFn: () => api.listActivity({ resourceId: docId }),
  });
  if (isLoading) {
    return (
      <div className="muted" data-testid="doc-detail-audit-loading">
        Loading audit…
      </div>
    );
  }
  const events = data?.items ?? [];
  if (events.length === 0) {
    return (
      <div className="muted" data-testid="doc-detail-audit-empty">
        No audit events for this document.
      </div>
    );
  }
  return (
    <ul className="doc-audit" data-testid="doc-detail-audit-list">
      {events.map((e) => (
        <li key={e.id} className="doc-audit-row">
          <div className="doc-audit-line1">
            <strong>{e.summary}</strong>
            <span className={`sev sev-${e.sev}`}>{e.sev}</span>
          </div>
          <div className="muted">
            {e.actor.user} · {relativeTime(e.ts, nowMs ?? data?.nowMs)}
          </div>
        </li>
      ))}
    </ul>
  );
}
