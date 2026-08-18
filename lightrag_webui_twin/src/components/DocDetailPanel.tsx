/**
 * DocDetailPanel — right-side document inspector with 3 tabs: Chunks, Lineage,
 * Audit. Opened from a row click in DocumentsTab; closes via X or backdrop.
 *
 * Footer actions (#105 + #149 spec révisée):
 *   - Retag: opens RetagModal via callback
 *   - View raw: surfaces a notice until the raw download endpoint is wired
 *   - Re-process: POST /documents/reprocess_failed (audit C7 —
 *     targeted ``/documents/{id}/scan`` is rejected because LightRAG
 *     has no safe per-document rescan; the host handler in App.tsx
 *     routes FAILED docs through the failed-batch endpoint instead,
 *     and toasts an explanatory cue on any other status).
 *   - Delete (cascade): DELETE /documents/{id}, both individual & multi
 */

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/resources';
import { ClassPill } from './ClassPill';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
import { relativeTime } from '../utils/relativeTime';
import { documentContentHash } from '../utils/documents';
import { ingestionFailureMessage } from '../lib/errorMessages';
import { policyRejectionKind, policyRejectionGuidance } from '../lib/policyRejection';
import type { ClassificationValue } from '../types/classification';
import type { Document } from '../types/document';

export type DetailTab = 'chunks' | 'lineage' | 'audit';

/** Half-open character range to highlight inside the drilled-down chunk. */
export interface ChunkAnchorRange {
  start: number;
  end: number;
}

export interface DocDetailPanelProps {
  doc: Document | null;
  onClose: () => void;
  onRetag?: (doc: Document) => void;
  onReprocess?: (doc: Document) => void;
  onDelete?: (doc: Document) => void;
  initialExpandedChunkId?: string | null;
  /** Paragraph anchor for `initialExpandedChunkId` (citation drill-down). */
  initialAnchor?: ChunkAnchorRange | null;
  initialTab?: DetailTab;
  nowMs?: number;
}

export function DocDetailPanel({
  doc,
  onClose,
  onRetag,
  onReprocess,
  onDelete,
  initialExpandedChunkId,
  initialAnchor,
  initialTab = 'chunks',
  nowMs,
}: Readonly<DocDetailPanelProps>) {
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
  const rejectionKind = policyRejectionKind(doc);
  const deleteArmed =
    deleteConfirm?.docId === doc.doc_id && deleteConfirm.armed;
  const tab = tabState.docId === doc.doc_id ? tabState.tab : initialTab;
  const setTab = (next: DetailTab) =>
    setTabState({ docId: doc.doc_id, tab: next });

  return (
    <aside
      className="doc-detail-panel"
      aria-label={`Detail: ${doc.file_path}`}
      data-testid="doc-detail-panel"
    >
      <header className="doc-detail-header">
        <div className="doc-detail-title">
          <SourceIcon type={doc.type} size={14} />
          <strong className={doc.type === 'file' ? '' : 'mono'}>
            {doc.file_path}
          </strong>
          {/* QA DOC-V4-001: the confidentiality level must be visible from
              the detail panel too, not only on the list rows. */}
          <ClassPill
            cls={doc.metadata?.classification as ClassificationValue}
            visibility={doc.visibility}
            docId={`detail-${doc.doc_id}`}
          />
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
        {tab === 'chunks' && (
          <ChunksTab
            docId={doc.doc_id}
            initialExpandedChunkId={initialExpandedChunkId}
            initialAnchor={initialAnchor}
          />
        )}
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
          disabled={rejectionKind !== null}
          title={
            rejectionKind !== null
              ? 'This document was rejected by policy — reprocessing cannot change the verdict.'
              : undefined
          }
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
        >
          <button
            type="button"
            className="modal-backdrop-dismiss"
            onClick={() => setRawNoticeOpen(false)}
            aria-label="Close raw notice"
            data-testid="doc-detail-raw-notice"
          />
          <dialog
            open
            className="modal modal-sm"
            aria-modal="true"
            aria-label="View raw notice"
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
              <p className="muted">
                Raw content is large and downloads as a stream. This is a
                notice gate — the actual download endpoint is wired in backend
                phase 2.
              </p>
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
          </dialog>
        </div>
      )}
    </aside>
  );
}

interface ChunksTabProps {
  docId: string;
  initialExpandedChunkId?: string | null;
  initialAnchor?: ChunkAnchorRange | null;
}

/**
 * Render a chunk's text with the anchored paragraph highlighted.
 * The anchor is a non-authoritative hint carried through URL params: any
 * range that does not fit the actual loaded text (stale offsets after a
 * re-index, hand-edited URL) degrades to the plain text, never to a
 * mis-sliced render.
 *
 * Offsets are Unicode CODE POINTS (Python string indices, the backend
 * contract in `TwinSourceAnchor`) — NOT UTF-16 units. `String.slice`
 * would drift by one for every astral character (emoji, some CJK)
 * before the paragraph, so both the bounds check and the slicing run
 * on `Array.from(text)` (PR #418 review, finding 2).
 */
function renderChunkText(text: string, anchor: ChunkAnchorRange | null) {
  if (!anchor) return text;
  const codePoints = Array.from(text);
  if (
    anchor.start < 0 ||
    anchor.end <= anchor.start ||
    anchor.end > codePoints.length
  ) {
    return text;
  }
  return (
    <>
      {codePoints.slice(0, anchor.start).join('')}
      <mark className="chunk-anchor-mark" data-testid="chunk-anchor-highlight">
        {codePoints.slice(anchor.start, anchor.end).join('')}
      </mark>
      {codePoints.slice(anchor.end).join('')}
    </>
  );
}

function ChunksTab({
  docId,
  initialExpandedChunkId,
  initialAnchor,
}: Readonly<ChunksTabProps>) {
  const [manualExpanded, setManualExpanded] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [manualCollapsed, setManualCollapsed] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const { data, isLoading } = useQuery({
    queryKey: ['doc-chunks', docId] as const,
    queryFn: () => api.listDocumentChunks(docId),
  });
  const expanded = useMemo(() => {
    const next = new Set(manualExpanded);
    if (initialExpandedChunkId && !manualCollapsed.has(initialExpandedChunkId)) {
      next.add(initialExpandedChunkId);
    }
    return next;
  }, [initialExpandedChunkId, manualCollapsed, manualExpanded]);

  useEffect(() => {
    if (!initialExpandedChunkId || !data?.length) return;
    globalThis.setTimeout(() => {
      const target = Array.from(
        document.querySelectorAll<HTMLElement>('[data-chunk-id]'),
      ).find((el) => el.dataset.chunkId === initialExpandedChunkId);
      target?.scrollIntoView?.({ block: 'center' });
    }, 0);
  }, [data, initialExpandedChunkId]);
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
        const isExpanded = expanded.has(c.chunk_id);
        const hasText = c.text.trim().length > 0;
        return (
          <li
            key={c.chunk_id}
            className="doc-chunk"
            data-chunk-id={c.chunk_id}
            data-testid={`doc-detail-chunk-${c.chunk_id}`}
          >
            <div className="doc-chunk-meta">
              <code className="mono">{c.chunk_id}</code>
              <span className="muted">#{c.order}</span>
            </div>
            <div
              className={`doc-chunk-text${isExpanded ? ' expanded' : ''}`}
              data-testid="doc-detail-chunk-text"
            >
              {hasText ? (
                renderChunkText(
                  c.text,
                  c.chunk_id === initialExpandedChunkId
                    ? (initialAnchor ?? null)
                    : null,
                )
              ) : (
                <span className="muted">No text stored for this chunk.</span>
              )}
            </div>
            {hasText && (
              <button
                type="button"
                className="doc-chunk-toggle"
                onClick={() => {
                  setManualExpanded((current) => {
                    const next = new Set(current);
                    if (isExpanded) next.delete(c.chunk_id);
                    else next.add(c.chunk_id);
                    return next;
                  });
                  setManualCollapsed((current) => {
                    const next = new Set(current);
                    if (isExpanded) next.add(c.chunk_id);
                    else next.delete(c.chunk_id);
                    return next;
                  });
                }}
                aria-expanded={isExpanded}
                aria-label={
                  isExpanded
                    ? `Collapse chunk ${c.chunk_id}`
                    : `Show full chunk ${c.chunk_id}`
                }
                data-testid={`doc-detail-chunk-toggle-${c.chunk_id}`}
              >
                {isExpanded ? 'Réduire' : 'Voir tout'}
              </button>
            )}
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

function LineageTab({ doc, nowMs }: Readonly<LineageTabProps>) {
  const hash = documentContentHash(doc);
  const rejectionKind = policyRejectionKind(doc);
  return (
    <div className="doc-lineage" data-testid="doc-detail-lineage">
      <dl className="settings-dl">
        <dt>Source path</dt>
        <dd className="mono">{doc.file_path}</dd>
        <dt>Folder</dt>
        <dd className="mono">{doc.folder}</dd>
        <dt>Uploader</dt>
        <dd>
          {typeof doc.metadata?.uploader === 'string'
            ? doc.metadata.uploader
            : 'unknown'}
        </dd>
        <dt>Created</dt>
        <dd>{relativeTime(doc.created_at, nowMs)}</dd>
        <dt>Updated</dt>
        <dd>{relativeTime(doc.updated_at, nowMs)}</dd>
        <dt>Track id</dt>
        <dd className="mono">{doc.track_id ?? 'n/a'}</dd>
        <dt>Hash</dt>
        <dd className="mono" data-testid="doc-detail-hash">
          {hash ? `${hash.label}: ${hash.value}` : 'n/a'}
        </dd>
        <dt>Status</dt>
        <dd>{doc.status}</dd>
        <dt>Chunks</dt>
        <dd>
          {doc.chunks_count ?? 0}
          {doc.status === 'FAILED' && (doc.chunks_count ?? 0) > 0 && (
            <span className="muted" style={{ marginLeft: 6 }}>
              (created before failure)
            </span>
          )}
        </dd>
        {doc.status === 'FAILED' && doc.error_msg && (
          <>
            <dt>Error</dt>
            <dd
              data-testid="doc-detail-error-msg"
              style={{ color: 'var(--twin-red-vivid)' }}
            >
              {rejectionKind !== null
                ? doc.error_msg
                : `Indexing failed: ${ingestionFailureMessage(doc.error_msg)}`}
            </dd>
          </>
        )}
        {rejectionKind !== null && (
          <>
            <dt>Verdict</dt>
            <dd data-testid="doc-detail-policy-rejection">
              {policyRejectionGuidance(rejectionKind)}
            </dd>
          </>
        )}
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

function AuditTab({ docId, nowMs }: Readonly<AuditTabProps>) {
  const { data, isLoading } = useQuery({
    queryKey: ['doc-audit', docId] as const,
    queryFn: () => api.listActivity({ resourceId: docId, limit: 200 }),
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
