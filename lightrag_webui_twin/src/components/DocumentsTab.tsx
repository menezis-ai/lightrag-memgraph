/**
 * DocumentsTab — table of sources with status filter, search, tag filter,
 * multi-select, retag, individual + bulk delete (#149).
 *
 * Aligned on LightRAG DocStatus schema (sprint Étape 0). Field names mirror the
 * API JSON exactly (`doc_id`, `file_path`, `content_summary`, `chunks_count`,
 * `updated_at`) so there is no remapping at the boundary.
 *
 * Status enum is uppercase (LightRAG-native) but displayed lowercased for UI
 * continuity; filter pills/labels use lowercase keys mapped via STATUS_TO_FILTER.
 */

import { useMemo, useState } from 'react';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
import { ClassPill } from './ClassPill';
import { QuotaBanner } from './QuotaBanner';
import { useIngestionDisabled } from '../hooks/useIngestionDisabled';
import { useUrlArrayParam, useUrlParam } from '../hooks/useUrlParam';
import { relativeTime } from '../utils/relativeTime';
import { tagMatchesQuery, tagSuggestionComparator } from '../utils/tags';
import type { PipelineStatusResponse } from '../api/resources';
import type { Document, DocumentStatus } from '../types/document';
import type { ClassificationValue } from '../types/classification';
import type { TagEntry } from '../types/tag';

type StatusFilterKey = 'all' | 'completed' | 'processing' | 'pending' | 'failed';

const FILTER_TO_STATUS: Record<Exclude<StatusFilterKey, 'all'>, DocumentStatus> = {
  completed: 'PROCESSED',
  processing: 'PROCESSING',
  pending: 'PENDING',
  failed: 'FAILED',
};

const STATUS_TO_FILTER: Record<DocumentStatus, StatusFilterKey> = {
  PROCESSED: 'completed',
  PROCESSING: 'processing',
  PENDING: 'pending',
  FAILED: 'failed',
};

const STATUS_LABELS: Record<StatusFilterKey, string> = {
  all: 'All',
  completed: 'Completed',
  processing: 'Processing',
  pending: 'Pending',
  failed: 'Failed',
};

const STATUS_KEYS: readonly StatusFilterKey[] = [
  'all',
  'completed',
  'processing',
  'pending',
  'failed',
];

const STATUS_FILTERS = ['all', 'completed', 'processing', 'pending', 'failed'] as const;

export interface DocumentsTabProps {
  docs: readonly Document[];
  tagCatalog: readonly TagEntry[];
  onOpenAdd: () => void;
  onOpenRetag: (doc: Document) => void;
  onOpenBulkRetag: (docs: readonly Document[]) => void;
  onAddToast: (title: string, sub?: string) => void;
  onScanRetry?: (failedCount: number) => void;
  pipelineStatus?: PipelineStatusResponse | null;
  pipelineOpen?: boolean;
  pipelineLoading?: boolean;
  pipelineError?: string | null;
  onTogglePipeline?: () => void;
  onRefreshPipeline?: () => void;
  /** Open the document detail panel with chunks, lineage, audit and delete actions. */
  onOpenDetail?: (doc: Document) => void;
  /** @deprecated Use onOpenDetail. Kept for older callers/tests. */
  onDeleteDoc?: (doc: Document) => void;
  /** Bulk delete the selected documents (cascade), per #149. */
  onBulkDelete?: (docs: readonly Document[]) => void;
  /** Now in ms for deterministic relative-time rendering in tests. */
  nowMs?: number;
  /**
   * Optional slot rendered FULL-WIDTH between the header (h1 + actions) and
   * the filter pills. The host (App.tsx) injects PendingDocsSection here so
   * the "To be validated by your reviewer" panel sits inside the Documents
   * tab content rather than above it. Pattern mirrors the design prototype
   * (~/Downloads/prototype/src/app.jsx — `pendingSlot` prop).
   */
  pendingSlot?: React.ReactNode;
}

export function DocumentsTab({
  docs,
  tagCatalog,
  onOpenAdd,
  onOpenRetag,
  onOpenBulkRetag,
  onAddToast,
  onScanRetry,
  pipelineStatus,
  pipelineOpen = false,
  pipelineLoading = false,
  pipelineError = null,
  onTogglePipeline,
  onRefreshPipeline,
  onOpenDetail,
  onDeleteDoc,
  onBulkDelete,
  nowMs,
  pendingSlot,
}: DocumentsTabProps) {
  const openDetail = onOpenDetail ?? onDeleteDoc;
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [statusFilter, setStatusFilter] = useUrlParam<StatusFilterKey>(
    'status',
    'all',
    {
      validate: (v) => (STATUS_FILTERS as readonly string[]).includes(v),
    },
  );
  const [search, setSearch] = useUrlParam<string>('q', '');
  const [tagFilters, setTagFilters] = useUrlArrayParam('tag', []);
  // Set by citation / drill-down navigation (?source=<file_path>). The
  // param survives topbar tab switches, so it MUST stay visible and
  // removable — an invisible filter silently shrank the table to one doc.
  const [sourceFilters, setSourceFilters] = useUrlArrayParam('source', []);
  const [tagAddOpen, setTagAddOpen] = useState(false);
  const [tagAddVal, setTagAddVal] = useState('');
  const [bulkDeleteArmed, setBulkDeleteArmed] = useState(false);

  const searchAndTagFiltered = useMemo(() => {
    return docs.filter((d) => {
      if (
        search &&
        !d.file_path.toLowerCase().includes(search.toLowerCase())
      )
        return false;
      if (tagFilters.length && !tagFilters.every((t) => d.tags.includes(t)))
        return false;
      if (
        sourceFilters.length &&
        !sourceFilters.includes(d.file_path)
      )
        return false;
      return true;
    });
  }, [docs, search, tagFilters, sourceFilters]);

  const counts = useMemo(() => {
    const c: Record<StatusFilterKey, number> = {
      all: searchAndTagFiltered.length,
      completed: 0,
      processing: 0,
      pending: 0,
      failed: 0,
    };
    searchAndTagFiltered.forEach((d) => {
      c[STATUS_TO_FILTER[d.status]]++;
    });
    return c;
  }, [searchAndTagFiltered]);
  const failedCount = counts.failed;

  const filtered = useMemo(() => {
    return searchAndTagFiltered.filter((d) => {
      if (
        statusFilter !== 'all' &&
        d.status !== FILTER_TO_STATUS[statusFilter]
      )
        return false;
      return true;
    });
  }, [searchAndTagFiltered, statusFilter]);

  const removeTagFilter = (t: string) =>
    setTagFilters(tagFilters.filter((x) => x !== t));
  const addTagFilter = (t: string) => {
    if (t && !tagFilters.includes(t)) setTagFilters([...tagFilters, t]);
    setTagAddVal('');
    setTagAddOpen(false);
  };

  const tagSuggestions = useMemo(() => {
    const v = tagAddVal.toLowerCase();
    return tagCatalog
      .filter((t) => !tagFilters.includes(t.tag))
      .filter((t) => tagMatchesQuery(t, v))
      .sort(tagSuggestionComparator(v))
      .slice(0, 5);
  }, [tagAddVal, tagFilters, tagCatalog]);

  const clickTagOnRow = (e: React.MouseEvent, tag: string) => {
    e.stopPropagation();
    if (!tagFilters.includes(tag)) setTagFilters([...tagFilters, tag]);
  };

  const toggleRow = (id: string) => {
    setBulkDeleteArmed(false);
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };
  const filteredIds = filtered.map((d) => d.doc_id);
  const allFilteredSelected =
    filteredIds.length > 0 && filteredIds.every((id) => selected.has(id));
  const someFilteredSelected = filteredIds.some((id) => selected.has(id));
  const toggleAll = () => {
    setBulkDeleteArmed(false);
    const next = new Set(selected);
    if (allFilteredSelected) filteredIds.forEach((id) => next.delete(id));
    else filteredIds.forEach((id) => next.add(id));
    setSelected(next);
  };
  const clearSelection = () => {
    setBulkDeleteArmed(false);
    setSelected(new Set());
  };
  const selectedDocs = docs.filter((d) => selected.has(d.doc_id));
  const openBulk = () => onOpenBulkRetag(selectedDocs);
  const pipelineMessages = useMemo(() => {
    const messages = [...(pipelineStatus?.history_messages ?? [])];
    const latest = pipelineStatus?.latest_message;
    if (latest && messages[messages.length - 1] !== latest) {
      messages.push(latest);
    }
    return messages.slice(-80);
  }, [pipelineStatus]);
  const triggerBulkDelete = () => {
    if (!onBulkDelete || selectedDocs.length === 0) return;
    if (!bulkDeleteArmed) {
      setBulkDeleteArmed(true);
      onAddToast(
        'Confirm bulk delete',
        `${selectedDocs.length} source${selectedDocs.length === 1 ? '' : 's'} selected · click Delete again to cascade-remove chunks, entities and relations`,
      );
      return;
    }
    onBulkDelete(selectedDocs);
    clearSelection();
  };

  const ingestionDisabled = useIngestionDisabled();

  return (
    <div className="docs">
      <QuotaBanner tone="block" />
      <div className="docs-header">
        <h1>Document management</h1>
        <div className="docs-header-actions">
          <div className="pipeline-control">
            <button
              type="button"
              className={`btn${pipelineOpen ? ' active' : ''}`}
              title="Pipeline logs"
              aria-expanded={pipelineOpen}
              aria-haspopup="dialog"
              onClick={() => onTogglePipeline?.()}
            >
              <Icon name="activity" size={14} />
              Pipeline
              <span
                className={`pipeline-state-badge ${
                  pipelineStatus?.busy ? 'pipeline-state-busy' : 'pipeline-state-idle'
                }`}
                aria-label={`Pipeline ${pipelineStatus?.busy ? 'busy' : 'idle'}`}
              >
                {pipelineStatus?.busy ? 'BUSY' : 'IDLE'}
              </span>
            </button>
            {pipelineOpen && (
              <div
                className="pipeline-popover"
                role="dialog"
                aria-label="Pipeline logs"
              >
                <div className="pp-header">
                  <div className="pp-title">
                    <Icon name="activity" size={14} />
                    Pipeline
                    <span
                      className={`pp-state-badge ${
                        pipelineStatus?.busy ? 'busy' : 'paused'
                      }`}
                    >
                      <span className="pp-state-dot" />
                      {pipelineStatus?.busy ? 'Busy' : 'Idle'}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="btn small"
                    aria-label="Close pipeline logs"
                    onClick={() => onTogglePipeline?.()}
                  >
                    <Icon name="x" size={13} />
                  </button>
                </div>

                <div className="pp-section">
                  <div className="pp-stats">
                    <div>
                      <span className="pp-stat-num">
                        {pipelineStatus?.job_count ?? 0}
                      </span>
                      <span className="pp-stat-lbl">Jobs</span>
                    </div>
                    <div>
                      <span className="pp-stat-num">
                        {pipelineMessages.length}
                      </span>
                      <span className="pp-stat-lbl">Messages</span>
                    </div>
                    <div className="pp-job-name">
                      <span className="pp-stat-num">
                        {pipelineStatus?.job_name ?? '—'}
                      </span>
                      <span className="pp-stat-lbl">Current job</span>
                    </div>
                  </div>
                </div>

                <div className="pp-section">
                  <h3>Latest</h3>
                  {pipelineError ? (
                    <p className="pp-error">{pipelineError}</p>
                  ) : pipelineLoading && !pipelineStatus ? (
                    <p className="pp-empty">Loading pipeline status…</p>
                  ) : pipelineStatus?.latest_message ? (
                    <p className="pp-latest">{pipelineStatus.latest_message}</p>
                  ) : (
                    <p className="pp-empty">No pipeline message reported yet.</p>
                  )}
                </div>

                <div className="pp-section">
                  <h3>History</h3>
                  {pipelineMessages.length > 0 ? (
                    <ol className="pp-log-list">
                      {pipelineMessages.map((message, index) => (
                        <li key={`${index}-${message}`}>
                          <span className="pp-log-index">
                            {String(index + 1).padStart(2, '0')}
                          </span>
                          <span className="pp-log-message">{message}</span>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="pp-empty">No history from backend.</p>
                  )}
                </div>

                <div className="pp-footer">
                  <span className="pp-footnote">
                    <span
                      className={`pp-live-pill ${
                        pipelineLoading
                          ? 'pp-live-pill--loading'
                          : 'pp-live-pill--idle'
                      }`}
                      aria-label={`Pipeline live ${
                        pipelineLoading ? 'refreshing' : 'idle'
                      }`}
                    >
                      <span className="pp-live-dot" />
                      LIVE
                    </span>
                  </span>
                  <button
                    type="button"
                    className="btn small"
                    onClick={() => onRefreshPipeline?.()}
                  >
                    <Icon name="refresh" size={13} />
                    Refresh
                  </button>
                </div>
              </div>
            )}
          </div>
          {/* Audit C7: the only backend this action calls is
              ``POST /documents/reprocess_failed``. It stays enabled
              only when at least one source is failed and is labelled
              for that exact batch retry. */}
          <button
            type="button"
            className={`btn${failedCount > 0 ? ' btn-retry' : ''}`}
            disabled={failedCount === 0 || ingestionDisabled}
            title={
              ingestionDisabled
                ? 'Memgraph instance quota reached — free space before re-processing'
                : failedCount === 0
                  ? 'No failed sources to re-process'
                  : `Re-process ${failedCount} failed source${
                      failedCount > 1 ? 's' : ''
                    } (POST /documents/reprocess_failed)`
            }
            onClick={() => {
              if (failedCount === 0 || ingestionDisabled) return;
              if (onScanRetry) {
                onScanRetry(failedCount);
                return;
              }
              onAddToast(
                'Re-processing failed sources',
                'POST /documents/reprocess_failed',
              );
            }}
          >
            <Icon name="refresh" size={14} />
            Re-process failed sources
            {failedCount > 0 && (
              <span
                className="pipeline-badge"
                aria-label={`${failedCount} failed`}
              >
                {failedCount}
              </span>
            )}
          </button>
          <button
            type="button"
            className="btn primary"
            onClick={onOpenAdd}
            disabled={ingestionDisabled}
            title={
              ingestionDisabled
                ? 'Memgraph instance quota reached — free space before uploading'
                : undefined
            }
            data-testid="docs-add-source"
          >
            <Icon name="cloud-upload" size={14} /> Add source
          </button>
        </div>
      </div>

      {pendingSlot}

      <div className="docs-filters">
        <span className="filter-label">Uploaded</span>
        <div className="filter-pills">
          {STATUS_KEYS.map((k) => (
            <button
              key={k}
              type="button"
              className={`pill ${k}${statusFilter === k ? ' active' : ''}`}
              onClick={() => setStatusFilter(k)}
            >
              {STATUS_LABELS[k]} ({counts[k]})
            </button>
          ))}
        </div>
        <input
          className="search-source"
          placeholder="Search source name…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search source"
        />
      </div>

      {sourceFilters.length > 0 && (
        <div className="tag-filter-row" data-testid="source-filter-row">
          <span className="lbl">Filtered to source</span>
          <div className="tag-chips">
            {sourceFilters.map((s) => (
              <span key={s} className="tag-chip" data-testid={`source-filter-${s}`}>
                {s}
                <button
                  type="button"
                  aria-label={`Remove source filter ${s}`}
                  onClick={() =>
                    setSourceFilters(sourceFilters.filter((x) => x !== s))
                  }
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="tag-filter-row">
        <span className="lbl">
          Filter by tag<em>— Twin</em>
        </span>
        <div className="tag-chips">
          {tagFilters.map((t) => (
            <TagChip key={t} tag={t} removable onRemove={removeTagFilter} />
          ))}
          {tagAddOpen ? (
            <div className="autocomplete-anchor">
              <input
                autoFocus
                value={tagAddVal}
                onChange={(e) => setTagAddVal(e.target.value)}
                onBlur={() => setTimeout(() => setTagAddOpen(false), 150)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && tagSuggestions[0])
                    addTagFilter(tagSuggestions[0].tag);
                  if (e.key === 'Escape') setTagAddOpen(false);
                }}
                placeholder="tag…"
                aria-label="Add tag filter"
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  padding: '3px 8px',
                  border: '0.5px solid var(--twin-accent)',
                  borderRadius: '999px',
                  width: 110,
                  background: 'var(--color-background-primary)',
                }}
              />
              {tagSuggestions.length > 0 && (
                <div
                  className="autocomplete floating-autocomplete"
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    marginTop: 4,
                    minWidth: 200,
                    zIndex: 30,
                  }}
                >
                  {tagSuggestions.map((s, i) => (
                    <div
                      key={s.tag}
                      className={`autocomplete-row${i === 0 ? ' focus' : ''}`}
                      onMouseDown={() => addTagFilter(s.tag)}
                      data-testid={`docs-tag-sugg-${s.tag}`}
                    >
                      <div className="row1">
                        <span>{s.tag}</span>
                        <span className="badge">{s.category}</span>
                      </div>
                      <div className="def">{s.def}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <button
              type="button"
              className="tag-add-btn"
              onClick={() => setTagAddOpen(true)}
            >
              + Add tag
            </button>
          )}
        </div>
      </div>

      {selected.size > 0 && (
        <div className="bulk-bar" role="region" aria-label="Bulk actions">
          <span className="bulk-count">
            <b>{selected.size}</b> selected
            <span className="bulk-of">of {filtered.length}</span>
          </span>
          <button
            type="button"
            className="bulk-action primary"
            onClick={openBulk}
          >
            <Icon name="tags" size={13} /> Retag {selected.size} sources
          </button>
          {onBulkDelete && (
            <button
              type="button"
              className="bulk-action danger"
              onClick={triggerBulkDelete}
              data-testid="docs-bulk-delete"
              aria-label={`Delete ${selected.size} sources`}
            >
              <Icon name="x" size={13} />{' '}
              {bulkDeleteArmed ? `Confirm delete ${selected.size}` : `Delete ${selected.size}`}
            </button>
          )}
          <button
            type="button"
            className="bulk-clear"
            onClick={clearSelection}
          >
            <Icon name="x" size={12} /> Clear selection
          </button>
        </div>
      )}

      <div className="docs-table-wrap">
        <div className="docs-table">
          <div className="docs-row header has-select">
            <div className="cell-select">
              <input
                type="checkbox"
                className="row-check"
                checked={allFilteredSelected}
                ref={(el) => {
                  if (el)
                    el.indeterminate =
                      !allFilteredSelected && someFilteredSelected;
                }}
                onChange={toggleAll}
                aria-label="Select all visible"
              />
            </div>
            <div>Source</div>
            <div>Indexed preview</div>
            <div>
              Tags{' '}
              <span
                style={{
                  textTransform: 'none',
                  color: 'var(--color-text-tertiary)',
                  letterSpacing: 0,
                  fontSize: 9.5,
                }}
              >
                — Twin
              </span>
            </div>
            <div>Status</div>
            <div>Chunks</div>
            <div>Updated</div>
            <div />
          </div>
          {filtered.length === 0 && (
            <div
              style={{
                padding: 30,
                textAlign: 'center',
                color: 'var(--color-text-tertiary)',
                fontSize: 12,
              }}
              data-testid="docs-empty"
            >
              No documents match the current filters.
            </div>
          )}
          {filtered.map((d) => (
            <DocRow
              key={d.doc_id}
              doc={d}
              checked={selected.has(d.doc_id)}
              onToggle={toggleRow}
              onOpenRetag={onOpenRetag}
              onClickTag={clickTagOnRow}
              onOpenDetail={openDetail}
              nowMs={nowMs}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

interface DocRowProps {
  doc: Document;
  checked: boolean;
  onToggle: (id: string) => void;
  onOpenRetag: (doc: Document) => void;
  onClickTag: (e: React.MouseEvent, tag: string) => void;
  onOpenDetail?: (doc: Document) => void;
  nowMs?: number;
}

function DocRow({
  doc,
  checked,
  onToggle,
  onOpenRetag,
  onClickTag,
  onOpenDetail,
  nowMs,
}: DocRowProps) {
  const isFail = doc.status === 'FAILED';
  const isDeleting = doc._deleting === true;
  const isOptimisticUpload = doc._optimisticUpload === true;
  const canOpenDetail = Boolean(onOpenDetail && !isOptimisticUpload);
  const visibleTags = doc.tags.slice(0, 2);
  const overflow = doc.tags.length - visibleTags.length;
  const filterStatus = STATUS_TO_FILTER[doc.status];
  return (
    <div
      className={`docs-row has-select${checked ? ' is-checked' : ''}${isFail ? ' is-failed' : ''}`}
      data-testid={`docs-row-${doc.doc_id}`}
    >
      <div className="cell-select" onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          className="row-check"
          checked={checked}
          disabled={isOptimisticUpload}
          onChange={() => onToggle(doc.doc_id)}
          aria-label={
            isOptimisticUpload
              ? `${doc.file_path} is waiting for ingestion`
              : `Select ${doc.file_path}`
          }
        />
      </div>
      <div className="cell-source">
        <SourceIcon type={doc.type} size={14} />
        {canOpenDetail ? (
          <button
            type="button"
            className={`source-name source-name-button${doc.type !== 'file' ? ' mono' : ''}`}
            title={doc.file_path}
            onClick={(e) => {
              e.stopPropagation();
              onOpenDetail?.(doc);
            }}
            data-testid={`docs-row-delete-${doc.doc_id}`}
            aria-label={`Open details for ${doc.file_path}`}
          >
            <span data-testid={`docs-row-filename-${doc.doc_id}`}>
              {doc.file_path}
            </span>
          </button>
        ) : (
          <span
            className={`source-name${doc.type !== 'file' ? ' mono' : ''}`}
            title={doc.file_path}
          >
            {doc.file_path}
          </span>
        )}
        <ClassPill
          cls={doc.metadata?.classification as ClassificationValue}
          docId={doc.doc_id}
        />
      </div>
      <div className="cell-summary">
        {isFail && (
          <Icon
            name="alert-triangle"
            size={13}
            color="var(--twin-red-vivid)"
            // Inline-flex via aria-hidden + style so the icon sits flush with
            // the preview text baseline. No extra span — the .is-failed row
            // class already paints everything red.
          />
        )}
        <span
          className="summary-text"
          style={isFail ? { marginLeft: 6 } : undefined}
          title={doc.content_summary}
        >
          {doc.content_summary || 'No indexed preview available.'}
        </span>
        {isFail && doc.error_msg && (
          <div
            className="summary-error"
            data-testid={`docs-row-error-${doc.doc_id}`}
            style={{
              marginTop: 4,
              fontSize: 11,
              color: 'var(--twin-red-vivid)',
            }}
          >
            Indexing failed: {doc.error_msg}
          </div>
        )}
      </div>
      <div className="cell-tags">
        {visibleTags.map((t) => (
          <span
            key={t}
            onClick={(e) => onClickTag(e, t)}
            data-testid={`row-tag-${doc.doc_id}-${t}`}
          >
            <TagChip tag={t} />
          </span>
        ))}
        {overflow > 0 && <span className="tag-overflow">+{overflow}</span>}
      </div>
      <div className="cell-status">
        {isDeleting ? (
          <span className="status-text deleting" data-testid="status-deleting">
            deleting…
          </span>
        ) : (
          <span className={`status-text ${filterStatus}`}>{filterStatus}</span>
        )}
      </div>
      <div
        className="cell-chunks"
        title={
          isFail && (doc.chunks_count ?? 0) > 0
            ? `${doc.chunks_count} chunks created before failure`
            : undefined
        }
        data-testid={`docs-row-chunks-${doc.doc_id}`}
      >
        {doc.chunks_count ?? 0}
      </div>
      <div className="cell-updated">{relativeTime(doc.updated_at, nowMs)}</div>
      <div className="cell-actions">
        {!isOptimisticUpload && (
          <>
            <button
              type="button"
              className="row-action"
              onClick={() => onOpenRetag(doc)}
              aria-label={`Retag ${doc.file_path}`}
            >
              <Icon name="tags" size={13} />
            </button>
            {canOpenDetail && (
              <button
                type="button"
                className="row-action"
                onClick={() => onOpenDetail?.(doc)}
                aria-label={`Open details for ${doc.file_path}`}
                title="Open details"
              >
                <Icon name="file-text" size={13} />
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
