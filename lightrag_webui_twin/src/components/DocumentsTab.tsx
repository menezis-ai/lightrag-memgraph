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

function documentMatchesSearchAndTags(
  doc: Document,
  search: string,
  tagFilters: readonly string[],
  sourceFilters: readonly string[],
): boolean {
  if (search && !doc.file_path.toLowerCase().includes(search.toLowerCase())) {
    return false;
  }
  if (tagFilters.length && !tagFilters.every((tag) => doc.tags.includes(tag))) {
    return false;
  }
  return sourceFilters.length === 0 || sourceFilters.includes(doc.file_path);
}

function statusCountsFor(
  docs: readonly Document[],
  statusCounts: Record<string, number> | null,
): Record<StatusFilterKey, number> {
  if (statusCounts) {
    return {
      all: Object.values(statusCounts).reduce((a, b) => a + b, 0),
      completed: statusCounts.processed ?? statusCounts.PROCESSED ?? 0,
      processing: statusCounts.processing ?? statusCounts.PROCESSING ?? 0,
      pending: statusCounts.pending ?? statusCounts.PENDING ?? 0,
      failed: statusCounts.failed ?? statusCounts.FAILED ?? 0,
    };
  }
  const counts: Record<StatusFilterKey, number> = {
    all: docs.length,
    completed: 0,
    processing: 0,
    pending: 0,
    failed: 0,
  };
  docs.forEach((doc) => {
    counts[STATUS_TO_FILTER[doc.status]] += 1;
  });
  return counts;
}

function documentMatchesStatus(doc: Document, statusFilter: StatusFilterKey) {
  return (
    statusFilter === 'all' ||
    doc.status === FILTER_TO_STATUS[statusFilter]
  );
}

function pipelineHistoryMessages(
  pipelineStatus: PipelineStatusResponse | null | undefined,
): string[] {
  const messages = [...(pipelineStatus?.history_messages ?? [])];
  const latest = pipelineStatus?.latest_message;
  if (latest && messages[messages.length - 1] !== latest) {
    messages.push(latest);
  }
  return messages.slice(-80);
}

interface PipelineControlProps {
  open: boolean;
  loading: boolean;
  error: string | null;
  status: PipelineStatusResponse | null | undefined;
  messages: readonly string[];
  onToggle?: () => void;
  onRefresh?: () => void;
}

function PipelineControl({
  open,
  loading,
  error,
  status,
  messages,
  onToggle,
  onRefresh,
}: Readonly<PipelineControlProps>) {
  return (
    <div className="pipeline-control">
      <button
        type="button"
        className={`btn${open ? ' active' : ''}`}
        title="Pipeline logs"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => onToggle?.()}
      >
        <Icon name="activity" size={14} />
        Pipeline
        <span
          className={`pipeline-state-badge ${
            status?.busy ? 'pipeline-state-busy' : 'pipeline-state-idle'
          }`}
          aria-label={`Pipeline ${status?.busy ? 'busy' : 'idle'}`}
        >
          {status?.busy ? 'BUSY' : 'IDLE'}
        </span>
      </button>
      {open && (
        <PipelinePopover
          loading={loading}
          error={error}
          status={status}
          messages={messages}
          onClose={onToggle}
          onRefresh={onRefresh}
        />
      )}
    </div>
  );
}

function PipelinePopover({
  loading,
  error,
  status,
  messages,
  onClose,
  onRefresh,
}: Readonly<{
  loading: boolean;
  error: string | null;
  status: PipelineStatusResponse | null | undefined;
  messages: readonly string[];
  onClose?: () => void;
  onRefresh?: () => void;
}>) {
  const liveState = loading ? 'refreshing' : 'idle';
  return (
    <dialog open className="pipeline-popover" aria-label="Pipeline logs">
      <div className="pp-header">
        <div className="pp-title">
          <Icon name="activity" size={14} />
          Pipeline
          <span className={`pp-state-badge ${status?.busy ? 'busy' : 'paused'}`}>
            <span className="pp-state-dot" />
            {status?.busy ? 'Busy' : 'Idle'}
          </span>
        </div>
        <button
          type="button"
          className="btn small"
          aria-label="Close pipeline logs"
          onClick={() => onClose?.()}
        >
          <Icon name="x" size={13} />
        </button>
      </div>

      <div className="pp-section">
        <div className="pp-stats">
          <div>
            <span className="pp-stat-num">{status?.job_count ?? 0}</span>
            <span className="pp-stat-lbl">Jobs</span>
          </div>
          <div>
            <span className="pp-stat-num">{messages.length}</span>
            <span className="pp-stat-lbl">Messages</span>
          </div>
          <div className="pp-job-name">
            <span className="pp-stat-num">{status?.job_name ?? '—'}</span>
            <span className="pp-stat-lbl">Current job</span>
          </div>
        </div>
      </div>

      <div className="pp-section">
        <h3>Latest</h3>
        <PipelineLatest error={error} loading={loading} status={status} />
      </div>

      <div className="pp-section">
        <h3>History</h3>
        {messages.length > 0 ? (
          <ol className="pp-log-list">
            {messages.map((message, index) => (
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
              loading ? 'pp-live-pill--loading' : 'pp-live-pill--idle'
            }`}
            aria-label={`Pipeline live ${liveState}`}
          >
            <span className="pp-live-dot" />
            LIVE
          </span>
        </span>
        <button type="button" className="btn small" onClick={() => onRefresh?.()}>
          <Icon name="refresh" size={13} />
          Refresh
        </button>
      </div>
    </dialog>
  );
}

function PipelineLatest({
  error,
  loading,
  status,
}: Readonly<{
  error: string | null;
  loading: boolean;
  status: PipelineStatusResponse | null | undefined;
}>) {
  if (error) return <p className="pp-error">{error}</p>;
  if (loading && !status) return <p className="pp-empty">Loading pipeline status…</p>;
  if (status?.latest_message) {
    return <p className="pp-latest">{status.latest_message}</p>;
  }
  return <p className="pp-empty">No pipeline message reported yet.</p>;
}

function RetryFailedButton({
  failedCount,
  ingestionDisabled,
  onScanRetry,
  onAddToast,
}: Readonly<{
  failedCount: number;
  ingestionDisabled: boolean;
  onScanRetry?: (failedCount: number) => void;
  onAddToast: (title: string, sub?: string) => void;
}>) {
  const disabled = failedCount === 0 || ingestionDisabled;
  const title = retryButtonTitle(failedCount, ingestionDisabled);
  const retry = () => {
    if (disabled) return;
    if (onScanRetry) {
      onScanRetry(failedCount);
      return;
    }
    onAddToast('Re-processing failed sources', 'POST /documents/reprocess_failed');
  };
  return (
    <button
      type="button"
      className={`btn${failedCount > 0 ? ' btn-retry' : ''}`}
      disabled={disabled}
      title={title}
      onClick={retry}
    >
      <Icon name="refresh" size={14} />
      Re-process failed sources
      {failedCount > 0 && (
        <span className="pipeline-badge" aria-label={`${failedCount} failed`}>
          {failedCount}
        </span>
      )}
    </button>
  );
}

function retryButtonTitle(failedCount: number, ingestionDisabled: boolean) {
  if (ingestionDisabled) {
    return 'Memgraph instance quota reached — free space before re-processing';
  }
  if (failedCount === 0) return 'No failed sources to re-process';
  return `Re-process ${failedCount} failed source${
    failedCount > 1 ? 's' : ''
  } (POST /documents/reprocess_failed)`;
}

function BulkActionsBar({
  selectedCount,
  filteredCount,
  bulkDeleteArmed,
  onOpenBulk,
  onBulkDelete,
  onTriggerBulkDelete,
  onClearSelection,
}: Readonly<{
  selectedCount: number;
  filteredCount: number;
  bulkDeleteArmed: boolean;
  onOpenBulk: () => void;
  onBulkDelete?: (docs: readonly Document[]) => void;
  onTriggerBulkDelete: () => void;
  onClearSelection: () => void;
}>) {
  return (
    <section className="bulk-bar" aria-label="Bulk actions">
      <span className="bulk-count">
        <b>{selectedCount}</b> selected
        <span className="bulk-of">of {filteredCount}</span>
      </span>
      <button type="button" className="bulk-action primary" onClick={onOpenBulk}>
        <Icon name="tags" size={13} /> Retag {selectedCount} sources
      </button>
      {onBulkDelete && (
        <button
          type="button"
          className="bulk-action danger"
          onClick={onTriggerBulkDelete}
          data-testid="docs-bulk-delete"
          aria-label={`Delete ${selectedCount} sources`}
        >
          <Icon name="x" size={13} />{' '}
          {bulkDeleteArmed
            ? `Confirm delete ${selectedCount}`
            : `Delete ${selectedCount}`}
        </button>
      )}
      <button type="button" className="bulk-clear" onClick={onClearSelection}>
        <Icon name="x" size={12} /> Clear selection
      </button>
    </section>
  );
}

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
  currentPage?: number;
  totalCount?: number;
  statusCounts?: Record<string, number> | null;
  hasNextPage?: boolean;
  isPageFetching?: boolean;
  onPreviousPage?: () => void;
  onNextPage?: () => void;
  statusFilter?: StatusFilterKey;
  onStatusFilterChange?: (status: StatusFilterKey) => void;
  search?: string;
  onSearchChange?: (search: string) => void;
  tagFilters?: readonly string[];
  onTagFiltersChange?: (tags: readonly string[]) => void;
  sourceFilters?: readonly string[];
  onSourceFiltersChange?: (sources: readonly string[]) => void;
  onFiltersChanged?: () => void;
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
  currentPage = 1,
  totalCount,
  statusCounts = null,
  hasNextPage = false,
  isPageFetching = false,
  onPreviousPage,
  onNextPage,
  statusFilter: controlledStatusFilter,
  onStatusFilterChange,
  search: controlledSearch,
  onSearchChange,
  tagFilters: controlledTagFilters,
  onTagFiltersChange,
  sourceFilters: controlledSourceFilters,
  onSourceFiltersChange,
  onFiltersChanged,
}: Readonly<DocumentsTabProps>) {
  const openDetail = onOpenDetail ?? onDeleteDoc;
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [localStatusFilter, setLocalStatusFilter] = useUrlParam<StatusFilterKey>(
    'status',
    'all',
    {
      validate: (v) => (STATUS_FILTERS as readonly string[]).includes(v),
    },
  );
  const [localSearch, setLocalSearch] = useUrlParam<string>('q', '');
  const [localTagFilters, setLocalTagFilters] = useUrlArrayParam('tag', []);
  // Set by citation / drill-down navigation (?source=<file_path>). The
  // param survives topbar tab switches, so it MUST stay visible and
  // removable — an invisible filter silently shrank the table to one doc.
  const [localSourceFilters, setLocalSourceFilters] = useUrlArrayParam('source', []);
  const [tagAddOpen, setTagAddOpen] = useState(false);
  const [tagAddVal, setTagAddVal] = useState('');
  const [activeTagSuggestionIndex, setActiveTagSuggestionIndex] = useState(0);
  const [bulkDeleteArmed, setBulkDeleteArmed] = useState(false);
  const statusFilter = controlledStatusFilter ?? localStatusFilter;
  const search = controlledSearch ?? localSearch;
  const tagFilters = controlledTagFilters ?? localTagFilters;
  const sourceFilters = controlledSourceFilters ?? localSourceFilters;
  const updateStatusFilter = (next: StatusFilterKey) => {
    (onStatusFilterChange ?? setLocalStatusFilter)(next);
    onFiltersChanged?.();
  };
  const updateSearch = (next: string) => {
    (onSearchChange ?? setLocalSearch)(next);
    onFiltersChanged?.();
  };
  const updateTagFilters = (next: readonly string[]) => {
    (onTagFiltersChange ?? setLocalTagFilters)([...next]);
    onFiltersChanged?.();
  };
  const updateSourceFilters = (next: readonly string[]) => {
    (onSourceFiltersChange ?? setLocalSourceFilters)([...next]);
    onFiltersChanged?.();
  };

  const searchAndTagFiltered = useMemo(() => {
    return docs.filter((doc) =>
      documentMatchesSearchAndTags(doc, search, tagFilters, sourceFilters),
    );
  }, [docs, search, tagFilters, sourceFilters]);

  const counts = useMemo(() => {
    return statusCountsFor(searchAndTagFiltered, statusCounts);
  }, [searchAndTagFiltered, statusCounts]);
  const failedCount = counts.failed;

  const filtered = useMemo(() => {
    return searchAndTagFiltered.filter((doc) =>
      documentMatchesStatus(doc, statusFilter),
    );
  }, [searchAndTagFiltered, statusFilter]);

  const removeTagFilter = (t: string) =>
    updateTagFilters(tagFilters.filter((x) => x !== t));
  const addTagFilter = (t: string) => {
    if (t && !tagFilters.includes(t)) updateTagFilters([...tagFilters, t]);
    setTagAddVal('');
    setTagAddOpen(false);
    setActiveTagSuggestionIndex(0);
  };

  const tagSuggestions = useMemo(() => {
    const v = tagAddVal.toLowerCase();
    return tagCatalog
      .filter((t) => !tagFilters.includes(t.tag))
      .filter((t) => tagMatchesQuery(t, v))
      .sort(tagSuggestionComparator(v))
      .slice(0, 5);
  }, [tagAddVal, tagFilters, tagCatalog]);
  const activeTagSuggestion =
    tagSuggestions[
      Math.min(activeTagSuggestionIndex, Math.max(tagSuggestions.length - 1, 0))
    ];
  const tagListboxId = 'documents-tag-suggestions';

  const clickTagOnRow = (e: React.MouseEvent, tag: string) => {
    e.stopPropagation();
    if (!tagFilters.includes(tag)) updateTagFilters([...tagFilters, tag]);
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
    return pipelineHistoryMessages(pipelineStatus);
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
          <PipelineControl
            open={pipelineOpen}
            loading={pipelineLoading}
            error={pipelineError}
            status={pipelineStatus}
            messages={pipelineMessages}
            onToggle={onTogglePipeline}
            onRefresh={onRefreshPipeline}
          />
          {/* Audit C7: the only backend this action calls is
              ``POST /documents/reprocess_failed``. It stays enabled
              only when at least one source is failed and is labelled
              for that exact batch retry. */}
          <RetryFailedButton
            failedCount={failedCount}
            ingestionDisabled={ingestionDisabled}
            onScanRetry={onScanRetry}
            onAddToast={onAddToast}
          />
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
              onClick={() => updateStatusFilter(k)}
            >
              {STATUS_LABELS[k]} ({counts[k]})
            </button>
          ))}
        </div>
        <div className="search-source-field">
          <Icon name="search" size={13} color="var(--color-text-tertiary)" />
          <input
            className="search-source"
            placeholder="Search source name…"
            value={search}
            onChange={(e) => updateSearch(e.target.value)}
            aria-label="Search source"
          />
        </div>
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
                    updateSourceFilters(sourceFilters.filter((x) => x !== s))
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
                className="tag-filter-input"
                value={tagAddVal}
                onChange={(e) => {
                  setTagAddVal(e.target.value);
                  setActiveTagSuggestionIndex(0);
                }}
                onBlur={() => setTimeout(() => setTagAddOpen(false), 150)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setActiveTagSuggestionIndex((i) =>
                      tagSuggestions.length === 0
                        ? 0
                        : (i + 1) % tagSuggestions.length,
                    );
                    return;
                  }
                  if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setActiveTagSuggestionIndex((i) =>
                      tagSuggestions.length === 0
                        ? 0
                        : (i - 1 + tagSuggestions.length) %
                          tagSuggestions.length,
                    );
                    return;
                  }
                  if (e.key === 'Enter' && activeTagSuggestion) {
                    e.preventDefault();
                    addTagFilter(activeTagSuggestion.tag);
                  }
                  if (e.key === 'Escape') {
                    setTagAddOpen(false);
                    setActiveTagSuggestionIndex(0);
                  }
                }}
                placeholder="tag…"
                aria-label="Add tag filter"
                aria-autocomplete="list"
                aria-controls={tagListboxId}
                aria-expanded={tagSuggestions.length > 0}
              />
              {tagSuggestions.length > 0 && (
                <div
                  id={tagListboxId}
                  className="autocomplete floating-autocomplete"
                  aria-label="Tag suggestions"
                >
                  {tagSuggestions.map((s, i) => (
                    <button
                      type="button"
                      key={s.tag}
                      className={`autocomplete-row${
                        i === activeTagSuggestionIndex ? ' focus' : ''
                      }`}
                      aria-selected={i === activeTagSuggestionIndex}
                      onMouseEnter={() => setActiveTagSuggestionIndex(i)}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => addTagFilter(s.tag)}
                      data-testid={`docs-tag-sugg-${s.tag}`}
                    >
                      <div className="row1">
                        <span>{s.tag}</span>
                        <span className="badge">{s.category}</span>
                      </div>
                      <div className="def">{s.def}</div>
                    </button>
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
        <BulkActionsBar
          selectedCount={selected.size}
          filteredCount={filtered.length}
          bulkDeleteArmed={bulkDeleteArmed}
          onOpenBulk={openBulk}
          onBulkDelete={onBulkDelete}
          onTriggerBulkDelete={triggerBulkDelete}
          onClearSelection={clearSelection}
        />
      )}

      <div className="docs-table-wrap">
        <div className="docs-table">
          <div className="docs-row header has-select">
            <div className="cell-select">
              <label className="row-check-target">
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
              </label>
            </div>
            <div>Source</div>
            <div>Indexed preview</div>
            <div>
              Tags{' '}
              <span className="docs-table-subhead">
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
              className="docs-empty"
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
        {(totalCount != null || currentPage > 1 || hasNextPage) && (
          <div className="docs-pagination" data-testid="docs-pagination">
            <span className="docs-pagination-label">
              Page {currentPage}
              {totalCount != null ? ` · ${totalCount.toLocaleString()} total` : ''}
            </span>
            <button
              type="button"
              className="ghost-btn"
              onClick={onPreviousPage}
              disabled={currentPage <= 1 || isPageFetching}
              data-testid="docs-page-prev"
            >
              Previous
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={onNextPage}
              disabled={!hasNextPage || isPageFetching}
              data-testid="docs-page-next"
            >
              {isPageFetching ? 'Loading...' : 'Next'}
            </button>
          </div>
        )}
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
}: Readonly<DocRowProps>) {
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
      <div className="cell-select">
        <label className="row-check-target">
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
        </label>
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
          >
            Indexing failed: {doc.error_msg}
          </div>
        )}
      </div>
      <div className="cell-tags">
        {visibleTags.map((t) => (
          <button
            type="button"
            key={t}
            className="tag-chip-button"
            onClick={(e) => onClickTag(e, t)}
            data-testid={`row-tag-${doc.doc_id}-${t}`}
          >
            <TagChip tag={t} />
          </button>
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
