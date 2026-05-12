/**
 * DocumentsTab — table of sources with status filter, search, tag filter,
 * multi-select, and bulk retag.
 *
 * Ported from Desktop/UI/documents.jsx. Scope of this port:
 *   - main shell (header + filters + table + bulk bar)
 *   - DocRow (private, internal)
 *
 * Deferred to later slice/sprint:
 *   - PipelineStatusPopover (pipeline-status modal)
 *   - DocDetailPanel (right-side doc detail panel)
 *
 * Behavior delta vs the proto:
 *   - thesaurus injected via prop (no window.MOCK_THESAURUS)
 *   - onAddToast typed as (title, sub?) — host owns the toast queue
 *   - URL state via useUrlParam/useUrlArrayParam from S1
 */

import { useMemo, useState } from 'react';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
import {
  useUrlArrayParam,
  useUrlParam,
} from '../hooks/useUrlParam';
import type { Document, DocumentStatus } from '../types/document';
import type { ThesaurusEntry } from '../types/thesaurus';

const STATUS_LABELS: Record<'all' | DocumentStatus, string> = {
  all: 'All',
  completed: 'Completed',
  processing: 'Processing',
  pending: 'Pending',
  failed: 'Failed',
};

const STATUS_KEYS: readonly ('all' | DocumentStatus)[] = [
  'all',
  'completed',
  'processing',
  'pending',
  'failed',
];

export interface DocumentsTabProps {
  docs: readonly Document[];
  thesaurus: readonly ThesaurusEntry[];
  onOpenAdd: () => void;
  onOpenRetag: (doc: Document) => void;
  onOpenBulkRetag: (docs: readonly Document[]) => void;
  onAddToast: (title: string, sub?: string) => void;
}

type StatusFilter = 'all' | DocumentStatus;

export function DocumentsTab({
  docs,
  thesaurus,
  onOpenAdd,
  onOpenRetag,
  onOpenBulkRetag,
  onAddToast,
}: DocumentsTabProps) {
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [statusFilter, setStatusFilter] = useUrlParam<StatusFilter>(
    'status',
    'all',
    {
      validate: (v) =>
        (['all', 'completed', 'processing', 'pending', 'failed'] as const).includes(
          v as StatusFilter,
        ),
    },
  );
  const [search, setSearch] = useUrlParam<string>('q', '');
  const [tagFilters, setTagFilters] = useUrlArrayParam('tag', []);
  const [tagAddOpen, setTagAddOpen] = useState(false);
  const [tagAddVal, setTagAddVal] = useState('');

  const counts = useMemo(() => {
    const c: Record<StatusFilter, number> = {
      all: docs.length,
      completed: 0,
      processing: 0,
      pending: 0,
      failed: 0,
    };
    docs.forEach((d) => {
      c[d.status]++;
    });
    return c;
  }, [docs]);
  const failedCount = counts.failed;

  const filtered = useMemo(() => {
    return docs.filter((d) => {
      if (statusFilter !== 'all' && d.status !== statusFilter) return false;
      if (search && !d.source.toLowerCase().includes(search.toLowerCase()))
        return false;
      if (
        tagFilters.length &&
        !tagFilters.every((t) => d.tags.includes(t))
      )
        return false;
      return true;
    });
  }, [docs, statusFilter, search, tagFilters]);

  const removeTagFilter = (t: string) =>
    setTagFilters(tagFilters.filter((x) => x !== t));
  const addTagFilter = (t: string) => {
    if (t && !tagFilters.includes(t)) setTagFilters([...tagFilters, t]);
    setTagAddVal('');
    setTagAddOpen(false);
  };

  const thesaurusSuggestions = useMemo(() => {
    const v = tagAddVal.toLowerCase();
    return thesaurus
      .filter((t) => !tagFilters.includes(t.tag))
      .filter((t) => !v || t.tag.includes(v))
      .slice(0, 5);
  }, [tagAddVal, tagFilters, thesaurus]);

  const clickTagOnRow = (e: React.MouseEvent, tag: string) => {
    e.stopPropagation();
    if (!tagFilters.includes(tag)) setTagFilters([...tagFilters, tag]);
  };

  const toggleRow = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };
  const filteredIds = filtered.map((d) => d.id);
  const allFilteredSelected =
    filteredIds.length > 0 && filteredIds.every((id) => selected.has(id));
  const someFilteredSelected = filteredIds.some((id) => selected.has(id));
  const toggleAll = () => {
    const next = new Set(selected);
    if (allFilteredSelected) filteredIds.forEach((id) => next.delete(id));
    else filteredIds.forEach((id) => next.add(id));
    setSelected(next);
  };
  const clearSelection = () => setSelected(new Set());
  const selectedDocs = docs.filter((d) => selected.has(d.id));
  const openBulk = () => onOpenBulkRetag(selectedDocs);

  const hasFilters =
    statusFilter !== 'all' ||
    !!search ||
    tagFilters.length > 0 ||
    selected.size > 0;
  const clearAllFilters = () => {
    const summary = [
      statusFilter !== 'all' && `status: ${statusFilter}`,
      search && `q: ${search}`,
      tagFilters.length > 0 && `tags: ${tagFilters.join(', ')}`,
      selected.size > 0 && `${selected.size} selected`,
    ]
      .filter(Boolean)
      .join(' · ');
    setStatusFilter('all');
    setSearch('');
    setTagFilters([]);
    setSelected(new Set());
    if (summary) onAddToast('Filters cleared', summary);
  };

  return (
    <div className="docs">
      <div className="docs-header">
        <h1>Document management</h1>
        <div className="docs-header-actions">
          <button
            type="button"
            className={`btn${failedCount > 0 ? ' btn-retry' : ''}`}
            onClick={() =>
              onAddToast(
                failedCount > 0
                  ? `Scan started · retrying ${failedCount} failed source${failedCount > 1 ? 's' : ''}`
                  : 'Pipeline scan started',
                failedCount > 0
                  ? 'POST /documents/scan?retry=failed · workers picking up now'
                  : 'POST /documents/scan · re-scanning sources for changes',
              )
            }
          >
            <Icon name="refresh" size={14} />
            {failedCount > 0 ? 'Scan / Retry' : 'Scan'}
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
            className="btn"
            onClick={clearAllFilters}
            disabled={!hasFilters}
            title={
              hasFilters
                ? 'Clear status, search, tag filters and selection'
                : 'No filters active'
            }
          >
            <Icon name="x" size={14} /> Clear
          </button>
          <button type="button" className="btn primary" onClick={onOpenAdd}>
            <Icon name="cloud-upload" size={14} /> Add source
          </button>
        </div>
      </div>

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

      <div className="tag-filter-row">
        <span className="lbl">
          Filter by tag<em>— Twin</em>
        </span>
        <div className="tag-chips">
          {tagFilters.map((t) => (
            <TagChip key={t} tag={t} removable onRemove={removeTagFilter} />
          ))}
          {tagAddOpen ? (
            <div style={{ position: 'relative' }}>
              <input
                autoFocus
                value={tagAddVal}
                onChange={(e) => setTagAddVal(e.target.value)}
                onBlur={() => setTimeout(() => setTagAddOpen(false), 150)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && thesaurusSuggestions[0])
                    addTagFilter(thesaurusSuggestions[0].tag);
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
              {thesaurusSuggestions.length > 0 && (
                <div
                  className="autocomplete"
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    marginTop: 4,
                    minWidth: 200,
                    zIndex: 30,
                  }}
                >
                  {thesaurusSuggestions.map((s, i) => (
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
          <button
            type="button"
            className="bulk-action"
            onClick={() =>
              onAddToast(
                `Re-processing ${selected.size} sources`,
                'Queued at pipeline · workers picking up now',
              )
            }
          >
            <Icon name="refresh" size={13} /> Re-process
          </button>
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
            <div>Summary</div>
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
              key={d.id}
              doc={d}
              checked={selected.has(d.id)}
              onToggle={toggleRow}
              onOpenRetag={onOpenRetag}
              onClickTag={clickTagOnRow}
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
}

function DocRow({ doc, checked, onToggle, onOpenRetag, onClickTag }: DocRowProps) {
  const isFail = doc.status === 'failed';
  const visibleTags = doc.tags.slice(0, 2);
  const overflow = doc.tags.length - visibleTags.length;
  return (
    <div
      className={`docs-row has-select${checked ? ' is-checked' : ''}`}
      data-testid={`docs-row-${doc.id}`}
    >
      <div className="cell-select" onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          className="row-check"
          checked={checked}
          onChange={() => onToggle(doc.id)}
          aria-label={`Select ${doc.source}`}
        />
      </div>
      <div className="cell-source">
        <SourceIcon type={doc.type} size={14} />
        <span className={doc.type !== 'file' ? 'mono' : ''}>{doc.source}</span>
      </div>
      <div className="cell-summary">{doc.summary}</div>
      <div className="cell-tags">
        {visibleTags.map((t) => (
          <span
            key={t}
            onClick={(e) => onClickTag(e, t)}
            data-testid={`row-tag-${doc.id}-${t}`}
          >
            <TagChip tag={t} />
          </span>
        ))}
        {overflow > 0 && (
          <span className="tag-overflow">+{overflow}</span>
        )}
      </div>
      <div className={`cell-status status-${doc.status}`}>{doc.status}</div>
      <div className="cell-chunks">{doc.chunks}</div>
      <div className="cell-updated">{doc.updated}</div>
      <div className="cell-actions">
        <button
          type="button"
          className="row-action"
          onClick={() => onOpenRetag(doc)}
          aria-label={`Retag ${doc.source}`}
        >
          <Icon name="tags" size={13} />
        </button>
        {isFail && (
          <span className="row-fail">
            <Icon name="alert-triangle" size={13} />
          </span>
        )}
      </div>
    </div>
  );
}
