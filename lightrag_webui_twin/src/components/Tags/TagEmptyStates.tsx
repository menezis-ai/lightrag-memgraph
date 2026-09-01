import { Icon } from '../Icon';
import type { TagCategory, TagEntry, TagStatusFilter } from '../../types/tag';

interface TagsEmptyZeroProps {
  canSuggest: boolean;
  onRequest: () => void;
}

function TagsEmptyZero({ canSuggest, onRequest }: Readonly<TagsEmptyZeroProps>) {
  return (
    <div className="tags-empty zero" data-testid="tags-empty-zero">
      <div className="tags-empty-illus" aria-hidden="true">
        <svg width="120" height="80" viewBox="0 0 120 80" fill="none">
          <rect x="6" y="18" width="44" height="18" rx="3" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.45" />
          <rect x="54" y="32" width="56" height="18" rx="3" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.35" />
          <rect x="20" y="48" width="38" height="18" rx="3" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.25" />
        </svg>
      </div>
      <div className="tags-empty-title">No tags in this folder yet</div>
      <p className="tags-empty-body">
        The tag catalog is empty. Start by requesting your first tag — a steward will
        review and promote it to a Tier 1 / 2 / 3 slot. Every tagged source then
        becomes filterable in Retrieval.
      </p>
      <div className="tags-empty-actions">
        {canSuggest ? (
          <button className="primary-btn" onClick={onRequest}>
            <Icon name="plus" size={12} /> Request the first tag
          </button>
        ) : (
          <span className="tags-empty-hint">
            Your role doesn't allow tag requests. Ask a Tier 2+ reviewer.
          </span>
        )}
      </div>
      <ul className="tags-empty-tips">
        <li><Icon name="info-circle" size={11} /> Tier 1 (Trunk) — gov-validated, applies cross-folder</li>
        <li><Icon name="info-circle" size={11} /> Tier 2 (Branch) — dept-scoped, steward-approved</li>
        <li><Icon name="info-circle" size={11} /> Tier 3 (Leaf) — user-proposed, lightweight review</li>
      </ul>
    </div>
  );
}

export interface TagsEmptyFilteredProps {
  q: string;
  selectedCat: string;
  selectedStatus: TagStatusFilter;
  categories: readonly TagCategory[];
  suggestions: readonly TagEntry[];
  canSuggest: boolean;
  onClear: () => void;
  onPickTag: (name: string) => void;
  onRequest: () => void;
}

function TagsEmptyFiltered({
  q,
  selectedCat,
  selectedStatus,
  categories,
  suggestions,
  canSuggest,
  onClear,
  onPickTag,
  onRequest,
}: Readonly<TagsEmptyFilteredProps>) {
  const catLabel =
    selectedCat === 'all' ? null : categories.find((c) => c.id === selectedCat);
  const active = [
    q.trim() ? { key: 'q', label: `search: "${q.trim()}"` } : null,
    catLabel ? { key: 'cat', label: `category: ${catLabel.label}` } : null,
    selectedStatus === 'all'
      ? null
      : { key: 'status', label: `status: ${selectedStatus}` },
  ].filter(Boolean) as { key: string; label: string }[];

  return (
    <div className="tags-empty filtered" data-testid="tags-empty-filtered">
      <div className="tags-empty-ico">
        <Icon name="search" size={20} color="var(--color-text-tertiary)" />
      </div>
      <div className="tags-empty-title">No tags match the current filter</div>
      {active.length > 0 && (
        <div className="tags-empty-chips">
          {active.map((item) => (
            <span key={item.key} className="tags-empty-chip">{item.label}</span>
          ))}
        </div>
      )}
      <div className="tags-empty-actions">
        <button className="primary-btn" onClick={onClear}>Clear filters</button>
        {canSuggest && q.trim() && (
          <button className="ghost-btn" onClick={onRequest}>
            <Icon name="plus" size={12} /> Request{' '}
            <code>{q.trim().toLowerCase().replaceAll(/\s+/g, '-')}</code> as new tag
          </button>
        )}
      </div>
      {suggestions.length > 0 && (
        <div className="tags-empty-suggest">
          <div className="tags-empty-suggest-h">Try one of these instead</div>
          <div className="tags-empty-suggest-row">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion.tag}
                className="tags-empty-suggest-chip"
                onClick={() => {
                  onClear();
                  onPickTag(suggestion.tag);
                }}
              >
                <code>{suggestion.tag}</code>
                <span className="tags-empty-suggest-meta">{suggestion.sources_count} docs</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Empty grid body: zero-tags onboarding vs filtered-to-nothing state. */
export function TagsGridEmpty({
  totalActive,
  ...filteredProps
}: Readonly<TagsEmptyFilteredProps & { totalActive: number }>) {
  if (totalActive === 0) {
    return <TagsEmptyZero canSuggest={filteredProps.canSuggest} onRequest={filteredProps.onRequest} />;
  }
  return <TagsEmptyFiltered {...filteredProps} />;
}
