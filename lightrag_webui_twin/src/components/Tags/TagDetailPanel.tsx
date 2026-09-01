import { Icon } from '../Icon';
import { StatusBadge } from '../StatusBadge';
import type { TagActionCommit } from '../TagActionModal';
import type { TagAction, TagCategory, TagEntry } from '../../types/tag';

export interface TagDetailPanelProps {
  t: TagEntry | null;
  allTags: readonly TagEntry[];
  categories: readonly TagCategory[];
  onSelect: (name: string) => void;
  onAction: (action: TagAction) => void;
  onCommit?: (commit: TagActionCommit) => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
  canEdit: boolean;
  canSuggest: boolean;
}

/** Read-only tag facts and governance actions for the selected catalog entry. */
export function TagDetailPanel({
  t,
  allTags,
  categories,
  onSelect,
  onAction,
  onCommit,
  onNavigate,
  canEdit,
  canSuggest,
}: Readonly<TagDetailPanelProps>) {
  if (!t) return null;
  const cat = categories.find((c) => c.id === t.category);
  return (
    <aside className="tag-detail">
      <div className="detail-head">
        <div
          className="detail-kind"
          style={{ color: cat ? cat.color : 'var(--color-text-secondary)' }}
        >
          <span
            className="rail-dot"
            style={{ background: cat ? cat.color : 'var(--color-text-tertiary)' }}
          />
          {cat ? cat.label : 'Uncategorized'}
        </div>
        <div className="tag-detail-h">
          <code className="tag-detail-name">{t.tag}</code>
          <StatusBadge status={t.status} size="md" />
        </div>
        {t.aliases.length > 0 && (
          <div className="tag-detail-aliases">
            <span className="al-label">Synonyms:</span>
            {t.aliases.map((a) => (
              <code key={a}>{a}</code>
            ))}
          </div>
        )}
        <div className="detail-summary">{t.def}</div>
      </div>

      <div className="detail-section">
        <div className="detail-section-h">Usage</div>
        <div className="usage-grid">
          <button
            type="button"
            className="usage-cell usage-cell-link"
            disabled={!onNavigate || t.sources_count === 0}
            onClick={() => onNavigate?.('documents', { tag: t.tag })}
            aria-label={`View ${t.sources_count} documents tagged ${t.tag}`}
          >
            <div className="usage-num">{t.sources_count}</div>
            <div className="usage-lbl">Docs</div>
          </button>
          <div className="usage-cell">
            <div className="usage-num">{t.chunks_count.toLocaleString()}</div>
            <div className="usage-lbl">Chunks</div>
          </div>
          <div className="usage-cell">
            <div className="usage-num">{t.query_freq_30d}</div>
            <div className="usage-lbl">Queries / 30d</div>
          </div>
        </div>
      </div>

      {t.examples.length > 0 && (
        <div className="detail-section">
          <div className="detail-section-h">Last tagged docs</div>
          <div className="example-list">
            {t.examples.map((e) => (
              <button
                key={e}
                className="example-row"
                onClick={() => onNavigate?.('documents', { q: e })}
              >
                <Icon
                  name={e.includes('/') ? 'brand-confluence' : 'file-text'}
                  size={12}
                  color="var(--color-text-tertiary)"
                />
                <span>{e}</span>
                <Icon
                  name="arrow-right"
                  size={11}
                  color="var(--color-text-tertiary)"
                />
              </button>
            ))}
            {t.sources_count > t.examples.length && (
              <button
                className="example-more"
                onClick={() => onNavigate?.('documents', { tag: t.tag })}
              >
                View all {t.sources_count} docs in Documents →
              </button>
            )}
          </div>
        </div>
      )}

      {t.related.length > 0 && (
        <div className="detail-section">
          <div className="detail-section-h">Co-occurring tags</div>
          <div className="related-list">
            {t.related.map((r) => {
              const rt = allTags.find((x) => x.tag === r.tag);
              if (!rt) return null;
              return (
                <button
                  key={r.tag}
                  className="related-chip"
                  onClick={() => onSelect(r.tag)}
                  data-testid={`related-${r.tag}`}
                >
                  <code>{r.tag}</code>
                  <span className="related-strength">
                    {(r.strength * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="detail-section">
        <div className="detail-section-h">History</div>
        <div className="history-list">
          <div className="hist-item">
            <span className="hist-when">{t.last_edit.at}</span>
            <span className="hist-what">{t.last_edit.action ?? 'edited'}</span>
            <span className="hist-who">by {t.last_edit.by}</span>
          </div>
          <div className="hist-item">
            <span className="hist-when">{t.created.at}</span>
            <span className="hist-what">created</span>
            <span className="hist-who">by {t.created.by}</span>
          </div>
        </div>
      </div>

      <div className="detail-section detail-cta">
        <button
          className="ghost-btn"
          type="button"
          disabled={!onNavigate || t.sources_count === 0}
          onClick={() => onNavigate?.('documents', { tag: t.tag })}
        >
          <Icon name="external-link" size={11} /> See documents containing this tag
        </button>
      </div>

      <div className="detail-actions wrap">
        {!canSuggest && (
          <span className="muted-italic">
            Palier 1 — read-only. Upgrade to palier 2 to suggest edits.
          </span>
        )}
        {canSuggest && !canEdit && (
          <button
            className="ghost-btn small"
            onClick={() => onAction({ kind: 'suggest', tag: t })}
          >
            Suggest edit
          </button>
        )}
        {canEdit && (
          <>
            <button
              className="ghost-btn small"
              onClick={() => onAction({ kind: 'edit', tag: t })}
            >
              Edit
            </button>
            <button
              className="ghost-btn small"
              onClick={() => onAction({ kind: 'synonyms', tag: t })}
            >
              Manage synonyms
            </button>
            {t.status === 'deprecated' ? (
              <button
                className="ghost-btn small"
                onClick={() => onCommit?.({ kind: 'reactivate', tag: t })}
              >
                Reactivate
              </button>
            ) : (
              <button
                className="ghost-btn small"
                onClick={() => onAction({ kind: 'deprecate', tag: t })}
              >
                Deprecate
              </button>
            )}
            <button
              className="ghost-btn small danger"
              onClick={() => onAction({ kind: 'delete', tag: t })}
            >
              Delete
            </button>
          </>
        )}
      </div>
    </aside>
  );
}
