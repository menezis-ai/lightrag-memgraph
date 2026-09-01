import type { TagCategory } from '../../types/tag';
import { Icon } from '../Icon';

export interface DomainDraft {
  key: string;
  id: string;
  label: string;
  color: string;
  existing: boolean;
}

export interface DomainRailEditorProps {
  draft: readonly DomainDraft[];
  error: string | null;
  tagCounts: Record<string, number>;
  removedDomainsWithTags: readonly (TagCategory & { count: number })[];
  isSaving: boolean;
  onAdd: () => void;
  onUpdate: (key: string, patch: Partial<Pick<DomainDraft, 'id' | 'label' | 'color'>>) => void;
  onRemove: (key: string) => void;
  onCancel: () => void;
  onSave: () => void;
}

function normalizeDomainId(value: string): string {
  return value.trim().toLowerCase().replaceAll(/\s+/g, '-');
}

export function DomainRailEditor({
  draft,
  error,
  tagCounts,
  removedDomainsWithTags,
  isSaving,
  onAdd,
  onUpdate,
  onRemove,
  onCancel,
  onSave,
}: Readonly<DomainRailEditorProps>) {
  return (
    <div className="domain-rail-editor" data-testid="domain-rail-editor">
      <div className="domain-rail-rows">
        {draft.map((row) => (
          <div className="domain-rail-row" key={row.key}>
            <input
              className="domain-color-input domain-rail-color"
              type="color"
              value={row.color}
              aria-label={`${row.label || row.id} color`}
              required
              aria-required="true"
              onChange={(event) => onUpdate(row.key, { color: event.target.value.toUpperCase() })}
            />
            <div className="domain-rail-fields">
              <input
                className="text-input domain-rail-label"
                value={row.label}
                aria-label={`${row.id} domain label`}
                placeholder="Domain name"
                required
                aria-required="true"
                onChange={(event) => onUpdate(row.key, { label: event.target.value })}
              />
              {!row.existing && (
                <input
                  className="text-input domain-rail-id"
                  value={row.id}
                  aria-label={`${row.label || row.id} domain id`}
                  placeholder="domain-id"
                  required
                  aria-required="true"
                  onChange={(event) => onUpdate(row.key, { id: normalizeDomainId(event.target.value) })}
                />
              )}
            </div>
            <span className="domain-rail-count">{row.existing ? (tagCounts[row.id] ?? 0) : 0}</span>
            <button className="rail-tool-btn danger" type="button" onClick={() => onRemove(row.key)} aria-label={`Remove ${row.label || row.id}`}>
              <Icon name="trash" size={12} />
            </button>
          </div>
        ))}
      </div>
      <button className="ghost-btn small domain-rail-add" type="button" onClick={onAdd}><Icon name="plus" size={12} /> Add domain</button>
      {removedDomainsWithTags.length > 0 && (
        <output className="impact-box warning domain-rail-impact">
          <Icon name="alert-triangle" size={14} />
          <span>
            Removed domains with tags become uncategorized until retagged:{' '}
            {removedDomainsWithTags.map((category) => <code key={category.id}>{category.label} ({category.count})</code>)}
          </span>
        </output>
      )}
      {error && <div className="impact-box danger domain-rail-impact" role="alert"><Icon name="alert-triangle" size={14} /><span>{error}</span></div>}
      <div className="domain-rail-actions">
        <button className="ghost-btn small" type="button" onClick={onCancel} disabled={isSaving}>Cancel</button>
        <button className="primary-btn small" type="button" onClick={onSave} disabled={isSaving}>{isSaving ? 'Saving…' : 'Save'}</button>
      </div>
    </div>
  );
}
