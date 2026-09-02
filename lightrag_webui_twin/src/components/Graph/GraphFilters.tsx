import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from '../Icon';
import { GRAPH_TYPE_LABEL, type GraphEntityType } from '../../types/graph';
import { TYPE_KEYS } from './graphLayout';
import type { MatchMode } from './graphTypes';

interface GraphFiltersProps {
  activeTypes: readonly string[];
  typeCounts: Partial<Record<GraphEntityType, number>>;
  colors: Record<GraphEntityType, string>;
  allTags: readonly string[];
  deprecatedTags?: readonly string[];
  tagFilter: readonly string[];
  onTagFilterChange: (next: string[]) => void;
  tagMatchMode: MatchMode;
  onTagMatchModeChange: (value: MatchMode) => void;
  allSourceDocs: readonly string[];
  docFilter: readonly string[];
  onDocFilterChange: (next: string[]) => void;
  docMatchMode: MatchMode;
  onDocMatchModeChange: (value: MatchMode) => void;
  docLabels?: Readonly<Record<string, string>>;
  onToggleType: (type: GraphEntityType) => void;
}

export function GraphFilters({
  activeTypes,
  typeCounts,
  colors,
  allTags,
  deprecatedTags = [],
  tagFilter,
  onTagFilterChange,
  tagMatchMode,
  onTagMatchModeChange,
  allSourceDocs,
  docFilter,
  onDocFilterChange,
  docMatchMode,
  onDocMatchModeChange,
  docLabels,
  onToggleType,
}: Readonly<GraphFiltersProps>) {
  const toggleType = onToggleType;
  const setTagFilter = onTagFilterChange;
  const setTagMatchMode = onTagMatchModeChange;
  const setDocFilter = onDocFilterChange;
  const setDocMatchMode = onDocMatchModeChange;
  return (
    <aside className="kg-rail">
          <div className="kg-rail-h">Entity types</div>
          <ul className="kg-type-list">
            {TYPE_KEYS.map((t) => {
              const on = activeTypes.includes(t);
              return (
                <li key={t}>
                  <button
                    className={`kg-type-row${on ? ' is-on' : ''}`}
                    onClick={() => toggleType(t)}
                    aria-pressed={on}
                    data-testid={`kg-type-${t}`}
                  >
                    <span
                      className="kg-type-swatch"
                      style={{ background: colors[t] }}
                    />
                    <span className="kg-type-name">{GRAPH_TYPE_LABEL[t]}</span>
                    <span className="kg-type-count">{typeCounts[t] ?? 0}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          <FilterPicker
            label="Filter by tag"
            options={allTags}
            deprecatedOptions={deprecatedTags}
            selected={tagFilter}
            onChange={setTagFilter}
            placeholder="Search tags…"
            listboxId="kg-tag-filter-suggestions"
          />
          <FilterMatchMode
            label="Tag filter mode"
            value={tagMatchMode}
            onChange={setTagMatchMode}
            disabled={tagFilter.length < 2}
          />
          <FilterPicker
            label="Filter by document"
            options={allSourceDocs}
            selected={docFilter}
            onChange={setDocFilter}
            placeholder="Search documents…"
            format={(id) => docLabels?.[id] ?? id}
            listboxId="kg-doc-filter-suggestions"
          />
          <FilterMatchMode
            label="Document filter mode"
            value={docMatchMode}
            onChange={setDocMatchMode}
            disabled={docFilter.length < 2}
          />
          <div className="kg-legend">
            <div className="kg-legend-h">Legend</div>
            <ul>
              <li>
                <span className="kg-legend-line" /> relation
              </li>
              <li>
                <span className="kg-legend-line strong" /> relation (high confidence)
              </li>
              <li>
                <span className="kg-legend-dot" /> node size = mentions
              </li>
            </ul>
          </div>
    </aside>


  );
}

// ─────────────────────────────────────────── FilterPicker (B4) ──
// Fuzzy-search picker scaled past chip-walls (200+ sources / 50+ tags).
// Layout (user request 2026-05-31): search input FIRST, removable pills
// BELOW the search bar — inverse of the design prototype, more intuitive
// for "search → click to add → see what's selected just below".
export function FilterMatchMode({
  label,
  value,
  onChange,
  disabled,
}: Readonly<{
  label: string;
  value: 'any' | 'all';
  onChange: (value: 'any' | 'all') => void;
  disabled?: boolean;
}>) {
  return (
    <div
      className={`kg-filter-mode${disabled ? ' is-disabled' : ''}`}
      aria-label={label}
    >
      <button
        type="button"
        className={value === 'any' ? 'is-on' : ''}
        onClick={() => onChange('any')}
        aria-pressed={value === 'any'}
        disabled={disabled}
      >
        Any
      </button>
      <button
        type="button"
        className={value === 'all' ? 'is-on' : ''}
        onClick={() => onChange('all')}
        aria-pressed={value === 'all'}
        disabled={disabled}
      >
        All
      </button>
    </div>
  );
}

interface FilterPickerProps {
  label: string;
  options: readonly string[];
  selected: readonly string[];
  onChange: (next: string[]) => void;
  placeholder: string;
  format?: (x: string) => string;
  listboxId?: string;
  /** Lifecycle decoration for historical values already selected through a
   * saved URL. Deprecated values are deliberately absent from `options`. */
  deprecatedOptions?: readonly string[];
}

export function FilterPicker({
  label,
  options,
  selected,
  onChange,
  placeholder,
  format,
  listboxId,
  deprecatedOptions = [],
}: Readonly<FilterPickerProps>) {
  const fmt = format ?? ((x: string) => x);
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [focus, setFocus] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const deprecatedSet = useMemo(
    () => new Set(deprecatedOptions),
    [deprecatedOptions],
  );
  const listId =
    listboxId ??
    `${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}-filter-suggestions`;

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  const fuzzy = (opt: string, q: string): boolean => {
    opt = opt.toLowerCase();
    q = q.toLowerCase().trim();
    if (!q) return true;
    if (opt.includes(q)) return true;
    let i = 0;
    for (const char of opt) {
      if (char === q[i]) i++;
      if (i >= q.length) return true;
    }
    return false;
  };

  const avail = options.filter((o) => !selected.includes(o));
  const rank = (o: string) => {
    const q = query.toLowerCase().trim();
    const a = fmt(o).toLowerCase().indexOf(q);
    const b = o.toLowerCase().indexOf(q);
    if (a >= 0) return a;
    if (b >= 0) return b;
    return 999;
  };
  const results = (
    query
      ? avail
          .filter((o) => fuzzy(fmt(o), query) || fuzzy(o, query))
          .sort((a, b) => rank(a) - rank(b) || fmt(a).length - fmt(b).length)
      : avail
  ).slice(0, 8);
  const activeOption = results[Math.min(focus, Math.max(results.length - 1, 0))];
  const activeOptionId =
    open && activeOption ? `${listId}-option-${focus}` : undefined;

  const add = (o: string) => {
    onChange(selected.concat([o]));
    setQuery('');
    setFocus(0);
  };
  const remove = (o: string) => onChange(selected.filter((x) => x !== o));

  return (
    <div className="kg-rail-filter" ref={boxRef}>
      <div className="kg-rail-h">
        {label}
        {selected.length > 0 && (
          <button
            type="button"
            className="kg-rail-clear"
            onClick={() => onChange([])}
          >
            clear ({selected.length})
          </button>
        )}
      </div>
      {/* Search input FIRST (per 2026-05-31 user request — inverse of prototype) */}
      <div className="kg-picker">
          <input
            className="kg-picker-input"
            value={query}
            placeholder={placeholder}
            role="combobox"
            aria-label={label}
            aria-autocomplete="list"
            aria-expanded={open && results.length > 0}
            aria-controls={listId}
            aria-activedescendant={activeOptionId}
            onFocus={() => setOpen(true)}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
              setFocus(0);
            }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (results.length === 0) return;
                setFocus((f) => (f + 1) % results.length);
              } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (results.length === 0) return;
                setFocus((f) => (f - 1 + results.length) % results.length);
              } else if (e.key === 'Enter' && activeOption) {
                e.preventDefault();
                add(activeOption);
              } else if (e.key === 'Escape') {
                e.preventDefault();
                setOpen(false);
              }
            }}
          />
        {open && results.length > 0 && (
          <div id={listId} role="listbox" aria-label={`${label} suggestions`} className="kg-picker-menu">
            {results.map((o, i) => (
              <button
                key={o}
                type="button"
                id={`${listId}-option-${i}`}
                role="option"
                aria-selected={i === focus}
                className={`kg-picker-opt${i === focus ? ' focus' : ''}`}
                onMouseEnter={() => setFocus(i)}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => add(o)}
                title={o}
                data-testid={`kg-pick-${o}`}
              >
                {fmt(o)}
              </button>
            ))}
          </div>
        )}
        {open && query && results.length === 0 && (
          <div className="kg-picker-menu">
            <div className="kg-picker-empty">No match</div>
          </div>
        )}
      </div>
      {/* Selected pills BELOW the search bar (per 2026-05-31 user request) */}
      {selected.length > 0 && (
        <div className="kg-picker-pills">
          {selected.map((o) => {
            const isDeprecated = deprecatedSet.has(o);
            return (
              <span
                key={o}
                className={`kg-picker-pill${isDeprecated ? ' is-deprecated' : ''}`}
                title={o}
                data-testid={`kg-picked-${o}`}
              >
                <span className="lbl">{fmt(o)}</span>
                {isDeprecated && (
                  <span className="kg-deprecated-badge">deprecated</span>
                )}
                <button
                  type="button"
                  onClick={() => remove(o)}
                  aria-label={`Remove ${fmt(o)}`}
                >
                  <Icon name="x" size={10} />
                </button>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
