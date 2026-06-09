/**
 * GraphTab — knowledge graph tab.
 *
 * Ported from Desktop/UI/graph.jsx. Read-only teaser of the LightRAG-extracted
 * entities + relations. SVG layout uses precomputed `x`, `y` from fixtures
 * (no in-browser force simulation). Pan/zoom is a vanilla SVG transform.
 *
 * Behavior delta vs the proto:
 *   - Entities, relations, colors injected via props.
 *   - `onNavigate(tab, params)` instead of direct window.history mutation.
 *   - Wheel/pan use refs to avoid re-attaching listeners on each render.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from './Icon';
import {
  useUrlArrayParam,
  useUrlParam,
} from '../hooks/useUrlParam';
import {
  GRAPH_TYPE_COLORS,
  GRAPH_TYPE_LABEL,
  type GraphEntity,
  type GraphEntityPatch,
  type GraphEntityType,
  type GraphRelation,
  type GraphRelationPatch,
} from '../types/graph';
// Mock-kill F3 — the legacy `GRAPH_ENTITY_TAGS` / `GRAPH_ENTITY_DOCS`
// fixtures were keyed by prototype entity ids (`e_oracle`, `e_rman`…)
// and always returned `[]` for real Memgraph entities (hashed ids),
// showing misleading "0 tags · 0 sources" on every detail panel. The
// fixture maps were removed; tag data now reads `entity.tags` (an
// optional property already in the GraphEntity contract). Source-id
// lookup is dropped entirely until graph_reader.py exposes
// `source_doc_ids` per entity.
import {
  useCreateGraphEntity,
  useCreateGraphRelation,
  useDeleteGraphEntity,
  useDeleteGraphRelation,
  useUpdateGraphEntity,
  useUpdateGraphRelation,
} from '../api/queries';

const tagsOf = (e: GraphEntity): readonly string[] => e.tags ?? [];

const TYPE_KEYS = Object.keys(GRAPH_TYPE_LABEL) as readonly GraphEntityType[];
const PINNED_STORAGE_KEY = 'twin.kg.pinned.v1';

const readPinnedEntityIds = (): readonly string[] => {
  if (typeof window === 'undefined') return [];
  try {
    const parsed = JSON.parse(
      window.localStorage.getItem(PINNED_STORAGE_KEY) ?? '[]',
    );
    return Array.isArray(parsed)
      ? parsed.filter((id): id is string => typeof id === 'string')
      : [];
  } catch {
    return [];
  }
};

export interface GraphTabProps {
  entities: readonly GraphEntity[];
  relations: readonly GraphRelation[];
  /** Optional color override; defaults to the package palette. */
  colors?: Record<GraphEntityType, string>;
  /** Host-controlled tab navigation. */
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

export function GraphTab({
  entities,
  relations,
  colors = GRAPH_TYPE_COLORS,
  onNavigate,
}: GraphTabProps) {
  const [q, setQ] = useUrlParam<string>('gq', '');
  const [activeTypes, setActiveTypes] = useUrlArrayParam(
    'gtype',
    TYPE_KEYS,
  );
  const [tagFilter, setTagFilter] = useUrlArrayParam('gtag', []);
  const [docFilter, setDocFilter] = useUrlArrayParam('gdoc', []);
  const [addOpen, setAddOpen] = useState(false);
  const createEntity = useCreateGraphEntity();

  const allTags = useMemo(() => {
    const s = new Set<string>();
    entities.forEach((e) => tagsOf(e).forEach((t) => s.add(t)));
    return Array.from(s).sort();
  }, [entities]);
  const allSourceDocs = useMemo(() => {
    const s = new Set<string>();
    entities.forEach((e) => {
      (e.source_docs ?? []).forEach((doc) => s.add(doc));
    });
    return Array.from(s).sort();
  }, [entities]);
  const [selectedId, setSelectedId] = useUrlParam<string>(
    'gent',
    entities[0]?.id ?? '',
  );
  const [selectedRelId, setSelectedRelId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [pinnedIds, setPinnedIds] = useState<readonly string[]>(() =>
    readPinnedEntityIds(),
  );
  const canvasRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(
    null,
  );

  const nonTypeFiltered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return entities.filter((e) => {
      if (
        tagFilter.length > 0 &&
        !tagsOf(e).some((t) => tagFilter.includes(t))
      )
        return false;
      if (
        docFilter.length > 0 &&
        !(e.source_docs ?? []).some((doc) => docFilter.includes(doc))
      )
        return false;
      if (!needle) return true;
      return (
        e.name.toLowerCase().includes(needle) ||
        e.summary.toLowerCase().includes(needle)
      );
    });
  }, [entities, q, tagFilter, docFilter]);

  const typeCounts = useMemo(() => {
    const c: Partial<Record<GraphEntityType, number>> = {};
    nonTypeFiltered.forEach((e) => {
      c[e.type] = (c[e.type] ?? 0) + 1;
    });
    return c;
  }, [nonTypeFiltered]);

  const matches = useMemo(() => {
    return nonTypeFiltered.filter((e) => {
      if (!activeTypes.includes(e.type)) return false;
      return true;
    });
  }, [nonTypeFiltered, activeTypes]);

  const visibleIds = useMemo(() => new Set(matches.map((e) => e.id)), [matches]);
  const visibleRels = useMemo(
    () =>
      relations.filter(
        (r) => visibleIds.has(r.source) && visibleIds.has(r.target),
      ),
    [relations, visibleIds],
  );

  // Neutral state doctrine: only auto-pick the first entity when the
  // user has not yet made an explicit selection (selectedId === '').
  // Once a node has been chosen, a missing match (entity deleted,
  // refetch dropped it) must surface the empty inspector — falling
  // back to entities[0] hides the cascade and makes deletes look like
  // no-ops.
  const selected = !selectedId
    ? (entities[0] ?? null)
    : (entities.find((e) => e.id === selectedId) ?? null);
  const neighbors = useMemo(() => {
    if (!selected) return { rels: [] as GraphRelation[], nodes: [] as GraphEntity[] };
    const rels = relations.filter(
      (r) => r.source === selected.id || r.target === selected.id,
    );
    const nodeIds = new Set<string>();
    rels.forEach((r) => {
      nodeIds.add(r.source);
      nodeIds.add(r.target);
    });
    nodeIds.delete(selected.id);
    return { rels, nodes: entities.filter((e) => nodeIds.has(e.id)) };
  }, [selected, entities, relations]);
  const highlightIds = useMemo(() => {
    const ids = new Set<string>(selected ? [selected.id] : []);
    neighbors.nodes.forEach((n) => ids.add(n.id));
    return ids;
  }, [selected, neighbors]);

  useEffect(() => {
    window.localStorage.setItem(PINNED_STORAGE_KEY, JSON.stringify(pinnedIds));
  }, [pinnedIds]);

  const togglePinned = (id: string) => {
    setPinnedIds((current) =>
      current.includes(id)
        ? current.filter((pinnedId) => pinnedId !== id)
        : [...current, id],
    );
  };

  const toggleType = (t: GraphEntityType) => {
    if (activeTypes.includes(t)) {
      setActiveTypes(activeTypes.filter((x) => x !== t));
    } else {
      setActiveTypes([...activeTypes, t]);
    }
  };

  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    dragRef.current = {
      x: e.clientX,
      y: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  };
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      setPan({
        x: dragRef.current.panX + (e.clientX - dragRef.current.x),
        y: dragRef.current.panY + (e.clientY - dragRef.current.y),
      });
    };
    const onUp = () => {
      dragRef.current = null;
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);
  useEffect(() => {
    const node = canvasRef.current;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const dz = e.deltaY < 0 ? 1.1 : 0.9;
      setZoom((z) => Math.max(0.4, Math.min(3, z * dz)));
    };
    node.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      node.removeEventListener('wheel', onWheel);
    };
  }, []);

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="kg">
      <div className="kg-header">
        <div>
          <h1>Knowledge Graph</h1>
          <div className="kg-sub">
            <span>
              {entities.length} entities · {relations.length} relations · folder{' '}
              <code>cib</code>
            </span>
            <span className="dot-sep">·</span>
            <span
              className="kg-tier-note"
              title="Read-only view of LightRAG entity extraction. :MENTIONED_IN traversal + tag-filtered graph reasoning are Twin Graph tier features."
            >
              <Icon name="info-circle" size={11} /> read-only · Twin Graph tier for
              traversal
            </span>
          </div>
        </div>
        <div className="kg-header-actions">
          <div className="kg-search">
            <Icon name="search" size={12} color="var(--color-text-tertiary)" />
            <input
              type="text"
              placeholder="Search entities…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              aria-label="Search entities"
            />
            {q && (
              <button
                className="kg-search-clear"
                onClick={() => setQ('')}
                aria-label="Clear search"
              >
                <Icon name="x" size={11} />
              </button>
            )}
          </div>
          <button
            className="ghost-btn primary"
            onClick={() => setAddOpen(true)}
            type="button"
            data-testid="kg-add-entity-btn"
          >
            <Icon name="plus" size={12} /> Add entity
          </button>
          <button className="ghost-btn" onClick={resetView} title="Reset view">
            <Icon name="refresh" size={12} /> Reset view
          </button>
        </div>
      </div>

      {addOpen && (
        <AddEntityForm
          colors={colors}
          existingNames={entities.map((e) => e.name)}
          pending={createEntity.isPending}
          onCancel={() => setAddOpen(false)}
          onSubmit={(payload) => {
            createEntity.mutate(payload, {
              onSuccess: (created) => {
                setAddOpen(false);
                setSelectedId(created.id);
              },
            });
          }}
        />
      )}

      <div className="kg-body">
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
            selected={tagFilter}
            onChange={setTagFilter}
            placeholder="Search tags…"
          />
          <FilterPicker
            label="Filter by document"
            options={allSourceDocs}
            selected={docFilter}
            onChange={setDocFilter}
            placeholder="Search documents…"
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

        <div
          ref={canvasRef}
          className="kg-canvas"
          onMouseDown={onMouseDown}
          data-testid="kg-canvas"
          style={{ touchAction: 'none', overscrollBehavior: 'none' }}
        >
          <svg
            viewBox="0 0 1000 680"
            preserveAspectRatio="xMidYMid meet"
            className="kg-svg"
          >
            <defs>
              <marker
                id="kg-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M0,0 L10,5 L0,10 z" fill="var(--color-text-tertiary)" opacity="0.55" />
              </marker>
              <marker
                id="kg-arrow-hi"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M0,0 L10,5 L0,10 z" fill="var(--twin-accent)" />
              </marker>
            </defs>
            <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
              {/* Edges first so nodes sit on top */}
              {visibleRels.map((r) => {
                const s = entities.find((e) => e.id === r.source);
                const t = entities.find((e) => e.id === r.target);
                if (!s || !t) return null;
                const hi =
                  selected && (r.source === selected.id || r.target === selected.id);
                const dim = selected && !hi;
                const strong = r.strength >= 0.75;
                return (
                  <g
                    key={r.id}
                    className={`kg-edge${hi ? ' is-hi' : ''}${dim ? ' is-dim' : ''}`}
                  >
                    <line
                      x1={s.x}
                      y1={s.y}
                      x2={t.x}
                      y2={t.y}
                      stroke={hi ? 'var(--twin-accent)' : 'currentColor'}
                      strokeWidth={hi ? 1.6 : strong ? 1.1 : 0.7}
                      strokeOpacity={hi ? 0.9 : dim ? 0.08 : 0.32}
                      markerEnd={hi ? 'url(#kg-arrow-hi)' : 'url(#kg-arrow)'}
                    />
                    {hi && (
                      <text
                        x={(s.x + t.x) / 2}
                        y={(s.y + t.y) / 2 - 4}
                        textAnchor="middle"
                        className="kg-edge-label"
                      >
                        {r.label}
                      </text>
                    )}
                  </g>
                );
              })}
              {/* Nodes */}
              {matches.map((e) => {
                const radius = 8 + Math.min(18, Math.sqrt(e.mentions) * 0.9);
                const isSelected = !!selected && selected.id === e.id;
                const isNeighbor = highlightIds.has(e.id) && !isSelected;
                const isDim = !!selected && !highlightIds.has(e.id);
                const isHover = hoverId === e.id;
                return (
                  <g
                    key={e.id}
                    className={`kg-node${isSelected ? ' is-selected' : ''}${isDim ? ' is-dim' : ''}`}
                    transform={`translate(${e.x}, ${e.y})`}
                    role="button"
                    tabIndex={0}
                    aria-label={`Select entity ${e.name}`}
                    aria-pressed={isSelected}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setSelectedId(e.id);
                    }}
                    onKeyDown={(ev) => {
                      if (ev.key === 'Enter' || ev.key === ' ') {
                        ev.preventDefault();
                        ev.stopPropagation();
                        setSelectedId(e.id);
                      }
                    }}
                    onMouseEnter={() => setHoverId(e.id)}
                    onMouseLeave={() => setHoverId(null)}
                    style={{ cursor: 'pointer', opacity: isDim ? 0.35 : 1 }}
                    data-testid={`kg-node-${e.id}`}
                  >
                    {isSelected && <circle r={radius + 7} className="kg-node-halo" />}
                    {isNeighbor && (
                      <circle r={radius + 4} className="kg-node-halo subtle" />
                    )}
                    <circle
                      r={radius}
                      fill={colors[e.type]}
                      stroke={
                        isSelected
                          ? 'var(--twin-accent)'
                          : 'var(--color-background-primary)'
                      }
                      strokeWidth={isSelected ? 2.5 : 1.5}
                    />
                    <text
                      y={radius + 12}
                      textAnchor="middle"
                      className={`kg-node-label${isSelected ? ' is-selected' : ''}`}
                      style={{ fontWeight: isSelected || isHover ? 600 : 500 }}
                    >
                      {e.name}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
          {matches.length === 0 && (
            <div className="kg-empty">
              <Icon name="search" size={20} color="var(--color-text-tertiary)" />
              <div>No entities match the current filter.</div>
              <button
                className="ghost-btn"
                onClick={() => {
                  setQ('');
                  setActiveTypes(TYPE_KEYS);
                }}
              >
                Clear filter
              </button>
            </div>
          )}
          <div className="kg-zoom-pill">
            <button
              onClick={() => setZoom((z) => Math.max(0.4, z * 0.85))}
              aria-label="Zoom out"
            >
              <Icon name="minus" size={11} />
            </button>
            <span data-testid="kg-zoom-value">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(3, z * 1.18))}
              aria-label="Zoom in"
            >
              <Icon name="plus" size={11} />
            </button>
          </div>
        </div>

        <GraphDetailPanel
          entity={selected}
          selectedRel={
            selectedRelId
              ? (relations.find((r) => r.id === selectedRelId) ?? null)
              : null
          }
          neighbors={neighbors}
          colors={colors}
          entities={entities}
          relations={relations}
          typeLabels={GRAPH_TYPE_LABEL}
          onSelect={(id) => {
            setSelectedRelId(null);
            setSelectedId(id);
          }}
          onSelectRelation={(id) => setSelectedRelId(id)}
          onClearRelation={() => setSelectedRelId(null)}
          pinnedIds={pinnedIds}
          onTogglePinned={togglePinned}
          onNavigate={onNavigate}
        />
      </div>
    </div>
  );
}

interface GraphDetailPanelProps {
  entity: GraphEntity | null;
  selectedRel: GraphRelation | null;
  neighbors: { rels: GraphRelation[]; nodes: GraphEntity[] };
  colors: Record<GraphEntityType, string>;
  entities: readonly GraphEntity[];
  relations: readonly GraphRelation[];
  typeLabels: Record<GraphEntityType, string>;
  onSelect: (id: string) => void;
  onSelectRelation: (id: string) => void;
  onClearRelation: () => void;
  pinnedIds: readonly string[];
  onTogglePinned: (id: string) => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

function GraphDetailPanel({
  entity,
  selectedRel,
  neighbors,
  colors,
  entities,
  relations,
  typeLabels,
  onSelect,
  onSelectRelation,
  onClearRelation,
  pinnedIds,
  onTogglePinned,
  onNavigate,
}: GraphDetailPanelProps) {
  // Relation editor takes priority when an edge is selected.
  if (selectedRel) {
    const src = entities.find((n) => n.id === selectedRel.source) ?? null;
    const tgt = entities.find((n) => n.id === selectedRel.target) ?? null;
    return (
      <aside className="kg-detail" data-testid="kg-detail-relation">
        <RelationEditor
          rel={selectedRel}
          src={src}
          tgt={tgt}
          colors={colors}
          onSelectNode={(id) => {
            onClearRelation();
            onSelect(id);
          }}
          onBack={onClearRelation}
        />
      </aside>
    );
  }

  if (!entity) {
    return (
      <aside className="kg-detail">
        <div className="kg-detail-empty">
          <Icon name="circle-dot" size={20} color="var(--color-text-tertiary)" />
          <div>Select a node to inspect</div>
        </div>
      </aside>
    );
  }
  return (
    <aside className="kg-detail" data-testid="kg-detail-entity">
      <EntityEditor
        entity={entity}
        neighbors={neighbors}
        entities={entities}
        relations={relations}
        colors={colors}
        typeLabels={typeLabels}
        onSelectRelation={onSelectRelation}
        isPinned={pinnedIds.includes(entity.id)}
        onTogglePinned={() => onTogglePinned(entity.id)}
        onNavigate={onNavigate}
      />
    </aside>
  );
}

// ─── Entity editor — view + edit name, type, summary, tags, properties ──
interface EntityEditorProps {
  entity: GraphEntity;
  neighbors: { rels: GraphRelation[]; nodes: GraphEntity[] };
  /** Full entity list — used as the target picker source when drawing
   *  a new outgoing relation. The current entity is filtered out. */
  entities: readonly GraphEntity[];
  /** Full relation list — used to dedupe a new outgoing relation
   *  against an existing edge between the same endpoints. */
  relations: readonly GraphRelation[];
  colors: Record<GraphEntityType, string>;
  typeLabels: Record<GraphEntityType, string>;
  onSelectRelation: (id: string) => void;
  isPinned: boolean;
  onTogglePinned: () => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

interface EntityDraft {
  name: string;
  type: GraphEntityType;
  summary: string;
  tags: string[];
  properties: Record<string, string>;
}

function EntityEditor({
  entity,
  neighbors,
  entities,
  relations,
  colors,
  typeLabels,
  onSelectRelation,
  isPinned,
  onTogglePinned,
  onNavigate,
}: EntityEditorProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<EntityDraft | null>(null);
  const updateEntity = useUpdateGraphEntity();
  const deleteEntity = useDeleteGraphEntity();
  const createRelation = useCreateGraphRelation();
  const [addRelOpen, setAddRelOpen] = useState(false);
  // Two-step destructive confirmation. First click arms the action,
  // second click within the timeout fires the mutation. Reset on
  // entity change so navigating away cancels.
  const [armedDelete, setArmedDelete] = useState(false);
  useEffect(() => {
    if (!armedDelete) return;
    const t = window.setTimeout(() => setArmedDelete(false), 4000);
    return () => window.clearTimeout(t);
  }, [armedDelete]);

  // Reset edit mode + armed-delete + Add relation form when switching
  // entities — every transient panel should start fresh on the next node.
  /* eslint-disable react-hooks/set-state-in-effect -- intentional reset on prop change; refactoring to a key prop would shed unrelated state. */
  useEffect(() => {
    setEditing(false);
    setDraft(null);
    setArmedDelete(false);
    setAddRelOpen(false);
  }, [entity.id]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const startEdit = () => {
    setDraft({
      name: entity.name,
      type: entity.type,
      summary: entity.summary || '',
      tags: [...(entity.tags ?? [])],
      properties: { ...(entity.properties ?? {}) },
    });
    setEditing(true);
  };
  const cancel = () => {
    setEditing(false);
    setDraft(null);
  };
  const commit = () => {
    if (!draft) return;
    const patch: GraphEntityPatch = {
      name: draft.name.trim() || entity.name,
      type: draft.type,
      summary: draft.summary,
      tags: draft.tags,
      properties: draft.properties,
    };
    updateEntity.mutate({ id: entity.id, patch });
    setEditing(false);
    setDraft(null);
  };

  const incoming = neighbors.rels.filter((r) => r.target === entity.id);
  const outgoing = neighbors.rels.filter((r) => r.source === entity.id);
  const propEntries = Object.entries(entity.properties ?? {});

  return (
    <>
      <div className="kg-detail-h">
        <div className="kg-detail-title">
          <span
            className="kg-detail-swatch"
            style={{ background: colors[entity.type] }}
          />
          {editing && draft ? (
            <input
              className="kg-edit-input kg-edit-name"
              value={draft.name}
              onChange={(e) =>
                setDraft((d) => (d ? { ...d, name: e.target.value } : d))
              }
              placeholder="Name"
              aria-label="Entity name"
              autoFocus
            />
          ) : (
            <h2>{entity.name}</h2>
          )}
          {!editing && (
            <div className="kg-detail-title-actions">
              <button
                className={`ghost-btn small${isPinned ? ' primary' : ''}`}
                onClick={onTogglePinned}
                title={isPinned ? 'Unpin entity' : 'Pin entity'}
                data-testid="kg-entity-pin"
                aria-pressed={isPinned}
              >
                <Icon name="pin" size={11} /> {isPinned ? 'Pinned' : 'Pin'}
              </button>
              <button
                className="ghost-btn small"
                onClick={startEdit}
                title="Edit metadata"
                data-testid="kg-entity-edit"
              >
                <Icon name="edit" size={11} /> Edit
              </button>
            </div>
          )}
        </div>
        {editing && draft ? (
          <div className="kg-edit-row" style={{ marginTop: 6 }}>
            <span className="muted-sm" style={{ marginRight: 6 }}>
              Type
            </span>
            <select
              className="kg-edit-select"
              value={draft.type}
              onChange={(e) =>
                setDraft((d) =>
                  d ? { ...d, type: e.target.value as GraphEntityType } : d,
                )
              }
              aria-label="Entity type"
            >
              {(Object.keys(typeLabels) as GraphEntityType[]).map((t) => (
                <option key={t} value={t}>
                  {typeLabels[t]}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="kg-detail-type" style={{ color: colors[entity.type] }}>
            {typeLabels[entity.type]}
          </div>
        )}
        {editing && draft ? (
          <textarea
            className="kg-edit-input kg-edit-summary"
            rows={3}
            value={draft.summary}
            onChange={(e) =>
              setDraft((d) => (d ? { ...d, summary: e.target.value } : d))
            }
            placeholder="Short description"
            aria-label="Entity summary"
            style={{ marginTop: 6, width: '100%' }}
          />
        ) : (
          <p className="kg-detail-summary">{entity.summary || '—'}</p>
        )}
        {!editing && (
          <div className="kg-detail-stats">
            <div>
              <span className="kg-stat-n">{entity.mentions}</span>
              <span className="kg-stat-l">mentions</span>
            </div>
            <div>
              <span className="kg-stat-n">{entity.sources}</span>
              <span className="kg-stat-l">sources</span>
            </div>
            <div>
              <span className="kg-stat-n">{neighbors.rels.length}</span>
              <span className="kg-stat-l">relations</span>
            </div>
          </div>
        )}
      </div>

      {/* Tags — node-attribute strings, decoupled from the WebuiTag taxonomy. */}
      <div className="kg-detail-section">
        <div className="section-label">
          <span>
            Tags{' '}
            {editing ? <em>— edit</em> : <em>— node attributes</em>}
          </span>
        </div>
        {editing && draft ? (
          <TagAttrEditor
            tags={draft.tags}
            onChange={(tags) =>
              setDraft((d) => (d ? { ...d, tags } : d))
            }
          />
        ) : (entity.tags ?? []).length > 0 ? (
          <div className="tag-chips">
            {(entity.tags ?? []).map((t) => (
              <span key={t} className="tag-chip">
                {t}
              </span>
            ))}
          </div>
        ) : (
          <div className="muted-sm">No tags.</div>
        )}
      </div>

      {/* Properties — custom k/v metadata. */}
      <div className="kg-detail-section">
        <div className="section-label">
          <span>
            Properties{' '}
            {editing ? (
              <em>— add / remove</em>
            ) : (
              <em>— custom metadata</em>
            )}
          </span>
          {!editing && propEntries.length > 0 && (
            <span className="kg-prop-count">{propEntries.length}</span>
          )}
        </div>
        {editing && draft ? (
          <PropEditor
            properties={draft.properties}
            onChange={(properties) =>
              setDraft((d) => (d ? { ...d, properties } : d))
            }
          />
        ) : propEntries.length === 0 ? (
          <div className="muted-sm">
            No custom properties.{' '}
            <button
              className="kg-inline-add"
              onClick={startEdit}
              type="button"
            >
              + Add some
            </button>
          </div>
        ) : (
          <dl className="kg-prop-list">
            {propEntries.map(([k, v]) => (
              <div key={k} className="kg-prop-row">
                <dt>{k}</dt>
                <dd>{String(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>

      {editing && (
        <div className="kg-detail-section kg-edit-actions">
          <button className="ghost-btn" onClick={cancel} type="button">
            Cancel
          </button>
          <button
            className="ghost-btn primary"
            onClick={commit}
            disabled={updateEntity.isPending}
            type="button"
            data-testid="kg-entity-save"
          >
            <Icon name="check" size={11} />{' '}
            {updateEntity.isPending ? 'Saving…' : 'Save'}
          </button>
        </div>
      )}

      {!editing && (
        <>
          <div className="kg-detail-section">
            <div
              className="section-label"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 8,
              }}
            >
              <span>
                Outgoing ({outgoing.length}) <em>— click to edit</em>
              </span>
              {!addRelOpen && (
                <button
                  type="button"
                  className="ghost-btn small"
                  onClick={() => setAddRelOpen(true)}
                  data-testid="kg-add-rel-btn"
                >
                  <Icon name="plus" size={11} /> Add relation
                </button>
              )}
            </div>
            {addRelOpen && (
              <AddRelationForm
                source={entity}
                entities={entities}
                relations={relations}
                colors={colors}
                pending={createRelation.isPending}
                onCancel={() => setAddRelOpen(false)}
                onSubmit={(payload) => {
                  createRelation.mutate(payload, {
                    onSuccess: () => setAddRelOpen(false),
                  });
                }}
              />
            )}
            {outgoing.length === 0 ? (
              <div className="muted-sm">No outgoing relations.</div>
            ) : (
              <ul className="kg-rel-list">
                {outgoing.map((r) => {
                  const t = neighbors.nodes.find((n) => n.id === r.target);
                  if (!t) return null;
                  return (
                    <li
                      key={r.id}
                      className="kg-rel-row"
                      onClick={() => onSelectRelation(r.id)}
                      onKeyDown={(ev) => {
                        if (ev.key === 'Enter' || ev.key === ' ') {
                          ev.preventDefault();
                          onSelectRelation(r.id);
                        }
                      }}
                      role="button"
                      tabIndex={0}
                      data-testid={`kg-rel-row-${r.id}`}
                    >
                      <span className="kg-rel-arrow">→</span>
                      <code className="kg-rel-label">{r.label}</code>
                      <span className="kg-rel-target">
                        <span
                          className="kg-rel-swatch"
                          style={{ background: colors[t.type] }}
                        />
                        <span className="kg-rel-target-name">{t.name}</span>
                      </span>
                      <span
                        className="kg-rel-strength"
                        title={`strength ${r.strength.toFixed(2)}`}
                      >
                        {Math.round(r.strength * 100)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="kg-detail-section">
            <div className="section-label">
              <span>
                Incoming ({incoming.length}) <em>— click to edit</em>
              </span>
            </div>
            {incoming.length === 0 ? (
              <div className="muted-sm">No incoming relations.</div>
            ) : (
              <ul className="kg-rel-list">
                {incoming.map((r) => {
                  const s = neighbors.nodes.find((n) => n.id === r.source);
                  if (!s) return null;
                  return (
                    <li
                      key={r.id}
                      className="kg-rel-row"
                      onClick={() => onSelectRelation(r.id)}
                      onKeyDown={(ev) => {
                        if (ev.key === 'Enter' || ev.key === ' ') {
                          ev.preventDefault();
                          onSelectRelation(r.id);
                        }
                      }}
                      role="button"
                      tabIndex={0}
                      data-testid={`kg-rel-row-${r.id}`}
                    >
                      <span className="kg-rel-target">
                        <span
                          className="kg-rel-swatch"
                          style={{ background: colors[s.type] }}
                        />
                        <span className="kg-rel-target-name">{s.name}</span>
                      </span>
                      <code className="kg-rel-label">{r.label}</code>
                      <span className="kg-rel-arrow">→</span>
                      <span
                        className="kg-rel-strength"
                        title={`strength ${r.strength.toFixed(2)}`}
                      >
                        {Math.round(r.strength * 100)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="kg-detail-section kg-detail-cta">
            <button
              className="ghost-btn"
              onClick={() => {
                // Mock-kill F3 — navigation falls back to a text query
                // on the entity name; the previous per-entity source
                // list came from a fixture map keyed by prototype ids.
                onNavigate?.('documents', { q: entity.name });
              }}
              type="button"
            >
              <Icon name="external-link" size={11} /> View documents
              mentioning this entity
            </button>
            <div className="kg-detail-locked">
              <Icon name="lock" size={11} />
              <span>
                Traverse relations with tag filter — <b>Twin Graph</b>
              </span>
            </div>
          </div>

          <div
            className="kg-detail-section kg-lifecycle"
            data-testid="kg-entity-lifecycle"
            style={{ borderTop: '1px solid var(--color-border, #e2e6ec)' }}
          >
            <button
              type="button"
              className={armedDelete ? 'ghost-btn danger' : 'ghost-btn'}
              onClick={() => {
                if (!armedDelete) {
                  setArmedDelete(true);
                  return;
                }
                deleteEntity.mutate(entity.id);
                setArmedDelete(false);
              }}
              disabled={deleteEntity.isPending}
              data-testid="kg-entity-delete"
              style={
                armedDelete
                  ? {
                      color: 'var(--twin-red-vivid, #b03060)',
                      borderColor: 'var(--twin-red-vivid, #b03060)',
                    }
                  : undefined
              }
            >
              <Icon name={armedDelete ? 'alert-triangle' : 'trash'} size={11} />{' '}
              {deleteEntity.isPending
                ? 'Deleting…'
                : armedDelete
                  ? 'Click again to confirm'
                  : 'Delete entity'}
            </button>
            {armedDelete && (
              <button
                type="button"
                className="ghost-btn small"
                onClick={() => setArmedDelete(false)}
                data-testid="kg-entity-delete-cancel"
              >
                Cancel
              </button>
            )}
          </div>
        </>
      )}
    </>
  );
}

// ─── Relation editor — label, strength, custom properties ──────────────
interface RelationEditorProps {
  rel: GraphRelation;
  src: GraphEntity | null;
  tgt: GraphEntity | null;
  colors: Record<GraphEntityType, string>;
  onSelectNode: (id: string) => void;
  onBack: () => void;
}

interface RelationDraft {
  label: string;
  strength: number;
  properties: Record<string, string>;
}

function RelationEditor({
  rel,
  src,
  tgt,
  colors,
  onSelectNode,
  onBack,
}: RelationEditorProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<RelationDraft | null>(null);
  const updateRelation = useUpdateGraphRelation();
  const deleteRelation = useDeleteGraphRelation();
  const [armedDelete, setArmedDelete] = useState(false);
  useEffect(() => {
    if (!armedDelete) return;
    const t = window.setTimeout(() => setArmedDelete(false), 4000);
    return () => window.clearTimeout(t);
  }, [armedDelete]);

  /* eslint-disable react-hooks/set-state-in-effect -- intentional reset of the relation editor panel when switching to a different edge. */
  useEffect(() => {
    setEditing(false);
    setDraft(null);
    setArmedDelete(false);
  }, [rel.id]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const startEdit = () => {
    setDraft({
      label: rel.label,
      strength: rel.strength,
      properties: { ...(rel.properties ?? {}) },
    });
    setEditing(true);
  };
  const cancel = () => {
    setEditing(false);
    setDraft(null);
  };
  const commit = () => {
    if (!draft) return;
    const cleaned = draft.label.trim().toUpperCase().replace(/\s+/g, '_');
    const patch: GraphRelationPatch = {
      label: cleaned || rel.label,
      strength: Math.max(0, Math.min(1, draft.strength)),
      properties: draft.properties,
    };
    updateRelation.mutate({ id: rel.id, patch });
    setEditing(false);
    setDraft(null);
  };

  const propEntries = Object.entries(rel.properties ?? {});

  return (
    <>
      <div className="kg-detail-h">
        <button
          className="ghost-btn small"
          onClick={onBack}
          title="Back to entity"
          type="button"
          data-testid="kg-rel-back"
        >
          <Icon name="chevron-left" size={11} /> Back
        </button>
        <div className="kg-detail-title" style={{ marginTop: 4 }}>
          {editing && draft ? (
            <input
              className="kg-edit-input kg-edit-name"
              value={draft.label}
              onChange={(e) =>
                setDraft((d) => (d ? { ...d, label: e.target.value } : d))
              }
              placeholder="RELATION_LABEL"
              aria-label="Relation label"
              autoFocus
              style={{
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
              }}
            />
          ) : (
            <h2 style={{ fontFamily: 'var(--font-mono)', fontSize: 14 }}>
              {rel.label}
            </h2>
          )}
          {!editing && (
            <button
              className="ghost-btn small"
              onClick={startEdit}
              style={{ marginLeft: 'auto' }}
              type="button"
              data-testid="kg-rel-edit"
            >
              <Icon name="edit" size={11} /> Edit
            </button>
          )}
        </div>
        <div className="kg-detail-type" style={{ marginTop: 2 }}>
          Relation
        </div>
        <div className="kg-rel-endpoints" style={{ marginTop: 6 }}>
          <button
            className="kg-rel-endpoint"
            onClick={() => src && onSelectNode(src.id)}
            type="button"
            disabled={!src}
          >
            <span
              className="kg-rel-swatch"
              style={{ background: src ? colors[src.type] : '#888' }}
            />
            {src ? src.name : '?'}
          </button>
          <span className="kg-rel-arrow">→</span>
          <button
            className="kg-rel-endpoint"
            onClick={() => tgt && onSelectNode(tgt.id)}
            type="button"
            disabled={!tgt}
          >
            <span
              className="kg-rel-swatch"
              style={{ background: tgt ? colors[tgt.type] : '#888' }}
            />
            {tgt ? tgt.name : '?'}
          </button>
        </div>
      </div>

      <div className="kg-detail-section">
        <div className="section-label">
          <span>
            Strength{' '}
            {editing ? <em>— 0.00–1.00</em> : null}
          </span>
        </div>
        {editing && draft ? (
          <div className="kg-strength-edit">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={draft.strength}
              onChange={(e) =>
                setDraft((d) =>
                  d ? { ...d, strength: parseFloat(e.target.value) } : d,
                )
              }
              aria-label="Relation strength"
            />
            <code>{draft.strength.toFixed(2)}</code>
          </div>
        ) : (
          <div className="kg-strength-view">
            <div className="kg-strength-bar">
              <div style={{ width: `${rel.strength * 100}%` }} />
            </div>
            <code>{Math.round(rel.strength * 100)}%</code>
          </div>
        )}
      </div>

      <div className="kg-detail-section">
        <div className="section-label">
          <span>
            Properties{' '}
            {editing ? (
              <em>— add / remove</em>
            ) : (
              <em>— custom metadata</em>
            )}
          </span>
          {!editing && propEntries.length > 0 && (
            <span className="kg-prop-count">{propEntries.length}</span>
          )}
        </div>
        {editing && draft ? (
          <PropEditor
            properties={draft.properties}
            onChange={(properties) =>
              setDraft((d) => (d ? { ...d, properties } : d))
            }
          />
        ) : propEntries.length === 0 ? (
          <div className="muted-sm">
            No custom properties.{' '}
            <button
              className="kg-inline-add"
              onClick={startEdit}
              type="button"
            >
              + Add some
            </button>
          </div>
        ) : (
          <dl className="kg-prop-list">
            {propEntries.map(([k, v]) => (
              <div key={k} className="kg-prop-row">
                <dt>{k}</dt>
                <dd>{String(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>

      {editing && (
        <div className="kg-detail-section kg-edit-actions">
          <button className="ghost-btn" onClick={cancel} type="button">
            Cancel
          </button>
          <button
            className="ghost-btn primary"
            onClick={commit}
            disabled={updateRelation.isPending}
            type="button"
            data-testid="kg-rel-save"
          >
            <Icon name="check" size={11} />{' '}
            {updateRelation.isPending ? 'Saving…' : 'Save'}
          </button>
        </div>
      )}

      {!editing && (
        <div
          className="kg-detail-section kg-lifecycle"
          data-testid="kg-rel-lifecycle"
          style={{ borderTop: '1px solid var(--color-border, #e2e6ec)' }}
        >
          <button
            type="button"
            className={armedDelete ? 'ghost-btn danger' : 'ghost-btn'}
            onClick={() => {
              if (!armedDelete) {
                setArmedDelete(true);
                return;
              }
              deleteRelation.mutate(rel.id, {
                onSuccess: () => onBack(),
              });
              setArmedDelete(false);
            }}
            disabled={deleteRelation.isPending}
            data-testid="kg-rel-delete"
            style={
              armedDelete
                ? {
                    color: 'var(--twin-red-vivid, #b03060)',
                    borderColor: 'var(--twin-red-vivid, #b03060)',
                  }
                : undefined
            }
          >
            <Icon name={armedDelete ? 'alert-triangle' : 'trash'} size={11} />{' '}
            {deleteRelation.isPending
              ? 'Deleting…'
              : armedDelete
                ? 'Click again to confirm'
                : 'Delete relation'}
          </button>
          {armedDelete && (
            <button
              type="button"
              className="ghost-btn small"
              onClick={() => setArmedDelete(false)}
              data-testid="kg-rel-delete-cancel"
            >
              Cancel
            </button>
          )}
        </div>
      )}
    </>
  );
}

// ─── Tag chip editor (node attribute strings) ──────────────────────────
function TagAttrEditor({
  tags,
  onChange,
}: {
  tags: readonly string[];
  onChange: (next: string[]) => void;
}) {
  const [v, setV] = useState('');
  const add = () => {
    const t = v.trim().toLowerCase().replace(/\s+/g, '-');
    if (!t || tags.includes(t)) return;
    onChange([...tags, t]);
    setV('');
  };
  const remove = (t: string) => onChange(tags.filter((x) => x !== t));
  return (
    <div className="kg-tag-editor">
      <div className="tag-chips">
        {tags.map((t) => (
          <span key={t} className="tag-chip">
            {t}{' '}
            <button
              onClick={() => remove(t)}
              aria-label={`Remove ${t}`}
              type="button"
            >
              <Icon name="x" size={9} />
            </button>
          </span>
        ))}
      </div>
      <div className="kg-tag-add-row" style={{ marginTop: 6 }}>
        <input
          value={v}
          onChange={(e) => setV(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              add();
            }
          }}
          placeholder="Add tag…"
          aria-label="Add tag"
        />
        <button
          className="ghost-btn small"
          onClick={add}
          disabled={!v.trim()}
          type="button"
        >
          Add
        </button>
      </div>
    </div>
  );
}

// ─── Properties editor — k/v list with add / rename / remove ───────────
function PropEditor({
  properties,
  onChange,
}: {
  properties: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}) {
  const entries = Object.entries(properties);
  const [draftKey, setDraftKey] = useState('');
  const [draftVal, setDraftVal] = useState('');

  const editValue = (k: string, newVal: string) =>
    onChange({ ...properties, [k]: newVal });
  const renameKey = (oldK: string, newK: string) => {
    if (!newK || newK === oldK || properties[newK] !== undefined) return;
    const next: Record<string, string> = {};
    for (const [k, v] of Object.entries(properties)) {
      next[k === oldK ? newK : k] = v;
    }
    onChange(next);
  };
  const removeKey = (k: string) => {
    const next = { ...properties };
    delete next[k];
    onChange(next);
  };
  const addProp = () => {
    const k = draftKey.trim();
    if (!k || properties[k] !== undefined) return;
    onChange({ ...properties, [k]: draftVal });
    setDraftKey('');
    setDraftVal('');
  };

  return (
    <div className="kg-prop-editor">
      {entries.length === 0 && (
        <div className="muted-sm" style={{ marginBottom: 6 }}>
          No properties yet — add the first one below.
        </div>
      )}
      {entries.map(([k, v]) => (
        <div key={k} className="kg-prop-edit-row">
          <input
            className="kg-prop-key"
            value={k}
            onChange={(e) => renameKey(k, e.target.value.trim())}
            placeholder="key"
            aria-label={`Property key ${k}`}
          />
          <span className="kg-prop-sep">:</span>
          <input
            className="kg-prop-val"
            value={String(v)}
            onChange={(e) => editValue(k, e.target.value)}
            placeholder="value"
            aria-label={`Property value ${k}`}
          />
          <button
            className="kg-prop-x"
            onClick={() => removeKey(k)}
            aria-label={`Remove ${k}`}
            type="button"
          >
            <Icon name="x" size={10} />
          </button>
        </div>
      ))}
      <div className="kg-prop-add-row">
        <input
          className="kg-prop-key"
          value={draftKey}
          onChange={(e) => setDraftKey(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && draftKey.trim()) {
              e.preventDefault();
              addProp();
            }
          }}
          placeholder="new key"
          aria-label="New property key"
        />
        <span className="kg-prop-sep">:</span>
        <input
          className="kg-prop-val"
          value={draftVal}
          onChange={(e) => setDraftVal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && draftKey.trim()) {
              e.preventDefault();
              addProp();
            }
          }}
          placeholder="value"
          aria-label="New property value"
        />
        <button
          className="kg-prop-add ghost-btn small"
          onClick={addProp}
          disabled={!draftKey.trim() || properties[draftKey.trim()] !== undefined}
          type="button"
          data-testid="kg-prop-add"
        >
          <Icon name="plus" size={10} /> Add
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── FilterPicker (B4) ──
// Fuzzy-search picker scaled past chip-walls (200+ sources / 50+ tags).
// Layout (user request 2026-05-31): search input FIRST, removable pills
// BELOW the search bar — inverse of the design prototype, more intuitive
// for "search → click to add → see what's selected just below".

interface FilterPickerProps {
  label: string;
  options: readonly string[];
  selected: readonly string[];
  onChange: (next: string[]) => void;
  placeholder: string;
  format?: (x: string) => string;
}

function FilterPicker({
  label,
  options,
  selected,
  onChange,
  placeholder,
  format,
}: FilterPickerProps) {
  const fmt = format ?? ((x: string) => x);
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [focus, setFocus] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

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
    if (opt.indexOf(q) >= 0) return true;
    let i = 0;
    for (let k = 0; k < opt.length; k++) {
      if (opt[k] === q[i]) i++;
      if (i >= q.length) return true;
    }
    return false;
  };

  const avail = options.filter((o) => !selected.includes(o));
  const rank = (o: string) => {
    const q = query.toLowerCase().trim();
    const a = fmt(o).toLowerCase().indexOf(q);
    const b = o.toLowerCase().indexOf(q);
    return a >= 0 ? a : b >= 0 ? b : 999;
  };
  const results = (
    query
      ? avail
          .filter((o) => fuzzy(fmt(o), query) || fuzzy(o, query))
          .sort((a, b) => rank(a) - rank(b) || fmt(a).length - fmt(b).length)
      : avail
  ).slice(0, 8);

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
          aria-label={label}
          onFocus={() => setOpen(true)}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
            setFocus(0);
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setFocus((f) => Math.min(results.length - 1, f + 1));
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setFocus((f) => Math.max(0, f - 1));
            } else if (e.key === 'Enter' && results[focus]) {
              e.preventDefault();
              add(results[focus]);
            } else if (e.key === 'Escape') {
              setOpen(false);
            }
          }}
        />
        {open && results.length > 0 && (
          <div className="kg-picker-menu">
            {results.map((o, i) => (
              <button
                key={o}
                type="button"
                className={`kg-picker-opt${i === focus ? ' focus' : ''}`}
                onMouseEnter={() => setFocus(i)}
                onMouseDown={() => add(o)}
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
          {selected.map((o) => (
            <span
              key={o}
              className="kg-picker-pill"
              title={o}
              data-testid={`kg-picked-${o}`}
            >
              <span className="lbl">{fmt(o)}</span>
              <button
                type="button"
                onClick={() => remove(o)}
                aria-label={`Remove ${fmt(o)}`}
              >
                <Icon name="x" size={10} />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Lifecycle: Add entity inline form ────────────────────────────────
interface AddEntityFormProps {
  colors: Record<GraphEntityType, string>;
  existingNames: readonly string[];
  pending: boolean;
  onCancel: () => void;
  onSubmit: (payload: {
    name: string;
    type: GraphEntityType;
    summary?: string;
  }) => void;
}

function AddEntityForm({
  colors,
  existingNames,
  pending,
  onCancel,
  onSubmit,
}: AddEntityFormProps) {
  const [name, setName] = useState('');
  const [type, setType] = useState<GraphEntityType>('PRODUCT');
  const [summary, setSummary] = useState('');
  const trimmed = name.trim();
  const duplicate = trimmed.length > 0 && existingNames.includes(trimmed);
  const canSubmit = trimmed.length > 0 && !duplicate && !pending;

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      name: trimmed,
      type,
      summary: summary.trim() || undefined,
    });
  };

  return (
    <form
      className="kg-add-entity"
      data-testid="kg-add-entity-form"
      onSubmit={submit}
      style={{
        display: 'flex',
        gap: 10,
        flexWrap: 'wrap',
        alignItems: 'flex-end',
        padding: '10px 14px',
        margin: '8px 0',
        background: 'var(--color-surface-alt, #f5f7fa)',
        border: '1px solid var(--color-border, #e2e6ec)',
        borderRadius: 6,
      }}
    >
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Name
        </span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Oracle Database"
          aria-label="New entity name"
          autoFocus
          data-testid="kg-add-entity-name"
          style={{ minWidth: 200 }}
        />
      </label>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Type
        </span>
        <select
          value={type}
          onChange={(e) => setType(e.target.value as GraphEntityType)}
          aria-label="New entity type"
          data-testid="kg-add-entity-type"
        >
          {TYPE_KEYS.map((t) => (
            <option key={t} value={t}>
              {GRAPH_TYPE_LABEL[t]}
            </option>
          ))}
        </select>
      </label>
      <label
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          flex: '1 1 240px',
        }}
      >
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Summary <em style={{ opacity: 0.6 }}>(optional)</em>
        </span>
        <input
          type="text"
          value={summary}
          onChange={(e) => setSummary(e.target.value)}
          placeholder="What is this?"
          aria-label="New entity summary"
          data-testid="kg-add-entity-summary"
        />
      </label>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <span
          className="kg-type-swatch"
          style={{ background: colors[type], width: 14, height: 14 }}
          aria-hidden
        />
        <button type="button" className="ghost-btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="submit"
          className="ghost-btn primary"
          disabled={!canSubmit}
          data-testid="kg-add-entity-submit"
        >
          <Icon name="check" size={11} /> {pending ? 'Adding…' : 'Add'}
        </button>
      </div>
      {duplicate && (
        <div
          role="alert"
          style={{ fontSize: 11, color: 'var(--twin-red-vivid, #b03060)' }}
          data-testid="kg-add-entity-duplicate"
        >
          An entity named “{trimmed}” already exists.
        </div>
      )}
    </form>
  );
}

// ─── Lifecycle: Add outgoing relation inline form ─────────────────────
interface AddRelationFormProps {
  source: GraphEntity;
  entities: readonly GraphEntity[];
  relations: readonly GraphRelation[];
  colors: Record<GraphEntityType, string>;
  pending: boolean;
  onCancel: () => void;
  onSubmit: (payload: {
    source: string;
    target: string;
    label: string;
    strength: number;
  }) => void;
}

function AddRelationForm({
  source,
  entities,
  relations,
  colors,
  pending,
  onCancel,
  onSubmit,
}: AddRelationFormProps) {
  // Targets = every other entity in the graph. Sorted by name for a
  // predictable picker order.
  const targetOptions = useMemo(
    () =>
      entities
        .filter((e) => e.id !== source.id)
        .sort((a, b) => a.name.localeCompare(b.name)),
    [entities, source.id],
  );
  const [targetId, setTargetId] = useState<string>(
    targetOptions[0]?.id ?? '',
  );
  const [label, setLabel] = useState('');
  const [strength, setStrength] = useState(0.7);

  const trimmedLabel = label.trim().toUpperCase().replace(/\s+/g, '_');
  const duplicate =
    targetId !== '' &&
    relations.some((r) => r.source === source.id && r.target === targetId);
  const canSubmit =
    targetId !== '' && trimmedLabel.length > 0 && !duplicate && !pending;

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      source: source.id,
      target: targetId,
      label: trimmedLabel,
      strength: Math.max(0, Math.min(1, strength)),
    });
  };

  const target = targetOptions.find((e) => e.id === targetId) ?? null;

  return (
    <form
      className="kg-add-relation"
      data-testid="kg-add-rel-form"
      onSubmit={submit}
      style={{
        display: 'flex',
        gap: 10,
        flexWrap: 'wrap',
        alignItems: 'flex-end',
        padding: '10px 12px',
        margin: '8px 0',
        background: 'var(--color-surface-alt, #f5f7fa)',
        border: '1px solid var(--color-border, #e2e6ec)',
        borderRadius: 6,
      }}
    >
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          From
        </span>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 8px',
            border: '1px solid var(--color-border, #e2e6ec)',
            borderRadius: 4,
            fontSize: 12,
            background: 'var(--color-surface, #fff)',
          }}
        >
          <span
            className="kg-type-swatch"
            style={{ background: colors[source.type] }}
            aria-hidden
          />
          {source.name}
        </span>
      </label>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          To
        </span>
        <select
          value={targetId}
          onChange={(e) => setTargetId(e.target.value)}
          aria-label="Relation target entity"
          data-testid="kg-add-rel-target"
        >
          {targetOptions.length === 0 ? (
            <option value="">No other entities</option>
          ) : (
            targetOptions.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name} · {e.type}
              </option>
            ))
          )}
        </select>
      </label>
      <label
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          flex: '1 1 180px',
        }}
      >
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Label
        </span>
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="USES, RUNS_ON, …"
          aria-label="New relation label"
          data-testid="kg-add-rel-label"
          style={{
            fontFamily: 'var(--font-mono)',
            textTransform: 'uppercase',
          }}
        />
      </label>
      <label
        style={{ display: 'flex', flexDirection: 'column', gap: 4, width: 160 }}
      >
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Strength — {strength.toFixed(2)}
        </span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={strength}
          onChange={(e) => setStrength(parseFloat(e.target.value))}
          aria-label="New relation strength"
          data-testid="kg-add-rel-strength"
        />
      </label>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        {target && (
          <span
            className="kg-type-swatch"
            style={{ background: colors[target.type], width: 14, height: 14 }}
            aria-hidden
          />
        )}
        <button type="button" className="ghost-btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="submit"
          className="ghost-btn primary"
          disabled={!canSubmit}
          data-testid="kg-add-rel-submit"
        >
          <Icon name="check" size={11} /> {pending ? 'Adding…' : 'Add'}
        </button>
      </div>
      {duplicate && (
        <div
          role="alert"
          style={{ fontSize: 11, color: 'var(--twin-red-vivid, #b03060)' }}
          data-testid="kg-add-rel-duplicate"
        >
          A relation from “{source.name}” to this target already exists.
        </div>
      )}
    </form>
  );
}
