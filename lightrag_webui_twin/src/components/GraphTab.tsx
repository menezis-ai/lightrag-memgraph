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
  type GraphEntityType,
  type GraphRelation,
} from '../types/graph';

const TYPE_KEYS = Object.keys(GRAPH_TYPE_LABEL) as readonly GraphEntityType[];

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
  const [selectedId, setSelectedId] = useUrlParam<string>(
    'gent',
    entities[0]?.id ?? '',
  );
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(
    null,
  );

  const typeCounts = useMemo(() => {
    const c: Partial<Record<GraphEntityType, number>> = {};
    entities.forEach((e) => {
      c[e.type] = (c[e.type] ?? 0) + 1;
    });
    return c;
  }, [entities]);

  const matches = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return entities.filter((e) => {
      if (!activeTypes.includes(e.type)) return false;
      if (!needle) return true;
      return (
        e.name.toLowerCase().includes(needle) ||
        e.summary.toLowerCase().includes(needle)
      );
    });
  }, [entities, q, activeTypes]);

  const visibleIds = useMemo(() => new Set(matches.map((e) => e.id)), [matches]);
  const visibleRels = useMemo(
    () =>
      relations.filter(
        (r) => visibleIds.has(r.source) && visibleIds.has(r.target),
      ),
    [relations, visibleIds],
  );

  const selected =
    entities.find((e) => e.id === selectedId) ?? entities[0] ?? null;
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
  const onWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const dz = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.max(0.4, Math.min(3, z * dz)));
  };
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
              {entities.length} entities · {relations.length} relations · workspace{' '}
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
          <button className="ghost-btn" onClick={resetView} title="Reset view">
            <Icon name="refresh" size={12} /> Reset view
          </button>
        </div>
      </div>

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
          className="kg-canvas"
          onMouseDown={onMouseDown}
          onWheel={onWheel}
          data-testid="kg-canvas"
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
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setSelectedId(e.id);
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
          neighbors={neighbors}
          colors={colors}
          onSelect={(id) => setSelectedId(id)}
          onNavigate={onNavigate}
        />
      </div>
    </div>
  );
}

interface GraphDetailPanelProps {
  entity: GraphEntity | null;
  neighbors: { rels: GraphRelation[]; nodes: GraphEntity[] };
  colors: Record<GraphEntityType, string>;
  onSelect: (id: string) => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

function GraphDetailPanel({
  entity,
  neighbors,
  colors,
  onSelect,
  onNavigate,
}: GraphDetailPanelProps) {
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
  const incoming = neighbors.rels.filter((r) => r.target === entity.id);
  const outgoing = neighbors.rels.filter((r) => r.source === entity.id);
  return (
    <aside className="kg-detail">
      <div className="kg-detail-h">
        <div className="kg-detail-title">
          <span
            className="kg-detail-swatch"
            style={{ background: colors[entity.type] }}
          />
          <h2>{entity.name}</h2>
        </div>
        <div className="kg-detail-type" style={{ color: colors[entity.type] }}>
          {GRAPH_TYPE_LABEL[entity.type]}
        </div>
        <p className="kg-detail-summary">{entity.summary}</p>
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
      </div>

      <div className="kg-detail-section">
        <div className="section-label">
          <span>Outgoing ({outgoing.length})</span>
        </div>
        {outgoing.length === 0 ? (
          <div className="muted-sm">No outgoing relations.</div>
        ) : (
          <ul className="kg-rel-list">
            {outgoing.map((r) => {
              const t = neighbors.nodes.find((n) => n.id === r.target);
              if (!t) return null;
              return (
                <li key={r.id}>
                  <span className="kg-rel-arrow">→</span>
                  <code className="kg-rel-label">{r.label}</code>
                  <button
                    className="kg-rel-target"
                    onClick={() => onSelect(t.id)}
                  >
                    <span
                      className="kg-rel-swatch"
                      style={{ background: colors[t.type] }}
                    />
                    {t.name}
                  </button>
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
          <span>Incoming ({incoming.length})</span>
        </div>
        {incoming.length === 0 ? (
          <div className="muted-sm">No incoming relations.</div>
        ) : (
          <ul className="kg-rel-list">
            {incoming.map((r) => {
              const s = neighbors.nodes.find((n) => n.id === r.source);
              if (!s) return null;
              return (
                <li key={r.id}>
                  <button
                    className="kg-rel-target"
                    onClick={() => onSelect(s.id)}
                  >
                    <span
                      className="kg-rel-swatch"
                      style={{ background: colors[s.type] }}
                    />
                    {s.name}
                  </button>
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
          onClick={() => onNavigate?.('documents', { q: entity.name })}
        >
          <Icon name="external-link" size={11} /> View {entity.sources} sources
          mentioning this entity
        </button>
        <div className="kg-detail-locked">
          <Icon name="lock" size={11} />
          <span>
            Traverse relations with tag filter — <b>Twin Graph</b>
          </span>
        </div>
      </div>
    </aside>
  );
}
