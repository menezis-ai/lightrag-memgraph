import type { Dispatch, MouseEventHandler, RefObject, SetStateAction } from 'react';
import { Icon } from '../Icon';
import { type GraphEntity, type GraphEntityType, type GraphRelation } from '../../types/graph';
import { entityRadius } from './graphLayout';

interface GraphCanvasProps {
  canvasRef: RefObject<HTMLDivElement | null>;
  entities: readonly GraphEntity[];
  matches: readonly GraphEntity[];
  visibleRels: readonly GraphRelation[];
  selected: GraphEntity | null;
  highlightIds: ReadonlySet<string>;
  hoverId: string | null;
  pan: { x: number; y: number };
  zoom: number;
  colors: Record<GraphEntityType, string>;
  onMouseDown: MouseEventHandler<SVGSVGElement>;
  onSelectEntity: (id: string) => void;
  onHoverEntity: (id: string | null) => void;
  onZoomChange: Dispatch<SetStateAction<number>>;
  onClearFilters: () => void;
}

function edgeStrokeWidth(isHighlighted: boolean, isStrong: boolean): number {
  if (isHighlighted) return 1.6;
  if (isStrong) return 1.1;
  return 0.7;
}

function edgeStrokeOpacity(isHighlighted: boolean, isDimmed: boolean): number {
  if (isHighlighted) return 0.9;
  if (isDimmed) return 0.08;
  return 0.32;
}

export function GraphCanvas({
  canvasRef,
  entities,
  matches,
  visibleRels,
  selected,
  highlightIds,
  hoverId,
  pan,
  zoom,
  colors,
  onMouseDown,
  onSelectEntity,
  onHoverEntity,
  onZoomChange,
  onClearFilters,
}: Readonly<GraphCanvasProps>) {
  return (
    <div
          ref={canvasRef}
          className="kg-canvas"
          data-testid="kg-canvas"
          style={{ touchAction: 'none', overscrollBehavior: 'none' }}
        >
          <svg
            onMouseDown={onMouseDown}
            viewBox="0 0 1000 680"
            preserveAspectRatio="xMidYMid meet"
            className="kg-svg"
            aria-label="Knowledge graph canvas"
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
                  selected !== null &&
                  (r.source === selected.id || r.target === selected.id);
                const dim = selected !== null && !hi;
                const strong = r.strength >= 0.75;
                const dx = t.x - s.x;
                const dy = t.y - s.y;
                const len = Math.hypot(dx, dy) || 1;
                const sourceRadius = entityRadius(s) + 1.5;
                const targetRadius = entityRadius(t) + 2;
                const x1 = s.x + (dx / len) * sourceRadius;
                const y1 = s.y + (dy / len) * sourceRadius;
                const x2 = t.x - (dx / len) * targetRadius;
                const y2 = t.y - (dy / len) * targetRadius;
                return (
                  <g
                    key={r.id}
                    className={`kg-edge${hi ? ' is-hi' : ''}${dim ? ' is-dim' : ''}`}
                  >
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={hi ? 'var(--twin-accent)' : 'currentColor'}
                      strokeWidth={edgeStrokeWidth(hi, strong)}
                      strokeOpacity={edgeStrokeOpacity(hi, dim)}
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
                const radius = entityRadius(e);
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
                      onSelectEntity(e.id);
                    }}
                    onKeyDown={(ev) => {
                      if (ev.key === 'Enter' || ev.key === ' ') {
                        ev.preventDefault();
                        ev.stopPropagation();
                        onSelectEntity(e.id);
                      }
                    }}
                    onMouseEnter={() => onHoverEntity(e.id)}
                    onMouseLeave={() => onHoverEntity(null)}
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
                onClick={onClearFilters}
              >
                Clear filter
              </button>
            </div>
          )}
          <div className="kg-zoom-pill">
            <button
              onClick={() => onZoomChange((z) => Math.max(0.4, z * 0.85))}
              aria-label="Zoom out"
            >
              <Icon name="minus" size={11} />
            </button>
            <span data-testid="kg-zoom-value">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => onZoomChange((z) => Math.min(3, z * 1.18))}
              aria-label="Zoom in"
            >
              <Icon name="plus" size={11} />
            </button>
          </div>
    </div>
  );
}
