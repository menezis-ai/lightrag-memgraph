/**
 * GraphTab — knowledge graph tab.
 *
 * Ported from Desktop/UI/graph.jsx. Read-only teaser of the LightRAG-extracted
 * entities + relations. SVG layout uses precomputed `x`, `y` from the API
 * (no in-browser force simulation). Pan/zoom is a vanilla SVG transform.
 *
 * Behavior delta vs the proto:
 *   - Entities, relations, colors injected via props.
 *   - `onNavigate(tab, params)` instead of direct globalThis.history mutation.
 *   - Wheel/pan use refs to avoid re-attaching listeners on each render.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from '../Icon';
import {
  useUrlArrayParam,
  useUrlParam,
} from '../../hooks/useUrlParam';
import {
  GRAPH_TYPE_COLORS,
  GRAPH_TYPE_LABEL,
  type GraphEntity,
  type GraphEntityType,
  type GraphRelation,
} from '../../types/graph';
import {
  useCreateGraphEntity,
} from '../../api/queries';
import { mapCreateEntityError } from '../../api/errors';
import { useQueryClient } from '@tanstack/react-query';
import { GraphCanvas } from './GraphCanvas';
import { GraphFilters } from './GraphFilters';
import { AddEntityForm, GraphDetailPanel, TagAttrEditor } from './GraphInspector';
import { TYPE_KEYS } from './graphLayout';
import { PINNED_STORAGE_KEY, readPinnedEntityIds, tagsOf } from './graphSelection';
import type { GraphTabProps } from './graphTypes';

function collectEntityTags(
  entities: readonly GraphEntity[],
  docTags?: Readonly<Record<string, readonly string[]>>,
): Map<string, readonly string[]> {
  const map = new Map<string, readonly string[]>();
  for (const entity of entities) {
    const tags = new Set<string>(tagsOf(entity));
    for (const doc of entity.source_docs ?? []) {
      for (const tag of docTags?.[doc] ?? []) {
        tags.add(tag);
      }
    }
    map.set(entity.id, Array.from(tags));
  }
  return map;
}

export function GraphTab({
  entities,
  relations,
  colors = GRAPH_TYPE_COLORS,
  docLabels,
  docTags,
  tagCatalog = [],
  folderLabel,
  onNavigate,
  onToast,
}: GraphTabProps) {
  const [q, setQ] = useUrlParam<string>('gq', '');
  const [activeTypes, setActiveTypes] = useUrlArrayParam(
    'gtype',
    TYPE_KEYS,
  );
  const [tagFilter, setTagFilter] = useUrlArrayParam('gtag', []);
  const [docFilter, setDocFilter] = useUrlArrayParam('gdoc', []);
  const [tagMatchMode, setTagMatchMode] = useUrlParam<'any' | 'all'>(
    'gtagmode',
    'any',
    { validate: (v) => v === 'any' || v === 'all' },
  );
  const [docMatchMode, setDocMatchMode] = useUrlParam<'any' | 'all'>(
    'gdocmode',
    'any',
    { validate: (v) => v === 'any' || v === 'all' },
  );
  const [addOpen, setAddOpen] = useState(false);
  const [addEntityError, setAddEntityError] = useState<string | null>(null);
  const createEntity = useCreateGraphEntity();
  const qc = useQueryClient();

  // Effective tags = own twin tags ∪ tags inherited from source documents.
  const entityTags = useMemo(
    () => collectEntityTags(entities, docTags),
    [entities, docTags],
  );
  const allTags = useMemo(() => {
    const s = new Set<string>();
    tagCatalog.forEach((t) => s.add(t));
    entityTags.forEach((tags) => tags.forEach((t) => s.add(t)));
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [entityTags, tagCatalog]);
  const allSourceDocs = useMemo(() => {
    const s = new Set<string>();
    entities.forEach((e) => {
      (e.source_docs ?? []).forEach((doc) => s.add(doc));
    });
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [entities]);
  const propertyKeySuggestions = useMemo(() => {
    const s = new Set<string>();
    entities.forEach((e) => {
      Object.keys(e.properties ?? {}).forEach((key) => s.add(key));
    });
    return Array.from(s).sort((a, b) => a.localeCompare(b));
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
      const tags = entityTags.get(e.id) ?? [];
      const sourceDocs = e.source_docs ?? [];
      if (
        tagFilter.length > 0 &&
        (tagMatchMode === 'all'
          ? !tagFilter.every((t) => tags.includes(t))
          : !tags.some((t) => tagFilter.includes(t)))
      )
        return false;
      if (
        docFilter.length > 0 &&
        (docMatchMode === 'all'
          ? !docFilter.every((doc) => sourceDocs.includes(doc))
          : !sourceDocs.some((doc) => docFilter.includes(doc)))
      )
        return false;
      if (!needle) return true;
      return (
        e.name.toLowerCase().includes(needle) ||
        e.summary.toLowerCase().includes(needle)
      );
    });
  }, [
    entities,
    entityTags,
    q,
    tagFilter,
    tagMatchMode,
    docFilter,
    docMatchMode,
  ]);

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
    globalThis.localStorage.setItem(PINNED_STORAGE_KEY, JSON.stringify(pinnedIds));
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
    globalThis.addEventListener('mousemove', onMove);
    globalThis.addEventListener('mouseup', onUp);
    return () => {
      globalThis.removeEventListener('mousemove', onMove);
      globalThis.removeEventListener('mouseup', onUp);
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
    setQ('');
    setActiveTypes(TYPE_KEYS);
    setTagFilter([]);
    setDocFilter([]);
    setSelectedId('');
    setSelectedRelId(null);
    setHoverId(null);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const hasTransferableFilters = tagFilter.length > 0 || docFilter.length > 0;
  const transferFiltersToRetrieval = () => {
    if (!hasTransferableFilters) return;
    const params: Record<string, string> = {};
    if (tagFilter.length > 0) {
      params.rtag = tagFilter.join(',');
      params.rtagmode = tagMatchMode;
    }
    if (docFilter.length > 0) {
      params.rdoc = docFilter.join(',');
      params.rdocmode = docMatchMode;
    }
    onNavigate?.('retrieval', params);
  };

  return (
    <div className="kg">
      <div className="kg-header">
        <div>
          <h1>Knowledge Graph</h1>
          <div className="kg-sub">
            <span>
              {entities.length} entities · {relations.length} relations
              {folderLabel && (
                <>
                  {' '}· folder <code>{folderLabel}</code>
                </>
              )}
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
            onClick={() => {
              setAddEntityError(null);
              setAddOpen(true);
            }}
            type="button"
            data-testid="kg-add-entity-btn"
          >
            <Icon name="plus" size={12} /> Add entity
          </button>
          <button className="ghost-btn" onClick={resetView} title="Reset view">
            <Icon name="refresh" size={12} /> Reset view
          </button>
          <button
            className="ghost-btn"
            onClick={transferFiltersToRetrieval}
            disabled={!hasTransferableFilters}
            title="Open Retrieval with these tag/document filters"
            data-testid="kg-transfer-filters"
          >
            <Icon name="message-circle" size={12} /> Chat with filters
          </button>
        </div>
      </div>

      {addOpen && (
        <AddEntityForm
          colors={colors}
          existingNames={entities.map((e) => e.name)}
          pending={createEntity.isPending}
          error={addEntityError}
          onCancel={() => {
            setAddEntityError(null);
            setAddOpen(false);
          }}
          onSubmit={(payload) => {
            setAddEntityError(null);
            createEntity.mutate(payload, {
              onSuccess: (created) => {
                setAddOpen(false);
                setSelectedId(created.id);
              },
              onError: (error) => {
                const mapped = mapCreateEntityError(error, payload.name);
                // TR-KG-01: a 500 from the server means the write
                // committed but the projection failed. Treat it as a
                // half-success — close the form, invalidate the
                // entities query (onSettled would also fire this, the
                // explicit call here pins the intent next to the
                // toast), and surface a soft "done" toast instead of
                // a hard error that would push the operator to retry
                // and collide with the now-existing entity.
                if (mapped.kind === 'projection') {
                  setAddOpen(false);
                  void qc.invalidateQueries({ queryKey: ['graph-entities'] });
                  onToast?.({
                    kind: 'done',
                    title: 'Entity created',
                    sub: mapped.message,
                  });
                  return;
                }
                setAddEntityError(mapped.message);
              },
            });
          }}
        />
      )}

      <div className="kg-body">
        <GraphFilters
          activeTypes={activeTypes}
          typeCounts={typeCounts}
          colors={colors}
          allTags={allTags}
          tagFilter={tagFilter}
          onTagFilterChange={setTagFilter}
          tagMatchMode={tagMatchMode}
          onTagMatchModeChange={setTagMatchMode}
          allSourceDocs={allSourceDocs}
          docFilter={docFilter}
          onDocFilterChange={setDocFilter}
          docMatchMode={docMatchMode}
          onDocMatchModeChange={setDocMatchMode}
          docLabels={docLabels}
          onToggleType={toggleType}
        />

        <GraphCanvas
          canvasRef={canvasRef}
          entities={entities}
          matches={matches}
          visibleRels={visibleRels}
          selected={selected}
          highlightIds={highlightIds}
          hoverId={hoverId}
          pan={pan}
          zoom={zoom}
          colors={colors}
          onMouseDown={onMouseDown}
          onSelectEntity={setSelectedId}
          onHoverEntity={setHoverId}
          onZoomChange={setZoom}
          onClearFilters={() => {
            setQ('');
            setActiveTypes(TYPE_KEYS);
          }}
        />

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
          tagCatalog={tagCatalog}
          propertyKeySuggestions={propertyKeySuggestions}
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

export { TagAttrEditor };
export type { GraphTabProps };
