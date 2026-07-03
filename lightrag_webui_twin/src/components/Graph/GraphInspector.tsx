import { useEffect, useMemo, useState } from 'react';
import { Icon } from '../Icon';
import {
  useCreateGraphRelation,
  useDeleteGraphEntity,
  useDeleteGraphRelation,
  useUpdateGraphEntity,
  useUpdateGraphRelation,
} from '../../api/queries';
import {
  GRAPH_TYPE_LABEL,
  type GraphEntity,
  type GraphEntityPatch,
  type GraphEntityType,
  type GraphRelation,
  type GraphRelationPatch,
} from '../../types/graph';
import { TYPE_KEYS } from './graphLayout';

interface GraphDetailPanelProps {
  entity: GraphEntity | null;
  selectedRel: GraphRelation | null;
  neighbors: { rels: GraphRelation[]; nodes: GraphEntity[] };
  colors: Record<GraphEntityType, string>;
  entities: readonly GraphEntity[];
  relations: readonly GraphRelation[];
  typeLabels: Record<GraphEntityType, string>;
  docLabels?: Readonly<Record<string, string>>;
  /** TR-KG-03: active tag catalog. Empty list disables the node
   *  tag editor's autocomplete + lets any free text through (used
   *  by the legacy seed tests that don't render a catalog). */
  tagCatalog: readonly string[];
  propertyKeySuggestions: readonly string[];
  onSelect: (id: string) => void;
  onSelectRelation: (id: string) => void;
  onClearRelation: () => void;
  pinnedIds: readonly string[];
  onTogglePinned: (id: string) => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

export function GraphDetailPanel({
  entity,
  selectedRel,
  neighbors,
  colors,
  entities,
  relations,
  typeLabels,
  docLabels,
  tagCatalog,
  propertyKeySuggestions,
  onSelect,
  onSelectRelation,
  onClearRelation,
  pinnedIds,
  onTogglePinned,
  onNavigate,
}: Readonly<GraphDetailPanelProps>) {
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
        docLabels={docLabels}
        tagCatalog={tagCatalog}
        propertyKeySuggestions={propertyKeySuggestions}
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
  docLabels?: Readonly<Record<string, string>>;
  /** TR-KG-03: active tag catalog passed down to the in-editor
   *  TagAttrEditor so unknown tags can't reach the backend.
   *  Empty list disables the binding (mirrors GraphTab's default). */
  tagCatalog: readonly string[];
  propertyKeySuggestions: readonly string[];
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

function EntityRelationList({
  direction,
  rels,
  nodes,
  colors,
  onSelectRelation,
}: Readonly<{
  direction: 'outgoing' | 'incoming';
  rels: readonly GraphRelation[];
  nodes: readonly GraphEntity[];
  colors: Record<GraphEntityType, string>;
  onSelectRelation: (id: string) => void;
}>) {
  if (rels.length === 0) {
    return (
      <div className="muted-sm">
        No {direction === 'outgoing' ? 'outgoing' : 'incoming'} relations.
      </div>
    );
  }
  return (
    <ul className="kg-rel-list">
      {rels.map((rel) => {
        const endpoint = relationEndpoint(direction, nodes, rel);
        if (!endpoint) return null;
        return (
          <GraphRelationRow
            key={rel.id}
            rel={rel}
            direction={direction}
            endpoint={endpoint}
            colors={colors}
            onSelectRelation={onSelectRelation}
          />
        );
      })}
    </ul>
  );
}

interface GraphRelationRowProps {
  rel: GraphRelation;
  direction: 'outgoing' | 'incoming';
  endpoint: GraphEntity;
  colors: Record<GraphEntityType, string>;
  onSelectRelation: (id: string) => void;
}

function relationEndpoint(
  direction: 'outgoing' | 'incoming',
  nodes: readonly GraphEntity[],
  rel: GraphRelation,
): GraphEntity | undefined {
  const relationEndpointId = direction === 'outgoing' ? rel.target : rel.source;
  return nodes.find((node) => node.id === relationEndpointId);
}

function GraphRelationRow({
  rel,
  direction,
  endpoint,
  colors,
  onSelectRelation,
}: Readonly<GraphRelationRowProps>) {
  const relationLabel = <code className="kg-rel-label">{rel.label}</code>;

  return (
    <li>
      <button
        type="button"
        className="kg-rel-row"
        onClick={() => onSelectRelation(rel.id)}
        data-testid={`kg-rel-row-${rel.id}`}
      >
        {direction === 'outgoing' && <span className="kg-rel-arrow">→</span>}
        {direction === 'outgoing' && relationLabel}
        <span className="kg-rel-target">
          <span
            className="kg-rel-swatch"
            style={{ background: colors[endpoint.type] }}
          />
          <span className="kg-rel-target-name">{endpoint.name}</span>
        </span>
        {direction === 'incoming' && (
          <>
            {relationLabel}
            <span className="kg-rel-arrow">→</span>
          </>
        )}
        <span
          className="kg-rel-strength"
          title={`strength ${rel.strength.toFixed(2)}`}
        >
          {Math.round(rel.strength * 100)}
        </span>
      </button>
    </li>
  );
}

function GraphDeleteConfirm({
  armed,
  pending,
  noun,
  testId,
  onArm,
  onConfirm,
  onCancel,
}: Readonly<{
  armed: boolean;
  pending: boolean;
  noun: 'entity' | 'relation';
  testId: string;
  onArm: () => void;
  onConfirm: () => void;
  onCancel: () => void;
}>) {
  const confirm = () => {
    if (!armed) {
      onArm();
      return;
    }
    onConfirm();
  };
  const buttonLabel = deleteConfirmLabel(pending, armed, noun);
  return (
    <div
      className="kg-detail-section kg-lifecycle"
      data-testid={`kg-${noun}-lifecycle`}
      style={{ borderTop: '1px solid var(--color-border, #e2e6ec)' }}
    >
      <button
        type="button"
        className={armed ? 'ghost-btn danger' : 'ghost-btn'}
        onClick={confirm}
        disabled={pending}
        data-testid={testId}
        style={
          armed
            ? {
                color: 'var(--twin-red-vivid, #b03060)',
                borderColor: 'var(--twin-red-vivid, #b03060)',
              }
            : undefined
        }
      >
        <Icon name={armed ? 'alert-triangle' : 'trash'} size={11} />{' '}
        {buttonLabel}
      </button>
      {armed && (
        <button
          type="button"
          className="ghost-btn small"
          onClick={onCancel}
          data-testid={`${testId}-cancel`}
        >
          Cancel
        </button>
      )}
    </div>
  );
}

function deleteConfirmLabel(
  pending: boolean,
  armed: boolean,
  noun: 'entity' | 'relation',
): string {
  if (pending) return 'Deleting…';
  if (armed) return 'Click again to confirm';
  return `Delete ${noun}`;
}

function EntityHeader({
  entity,
  neighborsCount,
  editing,
  draft,
  colors,
  typeLabels,
  isPinned,
  onTogglePinned,
  onStartEdit,
  onDraftChange,
}: Readonly<{
  entity: GraphEntity;
  neighborsCount: number;
  editing: boolean;
  draft: EntityDraft | null;
  colors: Record<GraphEntityType, string>;
  typeLabels: Record<GraphEntityType, string>;
  isPinned: boolean;
  onTogglePinned: () => void;
  onStartEdit: () => void;
  onDraftChange: (draft: EntityDraft) => void;
}>) {
  const updateDraft = (patch: Partial<EntityDraft>) => {
    if (draft) onDraftChange({ ...draft, ...patch });
  };
  return (
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
            onChange={(e) => updateDraft({ name: e.target.value })}
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
              onClick={onStartEdit}
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
              updateDraft({ type: e.target.value as GraphEntityType })
            }
            aria-label="Entity type"
          >
            {(Object.keys(typeLabels) as GraphEntityType[]).map((type) => (
              <option key={type} value={type}>
                {typeLabels[type]}
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
          onChange={(e) => updateDraft({ summary: e.target.value })}
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
            <span className="kg-stat-n">{neighborsCount}</span>
            <span className="kg-stat-l">relations</span>
          </div>
        </div>
      )}
    </div>
  );
}

function EntityTagsSection({
  editing,
  tags,
  tagCatalog,
  onChange,
}: Readonly<{
  editing: boolean;
  tags: readonly string[];
  tagCatalog: readonly string[];
  onChange: (tags: string[]) => void;
}>) {
  const tagBody = renderEntityTagsBody(editing, tags, tagCatalog, onChange);
  return (
    <div className="kg-detail-section">
      <div className="section-label">
        <span>
          Tags {editing ? <em>— edit</em> : <em>— node attributes</em>}
        </span>
      </div>
      {tagBody}
    </div>
  );
}

function renderEntityTagsBody(
  editing: boolean,
  tags: readonly string[],
  tagCatalog: readonly string[],
  onChange: (tags: string[]) => void,
) {
  if (editing) {
    return (
      <TagAttrEditor tags={tags} tagCatalog={tagCatalog} onChange={onChange} />
    );
  }
  if (tags.length === 0) return <div className="muted-sm">No tags.</div>;
  return (
    <div className="tag-chips">
      {tags.map((tag) => (
        <span key={tag} className="tag-chip">
          {tag}
        </span>
      ))}
    </div>
  );
}

function PropertyEmptyState({
  onStartEdit,
}: Readonly<{ onStartEdit: () => void }>) {
  return (
    <div className="muted-sm">
      No custom properties.{' '}
      <button className="kg-inline-add" onClick={onStartEdit} type="button">
        + Add some
      </button>
    </div>
  );
}

function PropertyList({
  propEntries,
}: Readonly<{ propEntries: readonly [string, unknown][] }>) {
  return (
    <dl className="kg-prop-list">
      {propEntries.map(([key, value]) => (
        <div key={key} className="kg-prop-row">
          <dt>{key}</dt>
          <dd>{String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function renderEntityPropertiesBody({
  editing,
  properties,
  propEntries,
  suggestions,
  onChange,
  onStartEdit,
}: Readonly<{
  editing: boolean;
  properties: Record<string, string>;
  propEntries: readonly [string, unknown][];
  suggestions: readonly string[];
  onChange: (properties: Record<string, string>) => void;
  onStartEdit: () => void;
}>) {
  if (editing) {
    return (
      <PropEditor
        properties={properties}
        suggestions={suggestions}
        onChange={onChange}
      />
    );
  }
  if (propEntries.length === 0) {
    return <PropertyEmptyState onStartEdit={onStartEdit} />;
  }
  return <PropertyList propEntries={propEntries} />;
}

function renderRelationPropertiesBody({
  editing,
  draft,
  propEntries,
  onDraftPropertiesChange,
  onStartEdit,
}: Readonly<{
  editing: boolean;
  draft: RelationDraft | null;
  propEntries: readonly [string, unknown][];
  onDraftPropertiesChange: (properties: Record<string, string>) => void;
  onStartEdit: () => void;
}>) {
  if (editing && draft) {
    return (
      <PropEditor
        properties={draft.properties}
        suggestions={[]}
        onChange={onDraftPropertiesChange}
      />
    );
  }
  if (propEntries.length === 0) {
    return <PropertyEmptyState onStartEdit={onStartEdit} />;
  }
  return <PropertyList propEntries={propEntries} />;
}

function EntityPropertiesSection({
  editing,
  properties,
  propEntries,
  suggestions,
  onChange,
  onStartEdit,
}: Readonly<{
  editing: boolean;
  properties: Record<string, string>;
  propEntries: readonly [string, unknown][];
  suggestions: readonly string[];
  onChange: (properties: Record<string, string>) => void;
  onStartEdit: () => void;
}>) {
  const propertyBody = renderEntityPropertiesBody({
    editing,
    properties,
    propEntries,
    suggestions,
    onChange,
    onStartEdit,
  });
  return (
    <div className="kg-detail-section">
      <div className="section-label">
        <span>
          Properties{' '}
          {editing ? <em>— add / remove</em> : <em>— custom metadata</em>}
        </span>
        {!editing && propEntries.length > 0 && (
          <span className="kg-prop-count">{propEntries.length}</span>
        )}
      </div>
      {propertyBody}
    </div>
  );
}

function EntityEditor({
  entity,
  neighbors,
  entities,
  relations,
  colors,
  typeLabels,
  docLabels,
  tagCatalog,
  propertyKeySuggestions,
  onSelectRelation,
  isPinned,
  onTogglePinned,
  onNavigate,
}: Readonly<EntityEditorProps>) {
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
    const t = globalThis.setTimeout(() => setArmedDelete(false), 4000);
    return () => globalThis.clearTimeout(t);
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
      properties: { ...entity.properties },
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
      <EntityHeader
        entity={entity}
        neighborsCount={neighbors.rels.length}
        editing={editing}
        draft={draft}
        colors={colors}
        typeLabels={typeLabels}
        isPinned={isPinned}
        onTogglePinned={onTogglePinned}
        onStartEdit={startEdit}
        onDraftChange={setDraft}
      />

      {/* Tags — node-attribute strings, decoupled from the WebuiTag taxonomy. */}
      <EntityTagsSection
        editing={editing && Boolean(draft)}
        tags={draft?.tags ?? entity.tags ?? []}
        tagCatalog={tagCatalog}
        onChange={(tags) => setDraft((d) => (d ? { ...d, tags } : d))}
      />

      {/* Properties — custom k/v metadata. */}
      <EntityPropertiesSection
        editing={editing && Boolean(draft)}
        properties={draft?.properties ?? {}}
        propEntries={propEntries}
        suggestions={propertyKeySuggestions}
        onChange={(properties) =>
          setDraft((d) => (d ? { ...d, properties } : d))
        }
        onStartEdit={startEdit}
      />

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
            <EntityRelationList
              direction="outgoing"
              rels={outgoing}
              nodes={neighbors.nodes}
              colors={colors}
              onSelectRelation={onSelectRelation}
            />
          </div>

          <div className="kg-detail-section">
            <div className="section-label">
              <span>
                Incoming ({incoming.length}) <em>— click to edit</em>
              </span>
            </div>
            <EntityRelationList
              direction="incoming"
              rels={incoming}
              nodes={neighbors.nodes}
              colors={colors}
              onSelectRelation={onSelectRelation}
            />
          </div>

          <div className="kg-detail-section kg-detail-cta">
            <button
              className="ghost-btn"
              onClick={() => {
                const sourceDocs = entity.source_docs ?? [];
                if (sourceDocs.length > 0) {
                  const sources = sourceDocs.map((doc) => docLabels?.[doc] ?? doc);
                  const params: Record<string, string> = {
                    source: sources.join(','),
                  };
                  if (sourceDocs.length === 1) params.doc = sourceDocs[0];
                  onNavigate?.('documents', params);
                  return;
                }
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

          <GraphDeleteConfirm
            armed={armedDelete}
            pending={deleteEntity.isPending}
            noun="entity"
            testId="kg-entity-delete"
            onArm={() => setArmedDelete(true)}
            onConfirm={() => {
              deleteEntity.mutate(entity.id);
              setArmedDelete(false);
            }}
            onCancel={() => setArmedDelete(false)}
          />
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
}: Readonly<RelationEditorProps>) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<RelationDraft | null>(null);
  const updateRelation = useUpdateGraphRelation();
  const deleteRelation = useDeleteGraphRelation();
  const [armedDelete, setArmedDelete] = useState(false);
  useEffect(() => {
    if (!armedDelete) return;
    const t = globalThis.setTimeout(() => setArmedDelete(false), 4000);
    return () => globalThis.clearTimeout(t);
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
      properties: { ...rel.properties },
    });
    setEditing(true);
  };
  const cancel = () => {
    setEditing(false);
    setDraft(null);
  };
  const commit = () => {
    if (!draft) return;
    const cleaned = draft.label.trim().toUpperCase().replaceAll(/\s+/g, '_');
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
  const relationPropertiesBody = renderRelationPropertiesBody({
    editing,
    draft,
    propEntries,
    onDraftPropertiesChange: (properties) =>
      setDraft((d) => (d ? { ...d, properties } : d)),
    onStartEdit: startEdit,
  });

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
                  d ? { ...d, strength: Number.parseFloat(e.target.value) } : d,
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
        {relationPropertiesBody}
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
        <GraphDeleteConfirm
          armed={armedDelete}
          pending={deleteRelation.isPending}
          noun="relation"
          testId="kg-rel-delete"
          onArm={() => setArmedDelete(true)}
          onConfirm={() => {
            deleteRelation.mutate(rel.id, {
              onSuccess: () => onBack(),
            });
            setArmedDelete(false);
          }}
          onCancel={() => setArmedDelete(false)}
        />
      )}
    </>
  );
}

// ─── Tag chip editor (node attribute strings) ──────────────────────────
//
// TR-KG-03 (QA report 2026-06-12): node tags must come from the
// active tag catalog. The previous free-text input accepted any value
// and bypassed the canonical vocabulary; the backend now 422s on
// unknown tags (see ``server/webui_router._validate_graph_entity_tags``)
// and this editor mirrors the same rule client-side: autocomplete
// proposes catalog matches on the typed prefix. Enter only commits an
// exact catalog match; unknown values surface an inline warning.
export function TagAttrEditor({
  tags,
  tagCatalog,
  onChange,
}: Readonly<{
  tags: readonly string[];
  /** Active tag catalog, e.g. derived from ``/tags`` via
   *  ``tagCatalogForSuggestions`` upstream. An empty list disables
   *  the binding (isolated tests). */
  tagCatalog: readonly string[];
  onChange: (next: string[]) => void;
}>) {
  const [v, setV] = useState('');
  const [focused, setFocused] = useState(false);
  const normalized = v.trim().toLowerCase().replaceAll(/\s+/g, '-');
  const catalogSet = useMemo(
    () => new Set(tagCatalog.map((t) => t.toLowerCase())),
    [tagCatalog],
  );
  const bindingActive = tagCatalog.length > 0;
  const [tagSuggestionIndex, setTagSuggestionIndex] = useState(0);
  const suggestions = useMemo(() => {
    if (!bindingActive || (!focused && !normalized)) return [];
    return tagCatalog
      .filter(
        (t) =>
          !tags.includes(t) &&
          (!normalized || t.toLowerCase().startsWith(normalized)),
      )
      .slice(0, 6);
  }, [tagCatalog, tags, normalized, bindingActive, focused]);
  const isKnown = !bindingActive || catalogSet.has(normalized);
  const activeSuggestion =
    suggestions.length === 0
      ? undefined
      : suggestions[Math.min(tagSuggestionIndex, suggestions.length - 1)];
  const suggestionListId = 'kg-tag-suggestions';

  /* eslint-disable react-hooks/set-state-in-effect -- intentional reset of the highlighted tag-suggestion index when the filter query / focus changes. */
  useEffect(() => {
    setTagSuggestionIndex(0);
  }, [normalized, focused, bindingActive]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const add = (value?: string) => {
    const t = (value ?? normalized).trim().toLowerCase().replaceAll(/\s+/g, '-');
    if (!t || tags.includes(t)) return;
    if (bindingActive && !catalogSet.has(t)) return;
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
          onFocus={() => setFocused(true)}
          onBlur={() => globalThis.setTimeout(() => setFocused(false), 120)}
          onKeyDownCapture={(e) => {
            if (e.key === 'Escape' && v) {
              e.stopPropagation();
              setV('');
              setTagSuggestionIndex(0);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              if (suggestions.length === 0) return;
              e.preventDefault();
              setTagSuggestionIndex((idx) => (idx + 1) % suggestions.length);
              return;
            }
            if (e.key === 'ArrowUp') {
              if (suggestions.length === 0) return;
              e.preventDefault();
              setTagSuggestionIndex(
                (idx) => (idx - 1 + suggestions.length) % suggestions.length,
              );
              return;
            }
            if (e.key === 'Enter') {
              e.preventDefault();
              if (activeSuggestion) {
                add(activeSuggestion);
                return;
              }
              add();
            }
          }}
          placeholder="Add tag…"
          role="combobox"
          aria-label="Add node tag"
          aria-autocomplete="list"
          aria-expanded={suggestions.length > 0}
          aria-controls={suggestionListId}
          aria-activedescendant={
            activeSuggestion ? `kg-tag-sugg-${activeSuggestion}` : undefined
          }
        />
      </div>
      {bindingActive && normalized && !isKnown && (
        <div
          className="muted-sm"
          role="alert"
          data-testid="kg-tag-not-in-catalog"
          style={{ marginTop: 4, fontSize: 11 }}
        >
          “{normalized}” is not in the tag catalog. Pick an existing tag
          or approve it first in the Tags tab.
        </div>
      )}
      {suggestions.length > 0 && (
        <div
          id={suggestionListId}
          role="listbox"
          aria-label="Tag suggestions"
          className="autocomplete panel-autocomplete"
          data-testid="kg-tag-suggestions"
          style={{ marginTop: 4 }}
        >
          {suggestions.map((s, idx) => (
            <button
              type="button"
              key={s}
              id={`kg-tag-sugg-${s}`}
              role="option"
              aria-selected={idx === tagSuggestionIndex}
              className={`autocomplete-row${
                idx === tagSuggestionIndex ? ' focus' : ''
              }`}
              onMouseEnter={() => setTagSuggestionIndex(idx)}
              onMouseDown={(e) => e.preventDefault()}
              data-testid={`kg-tag-sugg-${s}`}
              onClick={() => add(s)}
              style={{ cursor: 'pointer' }}
            >
              <span style={{ fontSize: 12 }}>{s}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Properties editor — k/v list with add / rename / remove ───────────
function PropEditor({
  properties,
  suggestions,
  onChange,
}: Readonly<{
  properties: Record<string, string>;
  suggestions: readonly string[];
  onChange: (next: Record<string, string>) => void;
}>) {
  const entries = Object.entries(properties);
  const [draftKey, setDraftKey] = useState('');
  const [draftVal, setDraftVal] = useState('');
  const availableSuggestions = suggestions.filter(
    (key) => properties[key] === undefined,
  );

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
          list="kg-prop-key-suggestions"
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
        {availableSuggestions.length > 0 && (
          <datalist id="kg-prop-key-suggestions">
            {availableSuggestions.map((key) => (
              <option key={key} value={key} />
            ))}
          </datalist>
        )}
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

// ─── Lifecycle: Add entity inline form ────────────────────────────────
interface AddEntityFormProps {
  colors: Record<GraphEntityType, string>;
  existingNames: readonly string[];
  pending: boolean;
  error?: string | null;
  onCancel: () => void;
  onSubmit: (payload: {
    name: string;
    type: GraphEntityType;
    summary?: string;
  }) => void;
}

export function AddEntityForm({
  colors,
  existingNames,
  pending,
  error,
  onCancel,
  onSubmit,
}: Readonly<AddEntityFormProps>) {
  const [name, setName] = useState('');
  const [type, setType] = useState<GraphEntityType>('PRODUCT');
  const [summary, setSummary] = useState('');
  const trimmed = name.trim();
  const duplicate = trimmed.length > 0 && existingNames.includes(trimmed);
  const canSubmit = trimmed.length > 0 && !duplicate && !pending;

  const submit = (e?: React.SyntheticEvent<HTMLFormElement>) => {
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
    >
      <label className="kg-form-field kg-form-name">
        <span>Name</span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Entity name"
          aria-label="New entity name"
          autoFocus
          data-testid="kg-add-entity-name"
        />
      </label>
      <label className="kg-form-field kg-form-type">
        <span>Type</span>
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
      <label className="kg-form-field kg-form-summary">
        <span>
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
      <div className="kg-form-actions">
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
          className="kg-form-error"
          data-testid="kg-add-entity-duplicate"
        >
          An entity named “{trimmed}” already exists.
        </div>
      )}
      {error && (
        <div
          role="alert"
          className="kg-form-error"
          data-testid="kg-add-entity-error"
        >
          {error}
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
}: Readonly<AddRelationFormProps>) {
  // Targets = every other entity in the graph. Sorted by name for a
  // predictable picker order.
  const targetOptions = useMemo(
    () =>
      entities
        .filter((e) => e.id !== source.id)
        .sort((a, b) => a.name.localeCompare(b.name)),
    [entities, source.id],
  );
  const [selectedTargetId, setSelectedTargetId] = useState<string>(
    targetOptions[0]?.id ?? '',
  );
  const [targetQuery, setTargetQuery] = useState('');
  const [activeTargetIndex, setActiveTargetIndex] = useState(0);
  const [label, setLabel] = useState('');
  const [strength, setStrength] = useState(0.7);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onCancel]);

  const filteredTargetOptions = useMemo(() => {
    const query = targetQuery.trim().toLocaleLowerCase();
    if (!query) return targetOptions;
    return targetOptions.filter((e) =>
      `${e.name} ${e.type}`.toLocaleLowerCase().includes(query),
    );
  }, [targetOptions, targetQuery]);
  const visibleTargetOptions = filteredTargetOptions.slice(0, 12);
  const clampedActiveTargetIndex = Math.min(
    activeTargetIndex,
    Math.max(visibleTargetOptions.length - 1, 0),
  );
  const activeTarget = visibleTargetOptions[clampedActiveTargetIndex] ?? null;
  const effectiveTargetId = useMemo(() => {
    if (
      selectedTargetId &&
      filteredTargetOptions.some((e) => e.id === selectedTargetId)
    ) {
      return selectedTargetId;
    }
    return filteredTargetOptions[0]?.id ?? '';
  }, [filteredTargetOptions, selectedTargetId]);

  const trimmedLabel = label.trim().toUpperCase().replaceAll(/\s+/g, '_');
  const duplicate =
    effectiveTargetId !== '' &&
    relations.some((r) => r.source === source.id && r.target === effectiveTargetId);
  const canSubmit =
    effectiveTargetId !== '' && trimmedLabel.length > 0 && !duplicate && !pending;

  const submit = (e?: React.SyntheticEvent<HTMLFormElement>) => {
    e?.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      source: source.id,
      target: effectiveTargetId,
      label: trimmedLabel,
      strength: Math.max(0, Math.min(1, strength)),
    });
  };

  const target = targetOptions.find((e) => e.id === effectiveTargetId) ?? null;
  const selectTarget = (targetEntity: GraphEntity) => {
    setSelectedTargetId(targetEntity.id);
    setTargetQuery(targetEntity.name);
  };
  const onTargetKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (visibleTargetOptions.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveTargetIndex((i) =>
        Math.min(i + 1, visibleTargetOptions.length - 1),
      );
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveTargetIndex((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === 'Home') {
      e.preventDefault();
      setActiveTargetIndex(0);
      return;
    }
    if (e.key === 'End') {
      e.preventDefault();
      setActiveTargetIndex(visibleTargetOptions.length - 1);
      return;
    }
    if (e.key === 'Enter' && activeTarget) {
      e.preventDefault();
      selectTarget(activeTarget);
    }
  };

  return (
    <form
      className="kg-add-relation"
      data-testid="kg-add-rel-form"
      onSubmit={submit}
    >
      <div className="kg-form-field kg-form-endpoint">
        <span>From</span>
        <span className="kg-form-readonly">
          <span
            className="kg-type-swatch"
            style={{ background: colors[source.type] }}
            aria-hidden
          />
          <span className="kg-form-readonly-text">{source.name}</span>
        </span>
      </div>
      <label className="kg-form-field kg-form-target">
        <span>To</span>
        <div className="kg-target-picker">
          <input
            type="search"
            value={targetQuery}
            onChange={(e) => {
              setTargetQuery(e.target.value);
              setActiveTargetIndex(0);
            }}
            onKeyDown={onTargetKeyDown}
            role="combobox"
            aria-expanded="true"
            aria-controls="kg-add-rel-target-list"
            aria-activedescendant={
              activeTarget ? `kg-add-rel-target-${activeTarget.id}` : undefined
            }
            aria-label="Relation target entity"
            placeholder={
              targetOptions.length === 0
                ? 'No other entities'
                : 'Search entity name'
            }
            data-testid="kg-add-rel-target"
          />
          <div
            id="kg-add-rel-target-list"
            className="kg-target-list"
            role="listbox"
            aria-label="Matching target entities"
          >
            {visibleTargetOptions.map((e, index) => (
              <button
                key={e.id}
                id={`kg-add-rel-target-${e.id}`}
                type="button"
                className={`kg-target-option${e.id === effectiveTargetId ? ' is-selected' : ''}${index === clampedActiveTargetIndex ? ' is-active' : ''}`}
                role="option"
                aria-selected={e.id === effectiveTargetId}
                onMouseEnter={() => setActiveTargetIndex(index)}
                onClick={() => selectTarget(e)}
                data-testid={`kg-add-rel-target-option-${e.id}`}
              >
                <span
                  className="kg-type-swatch"
                  style={{ background: colors[e.type] }}
                  aria-hidden
                />
                <span className="kg-target-name">{e.name}</span>
                <span className="kg-target-type">{e.type}</span>
              </button>
            ))}
            {filteredTargetOptions.length === 0 && (
              <div className="kg-target-empty" role="status">
                No matching entity
              </div>
            )}
          </div>
          {filteredTargetOptions.length > visibleTargetOptions.length && (
            <div className="kg-target-result-count">
              Showing {visibleTargetOptions.length} of{' '}
              {filteredTargetOptions.length}; refine search
            </div>
          )}
        </div>
      </label>
      <label className="kg-form-field kg-form-relation-label">
        <span>Label</span>
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
      <label className="kg-form-field kg-form-strength">
        <span>Strength — {strength.toFixed(2)}</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={strength}
          onChange={(e) => setStrength(Number.parseFloat(e.target.value))}
          aria-label="New relation strength"
          data-testid="kg-add-rel-strength"
        />
      </label>
      <div className="kg-form-actions">
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
          className="kg-form-error"
          data-testid="kg-add-rel-duplicate"
        >
          A relation from “{source.name}” to this target already exists.
        </div>
      )}
    </form>
  );
}
