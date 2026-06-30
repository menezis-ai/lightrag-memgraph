/**
 * TagsTab — tag catalog governance (header → review queue → filters →
 * category rail + card grid + detail panel).
 *
 * Ported from Desktop/UI/tags.jsx. Sub-components extracted out:
 *   - StatusBadge → ./StatusBadge.tsx (exported)
 *   - TagActionModal → ./TagActionModal.tsx (exported, the 8-kind dispatch)
 *
 * Behavior delta vs the proto:
 *   - `tags`, `categories`, `currentUser` injected via props.
 *   - The Approve action emits a structured `TagApproveAction` to the host
 *     rather than pushing a toast inline; the host owns the toast queue.
 *   - Modal commit returns a structured `TagActionCommit` — the host
 *     translates it to a toast payload.
 */

import { type ReactNode, useMemo, useRef, useState } from 'react';
import { Icon } from './Icon';
import { StatusBadge } from './StatusBadge';
import { TagActionModal, type TagActionCommit } from './TagActionModal';
import { useUrlParam } from '../hooks/useUrlParam';
import { useImportCategories } from '../api/queries';
import { api } from '../api/resources';
import { ApiError } from '../api/client';
import type {
  TagAction,
  TagCategory,
  TagCurrentUser,
  TagEntry,
  TagStatusFilter,
} from '../types/tag';
import { TAG_STATUS_FILTERS } from '../types/tag';

export interface TagApproveAction {
  tag: TagEntry;
}

export interface TagsTabProps {
  tags: readonly TagEntry[];
  categories: readonly TagCategory[];
  currentUser: TagCurrentUser;
  folderLabel?: string;
  /** Direct approve (palier-3 fast path for a pending request). */
  onApprove?: (action: TagApproveAction) => void | Promise<void>;
  /** Commit handler for the 8-kind modal dispatch. */
  onCommit?: (commit: TagActionCommit) => void;
  /** Host-controlled tab navigation (replaces direct window.history mutation). */
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
  /** Expand pending review cards on first render for operator shells. */
  defaultPendingOpen?: boolean;
}

interface DomainDraft {
  key: string;
  id: string;
  label: string;
  color: string;
  existing: boolean;
}

const DOMAIN_ID_RE = /^[a-z0-9][a-z0-9-]*$/;
const DOMAIN_COLOR_RE = /^#[0-9a-fA-F]{6}$/;

function normalizeDomainId(value: string): string {
  return value.trim().toLowerCase().replaceAll(/\s+/g, '-');
}

function draftFromCategories(categories: readonly TagCategory[]): DomainDraft[] {
  return categories.map((cat) => ({
    key: cat.id,
    id: cat.id,
    label: cat.label,
    color: cat.color,
    existing: true,
  }));
}

function appendNewDomainDraft(current: readonly DomainDraft[]): DomainDraft[] {
  const existingIds = new Set(current.map((row) => normalizeDomainId(row.id)));
  let nextId = 'new-domain';
  let index = 2;
  while (existingIds.has(nextId)) {
    nextId = `new-domain-${index}`;
    index += 1;
  }
  return [
    ...current,
    {
      key: `draft-${Date.now()}-${index}`,
      id: nextId,
      label: 'New domain',
      color: '#5A7FB4',
      existing: false,
    },
  ];
}

function validateDomainDraft(draft: readonly DomainDraft[]): string | null {
  if (draft.length === 0) return 'At least one domain is required.';
  const seen = new Set<string>();
  for (const row of draft) {
    const id = normalizeDomainId(row.id);
    if (!id || !DOMAIN_ID_RE.test(id)) {
      return 'Domain ids must use lowercase letters, numbers and hyphens.';
    }
    if (seen.has(id)) return `Domain id "${id}" is duplicated.`;
    seen.add(id);
    if (!row.label.trim()) return `Domain "${id}" needs a label.`;
    if (!DOMAIN_COLOR_RE.test(row.color.trim())) {
      return `Domain "${id}" needs a hex color like #5A7FB4.`;
    }
  }
  return null;
}

function apiErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError && typeof err.body === 'object' && err.body) {
    return (err.body as { detail?: string }).detail ?? err.message;
  }
  if (err instanceof Error) return err.message;
  return fallback;
}

export function TagsTab({
  tags,
  categories,
  currentUser,
  folderLabel = 'default',
  onApprove,
  onCommit,
  onNavigate,
  defaultPendingOpen = false,
}: Readonly<TagsTabProps>) {
  const [selectedCat, setSelectedCat] = useUrlParam<string>('cat', 'all');
  const [selectedStatus, setSelectedStatus] = useUrlParam<TagStatusFilter>(
    'status',
    'all',
    {
      validate: (v): boolean =>
        (TAG_STATUS_FILTERS as readonly string[]).includes(v),
    },
  );
  const [q, setQ] = useUrlParam<string>('q', '');
  const [selectedTag, setSelectedTag] = useUrlParam<string>('tag', tags[0]?.tag ?? '');
  const [pendingOpen, setPendingOpen] = useState(defaultPendingOpen);
  const [modal, setModal] = useState<TagAction | null>(null);
  const [approvingTags, setApprovingTags] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const approveRequestedTag = (tag: TagEntry) => {
    if (approvingTags.has(tag.tag)) return;
    setApprovingTags((current) => new Set(current).add(tag.tag));
    void Promise.resolve(onApprove?.({ tag })).finally(() => {
      setApprovingTags((current) => {
        const next = new Set(current);
        next.delete(tag.tag);
        return next;
      });
    });
  };

  // ── Taxonomy import / domain editor / template download ─────────
  // Categories remain a folder-wide governance taxonomy. Admins can
  // update the canonical JSON directly or edit the same replacement
  // payload through the UI below.
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importStatus, setImportStatus] = useState<
    | { kind: 'idle' }
    | { kind: 'success'; count: number }
    | { kind: 'error'; message: string }
  >({ kind: 'idle' });
  const [domainDraft, setDomainDraft] = useState<DomainDraft[]>(() =>
    draftFromCategories(categories),
  );
  const [domainError, setDomainError] = useState<string | null>(null);
  const [domainRailEditing, setDomainRailEditing] = useState(false);
  const importCategories = useImportCategories();

  const openDomainRailEditor = (options: { addNew?: boolean } = {}): void => {
    const draft = draftFromCategories(categories);
    setDomainDraft(options.addNew ? appendNewDomainDraft(draft) : draft);
    setDomainError(null);
    setDomainRailEditing(true);
  };

  const updateDomainDraft = (
    key: string,
    patch: Partial<Pick<DomainDraft, 'id' | 'label' | 'color'>>,
  ): void => {
    setDomainDraft((current) =>
      current.map((row) => (row.key === key ? { ...row, ...patch } : row)),
    );
    setDomainError(null);
  };

  const addDomainDraft = (): void => {
    setDomainDraft((current) => appendNewDomainDraft(current));
    setDomainError(null);
  };

  const removeDomainDraft = (key: string): void => {
    setDomainDraft((current) => current.filter((row) => row.key !== key));
    setDomainError(null);
  };

  const saveDomainDraft = async (): Promise<void> => {
    const validationError = validateDomainDraft(domainDraft);
    if (validationError) {
      setDomainError(validationError);
      return;
    }
    const payload = domainDraft.map((row) => ({
      id: normalizeDomainId(row.id),
      label: row.label.trim(),
      color: row.color.trim(),
    }));
    try {
      await importCategories.mutateAsync(payload);
      setImportStatus({ kind: 'success', count: payload.length });
      setDomainRailEditing(false);
      if (
        selectedCat !== 'all' &&
        selectedCat !== 'uncategorized' &&
        !payload.some((row) => row.id === selectedCat)
      ) {
        setSelectedCat('all');
      }
    } catch (err) {
      const message = apiErrorMessage(err, 'Domain update failed.');
      setDomainError(message);
      setImportStatus({ kind: 'error', message });
    }
  };

  const handleDownloadTemplate = async (): Promise<void> => {
    try {
      const data = await api.downloadCategoriesTemplate();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'twin-categories.template.json';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setImportStatus({
        kind: 'error',
        message:
          err instanceof Error
            ? `Template download failed: ${err.message}`
            : 'Template download failed.',
      });
    }
  };

  const handleImportFile = async (file: File): Promise<void> => {
    setImportStatus({ kind: 'idle' });
    let parsed: unknown;
    try {
      const text = await file.text();
      parsed = JSON.parse(text);
    } catch (err) {
      setImportStatus({
        kind: 'error',
        message: `Invalid JSON file: ${
          err instanceof Error ? err.message : String(err)
        }`,
      });
      return;
    }
    if (!Array.isArray(parsed)) {
      setImportStatus({
        kind: 'error',
        message:
          'Root must be a JSON array of category objects. See ' +
          'docs/templates/twin-categories.schema.json for the expected shape.',
      });
      return;
    }
    try {
      await importCategories.mutateAsync(
        parsed as Parameters<typeof api.importCategories>[0],
      );
      setImportStatus({ kind: 'success', count: parsed.length });
    } catch (err) {
      const message = apiErrorMessage(err, 'Import failed.');
      setImportStatus({ kind: 'error', message });
    }
  };

  const requested = useMemo(
    () => tags.filter((t) => t.tier === 'requested' && t.status === 'pending-review'),
    [tags],
  );
  const pendingLabel = requested.length === 1 ? 'pending item' : 'pending items';
  const knownCategories = useMemo(
    () => new Set(categories.map((cat) => cat.id)),
    [categories],
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = {
      all: tags.filter((t) => t.tier !== 'requested').length,
    };
    categories.forEach((cat) => {
      c[cat.id] = tags.filter(
        (t) => t.category === cat.id && t.tier !== 'requested',
      ).length;
    });
    c.uncategorized = tags.filter(
      (t) => t.tier !== 'requested' && !knownCategories.has(t.category),
    ).length;
    return c;
  }, [tags, categories, knownCategories]);

  const removedDomainsWithTags = useMemo(() => {
    const draftIds = new Set(domainDraft.map((row) => normalizeDomainId(row.id)));
    return categories
      .filter((cat) => !draftIds.has(cat.id) && (counts[cat.id] ?? 0) > 0)
      .map((cat) => ({
        ...cat,
        count: counts[cat.id] ?? 0,
      }));
  }, [categories, counts, domainDraft]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return tags.filter((t) => {
      if (t.tier === 'requested') return false;
      if (selectedCat === 'uncategorized') {
        if (knownCategories.has(t.category)) return false;
      } else if (selectedCat !== 'all' && t.category !== selectedCat) {
        return false;
      }
      if (selectedStatus !== 'all' && t.status !== selectedStatus) return false;
      if (needle) {
        const hay = (t.tag + ' ' + t.def + ' ' + t.aliases.join(' ')).toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [tags, q, selectedCat, selectedStatus, knownCategories]);

  const detail = tags.find((t) => t.tag === selectedTag) ?? tags[0] ?? null;
  const canEdit = currentUser.palier >= 3;
  const canSuggest = currentUser.palier >= 2;
  const totalActive = counts.all;

  const clearFilters = () => {
    setSelectedCat('all');
    setSelectedStatus('all');
    setQ('');
  };

  const suggestions = useMemo(
    () =>
      tags
        .filter((t) => t.tier !== 'requested' && t.status === 'active')
        .slice(0, 4),
    [tags],
  );

  let tagsGridContent: ReactNode;
  if (filtered.length > 0) {
    tagsGridContent = (
      <div className="tags-grid">
        {filtered.map((t) => {
          const cat = categories.find((c) => c.id === t.category);
          return (
            <button
              key={t.tag}
              className={'tag-card ' + (selectedTag === t.tag ? 'is-selected' : '')}
              onClick={() => setSelectedTag(t.tag)}
              data-testid={`tag-card-${t.tag}`}
            >
              <div className="tag-card-h">
                <code className="tag-card-name">{t.tag}</code>
                {cat && (
                  <span
                    className="domain-badge"
                    style={{ borderColor: cat.color, color: cat.color }}
                  >
                    {cat.label}
                  </span>
                )}
              </div>
              <div className="tag-card-def">{t.def}</div>
              {t.aliases.length > 0 && (
                <div className="tag-card-aliases">
                  <span className="al-label">syn:</span>
                  {t.aliases.map((a) => (
                    <code key={a}>{a}</code>
                  ))}
                </div>
              )}
              <div className="tag-card-footer">
                <span>
                  <b>{t.sources_count}</b> docs
                </span>
                <span className="dot-sep">·</span>
                <span>{t.query_freq_30d}/30d</span>
                <span className="spacer" />
                <StatusBadge status={t.status} />
              </div>
            </button>
          );
        })}
      </div>
    );
  } else if (totalActive === 0) {
    tagsGridContent = (
      <TagsEmptyZero
        canSuggest={canSuggest}
        onRequest={() => setModal({ kind: 'request' })}
      />
    );
  } else {
    tagsGridContent = (
      <TagsEmptyFiltered
        q={q}
        selectedCat={selectedCat}
        selectedStatus={selectedStatus}
        categories={categories}
        suggestions={suggestions}
        canSuggest={canSuggest}
        onClear={clearFilters}
        onPickTag={(name) => setSelectedTag(name)}
        onRequest={() => setModal({ kind: 'request' })}
      />
    );
  }

  return (
    <div className="tags-screen">
      <div className="tags-header">
        <div>
          <h1>Tags</h1>
          <div className="tags-sub">
            <span>
              Tag catalog governance · {totalActive} active tags · {requested.length}{' '}
              {pendingLabel} · folder <code>{folderLabel}</code>
            </span>
            {/*
              palier-pill killed per the 30/05 cleanup. Role is JWT-only —
              the operator's capabilities (canEdit / canSuggest) are gated
              silently below; surfacing "palier 3 · admin / steward" as a
              header chrome chip was a gimmick that doesn't carry its
              weight in production.
            */}
          </div>
        </div>
        <div className="tags-header-actions">
          {canEdit && (
            <>
              <button
                className="ghost-btn"
                onClick={() => void handleDownloadTemplate()}
                title="Download the canonical category template JSON"
                data-testid="taxonomy-download-template"
              >
                <Icon name="external-link" size={12} /> Download template
              </button>
              <button
                className="ghost-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={importCategories.isPending}
                title="Mirror an uploaded JSON taxonomy into the active folder (replace, not merge)"
                data-testid="taxonomy-import"
              >
                <Icon name="cloud-upload" size={12} />{' '}
                {importCategories.isPending
                  ? 'Importing…'
                  : 'Import categories'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  // Reset the input so re-selecting the same file fires
                  // another change event (browser would otherwise no-op).
                  e.target.value = '';
                  if (file) void handleImportFile(file);
                }}
                data-testid="taxonomy-import-file"
              />
            </>
          )}
          {canSuggest && (
            <button
              className="ghost-btn"
              onClick={() => exportTagCatalogJson(tags, categories, currentUser.name)}
              title="Download full tag catalog as JSON"
            >
              <Icon name="external-link" size={12} /> Export tag catalog
            </button>
          )}
          <button
            className="primary-btn"
            onClick={() => setModal({ kind: 'request' })}
          >
            <Icon name="plus" size={12} /> Request new tag
          </button>
        </div>
      </div>

      {importStatus.kind === 'success' && (
        <output
          className={
            'taxonomy-import-status taxonomy-import-status--' +
            importStatus.kind
          }
          data-testid="taxonomy-import-status"
        >
          <Icon name="circle-check" size={14} />{' '}
          Categories imported · {importStatus.count} domain
          {importStatus.count === 1 ? '' : 's'} applied.
        </output>
      )}
      {importStatus.kind === 'error' && (
        <div
          className="taxonomy-import-status taxonomy-import-status--error"
          role="alert"
          data-testid="taxonomy-import-status"
        >
          <Icon name="alert-triangle" size={14} /> {importStatus.message}
        </div>
      )}

      {requested.length > 0 && canSuggest && (
        <div className={'pending-section ' + (pendingOpen ? 'is-open' : '')}>
          <button
            className="pending-h"
            onClick={() => setPendingOpen((o) => !o)}
            aria-expanded={pendingOpen}
          >
            <Icon name="alert-triangle" size={14} color="var(--twin-amber-vivid)" />
            <span className="pending-title">Tag review queue</span>
            <span className="pending-counts">
              <b>{requested.length}</b> governance item
              {requested.length === 1 ? '' : 's'} awaiting review
            </span>
            <span
              style={{
                display: 'inline-flex',
                transform: pendingOpen ? 'none' : 'rotate(-90deg)',
                transition: 'transform .15s',
              }}
            >
              <Icon
                name="chevron-down"
                size={14}
                color="var(--color-text-tertiary)"
              />
            </span>
          </button>
          {pendingOpen && (
            <div className="pending-grid">
              {requested.map((t) => {
                const isEditProposal = t.proposal_kind === 'edit' && t.target_tag;
                const pendingTitle = isEditProposal ? t.target_tag! : t.tag;
                const pendingKind = isEditProposal ? 'Edit suggestion' : 'New tag request';
                let approveLabel = 'Approve';
                if (approvingTags.has(t.tag)) {
                  approveLabel = 'Approving…';
                } else if (isEditProposal) {
                  approveLabel = 'Approve edit';
                }
                return (
                  <div
                    key={t.tag}
                    className="pending-card requested"
                    data-testid={`pending-${t.tag}`}
                  >
                    <div className="pending-card-h">
                      <code className="pending-tagname">{pendingTitle}</code>
                      <span className="pending-kind">{pendingKind}</span>
                      <span style={{ marginLeft: 'auto' }}>
                        <StatusBadge status="pending-review" />
                      </span>
                    </div>
                    <div className="pending-justif">{t.justification}</div>
                    <div className="pending-meta">
                      Proposed by <b>{t.requested_by}</b> · {t.requested_at} · category{' '}
                      <code>{t.category}</code>
                      {isEditProposal && t.proposed_fields?.length ? (
                        <>
                          {' '}· fields <code>{t.proposed_fields.join(', ')}</code>
                        </>
                      ) : null}
                    </div>
                    {canEdit ? (
                      <div className="pending-actions">
                        <button
                          className="primary-btn small"
                          disabled={approvingTags.has(t.tag)}
                          onClick={() => approveRequestedTag(t)}
                        >
                          {approveLabel}
                        </button>
                        {!isEditProposal && (
                          <button
                            className="ghost-btn small"
                            onClick={() => setModal({ kind: 'edit-approve', tag: t })}
                          >
                            Edit & approve
                          </button>
                        )}
                        <button
                          className="ghost-btn small danger"
                          onClick={() => setModal({ kind: 'reject', tag: t })}
                        >
                          Reject
                        </button>
                      </div>
                    ) : (
                      <div className="pending-actions">
                        <span className="muted">Awaiting reviewer approval</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      <div className="tags-filters">
        <div className="tags-search">
          <Icon name="search" size={13} color="var(--color-text-tertiary)" />
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by name, definition or synonym…"
            aria-label="Search tags"
          />
          {q && (
            <button
              className="x"
              onClick={() => setQ('')}
              aria-label="Clear search"
            >
              <Icon name="x" size={11} />
            </button>
          )}
        </div>
        <select
          className="mini-select"
          value={selectedStatus}
          onChange={(e) => setSelectedStatus(e.target.value as TagStatusFilter)}
          aria-label="Status filter"
        >
          <option value="all">All statuses</option>
          <option value="active">Active</option>
          <option value="pending-promotion">Pending</option>
          <option value="deprecated">Deprecated</option>
        </select>
      </div>

      <div className="tags-body">
        <aside className={'tags-rail ' + (domainRailEditing ? 'is-managing' : '')}>
          <div className="tags-rail-head">
            <span>Domains</span>
            {canEdit && !domainRailEditing && (
              <span className="tags-rail-tools">
                <button
                  className="rail-tool-btn"
                  type="button"
                  onClick={() => openDomainRailEditor()}
                  aria-label="Manage domains"
                  title="Manage domains"
                  data-testid="rail-manage-domains"
                >
                  <Icon name="settings" size={12} />
                </button>
                <button
                  className="rail-tool-btn"
                  type="button"
                  onClick={() => openDomainRailEditor({ addNew: true })}
                  aria-label="Add domain"
                  title="Add domain"
                  data-testid="rail-add-domain"
                >
                  <Icon name="plus" size={12} />
                </button>
              </span>
            )}
          </div>
          {domainRailEditing ? (
            <DomainRailEditor
              draft={domainDraft}
              error={domainError}
              tagCounts={counts}
              removedDomainsWithTags={removedDomainsWithTags}
              isSaving={importCategories.isPending}
              onAdd={addDomainDraft}
              onUpdate={updateDomainDraft}
              onRemove={removeDomainDraft}
              onCancel={() => {
                setDomainRailEditing(false);
                setDomainError(null);
              }}
              onSave={() => void saveDomainDraft()}
            />
          ) : (
            <>
              <button
                className={'rail-item ' + (selectedCat === 'all' ? 'is-active' : '')}
                onClick={() => setSelectedCat('all')}
                aria-pressed={selectedCat === 'all'}
                data-testid="rail-all"
              >
                <span
                  className="rail-dot"
                  style={{ background: 'var(--color-text-tertiary)' }}
                />
                <span className="rail-label">All domains</span>
                <span className="rail-count">{counts.all}</span>
              </button>
              {categories.map((c) => (
                <button
                  key={c.id}
                  className={
                    'rail-item ' + (selectedCat === c.id ? 'is-active' : '')
                  }
                  onClick={() => setSelectedCat(c.id)}
                  aria-pressed={selectedCat === c.id}
                  data-testid={`rail-${c.id}`}
                >
                  <span className="rail-dot" style={{ background: c.color }} />
                  <span className="rail-label">{c.label}</span>
                  <span className="rail-count">{counts[c.id] ?? 0}</span>
                </button>
              ))}
              <button
                className={
                  'rail-item ' + (selectedCat === 'uncategorized' ? 'is-active' : '')
                }
                onClick={() => setSelectedCat('uncategorized')}
                aria-pressed={selectedCat === 'uncategorized'}
                data-testid="rail-uncategorized"
              >
                <span
                  className="rail-dot"
                  style={{ background: 'var(--color-text-tertiary)' }}
                />
                <span className="rail-label">Uncategorized</span>
                <span className="rail-count">{counts.uncategorized ?? 0}</span>
              </button>
            </>
          )}
        </aside>

        <main className="tags-grid-wrap">{tagsGridContent}</main>

        <TagDetailPanel
          t={detail}
          allTags={tags}
          categories={categories}
          onSelect={setSelectedTag}
          onAction={setModal}
          onCommit={onCommit}
          onNavigate={onNavigate}
          canEdit={canEdit}
          canSuggest={canSuggest}
        />
      </div>

      {modal && (
        <TagActionModal
          action={modal}
          allTags={tags}
          categories={categories}
          onClose={() => setModal(null)}
          onCommit={(commit) => {
            setModal(null);
            onCommit?.(commit);
          }}
        />
      )}
    </div>
  );
}

interface DomainRailEditorProps {
  draft: readonly DomainDraft[];
  error: string | null;
  tagCounts: Record<string, number>;
  removedDomainsWithTags: readonly (TagCategory & { count: number })[];
  isSaving: boolean;
  onAdd: () => void;
  onUpdate: (
    key: string,
    patch: Partial<Pick<DomainDraft, 'id' | 'label' | 'color'>>,
  ) => void;
  onRemove: (key: string) => void;
  onCancel: () => void;
  onSave: () => void;
}

function DomainRailEditor({
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
              onChange={(e) =>
                onUpdate(row.key, { color: e.target.value.toUpperCase() })
              }
            />
            <div className="domain-rail-fields">
              <input
                className="text-input domain-rail-label"
                value={row.label}
                aria-label={`${row.id} domain label`}
                placeholder="Domain name"
                required
                aria-required="true"
                onChange={(e) => onUpdate(row.key, { label: e.target.value })}
              />
              {!row.existing && (
                <input
                  className="text-input domain-rail-id"
                  value={row.id}
                  aria-label={`${row.label || row.id} domain id`}
                  placeholder="domain-id"
                  required
                  aria-required="true"
                  onChange={(e) =>
                    onUpdate(row.key, { id: normalizeDomainId(e.target.value) })
                  }
                />
              )}
            </div>
            <span className="domain-rail-count">
              {row.existing ? (tagCounts[row.id] ?? 0) : 0}
            </span>
            <button
              className="rail-tool-btn danger"
              type="button"
              onClick={() => onRemove(row.key)}
              aria-label={`Remove ${row.label || row.id}`}
            >
              <Icon name="trash" size={12} />
            </button>
          </div>
        ))}
      </div>
      <button className="ghost-btn small domain-rail-add" type="button" onClick={onAdd}>
        <Icon name="plus" size={12} /> Add domain
      </button>
      {removedDomainsWithTags.length > 0 && (
        <output className="impact-box warning domain-rail-impact">
          <Icon name="alert-triangle" size={14} />
          <span>
            Removed domains with tags become uncategorized until retagged:{' '}
            {removedDomainsWithTags.map((cat) => (
              <code key={cat.id}>
                {cat.label} ({cat.count})
              </code>
            ))}
          </span>
        </output>
      )}
      {error && (
        <div className="impact-box danger domain-rail-impact" role="alert">
          <Icon name="alert-triangle" size={14} />
          <span>{error}</span>
        </div>
      )}
      <div className="domain-rail-actions">
        <button
          className="ghost-btn small"
          type="button"
          onClick={onCancel}
          disabled={isSaving}
        >
          Cancel
        </button>
        <button
          className="primary-btn small"
          type="button"
          onClick={onSave}
          disabled={isSaving}
        >
          {isSaving ? 'Saving…' : 'Save'}
        </button>
      </div>
    </div>
  );
}

interface TagsEmptyZeroProps {
  canSuggest: boolean;
  onRequest: () => void;
}

function TagsEmptyZero({ canSuggest, onRequest }: Readonly<TagsEmptyZeroProps>) {
  return (
    <div className="tags-empty zero" data-testid="tags-empty-zero">
      <div className="tags-empty-illus" aria-hidden="true">
        <svg width="120" height="80" viewBox="0 0 120 80" fill="none">
          <rect
            x="6"
            y="18"
            width="44"
            height="18"
            rx="3"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity="0.45"
          />
          <rect
            x="54"
            y="32"
            width="56"
            height="18"
            rx="3"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity="0.35"
          />
          <rect
            x="20"
            y="48"
            width="38"
            height="18"
            rx="3"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity="0.25"
          />
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
        <li>
          <Icon name="info-circle" size={11} /> Tier 1 (Trunk) — gov-validated, applies
          cross-folder
        </li>
        <li>
          <Icon name="info-circle" size={11} /> Tier 2 (Branch) — dept-scoped,
          steward-approved
        </li>
        <li>
          <Icon name="info-circle" size={11} /> Tier 3 (Leaf) — user-proposed,
          lightweight review
        </li>
      </ul>
    </div>
  );
}

interface TagsEmptyFilteredProps {
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
          {active.map((a) => (
            <span key={a.key} className="tags-empty-chip">
              {a.label}
            </span>
          ))}
        </div>
      )}
      <div className="tags-empty-actions">
        <button className="primary-btn" onClick={onClear}>
          Clear filters
        </button>
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
            {suggestions.map((s) => (
              <button
                key={s.tag}
                className="tags-empty-suggest-chip"
                onClick={() => {
                  onClear();
                  onPickTag(s.tag);
                }}
              >
                <code>{s.tag}</code>
                <span className="tags-empty-suggest-meta">{s.sources_count} docs</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface TagDetailPanelProps {
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

function TagDetailPanel({
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
          <div className="usage-cell">
            <div className="usage-num">{t.sources_count}</div>
            <div className="usage-lbl">Docs</div>
          </div>
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
            <span className="hist-what">
              {t.last_edit.action ?? 'edited'}
            </span>
            <span className="hist-who">by {t.last_edit.by}</span>
          </div>
          <div className="hist-item">
            <span className="hist-when">{t.created.at}</span>
            <span className="hist-what">created</span>
            <span className="hist-who">by {t.created.by}</span>
          </div>
        </div>
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

/**
 * Build a tag catalog JSON snapshot and trigger a download.
 * Exported for unit testing without rendering the whole tab.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function exportTagCatalogJson(
  tags: readonly TagEntry[],
  categories: readonly TagCategory[],
  exportedBy: string,
  folder: string = 'main',
): void {
  const payload = {
    folder,
    exported_at: new Date().toISOString(),
    exported_by: exportedBy,
    categories,
    tags: tags.map((t) => ({
      tag: t.tag,
      tier: t.tier,
      category: t.category,
      status: t.status,
      def: t.def,
      aliases: t.aliases,
      deprecates: t.deprecates,
      sources_count: t.sources_count,
      chunks_count: t.chunks_count,
      query_freq_30d: t.query_freq_30d,
      created: t.created,
      last_edit: t.last_edit,
    })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const stamp = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `twin-rag-tag-catalog-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 0);
}
