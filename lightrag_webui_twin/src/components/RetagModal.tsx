/**
 * RetagModal — tag a single document or a bulk selection.
 *
 * Ported from Desktop/UI/modals.jsx. This is the **P0 feature of the
 * Twin WebUI fork (sprint 1)** — UI tag rétroactif with tag catalog
 * autocomplete, preview impact, commit-deferred via Undo.
 *
 * Behavior delta vs the proto:
 *   - Tag catalog is injected via props instead of read from globals, so
 *     the modal is fully testable.
 *   - The "submit" payload (RetagAction) is decoupled from toast lifecycle:
 *     the host receives the structured RetagAction and is responsible for
 *     queuing the propagating/done toasts (and the undo). This keeps the
 *     modal free of timing concerns.
 *   - A11y trap via useModalA11y is wired through a single hook call.
 */

import { useMemo, useRef, useState, type RefObject } from 'react';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
import { useModalA11y } from '../hooks/useModalA11y';
import type { Document } from '../types/document';
import type { TagEntry } from '../types/tag';
import { tagMatchesQuery, tagSuggestionComparator } from '../utils/tags';

export interface RetagAction {
  /** Primary target — first doc when in bulk mode. */
  primary: Document;
  /** All targets. Length 1 for single-doc mode, >1 for bulk. */
  targets: readonly Document[];
  bulk: boolean;
  /** Tags the user requested to add (resolved from autocomplete). */
  adds: readonly string[];
  /** Tags the user requested to remove from `current`. */
  removes: readonly string[];
}

export interface RetagModalProps {
  open: boolean;
  /** Single-doc mode. Ignored if `docs` is set. */
  doc?: Document | null;
  /** Bulk mode. If non-empty, takes precedence over `doc`. */
  docs?: readonly Document[];
  tagCatalog: readonly TagEntry[];
  onClose: () => void;
  onSubmit: (action: RetagAction) => void;
}

export function RetagModal({
  open,
  doc,
  docs,
  tagCatalog,
  onClose,
  onSubmit,
}: Readonly<RetagModalProps>) {
  const modalRef = useRef<HTMLDialogElement>(null);

  const targets = useMemo<readonly Document[]>(() => {
    if (docs && docs.length > 0) return docs;
    if (doc) return [doc];
    return [];
  }, [doc, docs]);
  const bulk = targets.length > 1;
  const isReady = open && targets.length > 0;

  useModalA11y({ open: isReady, onClose, ref: modalRef });

  // Shared = present on every selected doc. Partial = present on some.
  const { sharedTags, partialTags } = useMemo(() => {
    if (targets.length === 0) return { sharedTags: [], partialTags: [] };
    const shared = targets.reduce<string[]>(
      (acc, d, i) =>
        i === 0 ? [...d.tags] : acc.filter((t) => d.tags.includes(t)),
      [],
    );
    const partial = bulk
      ? Array.from(new Set(targets.flatMap((d) => d.tags))).filter(
          (t) => !shared.includes(t),
        )
      : [];
    return { sharedTags: shared, partialTags: partial };
  }, [targets, bulk]);

  const targetsKey = targets.map((d) => d.doc_id).join(',');
  if (!isReady) return null;

  return (
    <RetagModalBody
      key={targetsKey}
      modalRef={modalRef}
      targets={targets}
      bulk={bulk}
      sharedTags={sharedTags}
      partialTags={partialTags}
      tagCatalog={tagCatalog}
      onClose={onClose}
      onSubmit={onSubmit}
    />
  );
}

interface RetagModalBodyProps {
  modalRef: RefObject<HTMLDialogElement | null>;
  targets: readonly Document[];
  bulk: boolean;
  sharedTags: readonly string[];
  partialTags: readonly string[];
  tagCatalog: readonly TagEntry[];
  onClose: () => void;
  onSubmit: (action: RetagAction) => void;
}

function retagApplyLabel(totalChanges: number, pendingRemoveCount: number) {
  if (totalChanges === 0) return 'Apply tag';
  if (pendingRemoveCount > 0 && pendingRemoveCount === totalChanges) {
    return `Remove ${pendingRemoveCount} tag${pendingRemoveCount > 1 ? 's' : ''}`;
  }
  return totalChanges === 1 ? 'Apply tag' : `Apply ${totalChanges} changes`;
}

function suggestionHeaderLabel(count: number): string {
  if (count === 0) return 'No matches';
  return `${count} match${count > 1 ? 'es' : ''} in tags`;
}

function RetagHeaderContext({
  bulk,
  targets,
  primary,
  totalChunks,
}: Readonly<{
  bulk: boolean;
  targets: readonly Document[];
  primary: Document;
  totalChunks: number;
}>) {
  if (bulk) {
    return (
      <>
        <div className="ctx">
          <Icon name="tags" size={13} />
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Bulk · changes propagate to every selected source
          </span>
          <span className="sep">·</span>
          <span className="vis-chip">
            <Icon name="lock" size={11} />
            {targets[0].folder} · {totalChunks.toLocaleString()} chunks total
          </span>
        </div>
        <div className="bulk-target-strip">
          {targets.slice(0, 4).map((doc) => (
            <span key={doc.doc_id} className="bulk-target-chip" title={doc.file_path}>
              <SourceIcon type={doc.type} size={11} />
              <span className={doc.type === 'file' ? '' : 'mono'}>
                {doc.file_path}
              </span>
            </span>
          ))}
          {targets.length > 4 && (
            <span className="bulk-target-more">+{targets.length - 4} more</span>
          )}
        </div>
      </>
    );
  }
  return (
    <div className="ctx">
      <SourceIcon type={primary.type} size={13} />
      <span
        style={{
          fontFamily: primary.type === 'file' ? 'inherit' : 'var(--font-mono)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          maxWidth: 320,
          whiteSpace: 'nowrap',
        }}
      >
        {primary.file_path}
      </span>
      <span className="sep">·</span>
      <span className="vis-chip">
        <Icon name={primary.visibility === 'private' ? 'lock' : 'world'} size={11} />
        {primary.folder} · {primary.visibility}
      </span>
    </div>
  );
}

function RetagPreviewImpact({
  isRemoving,
  pendingAdd,
  pendingRemove,
  previewChunks,
  previewDocs,
}: Readonly<{
  isRemoving: boolean;
  pendingAdd: readonly string[];
  pendingRemove: readonly string[];
  previewChunks: number;
  previewDocs: number;
}>) {
  const tag = isRemoving ? pendingRemove[0] : pendingAdd[0];
  const verb = isRemoving ? 'Removing' : 'Adding';
  return (
    <div
      className={isRemoving ? 'preview-impact removing' : 'preview-impact'}
      data-testid="preview-impact"
    >
      <div className="head">
        <Icon name="eye" size={14} /> Preview impact
      </div>
      <div className="stats">
        <div className="stat">
          <div className="num">{previewChunks}</div>
          <div className="lbl">chunks</div>
        </div>
        <div className="stat">
          <div className="num">{previewDocs}</div>
          <div className="lbl">selected docs</div>
        </div>
      </div>
      <div className="foot">
        <Icon name="arrow-right" size={12} />
        {verb} <span className="mini-chip">{tag}</span>{' '}
        {isRemoving ? 'will update' : '· will update'}{' '}
        {previewChunks.toLocaleString()} chunks across {previewDocs} selected doc
        {previewDocs > 1 ? 's' : ''}
      </div>
    </div>
  );
}

function RetagModalBody({
  modalRef,
  targets,
  bulk,
  sharedTags,
  partialTags,
  tagCatalog,
  onClose,
  onSubmit,
}: Readonly<RetagModalBodyProps>) {
  const current = sharedTags;
  const [pendingAdd, setPendingAdd] = useState<readonly string[]>([]);
  const [pendingRemove, setPendingRemove] = useState<readonly string[]>([]);
  const [input, setInput] = useState('');
  const [focusIdx, setFocusIdx] = useState(0);
  const suggestionListId = 'retag-tag-suggestions';

  const sugg = useMemo(() => {
    // Defence in depth: callers normally pass tagCatalogForSuggestions(), but
    // this mutation dialog also enforces the backend's active-tag invariant so
    // a stale/raw catalogue cannot offer a pending tag that will 422.
    const all = tagCatalog.filter(
      (t) =>
        t.status === 'active' &&
        t.tier !== 'requested' &&
        !current.includes(t.tag) &&
        !pendingAdd.includes(t.tag),
    );
    if (!input) return all.slice(0, 4);
    return all
      .filter((t) => tagMatchesQuery(t, input))
      .sort(tagSuggestionComparator(input))
      .slice(0, 5);
  }, [input, current, pendingAdd, tagCatalog]);
  const activeTagSuggestion = sugg[Math.min(focusIdx, Math.max(sugg.length - 1, 0))];
  const activeTagSuggestionId = activeTagSuggestion
    ? `${suggestionListId}-option-${focusIdx}`
    : undefined;

  const primary = targets[0];

  const addTag = (t: string) => {
    setPendingAdd([...pendingAdd, t]);
    setInput('');
  };
  const undoAdd = (t: string) =>
    setPendingAdd(pendingAdd.filter((x) => x !== t));
  const removeCurrent = (t: string) => {
    if (pendingRemove.includes(t))
      setPendingRemove(pendingRemove.filter((x) => x !== t));
    else setPendingRemove([...pendingRemove, t]);
  };

  const isRemoving = pendingRemove.length > 0 && pendingAdd.length === 0;
  const totalChunks = targets.reduce(
    (s, d) => s + (d.chunks_count ?? 0),
    0,
  );
  const previewChunks = totalChunks;
  const previewDocs = targets.length;

  const totalChanges = pendingAdd.length + pendingRemove.length;
  const applyLabel = retagApplyLabel(totalChanges, pendingRemove.length);

  const submit = () => {
    if (totalChanges === 0) return;
    onSubmit({
      primary,
      targets,
      bulk,
      adds: pendingAdd,
      removes: pendingRemove,
    });
    onClose();
  };

  return (
    <div
      className="modal-backdrop"
    >
      <button
        type="button"
        className="modal-backdrop-dismiss"
        onClick={onClose}
        aria-label="Close retag dialog"
        data-testid="retag-backdrop"
      />
      <dialog
        open
        ref={modalRef}
        className="modal modal-md"
        aria-modal="true"
        aria-labelledby="retag-title"
        tabIndex={-1}
      >
        <div className="modal-header">
          <div style={{ flex: 1 }}>
            <h2 id="retag-title">
              {bulk ? `Retag ${targets.length} sources` : 'Retag document'}
            </h2>
            <RetagHeaderContext
              bulk={bulk}
              targets={targets}
              primary={primary}
              totalChunks={totalChunks}
            />
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close dialog"
          >
            <Icon name="x" size={18} />
          </button>
        </div>

        <div className="modal-body">
          {(current.length > 0 || pendingRemove.length > 0) && (
            <div>
              <div className="section-label">
                <span>{bulk ? 'Tags on all selected' : 'Currently applied'}</span>
              </div>
              <div className="tag-chips" style={{ marginTop: 6 }}>
                {current.map((t) => (
                  <span
                    key={t}
                    style={{
                      opacity: pendingRemove.includes(t) ? 0.45 : 1,
                      textDecoration: pendingRemove.includes(t)
                        ? 'line-through'
                        : 'none',
                    }}
                  >
                    <TagChip tag={t} removable onRemove={removeCurrent} />
                  </span>
                ))}
                {current.length === 0 && (
                  <span className="muted" style={{ fontSize: 11 }}>
                    None shared across all selected sources.
                  </span>
                )}
              </div>
            </div>
          )}
          {bulk && partialTags.length > 0 && (
            <div>
              <div className="section-label">
                <span>
                  On some selected{' '}
                  <span className="hint">
                    — removing these only affects sources where present
                  </span>
                </span>
              </div>
              <div className="tag-chips" style={{ marginTop: 6, opacity: 0.7 }}>
                {partialTags.map((t) => (
                  <TagChip key={t} tag={t} />
                ))}
              </div>
            </div>
          )}

          <div>
            <div className="section-label">
              <span>Add tags</span>
            </div>
            {pendingAdd.length > 0 && (
              <div className="tag-chips" style={{ marginTop: 6, marginBottom: 6 }}>
                {pendingAdd.map((t) => (
                  <span key={t}>
                    <TagChip tag={t} removable onRemove={undoAdd} />
                  </span>
                ))}
              </div>
            )}
            <input
              type="text"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                setFocusIdx(0);
              }}
              onKeyDownCapture={(e) => {
                if (e.key === 'Escape') {
                  e.stopPropagation();
                  setInput('');
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'ArrowDown') {
                  if (sugg.length === 0) return;
                  e.preventDefault();
                  setFocusIdx((idx) => (idx + 1) % sugg.length);
                } else if (e.key === 'ArrowUp') {
                  if (sugg.length === 0) return;
                  e.preventDefault();
                  setFocusIdx((idx) => (idx - 1 + sugg.length) % sugg.length);
                } else if (e.key === 'Enter' && activeTagSuggestion) {
                  e.preventDefault();
                  addTag(activeTagSuggestion.tag);
                }
              }}
              placeholder="Start typing — autocomplete from tags"
              role="combobox"
              aria-label="Tag input"
              aria-autocomplete="list"
              aria-expanded={sugg.length > 0}
              aria-controls={suggestionListId}
              aria-activedescendant={activeTagSuggestionId}
              style={{
                width: '100%',
                padding: '8px 10px',
                fontSize: 13,
                fontFamily: 'var(--font-mono)',
                border: '0.5px solid var(--color-border-tertiary)',
                borderRadius: 'var(--border-radius-md)',
                background: 'var(--color-background-primary)',
                marginTop: 6,
              }}
            />
            <div className="autocomplete modal-autocomplete">
              <div className="autocomplete-header">
                {suggestionHeaderLabel(sugg.length)}
              </div>
              {sugg.length > 0 && (
                <div
                  id={suggestionListId}
                  role="listbox"
                  aria-label="Tag suggestions"
                  className="autocomplete-list"
                >
                  {sugg.map((s, i) => (
                    <button
                      type="button"
                      key={s.tag}
                      id={`${suggestionListId}-option-${i}`}
                      role="option"
                      aria-selected={i === focusIdx}
                      className={`autocomplete-row${i === focusIdx ? ' focus' : ''}`}
                      onMouseEnter={() => setFocusIdx(i)}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => addTag(s.tag)}
                      tabIndex={-1}
                      data-testid={`sugg-${s.tag}`}
                    >
                      <div className="row1">
                        <span>{s.tag}</span>
                        <span className="badge">{s.category}</span>
                      </div>
                      <div className="def">{s.def}</div>
                    </button>
                  ))}
                </div>
              )}
              <div className="autocomplete-footer">
                <Icon name="info-circle" size={12} /> No match? Request a new
                tag in the Tags tab.
              </div>
            </div>
          </div>

          {totalChanges > 0 && (
            <RetagPreviewImpact
              isRemoving={isRemoving}
              pendingAdd={pendingAdd}
              pendingRemove={pendingRemove}
              previewChunks={previewChunks}
              previewDocs={previewDocs}
            />
          )}
        </div>

        <div className="modal-footer">
          <div className="status" />
          <div className="actions">
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              className="btn primary"
              disabled={totalChanges === 0}
              onClick={submit}
            >
              {applyLabel}
            </button>
          </div>
        </div>
      </dialog>
    </div>
  );
}
