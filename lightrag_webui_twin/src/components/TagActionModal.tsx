/**
 * TagActionModal — 8-kind modal dispatch for tag governance actions.
 *
 * Ported from Desktop/UI/tags.jsx (the `TagActionModal` private component).
 *
 * The 8 kinds map to tag catalog governance workflows:
 *   edit         — palier-3 in-place edit of a tag's definition.
 *   suggest      — palier-2 proposes an edit, queued for palier-3 review.
 *   synonyms     — palier-3 manages the alias list at the gateway.
 *   deprecate    — flag the tag as excluded from default retrieval.
 *   delete       — palier-3 destructive action; requires a migration strategy.
 *   reject       — palier-3 rejects a pending tag request with a reason.
 *   edit-approve — palier-3 tweaks then accepts a request.
 *   request      — anyone proposes a new Tier-3 tag.
 *
 * Like the other modals (RetagModal, AddSourceModal) this is a controlled
 * component that emits a structured *Action on commit instead of toasts —
 * the host (TagsTab → App.tsx) owns the toast queue.
 */

import { useMemo, useRef, useState } from 'react';
import { Icon } from './Icon';
import { StatusBadge } from './StatusBadge';
import { useModalA11y } from '../hooks/useModalA11y';
import type {
  TagAction,
  TagActionKind,
  TagCategory,
  TagEntry,
} from '../types/tag';

export interface TagActionCommit {
  kind: TagActionKind;
  tag: TagEntry | null;
  /** Proposed canonical name (only for `request` / `edit` / `edit-approve`). */
  name?: string;
  /** Definition text for request/edit flows. */
  def?: string;
  /** Optional longer governance description for edit flows. */
  longDescription?: string;
  /** Governance domain/category for request/edit flows. */
  category?: string;
  /** Reason for `reject`. */
  reason?: string;
  /** Request/suggest justification text. */
  justification?: string;
  /** New synonym to add (only for `synonyms`). */
  newSynonym?: string;
  /** Full synonym list when managing aliases. */
  aliases?: readonly string[];
  /** Delete-only: migration strategy + replacement tag. */
  migrate?: { strategy: 'migrate' | 'untag'; to?: string };
}

export interface TagActionModalProps {
  action: TagAction;
  allTags: readonly TagEntry[];
  categories: readonly TagCategory[];
  onClose: () => void;
  onCommit: (commit: TagActionCommit) => void;
}

const TITLE_MAP: Record<TagActionKind, string> = {
  edit: 'Edit tag',
  suggest: 'Suggest tag edit',
  synonyms: 'Manage synonyms',
  deprecate: 'Deprecate tag',
  delete: 'Delete tag',
  reject: 'Reject request',
  'edit-approve': 'Edit & approve request',
  request: 'Request new tag',
};

const SUBMIT_LABEL: Record<TagActionKind, (state: { migrateStrategy: 'migrate' | 'untag' }) => string> = {
  edit: () => 'Save',
  suggest: () => 'Submit suggestion',
  synonyms: () => 'Save synonyms',
  deprecate: () => 'Deprecate',
  delete: (s) => (s.migrateStrategy === 'migrate' ? 'Migrate and delete' : 'Untag and delete'),
  reject: () => 'Reject request',
  'edit-approve': () => 'Approve with edits',
  request: () => 'Submit request',
};

export function TagActionModal({
  action,
  allTags,
  categories,
  onClose,
  onCommit,
}: Readonly<TagActionModalProps>) {
  const modalRef = useRef<HTMLDialogElement>(null);
  useModalA11y({ open: true, onClose, ref: modalRef });

  const tag = action.tag ?? null;
  const [name, setName] = useState(tag?.tag ?? '');
  const [definition, setDefinition] = useState(tag?.def ?? '');
  const [longDescription, setLongDescription] = useState(
    tag?.long_description ?? '',
  );
  const [category, setCategory] = useState(tag?.category ?? categories[0]?.id ?? '');
  const [migrateTo, setMigrateTo] = useState('');
  const [migrateStrategy, setMigrateStrategy] = useState<'migrate' | 'untag'>('migrate');
  const [newSyn, setNewSyn] = useState('');
  const [aliases, setAliases] = useState<readonly string[]>(tag?.aliases ?? []);
  const [reason, setReason] = useState('');
  const [requestSynonyms, setRequestSynonyms] = useState('');
  const [justification, setJustification] = useState('');

  const eligible = useMemo(
    () =>
      tag
        ? allTags.filter(
            (x) => x.tag !== tag.tag && x.tier !== 'requested' && x.status === 'active',
          )
        : [],
    [allTags, tag],
  );

  const isDanger = action.kind === 'delete' || action.kind === 'reject';
  const submitLabel = SUBMIT_LABEL[action.kind]({ migrateStrategy });

  const commit = () => {
    const payload: TagActionCommit = { kind: action.kind, tag };
    if (action.kind === 'request' || action.kind === 'edit' || action.kind === 'edit-approve') {
      payload.name = name;
      payload.def = definition;
      payload.longDescription = longDescription;
      payload.category = category;
    }
    if (action.kind === 'synonyms') {
      const nextAliases = newSyn.trim()
        ? [...aliases, newSyn.trim()]
        : [...aliases];
      payload.aliases = Array.from(new Set(nextAliases));
      if (newSyn.trim()) {
        payload.newSynonym = newSyn.trim();
      }
    }
    if (action.kind === 'request') {
      payload.aliases = requestSynonyms
        .split(',')
        .map((alias) => alias.trim())
        .filter(Boolean);
      payload.justification = justification.trim();
    }
    if (action.kind === 'reject') {
      payload.reason = reason.trim();
    }
    if (action.kind === 'delete') {
      payload.migrate = {
        strategy: migrateStrategy,
        to: migrateStrategy === 'migrate' ? migrateTo : undefined,
      };
    }
    onCommit(payload);
  };

  // Submit gate — block destructive submits until prerequisites are met.
  const submitDisabled =
    (action.kind === 'reject' && !reason.trim()) ||
    (action.kind === 'request' && (!name.trim() || !definition.trim())) ||
    (action.kind === 'delete' && migrateStrategy === 'migrate' && !migrateTo);

  return (
    <div
      className="modal-bg"
      onClick={(e) => {
        if (e.currentTarget === e.target) onClose();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
      }}
      data-testid="tagaction-backdrop"
    >
      <dialog
        open
        ref={modalRef}
        className="modal tag-action-modal"
        aria-modal="true"
        aria-labelledby="tagaction-title"
        tabIndex={-1}
      >
        <div className="modal-h">
          <h3 id="tagaction-title">{TITLE_MAP[action.kind]}</h3>
          {tag && action.kind !== 'request' && (
            <div className="modal-h-sub">
              <code>{tag.tag}</code>
              <StatusBadge status={tag.status} />
              <span className="dot-sep">·</span>
              <span>
                {tag.sources_count} docs · {tag.chunks_count.toLocaleString()} chunks
              </span>
            </div>
          )}
          <button className="modal-x" onClick={onClose} aria-label="Close dialog">
            <Icon name="x" size={14} />
          </button>
        </div>

        <div className="modal-body">
          {(action.kind === 'edit' ||
            action.kind === 'suggest' ||
            action.kind === 'edit-approve') &&
            tag && (
              <>
                <label className="field-label" htmlFor="tagaction-name">
                  Name (canonical)
                </label>
                <input
                  id="tagaction-name"
                  className="text-input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={action.kind !== 'edit-approve' && action.kind !== 'edit'}
                />
                <label className="field-label" htmlFor="tagaction-def">
                  Short definition
                </label>
                <textarea
                  id="tagaction-def"
                  className="text-input"
                  rows={3}
                  value={definition}
                  onChange={(e) => setDefinition(e.target.value)}
                />
                <label className="field-label" htmlFor="tagaction-longdef">
                  Long description (optional)
                </label>
                <textarea
                  id="tagaction-longdef"
                  className="text-input"
                  rows={3}
                  value={longDescription}
                  onChange={(e) => setLongDescription(e.target.value)}
                  placeholder="For complex tags — surfaced in autocomplete tooltip."
                />
                <div className="impact-box">
                  <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                  <span>
                    {action.kind === 'suggest' &&
                      'Your edit will be queued for palier-3 review. Existing chunks remain untouched until approval.'}
                    {action.kind === 'edit' &&
                      'Definition changes are non-destructive — only the autocomplete and detail panel update. Synonyms and structure are managed separately.'}
                    {action.kind === 'edit-approve' && (
                      <>
                        You can tweak the proposed definition before accepting. The tag will
                        enter the tag catalog as <b>active</b>.
                      </>
                    )}
                  </span>
                </div>
              </>
            )}

          {action.kind === 'synonyms' && tag && (
            <>
              <label className="field-label">Current synonyms</label>
              <div className="alias-chips">
                {aliases.length === 0 && <span className="muted">No synonyms.</span>}
                {aliases.map((a) => (
                  <span key={a} className="alias-chip">
                    <code>{a}</code>
                    <button
                      type="button"
                      aria-label={`Remove synonym ${a}`}
                      onClick={() =>
                        setAliases((current) => current.filter((x) => x !== a))
                      }
                    >
                      <Icon name="x" size={10} />
                    </button>
                  </span>
                ))}
              </div>
              <label className="field-label" htmlFor="tagaction-newsyn">
                Add synonym
              </label>
              <input
                id="tagaction-newsyn"
                className="text-input"
                value={newSyn}
                onChange={(e) => setNewSyn(e.target.value)}
                placeholder="e.g. recovery-manager"
              />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>
                  Synonyms are matched by query rewriting at the gateway. They do not
                  duplicate the index; the canonical tag is preserved on chunks.
                </span>
              </div>
            </>
          )}

          {action.kind === 'deprecate' && tag && (
            <>
              <div className="impact-box warning">
                <Icon name="alert-triangle" size={13} color="var(--twin-amber-vivid)" />
                <span>
                  Deprecating <code>{tag.tag}</code> excludes its {tag.sources_count} docs
                  from default retrieval. Existing tags on chunks are preserved; queries need{' '}
                  <code>include_deprecated: true</code> to surface them.
                </span>
              </div>
              <label className="field-label" htmlFor="tagaction-depreason">
                Reason (optional)
              </label>
              <textarea
                id="tagaction-depreason"
                className="text-input"
                rows={3}
                placeholder="e.g. Superseded by iso20022 — see HLA §4.2"
              />
            </>
          )}

          {action.kind === 'delete' && tag && (
            <>
              <div className="impact-box danger">
                <Icon name="alert-triangle" size={13} color="var(--twin-red-vivid)" />
                <span>
                  <b>{tag.tag}</b> is used on <b>{tag.sources_count} docs</b> (
                  {tag.chunks_count.toLocaleString()} chunks). Deletion cannot be undone —
                  choose a migration strategy:
                </span>
              </div>
              <div className="strategy-radios">
                <label
                  className={'strategy ' + (migrateStrategy === 'migrate' ? 'is-active' : '')}
                >
                  <input
                    type="radio"
                    name="migrate-strategy"
                    checked={migrateStrategy === 'migrate'}
                    onChange={() => setMigrateStrategy('migrate')}
                  />
                  <div>
                    <div className="strategy-h">Migrate to another tag</div>
                    <div className="strategy-sub">
                      Re-tag all {tag.sources_count} docs with a replacement.
                    </div>
                    {migrateStrategy === 'migrate' && (
                      <select
                        className="text-input mt8"
                        value={migrateTo}
                        onChange={(e) => setMigrateTo(e.target.value)}
                        aria-label="Replacement tag"
                      >
                        <option value="">— select replacement —</option>
                        {eligible.map((x) => (
                          <option key={x.tag} value={x.tag}>
                            {x.tag} ({x.sources_count} docs)
                          </option>
                        ))}
                      </select>
                    )}
                  </div>
                </label>
                <label
                  className={'strategy ' + (migrateStrategy === 'untag' ? 'is-active' : '')}
                >
                  <input
                    type="radio"
                    name="migrate-strategy"
                    checked={migrateStrategy === 'untag'}
                    onChange={() => setMigrateStrategy('untag')}
                  />
                  <div>
                    <div className="strategy-h">Untag and delete</div>
                    <div className="strategy-sub">
                      Docs lose the tag and become untagged on this axis.
                    </div>
                  </div>
                </label>
              </div>
            </>
          )}

          {action.kind === 'reject' && tag && (
            <>
              <label className="field-label" htmlFor="tagaction-reason">
                Reason
              </label>
              <textarea
                id="tagaction-reason"
                className="text-input"
                rows={3}
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="The author of the request will receive this message. Be specific."
              />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>
                  A <code>tag.rejected</code> event is emitted to Activity with this
                  reason. The requester is notified by email.
                </span>
              </div>
            </>
          )}

          {action.kind === 'request' && (
            <>
              <label className="field-label" htmlFor="tagaction-reqname">
                Proposed name <span className="hint">lowercase, no spaces</span>
              </label>
              <input
                id="tagaction-reqname"
                className="text-input"
                value={name}
                onChange={(e) =>
                  setName(e.target.value.toLowerCase().replace(/\s+/g, '-'))
                }
                placeholder="e.g. argocd"
              />
              <label className="field-label" htmlFor="tagaction-reqdef">
                Definition <span className="hint">200 chars max</span>
              </label>
              <textarea
                id="tagaction-reqdef"
                className="text-input"
                rows={3}
                maxLength={200}
                value={definition}
                onChange={(e) => setDefinition(e.target.value)}
                placeholder="What should this tag mean? When should it be applied?"
              />
              <label className="field-label" htmlFor="tagaction-reqlongdef">
                Long description <span className="hint">optional</span>
              </label>
              <textarea
                id="tagaction-reqlongdef"
                className="text-input"
                rows={3}
                value={longDescription}
                onChange={(e) => setLongDescription(e.target.value)}
                placeholder="Add governance notes, examples, or boundary cases for reviewers."
              />
              <label className="field-label" htmlFor="tagaction-reqdomain">
                Domain
              </label>
              <select
                id="tagaction-reqdomain"
                className="text-input"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              >
                {categories.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
                <option value="other">Other (specify in justification)</option>
              </select>
              <label className="field-label" htmlFor="tagaction-reqsyn">
                Synonyms <span className="hint">optional</span>
              </label>
              <input
                id="tagaction-reqsyn"
                className="text-input"
                value={requestSynonyms}
                onChange={(e) => setRequestSynonyms(e.target.value)}
                placeholder="comma-separated, e.g. recovery-manager, backup-tool"
              />
              <label className="field-label" htmlFor="tagaction-reqjustif">
                Justification
              </label>
              <textarea
                id="tagaction-reqjustif"
                className="text-input"
                rows={3}
                value={justification}
                onChange={(e) => setJustification(e.target.value)}
                placeholder="Why is the existing taxonomy insufficient? Cite an example use."
              />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>
                  Visibility is auto-set to <code>private</code> (inherited from
                  folder). Requests reach a palier-3 reviewer within 2 business days.
                  An accepted tag enters as <b>active</b>.
                </span>
              </div>
            </>
          )}
        </div>

        <div className="modal-footer">
          <button className="ghost-btn" onClick={onClose}>
            Cancel
          </button>
          <button
            className={'primary-btn' + (isDanger ? ' danger' : '')}
            onClick={commit}
            disabled={submitDisabled}
          >
            {submitLabel}
          </button>
        </div>
      </dialog>
    </div>
  );
}
