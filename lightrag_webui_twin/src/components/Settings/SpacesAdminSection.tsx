/**
 * Settings → Spaces (Admin CRUD).
 *
 * Lists every Twin space currently provisioned (env seed + runtime
 * additions) and lets an operator add, edit, or delete the runtime
 * ones. Env-seeded spaces are marked read-only with an explicit badge
 * so the operator can't waste a click on them.
 *
 * Backend contract:
 *   GET    /twin/api/spaces           → readonly Workspace[]
 *   POST   /twin/api/spaces           → 201 / 409 / 422
 *   PATCH  /twin/api/spaces/{id}      → 200 / 403 / 404
 *   DELETE /twin/api/spaces/{id}      → 204 / 403 / 404 / 409
 *
 * The 409 returned by DELETE means the space still has data (docs
 * or tags) — we surface it as an inline warning instead of silently
 * dropping the click.
 */

import { useEffect, useState } from 'react';
import { Icon } from '../Icon';
import {
  useCreateSpace,
  useDeleteSpace,
  useSpaces,
  useUpdateSpace,
} from '../../api/queries';
import { ApiError } from '../../api/client';
import { canManageSpaces } from '../../lib/permissions';
import type { AuthenticatedUser } from '../../types/auth';
import type { Toast } from '../../types/toast';
import type { Workspace } from '../../types/topbar';

const SPACE_ID_RE = /^[A-Za-z0-9_-]+$/;
const MAX_SPACES = 5;

export interface SpacesAdminSectionProps {
  user?: AuthenticatedUser | null;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
  /** When the operator deletes the currently-active space, the host
   *  needs to swap to a fallback. Returning the new active id is
   *  optional — leave undefined to let the host pick. */
  onActiveSpaceDeleted?: (deletedId: string) => void;
}

export function SpacesAdminSection({
  user = null,
  onToast,
  onActiveSpaceDeleted,
}: SpacesAdminSectionProps = {}) {
  const spaces = useSpaces();
  const createSpace = useCreateSpace();
  const updateSpace = useUpdateSpace();
  const deleteSpace = useDeleteSpace();

  const [addOpen, setAddOpen] = useState(false);
  const list = spaces.data ?? [];
  const atMax = list.length >= MAX_SPACES;
  const canManage = canManageSpaces(user);

  const notifyAdminScope403 = (err: unknown) => {
    if (!(err instanceof ApiError) || err.status !== 403) return;
    onToast?.({
      kind: 'error',
      title: 'Admin scope required',
      sub: "Admin scope 'admin:spaces' required",
    });
  };

  return (
    <div className="settings-section" data-testid="settings-spaces-admin">
      <h3>Spaces</h3>
      <p className="muted">
        Runtime Twin spaces. The SRE-provisioned default is read-only and
        managed via the deploy env — operator additions are persisted to
        the runtime catalog.
      </p>
      {!canManage && (
        <span className="env-badge" data-testid="spaces-admin-readonly-badge">
          <Icon name="lock" size={10} /> Read-only — admin scope required
        </span>
      )}

      {canManage && (
        addOpen ? (
          <AddSpaceForm
            existingIds={list.map((s) => s.id)}
            pending={createSpace.isPending}
            error={errorToMessage(createSpace.error)}
            onCancel={() => setAddOpen(false)}
            onSubmit={(payload) => {
              createSpace.mutate(payload, {
                onSuccess: () => setAddOpen(false),
                onError: notifyAdminScope403,
              });
            }}
          />
        ) : (
          <div style={{ margin: '8px 0' }}>
            <button
              type="button"
              className="ghost-btn primary"
              onClick={() => setAddOpen(true)}
              disabled={atMax}
              data-testid="settings-add-space-btn"
              title={
                atMax
                  ? `Cannot add more spaces — already at the cap of ${MAX_SPACES}.`
                  : undefined
              }
            >
              <Icon name="plus" size={12} /> Add space
            </button>
            {atMax && (
              <span
                className="muted"
                style={{ marginLeft: 10, fontSize: 12 }}
                data-testid="settings-spaces-at-max"
              >
                At max — {MAX_SPACES} space cap reached.
              </span>
            )}
          </div>
        )
      )}

      <ul className="settings-spaces-list" style={{ padding: 0, margin: 0 }}>
        {list.map((space, idx) => {
          // First fixture / first env-injected entry is the
          // SRE-provisioned default — the picker uses `current` for
          // selection but the catalog returns env first.
          const envSeeded = idx === 0;
          return (
            <SpaceRow
              key={space.id}
              space={space}
              envSeeded={envSeeded}
              pendingEdit={updateSpace.isPending}
              pendingDelete={deleteSpace.isPending}
              updateError={errorToMessage(updateSpace.error)}
              deleteError={errorToMessage(deleteSpace.error)}
              canManage={canManage}
              onSave={(patch) =>
                updateSpace.mutate(
                  { id: space.id, patch },
                  { onError: notifyAdminScope403 },
                )
              }
              onDelete={() =>
                deleteSpace.mutate(space.id, {
                  onSuccess: () => {
                    if (space.current) onActiveSpaceDeleted?.(space.id);
                  },
                  onError: notifyAdminScope403,
                })
              }
            />
          );
        })}
      </ul>
    </div>
  );
}

// ─── Add Space form ───────────────────────────────────────────────────
interface AddSpaceFormProps {
  existingIds: readonly string[];
  pending: boolean;
  error: string | null;
  onCancel: () => void;
  onSubmit: (payload: {
    id: string;
    label: string;
    kind: string;
    description?: string;
  }) => void;
}

function AddSpaceForm({
  existingIds,
  pending,
  error,
  onCancel,
  onSubmit,
}: AddSpaceFormProps) {
  const [id, setId] = useState('');
  const [label, setLabel] = useState('');
  const [kind, setKind] = useState('custom');
  const [description, setDescription] = useState('');

  const trimmedId = id.trim();
  const trimmedLabel = label.trim();
  const idValid = trimmedId.length > 0 && SPACE_ID_RE.test(trimmedId);
  const duplicate = idValid && existingIds.includes(trimmedId);
  const canSubmit =
    idValid && !duplicate && trimmedLabel.length > 0 && !pending;

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      id: trimmedId,
      label: trimmedLabel,
      kind: kind.trim() || 'custom',
      description: description.trim() || undefined,
    });
  };

  return (
    <form
      className="set-card"
      data-testid="settings-add-space-form"
      onSubmit={submit}
      style={{ display: 'flex', flexDirection: 'column', gap: 10 }}
    >
      <div className="set-card-h">
        Add space{' '}
        <span className="env-badge">
          <Icon name="plus" size={10} /> runtime
        </span>
      </div>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          ID <em>(alphanumeric, underscore or dash)</em>
        </span>
        <input
          type="text"
          value={id}
          onChange={(e) => setId(e.target.value)}
          placeholder="sandbox"
          autoFocus
          aria-label="New space id"
          data-testid="settings-add-space-id"
          className="mono"
        />
      </label>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Display name
        </span>
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Sandbox"
          aria-label="New space label"
          data-testid="settings-add-space-label"
        />
      </label>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Kind
        </span>
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value)}
          aria-label="New space kind"
          data-testid="settings-add-space-kind"
        >
          <option value="custom">Custom</option>
          <option value="sandbox">Sandbox</option>
          <option value="project">Project</option>
          <option value="team">Team</option>
        </select>
      </label>
      <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>
          Description <em>(optional)</em>
        </span>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What is this space for?"
          aria-label="New space description"
          data-testid="settings-add-space-description"
        />
      </label>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button type="button" className="ghost-btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="submit"
          className="ghost-btn primary"
          disabled={!canSubmit}
          data-testid="settings-add-space-submit"
        >
          <Icon name="check" size={11} /> {pending ? 'Adding…' : 'Add'}
        </button>
      </div>
      {!idValid && trimmedId.length > 0 && (
        <div
          role="alert"
          style={{ fontSize: 11, color: 'var(--twin-red-vivid, #b03060)' }}
          data-testid="settings-add-space-id-invalid"
        >
          Invalid id — use only alphanumeric characters, underscore or dash.
        </div>
      )}
      {duplicate && (
        <div
          role="alert"
          style={{ fontSize: 11, color: 'var(--twin-red-vivid, #b03060)' }}
          data-testid="settings-add-space-duplicate"
        >
          A space with id “{trimmedId}” already exists.
        </div>
      )}
      {error && (
        <div
          role="alert"
          style={{ fontSize: 11, color: 'var(--twin-red-vivid, #b03060)' }}
          data-testid="settings-add-space-error"
        >
          {error}
        </div>
      )}
    </form>
  );
}

// ─── Row: shows + edits + deletes one space ───────────────────────────
interface SpaceRowProps {
  space: Workspace;
  envSeeded: boolean;
  pendingEdit: boolean;
  pendingDelete: boolean;
  updateError: string | null;
  deleteError: string | null;
  canManage: boolean;
  onSave: (patch: { label?: string }) => void;
  onDelete: () => void;
}

function SpaceRow({
  space,
  envSeeded,
  pendingEdit,
  pendingDelete,
  updateError,
  deleteError,
  canManage,
  onSave,
  onDelete,
}: SpaceRowProps) {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(space.kb);
  const [armedDelete, setArmedDelete] = useState(false);

  // Reset edit state when the cached space object changes (after a
  // refetch following a successful save).
  /* eslint-disable react-hooks/set-state-in-effect -- intentional re-sync with the new server-side value. */
  useEffect(() => {
    setLabel(space.kb);
    setEditing(false);
  }, [space.id, space.kb]);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!armedDelete) return;
    const t = window.setTimeout(() => setArmedDelete(false), 4000);
    return () => window.clearTimeout(t);
  }, [armedDelete]);

  return (
    <li
      className="set-card"
      data-testid={`settings-space-row-${space.id}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        marginBottom: 10,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexWrap: 'wrap',
        }}
      >
        <code
          className="mono"
          style={{ fontSize: 12, fontWeight: 600 }}
          data-testid={`settings-space-id-${space.id}`}
        >
          {space.id}
        </code>
        {space.current && (
          <span
            className="env-badge"
            style={{
              background: 'var(--twin-accent-soft, rgba(56,113,180,0.12))',
              color: 'var(--twin-accent, #3871b4)',
            }}
          >
            active
          </span>
        )}
        {envSeeded ? (
          <span className="env-badge">
            <Icon name="lock" size={10} /> env-seeded
          </span>
        ) : (
          <span className="env-badge">
            <Icon name="circle-dot" size={10} /> runtime
          </span>
        )}
      </div>

      {editing ? (
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            aria-label={`Edit label for space ${space.id}`}
            data-testid={`settings-space-edit-label-${space.id}`}
            style={{ flex: 1 }}
          />
          <button
            type="button"
            className="ghost-btn"
            onClick={() => {
              setLabel(space.kb);
              setEditing(false);
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            className="ghost-btn primary"
            disabled={label.trim().length === 0 || pendingEdit}
            onClick={() => onSave({ label: label.trim() })}
            data-testid={`settings-space-save-${space.id}`}
          >
            <Icon name="check" size={11} /> {pendingEdit ? 'Saving…' : 'Save'}
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ flex: 1, fontSize: 13 }}>{space.kb}</span>
          {!envSeeded && canManage && (
            <>
              <button
                type="button"
                className="ghost-btn small"
                onClick={() => setEditing(true)}
                data-testid={`settings-space-edit-${space.id}`}
              >
                <Icon name="edit" size={11} /> Edit
              </button>
              <button
                type="button"
                className={armedDelete ? 'ghost-btn danger' : 'ghost-btn small'}
                onClick={() => {
                  if (!armedDelete) {
                    setArmedDelete(true);
                    return;
                  }
                  onDelete();
                  setArmedDelete(false);
                }}
                disabled={pendingDelete}
                data-testid={`settings-space-delete-${space.id}`}
                style={
                  armedDelete
                    ? {
                        color: 'var(--twin-red-vivid, #b03060)',
                        borderColor: 'var(--twin-red-vivid, #b03060)',
                      }
                    : undefined
                }
              >
                <Icon
                  name={armedDelete ? 'alert-triangle' : 'trash'}
                  size={11}
                />{' '}
                {pendingDelete
                  ? 'Deleting…'
                  : armedDelete
                    ? 'Click again'
                    : 'Delete'}
              </button>
            </>
          )}
        </div>
      )}

      {(updateError || deleteError) && (
        <div
          role="alert"
          style={{ fontSize: 11, color: 'var(--twin-red-vivid, #b03060)' }}
          data-testid={`settings-space-error-${space.id}`}
        >
          {updateError ?? deleteError}
        </div>
      )}
    </li>
  );
}

function errorToMessage(err: unknown): string | null {
  if (err === null || err === undefined) return null;
  if (err instanceof ApiError) {
    const detail = (err.body as { detail?: string } | undefined)?.detail;
    return detail || err.message;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}
