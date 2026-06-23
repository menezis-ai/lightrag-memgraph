/**
 * Settings → Folders (Admin CRUD).
 *
 * Lists every Twin folder currently provisioned (env seed + runtime
 * additions) and lets an operator add, edit, or delete the runtime
 * ones. Env-seeded folders are marked read-only with an explicit badge
 * so the operator can't waste a click on them.
 *
 * Backend contract:
 *   GET    /twin/api/folders          → readonly Folder[]
 *   POST   /twin/api/folders          → 201 / 409 / 422
 *   PATCH  /twin/api/folders/{id}     → 200 / 403 / 404
 *   DELETE /twin/api/folders/{id}     → 204 / 403 / 404 / 409
 *
 * The 409 returned by DELETE means the folder still has data (docs
 * or tags) — we surface it as an inline warning instead of silently
 * dropping the click.
 */

import { useEffect, useState } from 'react';
import { Icon } from '../Icon';
import {
  useCreateFolder,
  useDeleteFolder,
  useFolders,
  useUpdateFolder,
} from '../../api/queries';
import { ApiError } from '../../api/client';
import { canManageFolders } from '../../lib/permissions';
import type { AuthenticatedUser } from '../../types/auth';
import type { Toast } from '../../types/toast';
import type { Folder } from '../../types/topbar';

const FOLDER_ID_RE = /^[A-Za-z0-9_-]+$/;
const MAX_FOLDERS = 5;

function folderDeleteLabel(pendingDelete: boolean, armedDelete: boolean): string {
  if (pendingDelete) return 'Deleting…';
  if (armedDelete) return 'Click again';
  return 'Delete';
}

export interface FoldersAdminSectionProps {
  user?: AuthenticatedUser | null;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
  /** When the operator deletes the currently-active folder, the host
   *  needs to swap to a fallback. Returning the new active id is
   *  optional — leave undefined to let the host pick. */
  onActiveFolderDeleted?: (deletedId: string) => void;
}

export function FoldersAdminSection({
  user = null,
  onToast,
  onActiveFolderDeleted,
}: FoldersAdminSectionProps = {}) {
  const folders = useFolders();
  const createFolder = useCreateFolder();
  const updateFolder = useUpdateFolder();
  const deleteFolder = useDeleteFolder();

  const [addOpen, setAddOpen] = useState(false);
  const list = folders.data ?? [];
  const atMax = list.length >= MAX_FOLDERS;
  const canManage = canManageFolders(user);

  const notifyAdminScope403 = (err: unknown) => {
    if (!(err instanceof ApiError) || err.status !== 403) return;
    onToast?.({
      kind: 'error',
      title: 'Admin scope required',
      sub: "Admin scope 'admin:folders' required",
    });
  };

  return (
    <div className="settings-section" data-testid="settings-folders-admin">
      <h3>Folders</h3>
      <p className="muted">
        Runtime Twin folders. The SRE-provisioned default is read-only and
        managed via the deploy env — operator additions are persisted to
        the runtime catalog.
      </p>
      {!canManage && (
        <span className="env-badge" data-testid="folders-admin-readonly-badge">
          <Icon name="lock" size={10} /> Read-only — admin scope required
        </span>
      )}

      {canManage && (
        addOpen ? (
          <AddFolderForm
            existingIds={list.map((s) => s.id)}
            pending={createFolder.isPending}
            error={errorToMessage(createFolder.error)}
            onCancel={() => setAddOpen(false)}
            onSubmit={(payload) => {
              createFolder.mutate(payload, {
                onSuccess: () => setAddOpen(false),
                onError: notifyAdminScope403,
              });
            }}
          />
        ) : (
          <div className="settings-folder-add-row">
            <button
              type="button"
              className="ghost-btn primary"
              onClick={() => setAddOpen(true)}
              disabled={atMax}
              data-testid="settings-add-folder-btn"
              title={
                atMax
                  ? `Cannot add more folders — already at the cap of ${MAX_FOLDERS}.`
                  : undefined
              }
            >
              <Icon name="plus" size={12} /> Add folder
            </button>
            {atMax && (
              <span
                className="muted"
                data-testid="settings-folders-at-max"
              >
                At max — {MAX_FOLDERS} folder cap reached.
              </span>
            )}
          </div>
        )
      )}

      <ul className="settings-folders-list">
        {list.map((folder, idx) => {
          // First fixture / first env-injected entry is the
          // SRE-provisioned default — the picker uses `current` for
          // selection but the catalog returns env first.
          const envSeeded = idx === 0;
          return (
            <FolderRow
              key={folder.id}
              folder={folder}
              envSeeded={envSeeded}
              pendingEdit={updateFolder.isPending}
              pendingDelete={deleteFolder.isPending}
              updateError={errorToMessage(updateFolder.error)}
              deleteError={errorToMessage(deleteFolder.error)}
              canManage={canManage}
              onSave={(patch) =>
                updateFolder.mutate(
                  { id: folder.id, patch },
                  { onError: notifyAdminScope403 },
                )
              }
              onDelete={() =>
                deleteFolder.mutate(folder.id, {
                  onSuccess: () => {
                    if (folder.current) onActiveFolderDeleted?.(folder.id);
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

// ─── Add Folder form ──────────────────────────────────────────────────
interface AddFolderFormProps {
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

function AddFolderForm({
  existingIds,
  pending,
  error,
  onCancel,
  onSubmit,
}: Readonly<AddFolderFormProps>) {
  const [id, setId] = useState('');
  const [label, setLabel] = useState('');
  const [kind, setKind] = useState('custom');
  const [description, setDescription] = useState('');

  const trimmedId = id.trim();
  const trimmedLabel = label.trim();
  const idValid = trimmedId.length > 0 && FOLDER_ID_RE.test(trimmedId);
  const duplicate = idValid && existingIds.includes(trimmedId);
  const canSubmit =
    idValid && !duplicate && trimmedLabel.length > 0 && !pending;

  const submit = (e?: React.SyntheticEvent<HTMLFormElement>) => {
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
      className="set-card folder-form"
      data-testid="settings-add-folder-form"
      onSubmit={submit}
    >
      <div className="set-card-h">
        Add folder{' '}
        <span className="env-badge">
          <Icon name="plus" size={10} /> runtime
        </span>
      </div>
      <label className="settings-field">
        <span>
          ID <em>(alphanumeric, underscore or dash)</em>
        </span>
        <input
          type="text"
          value={id}
          onChange={(e) => setId(e.target.value)}
          placeholder="sandbox"
          autoFocus
          aria-label="New folder id"
          data-testid="settings-add-folder-id"
          className="mono"
        />
      </label>
      <label className="settings-field">
        <span>Display name</span>
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Sandbox"
          aria-label="New folder label"
          data-testid="settings-add-folder-label"
        />
      </label>
      <label className="settings-field">
        <span>Kind</span>
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value)}
          aria-label="New folder kind"
          data-testid="settings-add-folder-kind"
        >
          <option value="custom">Custom</option>
          <option value="sandbox">Sandbox</option>
          <option value="project">Project</option>
          <option value="team">Team</option>
        </select>
      </label>
      <label className="settings-field">
        <span>
          Description <em>(optional)</em>
        </span>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What is this folder for?"
          aria-label="New folder description"
          data-testid="settings-add-folder-description"
        />
      </label>
      <div className="settings-form-actions">
        <button type="button" className="ghost-btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="submit"
          className="ghost-btn primary"
          disabled={!canSubmit}
          data-testid="settings-add-folder-submit"
        >
          <Icon name="check" size={11} /> {pending ? 'Adding…' : 'Add'}
        </button>
      </div>
      {!idValid && trimmedId.length > 0 && (
        <div
          role="alert"
          className="settings-error"
          data-testid="settings-add-folder-id-invalid"
        >
          Invalid id — use only alphanumeric characters, underscore or dash.
        </div>
      )}
      {duplicate && (
        <div
          role="alert"
          className="settings-error"
          data-testid="settings-add-folder-duplicate"
        >
          A folder with id “{trimmedId}” already exists.
        </div>
      )}
      {error && (
        <div
          role="alert"
          className="settings-error"
          data-testid="settings-add-folder-error"
        >
          {error}
        </div>
      )}
    </form>
  );
}

// ─── Row: shows + edits + deletes one folder ──────────────────────────
interface FolderRowProps {
  folder: Folder;
  envSeeded: boolean;
  pendingEdit: boolean;
  pendingDelete: boolean;
  updateError: string | null;
  deleteError: string | null;
  canManage: boolean;
  onSave: (patch: { label?: string }) => void;
  onDelete: () => void;
}

function FolderRow({
  folder,
  envSeeded,
  pendingEdit,
  pendingDelete,
  updateError,
  deleteError,
  canManage,
  onSave,
  onDelete,
}: Readonly<FolderRowProps>) {
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(folder.kb);
  const [armedDelete, setArmedDelete] = useState(false);

  // Reset edit state when the cached folder object changes (after a
  // refetch following a successful save).
  /* eslint-disable react-hooks/set-state-in-effect -- intentional re-sync with the new server-side value. */
  useEffect(() => {
    setLabel(folder.kb);
    setEditing(false);
  }, [folder.id, folder.kb]);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!armedDelete) return;
    const t = globalThis.setTimeout(() => setArmedDelete(false), 4000);
    return () => globalThis.clearTimeout(t);
  }, [armedDelete]);

  return (
    <li
      className="set-card settings-folder-row"
      data-testid={`settings-folder-row-${folder.id}`}
    >
      <div className="settings-folder-row-head">
        <code
          className="mono"
          data-testid={`settings-folder-id-${folder.id}`}
        >
          {folder.id}
        </code>
        {folder.current && (
          <span className="env-badge active">
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
        <div className="settings-folder-edit-row">
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            aria-label={`Edit label for folder ${folder.id}`}
            data-testid={`settings-folder-edit-label-${folder.id}`}
          />
          <button
            type="button"
            className="ghost-btn"
            onClick={() => {
              setLabel(folder.kb);
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
            data-testid={`settings-folder-save-${folder.id}`}
          >
            <Icon name="check" size={11} /> {pendingEdit ? 'Saving…' : 'Save'}
          </button>
        </div>
      ) : (
        <div className="settings-folder-row-body">
          <span>{folder.kb}</span>
          {!envSeeded && canManage && (
            <>
              <button
                type="button"
                className="ghost-btn small"
                onClick={() => setEditing(true)}
                data-testid={`settings-folder-edit-${folder.id}`}
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
                data-testid={`settings-folder-delete-${folder.id}`}
              >
                <Icon
                  name={armedDelete ? 'alert-triangle' : 'trash'}
                  size={11}
                />{' '}
                {folderDeleteLabel(pendingDelete, armedDelete)}
              </button>
            </>
          )}
        </div>
      )}

      {(updateError || deleteError) && (
        <div
          role="alert"
          className="settings-error"
          data-testid={`settings-folder-error-${folder.id}`}
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
  if (typeof err === 'string') return err;
  return 'Unexpected folder operation error';
}
