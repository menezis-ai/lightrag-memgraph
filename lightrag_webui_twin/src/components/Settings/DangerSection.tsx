/**
 * Danger sub-section — destructive ops, gated by palier.
 *
 * Only "Delete workspace" is exposed: it triggers a confirmation modal that
 * requires typing the workspace id verbatim. The actual mutation is a stub
 * for the demo (no API call yet — the server-side wipe semantics need a
 * separate Salah/Geoffrey sync before we ship it).
 */

import { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';

interface DangerSectionProps {
  activeWorkspace: string;
  onDeleteWorkspace?: (id: string) => void;
}

export function DangerSection({
  activeWorkspace,
  onDeleteWorkspace,
}: DangerSectionProps) {
  const { user } = useAuth();
  const isSteward = user?.palier.level === 3;
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState('');

  if (!isSteward) {
    return (
      <div className="settings-section" data-testid="settings-danger">
        <h3>Danger zone</h3>
        <p className="muted">
          Destructive operations are restricted to Steward palier (level 3).
        </p>
      </div>
    );
  }

  return (
    <div className="settings-section" data-testid="settings-danger">
      <h3>Danger zone</h3>
      <div className="settings-danger-row">
        <div>
          <strong>Delete workspace</strong>
          <div className="muted">
            Removes <code className="mono">{activeWorkspace}</code> and every
            indexed document, tag, audit event under it. Irreversible.
          </div>
        </div>
        <button
          type="button"
          className="btn danger"
          data-testid="settings-danger-delete-ws"
          onClick={() => {
            setConfirmText('');
            setConfirmOpen(true);
          }}
        >
          Delete workspace…
        </button>
      </div>
      {confirmOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setConfirmOpen(false)}
          data-testid="settings-danger-confirm"
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="Confirm workspace delete"
            style={{ width: 460 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h2>Delete workspace</h2>
            </div>
            <div className="modal-body">
              <p>
                Type the workspace id <code>{activeWorkspace}</code> to confirm.
              </p>
              <input
                type="text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                aria-label="Type workspace id to confirm"
                data-testid="settings-danger-confirm-input"
                className="mono"
              />
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn"
                onClick={() => setConfirmOpen(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn danger"
                disabled={confirmText !== activeWorkspace}
                data-testid="settings-danger-confirm-submit"
                onClick={() => {
                  onDeleteWorkspace?.(activeWorkspace);
                  setConfirmOpen(false);
                }}
              >
                Delete forever
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
