/**
 * Settings → Folder section.
 *
 * Read-only view of the active Twin folder identity (id + display name).
 * Both values come from props (the AppShell already resolves them from
 * the runtime config + active-folder state); we avoid duplicating that
 * resolution here.
 *
 * Historical note: this section used to render hardcoded
 * visibility / region / retention TTL cards from
 * `fixtures/settings.ts`. They were removed 2026-06-04 as part of the
 * mock-kill audit (Fabrice 2026-06-01 — "je ne veux plus de moquer")
 * because the backend doesn't expose those fields and the displayed
 * values were inventions (`eu-west-3 · dc-paris`,
 * `twin-default-folder-retention-v1`, hardcoded 90d/30d/1y/7y TTLs).
 * They risked being read as a compliance commitment.
 */

import { Icon } from '../Icon';

export interface WorkspaceSectionProps {
  /** Active Twin folder id — comes from AppShell state, kept in sync with
   *  `setActiveFolder()` in `api/client.ts`. */
  activeSpaceId: string;
  /** Display name of the active folder — derived from the runtime config
   *  catalog at the AppShell level. */
  displayName: string;
}

export function WorkspaceSection({
  activeSpaceId,
  displayName,
}: WorkspaceSectionProps) {
  return (
    <div className="settings-section" data-testid="settings-workspace">
      <h3>Folder</h3>
      <p className="muted">
        Configuration for folder {activeSpaceId}. Identity is set at
        deployment time and cannot be changed from the UI.
      </p>

      <div className="set-card">
        <div className="set-card-h">
          Identity{' '}
          <span className="env-badge">
            <Icon name="lock" size={10} /> env-controlled
          </span>
        </div>
        <dl className="set-dl">
          <dt>Folder ID</dt>
          <dd className="mono" data-testid="settings-active-ws">
            {activeSpaceId}
          </dd>
          <dt>Display name</dt>
          <dd data-testid="settings-space-display-name">
            {displayName || <span className="muted">(unset)</span>}
          </dd>
        </dl>
      </div>
    </div>
  );
}
