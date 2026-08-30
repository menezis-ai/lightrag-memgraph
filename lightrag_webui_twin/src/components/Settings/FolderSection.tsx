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
 * mock-kill audit ("no mock data displayed as if it were real")
 * because the backend doesn't expose those fields and the displayed
 * values were inventions (`primary-demo-region`,
 * `twin-default-folder-retention-v1`, hardcoded 90d/30d/1y/7y TTLs).
 * They risked being read as a compliance commitment.
 */

import { Icon } from '../Icon';

export interface FolderSectionProps {
  /** Active Twin folder id — comes from AppShell state, kept in sync with
   *  `setActiveFolder()` in `api/client.ts`. */
  activeFolderId: string;
  /** Display name of the active folder — derived from the runtime config
   *  catalog at the AppShell level. */
  displayName: string;
}

export function FolderSection({
  activeFolderId,
  displayName,
}: Readonly<FolderSectionProps>) {
  return (
    <div className="settings-section" data-testid="settings-folder">
      <h3>Folder</h3>
      <p className="muted">
        Configuration for folder {activeFolderId}. Identity is set at
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
          <dd className="mono" data-testid="settings-active-folder">
            {activeFolderId}
          </dd>
          <dt>Display name</dt>
          <dd data-testid="settings-folder-display-name">
            {displayName || <span className="muted">(unset)</span>}
          </dd>
        </dl>
      </div>
    </div>
  );
}
