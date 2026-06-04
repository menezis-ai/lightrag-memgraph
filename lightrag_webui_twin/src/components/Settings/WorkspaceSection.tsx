/**
 * Settings → Space section.
 *
 * Read-only view of the active Twin space identity (id + display name).
 * Both values come from props (the AppShell already resolves them from
 * the runtime config + active-space state); we avoid duplicating that
 * resolution here.
 *
 * Historical note: this section used to render hardcoded
 * visibility / region / retention TTL cards from
 * `fixtures/settings.ts`. They were removed 2026-06-04 as part of the
 * mock-kill audit (Fabrice 2026-06-01 — "je ne veux plus de moquer")
 * because the backend doesn't expose those fields and the displayed
 * values were inventions (`eu-west-3 · dc-paris`,
 * `twin-default-space-retention-v1`, hardcoded 90d/30d/1y/7y TTLs).
 * They risked being read as a compliance commitment.
 * See `docs/audits/webui-fork/mock-kill-audit-2026-06-04.md` finding F1.
 */

import { Icon } from '../Icon';

export interface WorkspaceSectionProps {
  /** Active Twin space id — comes from AppShell state, kept in sync with
   *  `setActiveSpace()` in `api/client.ts`. */
  activeSpaceId: string;
  /** Display name of the active space — derived from the runtime config
   *  catalog at the AppShell level. */
  displayName: string;
}

export function WorkspaceSection({
  activeSpaceId,
  displayName,
}: WorkspaceSectionProps) {
  return (
    <div className="settings-section" data-testid="settings-workspace">
      <h3>Space</h3>
      <p className="muted">
        Configuration for space {activeSpaceId}. Identity is set at
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
          <dt>Space ID</dt>
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
