/**
 * Workspace sub-section — env vars / runtime config, read-only.
 *
 * Shows the resolved Twin runtime config (api base, lightrag base, IdP logout)
 * and the currently active workspace id. These are operational facts, not
 * user preferences; modifications happen via Twin deployment config, not the
 * WebUI.
 */

import { useAuth } from '../../hooks/useAuth';

interface WorkspaceSectionProps {
  activeWorkspace: string;
  kbName: string;
}

export function WorkspaceSection({
  activeWorkspace,
  kbName,
}: WorkspaceSectionProps) {
  const { config } = useAuth();
  return (
    <div className="settings-section" data-testid="settings-workspace">
      <h3>Workspace</h3>
      <p className="muted">
        Runtime config injected by the Twin FastAPI overlay. Read-only.
      </p>
      <dl className="settings-dl">
        <dt>Active workspace</dt>
        <dd className="mono" data-testid="settings-active-ws">
          {activeWorkspace}
        </dd>
        <dt>KB display name</dt>
        <dd>{kbName}</dd>
        <dt>Twin API base</dt>
        <dd className="mono">{config.apiBaseUrl}</dd>
        <dt>LightRAG base</dt>
        <dd className="mono">{config.lightragBaseUrl}</dd>
        <dt>IdP logout</dt>
        <dd className="mono">{config.idpLogoutUrl}</dd>
      </dl>
    </div>
  );
}
