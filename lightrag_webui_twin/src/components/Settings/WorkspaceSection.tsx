/**
 * Settings → Workspace section.
 *
 * Read-only view of the workspace identity (id / display name / visibility /
 * region) and the retention policy table. Values are env-controlled at Helm
 * install time — the env-badge makes this explicit so an operator doesn't
 * waste a click looking for an "Edit" button.
 *
 * Retention table covers the 6 retention areas (Source mgmt / Tag mgmt /
 * Retrieval / Admin / Auth / Policy·System) the BNP doctrine demands. TTLs
 * align with the `twin-cib-retention-v2.1` policy.
 */

import { WORKSPACE_SETTINGS } from '../../fixtures/settings';
import { Icon } from '../Icon';

export function WorkspaceSection() {
  const ws = WORKSPACE_SETTINGS;
  return (
    <div className="settings-section" data-testid="settings-workspace">
      <h3>Workspace</h3>
      <p className="muted">
        Configuration for workspace {ws.workspace_id}. Some values are set at
        Helm install time and cannot be changed at runtime.
      </p>

      <div className="set-card">
        <div className="set-card-h">
          Identity{' '}
          <span className="env-badge">
            <Icon name="lock" size={10} /> env-controlled
          </span>
        </div>
        <dl className="set-dl">
          <dt>Workspace ID</dt>
          <dd className="mono" data-testid="settings-active-ws">
            {ws.workspace_id}
          </dd>
          <dt>Display name</dt>
          <dd>{ws.display_name}</dd>
          <dt>Visibility</dt>
          <dd className="mono">
            {ws.visibility}{' '}
            <span className="muted">({ws.visibility_env})</span>
          </dd>
          <dt>Region</dt>
          <dd className="mono">{ws.region}</dd>
        </dl>
      </div>

      <div className="set-card">
        <div className="set-card-h">
          Retention policy{' '}
          <span className="env-badge">
            <Icon name="lock" size={10} /> env-controlled
          </span>
        </div>
        <table className="retention-table">
          <thead>
            <tr>
              <th>Area</th>
              <th>TTL</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {ws.retention.map((r) => (
              <tr key={r.area}>
                <td>{r.area}</td>
                <td className="mono">{r.ttl}</td>
                <td className="muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="retention-foot">
          Aligned with policy <code>{ws.retention_policy}</code>. Override
          requires a Tier-1 governance ticket.
        </div>
      </div>
    </div>
  );
}
