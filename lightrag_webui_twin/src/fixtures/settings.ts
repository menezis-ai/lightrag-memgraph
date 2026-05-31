/**
 * Settings tab fixtures — Workspace identity + retention table.
 *
 * Mirrors the prototype `data.js` `WORKSPACE_SETTINGS` shape. Real runtime
 * values come from `/twin/api/workspaces/{id}/settings` once the server
 * sub-app exposes the endpoint (Couche 3 of the prototype port plan).
 *
 * Note: PROVIDER_SETTINGS exists in the prototype data but is intentionally
 * NOT exported here — Providers section was removed from the rail per the
 * 30/05 review. Keeping the data out keeps the bundle lean.
 */

export interface RetentionRow {
  area: string;
  ttl: string;
  note: string;
}

export interface WorkspaceSettings {
  workspace_id: string;
  display_name: string;
  visibility: 'private' | 'internal' | 'public';
  /** Env var name that controls visibility — surfaced to ops, not editable from UI. */
  visibility_env: string;
  region: string;
  retention_policy: string;
  retention: readonly RetentionRow[];
}

export const WORKSPACE_SETTINGS: WorkspaceSettings = {
  workspace_id: 'cib',
  display_name: 'CIB KB',
  visibility: 'private',
  visibility_env: 'TWIN_INSTANCE_VISIBILITY',
  region: 'eu-west-3 · dc-paris',
  retention_policy: 'twin-cib-retention-v2.1',
  retention: [
    { area: 'Source mgmt', ttl: '90d', note: 'uploads, deletes, re-ingests' },
    { area: 'Tag mgmt', ttl: '90d', note: 'requests, approvals, deprecations' },
    { area: 'Retrieval', ttl: '30d', note: 'queries + cited sources' },
    { area: 'Admin', ttl: '1y', note: 'workspace + provider changes' },
    { area: 'Auth', ttl: '1y', note: 'logins, token mints' },
    { area: 'Policy / System', ttl: '7y', note: 'policy violations, system actions' },
  ],
};
