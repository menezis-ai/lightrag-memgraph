/**
 * Profile sub-section — read-only.
 *
 * Identity fields come exclusively from `useAuth().user` (MyAccess JWT). The
 * UI must NEVER offer to edit them — those operations live in MyAccess, not
 * Twin (Louis 2026-05-28).
 */

import { useAuth } from '../../hooks/useAuth';

export function ProfileSection() {
  const { user } = useAuth();
  if (!user) {
    return (
      <div className="settings-section" data-testid="settings-profile">
        <h3>Profile</h3>
        <p className="muted">No authenticated user (debug fallback missing).</p>
      </div>
    );
  }
  return (
    <div className="settings-section" data-testid="settings-profile">
      <h3>Profile</h3>
      <p className="muted">
        Identity & rôles inherited from MyAccess. Modifications are made in
        MyAccess, not in Twin.
      </p>
      <dl className="settings-dl">
        <dt>Display name</dt>
        <dd data-testid="settings-profile-name">{user.name}</dd>
        <dt>Email</dt>
        <dd>{user.email}</dd>
        <dt>SSO subject</dt>
        <dd className="mono">{user.sso_subject}</dd>
        <dt>Palier</dt>
        <dd>
          <strong>{user.palier.label}</strong>{' '}
          <span className="muted">(level {user.palier.level})</span>
        </dd>
        <dt>Scopes</dt>
        <dd className="mono">{user.palier.scopes.join(', ')}</dd>
        <dt>Workspaces</dt>
        <dd className="mono">{user.workspaces.join(', ')}</dd>
      </dl>
    </div>
  );
}
