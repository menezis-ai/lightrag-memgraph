/**
 * Settings → Profile section.
 *
 * Account info inherited from Keycloak (no edit affordance — corrections
 * happen in the corporate IdP). Renders:
 *
 *   - Identity card  : initials avatar + name + email + role-badge
 *   - Identity provider trace : IdP / realm / sub / session_expires (env-controlled)
 *   - Gateway scopes : chip list (one per OAuth2 scope on the bearer token)
 *   - Session card   : Sign out
 *
 * Wording: "Role" not "Palier" (palier = JWT-only term). The `role-badge` class
 * replaces the killed `palier-pill` from the pre-30/05 cleanup.
 */

import { useAuth } from '../../hooks/useAuth';
import { Icon } from '../Icon';

export interface ProfileSectionProps {
  onSignOut?: () => void;
}

function initialsOf(name: string): string {
  return name
    .split(/\s+/)
    .map((s) => s[0] ?? '')
    .join('')
    .slice(0, 2)
    .toUpperCase();
}

export function ProfileSection({ onSignOut }: Readonly<ProfileSectionProps>) {
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
        Account info inherited from your Keycloak session. Update name/email in
        the corporate IDP.
      </p>

      <div className="set-card">
        <div className="set-identity">
          <div className="set-avatar" aria-hidden>
            {initialsOf(user.name)}
          </div>
          <div className="set-identity-main">
            <div className="set-identity-name" data-testid="settings-profile-name">
              {user.name}
            </div>
            <div className="set-identity-email">{user.email}</div>
            <span className="role-badge">{user.palier.label}</span>
          </div>
        </div>
        <dl className="set-dl">
          <dt>Identity provider</dt>
          <dd className="mono">
            {user.idp} · {user.idp_realm} · sub={user.sub}
          </dd>
          <dt>Session expires</dt>
          <dd className="mono">{user.session_expires}</dd>
        </dl>
      </div>

      <div className="set-card">
        <div className="set-card-h">
          Scopes{' '}
          <span className="set-card-hint">
            Permissions attached to your bearer token at gateway level.
          </span>
        </div>
        <div className="scope-chips">
          {user.gateway_scopes.map((s) => (
            <code key={s} className="scope-chip">
              {s}
            </code>
          ))}
        </div>
      </div>

      <div className="set-card">
        <div className="set-card-h">Session</div>
        <div className="set-row">
          {user.idp === 'local-debug' ? (
            // Open-access deployment (no auth backend): there is no session
            // to terminate — a Sign out button here would purge local state
            // and silently re-enter as the same debug identity, which reads
            // like an auth bypass to an auditor.
            <span className="set-row-note" data-testid="settings-open-access-note">
              Open access — no authentication backend is configured on this
              deployment, so there is no session to sign out of. Configure
              LIGHTRAG_API_KEY, LIGHTRAG_JWT_SECRET or TWIN_IDP_JWKS_URL to
              enable authentication.
            </span>
          ) : (
            <>
              <button
                type="button"
                className="btn"
                data-testid="settings-signout"
                onClick={() => onSignOut?.()}
              >
                <Icon name="arrow-right" size={13} /> Sign out
              </button>
              <span className="set-row-note">
                Local cache (threads, tweaks) is preserved in this browser.
              </span>
            </>
          )}
        </div>
      </div>

    </div>
  );
}
