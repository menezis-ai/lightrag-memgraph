/**
 * SettingsTab — 3 sections only (review 2026-05-30):
 *
 *   - Profile   : read-only, lit useAuth()
 *   - API       : OpenAPI browser (delegates to ApiTab)
 *   - Space     : env vars + retention table read-only
 *
 * REMOVED from the maquette pre-30/05:
 *   - Providers (removed 30/05 cleanup)
 *   - Members editable (lives in MyAccess)
 *   - Tokens / OAuth2 client management
 *   - API key generation
 *   - Danger zone (space deletion moves to ops tooling)
 *
 * Rationale: Twin is a knowledge-management console, not an identity / billing
 * console. Every section that asked the operator to manage capability tokens
 * or destructive space state was moved out of the UI surface — Louis 28/05
 * + cleanup 30/05.
 */

import { useState } from 'react';
import { ApiTab } from './ApiTab';
import { ProfileSection } from './Settings/ProfileSection';
import { SpacesAdminSection } from './Settings/SpacesAdminSection';
import { WorkspaceSection } from './Settings/WorkspaceSection';
import { Icon } from './Icon';
import {
  API_BASE_URL,
  API_SERVERS,
  API_VERSION,
  OPENAPI_GROUPS,
} from '../fixtures';

type SectionKey = 'profile' | 'api' | 'workspace';

const SECTIONS: { key: SectionKey; label: string; icon: 'circle-dot' | 'world' | 'folder' }[] = [
  { key: 'profile', label: 'Profile', icon: 'circle-dot' },
  { key: 'api', label: 'API', icon: 'world' },
  { key: 'workspace', label: 'Space', icon: 'folder' },
];

export interface SettingsTabProps {
  /** Active space id. Currently informational — display lives in WorkspaceSection. */
  activeWorkspace?: string;
  /** Active space display name. Same status as above. */
  kbName?: string;
  /** Bearer-token revoke + redirect to IdP. Pushed up so the host owns the toast queue. */
  onSignOut?: () => void;
  /** Reopen the onboarding wizard at step 1. */
  onRestartTutorial?: () => void;
}

export function SettingsTab({
  onSignOut,
  onRestartTutorial,
}: SettingsTabProps) {
  const [section, setSection] = useState<SectionKey>('profile');

  return (
    <div className="settings" data-testid="settings-tab">
      <aside className="settings-rail">
        <h2>Settings</h2>
        <ul>
          {SECTIONS.map((s) => (
            <li key={s.key}>
              <button
                type="button"
                className={`settings-rail-btn${section === s.key ? ' active' : ''}`}
                onClick={() => setSection(s.key)}
                data-testid={`settings-rail-${s.key}`}
                aria-current={section === s.key}
              >
                <Icon name={s.icon} size={14} /> {s.label}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="settings-main">
        {section === 'profile' && (
          <ProfileSection
            onSignOut={onSignOut}
            onRestartTutorial={onRestartTutorial}
          />
        )}
        {section === 'api' && (
          <div className="settings-section settings-api" data-testid="settings-api">
            <h3>API</h3>
            <p className="muted">
              LightRAG OpenAPI surface. Bearer (OIDC) auth only — the gateway
              injects <code>tag_filter</code> and <code>visibility</code> scoping
              from the active space.
            </p>
            <ApiTab
              apiVersion={API_VERSION}
              groups={OPENAPI_GROUPS}
              servers={API_SERVERS}
              baseUrl={API_BASE_URL}
            />
          </div>
        )}
        {section === 'workspace' && (
          <>
            <WorkspaceSection />
            <SpacesAdminSection />
          </>
        )}
      </main>
    </div>
  );
}
