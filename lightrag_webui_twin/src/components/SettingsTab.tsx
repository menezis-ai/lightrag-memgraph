/**
 * SettingsTab — 3 sections only (review 2026-05-30):
 *
 *   - Profile   : read-only, lit useAuth()
 *   - API       : OpenAPI browser (delegates to ApiTab)
 *   - Folder    : env vars + retention table read-only
 *
 * REMOVED from the maquette pre-30/05:
 *   - Providers (removed 30/05 cleanup)
 *   - Members editable (lives in MyAccess)
 *   - Tokens / OAuth2 client management
 *   - API key generation
 *   - Danger zone (folder deletion moves to ops tooling)
 *
 * Rationale: Twin is a knowledge-management console, not an identity / billing
 * console. Every section that asked the operator to manage capability tokens
 * or destructive folder state was moved out of the UI surface — Louis 28/05
 * + cleanup 30/05.
 */

import { useState } from 'react';
import { ApiTab } from './ApiTab';
import { ProfileSection } from './Settings/ProfileSection';
import { FoldersAdminSection } from './Settings/FoldersAdminSection';
import { FolderSection } from './Settings/FolderSection';
import { Icon } from './Icon';
import { useAuth } from '../hooks/useAuth';
import { useOpenApi } from '../api/queries';
import type { Toast } from '../types/toast';

export type SettingsSectionKey = 'profile' | 'api' | 'folder';

const SECTIONS: { key: SettingsSectionKey; label: string; icon: 'circle-dot' | 'world' | 'folder' }[] = [
  { key: 'profile', label: 'Profile', icon: 'circle-dot' },
  { key: 'api', label: 'API', icon: 'world' },
  { key: 'folder', label: 'Folder', icon: 'folder' },
];

export interface SettingsTabProps {
  /** Active folder id — forwarded to FolderSection's Identity card.
   *  Comes from the AppShell state kept in sync with `setActiveFolder()`. */
  activeFolder?: string;
  /** Active folder display name — forwarded to FolderSection's
   *  Identity card. Resolved by AppShell from the runtime folder catalog. */
  kbName?: string;
  /** Bearer-token revoke + redirect to IdP. Pushed up so the host owns the toast queue. */
  onSignOut?: () => void;
  /** Reopen the onboarding wizard at step 1. */
  onRestartTutorial?: () => void;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
  initialSection?: SettingsSectionKey;
}

export function SettingsTab({
  activeFolder,
  kbName,
  onSignOut,
  onRestartTutorial,
  onToast,
  initialSection = 'profile',
}: SettingsTabProps) {
  const [sectionState, setSectionState] = useState<{
    initial: SettingsSectionKey;
    value: SettingsSectionKey;
  }>({ initial: initialSection, value: initialSection });
  const section =
    sectionState.initial === initialSection ? sectionState.value : initialSection;
  const setSection = (value: SettingsSectionKey) => {
    setSectionState({ initial: initialSection, value });
  };
  const { user } = useAuth();

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
        {section === 'api' && <ApiSection />}
        {section === 'folder' && (
          <>
            <FolderSection
              activeFolderId={activeFolder ?? 'default'}
              displayName={kbName ?? ''}
            />
            <FoldersAdminSection user={user} onToast={onToast} />
          </>
        )}
      </main>
    </div>
  );
}

/**
 * API section — fetches the live OpenAPI surface from
 * `/twin/api/openapi` instead of bundling a hardcoded fixture (mock-kill
 * F2). Pure prod URL list (`API_SERVERS`) was also removed because the
 * `cib-kb.twin.internal` hostnames didn't exist; the curl preview now
 * uses the current browser origin which always reflects the live deploy.
 */
function ApiSection() {
  const { data, isLoading, isError, error, refetch } = useOpenApi();
  const baseUrl =
    typeof window !== 'undefined' ? window.location.origin : '';

  return (
    <div className="settings-section settings-api" data-testid="settings-api">
      <h3>API</h3>
      <p className="muted">
        LightRAG OpenAPI surface. Bearer (OIDC) auth only — the gateway
        injects <code>tag_filter</code> and <code>visibility</code>{' '}
        scoping from the active folder.
      </p>
      {isLoading && (
        <div className="muted" data-testid="settings-api-loading">
          Loading OpenAPI surface…
        </div>
      )}
      {isError && (
        <div
          className="error-banner"
          role="alert"
          data-testid="settings-api-error"
        >
          OpenAPI surface unavailable
          {error instanceof Error ? ` — ${error.message}` : ''}.{' '}
          <button
            type="button"
            className="ghost-btn"
            onClick={() => refetch()}
          >
            Retry
          </button>
        </div>
      )}
      {data && (
        <ApiTab
          apiVersion={data.version}
          groups={data.groups}
          baseUrl={baseUrl}
        />
      )}
    </div>
  );
}
