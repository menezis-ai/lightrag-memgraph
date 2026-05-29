/**
 * SettingsTab — 4 sub-sections, no Tokens, no Members edit, no API generation.
 *
 * Sections gardées (Louis HORVAT 2026-05-28):
 *   - Profile     : read-only, lit useAuth()
 *   - Workspace   : env vars read-only
 *   - Providers   : LLM / Embedder / Reranker avec real Configure panels
 *   - Danger      : Delete workspace (gated by Steward palier)
 *
 * Sections explicitly REMOVED:
 *   - Tokens / OAuth2 client management
 *   - Members editable (lives in MyAccess)
 *   - API key generation
 *
 * The component takes the active workspace + kb name from props so it does
 * not duplicate App-level state; the host owns workspace switching.
 */

import { useState } from 'react';
import { ProfileSection } from './Settings/ProfileSection';
import { WorkspaceSection } from './Settings/WorkspaceSection';
import { ProvidersSection } from './Settings/ProvidersSection';
import { DangerSection } from './Settings/DangerSection';

type SectionKey = 'profile' | 'workspace' | 'providers' | 'danger';

const SECTIONS: { key: SectionKey; label: string }[] = [
  { key: 'profile', label: 'Profile' },
  { key: 'workspace', label: 'Workspace' },
  { key: 'providers', label: 'Providers' },
  { key: 'danger', label: 'Danger zone' },
];

export interface SettingsTabProps {
  activeWorkspace: string;
  kbName: string;
  onDeleteWorkspace?: (id: string) => void;
}

export function SettingsTab({
  activeWorkspace,
  kbName,
  onDeleteWorkspace,
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
                {s.label}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="settings-main">
        {section === 'profile' && <ProfileSection />}
        {section === 'workspace' && (
          <WorkspaceSection
            activeWorkspace={activeWorkspace}
            kbName={kbName}
          />
        )}
        {section === 'providers' && <ProvidersSection />}
        {section === 'danger' && (
          <DangerSection
            activeWorkspace={activeWorkspace}
            onDeleteWorkspace={onDeleteWorkspace}
          />
        )}
      </main>
    </div>
  );
}
