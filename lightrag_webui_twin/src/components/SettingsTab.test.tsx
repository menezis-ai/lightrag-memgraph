/**
 * Unit tests for SettingsTab (3-section redesign per 2026-05-30).
 *
 * Behaviors under test:
 *   - rail exposes exactly Profile / API / Folder (Providers / Members /
 *     Danger zone / Tokens / API generation all absent)
 *   - Profile renders the MyAccess identity from useAuth (Steward in dev)
 *   - Profile Sign out button fires onSignOut
 *   - Folder section shows the folder id + retention table
 *   - API section shows the ApiTab (proxy)
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SettingsTab } from './SettingsTab';
import { __resetAuthConfigCacheForTests } from '../hooks/useAuth';

function renderWith(
  qc: QueryClient,
  props: Partial<Parameters<typeof SettingsTab>[0]> = {},
) {
  return render(
    <QueryClientProvider client={qc}>
      <SettingsTab activeFolder="default" kbName="Default folder" {...props} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  __resetAuthConfigCacheForTests();
  (window as Window & typeof globalThis).__twinConfig = undefined;
});

afterEach(() => {
  (window as Window & typeof globalThis).__twinConfig = undefined;
});

describe('SettingsTab — rail', () => {
  it('exposes exactly 3 sections: profile, api, folder', () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-rail-profile')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-api')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-folder')).toBeInTheDocument();
    // Removed sections must NOT appear (30/05 cleanup)
    expect(screen.queryByTestId('settings-rail-providers')).toBeNull();
    expect(screen.queryByTestId('settings-rail-members')).toBeNull();
    expect(screen.queryByTestId('settings-rail-danger')).toBeNull();
    expect(screen.queryByTestId('settings-rail-tokens')).toBeNull();
  });

  it('keeps editable token/member/provider Settings surfaces out of scope', () => {
    renderWith(new QueryClient());
    expect(screen.queryByText(/Default ingestion tags/i)).toBeNull();
    expect(screen.queryByRole('button', { name: /Invite member/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /Delete member/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /Revoke token/i })).toBeNull();
  });
});

describe('SettingsTab — Profile', () => {
  it('shows MyAccess identity from useAuth (dev fallback = Steward)', () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-profile')).toBeInTheDocument();
    expect(
      screen.getByTestId('settings-profile-name').textContent,
    ).toBe('Claire Benoit');
    expect(screen.getByText('Steward')).toBeInTheDocument();
  });

  it('Sign out button fires onSignOut', async () => {
    const onSignOut = vi.fn();
    renderWith(new QueryClient(), { onSignOut });
    await userEvent.click(screen.getByTestId('settings-signout'));
    expect(onSignOut).toHaveBeenCalledTimes(1);
  });

  it('open-access deployment (local-debug idp) hides Sign out and explains why', () => {
    (window as Window & typeof globalThis).__twinConfig = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultFolderId: 'default',
      maxFolders: 5,
      folders: [
        { id: 'default', label: 'Default folder', kind: 'primary', sources: 0 },
      ],
      debugUser: {
        sso_subject: 'operator@twin.local',
        email: 'operator@twin.local',
        name: 'operator@twin.local',
        palier: {
          level: 3,
          label: 'Steward',
          scopes: ['twin:read', 'twin:write', 'twin:approve'],
        },
        folders: ['default'],
        idp: 'local-debug',
        idp_realm: 'twin-local',
        sub: 'local-debug-sub',
        session_expires: '2099-12-31T23:59:00Z',
        gateway_scopes: ['read:documents'],
      },
    };
    __resetAuthConfigCacheForTests();
    renderWith(new QueryClient());

    expect(screen.queryByTestId('settings-signout')).toBeNull();
    expect(screen.getByTestId('settings-open-access-note')).toBeInTheDocument();
  });

  it('renders gateway scopes as chip list', () => {
    renderWith(new QueryClient());
    // Steward dev fallback has 6 scopes
    expect(screen.getByText('read:documents')).toBeInTheDocument();
    expect(screen.getByText('admin:folders')).toBeInTheDocument();
  });
});

describe('SettingsTab — Folder', () => {
  it('can open directly on the folder section', () => {
    renderWith(new QueryClient(), { initialSection: 'folder' });
    expect(screen.getByTestId('settings-folder')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-folder')).toHaveAttribute(
      'aria-current',
      'true',
    );
  });

  it('renders the active folder identity from props (not a fixture)', async () => {
    renderWith(new QueryClient(), {
      activeFolder: 'cib-prod',
      kbName: 'CIB Production',
    });
    await userEvent.click(screen.getByTestId('settings-rail-folder'));
    expect(screen.getByTestId('settings-active-folder').textContent).toBe(
      'cib-prod',
    );
    expect(
      screen.getByTestId('settings-folder-display-name').textContent,
    ).toBe('CIB Production');
  });

  it('no longer renders the removed visibility / region / retention cards', async () => {
    // Mock-kill F1 — these were fixture-only invented values
    // (eu-west-3, twin-default-folder-retention-v1, hardcoded TTLs) and
    // were dropped 2026-06-04.
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-folder'));
    expect(screen.queryByText('Source mgmt')).toBeNull();
    expect(screen.queryByText('Retention policy')).toBeNull();
    expect(screen.queryByText(/eu-west-3/i)).toBeNull();
    expect(screen.queryByText(/twin-default-folder-retention/i)).toBeNull();
  });
});

describe('SettingsTab — API', () => {
  it('renders the ApiTab when API section is selected', async () => {
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-api'));
    expect(screen.getByTestId('settings-api')).toBeInTheDocument();
  });
});
