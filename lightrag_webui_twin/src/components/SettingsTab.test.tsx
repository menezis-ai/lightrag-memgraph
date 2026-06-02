/**
 * Unit tests for SettingsTab (3-section redesign per 2026-05-30).
 *
 * Behaviors under test:
 *   - rail exposes exactly Profile / API / Space (Providers / Members /
 *     Danger zone / Tokens / API generation all absent)
 *   - Profile renders the MyAccess identity from useAuth (Steward in dev)
 *   - Profile Sign out button fires onSignOut
 *   - Profile Restart tutorial button fires onRestartTutorial
 *   - Space section shows the space id + retention table
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
      <SettingsTab activeWorkspace="default" kbName="Default space" {...props} />
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
  it('exposes exactly 3 sections: profile, api, space', () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-rail-profile')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-api')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-workspace')).toBeInTheDocument();
    // Removed sections must NOT appear (30/05 cleanup)
    expect(screen.queryByTestId('settings-rail-providers')).toBeNull();
    expect(screen.queryByTestId('settings-rail-members')).toBeNull();
    expect(screen.queryByTestId('settings-rail-danger')).toBeNull();
    expect(screen.queryByTestId('settings-rail-tokens')).toBeNull();
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

  it('Restart tutorial button fires onRestartTutorial', async () => {
    const onRestart = vi.fn();
    renderWith(new QueryClient(), { onRestartTutorial: onRestart });
    await userEvent.click(screen.getByTestId('settings-restart-tutorial'));
    expect(onRestart).toHaveBeenCalledTimes(1);
  });

  it('renders gateway scopes as chip list', () => {
    renderWith(new QueryClient());
    // Steward dev fallback has 6 scopes
    expect(screen.getByText('read:documents')).toBeInTheDocument();
    expect(screen.getByText('admin:workspace')).toBeInTheDocument();
  });
});

describe('SettingsTab — Space', () => {
  it('renders the space id + retention table rows', async () => {
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-workspace'));
    expect(screen.getByTestId('settings-active-ws').textContent).toBe('default');
    // 6 retention rows present
    expect(screen.getByText('Source mgmt')).toBeInTheDocument();
    expect(screen.getByText('Tag mgmt')).toBeInTheDocument();
    expect(screen.getByText('Retrieval')).toBeInTheDocument();
    expect(screen.getByText('Admin')).toBeInTheDocument();
    expect(screen.getByText('Auth')).toBeInTheDocument();
    expect(screen.getByText('Policy / System')).toBeInTheDocument();
  });
});

describe('SettingsTab — API', () => {
  it('renders the ApiTab when API section is selected', async () => {
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-api'));
    expect(screen.getByTestId('settings-api')).toBeInTheDocument();
  });
});
