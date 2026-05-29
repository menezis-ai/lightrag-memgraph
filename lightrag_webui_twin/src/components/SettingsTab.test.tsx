/**
 * Unit tests for SettingsTab and its sub-sections.
 *
 * Behaviors under test:
 *   - Profile reads useAuth() and renders display name + palier
 *   - Workspace renders the active workspace id + runtime config
 *   - Providers Configure opens a panel; Reader cannot Request change
 *   - Danger zone is hidden for non-Steward
 *   - Danger zone delete requires typing the workspace id verbatim
 *   - 4 sections present, no Tokens/Members/API generation rail
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SettingsTab } from './SettingsTab';
import { __resetAuthConfigCacheForTests } from '../hooks/useAuth';

function renderWith(qc: QueryClient, props: Partial<Parameters<typeof SettingsTab>[0]> = {}) {
  return render(
    <QueryClientProvider client={qc}>
      <SettingsTab
        activeWorkspace="cib"
        kbName="CIB Knowledge"
        {...props}
      />
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
  it('exposes exactly 4 sections: profile, workspace, providers, danger', () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-rail-profile')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-workspace')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-providers')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-danger')).toBeInTheDocument();
    // Removed sections must not appear
    expect(screen.queryByText(/tokens/i)).toBeNull();
    expect(screen.queryByText(/api generation/i)).toBeNull();
  });
});

describe('SettingsTab — Profile', () => {
  it('shows MyAccess identity from useAuth (dev fallback = Steward)', () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-profile')).toBeInTheDocument();
    expect(screen.getByTestId('settings-profile-name').textContent).toBe(
      'dev.steward',
    );
    expect(screen.getByText('Steward')).toBeInTheDocument();
  });
});

describe('SettingsTab — Workspace', () => {
  it('renders the active workspace id + Twin api base', async () => {
    renderWith(new QueryClient(), { activeWorkspace: 'wm' });
    await userEvent.click(screen.getByTestId('settings-rail-workspace'));
    expect(screen.getByTestId('settings-active-ws').textContent).toBe('wm');
    expect(screen.getByText('/twin/api')).toBeInTheDocument();
  });
});

describe('SettingsTab — Providers', () => {
  it('opens a Configure panel and shows Request change for Steward', async () => {
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-providers'));
    await userEvent.click(
      screen.getByTestId('settings-providers-configure-llm'),
    );
    expect(screen.getByTestId('settings-provider-panel-llm')).toBeInTheDocument();
    expect(
      screen.getByTestId('settings-provider-request-llm'),
    ).toBeInTheDocument();
  });
});

describe('SettingsTab — Danger', () => {
  it('requires typing the workspace id verbatim before Delete is enabled', async () => {
    const onDelete = vi.fn();
    renderWith(new QueryClient(), {
      activeWorkspace: 'cib',
      onDeleteWorkspace: onDelete,
    });
    await userEvent.click(screen.getByTestId('settings-rail-danger'));
    await userEvent.click(screen.getByTestId('settings-danger-delete-ws'));

    const submit = screen.getByTestId('settings-danger-confirm-submit');
    expect(submit).toBeDisabled();

    const input = screen.getByTestId('settings-danger-confirm-input');
    await userEvent.type(input, 'wrong-ws');
    expect(submit).toBeDisabled();

    await userEvent.clear(input);
    await userEvent.type(input, 'cib');
    expect(submit).not.toBeDisabled();

    await userEvent.click(submit);
    expect(onDelete).toHaveBeenCalledWith('cib');
  });

  it('hides destructive actions for non-Steward users', async () => {
    __resetAuthConfigCacheForTests();
    (window as Window & typeof globalThis).__twinConfig = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '/api',
      idpLogoutUrl: 'http://localhost/logout',
      debugUser: {
        sso_subject: 'r@x.local',
        email: 'r@x.local',
        name: 'reader.bob',
        palier: { level: 1, label: 'Reader', scopes: ['twin:read'] },
        workspaces: ['cib'],
      },
    };
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-danger'));
    expect(screen.queryByTestId('settings-danger-delete-ws')).toBeNull();
  });
});
