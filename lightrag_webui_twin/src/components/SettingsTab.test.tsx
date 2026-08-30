/**
 * Unit tests for the governed Settings rail.
 *
 * Behaviors under test:
 *   - rail exposes the supported runtime sections (Providers / Members /
 *     Danger zone / Tokens all remain absent)
 *   - Profile renders the MyAccess identity after open access is confirmed
 *   - Profile Sign out button fires onSignOut
 *   - Folder section shows the folder id + retention table
 *   - API section shows the ApiTab (proxy)
 */

import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { SettingsTab } from './SettingsTab';
import { __resetAuthConfigCacheForTests } from '../hooks/useAuth';

const server = setupServer(
  http.get('*/auth-status', () =>
    HttpResponse.json({
      auth_enabled: false,
      // Open access mirrors the backend contract: anonymous requests are
      // authenticated by policy once that posture is explicitly confirmed.
      authenticated: true,
      login_required: false,
      user: null,
      expires_at: null,
    }),
  ),
  http.get('*/twin/api/folders', () => HttpResponse.json([])),
  http.get('*/openapi.json', () =>
    HttpResponse.json({
      info: { version: 'test' },
      paths: {},
    }),
  ),
);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());

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
  server.resetHandlers();
});

describe('SettingsTab — rail', () => {
  it('exposes the supported sections, including admin portability', async () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-rail-profile')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-api')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-folder')).toBeInTheDocument();
    expect(await screen.findByTestId('settings-rail-portability')).toBeInTheDocument();
    expect(screen.getByTestId('settings-rail-about')).toBeInTheDocument();
    // Removed sections must NOT appear (30/05 cleanup)
    expect(screen.queryByTestId('settings-rail-providers')).toBeNull();
    expect(screen.queryByTestId('settings-rail-members')).toBeNull();
    expect(screen.queryByTestId('settings-rail-danger')).toBeNull();
    expect(screen.queryByTestId('settings-rail-tokens')).toBeNull();
  });

  it('hides the portability rail entry without admin scope', () => {
    (window as Window & typeof globalThis).__twinConfig = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: '',
      defaultFolderId: 'default',
      maxFolders: 5,
      folders: [],
      debugUser: {
        sso_subject: 'reader',
        email: 'reader@example.test',
        name: 'Reader',
        palier: { level: 1, label: 'Reader', scopes: ['twin:read'] },
        folders: ['default'],
        idp: 'local-debug',
        idp_realm: 'test',
        sub: 'reader',
        session_expires: '2099-12-31T23:59:00Z',
        gateway_scopes: ['read:documents'],
      },
    };
    __resetAuthConfigCacheForTests();
    renderWith(new QueryClient());
    expect(screen.queryByTestId('settings-rail-portability')).toBeNull();
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
  it('shows the open-access Steward identity after auth posture resolves', async () => {
    renderWith(new QueryClient());
    expect(screen.getByTestId('settings-profile')).toBeInTheDocument();
    expect(
      screen.getByTestId('settings-profile-name').textContent,
    ).toBe('Local Operator');
    expect(await screen.findByText('Steward')).toBeInTheDocument();
  });

  it('Sign out button fires onSignOut', async () => {
    const onSignOut = vi.fn();
    renderWith(new QueryClient(), { onSignOut });
    await screen.findByText('Steward');
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
        sso_subject: 'operator@example.com',
        email: 'operator@example.com',
        name: 'operator@example.com',
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

  it('renders gateway scopes as chip list after auth posture resolves', async () => {
    renderWith(new QueryClient());
    // Explicitly confirmed open access exposes the configured Steward scopes.
    expect(screen.getByText('read:documents')).toBeInTheDocument();
    expect(await screen.findByText('admin:folders')).toBeInTheDocument();
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
      activeFolder: 'demo-prod',
      kbName: 'Demo Production',
    });
    await userEvent.click(screen.getByTestId('settings-rail-folder'));
    expect(screen.getByTestId('settings-active-folder').textContent).toBe(
      'demo-prod',
    );
    expect(
      screen.getByTestId('settings-folder-display-name').textContent,
    ).toBe('Demo Production');
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

  it('blurb reflects the audit-C8 honest copy', async () => {
    renderWith(new QueryClient());
    await userEvent.click(screen.getByTestId('settings-rail-api'));
    const blurb = screen.getByTestId('settings-api-blurb');
    // Keep the section scannable: one compact contract strip, not card soup.
    expect(blurb.textContent).toContain('Contract');
    expect(blurb.textContent).toContain('/twin/api/*');
    expect(blurb.textContent).toContain('tag_filter');
    expect(blurb.textContent).toContain('TAGGED_WITH');
    expect(blurb.textContent).toContain('/query/data');
    expect(blurb.textContent).toContain('mix');
    expect(blurb.textContent).toContain('hybrid');
    expect(blurb.querySelectorAll('dt')).toHaveLength(0);
    // No more "gateway transparently injects tag_filter / visibility".
    expect(blurb.textContent).not.toMatch(/transparently injects/i);
    expect(blurb.textContent).not.toMatch(/visibility/i);
  });
});
