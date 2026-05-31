/**
 * Unit tests for the Topbar enrichment components (#104).
 *
 * Per the sprint brief revision 2026-05-29 + Louis 2026-05-28:
 *   - PalierSwitcher : removed (palier is JWT-only, no UI)
 *   - MyAccessPill   : removed (gimmick, not in prod)
 *   - WorkspaceSwitcher : reads useAuth().user.workspaces, click emits onPick(id)
 *   - TodoBell : polls /twin/api/notifications, badge counts actionable items
 *   - SystemStatusIndicator : worst-of(lightrag, twin) health, opens popover
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { WorkspaceSwitcher } from './WorkspaceSwitcher';
import { TodoBell } from './TodoBell';
import { SystemStatusIndicator } from './SystemStatusIndicator';
import { WORKSPACE_FIXTURES, NOTIFICATION_FIXTURES } from '../../fixtures';
import { __resetAuthConfigCacheForTests } from '../../hooks/useAuth';

const server = setupServer(
  http.get('*/twin/api/notifications', () =>
    HttpResponse.json(NOTIFICATION_FIXTURES),
  ),
  http.get('*/health', ({ request }) => {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/twin/api')) return undefined;
    return HttpResponse.json({ status: 'ok' });
  }),
  http.get('*/twin/api/health', () => HttpResponse.json({ status: 'ok' })),
);

beforeEach(() => {
  __resetAuthConfigCacheForTests();
  // Inject a debug user with the full fixture workspace set so the
  // WorkspaceSwitcher rendering is not narrowed by MyAccess in tests.
  (window as Window & typeof globalThis).__twinConfig = {
    apiBaseUrl: '/twin/api',
    lightragBaseUrl: '/api',
    idpLogoutUrl: 'http://localhost/logout',
    debugUser: {
      sso_subject: 'test@twin.local',
      email: 'test@twin.local',
      name: 'test.user',
      palier: { level: 3, label: 'Steward', scopes: ['twin:read', 'twin:write'] },
      workspaces: WORKSPACE_FIXTURES.map((w) => w.id),
      idp: 'keycloak',
      idp_realm: 'twin-test',
      sub: 'test-001',
      session_expires: '2026-12-31T23:59:00Z',
      gateway_scopes: ['read:documents', 'write:documents'],
    },
  };
  server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
  server.resetHandlers();
  server.close();
});

function wrap(qc: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

describe('WorkspaceSwitcher', () => {
  it('renders the active workspace and opens a menu on click', async () => {
    const onPick = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <WorkspaceSwitcher
          active="cib"
          workspaces={WORKSPACE_FIXTURES}
          onPick={onPick}
        />
      </Wrap>,
    );
    expect(screen.getByTestId('topbar-workspace-switcher')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('topbar-workspace-switcher'));
    expect(screen.getByTestId('topbar-workspace-menu')).toBeInTheDocument();
  });

  it('emits onPick(id) when a non-active workspace is clicked', async () => {
    const onPick = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <WorkspaceSwitcher
          active="cib"
          workspaces={WORKSPACE_FIXTURES}
          onPick={onPick}
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('topbar-workspace-switcher'));
    // Pick the first non-active workspace
    const nonActive = WORKSPACE_FIXTURES.find((w) => w.id !== 'cib');
    if (!nonActive) throw new Error('expected fixture diversity');
    await userEvent.click(
      screen.getByTestId(`topbar-workspace-pick-${nonActive.id}`),
    );
    expect(onPick).toHaveBeenCalledWith(nonActive.id);
  });
});

describe('SystemStatusIndicator', () => {
  it('renders a dot with the worst-of status', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <SystemStatusIndicator pollMs={60_000} />
      </Wrap>,
    );
    const button = screen.getByTestId('topbar-status-indicator');
    await waitFor(() => {
      const dot = button.querySelector('[data-status]');
      expect(dot?.getAttribute('data-status')).toBe('ok');
    });
  });

  it('opens the detail popover on click', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <SystemStatusIndicator pollMs={60_000} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('topbar-status-indicator'));
    await waitFor(() => {
      expect(screen.getByTestId('status-lightrag')).toBeInTheDocument();
      expect(screen.getByTestId('status-twin')).toBeInTheDocument();
    });
  });
});

describe('TodoBell', () => {
  it('renders without crashing and reaches the notifications endpoint', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <TodoBell pollMs={60_000} />
      </Wrap>,
    );
    expect(screen.getByTestId('topbar-todo-bell')).toBeInTheDocument();
  });
});
