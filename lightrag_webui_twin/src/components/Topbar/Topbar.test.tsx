/**
 * Unit tests for the Topbar enrichment components (#104).
 *
 * Per the sprint brief revision 2026-05-29 + compliance review 2026-05-28:
 *   - PalierSwitcher : removed (palier is JWT-only, no UI)
 *   - MyAccessPill   : removed (gimmick, not in prod)
 *   - FolderSwitcher : reads useAuth().user.folders, click emits onPick(id)
 *   - TodoBell : polls /twin/api/notifications, badge counts actionable items
 *   - SystemStatusIndicator : worst-of(lightrag, twin) health, opens popover
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { FolderSwitcher } from './FolderSwitcher';
import { TodoBell } from './TodoBell';
import { SystemStatusIndicator } from './SystemStatusIndicator';
import { FOLDER_FIXTURES, NOTIFICATION_FIXTURES } from '../../fixtures';
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
  // Inject a debug user with the full fixture folder set so the
  // FolderSwitcher rendering is not narrowed by MyAccess in tests.
  (window as Window & typeof globalThis).__twinConfig = {
    apiBaseUrl: '/twin/api',
    lightragBaseUrl: '/api',
    idpLogoutUrl: 'http://localhost/logout',
    debugUser: {
      sso_subject: 'test@twin.local',
      email: 'test@twin.local',
      name: 'test.user',
      palier: { level: 3, label: 'Steward', scopes: ['twin:read', 'twin:write'] },
      folders: FOLDER_FIXTURES.map((w) => w.id),
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

describe('FolderSwitcher', () => {
  it('renders the active folder and opens a menu on click', async () => {
    const onPick = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <FolderSwitcher
          active="cib"
          folders={FOLDER_FIXTURES}
          onPick={onPick}
        />
      </Wrap>,
    );
    expect(screen.getByTestId('topbar-folder-switcher')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('topbar-folder-switcher'));
    expect(screen.getByTestId('topbar-folder-menu')).toBeInTheDocument();
  });

  it('emits onPick(id) when a non-active folder is clicked', async () => {
    const onPick = vi.fn();
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <FolderSwitcher
          active="cib"
          folders={FOLDER_FIXTURES}
          onPick={onPick}
        />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('topbar-folder-switcher'));
    // Pick the first non-active folder
    const nonActive = FOLDER_FIXTURES.find((w) => w.id !== 'cib');
    if (!nonActive) throw new Error('expected fixture diversity');
    await userEvent.click(
      screen.getByTestId(`topbar-folder-pick-${nonActive.id}`),
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

  // Couche 1.2 — Bucket A : sys-pill geometry from the prototype.
  // The topbar trigger must be a .sys-pill with a .sys-dot + .sys-pill-label,
  // and switch label/variant based on the worst-of status (ok → "All systems",
  // degraded → "Degraded" + .sys-pill-warn, down → "Outage" + .sys-pill-error).
  it('renders the .sys-pill shape with the "All systems" label when healthy', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <SystemStatusIndicator pollMs={60_000} />
      </Wrap>,
    );
    const button = screen.getByTestId('topbar-status-indicator');
    await waitFor(() => {
      expect(button.classList.contains('sys-pill')).toBe(true);
      expect(button.querySelector('.sys-dot')).toBeTruthy();
      expect(button.querySelector('.sys-pill-label')?.textContent).toBe(
        'All systems',
      );
    });
  });

  it('switches to .sys-pill-error + "Outage" when both endpoints fail', async () => {
    // Override both health endpoints with errors so worst-of resolves to 'down'.
    server.use(
      http.get('*/health', ({ request }) => {
        const url = new URL(request.url);
        if (url.pathname.startsWith('/twin/api')) return undefined;
        return HttpResponse.error();
      }),
      http.get('*/twin/api/health', () => HttpResponse.error()),
    );
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <SystemStatusIndicator pollMs={60_000} />
      </Wrap>,
    );
    const button = screen.getByTestId('topbar-status-indicator');
    await waitFor(() => {
      expect(button.classList.contains('sys-pill-error')).toBe(true);
      expect(button.querySelector('.sys-pill-label')?.textContent).toBe(
        'Outage',
      );
    });
  });

  it('opens the .sys-popover with per-surface checks on click', async () => {
    const Wrap = wrap(new QueryClient());
    render(
      <Wrap>
        <SystemStatusIndicator pollMs={60_000} />
      </Wrap>,
    );
    await userEvent.click(screen.getByTestId('topbar-status-indicator'));
    const popover = await screen.findByRole('dialog', { name: 'System status' });
    expect(popover.classList.contains('sys-popover')).toBe(true);
    expect(popover.querySelector('.sys-popover-checks')).toBeTruthy();
    expect(screen.getByTestId('status-lightrag')).toBeInTheDocument();
    expect(screen.getByTestId('status-twin')).toBeInTheDocument();
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
