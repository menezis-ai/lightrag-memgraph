/**
 * Unit tests for useAuth.
 *
 * Covers: dev fallback (debug user), placeholder detection, signout flow
 * (API call → queryClient.clear → redirect), config caching.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { __resetAuthConfigCacheForTests, useAuth } from './useAuth';
import { apiFetch } from '../api/client';
import { resolveRuntimeConfig, DEV_CONFIG } from '../config/devConfig';

/** A runtime config with NO debugUser, so the identity derives from
 *  /auth-status alone (the production shape, unlike the dev fallback). */
const AUTH_CONFIG = {
  apiBaseUrl: '/twin/api',
  lightragBaseUrl: '',
  idpLogoutUrl: 'https://idp.example.com/logout',
  defaultFolderId: 'default',
  folders: [{ id: 'default', label: 'Default folder', kind: 'primary' as const }],
};

const authStatusMock = vi.hoisted(() => vi.fn());
const loginMock = vi.hoisted(() => vi.fn());
const logoutLocalMock = vi.hoisted(() => vi.fn());
const logoutMock = vi.hoisted(() => vi.fn());

vi.mock('../api/resources', () => ({
  api: {
    authStatus: authStatusMock,
    login: loginMock,
    logoutLocal: logoutLocalMock,
    logout: logoutMock,
  },
}));

function wrap(qc: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

const ORIGINAL_LOCATION = window.location;
const originalConfig = (window as Window & typeof globalThis).__twinConfig;
const originalE2eConfig = (window as Window & typeof globalThis).__twinE2eRuntimeConfig;

beforeEach(() => {
  __resetAuthConfigCacheForTests();
  window.sessionStorage.clear();
  authStatusMock.mockResolvedValue({
    auth_enabled: false,
    authenticated: true,
    user: null,
    expires_at: null,
    login_required: false,
  });
  loginMock.mockResolvedValue({
    access_token: 'local-jwt-token',
    token_type: 'bearer',
    expires_in: 3600,
  });
  logoutLocalMock.mockResolvedValue({ ok: true });
  logoutMock.mockResolvedValue({ ok: true });
  (window as Window & typeof globalThis).__twinConfig = undefined;
  (window as Window & typeof globalThis).__twinE2eRuntimeConfig = undefined;
  Object.defineProperty(window, 'location', {
    value: {
      ...ORIGINAL_LOCATION,
      href: 'http://localhost:5173/',
      origin: 'http://localhost:5173',
    },
    writable: true,
  });
});

afterEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  (window as Window & typeof globalThis).__twinConfig = originalConfig;
  (window as Window & typeof globalThis).__twinE2eRuntimeConfig = originalE2eConfig;
  Object.defineProperty(window, 'location', {
    value: ORIGINAL_LOCATION,
    writable: true,
  });
});

describe('resolveRuntimeConfig', () => {
  it('returns DEV_CONFIG when source is missing and isDev=true', () => {
    expect(resolveRuntimeConfig(undefined, true)).toBe(DEV_CONFIG);
  });

  it('returns DEV_CONFIG when source equals the placeholder and isDev=true', () => {
    expect(resolveRuntimeConfig('__TWIN_CONFIG_JSON__', true)).toBe(DEV_CONFIG);
  });

  it('throws when source is missing and isDev=false', () => {
    expect(() => resolveRuntimeConfig(undefined, false)).toThrow(
      /globalThis\.__twinConfig/,
    );
  });

  it('parses a stringified JSON config', () => {
    const cfg = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '/api',
      idpLogoutUrl: 'https://idp.example.com/logout',
    };
    expect(resolveRuntimeConfig(JSON.stringify(cfg), false)).toEqual(cfg);
  });

  it('returns the object as-is when given a non-placeholder object', () => {
    const cfg = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '/api',
      idpLogoutUrl: 'https://idp.example.com/logout',
    };
    expect(resolveRuntimeConfig(cfg, false)).toBe(cfg);
  });

  it('uses the e2e runtime override in dev before the HTML placeholder fallback', () => {
    const cfg = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '/api',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultFolderId: 'sandbox',
      folders: [{ id: 'sandbox', label: 'Sandbox', kind: 'sandbox' as const }],
    };
    (window as Window & typeof globalThis).__twinE2eRuntimeConfig = cfg;
    expect(resolveRuntimeConfig('__TWIN_CONFIG_JSON__', true)).toBe(cfg);
  });
});

describe('useAuth — dev fallback', () => {
  it('exposes the dev debugUser when no config is injected', () => {
    const qc = new QueryClient();
    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.palier.level).toBe(3);
    expect(result.current.user?.palier.label).toBe('Steward');
    expect(result.current.user?.name).toBe('Local Operator');
    expect(result.current.user?.folders).toContain('default');
  });

  it('returns the same config across re-renders (cached)', () => {
    const qc = new QueryClient();
    const { result, rerender } = renderHook(() => useAuth(), {
      wrapper: wrap(qc),
    });
    const cfg1 = result.current.config;
    rerender();
    const cfg2 = result.current.config;
    expect(cfg1).toBe(cfg2);
  });
});

describe('useAuth — signout', () => {
  it('calls only the canonical Twin logout, clears cache, and redirects', async () => {
    const qc = new QueryClient();
    qc.setQueryData(['documents'], { items: [], total: 0 });
    window.localStorage.setItem('twin-rag.threads.v3', JSON.stringify([]));
    expect(qc.getQueryData(['documents'])).not.toBeUndefined();
    expect(window.localStorage.getItem('twin-rag.threads.v3')).not.toBeNull();

    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });

    await act(async () => {
      await result.current.signout();
    });

    expect(logoutLocalMock).not.toHaveBeenCalled();
    expect(logoutMock).toHaveBeenCalledOnce();
    expect(qc.getQueryData(['documents'])).toBeUndefined();
    expect(window.localStorage.getItem('twin-rag.threads.v3')).toBeNull();
    expect(window.location.href).toMatch(/realms\/twin\/protocol/);
    expect(window.location.href).toMatch(/redirect_uri=http/);
  });

  it('still cycles the client even if the logout API call rejects', async () => {
    const qc = new QueryClient();
    qc.setQueryData(['notifications'], [{ id: 'n1' }]);

    logoutMock.mockRejectedValue(new Error('network down'));

    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });

    await act(async () => {
      await result.current.signout();
    });

    expect(logoutLocalMock).not.toHaveBeenCalled();
    expect(logoutMock).toHaveBeenCalledOnce();
    expect(qc.getQueryData(['notifications'])).toBeUndefined();
    expect(window.location.href).toMatch(/realms\/twin/);
  });
});

describe('useAuth — session expiry', () => {
  it('drops to the login screen when a mid-session request returns 401', async () => {
    (window as Window & typeof globalThis).__twinConfig = { ...AUTH_CONFIG };
    __resetAuthConfigCacheForTests();
    authStatusMock.mockResolvedValue({
      auth_enabled: true,
      authenticated: true,
      user: 'twinadmin',
      expires_at: null,
      login_required: false,
    });
    const qc = new QueryClient();
    qc.setQueryData(['documents'], { items: [], total: 0 });
    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });
    await waitFor(() => expect(result.current.isAuthenticated).toBe(true));

    // A real 401 flowing through apiFetch must fire the registered handler.
    const realFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Token expired' }), { status: 401 }),
    ) as unknown as typeof fetch;
    try {
      await act(async () => {
        await apiFetch('/documents').catch(() => undefined);
      });
    } finally {
      globalThis.fetch = realFetch;
    }

    await waitFor(() => expect(result.current.needsLogin).toBe(true));
    expect(result.current.isAuthenticated).toBe(false);
    expect(qc.getQueryData(['documents'])).toBeUndefined();
  });

  it('logs out proactively at the JWT exp instant even while idle', async () => {
    (window as Window & typeof globalThis).__twinConfig = { ...AUTH_CONFIG };
    __resetAuthConfigCacheForTests();
    const soon = new Date(Date.now() + 120).toISOString();
    authStatusMock.mockResolvedValue({
      auth_enabled: true,
      authenticated: true,
      user: 'twinadmin',
      expires_at: soon,
      login_required: false,
    });
    const qc = new QueryClient();
    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });
    await waitFor(() => expect(result.current.isAuthenticated).toBe(true));

    // No request is made — the expiry timer alone must drive the logout.
    await waitFor(() => expect(result.current.needsLogin).toBe(true), {
      timeout: 1500,
    });
    expect(result.current.isAuthenticated).toBe(false);
  });

  it('does NOT arm a timer for a beyond-ceiling expiry (stays authenticated)', async () => {
    (window as Window & typeof globalThis).__twinConfig = { ...AUTH_CONFIG };
    __resetAuthConfigCacheForTests();
    authStatusMock.mockResolvedValue({
      auth_enabled: true,
      authenticated: true,
      user: 'twinadmin',
      expires_at: '2099-12-31T23:59:00Z',
      login_required: false,
    });
    const qc = new QueryClient();
    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });
    await waitFor(() => expect(result.current.isAuthenticated).toBe(true));

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.needsLogin).toBe(false);
  });
});

describe('useAuth — local login', () => {
  it('exposes a login screen state and authenticates through LightRAG login', async () => {
    (window as Window & typeof globalThis).__twinConfig = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultFolderId: 'default',
      folders: [{ id: 'default', label: 'Default folder', kind: 'primary' }],
    };
    __resetAuthConfigCacheForTests();
    authStatusMock
      .mockResolvedValueOnce({
        auth_enabled: true,
        authenticated: false,
        user: null,
        expires_at: null,
        login_required: true,
      })
      .mockResolvedValueOnce({
        auth_enabled: true,
        authenticated: true,
        user: 'twinadmin',
        expires_at: '2099-12-31T23:59:00Z',
        login_required: false,
      });
    const qc = new QueryClient();
    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });

    await waitFor(() => expect(result.current.needsLogin).toBe(true));

    await act(async () => {
      await result.current.login('twinadmin', 'secret');
    });

    expect(loginMock).toHaveBeenCalledWith({
      username: 'twinadmin',
      password: 'secret',
    });
    expect(authStatusMock).toHaveBeenLastCalledWith({ token: 'local-jwt-token' });
    expect(window.sessionStorage.getItem('twin-rag.authToken')).toBe(
      'local-jwt-token',
    );
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.name).toBe('twinadmin');
    expect(result.current.user?.palier.label).toBe('Steward');
  });
});
