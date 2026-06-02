/**
 * Unit tests for useAuth.
 *
 * Covers: dev fallback (debug user), placeholder detection, signout flow
 * (API call → queryClient.clear → redirect), config caching.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { __resetAuthConfigCacheForTests, useAuth } from './useAuth';
import { resolveRuntimeConfig, DEV_CONFIG } from '../config/devConfig';

const logoutMock = vi.hoisted(() => vi.fn());

vi.mock('../api/resources', () => ({
  api: {
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

beforeEach(() => {
  __resetAuthConfigCacheForTests();
  logoutMock.mockResolvedValue({ ok: true });
  (window as Window & typeof globalThis).__twinConfig = undefined;
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
  (window as Window & typeof globalThis).__twinConfig = originalConfig;
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
      /window\.__twinConfig/,
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
});

describe('useAuth — dev fallback', () => {
  it('exposes the dev debugUser when no config is injected', () => {
    const qc = new QueryClient();
    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.palier.level).toBe(3);
    expect(result.current.user?.palier.label).toBe('Steward');
    expect(result.current.user?.workspaces).toContain('default');
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
  it('clears the query cache and redirects to the IdP logout endpoint', async () => {
    const qc = new QueryClient();
    qc.setQueryData(['documents'], { items: [], total: 0 });
    window.localStorage.setItem('twin-rag.threads.v2', JSON.stringify([]));
    expect(qc.getQueryData(['documents'])).not.toBeUndefined();
    expect(window.localStorage.getItem('twin-rag.threads.v2')).not.toBeNull();

    const { result } = renderHook(() => useAuth(), { wrapper: wrap(qc) });

    await act(async () => {
      await result.current.signout();
    });

    expect(logoutMock).toHaveBeenCalledOnce();
    expect(qc.getQueryData(['documents'])).toBeUndefined();
    expect(window.localStorage.getItem('twin-rag.threads.v2')).toBeNull();
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

    expect(logoutMock).toHaveBeenCalledOnce();
    expect(qc.getQueryData(['notifications'])).toBeUndefined();
    expect(window.location.href).toMatch(/realms\/twin/);
  });
});
