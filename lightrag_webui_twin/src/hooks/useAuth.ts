/**
 * useAuth — the single source of truth for the current MyAccess identity in
 * the WebUI.
 *
 * Reads `window.__twinConfig.debugUser` (dev) or the JWT-decoded payload the
 * server already attached at injection time. Components downstream (Topbar,
 * SettingsTab profile section, capability gating in DocumentsTab/TagsTab)
 * consume `palier` to decide what to enable. There is NO PalierSwitcher and
 * NO MyAccessPill — palier is JWT-only, never UI-mutable (Louis 2026-05-28
 * + PO 2026-05-29).
 *
 * `signout()` follows the spec from the sprint brief:
 *   1. POST /twin/api/auth/logout (revokes server-side session)
 *   2. queryClient.clear() (kills cached PII)
 *   3. window.location.href = idpLogoutUrl?redirect_uri=window.location.origin
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api } from '../api/resources';
import { setSessionAuthToken } from '../api/client';
import { resolveRuntimeConfig } from '../config/devConfig';
import type { AuthenticatedUser, TwinRuntimeConfig } from '../types/auth';

export interface UseAuthResult {
  user: AuthenticatedUser | null;
  isAuthenticated: boolean;
  isCheckingAuth: boolean;
  needsLogin: boolean;
  loginError: string | null;
  config: TwinRuntimeConfig;
  login: (username: string, password: string) => Promise<void>;
  signout: () => Promise<void>;
}

function getDevFlag(): boolean {
  // Vitest sets DEV=true by default; Vite dev also sets it.
  return Boolean(import.meta.env?.DEV);
}

let cachedConfig: TwinRuntimeConfig | null = null;

const TWIN_BROWSER_STORAGE_PREFIXES = ['twin-rag.'] as const;

export function clearTwinBrowserState(): void {
  if (typeof window === 'undefined') return;
  for (let i = window.localStorage.length - 1; i >= 0; i -= 1) {
    const key = window.localStorage.key(i);
    if (!key) continue;
    if (TWIN_BROWSER_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
      window.localStorage.removeItem(key);
    }
  }
}

function localUser(username: string | null | undefined, config: TwinRuntimeConfig): AuthenticatedUser {
  const name = username || 'operator';
  const isAdmin = /admin/i.test(name);
  return {
    sso_subject: name,
    email: name,
    name,
    palier: {
      level: isAdmin ? 3 : 2,
      label: isAdmin ? 'Steward' : 'Contributor',
      scopes: isAdmin
        ? ['twin:read', 'twin:write', 'twin:approve']
        : ['twin:read', 'twin:write'],
    },
    workspaces: (config.folders ?? config.spaces ?? []).map((folder) => folder.id),
    idp: 'local-jwt',
    idp_realm: 'local',
    sub: name,
    session_expires: 'session',
    gateway_scopes: isAdmin
      ? [
          'read:documents',
          'write:documents',
          'read:query',
          'read:activity',
          'admin:tags',
          'admin:spaces',
        ]
      : ['read:documents', 'write:documents', 'read:query', 'read:activity'],
  };
}

function getRuntimeConfig(): TwinRuntimeConfig {
  if (cachedConfig) return cachedConfig;
  const raw =
    typeof window !== 'undefined' ? window.__twinConfig : undefined;
  cachedConfig = resolveRuntimeConfig(raw, getDevFlag());
  return cachedConfig;
}

/** Test-only: reset memoized config between tests. */
export function __resetAuthConfigCacheForTests(): void {
  cachedConfig = null;
}

export function useAuth(): UseAuthResult {
  const queryClient = useQueryClient();
  const config = useMemo(() => getRuntimeConfig(), []);
  const [authState, setAuthState] = useState<{
    checked: boolean;
    authenticated: boolean;
    loginRequired: boolean;
    user: string | null;
    authEnabled: boolean;
  }>({
    checked: false,
    authenticated: config.debugUser !== undefined,
    loginRequired: false,
    user: null,
    authEnabled: false,
  });
  const [loginError, setLoginError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.authStatus()
      .then((status) => {
        if (cancelled) return;
        setAuthState({
          checked: true,
          authenticated: status.authenticated,
          loginRequired: status.login_required,
          user: status.user ?? null,
          authEnabled: status.auth_enabled,
        });
      })
      .catch(() => {
        if (cancelled) return;
        setAuthState({
          checked: true,
          authenticated: config.debugUser !== undefined,
          loginRequired: config.debugUser === undefined,
          user: null,
          authEnabled: false,
        });
      });
    return () => {
      cancelled = true;
    };
  }, [config.debugUser]);

  const user = config.debugUser ?? (
    authState.authenticated ? localUser(authState.user, config) : null
  );

  const doLogin = useCallback(
    async (username: string, password: string) => {
      setLoginError(null);
      try {
        const response = await api.login({ username, password });
        setSessionAuthToken(response.access_token);
        const status = await api.authStatus({ token: response.access_token });
        setAuthState({
          checked: true,
          authenticated: status.authenticated,
          loginRequired: status.login_required,
          user: status.user ?? username,
          authEnabled: status.auth_enabled,
        });
      } catch (err) {
        setSessionAuthToken(null);
        const message =
          err instanceof Error ? err.message : 'Authentication failed';
        setLoginError(message);
        setAuthState((prev) => ({
          ...prev,
          checked: true,
          authenticated: false,
          loginRequired: true,
        }));
        throw err;
      }
    },
    [],
  );

  const signout = useCallback(async () => {
    try {
      await api.logoutLocal();
    } catch {
      // Keep clearing client-side state even if the local endpoint is absent.
    }
    try {
      await api.logout();
    } catch {
      // Server reachability errors should still let us cycle the client side.
    }
    setSessionAuthToken(null);
    queryClient.clear();
    clearTwinBrowserState();
    if (typeof window !== 'undefined') {
      if (
        (window as Window & { __TWIN_E2E_BLOCK_SIGNOUT_NAVIGATION?: boolean })
          .__TWIN_E2E_BLOCK_SIGNOUT_NAVIGATION
      ) {
        return;
      }
      if (authState.authEnabled) {
        window.location.reload();
      } else {
        const target = new URL(config.idpLogoutUrl);
        target.searchParams.set('redirect_uri', window.location.origin);
        window.location.href = target.toString();
      }
    }
  }, [authState.authEnabled, config.idpLogoutUrl, queryClient]);

  return {
    user,
    isAuthenticated: user !== null,
    isCheckingAuth: !authState.checked,
    needsLogin: authState.checked && authState.loginRequired && user === null,
    loginError,
    config,
    login: doLogin,
    signout,
  };
}
