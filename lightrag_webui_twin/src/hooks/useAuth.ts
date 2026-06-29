/**
 * useAuth — the single source of truth for the current MyAccess identity in
 * the WebUI.
 *
 * Reads `globalThis.__twinConfig.debugUser` (dev) or the JWT-decoded payload the
 * server already attached at injection time. Components downstream (Topbar,
 * SettingsTab profile section, capability gating in DocumentsTab/TagsTab)
 * consume `palier` to decide what to enable. There is NO PalierSwitcher and
 * NO MyAccessPill — palier is JWT-only, never UI-mutable (compliance review 2026-05-28
 * + PO 2026-05-29).
 *
 * `signout()` follows the spec from the sprint brief:
 *   1. POST /twin/api/auth/logout (revokes server-side session)
 *   2. queryClient.clear() (kills cached PII)
 *   3. globalThis.location.href = idpLogoutUrl?redirect_uri=globalThis.location.origin
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api } from '../api/resources';
import { onUnauthorized, setSessionAuthToken } from '../api/client';
import { resolveRuntimeConfig } from '../config/devConfig';
import type { AuthenticatedUser, TwinRuntimeConfig } from '../types/auth';

/** setTimeout clamps delays above ~24.8 days (2^31-1 ms) to fire immediately,
 *  so an expiry that far out (or a sentinel like the test's year-2099 token)
 *  must NOT arm a timer — the reactive 401 path covers it instead. */
const MAX_EXPIRY_TIMER_MS = 2_147_483_647;

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
  if (globalThis.window === undefined) return;
  for (let i = globalThis.localStorage.length - 1; i >= 0; i -= 1) {
    const key = globalThis.localStorage.key(i);
    if (!key) continue;
    if (TWIN_BROWSER_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
      globalThis.localStorage.removeItem(key);
    }
  }
}

function localUser(username: string | null | undefined, config: TwinRuntimeConfig): AuthenticatedUser {
  const name = username || 'operator@twin.local';
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
    folders: (config.folders ?? []).map((folder) => folder.id),
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
          'admin:folders',
        ]
      : ['read:documents', 'write:documents', 'read:query', 'read:activity'],
  };
}

function getRuntimeConfig(): TwinRuntimeConfig {
  if (cachedConfig) return cachedConfig;
  const raw =
    globalThis.window === undefined
      ? undefined
      : globalThis.window.__twinConfig;
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
  const expiryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Tear the client back down to an unauthenticated state when the session
   *  ends mid-use — fired both reactively (a backend 401) and proactively
   *  (the expiry timer). Clears cached PII before dropping to the login
   *  screen. In open-access mode (`authEnabled === false`, LightRAG parity)
   *  we never force a login screen that does not exist. */
  const expireSession = useCallback(() => {
    if (expiryTimerRef.current) {
      clearTimeout(expiryTimerRef.current);
      expiryTimerRef.current = null;
    }
    setSessionAuthToken(null);
    queryClient.clear();
    clearTwinBrowserState();
    setAuthState((prev) => ({
      ...prev,
      checked: true,
      authenticated: false,
      loginRequired: prev.authEnabled,
      user: null,
    }));
  }, [queryClient]);

  /** Arm a single timer that expires the session at the JWT `exp` instant,
   *  so an idle operator is logged out without waiting for the next request.
   *  Skips unparseable / null / beyond-ceiling values (see MAX_EXPIRY_TIMER_MS). */
  const scheduleExpiry = useCallback(
    (expiresAt: string | null | undefined) => {
      if (expiryTimerRef.current) {
        clearTimeout(expiryTimerRef.current);
        expiryTimerRef.current = null;
      }
      if (!expiresAt) return;
      const ms = new Date(expiresAt).getTime() - Date.now();
      if (Number.isNaN(ms) || ms > MAX_EXPIRY_TIMER_MS) return;
      expiryTimerRef.current = setTimeout(expireSession, Math.max(0, ms));
    },
    [expireSession],
  );

  // Reactive path: any mid-session 401 (expired/revoked token) drops to login.
  useEffect(() => onUnauthorized(expireSession), [expireSession]);

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
        scheduleExpiry(status.expires_at);
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
  }, [config.debugUser, scheduleExpiry]);

  // Cancel any pending expiry timer when the hook unmounts.
  useEffect(
    () => () => {
      if (expiryTimerRef.current) clearTimeout(expiryTimerRef.current);
    },
    [],
  );

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
        scheduleExpiry(status.expires_at);
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
    [scheduleExpiry],
  );

  const signout = useCallback(async () => {
    if (expiryTimerRef.current) {
      clearTimeout(expiryTimerRef.current);
      expiryTimerRef.current = null;
    }
    try {
      await api.logout();
    } catch {
      // Server reachability errors should still let us cycle the client side.
    }
    setSessionAuthToken(null);
    queryClient.clear();
    clearTwinBrowserState();
    if (globalThis.window !== undefined) {
      if (
        (globalThis.window as Window & { __TWIN_E2E_BLOCK_SIGNOUT_NAVIGATION?: boolean })
          .__TWIN_E2E_BLOCK_SIGNOUT_NAVIGATION
      ) {
        return;
      }
      if (authState.authEnabled) {
        globalThis.location.reload();
      } else {
        const target = new URL(config.idpLogoutUrl);
        target.searchParams.set('redirect_uri', globalThis.location.origin);
        globalThis.location.href = target.toString();
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
