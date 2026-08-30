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
import { loginErrorMessage, logTechnicalError } from '../lib/errorMessages';
import { resolveRuntimeConfig } from '../config/devConfig';
import type { AuthenticatedUser, TwinRuntimeConfig } from '../types/auth';

/** setTimeout clamps delays above ~24.8 days (2^31-1 ms) to fire immediately,
 *  so an expiry that far out (or a sentinel like the test's year-2099 token)
 *  must NOT arm a timer — the reactive 401 path covers it instead. */
const MAX_EXPIRY_TIMER_MS = 2_147_483_647;

export interface UseAuthResult {
  user: AuthenticatedUser | null;
  /** Backend auth posture. `null` means not resolved or unavailable. */
  authEnabled: boolean | null;
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

function readOnlyCredentialUser(
  username: string,
  config: TwinRuntimeConfig,
): AuthenticatedUser {
  const name = username || 'authenticated-user';
  return {
    sso_subject: name,
    email: name,
    name,
    palier: {
      level: 1,
      label: 'Reader',
      scopes: ['twin:read'],
    },
    folders: (config.folders ?? []).map((folder) => folder.id),
    idp: 'credential-only',
    idp_realm: 'unclaimed',
    sub: name,
    session_expires: 'session',
    gateway_scopes: ['read:documents', 'read:query', 'read:activity'],
  };
}

function readOnlyUser(user: AuthenticatedUser): AuthenticatedUser {
  return {
    ...user,
    palier: {
      level: 1,
      label: 'Reader',
      scopes: ['twin:read'],
    },
    gateway_scopes: user.gateway_scopes.filter((scope) =>
      scope.startsWith('read:'),
    ),
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
    identity: AuthenticatedUser | null;
    authEnabled: boolean | null;
  }>({
    checked: false,
    authenticated: config.debugUser !== undefined,
    loginRequired: false,
    user: null,
    identity: null,
    authEnabled: null,
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
      loginRequired: prev.authEnabled === true,
      user: null,
      identity: null,
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
          identity: status.identity ?? null,
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
          identity: null,
          // A transport failure is not evidence of an open-access runtime.
          authEnabled: null,
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

  let user: AuthenticatedUser | null = null;
  if (authState.authenticated) {
    user =
      authState.identity ??
      (authState.user
        ? readOnlyCredentialUser(authState.user, config)
        : config.debugUser ?? null);
  } else if (!authState.checked) {
    // Preserve the dev bootstrap while /auth-status is pending. Consumers
    // still fail closed because authEnabled remains null until it resolves.
    user = config.debugUser ?? null;
  }
  if (authState.authEnabled === null && user) {
    user = readOnlyUser(user);
  }

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
          identity: status.identity ?? null,
          authEnabled: status.auth_enabled,
        });
        scheduleExpiry(status.expires_at);
      } catch (err) {
        setSessionAuthToken(null);
        // Operator-facing copy only — a 401 here is a failed credential
        // check ("Incorrect username or password."), never the raw
        // "POST /login → 401" transport string.
        logTechnicalError('login', err);
        setLoginError(loginErrorMessage(err));
        setAuthState((prev) => ({
          ...prev,
          checked: true,
          authenticated: false,
          loginRequired: true,
          user: null,
          identity: null,
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
    authEnabled: authState.authEnabled,
    isAuthenticated: user !== null,
    isCheckingAuth: !authState.checked,
    needsLogin: authState.checked && authState.loginRequired && user === null,
    loginError,
    config,
    login: doLogin,
    signout,
  };
}
