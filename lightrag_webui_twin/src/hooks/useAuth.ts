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

import { useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api } from '../api/resources';
import { resolveRuntimeConfig } from '../config/devConfig';
import type { AuthenticatedUser, TwinRuntimeConfig } from '../types/auth';

export interface UseAuthResult {
  user: AuthenticatedUser | null;
  isAuthenticated: boolean;
  config: TwinRuntimeConfig;
  signout: () => Promise<void>;
}

function getDevFlag(): boolean {
  // Vitest sets DEV=true by default; Vite dev also sets it.
  return Boolean(import.meta.env?.DEV);
}

let cachedConfig: TwinRuntimeConfig | null = null;

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
  const config = useMemo(getRuntimeConfig, []);
  const user = config.debugUser ?? null;

  const signout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // Server reachability errors should still let us cycle the client side.
    }
    queryClient.clear();
    if (typeof window !== 'undefined') {
      const target = new URL(config.idpLogoutUrl);
      target.searchParams.set('redirect_uri', window.location.origin);
      window.location.href = target.toString();
    }
  }, [config.idpLogoutUrl, queryClient]);

  return {
    user,
    isAuthenticated: user !== null,
    config,
    signout,
  };
}
