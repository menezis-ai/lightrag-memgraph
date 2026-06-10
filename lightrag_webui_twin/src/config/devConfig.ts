/**
 * Dev-only runtime config fallback.
 *
 * Used when:
 *   - `import.meta.env.DEV === true` (Vite dev server, vitest run); AND
 *   - `window.__twinConfig` is missing or still equals the FastAPI placeholder
 *     `'__TWIN_CONFIG_JSON__'`, which means the substitution layer didn't
 *     run (typical of `bun run dev` hitting `/index.html` raw).
 *
 * In production, the Twin FastAPI sub-app serves `index.html` and substitutes
 * the placeholder with real values (env vars + decoded JWT claims) before
 * sending the response — this file is never reached.
 */

import type { TwinRuntimeConfig } from '../types/auth';

export const DEV_CONFIG: TwinRuntimeConfig = {
  apiBaseUrl: '/twin/api',
  lightragBaseUrl: '',
  idpLogoutUrl: 'https://idp.twin.internal/realms/twin/protocol/openid-connect/logout',
  defaultFolderId: 'default',
  maxFolders: 5,
  folders: [
    {
      id: 'default',
      label: 'Default folder',
      kind: 'primary',
      description: 'SRE-provisioned default folder for this KB.',
    },
    {
      id: 'sandbox',
      label: 'Sandbox',
      kind: 'sandbox',
      description: 'Optional test folder isolated from agent-facing data.',
    },
  ],
  debugUser: {
    sso_subject: 'claire.benoit@demo.local',
    email: 'claire.benoit@demo.local',
    name: 'Claire Benoit',
    palier: {
      level: 3,
      label: 'Steward',
      scopes: ['twin:read', 'twin:write', 'twin:approve'],
    },
    folders: ['default', 'sandbox'],
    idp: 'keycloak',
    idp_realm: 'twin-main',
    sub: 'clb-7f4e',
    session_expires: '2026-05-19T23:59:00Z',
    gateway_scopes: [
      'read:documents',
      'write:documents',
      'read:query',
      'read:activity',
      'admin:tags',
      'admin:folders',
    ],
  },
};

const PLACEHOLDER = '__TWIN_CONFIG_JSON__';

/**
 * Resolve the active runtime config, applying the dev fallback when the
 * placeholder is still present (or `window.__twinConfig` is missing).
 *
 * In a PROD build, the absence of a substituted config is normally a hard
 * error — it means the FastAPI sub-app didn't run and the bearer / scopes
 * are not in the page. But for the standalone OVH demo deploy (Caddy + MSW
 * client-side, no FastAPI behind), `VITE_FORCE_MSW=true` is the explicit
 * opt-in for "we know there's no real backend, use the dev fallback".
 */
export function resolveRuntimeConfig(
  source: TwinRuntimeConfig | string | undefined,
  isDev: boolean,
): TwinRuntimeConfig {
  if (isDev && typeof window !== 'undefined' && window.__twinE2eRuntimeConfig) {
    return window.__twinE2eRuntimeConfig;
  }
  if (!source || source === PLACEHOLDER) {
    if (isDev) return DEV_CONFIG;
    // PROD standalone demo path — VITE_FORCE_MSW=true means we deliberately
    // shipped without a FastAPI substitution layer; use the demo fallback.
    if (import.meta.env.VITE_FORCE_MSW === 'true') return DEV_CONFIG;
    throw new Error(
      '[twin-webui] window.__twinConfig was not substituted by the server. ' +
        'Check that the FastAPI sub-app is serving index.html via register(mount_server=True).',
    );
  }
  if (typeof source === 'string') {
    return JSON.parse(source) as TwinRuntimeConfig;
  }
  return source;
}
