/**
 * Authentication & MyAccess types.
 *
 * Aligned on the doctrine confirmed by Louis HORVAT 2026-05-28: no internal
 * RBAC, everything is driven by MyAccess claims forwarded as JWT cookie. The
 * frontend reads the JWT-decoded payload from a server-injected runtime
 * config; it does NOT introspect or sign tokens itself.
 */

export type PalierLevel = 1 | 2 | 3;

export interface Palier {
  level: PalierLevel;
  label: 'Reader' | 'Contributor' | 'Steward';
  /** OAuth2-style scopes carried by the JWT. Used for capability gating. */
  scopes: readonly string[];
}

export interface AuthenticatedUser {
  /** Stable subject id from the IdP (typically email or UID). */
  sso_subject: string;
  email: string;
  name: string;
  palier: Palier;
  /** Workspace ids the user can switch into, fed by MyAccess claim. */
  workspaces: readonly string[];
}

/**
 * Runtime config injected by the Twin FastAPI sub-app via index.html string
 * substitution (placeholder `__TWIN_CONFIG_JSON__`). See sprint Étape 0 brief.
 */
export interface TwinRuntimeConfig {
  apiBaseUrl: string;
  lightragBaseUrl: string;
  idpLogoutUrl: string;
  /** Debug-only: bypass IdP and pretend to be this user. Stripped in prod. */
  debugUser?: AuthenticatedUser;
}

declare global {
  interface Window {
    __twinConfig?: TwinRuntimeConfig | string;
  }
}

export {};
