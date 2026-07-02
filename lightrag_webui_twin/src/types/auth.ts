/**
 * Authentication & MyAccess types.
 *
 * Aligned on the compliance doctrine: no internal RBAC, everything is driven
 * by IdP / MyAccess claims forwarded as a JWT cookie. The
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
  /** Folder ids the user can switch into, fed by MyAccess claim. */
  folders: readonly string[];

  // ── Identity-provider trace (rendered read-only in Settings → Profile) ──
  /** IdP identifier (e.g. "keycloak"). */
  idp: string;
  /** IdP realm name (e.g. "twin-main"). */
  idp_realm: string;
  /** Short identifier the IdP carries in the `sub` JWT claim. */
  sub: string;
  /** ISO timestamp when the bearer token expires. */
  session_expires: string;
  /**
   * Gateway-level OAuth2 scopes carried by the JWT. Distinct from `palier.scopes`
   * (which are the Twin-internal capability tokens) — these are what the API
   * gateway / FastAPI middleware checks. Rendered as chip list in Profile.
   */
  gateway_scopes: readonly string[];
}

export type TwinFolderKind = 'primary' | 'sandbox' | 'staging' | 'archive' | 'custom';

export interface TwinFolderConfig {
  id: string;
  label: string;
  kind?: TwinFolderKind;
  description?: string;
  sources?: number;
}

/**
 * Runtime config injected by the Twin FastAPI sub-app via index.html string
 * substitution (placeholder `__TWIN_CONFIG_JSON__`). See sprint Étape 0 brief.
 */
export interface TwinRuntimeConfig {
  apiBaseUrl: string;
  lightragBaseUrl: string;
  idpLogoutUrl: string;
  /** Default Twin folder selected at boot. SRE/DevOps owns this via env. */
  defaultFolderId?: string;
  /** Admin-created logical folders inside the same Memgraph DB / KB. Max 5. */
  folders?: readonly TwinFolderConfig[];
  maxFolders?: number;
  /** Debug-only: bypass IdP and pretend to be this user. Stripped in prod. */
  debugUser?: AuthenticatedUser;
}

declare global {
  interface Window {
    __twinConfig?: TwinRuntimeConfig | string;
    __twinE2eRuntimeConfig?: TwinRuntimeConfig;
  }
}
