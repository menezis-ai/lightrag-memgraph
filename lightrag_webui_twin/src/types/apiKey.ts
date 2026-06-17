/**
 * Per-operator API keys minted via Settings → API keys.
 *
 * Distinct from the static `LIGHTRAG_API_KEY` env value (infra root key,
 * managed via deploy env, never surfaced in the UI).
 *
 * Wire format mirrors `server/api_key_routes.py:ApiKeyPublic`. The
 * `hash` field is intentionally absent — the backend strips it on every
 * response. `full_value` is present ONLY on the POST response and must
 * be displayed once, copied by the operator, then forgotten by the UI.
 */

export interface ApiKeyPublic {
  /** Opaque public id (used to revoke). */
  id: string;
  /** Operator-chosen label. 1..120 chars. */
  name: string;
  /** Displayable preview, ends with an ellipsis: `twk_xxxxxxxx…`. */
  prefix: string;
  /** Creation timestamp, milliseconds since epoch (UTC). */
  created_at: number;
  /** Identity that minted the key (SSO subject / email / username). */
  created_by: string;
  /** Last time the key successfully authenticated, or null if unused. */
  last_used_at: number | null;
  /** When set, the key no longer authenticates. */
  revoked_at: number | null;
}

/**
 * POST /settings/api-keys response. The `full_value` is the only place
 * the raw secret is ever returned by the API — display it ONCE in a
 * one-time-reveal modal, copy-to-clipboard, then drop from state.
 */
export interface ApiKeyCreated extends ApiKeyPublic {
  full_value: string;
}
