/**
 * Runtime identity card behind Settings → About.
 *
 * Mirrors ``server/system_info_routes.py:AboutResponse``. The payload is
 * two-tier: any caller the backend serves gets `twin` + `lightrag`, and the
 * deployment-shape blocks (`memgraph`, `runtime`, `storage`, `overlay`)
 * arrive only when `admin` is true — they are `null` otherwise. The tier
 * split, not the route gate, is what keeps topology admin-only: the route
 * sits behind LightRAG-parity auth, which serves anonymous callers on an
 * instance with no auth backend configured.
 *
 * Treat every optional block as genuinely absent rather than merely empty:
 * a non-admin session renders a valid, shorter card.
 */

export interface LightRagVersions {
  /** LightRAG's own version, recovered from the composite string. */
  native: string | null;
  /** `register()`'s patched value, e.g. `1.4.9.11+memgraph-1.1.0`. */
  composite: string | null;
}

export interface MemgraphInfo {
  reachable: boolean;
  version: string | null;
  /**
   * Tri-state. `null` = the capability probe could not answer (Memgraph
   * unreachable, or the probe itself failed). Render it as "unknown" — an
   * absence of evidence is not evidence of the floor tier.
  */
  mage: boolean | null;
  /** Null when the capability probe failed or an override skipped the probe. */
  procedures: number | null;
  /** Exception class name when the probe failed; null when reachable. */
  error: string | null;
}

export interface RuntimeInfo {
  python: string;
  implementation: string;
  platform: string;
}

export interface AboutResponse {
  twin: string;
  lightrag: LightRagVersions;
  admin: boolean;
  memgraph: MemgraphInfo | null;
  runtime: RuntimeInfo | null;
  /** Storage slot → bound backend class name. */
  storage: Record<string, string> | null;
  overlay: Record<string, boolean> | null;
  /**
   * Configured storage limits (admin-only, optional on older backends).
   * `vector_index_capacity` is the capacity a new vector index would get;
   * an existing index keeps the one it was created with.
   */
  limits?: Record<string, number> | null;
}
