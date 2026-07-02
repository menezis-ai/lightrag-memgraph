/**
 * OpenAPI tab types — Swagger-style endpoint browser for the LightRAG surface.
 *
 * Twin overlay extends the LightRAG OpenAPI surface with `/twin/api/*`
 * routes. `tag_filter` is honored server-side on Twin query routes via
 * `TAGGED_WITH`; `/twin/api/query/data` also falls back from filtered
 * graph-only modes to `mix` when chunks exist but KG rows do not. Native
 * LightRAG routes pass through unchanged. The previous claim about
 * "gateway injects `tag_filter` / `visibility` scoping" was incorrect
 * and was retracted by audit C8. These types mirror what the
 * `/openapi.json` document exposes, narrowed down to the fields the UI
 * renders.
 */

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

export interface OpenApiEndpoint {
  /** HTTP method. */
  m: HttpMethod;
  /** Request path (e.g. `/documents/upload`). */
  p: string;
  /** Short human summary (e.g. `Upload Document`). */
  s: string;
}

export interface OpenApiGroup {
  id: string;
  name: string;
  desc: string;
  endpoints: readonly OpenApiEndpoint[];
}

export interface OpenApiServer {
  id: string;
  label: string;
}

export interface MockResponse {
  status: number;
  statusText: string;
  tookMs: number;
  body: string;
}

export interface MethodColor {
  bg: string;
  fg: string;
  border: string;
}

export const METHOD_COLOR: Record<HttpMethod, MethodColor> = {
  GET: { bg: '#E6EFFA', fg: '#1B5BAE', border: '#B5D4F4' },
  POST: { bg: '#E5F3EA', fg: '#1F7A3A', border: '#B6DDC1' },
  DELETE: { bg: '#FBE7E7', fg: '#A33030', border: '#F0B7B7' },
  PUT: { bg: '#FCEFDE', fg: '#9C5A0E', border: '#F0CFA0' },
  PATCH: { bg: '#E8F0EE', fg: '#15706B', border: '#A9D2CC' },
};
