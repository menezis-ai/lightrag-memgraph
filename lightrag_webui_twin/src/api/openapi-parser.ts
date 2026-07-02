/**
 * OpenAPI 3.1 → Twin ApiTab format converter.
 *
 * The Twin API tab is ISO with the LightRAG WebUI's OpenAPI surface by
 * design (same backend, same FastAPI auto-generated spec, +Twin overlay
 * routes that share `app.include_router`).
 *
 * Hitting `/openapi.json` (FastAPI auto) is the cheapest way to stay
 * truly aligned — every new route added to the host app appears here
 * automatically without a manual catalog. The shape we get back is the
 * standard OpenAPI 3.1; this module reshapes it into `OpenApiGroup[]`
 * for the existing `ApiTab` UI.
 */

import type { HttpMethod, OpenApiEndpoint, OpenApiGroup } from '../types/api';

const HTTP_METHODS: readonly HttpMethod[] = [
  'GET',
  'POST',
  'PUT',
  'PATCH',
  'DELETE',
];

interface RawOperation {
  tags?: readonly string[];
  summary?: string;
  description?: string;
  operationId?: string;
}

interface RawSpec {
  info?: { version?: string; title?: string };
  paths?: Record<string, Record<string, unknown>>;
  tags?: readonly { name?: string; description?: string }[];
}

export interface ParsedOpenApi {
  version: string;
  groups: readonly OpenApiGroup[];
}

/**
 * Convert a standard OpenAPI 3.1 document into the Twin ApiTab's
 * `OpenApiGroup[]` shape. Groups are ordered by tag name; endpoints
 * inside a group keep the spec order (FastAPI emits routes in
 * registration order).
 */
export function parseOpenApiSpec(spec: unknown): ParsedOpenApi {
  if (!spec || typeof spec !== 'object') {
    return { version: 'unknown', groups: [] };
  }
  const raw = spec as RawSpec;
  const version = raw.info?.version ?? 'unknown';
  const tagDescriptions = new Map<string, string>();
  for (const t of raw.tags ?? []) {
    if (t.name) tagDescriptions.set(t.name, t.description ?? '');
  }

  const byTag = new Map<string, OpenApiEndpoint[]>();
  for (const [path, methods] of Object.entries(raw.paths ?? {})) {
    if (!methods || typeof methods !== 'object') continue;
    for (const method of HTTP_METHODS) {
      const op = methods[method.toLowerCase()];
      if (!op || typeof op !== 'object') continue;
      const operation = op as RawOperation;
      const tag = operation.tags?.[0] ?? 'default';
      const summary =
        operation.summary?.trim() ||
        operation.operationId ||
        `${method} ${path}`;
      if (!byTag.has(tag)) byTag.set(tag, []);
      byTag.get(tag)!.push({ m: method, p: path, s: summary });
    }
  }

  const groups: OpenApiGroup[] = [...byTag.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([tag, endpoints]) => ({
      id: tag,
      name: tag,
      desc: tagDescriptions.get(tag) ?? '',
      endpoints,
    }));

  return { version, groups };
}
