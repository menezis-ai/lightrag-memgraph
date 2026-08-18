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
 * for the existing `ApiTab` UI, carrying the documentation the backend
 * declares: operation description, parameters, request-body example and
 * response codes. Every field is optional so a sparse spec (the MSW
 * fixture, an older backend) still renders.
 */

import type {
  HttpMethod,
  OpenApiEndpoint,
  OpenApiGroup,
  OpenApiParam,
  OpenApiResponseInfo,
} from '../types/api';

const HTTP_METHODS: readonly HttpMethod[] = [
  'GET',
  'POST',
  'PUT',
  'PATCH',
  'DELETE',
];

type RawObject = Record<string, unknown>;

interface RawSpec {
  info?: { version?: string; title?: string };
  paths?: Record<string, Record<string, unknown>>;
  tags?: readonly { name?: string; description?: string }[];
  components?: RawObject;
}

export interface ParsedOpenApi {
  version: string;
  groups: readonly OpenApiGroup[];
}

function isObject(value: unknown): value is RawObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Resolve a local `#/components/...` reference; returns the node itself
 *  when it is not a reference. Unresolvable refs yield `undefined`. */
function deref(node: unknown, spec: RawSpec, depth = 0): RawObject | undefined {
  if (!isObject(node) || depth > 8) return isObject(node) ? node : undefined;
  const ref = node.$ref;
  if (typeof ref !== 'string' || !ref.startsWith('#/')) return node;
  let target: unknown = spec;
  for (const part of ref.slice(2).split('/')) {
    if (!isObject(target)) return undefined;
    target = target[part];
  }
  return deref(target, spec, depth + 1);
}

/** Human-readable type label for a (dereferenced) schema node. */
function schemaTypeLabel(schema: RawObject | undefined, spec: RawSpec): string {
  if (!schema) return '';
  if (typeof schema.type === 'string') {
    if (schema.type === 'array') {
      const items = deref(schema.items, spec);
      const inner = schemaTypeLabel(items, spec);
      return inner ? `array of ${inner}` : 'array';
    }
    return schema.type;
  }
  const anyOf = schema.anyOf ?? schema.oneOf;
  if (Array.isArray(anyOf)) {
    const labels = anyOf
      .map((s) => schemaTypeLabel(deref(s, spec), spec))
      .filter((l) => l && l !== 'null');
    if (labels.length) return [...new Set(labels)].join(' | ');
  }
  if (Array.isArray(schema.enum)) return 'enum';
  return '';
}

/** First example advertised for a schema/parameter, stringified for display. */
function firstExample(node: RawObject | undefined): unknown {
  if (!node) return undefined;
  if (node.example !== undefined) return node.example;
  if (Array.isArray(node.examples) && node.examples.length) {
    return node.examples[0];
  }
  if (node.default !== undefined) return node.default;
  return undefined;
}

function displayValue(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function parseParameters(
  rawParams: unknown,
  spec: RawSpec,
): OpenApiParam[] | undefined {
  if (!Array.isArray(rawParams) || !rawParams.length) return undefined;
  const params: OpenApiParam[] = [];
  for (const raw of rawParams) {
    const p = deref(raw, spec);
    if (!p || typeof p.name !== 'string') continue;
    const where = p.in;
    if (
      where !== 'path' &&
      where !== 'query' &&
      where !== 'header' &&
      where !== 'cookie'
    ) {
      continue;
    }
    const schema = deref(p.schema, spec);
    params.push({
      name: p.name,
      in: where,
      type: schemaTypeLabel(schema, spec),
      required: p.required === true,
      desc: typeof p.description === 'string' ? p.description : '',
      example: displayValue(firstExample(p) ?? firstExample(schema)),
    });
  }
  return params.length ? params : undefined;
}

/** Schema-compatible placeholder for a required field that ships no
 *  example: respects the declared type (and minimum / first enum value)
 *  so the prefilled body round-trips validation once filled in. */
function placeholderForSchema(
  schema: RawObject | undefined,
  spec: RawSpec,
): unknown {
  if (!schema) return '';
  if (Array.isArray(schema.enum) && schema.enum.length) return schema.enum[0];
  const anyOf = schema.anyOf ?? schema.oneOf;
  if (Array.isArray(anyOf) && anyOf.length) {
    return placeholderForSchema(deref(anyOf[0], spec), spec);
  }
  switch (schema.type) {
    case 'integer':
    case 'number':
      return typeof schema.minimum === 'number' ? schema.minimum : 0;
    case 'boolean':
      return false;
    case 'array':
      return [];
    case 'object':
      return {};
    default:
      return '';
  }
}

/** Build a skeleton example object from an object schema's properties,
 *  using each property's own example/default when available. */
function skeletonFromSchema(
  schema: RawObject | undefined,
  spec: RawSpec,
  depth = 0,
): unknown {
  if (!schema || depth > 4) return undefined;
  const example = firstExample(schema);
  if (example !== undefined) return example;
  const anyOf = schema.anyOf ?? schema.oneOf;
  if (Array.isArray(anyOf) && anyOf.length) {
    for (const candidate of anyOf) {
      const value = skeletonFromSchema(deref(candidate, spec), spec, depth + 1);
      if (value !== undefined) return value;
    }
    return undefined;
  }
  if (schema.type === 'object' || isObject(schema.properties)) {
    const props = isObject(schema.properties) ? schema.properties : {};
    const required = Array.isArray(schema.required) ? schema.required : [];
    const out: RawObject = {};
    for (const [key, rawProp] of Object.entries(props)) {
      const prop = deref(rawProp, spec);
      const value = skeletonFromSchema(prop, spec, depth + 1);
      if (value !== undefined) {
        out[key] = value;
      } else if (required.includes(key)) {
        out[key] = placeholderForSchema(prop, spec);
      }
    }
    return Object.keys(out).length ? out : undefined;
  }
  if (schema.type === 'array') {
    const item = skeletonFromSchema(deref(schema.items, spec), spec, depth + 1);
    return item === undefined ? undefined : [item];
  }
  return undefined;
}

function parseRequestBody(
  rawBody: unknown,
  spec: RawSpec,
): { example?: string; hasBody: boolean } {
  const body = deref(rawBody, spec);
  if (!body) return { hasBody: false };
  const content = isObject(body.content) ? body.content : undefined;
  const json = content ? deref(content['application/json'], spec) : undefined;
  if (!json) return { hasBody: true };
  const schema = deref(json.schema, spec);
  const example = firstExample(json) ?? skeletonFromSchema(schema, spec);
  return {
    hasBody: true,
    example:
      example === undefined ? undefined : JSON.stringify(example, null, 2),
  };
}

function parseResponses(
  rawResponses: unknown,
  spec: RawSpec,
): OpenApiResponseInfo[] | undefined {
  if (!isObject(rawResponses)) return undefined;
  const out: OpenApiResponseInfo[] = [];
  for (const [code, raw] of Object.entries(rawResponses)) {
    const resp = deref(raw, spec);
    out.push({
      code,
      desc:
        resp && typeof resp.description === 'string' ? resp.description : '',
    });
  }
  out.sort((a, b) => a.code.localeCompare(b.code));
  return out.length ? out : undefined;
}

interface RawOperation {
  tags?: readonly string[];
  summary?: string;
  description?: string;
  operationId?: string;
  parameters?: unknown;
  requestBody?: unknown;
  responses?: unknown;
  security?: unknown;
}

/** Per-operation auth state from the spec's `security` field.
 *  `undefined` = the spec says nothing (sparse fixture); `false` = public
 *  (no requirement, or an empty requirement `{}` allows anonymous);
 *  `true` = every alternative requires credentials. */
function parseSecured(rawSecurity: unknown): boolean | undefined {
  if (!Array.isArray(rawSecurity)) return undefined;
  if (!rawSecurity.length) return false;
  const anonymousAllowed = rawSecurity.some(
    (entry) => isObject(entry) && Object.keys(entry).length === 0,
  );
  return !anonymousAllowed;
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
    // Path-level parameters apply to every operation of the path.
    const pathParams = (methods as RawObject).parameters;
    for (const method of HTTP_METHODS) {
      const op = methods[method.toLowerCase()];
      if (!op || typeof op !== 'object') continue;
      const operation = op as RawOperation;
      const tag = operation.tags?.[0] ?? 'default';
      const summary =
        operation.summary?.trim() ||
        operation.operationId ||
        `${method} ${path}`;
      const mergedParams = [
        ...(Array.isArray(pathParams) ? pathParams : []),
        ...(Array.isArray(operation.parameters) ? operation.parameters : []),
      ];
      const params = parseParameters(mergedParams, raw);
      const { example, hasBody } = parseRequestBody(operation.requestBody, raw);
      const responses = parseResponses(operation.responses, raw);
      const desc = operation.description?.trim();
      const secured = parseSecured(operation.security);
      if (!byTag.has(tag)) byTag.set(tag, []);
      byTag.get(tag)!.push({
        m: method,
        p: path,
        s: summary,
        ...(desc ? { desc } : {}),
        ...(params ? { params } : {}),
        ...(example ? { bodyExample: example } : {}),
        // Keep the explicit false value. The API explorer must distinguish
        // a real, body-less operation from a sparse legacy fixture that says
        // nothing about requestBody.
        hasBody,
        ...(responses ? { responses } : {}),
        ...(secured === undefined ? {} : { secured }),
      });
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
