import type { HttpMethod, OpenApiEndpoint, OpenApiParam } from '../../types/api';

const QUERY_ENDPOINTS = new Set([
  '/query', '/query/data', '/query/stream', '/twin/api/query',
  '/twin/api/query/data', '/twin/api/query/stream',
]);
const METHODS_WITH_BODY = new Set<HttpMethod>(['POST', 'PUT', 'PATCH']);

export function endpointHasBody(ep: OpenApiEndpoint): boolean {
  return ep.hasBody ?? METHODS_WITH_BODY.has(ep.m);
}

export function paramKey(p: OpenApiParam): string {
  return `${p.in}:${p.name}`;
}

export interface ResolvedTarget {
  path: string;
  headers: Record<string, string>;
  missingRequired: string[];
}

export function resolveRequestTarget(
  ep: OpenApiEndpoint,
  values: Record<string, string>,
): ResolvedTarget {
  let path = ep.p;
  const headers: Record<string, string> = {};
  const query = new URLSearchParams();
  const missingRequired: string[] = [];
  for (const p of ep.params ?? []) {
    const raw = (values[paramKey(p)] ?? '').trim();
    if (!raw) {
      if (p.required) missingRequired.push(p.name);
      continue;
    }
    if (p.in === 'path') path = path.replaceAll(`{${p.name}}`, encodeURIComponent(raw));
    else if (p.in === 'query') query.append(p.name, raw);
    else if (p.in === 'header') headers[p.name] = raw;
  }
  const qs = query.toString();
  return { path: qs ? `${path}?${qs}` : path, headers, missingRequired };
}

export function responsesFor(
  ep: OpenApiEndpoint,
  secured: boolean,
): { code: string; desc: string }[] {
  if (ep.responses?.length) return ep.responses.map((row) => ({ code: row.code, desc: row.desc }));
  const rows = [{ code: '200', desc: 'Request completed successfully.' }];
  if (ep.m !== 'GET') rows.push({ code: '422', desc: 'The request body or parameters failed validation.' });
  if (secured) rows.push({ code: '401', desc: 'Authentication credentials are missing, invalid, or expired.' });
  return rows;
}

export function requestBodyFor(ep: OpenApiEndpoint): string {
  if (!endpointHasBody(ep)) return '';
  if (ep.bodyExample) return ep.bodyExample;
  if (QUERY_ENDPOINTS.has(ep.p)) {
    const dataEndpoint = ep.p.endsWith('/query/data');
    const supportsTagFilter = ep.p.startsWith('/twin/api/query') || dataEndpoint;
    return JSON.stringify({
      query: 'Ask a question about the indexed knowledge base',
      mode: 'hybrid', top_k: 60,
      ...(dataEndpoint ? { chunk_top_k: 20 } : {}),
      response_type: 'Multiple Paragraphs',
      ...(supportsTagFilter ? { tag_filter: { all: ['tag-name'], any: [] } } : {}),
    }, null, 2);
  }
  if (ep.p === '/documents/text') return JSON.stringify({ text: '', file_source: '', tags: ['twin'] }, null, 2);
  if (ep.p.includes('/graph/entity/edit')) return JSON.stringify({ entity_name: '', updated_data: {} }, null, 2);
  return JSON.stringify({}, null, 2);
}

export function curlFor(
  ep: OpenApiEndpoint,
  body: string,
  token: string,
  baseUrl: string,
  target?: Pick<ResolvedTarget, 'path' | 'headers'>,
): string {
  const path = target?.path ?? ep.p;
  const lines = [`curl -X ${ep.m} '${baseUrl}${path}' \\`];
  lines.push("  -H 'Accept: application/json' \\");
  const requestHasBody = endpointHasBody(ep);
  if (requestHasBody) lines.push("  -H 'Content-Type: application/json' \\");
  for (const [name, value] of Object.entries(target?.headers ?? {})) lines.push(`  -H '${name}: ${value}' \\`);
  if (token) lines.push(`  -H 'Authorization: Bearer ${token.slice(0, 6)}…' \\`);
  if (!requestHasBody) lines[lines.length - 1] = lines.at(-1)?.replace(/ \\$/, '') ?? '';
  else lines.push(`  -d '${(body || '').replaceAll(/\n\s*/g, ' ')}'`);
  return lines.join('\n');
}
