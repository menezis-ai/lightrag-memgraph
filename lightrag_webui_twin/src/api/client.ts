/**
 * Thin typed fetch wrapper for the Twin RAG backend.
 *
 * Env contract:
 *   VITE_API_BASE_URL — backend phase-1 origin (e.g. https://cib-kb.twin.internal).
 *                       Empty string = same-origin (default, plays well with MSW).
 *   VITE_AUTH_TOKEN   — optional bearer token, attached as Authorization header.
 *                       In prod this comes from the Twin gateway (Keycloak OIDC).
 *
 * Errors throw `ApiError` with the HTTP status and parsed body (or the raw text
 * if the response was non-JSON, e.g. nginx 502 HTML — same failure mode that
 * burned the BNP front in v0.5.2's HTTP-e2e suite).
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN ?? '';

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

export interface ApiRequestInit {
  method?: string;
  /** Query string params, serialized via URLSearchParams. Null/undefined skipped. */
  query?: Record<string, string | number | boolean | null | undefined>;
  /** JSON body — stringified for POST/PUT/PATCH. */
  body?: unknown;
  /** Optional override for the global bearer token. */
  token?: string;
  signal?: AbortSignal;
}

function buildUrl(path: string, query?: ApiRequestInit['query']): string {
  const url = BASE_URL + path;
  if (!query) return url;
  const usp = new URLSearchParams();
  Object.entries(query).forEach(([k, v]) => {
    if (v === null || v === undefined) return;
    usp.set(k, String(v));
  });
  const qs = usp.toString();
  return qs ? `${url}?${qs}` : url;
}

async function parseBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function apiFetch<T>(path: string, init: ApiRequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    Accept: 'application/json',
  };
  const method = init.method ?? 'GET';
  if (method !== 'GET' && method !== 'HEAD') {
    headers['Content-Type'] = 'application/json';
  }
  const token = init.token ?? AUTH_TOKEN;
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(buildUrl(path, init.query), {
    method,
    headers,
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
    signal: init.signal,
  });

  if (!res.ok) {
    const body = await parseBody(res);
    throw new ApiError(
      `${method} ${path} → ${res.status} ${res.statusText}`,
      res.status,
      body,
    );
  }

  return (await parseBody(res)) as T;
}
