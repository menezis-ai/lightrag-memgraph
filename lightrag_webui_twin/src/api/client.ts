/**
 * Thin typed fetch wrapper for the Twin RAG backend.
 *
 * Runtime contract:
 *   window.__twinConfig.apiBaseUrl      — Twin overlay base, e.g. /twin/api.
 *   window.__twinConfig.lightragBaseUrl — LightRAG native base, usually empty.
 *   window.__twinConfig.defaultFolderId — SRE-provisioned default Twin folder.
 *   VITE_API_BASE_URL                   — optional dev/test origin fallback.
 *   VITE_AUTH_TOKEN                     — optional dev/test bearer fallback.
 *
 * Errors throw `ApiError` with the HTTP status and parsed body (or the raw text
 * if the response was non-JSON, e.g. nginx 502 HTML — same failure mode that
 * burned the production front in v0.5.2's HTTP-e2e suite).
 */

import { resolveRuntimeConfig } from '../config/devConfig';

const TWIN_PREFIX = '/twin/api';
const ENV_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const ENV_AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN ?? '';
const SESSION_AUTH_TOKEN_KEY = 'twin-rag.authToken';

let activeFolder: string | null = null;
let sessionAuthToken: string | null = null;

export function setActiveFolder(folder: string | null): void {
  activeFolder = folder;
}

export function getActiveFolder(): string | null {
  return activeFolder;
}

function readStoredAuthToken(): string {
  if (typeof window === 'undefined') return '';
  try {
    return window.sessionStorage.getItem(SESSION_AUTH_TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

export function setSessionAuthToken(token: string | null): void {
  sessionAuthToken = token;
  if (typeof window === 'undefined') return;
  try {
    if (token) {
      window.sessionStorage.setItem(SESSION_AUTH_TOKEN_KEY, token);
    } else {
      window.sessionStorage.removeItem(SESSION_AUTH_TOKEN_KEY);
    }
  } catch {
    // Session storage can be unavailable in privacy-restricted contexts.
  }
}

export function getSessionAuthToken(): string | null {
  return sessionAuthToken ?? (readStoredAuthToken() || null);
}

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

export interface UnauthorizedEvent {
  path: string;
  status: number;
}

type UnauthorizedHandler = (event: UnauthorizedEvent) => void;

const unauthorizedHandlers = new Set<UnauthorizedHandler>();

/**
 * Subscribe to backend 401s observed mid-session (expired or revoked JWT).
 * `useAuth` registers a handler that drops the operator back to the login
 * screen instead of leaving a stale "authenticated" shell rendering broken
 * per-component errors. Returns an unsubscribe function.
 *
 * The `/login` handshake is exempt: a wrong-password 401 is a failed login
 * attempt, not a session expiry, and is handled by `useAuth.login` directly.
 */
export function onUnauthorized(handler: UnauthorizedHandler): () => void {
  unauthorizedHandlers.add(handler);
  return () => {
    unauthorizedHandlers.delete(handler);
  };
}

function notifyUnauthorized(event: UnauthorizedEvent): void {
  for (const handler of unauthorizedHandlers) {
    try {
      handler(event);
    } catch {
      // A subscriber must not break the request's own error-propagation path.
    }
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
  /** Optional override for X-Twin-Folder. Null disables the header. */
  folder?: string | null;
  signal?: AbortSignal;
}

export function getTwinRuntimeConfig() {
  const raw =
    typeof window !== 'undefined' ? window.__twinConfig : undefined;
  return resolveRuntimeConfig(raw, Boolean(import.meta.env.DEV));
}

function isAbsoluteUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function trimSlashes(value: string): string {
  return value.replace(/^\/+|\/+$/g, '');
}

function joinUrl(base: string, path: string): string {
  if (!base) return path || '';
  if (!path) return base;
  if (isAbsoluteUrl(path)) return path;
  return `${base.replace(/\/$/, '')}/${trimSlashes(path)}`;
}

function runtimeBase(kind: 'twin' | 'lightrag'): string {
  const cfg = getTwinRuntimeConfig();
  const configured =
    kind === 'twin' ? cfg.apiBaseUrl : cfg.lightragBaseUrl;
  if (!ENV_BASE_URL) return configured.replace(/\/$/, '');
  if (!configured) return ENV_BASE_URL;
  if (isAbsoluteUrl(configured)) return configured.replace(/\/$/, '');
  return joinUrl(ENV_BASE_URL, configured);
}

export function buildApiUrl(
  path: string,
  query?: ApiRequestInit['query'],
): string {
  let url: string;
  if (isAbsoluteUrl(path)) {
    url = path;
  } else if (path === TWIN_PREFIX || path.startsWith(`${TWIN_PREFIX}/`)) {
    const suffix = path.slice(TWIN_PREFIX.length);
    url = joinUrl(runtimeBase('twin'), suffix);
  } else {
    url = joinUrl(runtimeBase('lightrag'), path);
  }
  if (!query) return url;
  const usp = new URLSearchParams();
  Object.entries(query).forEach(([k, v]) => {
    if (v === null || v === undefined) return;
    usp.set(k, String(v));
  });
  const qs = usp.toString();
  return qs ? `${url}?${qs}` : url;
}

export function buildApiHeaders(
  init: Pick<ApiRequestInit, 'token' | 'folder'> = {},
  options: { json?: boolean } = {},
): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: 'application/json',
  };
  if (options.json) {
    headers['Content-Type'] = 'application/json';
  }
  const cfg = getTwinRuntimeConfig();
  const usesOuterBasicAuth = cfg.debugUser?.idp === 'local-debug';
  if (usesOuterBasicAuth) {
    setSessionAuthToken(null);
  }
  const token = usesOuterBasicAuth
    ? ''
    : init.token ?? getSessionAuthToken() ?? ENV_AUTH_TOKEN;
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const folder = init.folder === undefined ? activeFolder : init.folder;
  if (folder) {
    headers['X-Twin-Folder'] = folder;
  }
  return headers;
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
  const method = init.method ?? 'GET';
  const headers = buildApiHeaders(init, {
    json: method !== 'GET' && method !== 'HEAD',
  });

  const res = await fetch(buildApiUrl(path, init.query), {
    method,
    headers,
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
    signal: init.signal,
    credentials: 'include',
  });

  if (!res.ok) {
    const body = await parseBody(res);
    if (res.status === 401 && !path.endsWith('/login')) {
      notifyUnauthorized({ path, status: res.status });
    }
    throw new ApiError(
      `${method} ${path} → ${res.status} ${res.statusText}`,
      res.status,
      body,
    );
  }

  return (await parseBody(res)) as T;
}
