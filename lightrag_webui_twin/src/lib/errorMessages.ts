/**
 * User-facing error copy — the single mapping layer between transport
 * errors (ApiError, fetch failures) and what the operator reads.
 *
 * Doctrine (error-UX pass, 2026-07-03): no raw HTTP status code and no
 * synthetic `METHOD /path → 500` string may reach a headline. The
 * technical string stays available on `describeError(...).technical`
 * and via `logTechnicalError` (console) for support/debug, never as
 * the primary copy.
 *
 * Backend `detail` strings (FastAPI `{detail: "..."}`) in this product
 * are written as operator-readable sentences ("Folder 'x' already
 * exists", "Invalid username or password") — for 4xx they are promoted
 * to the message when present. 5xx details are NOT trusted as user
 * copy (stack fragments, driver errors) and stay technical-only.
 */

import { ApiError } from '../api/client';

export interface ErrorContext {
  /**
   * Present-participle action for the generic fallback, e.g.
   * "uploading the file" → "Something went wrong while uploading the
   * file. Please retry or contact Twincore Team."
   */
  action?: string;
}

export interface UserFacingError {
  /** Clear, non-technical headline. Always safe to show. */
  message: string;
  /** Technical second-level string (method/path/status) — console or
   *  discreet secondary line only, never the headline. */
  technical?: string;
}

const CONTACT = 'Please retry or contact Twincore Team.';

function genericMessage(action?: string): string {
  return action
    ? `Something went wrong while ${action}. ${CONTACT}`
    : `Something went wrong. ${CONTACT}`;
}

/** Extract the FastAPI-style `detail` (or `message`) string from an
 *  error body. Arrays/objects (Pydantic 422 payloads) are skipped. */
export function backendDetail(body: unknown): string | undefined {
  if (!body || typeof body !== 'object') return undefined;
  for (const key of ['detail', 'message'] as const) {
    const value = (body as Record<string, unknown>)[key];
    if (typeof value === 'string' && value.trim() && value.length <= 300) {
      return value.trim();
    }
  }
  return undefined;
}

/** fetch() rejections surface as TypeError with browser-specific text —
 *  "Failed to fetch" (Chromium), "Load failed" (WebKit),
 *  "NetworkError…" (Firefox). */
function isNetworkFailure(err: Error): boolean {
  return /failed to fetch|networkerror|load failed|fetch failed|network request failed/i.test(
    err.message,
  );
}

function forbiddenMessage(detail: string | undefined): string {
  // Scope details from server/folder.py ("Folder not in user scope",
  // "No folder available for this KB…") — but NOT the admin-gate detail
  // "Admin scope 'admin:folders' required", which is a permission issue.
  if (detail && !/admin scope/i.test(detail) && /folder/i.test(detail)) {
    return 'You do not have access to this folder. Contact Twincore Team if you need access.';
  }
  return 'You do not have permission to perform this action. Contact Twincore Team if you believe you should.';
}

function statusMessage(
  status: number,
  detail: string | undefined,
  ctx?: ErrorContext,
): string {
  if (status >= 500) {
    return `The Twin backend is temporarily unavailable. Please retry in a moment or contact Twincore Team.`;
  }
  switch (status) {
    case 401:
      return 'Your session has expired. Please sign in again.';
    case 403:
      return forbiddenMessage(detail);
    case 404:
      return detail ?? 'The requested item could not be found. It may have been removed.';
    case 409:
      return detail ?? 'This conflicts with an existing item.';
    case 413:
      return 'This file is too large.';
    case 429:
      return 'Too many requests. Please wait a moment and retry.';
    case 400:
    case 422:
      return detail ?? genericMessage(ctx?.action);
    default:
      return detail ?? genericMessage(ctx?.action);
  }
}

/** Map any thrown value to operator-facing copy + optional technical string. */
export function describeError(err: unknown, ctx?: ErrorContext): UserFacingError {
  if (err instanceof ApiError) {
    return {
      message: statusMessage(err.status, backendDetail(err.body), ctx),
      technical: err.message,
    };
  }
  if (err instanceof Error) {
    if (isNetworkFailure(err)) {
      return {
        message: 'Cannot reach the Twin backend. Check your connection and retry.',
        technical: err.message,
      };
    }
    return { message: genericMessage(ctx?.action), technical: err.message };
  }
  return { message: genericMessage(ctx?.action) };
}

/** Headline-only convenience for toasts, banners and inline errors. */
export function userErrorMessage(err: unknown, ctx?: ErrorContext): string {
  return describeError(err, ctx).message;
}

/** Login-scoped mapping — a 401 on POST /login is a failed credential
 *  check, not a session expiry. */
export function loginErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return 'Incorrect username or password.';
    if (err.status === 429) {
      return 'Too many sign-in attempts. Please wait a moment and retry.';
    }
    if (err.status >= 500) {
      return 'The Twin backend is temporarily unavailable. Please retry in a moment or contact Twincore Team.';
    }
    return `Sign-in failed. ${CONTACT}`;
  }
  if (err instanceof Error && isNetworkFailure(err)) {
    return 'Cannot reach the Twin backend. Check your connection and retry.';
  }
  return `Sign-in failed. ${CONTACT}`;
}

function fileFormatLabel(fileName: string | undefined): string | undefined {
  if (!fileName) return undefined;
  const idx = fileName.lastIndexOf('.');
  const ext = idx >= 0 ? fileName.slice(idx + 1).trim() : '';
  return ext ? ext.toUpperCase() : undefined;
}

/** Client-side (pre-upload) copy for a file whose type is not accepted.
 *  Kept here so the AddSourceModal validation and the backend-rejection
 *  mapping below stay one single wording. */
export function unsupportedFileMessage(fileName?: string): string {
  const format = fileFormatLabel(fileName);
  return format
    ? `${format} format is not supported`
    : 'This file format is not supported';
}

/** Upload-scoped mapping. `fileName` lets the copy name the rejected
 *  format ("ZIP format is not supported") instead of echoing LightRAG's
 *  "Unsupported file type. Supported types: (...)" detail. */
export function uploadFailureMessage(err: unknown, fileName?: string): string {
  if (err instanceof ApiError) {
    const detail = backendDetail(err.body);
    if (err.status === 400 && detail && /unsupported file type/i.test(detail)) {
      return unsupportedFileMessage(fileName);
    }
    if (err.status === 413) {
      return fileName ? `${fileName} is too large.` : 'This file is too large.';
    }
    if (err.status === 409) {
      return detail ?? 'A document with this name already exists.';
    }
    return statusMessage(err.status, detail, { action: 'uploading the file' });
  }
  return describeError(err, { action: 'uploading the file' }).message;
}

/**
 * Console-log the technical detail of a caught error once, at catch
 * time (never during render — render-time surfaces stay pure).
 */
export function logTechnicalError(scope: string, err: unknown): void {
  const { technical } = describeError(err);
  console.warn(`[twin] ${scope}:`, technical ?? err);
}
