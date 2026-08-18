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
   * file. Please retry. If the problem continues, contact your platform
   * administrator."
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

const CONTACT =
  'Please retry. If the problem continues, contact your platform administrator.';

function genericMessage(action?: string): string {
  return action
    ? `Something went wrong while ${action}. ${CONTACT}`
    : `Something went wrong. ${CONTACT}`;
}

function recoveryRequiredMessage(body: unknown): string | undefined {
  if (!body || typeof body !== 'object') return undefined;
  const payload = body as Record<string, unknown>;
  if (payload.recovery_required !== true) return undefined;
  const detail = payload.detail;
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  return 'This workspace requires operator recovery before deletion can resume. Complete the documented recovery procedure, then retry.';
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
  // "No folder is provisioned…") — but NOT the admin-gate detail
  // "Admin scope 'admin:folders' required", which is a permission issue.
  if (detail && !/admin scope/i.test(detail) && /folder/i.test(detail)) {
    return 'You do not have access to this folder. Ask your platform administrator to grant access.';
  }
  return 'You do not have permission to perform this action. Ask your platform administrator if you believe you should have access.';
}

export function isPipelineBusyDetail(detail: string | undefined): boolean {
  return Boolean(
    detail &&
      /pipeline|ingestion|document scan|document processing|scanning|processing loop|destructive job/i.test(detail) &&
      /busy|classifying|clearing|deleting|in flight|running|wait/i.test(detail),
  );
}

function pipelineBusyMessage(action?: string): string {
  const suffix = action ? ` while ${action}` : '';
  return `Action not taken${suffix}: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.`;
}

function statusMessage(
  status: number,
  detail: string | undefined,
  ctx?: ErrorContext,
): string {
  if (status >= 500) {
    return 'The backend is temporarily unavailable. Retry in a moment. If the problem continues, contact your platform administrator.';
  }
  switch (status) {
    case 401:
      return 'Your session has expired. Please sign in again.';
    case 403:
      return forbiddenMessage(detail);
    case 404:
      return detail ?? 'The requested item could not be found. It may have been removed.';
    case 409:
      if (isPipelineBusyDetail(detail)) return pipelineBusyMessage(ctx?.action);
      return detail ?? 'This conflicts with an existing item.';
    case 423:
      return isPipelineBusyDetail(detail)
        ? pipelineBusyMessage(ctx?.action)
        : detail ?? 'This item is currently locked. Please retry in a moment.';
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
    const recoveryMessage = recoveryRequiredMessage(err.body);
    return {
      message:
        recoveryMessage ?? statusMessage(err.status, backendDetail(err.body), ctx),
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
      return 'The backend is temporarily unavailable. Retry in a moment. If the problem continues, contact your platform administrator.';
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
      if (isPipelineBusyDetail(detail)) {
        return statusMessage(err.status, detail, { action: 'uploading the file' });
      }
      return detail ?? 'A document with this name already exists.';
    }
    return statusMessage(err.status, detail, { action: 'uploading the file' });
  }
  return describeError(err, { action: 'uploading the file' }).message;
}

/** Translate stable image-pipeline reason codes into operator-facing copy.
 *
 * Existing FAILED rows keep their original `error_msg`, so this mapper accepts
 * both the legacy terse strings and the newer explanatory backend wording.
 * Unknown errors remain untouched: this layer must not hide useful failure
 * detail from operators.
 */
function pdfVisualRejection(
  value: string,
): { pages: string; classification: string } | undefined {
  const lower = value.toLowerCase();
  if (!lower.startsWith('pdf-vision-dropped:')) return undefined;
  const pageMarker = 'first rejected page(s)';
  const pageStart = lower.indexOf(pageMarker);
  if (pageStart < 0) return undefined;
  const pageTail = value.slice(pageStart + pageMarker.length).trimStart();
  const separator = pageTail.indexOf(':');
  if (separator < 0) return undefined;
  const pages = pageTail.slice(0, separator).trim();
  if (!/^[\d,\s]+$/.test(pages)) return undefined;

  const detail = pageTail.slice(separator + 1);
  const classificationMarker = 'classified as';
  const classificationStart = detail
    .toLowerCase()
    .indexOf(classificationMarker);
  if (classificationStart < 0) return undefined;
  const quoted = detail
    .slice(classificationStart + classificationMarker.length)
    .trimStart();
  const quote = quoted[0];
  if (quote !== "'" && quote !== '"') return undefined;
  const quoteEnd = quoted.indexOf(quote, 1);
  if (quoteEnd < 1) return undefined;
  return { pages, classification: quoted.slice(1, quoteEnd) };
}

function decimalAfter(value: string, marker: string): string | undefined {
  const start = value.toLowerCase().indexOf(marker);
  if (start < 0) return undefined;
  return /^\d+/.exec(value.slice(start + marker.length).trimStart())?.[0];
}

function visionSizeValues(
  value: string,
): { actual: string; maximum: string } | undefined {
  const prefix = 'vision-size-limit:';
  if (!value.toLowerCase().startsWith(prefix)) return undefined;

  const currentActual = decimalAfter(value, 'file size is');
  const currentMaximum = decimalAfter(value, 'configured maximum is');
  if (currentActual && currentMaximum) {
    return { actual: currentActual, maximum: currentMaximum };
  }

  const legacyTail = value.slice(prefix.length).trimStart();
  const legacyActual = /^\d+/.exec(legacyTail)?.[0];
  const equals = legacyTail.lastIndexOf('=');
  const legacyMaximum =
    equals < 0 ? undefined : /^\d+/.exec(legacyTail.slice(equals + 1))?.[0];
  return legacyActual && legacyMaximum
    ? { actual: legacyActual, maximum: legacyMaximum }
    : undefined;
}

export function ingestionFailureMessage(error: string): string {
  const value = error.trim();

  const legacyOcr = /^vision-prefilter:\s*OCR text below\s+(\d+)\s+chars\s+\((\d+)\)/i.exec(
    value,
  );
  const currentOcr = /^vision-prefilter:.*OCR detected\s+(\d+)\s+text characters,\s+below configured minimum\s+(\d+)/i.exec(
    value,
  );
  if (legacyOcr || currentOcr) {
    const detected = legacyOcr?.[2] ?? currentOcr?.[1];
    const minimum = legacyOcr?.[1] ?? currentOcr?.[2];
    return `Image rejected by the OCR pre-filter: ${detected} text characters detected; the configured minimum is ${minimum}. Images with too little readable text are excluded before Vision analysis.`;
  }

  const rejectedPdfVisual = pdfVisualRejection(value);
  if (rejectedPdfVisual) {
    const pages = rejectedPdfVisual.pages
      .split(',')
      .map((page) => page.trim())
      .join(', ');
    const classification = rejectedPdfVisual.classification.toLowerCase();
    return `PDF rejected: it contains no usable text and all detected visual content was excluded. The first rejected visual is on page(s) ${pages}, classified as “${classification}”.`;
  }

  const legacyClassification = /^image-dropped:\s*classification\s+['"]([^'"]+)['"]/i.exec(
    value,
  );
  const currentClassification = /^image-dropped:.*classified as\s+['"]([^'"]+)['"]/i.exec(value);
  const classification = legacyClassification?.[1] ?? currentClassification?.[1];
  if (classification) {
    return `Image rejected by the Vision filter: classified as “${classification.toLowerCase()}”, which is excluded by the active Vision settings.`;
  }

  if (/^image-dropped:.*no informational content/i.test(value)) {
    return 'Image rejected by the Vision filter: no informational content was detected.';
  }

  const size = visionSizeValues(value);
  if (size) {
    return `Image rejected: file size is ${size.actual} bytes; the configured maximum is ${size.maximum} bytes.`;
  }

  return value;
}

/**
 * Console-log the technical detail of a caught error once, at catch
 * time (never during render — render-time surfaces stay pure).
 */
export function logTechnicalError(scope: string, err: unknown): void {
  const { technical } = describeError(err);
  console.warn(`[twin] ${scope}:`, technical ?? err);
}
