/**
 * Unit tests for the user-facing error mapping layer.
 *
 * Contract under test: no raw HTTP status and no `METHOD /path → 500`
 * arrow string ever lands in `.message`; the technical string is
 * preserved on `.technical` only.
 */

import { describe, expect, it, vi } from 'vitest';

import { ApiError } from '../api/client';
import {
  backendDetail,
  describeError,
  loginErrorMessage,
  logTechnicalError,
  unsupportedFileMessage,
  uploadFailureMessage,
  userErrorMessage,
} from './errorMessages';

function apiError(status: number, body: unknown = null, path = '/twin/api/x'): ApiError {
  return new ApiError(`POST ${path} → ${status} Error`, status, body);
}

describe('describeError / userErrorMessage', () => {
  it('never leaks the arrow-style technical string into the message', () => {
    for (const status of [400, 401, 403, 404, 409, 413, 422, 429, 500, 502, 503]) {
      const out = describeError(apiError(status));
      expect(out.message).not.toContain('→');
      expect(out.message).not.toContain(String(status));
      expect(out.technical).toContain(`${status}`);
    }
  });

  it('maps 5xx to the backend-unavailable copy and ignores 5xx details', () => {
    const out = describeError(apiError(503, { detail: 'bolt driver ServiceUnavailable at pool.py:88' }));
    expect(out.message).toBe(
      'The Twin backend is temporarily unavailable. Please retry in a moment or contact Twincore Team.',
    );
  });

  it('maps a mid-session 401 to the session-expired copy', () => {
    expect(userErrorMessage(apiError(401))).toBe(
      'Your session has expired. Please sign in again.',
    );
  });

  it('maps a folder 403 to the folder-access copy', () => {
    expect(
      userErrorMessage(apiError(403, { detail: "Folder 'secret' is not in your scope" })),
    ).toBe(
      'You do not have access to this folder. Contact Twincore Team if you need access.',
    );
  });

  it('maps a non-folder 403 to the generic permission copy', () => {
    expect(
      userErrorMessage(apiError(403, { detail: "Admin scope 'admin:folders' required" })),
    ).toBe(
      'You do not have permission to perform this action. Contact Twincore Team if you believe you should.',
    );
  });

  it('promotes human backend detail on 409/404/422', () => {
    expect(userErrorMessage(apiError(409, { detail: "Folder 'ops' already exists" }))).toBe(
      "Folder 'ops' already exists",
    );
    expect(userErrorMessage(apiError(404, { detail: "Folder 'x' not found" }))).toBe(
      "Folder 'x' not found",
    );
    expect(userErrorMessage(apiError(422, { detail: "Invalid folder id 'a b'" }))).toBe(
      "Invalid folder id 'a b'",
    );
  });

  it('skips Pydantic array details and falls back to the action copy', () => {
    const out = userErrorMessage(
      apiError(422, { detail: [{ loc: ['body', 'name'], msg: 'field required' }] }),
      { action: 'creating the tag' },
    );
    expect(out).toBe(
      'Something went wrong while creating the tag. Please retry or contact Twincore Team.',
    );
  });

  it('maps fetch network failures to the connectivity copy', () => {
    expect(userErrorMessage(new TypeError('Failed to fetch'))).toBe(
      'Cannot reach the Twin backend. Check your connection and retry.',
    );
    expect(userErrorMessage(new TypeError('Load failed'))).toBe(
      'Cannot reach the Twin backend. Check your connection and retry.',
    );
  });

  it('maps unknown errors and non-Errors to the generic fallback', () => {
    expect(userErrorMessage(new Error('kaput'), { action: 'saving the graph' })).toBe(
      'Something went wrong while saving the graph. Please retry or contact Twincore Team.',
    );
    expect(userErrorMessage('weird string throw')).toBe(
      'Something went wrong. Please retry or contact Twincore Team.',
    );
  });
});

describe('loginErrorMessage', () => {
  it('maps a 401 to incorrect-credentials copy (not session expiry, no status code)', () => {
    const msg = loginErrorMessage(apiError(401, { detail: 'Invalid username or password' }, '/login'));
    expect(msg).toBe('Incorrect username or password.');
    expect(msg).not.toContain('401');
  });

  it('maps 5xx and network failures distinctly', () => {
    expect(loginErrorMessage(apiError(503))).toContain('temporarily unavailable');
    expect(loginErrorMessage(new TypeError('Failed to fetch'))).toContain(
      'Cannot reach the Twin backend',
    );
  });

  it('falls back to a clean sign-in failure for anything else', () => {
    expect(loginErrorMessage(apiError(418))).toBe(
      'Sign-in failed. Please retry or contact Twincore Team.',
    );
    expect(loginErrorMessage('boom')).toBe(
      'Sign-in failed. Please retry or contact Twincore Team.',
    );
  });
});

describe('uploadFailureMessage / unsupportedFileMessage', () => {
  it('names the rejected format on a backend unsupported-type 400', () => {
    const err = apiError(400, {
      detail: "Unsupported file type. Supported types: ('.pdf', '.docx')",
    });
    expect(uploadFailureMessage(err, 'archive.zip')).toBe('ZIP format is not supported');
    expect(uploadFailureMessage(err)).toBe('This file format is not supported');
  });

  it('client-side copy matches the backend-mapped copy', () => {
    expect(unsupportedFileMessage('archive.zip')).toBe('ZIP format is not supported');
    expect(unsupportedFileMessage('noextension')).toBe('This file format is not supported');
  });

  it('maps 413 with the file name', () => {
    expect(uploadFailureMessage(apiError(413), 'big.pdf')).toBe('big.pdf is too large.');
  });

  it('maps 409 to the duplicate copy (detail preferred)', () => {
    expect(uploadFailureMessage(apiError(409))).toBe(
      'A document with this name already exists.',
    );
  });

  it('maps 5xx and network failures through the generic layer', () => {
    expect(uploadFailureMessage(apiError(500), 'a.pdf')).toContain('temporarily unavailable');
    expect(uploadFailureMessage(new TypeError('Failed to fetch'), 'a.pdf')).toContain(
      'Cannot reach the Twin backend',
    );
  });
});

describe('backendDetail', () => {
  it('extracts detail then message, skipping non-strings and oversized text', () => {
    expect(backendDetail({ detail: 'clean' })).toBe('clean');
    expect(backendDetail({ message: 'fallback' })).toBe('fallback');
    expect(backendDetail({ detail: ['array'] })).toBeUndefined();
    expect(backendDetail({ detail: 'x'.repeat(301) })).toBeUndefined();
    expect(backendDetail('raw html body')).toBeUndefined();
    expect(backendDetail(null)).toBeUndefined();
  });
});

describe('logTechnicalError', () => {
  it('logs the technical string to console.warn without throwing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    logTechnicalError('login', apiError(401));
    expect(warn).toHaveBeenCalledWith('[twin] login:', 'POST /twin/api/x → 401 Error');
    warn.mockRestore();
  });
});
