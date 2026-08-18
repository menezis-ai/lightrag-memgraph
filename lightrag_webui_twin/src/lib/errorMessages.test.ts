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
  ingestionFailureMessage,
  loginErrorMessage,
  logTechnicalError,
  unsupportedFileMessage,
  uploadFailureMessage,
  userErrorMessage,
} from './errorMessages';

describe('ingestionFailureMessage', () => {
  it('explains the legacy OCR pre-filter reason with measured and configured values', () => {
    expect(
      ingestionFailureMessage(
        'vision-prefilter: OCR text below 20 chars (9) — set TWIN_VISION_MIN_OCR_CHARS=0 to caption everything',
      ),
    ).toBe(
      'Image rejected by the OCR pre-filter: 9 text characters detected; the configured minimum is 20. Images with too little readable text are excluded before Vision analysis.',
    );
  });

  it('explains the current OCR and size reasons only from their stable codes', () => {
    expect(
      ingestionFailureMessage(
        'vision-prefilter: image rejected before vision analysis; OCR detected 7 text characters, below configured minimum 20',
      ),
    ).toBe(
      'Image rejected by the OCR pre-filter: 7 text characters detected; the configured minimum is 20. Images with too little readable text are excluded before Vision analysis.',
    );
    expect(
      ingestionFailureMessage(
        'vision-size-limit: image rejected because file size is 200 bytes; configured maximum is 100 bytes',
      ),
    ).toBe(
      'Image rejected: file size is 200 bytes; the configured maximum is 100 bytes.',
    );
  });

  it('explains both legacy and current excluded-class reasons', () => {
    const expected =
      'Image rejected by the Vision filter: classified as “logo”, which is excluded by the active Vision settings.';
    expect(
      ingestionFailureMessage("image-dropped: classification 'Logo'"),
    ).toBe(expected);
    expect(
      ingestionFailureMessage(
        "image-dropped: image rejected by active Vision policy; classified as 'logo', an excluded class",
      ),
    ).toBe(expected);
  });

  it('explains an all-visuals-rejected PDF with page and classification', () => {
    expect(
      ingestionFailureMessage(
        "pdf-vision-dropped: all visual candidates excluded by policy; first rejected page(s) 2,5: pdf-image-dropped: classified as 'logo', an excluded class",
      ),
    ).toBe(
      'PDF rejected: it contains no usable text and all detected visual content was excluded. The first rejected visual is on page(s) 2, 5, classified as “logo”.',
    );
  });

  it('preserves unknown ingestion failures verbatim', () => {
    expect(ingestionFailureMessage('LLM extractor: invalid JSON')).toBe(
      'LLM extractor: invalid JSON',
    );
  });

  it('does not rewrite lure text without a stable pipeline reason prefix', () => {
    const lures = [
      "Unknown policy error: classified as 'confidential' but not rejected by Vision",
      'Parser copied vision-prefilter: OCR detected 2 text characters, below configured minimum 20',
      'Proxy note: file size is 99 bytes; configured maximum is 10 bytes',
      "Wrapper saw pdf-vision-dropped: first rejected page(s) 4: classified as 'logo'",
    ];
    for (const lure of lures) {
      expect(ingestionFailureMessage(lure)).toBe(lure);
    }
  });

  it('handles long adversarial ingestion errors without ambiguous regex backtracking', () => {
    const hostile = `pdf-vision-dropped:${' first rejected page(s) 1'.repeat(20_000)}`;
    expect(ingestionFailureMessage(hostile)).toBe(hostile);
  });
});

function apiError(status: number, body: unknown = null, path = '/twin/api/x'): ApiError {
  return new ApiError(`POST ${path} → ${status} Error`, status, body);
}

describe('describeError / userErrorMessage', () => {
  it('never leaks the arrow-style technical string into the message', () => {
    for (const status of [400, 401, 403, 404, 409, 413, 422, 423, 429, 500, 502, 503]) {
      const out = describeError(apiError(status));
      expect(out.message).not.toContain('→');
      expect(out.message).not.toContain(String(status));
      expect(out.technical).toContain(`${status}`);
    }
  });

  it('maps 5xx to the backend-unavailable copy and ignores 5xx details', () => {
    const out = describeError(apiError(503, { detail: 'bolt driver ServiceUnavailable at pool.py:88' }));
    expect(out.message).toBe(
      'The backend is temporarily unavailable. Retry in a moment. If the problem continues, contact your platform administrator.',
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
      'You do not have access to this folder. Ask your platform administrator to grant access.',
    );
  });

  it('maps a non-folder 403 to the generic permission copy', () => {
    expect(
      userErrorMessage(apiError(403, { detail: "Admin scope 'admin:folders' required" })),
    ).toBe(
      'You do not have permission to perform this action. Ask your platform administrator if you believe you should have access.',
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

  it('maps pipeline-busy conflicts to explicit action-not-taken copy', () => {
    const msg = userErrorMessage(
      apiError(409, { detail: 'Pipeline is busy. Please try again later' }),
      { action: 'deleting a.pdf' },
    );
    expect(msg).toBe(
      'Action not taken while deleting a.pdf: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.',
    );
  });

  it('maps lock-style pipeline refusals the same way', () => {
    const msg = userErrorMessage(
      apiError(423, {
        detail: 'Document scan is classifying files. Wait for the classification phase to finish before submitting new work.',
      }),
      { action: 'creating the relation' },
    );
    expect(msg).toBe(
      'Action not taken while creating the relation: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.',
    );
  });

  it('promotes a structured recovery-required 503 instead of advising a retry', () => {
    const detail =
      'Operator recovery is required before deletion can resume; 1 earlier document change was already committed.';
    const msg = userErrorMessage(
      apiError(503, { detail, recovery_required: true }),
      { action: 'deleting the selected sources' },
    );
    expect(msg).toBe(detail);
    expect(msg).not.toContain('temporarily unavailable');
  });

  it('skips Pydantic array details and falls back to the action copy', () => {
    const out = userErrorMessage(
      apiError(422, { detail: [{ loc: ['body', 'name'], msg: 'field required' }] }),
      { action: 'creating the tag' },
    );
    expect(out).toBe(
      'Something went wrong while creating the tag. Please retry. If the problem continues, contact your platform administrator.',
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
      'Something went wrong while saving the graph. Please retry. If the problem continues, contact your platform administrator.',
    );
    expect(userErrorMessage('weird string throw')).toBe(
      'Something went wrong. Please retry. If the problem continues, contact your platform administrator.',
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
      'Sign-in failed. Please retry. If the problem continues, contact your platform administrator.',
    );
    expect(loginErrorMessage('boom')).toBe(
      'Sign-in failed. Please retry. If the problem continues, contact your platform administrator.',
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

  it('maps upload 409 pipeline-busy to explicit action-not-taken copy', () => {
    expect(
      uploadFailureMessage(
        apiError(409, { detail: 'Pipeline is busy. Please try again later' }),
        'a.pdf',
      ),
    ).toBe(
      'Action not taken while uploading the file: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.',
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
