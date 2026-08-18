/**
 * Policy-rejection detection contract. The string anchors mirror the
 * Twin backend (`patches/registry.py::_report_error_document`,
 * `_classification_hook.py::_failed_status_for_rejection`,
 * `_vision.py` reason prefixes) — if a backend rewording breaks these
 * tests, update both sides in the same commit.
 */
import { describe, expect, it } from 'vitest';
import {
  isPolicyRejected,
  policyRejectionGuidance,
  policyRejectionKind,
} from './policyRejection';

function doc(overrides: {
  status?: string;
  content_summary?: string;
  error_msg?: string | null;
}) {
  return {
    status: (overrides.status ?? 'FAILED') as never,
    content_summary: overrides.content_summary ?? '',
    error_msg: overrides.error_msg ?? null,
  };
}

describe('policyRejectionKind', () => {
  it('detects the OCR pre-filter verdict (the logo-starbucks case)', () => {
    expect(
      policyRejectionKind(
        doc({
          content_summary: 'Image ingestion refused',
          error_msg:
            'vision-prefilter: image rejected before vision analysis; ' +
            'OCR detected 0 text characters, below configured minimum 20',
        }),
      ),
    ).toBe('vision');
  });

  it('detects the size-limit and drop-class verdicts', () => {
    expect(
      policyRejectionKind(
        doc({
          content_summary: 'Image ingestion refused',
          error_msg: 'vision-size-limit: image rejected because file size is…',
        }),
      ),
    ).toBe('vision');
    expect(
      policyRejectionKind(
        doc({
          content_summary: 'Image ingestion refused',
          error_msg:
            "image-dropped: image rejected by active Vision policy; classified as 'logo', an excluded class",
        }),
      ),
    ).toBe('vision');
  });

  it('detects the MIP classification rejection', () => {
    expect(
      policyRejectionKind(
        doc({
          content_summary:
            '[content withheld: classification C3 exceeds ceiling C2]',
          error_msg: 'classification C3 exceeds configured ceiling C2',
        }),
      ),
    ).toBe('classification');
  });

  it('keeps transient vision failures as plain failures (retryable)', () => {
    for (const reason of [
      'vision-timeout: no result within 60s',
      'vision-error: RuntimeError: boom',
      'vision-llm-error: APIConnectionError: down',
      'vision-input-error: OSError: unreadable',
    ]) {
      expect(
        policyRejectionKind(
          doc({ content_summary: 'Image ingestion refused', error_msg: reason }),
        ),
      ).toBeNull();
    }
  });

  it('never fires on non-FAILED docs or ordinary failures', () => {
    expect(
      policyRejectionKind(
        doc({ status: 'PROCESSED', content_summary: 'Image ingestion refused' }),
      ),
    ).toBeNull();
    expect(
      policyRejectionKind(
        doc({ content_summary: 'Quarterly report', error_msg: 'parse error' }),
      ),
    ).toBeNull();
    expect(policyRejectionKind(doc({ content_summary: '', error_msg: null }))).toBeNull();
  });
});

describe('guidance copy', () => {
  it('tells the operator a retry is pointless, for both kinds', () => {
    expect(policyRejectionGuidance('vision')).toMatch(/will not change the verdict/);
    expect(policyRejectionGuidance('classification')).toMatch(
      /will not change the verdict/,
    );
  });

  it('isPolicyRejected mirrors kind detection', () => {
    expect(
      isPolicyRejected(
        doc({
          content_summary: 'Image ingestion refused',
          error_msg: 'vision-prefilter: no text',
        }),
      ),
    ).toBe(true);
    expect(isPolicyRejected(doc({ content_summary: 'ok' }))).toBe(false);
  });
});
