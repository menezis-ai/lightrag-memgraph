/**
 * Typed document fixtures aligned on LightRAG DocStatus + Twin overlay.
 *
 * Mirrors the design-prototype `data.js` shape exactly so the React port
 * renders the same demo flow (Couche 1 of feat/webui-port-from-prototype):
 *
 *   - d2  : `review.state: 'modified'` — Confluence revalidation card variant
 *   - d5  : `metadata.classification: 'restricted'` — chunks tab truncates
 *   - d6  : `review.state: 'pending-review'` + `extracted_text` for Read source
 *   - d7  : `review.state: 'pending-review'` + `extracted_text` for Read source
 *
 * The string-based `metadata.classification` is the legacy baseline shape.
 * Couche 2 (tenant classification) will replace it with the structured
 * `ClassificationResult` payload (see `types/classification.ts`).
 */

import type { Document } from '../types/document';

const TWO_HOURS_AGO = '2026-05-29T14:00:00Z';
const ONE_DAY_AGO = '2026-05-28T16:00:00Z';
const THIRTY_MIN_AGO = '2026-05-29T15:30:00Z';
const TWENTY_FIVE_MIN_AGO = '2026-05-29T15:35:00Z';
const THREE_DAYS_AGO = '2026-05-26T16:00:00Z';

const D2_EXTRACTED = `=== Oracle PGA tuning — demo runbook (knowledge base) ===
Folder : DEMO-RUNBOOKS · page id : 84213 · last edit : demo.reviewer

[ Re-validation requested — 2 new sections added upstream ]

1. PGA sizing on RHEL 9 large pages   (NEW)
------------------------------------------
With HugePages enabled, set pga_aggregate_target to 12G and
disable Transparent HugePages (THP) per RHEL 9 hardening.

2. AWR snapshot interpretation for memory pressure   (NEW)
----------------------------------------------------------
Read the "PGA Memory Advisory" section of the AWR report;
target cache-hit > 95% before raising the aggregate target.

(Existing sections unchanged — see prior revision.)`;

const D6_EXTRACTED = `=== CFT Vendor API specification (draft v0.7) ===
Vendor : Acme Payments Iberia SL · contract IBPAY-2026-014
Submitted by : demo.operator · 2026-05-20

1. Overview
-----------
This document specifies the integration contract between the demo
payment-orchestration layer and the Acme Payments Iberia gateway.
Coverage : SEPA Credit Transfer (SCT), SEPA Instant (SCT Inst) and
domestic Spain bizum-rail acknowledgments.

Production rollout target : 2026-Q3.
Confidence rating (vendor self-declared) : 70%.
Internal verification : pending reviewer sign-off (this review).

2. Authentication
------------------
- mTLS, client cert issued by the shared demo PKI (trust-store: demo-root-2024).
- Bearer token in Authorization header (rotated every 30 days).
- Optional HMAC-SHA256 body signature in X-Acme-Signature for high-value
  transfers (> EUR 100K). Rejected without 401 if missing.

3. Endpoints (vendor side)
--------------------------
POST   /v1/sct/credit-transfer        SCT initiation
POST   /v1/sct-inst/credit-transfer   SCT Inst initiation (<10s SLA)
GET    /v1/sct/{id}/status            Status enquiry, polling cap 1/s
POST   /v1/recall                     R-message (return / reject)
POST   /v1/bizum/ack                  Bizum acknowledgment relay

4. Idempotency
--------------
Required idempotency-key header on every POST. Acme retains the key for
24h. Duplicate requests within window return the original response with
HTTP 200; outside window return 409 Conflict.

5. Concerns flagged by the demo operator (submitter)
---------------------------------------
- Section 4 retention window (24h) is shorter than general guidance (72h).
  Recommend negotiation with Acme account manager.
- HMAC threshold at EUR 100K vs demo policy threshold at EUR 50K —
  policy mismatch, requires either contract amendment OR an internal
  override gateway rule.
- No explicit dispute-resolution endpoint; relies on R-message which
  doesn't cover all demo business cases.`;

const D7_EXTRACTED = `=== Incident postmortem (DRAFT) — 2026-Q2 ===
Incident : INC-2026-0418 · severity S1 · duration 2h47m
Owner : demo.reviewer · status : draft, pending reviewer review

1. Summary
----------
Oracle PGA exhaustion on DEMO-DB-03 cascaded into connection
pool starvation across the payment-orchestration tier between
09:12 and 11:59 CET.

2. Client impact
----------------
- 1,284 SCT Inst transfers delayed beyond the 10s SLA.
- 3 corporate clients raised tickets (refs withheld in draft).
- No data loss; all transfers eventually settled.

3. Root cause
-------------
pga_aggregate_target left at staging value (4G) after the
2026-03 migration; a large AWR snapshot job pinned ~3.1G.

4. Follow-ups
-------------
- Raise pga_aggregate_target to 12G on prod (change CR-7781).
- Alert on PGA > 80% for 5 min.
- Backport tuning note to /demo/runbooks/oracle-pga-tuning.`;

export const DOCUMENT_FIXTURES: readonly Document[] = [
  {
    doc_id: 'd1',
    track_id: 'tk_2026-05-29_001',
    file_path: 'oracle-restart-procedure.pdf',
    content_summary: 'Step-by-step guide for restarting Oracle DB on RHEL 9 in a demo environment',
    content_length: 41800,
    status: 'PROCESSED',
    chunks_count: 418,
    created_at: TWO_HOURS_AGO,
    updated_at: TWO_HOURS_AGO,
    error_msg: null,
    metadata: {
      mime: 'application/pdf',
      uploader: 'demo.steward',
      // Structured MIP classification (post-ingestion via PR #157 hook).
      // C2 = Confidentiel — many demo runbooks land here. Visible as a yellow
      // pill in DocumentsTab + PendingDocs cards.
      classification: {
        class_id: 'C2',
        class_name: 'C2 Confidentiel',
        label_guid: '22222222-2222-2222-2222-222222222222',
        raw_name: 'C2 Confidentiel',
        set_date: '2026-03-12T14:22:01Z',
        method: 'Standard',
        source_format: 'ooxml',
        reason: null,
        meta: { Enabled: 'true', SiteId: '{99999999-tenant-id-here}' },
      },
    },
    type: 'file',
    tags: ['rman', 'oracle'],
    folder: 'default',
    visibility: 'private',
  },
  {
    doc_id: 'd2',
    track_id: null,
    file_path: '/demo/runbooks/oracle-pga-tuning',
    content_summary: 'Oracle PGA memory tuning recommendations and worked examples',
    content_length: 5400,
    status: 'PROCESSED',
    chunks_count: 54,
    created_at: ONE_DAY_AGO,
    updated_at: ONE_DAY_AGO,
    error_msg: null,
    metadata: { source: 'confluence', uploader: 'demo.steward', classification: 'internal' },
    type: 'confluence',
    tags: ['rman'],
    folder: 'default',
    visibility: 'private',
    extracted_text: D2_EXTRACTED,
    review: {
      state: 'modified',
      update: {
        requested_by: 'demo.reviewer',
        edited_rel: '2h ago',
        detected_at: '2026-05-26',
        chunks_indexed: 54,
        // Body = diff content only. The "edited by X · date" preamble belongs
        // in the meta line, not duplicated here.
        summary_diff:
          'Added 2 new sections: "PGA sizing on RHEL 9 large pages" and ' +
          '"AWR snapshot interpretation for memory pressure". Existing ' +
          'sections unchanged. ~340 words added.',
      },
    },
  },
  {
    doc_id: 'd3',
    track_id: 'tk_2026-05-29_002',
    file_path: 'huge-archive.zip',
    content_summary: 'Failed ingest — unsupported MIME',
    content_length: 0,
    status: 'FAILED',
    chunks_count: 0,
    created_at: THIRTY_MIN_AGO,
    updated_at: THIRTY_MIN_AGO,
    error_msg: 'Unsupported MIME type: application/zip',
    metadata: { mime: 'application/zip', uploader: 'demo.steward' },
    type: 'sharepoint',
    tags: [],
    folder: 'default',
    visibility: 'private',
  },
  {
    doc_id: 'd4',
    track_id: 'tk_2026-05-29_003',
    file_path: 'memgraph-mage-3.8-release-notes.md',
    content_summary: 'Memgraph MAGE 3.8 release notes — vector_search improvements',
    content_length: 4200,
    status: 'PROCESSED',
    chunks_count: 42,
    created_at: TWENTY_FIVE_MIN_AGO,
    updated_at: TWENTY_FIVE_MIN_AGO,
    error_msg: null,
    metadata: {
      mime: 'text/markdown',
      uploader: 'demo.steward',
      // C1 = Public — vendor release notes, no confidentiality. Visible as
      // a neutral grey pill (the pill won't grab attention).
      classification: {
        class_id: 'C1',
        class_name: 'C1 Public',
        label_guid: '11111111-1111-1111-1111-111111111111',
        raw_name: 'C1 Public',
        set_date: '2026-04-08T09:15:00Z',
        method: 'Standard',
        source_format: 'ooxml',
        reason: null,
        meta: { Enabled: 'true' },
      },
    },
    type: 'file',
    tags: ['memgraph', 'mage'],
    folder: 'default',
    visibility: 'private',
  },
  {
    doc_id: 'd5',
    track_id: null,
    file_path: '/demo/incidents/2026-04-prod-outage',
    content_summary: 'Postmortem: 2026-04 prod outage, Oracle PGA OOM cascade',
    content_length: 15600,
    status: 'PROCESSED',
    chunks_count: 156,
    created_at: THREE_DAYS_AGO,
    updated_at: THREE_DAYS_AGO,
    error_msg: null,
    // classification > internal → DocDetailPanel Chunks tab truncates (compliance doctrine)
    metadata: { source: 'confluence', uploader: 'demo.contributor', classification: 'restricted' },
    type: 'confluence',
    tags: ['incident', 'oracle', 'production'],
    folder: 'default',
    visibility: 'private',
  },
  {
    doc_id: 'd6',
    track_id: 'tk_2026-05-20_009',
    file_path: 'cft-vendor-api-spec-draft.pdf',
    content_summary:
      'Vendor-provided spec — needs sign-off by a reviewer before retrieval. Confidence sourcing uncertain.',
    content_length: 2355,
    status: 'PROCESSING',
    chunks_count: 47,
    created_at: '2026-05-20T09:00:00Z',
    updated_at: '2026-05-20T09:00:00Z',
    error_msg: null,
    metadata: {
      mime: 'application/pdf',
      uploader: 'demo.operator',
      // C3 = Strictement Confidentiel — vendor draft, restricted. Visible
      // as a red pill. DocDetailPanel chunks will be truncated.
      classification: {
        class_id: 'C3',
        class_name: 'C3 Strictement Confidentiel',
        label_guid: '33333333-3333-3333-3333-333333333333',
        raw_name: 'C3 Strictement Confidentiel',
        set_date: '2026-05-18T11:03:45Z',
        method: 'Standard',
        source_format: 'ooxml',
        reason: null,
        meta: { Enabled: 'true' },
      },
    },
    type: 'file',
    tags: ['cft', 'network'],
    folder: 'default',
    visibility: 'private',
    extracted_text: D6_EXTRACTED,
    review: {
      state: 'pending-review',
      requested_by: 'demo.operator',
      requested_at: '2026-05-20',
      justification:
        'Vendor-provided spec — needs sign-off by a reviewer before retrieval. ' +
        'Confidence sourcing uncertain.',
    },
  },
  {
    doc_id: 'd7',
    track_id: 'tk_2026-05-20_010',
    file_path: '/demo/runbooks/incident-2026-Q2-postmortem-draft',
    content_summary:
      'Contains client-impact figures — reviewer review required before exposure to broader retrieval.',
    content_length: 8900,
    status: 'PROCESSING',
    chunks_count: 89,
    created_at: '2026-05-20T10:00:00Z',
    updated_at: '2026-05-20T10:00:00Z',
    error_msg: null,
    metadata: { source: 'confluence', uploader: 'demo.reviewer' },
    type: 'confluence',
    tags: ['incident', 'production'],
    folder: 'default',
    visibility: 'private',
    extracted_text: D7_EXTRACTED,
    review: {
      state: 'pending-review',
      requested_by: 'demo.reviewer',
      requested_at: '2026-05-20',
      justification:
        'Contains client-impact figures — reviewer review required before exposure to broader retrieval.',
    },
  },
];
