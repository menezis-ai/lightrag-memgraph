/**
 * Typed document fixtures aligned on LightRAG DocStatus + Twin overlay.
 *
 * Used for tests, Storybook, MSW handlers, and first-paint seeding in App.tsx.
 * Real runtime data comes from `/documents` (LightRAG) merged with
 * `/twin/api/documents/{id}/metadata` (overlay).
 */

import type { Document } from '../types/document';

const NOW = '2026-05-29T16:00:00Z';
const TWO_HOURS_AGO = '2026-05-29T14:00:00Z';
const ONE_DAY_AGO = '2026-05-28T16:00:00Z';
const THIRTY_MIN_AGO = '2026-05-29T15:30:00Z';
const TWENTY_FIVE_MIN_AGO = '2026-05-29T15:35:00Z';
const THREE_DAYS_AGO = '2026-05-26T16:00:00Z';
const FIVE_HOURS_AGO = '2026-05-29T11:00:00Z';

export const DOCUMENT_FIXTURES: readonly Document[] = [
  {
    doc_id: 'd1',
    track_id: 'tk_2026-05-29_001',
    file_path: 'oracle-restart-procedure.pdf',
    content_summary: 'Step-by-step guide for restarting Oracle DB on RHEL 9 in CIB prod',
    content_length: 41800,
    status: 'PROCESSED',
    chunks_count: 418,
    created_at: TWO_HOURS_AGO,
    updated_at: TWO_HOURS_AGO,
    error_msg: null,
    metadata: { mime: 'application/pdf', uploader: 'claire.benoit' },
    type: 'file',
    tags: ['rman', 'oracle'],
    workspace: 'cib',
    visibility: 'private',
  },
  {
    doc_id: 'd2',
    track_id: null,
    file_path: '/cib/runbooks/oracle-pga-tuning',
    content_summary: 'Oracle PGA memory tuning recommendations and worked examples',
    content_length: 5400,
    status: 'PROCESSED',
    chunks_count: 54,
    created_at: ONE_DAY_AGO,
    updated_at: ONE_DAY_AGO,
    error_msg: null,
    metadata: { source: 'confluence', uploader: 'claire.benoit' },
    type: 'confluence',
    tags: ['rman'],
    workspace: 'cib',
    visibility: 'private',
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
    metadata: { mime: 'application/zip', uploader: 'claire.benoit' },
    type: 'sharepoint',
    tags: [],
    workspace: 'cib',
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
    metadata: { mime: 'text/markdown', uploader: 'claire.benoit' },
    type: 'file',
    tags: ['memgraph', 'mage'],
    workspace: 'cib',
    visibility: 'private',
  },
  {
    doc_id: 'd5',
    track_id: null,
    file_path: '/cib/incidents/2026-04-prod-outage',
    content_summary: 'Postmortem: 2026-04 prod outage, Oracle PGA OOM cascade',
    content_length: 15600,
    status: 'PROCESSED',
    chunks_count: 156,
    created_at: THREE_DAYS_AGO,
    updated_at: THREE_DAYS_AGO,
    error_msg: null,
    metadata: { source: 'confluence', uploader: 'manu.dev' },
    type: 'confluence',
    tags: ['incident', 'oracle', 'production'],
    workspace: 'cib',
    visibility: 'private',
  },
  {
    doc_id: 'd6',
    track_id: 'tk_2026-05-29_004',
    file_path: 'pending-review-doc.pdf',
    content_summary: 'Awaiting palier-2 review before publication',
    content_length: 2100,
    status: 'PROCESSING',
    chunks_count: 21,
    created_at: FIVE_HOURS_AGO,
    updated_at: FIVE_HOURS_AGO,
    error_msg: null,
    metadata: { mime: 'application/pdf', uploader: 'fatima.t' },
    type: 'file',
    tags: ['pending-review'],
    workspace: 'cib',
    visibility: 'private',
    review: {
      state: 'pending-review',
      requested_by: 'fatima.t',
      requested_at: FIVE_HOURS_AGO,
      justification: 'Source mentions vendor disclosure clause — needs Steward sign-off.',
    },
  },
];

export { NOW as DOCUMENT_FIXTURES_NOW };
