/**
 * Typed document fixtures, ported from Desktop/UI/data.js `window.MOCK_DOCUMENTS`.
 *
 * Usage:
 *   - In tests: import and feed components directly.
 *   - In Storybook stories: same.
 *   - In dev MSW handlers: return these arrays from request handlers.
 *
 * Edits in the proto -> mirror here when refining the contract. Once backend
 * phase 1 ships, these become Storybook/test-only — the runtime app reads
 * from /documents.
 */

import type { Document } from '../types/document';

export const DOCUMENT_FIXTURES: readonly Document[] = [
  {
    id: 'd1',
    type: 'file',
    source: 'oracle-restart-procedure.pdf',
    summary: 'Step-by-step guide for restarting Oracle DB on RHEL 9 in CIB prod',
    tags: ['rman', 'oracle'],
    status: 'completed',
    chunks: 418,
    updated: '2h ago',
    visibility: 'private',
    workspace: 'cib',
  },
  {
    id: 'd2',
    type: 'confluence',
    source: '/cib/runbooks/oracle-pga-tuning',
    summary: 'Oracle PGA memory tuning recommendations and worked examples',
    tags: ['rman'],
    status: 'completed',
    chunks: 54,
    updated: '1d ago',
    visibility: 'private',
    workspace: 'cib',
  },
  {
    id: 'd3',
    type: 'sharepoint',
    source: 'huge-archive.zip',
    summary: 'Failed ingest — unsupported MIME',
    tags: [],
    status: 'failed',
    chunks: 0,
    updated: '30m ago',
    visibility: 'private',
    workspace: 'cib',
  },
  {
    id: 'd4',
    type: 'file',
    source: 'memgraph-mage-3.8-release-notes.md',
    summary: 'Memgraph MAGE 3.8 release notes — vector_search improvements',
    tags: ['memgraph', 'mage'],
    status: 'completed',
    chunks: 42,
    updated: '25m ago',
    visibility: 'private',
    workspace: 'cib',
  },
  {
    id: 'd5',
    type: 'confluence',
    source: '/cib/incidents/2026-04-prod-outage',
    summary: 'Postmortem: 2026-04 prod outage, Oracle PGA OOM cascade',
    tags: ['incident', 'oracle', 'production'],
    status: 'completed',
    chunks: 156,
    updated: '3d ago',
    visibility: 'private',
    workspace: 'cib',
  },
  {
    id: 'd6',
    type: 'file',
    source: 'pending-review-doc.pdf',
    summary: 'Awaiting palier-2 review before publication',
    tags: ['pending-review'],
    status: 'processing',
    chunks: 21,
    updated: '5h ago',
    visibility: 'private',
    workspace: 'cib',
  },
];
