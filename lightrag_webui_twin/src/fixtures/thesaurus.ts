/**
 * Typed thesaurus fixtures, ported from Desktop/UI/data.js
 * `window.MOCK_THESAURUS`. Contract template for `GET /thesaurus`.
 */

import type { ThesaurusEntry } from '../types/thesaurus';

export const THESAURUS_FIXTURES: readonly ThesaurusEntry[] = [
  { tag: 'rman', category: 'oracle', def: 'Recovery Manager — Oracle backup and recovery tool' },
  { tag: 'rmf-validated', category: 'governance', def: 'Reviewed by risk management framework' },
  { tag: 'oracle', category: 'oracle', def: 'Oracle Database engine and ecosystem' },
  { tag: 'production', category: 'lifecycle', def: 'Document applies to production environments' },
  { tag: 'incident', category: 'lifecycle', def: 'Postmortem or active incident material' },
  { tag: 'vmware', category: 'infra', def: 'VMware vSphere and ESXi runtime' },
  { tag: 'memgraph', category: 'infra', def: 'Memgraph graph database (MAGE extensions)' },
  { tag: 'network', category: 'network', def: 'Networking, routing, firewall topics' },
  { tag: 'cft', category: 'network', def: 'Cross-platform file transfer (Axway CFT)' },
  { tag: 'swift', category: 'payment', def: 'SWIFT messaging and payment flows' },
  { tag: 'governance', category: 'governance', def: 'Internal governance, audit, or charter' },
  { tag: 'deprecated', category: 'lifecycle', def: 'Marked for archival, do not act on' },
  { tag: 'pending-review', category: 'lifecycle', def: 'Awaiting risk review before publish' },
];
