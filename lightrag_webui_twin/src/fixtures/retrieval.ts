/**
 * Retrieval fixtures, ported from Desktop/UI/data.js.
 *
 * Note: thread.created / .updated use Date.now() at proto-load time. Here we
 * keep them as static offsets relative to `now`; in tests the function form
 * `makeSampleThreads()` lets tests freeze the clock if needed.
 */

import type {
  AnswerToken,
  RetrievalSource,
  RetrievalThread,
} from '../types/retrieval';

export const ANSWER_TOKENS_FIXTURE: readonly AnswerToken[] = [
  'To restart Oracle on RHEL 9 in a CIB production setup, follow the documented runbook ',
  '{cite:1} ',
  'which mandates first stopping listeners and pending RMAN jobs before issuing a graceful ',
  '`shutdown immediate` ',
  'from SQL*Plus. ',
  'After the database is down, verify the PGA configuration matches the tuning recommendations ',
  '{cite:3} ',
  '— this avoids the memory contention seen in incidents Q4 2025. ',
  'Restart the listener (`lsnrctl start`), then bring the DB back up with `startup`, ',
  'and finally re-enable RMAN backup jobs. ',
  'If a restore is required during this maintenance window, prefer point-in-time recovery as described in the RMAN cookbook ',
  '{cite:4} ',
  'rather than a full database restore. ',
  'Kernel parameters (`vm.swappiness`, `vm.dirty_ratio`) should already match the RHEL 9 hardening profile ',
  '{cite:5}',
  '.',
];

export const RETRIEVAL_SOURCES_FIXTURE: readonly RetrievalSource[] = [
  { n: 1, type: 'file', name: 'oracle-restart-procedure.pdf', meta: 'p.4', score: 0.94 },
  { n: 2, type: 'file', name: 'oracle-restart-procedure.pdf', meta: 'p.7', score: 0.89 },
  { n: 3, type: 'confluence', name: '/cib/runbooks/oracle-pga-tuning', meta: null, score: 0.82 },
  { n: 4, type: 'confluence', name: '/cib/runbooks/rman-restore-cookbook', meta: null, score: 0.78 },
  { n: 5, type: 'file', name: 'rhel9-kernel-tuning.pdf', meta: 'p.12', score: 0.71 },
];

/**
 * Sample threads. Returns a fresh array each call so callers can mutate
 * without cross-test contamination.
 */
export function makeSampleThreads(): RetrievalThread[] {
  return [
    {
      id: 'th_seed_1',
      title: 'RMAN backup restart procedure',
      created: Date.now() - 3_600_000 * 6,
      updated: Date.now() - 3_600_000 * 6,
      messages: [
        { role: 'user', text: 'How do I restart Oracle RMAN after a failed backup?' },
        { role: 'assistant', tokens: ['To', ' restart', ' RMAN', '…'], sources: [] },
      ],
    },
    {
      id: 'th_seed_2',
      title: 'CFT troubleshooting checklist',
      created: Date.now() - 86_400_000 * 2,
      updated: Date.now() - 86_400_000 * 2,
      messages: [],
    },
  ];
}

export const THREAD_FIXTURES: readonly RetrievalThread[] = makeSampleThreads();
