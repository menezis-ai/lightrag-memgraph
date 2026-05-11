/**
 * Typed notification fixtures, ported from Desktop/UI/data.js
 * `window.MOCK_NOTIFICATIONS`. Contract template for `GET /notifications`.
 */

import type { Notification } from '../types/topbar';

export const NOTIFICATION_FIXTURES: readonly Notification[] = [
  {
    id: 'n_001',
    kind: 'tag-mutation',
    title: 'Tag',
    tagname: 'rman',
    suffix: 'applied',
    sub: 'oracle-restart-procedure.pdf · 418 chunks',
    rel: '12m ago',
    read: false,
  },
  {
    id: 'n_002',
    kind: 'source-failed',
    title: 'Ingestion failed',
    sub: 'huge-archive.zip · unsupported MIME',
    rel: '30m ago',
    read: false,
  },
  {
    id: 'n_003',
    kind: 'source-ready',
    title: 'Source ready',
    sub: 'memgraph-mage-3.8-release-notes.md · 42 chunks',
    rel: '25m ago',
    read: false,
  },
  {
    id: 'n_004',
    kind: 'pipeline-warning',
    title: 'Pipeline warning',
    sub: 'LLM extraction retrying · attempt 2/3',
    rel: '57m ago',
    read: true,
  },
  {
    id: 'n_005',
    kind: 'tag-mutation',
    title: 'Tag',
    tagname: 'iso20022',
    suffix: 'added',
    sub: 'palier 2 · category payment',
    rel: '22h ago',
    read: true,
  },
];
