/**
 * Typed folder fixtures, ported from Desktop/UI/data.js `window.MOCK_FOLDERS`.
 * Contract template for the `GET /folders` endpoint.
 */

import type { Folder } from '../types/topbar';

export const FOLDER_FIXTURES: readonly Folder[] = [
  {
    id: 'demo',
    kb: 'Demo KB',
    visibility: 'private',
    sources: 247,
    role: 'admin / steward',
    current: true,
  },
  {
    id: 'demo-edge',
    kb: 'Demo Edge KB',
    visibility: 'private',
    sources: 82,
    role: 'admin',
    current: false,
  },
  {
    id: 'payments',
    kb: 'Payments KB',
    visibility: 'internal',
    sources: 1318,
    role: 'reader',
    current: false,
  },
  {
    id: 'infra',
    kb: 'Infra Runbooks',
    visibility: 'internal',
    sources: 612,
    role: 'steward',
    current: false,
  },
  {
    id: 'sandbox',
    kb: 'Personal sandbox',
    visibility: 'private',
    sources: 9,
    role: 'owner',
    current: false,
  },
];
