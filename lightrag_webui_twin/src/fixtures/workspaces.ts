/**
 * Typed workspace fixtures, ported from Desktop/UI/data.js `window.MOCK_WORKSPACES`.
 * Contract template for the future `GET /workspaces` endpoint.
 */

import type { Workspace } from '../types/topbar';

export const WORKSPACE_FIXTURES: readonly Workspace[] = [
  {
    id: 'cib',
    kb: 'CIB KB',
    visibility: 'private',
    sources: 247,
    role: 'admin / steward',
    current: true,
  },
  {
    id: 'cib-edge',
    kb: 'CIB Edge KB',
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
