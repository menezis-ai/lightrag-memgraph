import { describe, expect, it } from 'vitest';
import { canManageSpaces } from './permissions';
import type { AuthenticatedUser } from '../types/auth';

const baseUser: AuthenticatedUser = {
  sso_subject: 'reader@example.test',
  email: 'reader@example.test',
  name: 'Reader',
  palier: {
    level: 1,
    label: 'Reader',
    scopes: ['twin:read'],
  },
  workspaces: ['cib'],
  idp: 'keycloak',
  idp_realm: 'twin-cib',
  sub: 'reader-1',
  session_expires: '2026-06-04T23:59:00Z',
  gateway_scopes: ['read:documents'],
};

describe('canManageSpaces', () => {
  it('allows null user for IdP-dormant mode', () => {
    expect(canManageSpaces(null)).toBe(true);
  });

  it('allows users carrying admin:spaces', () => {
    expect(
      canManageSpaces({
        ...baseUser,
        gateway_scopes: ['read:documents', 'admin:spaces'],
      }),
    ).toBe(true);
  });

  it('rejects users missing admin:spaces', () => {
    expect(canManageSpaces(baseUser)).toBe(false);
  });
});
