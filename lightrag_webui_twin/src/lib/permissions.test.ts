import { describe, expect, it } from 'vitest';
import { canManageFolders } from './permissions';
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
  folders: ['cib'],
  idp: 'keycloak',
  idp_realm: 'twin-cib',
  sub: 'reader-1',
  session_expires: '2026-06-04T23:59:00Z',
  gateway_scopes: ['read:documents'],
};

describe('canManageFolders', () => {
  it('allows null user for IdP-dormant mode', () => {
    expect(canManageFolders(null)).toBe(true);
  });

  it('allows users carrying admin:folders', () => {
    expect(
      canManageFolders({
        ...baseUser,
        gateway_scopes: ['read:documents', 'admin:folders'],
      }),
    ).toBe(true);
  });

  it('rejects users missing admin:folders', () => {
    expect(canManageFolders(baseUser)).toBe(false);
  });
});
