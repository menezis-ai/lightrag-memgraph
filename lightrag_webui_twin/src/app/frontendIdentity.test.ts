import { describe, expect, it } from 'vitest';
import type { AuthenticatedUser, PalierLevel } from '../types/auth';
import { resolveFrontendIdentity } from './frontendIdentity';

function authenticatedUser(
  level: PalierLevel,
  label: 'Reader' | 'Contributor' | 'Steward',
): AuthenticatedUser {
  return {
    sso_subject: `subject-${level}`,
    email: `${label.toLowerCase()}@example.com`,
    name: label,
    palier: { level, label, scopes: [] },
    folders: ['default'],
    idp: 'test-idp',
    idp_realm: 'test',
    sub: `subject-${level}`,
    session_expires: '2099-12-31T23:59:00Z',
    gateway_scopes: [],
  };
}

describe('resolveFrontendIdentity', () => {
  it.each([
    [1, 'Reader'],
    [2, 'Contributor'],
    [3, 'Steward'],
  ] as const)(
    'projects an authenticated palier-%i %s without elevating it',
    (level, label) => {
      const identity = resolveFrontendIdentity(
        authenticatedUser(level, label),
        true,
      );

      expect(identity.actor).toBe(`${label.toLowerCase()}@example.com`);
      expect(identity.tagUser).toEqual({
        name: `${label.toLowerCase()}@example.com`,
        palier: level,
        role: label,
      });
    },
  );

  it('grants the named steward fallback only in confirmed open-access mode', () => {
    expect(resolveFrontendIdentity(null, false)).toEqual({
      actor: 'operator@example.com',
      tagUser: {
        name: 'operator@example.com',
        palier: 3,
        role: 'open-access steward',
      },
    });
  });

  it('keeps a debug steward read-only while the auth posture is unresolved', () => {
    expect(
      resolveFrontendIdentity(authenticatedUser(3, 'Steward'), null),
    ).toEqual({
      actor: 'anonymous',
      tagUser: { name: 'anonymous', palier: 1, role: 'unresolved reader' },
    });
  });

  it.each([null, true] as const)(
    'keeps an absent user read-only when authEnabled is %s',
    (authEnabled) => {
      expect(resolveFrontendIdentity(null, authEnabled)).toEqual({
        actor: 'anonymous',
        tagUser: { name: 'anonymous', palier: 1, role: 'unresolved reader' },
      });
    },
  );
});
