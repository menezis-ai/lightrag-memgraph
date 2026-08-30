import type { AuthenticatedUser } from '../types/auth';
import type { TagCurrentUser } from '../types/tag';

export interface FrontendIdentity {
  /** Display/audit hint sent by legacy frontend payloads. Never auth proof. */
  actor: string;
  /** Capability projection used only to hide actions the server will reject. */
  tagUser: TagCurrentUser;
}

const OPEN_ACCESS_IDENTITY: FrontendIdentity = {
  actor: 'operator@example.com',
  tagUser: {
    name: 'operator@example.com',
    palier: 3,
    role: 'open-access steward',
  },
};

const UNRESOLVED_IDENTITY: FrontendIdentity = {
  actor: 'anonymous',
  tagUser: {
    name: 'anonymous',
    palier: 1,
    role: 'unresolved reader',
  },
};

function resolvedName(user: AuthenticatedUser): string {
  return (
    user.email.trim() ||
    user.sso_subject.trim() ||
    user.name.trim() ||
    user.sub.trim() ||
    'authenticated-user'
  );
}

/**
 * Project the authenticated user into legacy component props.
 *
 * `authEnabled === false` is deliberately strict: `null` means the auth
 * posture was not resolved (initial request or transport failure), so it must
 * fail closed instead of inheriting the open-access steward capability.
 */
export function resolveFrontendIdentity(
  user: AuthenticatedUser | null,
  authEnabled: boolean | null,
): FrontendIdentity {
  if (authEnabled === null) return UNRESOLVED_IDENTITY;
  if (user) {
    const name = resolvedName(user);
    return {
      actor: name,
      tagUser: {
        name,
        palier: user.palier.level,
        role: user.palier.label,
      },
    };
  }
  if (authEnabled === false) return OPEN_ACCESS_IDENTITY;
  return UNRESOLVED_IDENTITY;
}
