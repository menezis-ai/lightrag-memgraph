import type { AuthenticatedUser } from '../types/auth';

export function canManageFolders(user: AuthenticatedUser | null): boolean {
  if (!user) return true;
  return user.gateway_scopes.includes('admin:folders');
}
