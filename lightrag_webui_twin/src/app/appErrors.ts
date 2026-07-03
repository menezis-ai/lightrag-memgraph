import { userErrorMessage } from '../lib/errorMessages';

export type QueryLike<T> = {
  data?: T;
  isError: boolean;
  isLoading: boolean;
  error: unknown;
};

export interface BackendResourceError {
  label: string;
  message: string;
}

export function resourceError<T>(
  label: string,
  query: QueryLike<T>,
): BackendResourceError | null {
  if (query.data || query.isLoading || !query.isError) {
    return null;
  }
  return { label, message: formatBackendError(query.error) };
}

/**
 * Reduce any thrown value to operator-facing copy. Since the error-UX
 * pass (2026-07-03) this delegates to the shared mapping layer — the
 * raw `status METHOD /path → status` string never reaches a banner.
 */
export function formatBackendError(error: unknown): string {
  return userErrorMessage(error, { action: 'loading data from the backend' });
}
