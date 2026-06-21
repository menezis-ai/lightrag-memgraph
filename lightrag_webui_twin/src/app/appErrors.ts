import { ApiError } from '../api/client';

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

export function formatBackendError(error: unknown): string {
  if (error instanceof ApiError) return `${error.status} ${error.message}`;
  if (error instanceof Error) return error.message;
  return 'Backend request failed';
}
