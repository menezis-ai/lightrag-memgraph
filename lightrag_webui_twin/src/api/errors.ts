/**
 * Front-side error mapping for Twin mutation responses.
 *
 * Mirrors the server-side error contract introduced for TR-KG-01:
 *
 *   POST /twin/api/graph/entities
 *     201 → created
 *     409 → duplicate, unless the backend detail says the ingestion pipeline
 *           refused the write while busy
 *     422 → Pydantic validation (empty/whitespace name, missing type, …)
 *     503 → Memgraph backend rejected the write (driver down, lock, …)
 *     500 → entity was created, projection failed (half-success)
 *
 * The host component switches on ``kind`` to pick the inline copy / toast
 * shape. Keeping the mapping in a pure helper makes it testable in
 * isolation and prevents the route from drifting silently.
 */

import { ApiError } from './client';
import {
  backendDetail,
  isPipelineBusyDetail,
  userErrorMessage,
} from '../lib/errorMessages';

export type CreateEntityErrorKind =
  | 'busy'
  | 'duplicate'
  | 'validation'
  | 'backend'
  | 'projection'
  | 'unknown';

export interface CreateEntityErrorResult {
  kind: CreateEntityErrorKind;
  message: string;
}

/**
 * Map a thrown error from ``api.createGraphEntity`` to a typed result.
 *
 * Non-``ApiError`` throws (network failures, JSON parse errors, …) land
 * on ``unknown`` with shared operator-facing copy rather than a silent
 * dropped submit.
 */
export function mapCreateEntityError(
  err: unknown,
  entityName: string,
): CreateEntityErrorResult {
  if (err instanceof ApiError) {
    switch (err.status) {
      case 409:
        if (isPipelineBusyDetail(backendDetail(err.body))) {
          return {
            kind: 'busy',
            message: userErrorMessage(err, { action: 'creating the entity' }),
          };
        }
        return {
          kind: 'duplicate',
          message: `An entity named “${entityName}” already exists. Choose a different name.`,
        };
      case 422:
        return {
          kind: 'validation',
          message: 'Invalid entity payload. Check the name and type.',
        };
      case 503:
        return {
          kind: 'backend',
          message:
            'Memgraph backend unavailable. Please retry in a moment.',
        };
      case 500:
        return {
          kind: 'projection',
          message: `“${entityName}” was created server-side but the graph hasn’t projected it yet — refreshing now.`,
        };
      default:
        return {
          kind: 'unknown',
          message: userErrorMessage(err, { action: 'creating the entity' }),
        };
    }
  }
  return {
    kind: 'unknown',
    message: userErrorMessage(err, { action: 'creating the entity' }),
  };
}
