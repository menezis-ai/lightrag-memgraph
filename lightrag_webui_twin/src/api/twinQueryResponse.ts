/**
 * Projection helpers between the Twin overlay query contract
 * (``TwinQueryResponse`` in ``./resources``) and the
 * ``RetrievalTab`` host callback shape.
 *
 * Lives in its own file (rather than next to the ``App`` component
 * that hosts the wiring) so the ``react-refresh/only-export-components``
 * lint rule stays happy on the App module, and so the projection is
 * trivially unit-testable in isolation.
 */

import type { SourceType } from '../components/Icon';
import type { RetrievalSource } from '../types/retrieval';

import type { TwinAnswerStatus, TwinQueryResponse } from './resources';

/**
 * Project a Twin overlay ``/twin/api/query`` (or ``/stream``) response
 * into the shape ``RetrievalTab`` expects from ``onSendQuery`` /
 * ``onStreamQuery``.
 *
 * The non-trivial part is propagating ``answer_status`` end-to-end —
 * the field exists on ``TwinQueryResponse`` (TR-RET-02 step 1) and on
 * ``ChatMessage`` (via ``RetrievalTab``), and the absence of forwarding
 * here used to silently flatten every answer back to ``"grounded"``,
 * defeating the Sources-panel-suppression behaviour the React port
 * already supports. Codex review on PR fix/tag-filter-honesty caught
 * this latent bug; this helper makes the wiring explicit and tested.
 */
export function mapTwinQueryResponseForRetrievalTab(
  res: TwinQueryResponse,
): {
  response: string;
  sources: RetrievalSource[];
  answer_status?: TwinAnswerStatus;
} {
  const sources: RetrievalSource[] = (res.sources ?? []).map((s) => ({
    n: s.n,
    type:
      s.type === 'file' ||
      s.type === 'url' ||
      s.type === 'confluence' ||
      s.type === 'sharepoint'
        ? (s.type as SourceType)
        : ('file' as const),
    name: s.name,
    meta: s.meta ?? undefined,
    score: s.score,
  }));
  return {
    response: res.response,
    sources,
    answer_status: res.answer_status,
  };
}
