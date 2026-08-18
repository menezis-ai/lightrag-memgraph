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

import type { RetrievalSource, SourceAnchor } from '../types/retrieval';

import type {
  TwinAnswerStatus,
  TwinQueryResponse,
  TwinQuerySourceAnchor,
} from './resources';

/**
 * Shape-validate an anchor before letting it into the UI contract.
 * The backend only publishes structurally valid anchors, but the anchor is
 * an explicitly non-authoritative hint — a malformed one (legacy backend,
 * fixture drift) must degrade to "no anchor", never to NaN offsets reaching
 * a slice call. Bounds against the actual chunk text are checked at render
 * time, where the text is known.
 */
function sanitizeAnchor(
  anchor: TwinQuerySourceAnchor | null | undefined,
): SourceAnchor | undefined {
  if (!anchor) return undefined;
  const { start, end, paragraph_idx, paragraph_count, confidence, method } =
    anchor;
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    end <= start ||
    !Number.isInteger(paragraph_idx) ||
    paragraph_idx < 0 ||
    !Number.isInteger(paragraph_count) ||
    paragraph_count < 1 ||
    typeof confidence !== 'number' ||
    !Number.isFinite(confidence) ||
    typeof method !== 'string'
  ) {
    return undefined;
  }
  return { start, end, paragraph_idx, paragraph_count, confidence, method };
}

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
  model?: string;
} {
  const sources: RetrievalSource[] = (res.sources ?? []).map((s) => ({
    n: s.n,
    type:
      s.type === 'file' ||
      s.type === 'url' ||
      s.type === 'confluence' ||
      s.type === 'sharepoint'
        ? s.type
        : ('file' as const),
    name: s.name,
    meta: s.meta ?? undefined,
    score: s.score,
    retrieval_origin: s.retrieval_origin ?? undefined,
    doc_id: s.doc_id ?? undefined,
    chunk_id: s.chunk_id ?? undefined,
    anchor: sanitizeAnchor(s.anchor),
  }));
  return {
    response: res.response,
    sources,
    answer_status: res.answer_status,
    model: res.model,
  };
}
