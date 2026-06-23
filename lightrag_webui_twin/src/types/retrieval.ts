/**
 * Retrieval types — Threads, ChatMessages, AnswerTokens, Sources.
 *
 * Mirror the shapes consumed by the Retrieval tab and double as the contract
 * for backend phase 1:
 *   POST /retrieval                -> { tokens, sources }
 *   GET  /threads                  -> RetrievalThread[]
 *   POST /threads / DELETE /threads/{id}
 *
 * The streamed response is a flat array of strings, with
 * inline citation markers `{cite:N}` and backtick `code` spans. The client
 * parses them into AnswerParts for rendering.
 */

import type { SourceType } from '../components/Icon';

export type InlineAnswerPart =
  | { type: 'text'; value: string }
  | { type: 'bold'; value: string }
  | { type: 'code'; value: string }
  | { type: 'cite'; value: number };

export type AnswerPart =
  | InlineAnswerPart
  | { type: 'heading'; level: 1 | 2 | 3; children: readonly InlineAnswerPart[] }
  | { type: 'listItem'; ordered: boolean; children: readonly InlineAnswerPart[] }
  | { type: 'lineBreak' };

export interface RetrievalSource {
  /** Citation number, 1-indexed. Matches `{cite:n}` in the tokens. */
  n: number;
  type: SourceType;
  name: string;
  meta?: string | null;
  /** Cosine / hybrid similarity, 0..1. */
  score: number;
  /** Optional document id for direct drill-down from citations/sources. */
  doc_id?: string | null;
  /** Optional chunk id cited by the backend. */
  chunk_id?: string | null;
}

export type ChatRole = 'user' | 'assistant';

/**
 * Mirrors the backend ``answer_status`` field on ``/twin/api/query``
 * (TR-RET-02). ``insufficient_information`` is the canonical
 * machine-readable signal that LightRAG had no usable context — the
 * React port uses it to suppress the Sources panel cleanly instead of
 * parsing the LLM prose. Default = ``grounded``.
 */
export type AnswerStatus = 'grounded' | 'insufficient_information';

export interface ChatMessage {
  role: ChatRole;
  /** User messages carry a plain `text`. */
  text?: string;
  /** Assistant messages carry parsed-up tokens + sources. */
  tokens?: readonly string[];
  sources?: readonly RetrievalSource[];
  /** Assistant-only: propagated from the backend answer_status flag. */
  answerStatus?: AnswerStatus;
  /** Assistant-only: Top K selected when this answer was requested. */
  requestedTopK?: number;
}

export interface RetrievalThread {
  id: string;
  title: string;
  /** Epoch ms. */
  created: number;
  /** Epoch ms. */
  updated: number;
  messages: readonly ChatMessage[];
}

export type QueryMode = 'naive' | 'local' | 'global' | 'hybrid' | 'mix' | 'bypass';

export const QUERY_MODES: readonly QueryMode[] = [
  'naive',
  'local',
  'global',
  'hybrid',
  'mix',
  'bypass',
];

function stripTrailingReferencesSection(text: string): string {
  const lines = text.split('\n');
  const start = lines.findIndex((line) =>
    /^\s*#{1,6}\s*(?:references|références)\b/i.test(line) ||
    /^\s*(?:references|références)\s*:?\s*$/i.test(line),
  );
  if (start === -1) return text;
  return lines.slice(0, start).join('\n').trimEnd();
}

/**
 * Parse a token stream into typed AnswerParts. Tokens contain inline
 * `{cite:N}` (proto/fixture format) or `[N]` (LightRAG prompt output)
 * citation markers, plus backtick-code spans.
 */
export function parseAnswer(tokens: readonly string[]): AnswerPart[] {
  const out: AnswerPart[] = [];
  const parseInline = (text: string): InlineAnswerPart[] => {
    const parts: InlineAnswerPart[] = [];
    const re = /\*\*([^*]+)\*\*|\{cite:(\d+)\}|\[\^?(\d+)\]|`([^`]+)`/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) {
        parts.push({ type: 'text', value: text.slice(last, m.index) });
      }
      if (m[1]) {
        parts.push({ type: 'bold', value: m[1] });
      } else if (m[2] || m[3]) {
        parts.push({ type: 'cite', value: Number.parseInt(m[2] || m[3], 10) });
      } else if (m[4]) {
        parts.push({ type: 'code', value: m[4] });
      }
      last = re.lastIndex;
    }
    if (last < text.length) {
      parts.push({ type: 'text', value: text.slice(last) });
    }
    return parts;
  };

  const text = stripTrailingReferencesSection(tokens.join(''));
  const lines = text.split('\n');
  const hasMarkdownBlocks = lines.length > 1 || lines.some((line) =>
    /^(?:(?:#{1,3})\s+|\s*[-*]\s+|\s*\d+\.\s+)/.test(line),
  );

  if (!hasMarkdownBlocks) {
    return parseInline(text);
  }

  lines.forEach((line, index) => {
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    const bullet = /^\s*[-*]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+\.\s+(.+)$/.exec(line);
    if (heading) {
      out.push({
        type: 'heading',
        level: heading[1].length as 1 | 2 | 3,
        children: parseInline(heading[2]),
      });
    } else if (bullet || ordered) {
      out.push({
        type: 'listItem',
        ordered: Boolean(ordered),
        children: parseInline((bullet ?? ordered)?.[1] ?? ''),
      });
    } else if (line) {
      out.push(...parseInline(line));
    }
    if (index < lines.length - 1) {
      out.push({ type: 'lineBreak' });
    }
  });
  return out;
}

/**
 * Cheap epoch-ms -> "Xm" / "Xh" / "Xd" relative-time string.
 * Returns "" for null/invalid input, "now" for <60s.
 */
export function relTime(ts: number | string | null | undefined): string {
  if (ts === null || ts === undefined || ts === '') return '';
  const epoch = typeof ts === 'number' ? ts : Date.parse(ts);
  if (!Number.isFinite(epoch)) return '';
  const d = Date.now() - epoch;
  if (d < 0) return 'now';
  if (d < 60_000) return 'now';
  if (d < 3_600_000) return `${Math.round(d / 60_000)}m`;
  if (d < 86_400_000) return `${Math.round(d / 3_600_000)}h`;
  return `${Math.round(d / 86_400_000)}d`;
}
