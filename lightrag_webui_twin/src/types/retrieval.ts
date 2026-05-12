/**
 * Retrieval types — Threads, ChatMessages, AnswerTokens, Sources.
 *
 * Mirror the shapes consumed by the Retrieval tab and double as the contract
 * for backend phase 1:
 *   POST /retrieval                -> { tokens, sources }
 *   GET  /threads                  -> RetrievalThread[]
 *   POST /threads / DELETE /threads/{id}
 *
 * The streamed response is a flat array of `AnswerToken`s (strings), with
 * inline citation markers `{cite:N}` and backtick `code` spans. The client
 * parses them into AnswerParts for rendering.
 */

import type { SourceType } from '../components/Icon';

export type AnswerToken = string;

export type AnswerPart =
  | { type: 'text'; value: string }
  | { type: 'code'; value: string }
  | { type: 'cite'; value: number };

export interface RetrievalSource {
  /** Citation number, 1-indexed. Matches `{cite:n}` in the tokens. */
  n: number;
  type: SourceType;
  name: string;
  meta?: string | null;
  /** Cosine / hybrid similarity, 0..1. */
  score: number;
}

export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  /** User messages carry a plain `text`. */
  text?: string;
  /** Assistant messages carry parsed-up tokens + sources. */
  tokens?: readonly AnswerToken[];
  sources?: readonly RetrievalSource[];
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

/**
 * Parse a token stream into typed AnswerParts. Tokens contain inline
 * `{cite:N}` and backtick-code markers.
 */
export function parseAnswer(tokens: readonly AnswerToken[]): AnswerPart[] {
  const out: AnswerPart[] = [];
  tokens.forEach((tk) => {
    const re = /\{cite:(\d+)\}|`([^`]+)`/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(tk)) !== null) {
      if (m.index > last) {
        out.push({ type: 'text', value: tk.slice(last, m.index) });
      }
      if (m[1]) {
        out.push({ type: 'cite', value: parseInt(m[1], 10) });
      } else if (m[2]) {
        out.push({ type: 'code', value: m[2] });
      }
      last = re.lastIndex;
    }
    if (last < tk.length) {
      out.push({ type: 'text', value: tk.slice(last) });
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
