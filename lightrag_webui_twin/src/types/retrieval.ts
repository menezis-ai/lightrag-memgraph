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
import type { SourceLink } from './document';

export type InlineAnswerPart =
  | { type: 'text'; value: string }
  | { type: 'bold'; value: string }
  | { type: 'italic'; value: string }
  | { type: 'code'; value: string }
  | { type: 'link'; label: string; href: string }
  | { type: 'cite'; value: number };

export type AnswerPart =
  | InlineAnswerPart
  | { type: 'heading'; level: 1 | 2 | 3; children: readonly InlineAnswerPart[] }
  | {
      type: 'listItem';
      ordered: boolean;
      marker: string;
      children: readonly InlineAnswerPart[];
    }
  | { type: 'blockquote'; children: readonly InlineAnswerPart[] }
  | { type: 'codeBlock'; language?: string; value: string }
  | {
      type: 'table';
      headers: readonly (readonly InlineAnswerPart[])[];
      rows: readonly (readonly (readonly InlineAnswerPart[])[])[];
    }
  | { type: 'lineBreak' }
  | { type: 'paragraphBreak' };

/**
 * Intra-chunk paragraph anchor (docs/adr/008-paragraph-citation-anchor.md, phase A).
 * Offsets only, never paragraph text — the chunk content is loaded on
 * demand through the existing chunk routes, and threads persisted to
 * localStorage must not grow with the feature. Heuristic and
 * non-authoritative: every consumer must render correctly without it.
 *
 * Offsets count Unicode CODE POINTS (Python string indices — the
 * backend contract), not UTF-16 units: slice on `Array.from(content)`,
 * never `String.prototype.slice`, or any astral character before the
 * paragraph shifts the range.
 */
export interface SourceAnchor {
  /** Start offset (inclusive) into the chunk content, in code points. */
  start: number;
  /** End offset (exclusive) into the chunk content, in code points. */
  end: number;
  /** 0-based index of the anchored paragraph in the chunk. */
  paragraph_idx: number;
  /** Total paragraphs detected in the chunk. */
  paragraph_count: number;
  /** Anchor confidence in [0, 1]; low-confidence anchors are not sent. */
  confidence: number;
  /** Anchoring method identifier (phase A: "lexical_overlap"). */
  method: string;
}

export interface RetrievalSource {
  /** Citation number, 1-indexed. Matches `{cite:n}` in the tokens. */
  n: number;
  type: SourceType;
  name: string;
  meta?: string | null;
  /** Retrieval metric when exposed by the backend; null/absent means unavailable. */
  score?: number | null;
  /** Path that grounded this source in the answer. */
  retrieval_origin?: 'vector' | 'graph' | null;
  /** Optional document id for direct drill-down from citations/sources. */
  doc_id?: string | null;
  /** Optional chunk id cited by the backend. */
  chunk_id?: string | null;
  /** Document-level provenance; never a server-side fetch target. */
  source_links?: readonly SourceLink[];
  /** Optional paragraph anchor inside the cited chunk. */
  anchor?: SourceAnchor | null;
}

export type ChatRole = 'user' | 'assistant';

/**
 * Mirrors the backend ``answer_status`` field on ``/twin/api/query``
 * (TR-RET-02). ``insufficient_information`` is the canonical
 * machine-readable signal that LightRAG had no usable context — the
 * React port uses it to suppress the Sources panel cleanly instead of
 * parsing the LLM prose. ``source_projection_failed`` means the answer IS
 * grounded but its references could not be projected into the sources
 * contract — the answer is shown, the Sources panel is suppressed, and a
 * "sources unavailable" cue is rendered (never silently as no-sources).
 * ``no_retrieval`` means no sourced final answer was requested (currently
 * ``only_need_context``): the empty Sources panel is the contract, not a
 * projection failure. Ungrounded ``bypass`` and prompt disclosure are not
 * part of the external Twin API.
 * ``query_failed`` means a backend error occurred mid-stream (the HTTP status
 * was already committed to 200): the answer text is an ``[query failed: …]``
 * error notice, NOT a grounded answer — the Sources panel is suppressed and no
 * citation affordance is rendered.
 * Default = ``grounded``.
 */
export type AnswerStatus =
  | 'grounded'
  | 'insufficient_information'
  | 'source_projection_failed'
  | 'no_retrieval'
  | 'query_failed';

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
  /** Assistant-only: immutable execution details captured for this answer. */
  queryMeta?: QueryRunMetadata;
}

export interface QueryRunMetadata {
  /** Deployment-provided LLM name. Absent on legacy backends. */
  model?: string;
  mode: QueryMode;
  topK: number;
  chunkTopK: number;
  enableRerank: boolean;
  /** Client-observed request duration, including streamed delivery. */
  durationMs: number;
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

export type QueryMode = 'naive' | 'local' | 'global' | 'hybrid' | 'mix';

export const QUERY_MODES: readonly QueryMode[] = [
  'naive',
  'local',
  'global',
  'hybrid',
  'mix',
];

export function stripTrailingReferencesSection(text: string): string {
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
    const re = /(?:\*\*([^*]+)\*\*|\[([^\]]+)]\((https?:\/\/[^)\s]+)\)|\*(\S(?:[^*\n]*\S)?)\*|(?<![\w])_(\S(?:[^_\n]*\S)?)_(?![\w])|\{cite:(\d+)\}|\[\^?(\d+)]|`([^`]+)`)/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) {
        parts.push({ type: 'text', value: text.slice(last, m.index) });
      }
      if (m[1]) {
        parts.push({ type: 'bold', value: m[1] });
      } else if (m[2] && m[3]) {
        parts.push({ type: 'link', label: m[2], href: m[3] });
      } else if (m[4] || m[5]) {
        parts.push({ type: 'italic', value: m[4] || m[5] });
      } else if (m[6] || m[7]) {
        parts.push({ type: 'cite', value: Number.parseInt(m[6] || m[7], 10) });
      } else if (m[8]) {
        parts.push({ type: 'code', value: m[8] });
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

  const parseTableRow = (line: string): string[] => {
    let value = line.trim();
    if (value.startsWith('|')) value = value.slice(1);
    if (value.endsWith('|')) value = value.slice(0, -1);
    return value.split('|').map((cell) => cell.trim());
  };
  const isTableDivider = (line: string): boolean => {
    const cells = parseTableRow(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  };

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const nextLine = lines[index + 1];

    if (!line.trim()) {
      if (out.length > 0 && out.at(-1)?.type !== 'paragraphBreak') {
        out.push({ type: 'paragraphBreak' });
      }
      index += 1;
      continue;
    }

    if (/^\s*```/.test(line)) {
      const language = line.trim().slice(3).trim() || undefined;
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      out.push({ type: 'codeBlock', language, value: code.join('\n') });
      continue;
    }

    if (line.includes('|') && nextLine !== undefined && isTableDivider(nextLine)) {
      const headers = parseTableRow(line).map(parseInline);
      const rows: InlineAnswerPart[][][] = [];
      index += 2;
      while (index < lines.length && lines[index].trimStart().startsWith('|')) {
        rows.push(parseTableRow(lines[index]).map(parseInline));
        index += 1;
      }
      out.push({ type: 'table', headers, rows });
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    const bullet = /^\s*[-*]\s+(.+)$/.exec(line);
    const ordered = /^\s*(\d+)\.\s+(.+)$/.exec(line);
    const quote = /^\s*>\s?(.*)$/.exec(line);
    if (heading) {
      out.push({
        type: 'heading',
        level: heading[1].length as 1 | 2 | 3,
        children: parseInline(heading[2]),
      });
    } else if (quote) {
      out.push({ type: 'blockquote', children: parseInline(quote[1]) });
    } else if (bullet || ordered) {
      out.push({
        type: 'listItem',
        ordered: Boolean(ordered),
        marker: ordered ? `${ordered[1]}.` : '•',
        children: parseInline(bullet?.[1] ?? ordered?.[2] ?? ''),
      });
    } else if (line) {
      out.push(...parseInline(line));
    }
    if (
      index < lines.length - 1 &&
      lines[index + 1].trim() &&
      !heading &&
      !bullet &&
      !ordered &&
      !quote
    ) {
      out.push({ type: 'lineBreak' });
    }
    index += 1;
  }
  while (out.at(-1)?.type === 'paragraphBreak') out.pop();
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
