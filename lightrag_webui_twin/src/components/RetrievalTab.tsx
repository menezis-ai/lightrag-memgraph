/**
 * RetrievalTab — conversation + streamed answer with inline citations,
 * thread sidebar, parameters panel (Twin extras: tag filter + Twin params).
 *
 * Ported from Desktop/UI/retrieval.jsx. Scope:
 *   - History panel (new / switch / delete threads, localStorage persistence)
 *   - Main conversation with streamed tokens + clickable citations
 *   - Parameters panel (query mode, tag filter, top-k, max-tok, history,
 *     only-context, only-prompt)
 *
 * Behavior delta vs the proto:
 *   - tag catalog injected via prop (no window globals)
 *   - Real-backend callbacks drive assistant responses; tests inject
 *     callbacks when they need deterministic answer content.
 *   - The streaming timer (70ms/token in the proto) is preserved; tests use
 *     vi.useFakeTimers to drive it.
 *   - LocalStorage key matches the proto: "twin-rag.threads".
 *   - URL state via useUrlParam / useUrlArrayParam / useUrlNumberParam.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon, SourceIcon } from './Icon';
import {
  useUrlArrayParam,
  useUrlNumberParam,
  useUrlParam,
} from '../hooks/useUrlParam';
import {
  parseAnswer,
  QUERY_MODES,
  relTime,
  type AnswerPart,
  type InlineAnswerPart,
  type AnswerStatus,
  type AnswerToken,
  type ChatMessage,
  type QueryMode,
  type RetrievalSource,
  type RetrievalThread,
} from '../types/retrieval';

// Versioned key — bumped to invalidate stale pre-production local threads.
const THREADS_STORAGE_KEY = 'twin-rag.threads.v3';
const STREAM_TICK_MS = 70;
const INITIAL_VISIBLE_SOURCES = 5;

const makeThreadId = () => 'th_' + Math.random().toString(16).slice(2, 8);

export interface RetrievalTabProps {
  /** Real-backend callback. The returned `response` string is split into
   *  whitespace tokens and streamed via the existing animator. Sources are
   *  passed through as-is. */
  onSendQuery?: (params: {
    query: string;
    mode: QueryMode;
    topK: number;
    chunkTopK: number;
    maxTokens: number;
    historyTurns: number;
    conversationHistory: readonly ConversationHistoryMessage[];
    onlyContext: boolean;
    onlyPrompt: boolean;
    userPrompt: string;
    enableRerank: boolean;
    minScore: number;
    tagFilter?: RetrievalAdvancedFilter;
    docFilter?: RetrievalAdvancedFilter;
  }) => Promise<{
    response: string;
    sources?: readonly RetrievalSource[];
    /** TR-RET-02: propagated from the backend ``answer_status`` field
     *  so the host can suppress the Sources panel on
     *  ``insufficient_information`` answers. */
    answer_status?: AnswerStatus;
  }>;
  onStreamQuery?: (
    params: {
      query: string;
      mode: QueryMode;
      topK: number;
      chunkTopK: number;
      maxTokens: number;
      historyTurns: number;
      conversationHistory: readonly ConversationHistoryMessage[];
      onlyContext: boolean;
      onlyPrompt: boolean;
      userPrompt: string;
      enableRerank: boolean;
      minScore: number;
      tagFilter?: RetrievalAdvancedFilter;
      docFilter?: RetrievalAdvancedFilter;
    },
    onChunk: (chunk: string) => void,
  ) => Promise<{
    response: string;
    sources?: readonly RetrievalSource[];
    answer_status?: AnswerStatus;
  }>;
  /** Seed threads when localStorage is empty. */
  initialThreads?: readonly RetrievalThread[];
  /** Suggestions displayed in the empty state. */
  suggestions?: readonly string[];
  /** Canonical tags for source filtering. */
  tagOptions?: readonly string[];
  /** Document ids or source paths for source filtering. */
  docOptions?: readonly string[];
  /** Optional doc id -> display label. */
  docLabels?: Readonly<Record<string, string>>;
  /** Host-controlled tab navigation for citation/source drill-downs. */
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

const missingRetrievalBackend: NonNullable<RetrievalTabProps['onSendQuery']> =
  async () => ({
    response: '⚠ Retrieval backend is not configured',
    sources: [],
  });

const DEFAULT_SUGGESTIONS = [
  'Ask about a source in this folder',
  'Summarize recent indexed documents',
  'Find operational procedures by tag',
];

function splitThinkBlocks(text: string): { visible: string; thoughts: readonly string[] } {
  const thoughts: string[] = [];
  const visible = text
    .replace(/<think\b[^>]*>([\s\S]*?)(?:<\/think>|$)/gi, (_match, thought) => {
      const trimmed = String(thought ?? '').trim();
      if (trimmed) thoughts.push(trimmed);
      return '';
    })
    .replace(/<\/think>/gi, '')
    .trimStart();
  return { visible, thoughts };
}

interface ConversationHistoryMessage {
  role: ChatMessage['role'];
  content: string;
}

interface RetrievalAdvancedFilter {
  all?: readonly string[];
  any?: readonly string[];
}

type FilterMode = 'any' | 'all';

function filterPayload(
  selected: readonly string[],
  mode: FilterMode,
): RetrievalAdvancedFilter | undefined {
  if (selected.length === 0) return undefined;
  return mode === 'all' ? { all: selected } : { any: selected };
}

function sourceDrilldownParams(source: RetrievalSource): Record<string, string> {
  const looksLikePath = source.type === 'file' && source.name.includes('.');
  const params: Record<string, string> = looksLikePath
    ? { source: source.name }
    : { q: source.name };
  if (source.doc_id) params.doc = source.doc_id;
  if (source.chunk_id) params.chunk = source.chunk_id;
  return params;
}

export function RetrievalTab({
  onSendQuery,
  onStreamQuery,
  initialThreads = [],
  suggestions = DEFAULT_SUGGESTIONS,
  tagOptions = [],
  docOptions = [],
  docLabels,
  onNavigate,
}: Readonly<RetrievalTabProps>) {
  const [query, setQuery] = useState('');
  const [threads, setThreads] = useState<readonly RetrievalThread[]>(() => {
    try {
      const raw = localStorage.getItem(THREADS_STORAGE_KEY);
      if (raw) return JSON.parse(raw) as RetrievalThread[];
    } catch {
      /* ignore */
    }
    return initialThreads;
  });
  const [activeThreadId, setActiveThreadId] = useState<string | null>(
    () => initialThreads[0]?.id ?? null,
  );
  const [streaming, setStreaming] = useState(false);
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(null);
  const [streamedTokens, setStreamedTokens] = useState<readonly AnswerToken[]>([]);
  const [highlightSrc, setHighlightSrc] = useState<number | null>(null);

  const [queryMode, setQueryMode] = useUrlParam<QueryMode>('mode', 'mix', {
    validate: (v) => QUERY_MODES.includes(v as QueryMode),
  });
  const [topK, setTopK] = useUrlNumberParam('topk', 20);
  const [chunkTopK, setChunkTopK] = useUrlNumberParam('chunktopk', 20);
  const [maxTok, setMaxTok] = useUrlNumberParam('maxtok', 30000);
  const [history, setHistory] = useUrlNumberParam('hist', 3);
  const [minScore, setMinScore] = useUrlNumberParam('minscore', 0);
  const [tagFilter, setTagFilter] = useUrlArrayParam('rtag', []);
  const [docFilter, setDocFilter] = useUrlArrayParam('rdoc', []);
  const [tagFilterMode, setTagFilterMode] = useUrlParam<FilterMode>(
    'rtagmode',
    'any',
    { validate: (v) => v === 'any' || v === 'all' },
  );
  const [docFilterMode, setDocFilterMode] = useUrlParam<FilterMode>(
    'rdocmode',
    'any',
    { validate: (v) => v === 'any' || v === 'all' },
  );
  const [onlyCtx, setOnlyCtx] = useState(false);
  const [onlyPrompt, setOnlyPrompt] = useState(false);
  const [userPrompt, setUserPrompt] = useState('');
  const [enableRerank, setEnableRerank] = useState(true);

  const convRef = useRef<HTMLDivElement>(null);

  const activeThread = threads.find((t) => t.id === activeThreadId);
  const convo = useMemo<readonly ChatMessage[]>(
    () => activeThread?.messages ?? [],
    [activeThread],
  );

  // Persist threads to localStorage.
  useEffect(() => {
    try {
      localStorage.setItem(THREADS_STORAGE_KEY, JSON.stringify(threads));
    } catch {
      /* ignore quota */
    }
  }, [threads]);

  useEffect(() => {
    if (convRef.current) {
      convRef.current.scrollTop = convRef.current.scrollHeight;
    }
  }, [streamedTokens, convo]);

  // Resolve the thread that will receive the exchange ONCE, at send time.
  // The id then travels through the async response flow explicitly —
  // reading `activeThreadId` from a callback closure created the
  // "question in one thread, answer in a fresh New thread" split (the
  // closure still saw the pre-send null/stale id).
  const ensureActiveThread = (): string => {
    if (activeThreadId && threads.find((t) => t.id === activeThreadId)) {
      return activeThreadId;
    }
    const id = makeThreadId();
    setThreads((ts) => [
      {
        id,
        title: 'New thread',
        created: Date.now(),
        updated: Date.now(),
        messages: [],
      },
      ...ts,
    ]);
    setActiveThreadId(id);
    return id;
  };

  const appendToThread = (
    id: string,
    updater: (msgs: readonly ChatMessage[]) => readonly ChatMessage[],
  ) => {
    setThreads((ts) => {
      // Thread deleted mid-flight: recreate it so the exchange is kept.
      const arr = ts.find((t) => t.id === id)
        ? ts
        : [
            {
              id,
              title: 'New thread',
              created: Date.now(),
              updated: Date.now(),
              messages: [] as readonly ChatMessage[],
            },
            ...ts,
          ];
      return arr.map((t) => {
        if (t.id !== id) return t;
        const nextMsgs = updater(t.messages);
        const firstUser = nextMsgs.find((m) => m.role === 'user');
        const newTitle =
          t.messages.length === 0 && firstUser?.text
            ? firstUser.text.slice(0, 64)
            : t.title;
        return {
          ...t,
          updated: Date.now(),
          messages: nextMsgs,
          title: newTitle,
        };
      });
    });
  };

  const newThread = () => {
    const id = makeThreadId();
    setThreads((ts) => [
      {
        id,
        title: 'New thread',
        created: Date.now(),
        updated: Date.now(),
        messages: [],
      },
      ...ts,
    ]);
    setActiveThreadId(id);
    setStreamedTokens([]);
    setStreamingThreadId(null);
    setStreaming(false);
  };

  const deleteThread = (id: string) => {
    setThreads((ts) => {
      const next = ts.filter((t) => t.id !== id);
      if (id === activeThreadId) {
        setActiveThreadId(next[0]?.id ?? null);
      }
      return next;
    });
  };

  const streamTokens = (
    threadId: string,
    tokens: readonly AnswerToken[],
    sources: readonly RetrievalSource[],
    answerStatus: AnswerStatus = 'grounded',
    requestedTopK?: number,
  ) => {
    if (tokens.length === 0) {
      setStreaming(false);
      setStreamingThreadId(null);
      appendToThread(threadId, (c) => [
        ...c,
        { role: 'assistant', tokens: [], sources, answerStatus, requestedTopK },
      ]);
      return;
    }
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setStreamedTokens(tokens.slice(0, i));
      if (i >= tokens.length) {
        clearInterval(interval);
        setStreaming(false);
        setStreamingThreadId(null);
        appendToThread(threadId, (c) => [
          ...c,
          { role: 'assistant', tokens, sources, answerStatus, requestedTopK },
        ]);
        setStreamedTokens([]);
      }
    }, STREAM_TICK_MS);
  };

  const conversationHistoryFor = (
    messages: readonly ChatMessage[],
  ): readonly ConversationHistoryMessage[] => {
    const maxMessages = Math.max(0, history) * 2;
    if (maxMessages === 0) return [];
    return messages
      .map((message): ConversationHistoryMessage | null => {
        const content =
          message.role === 'user'
            ? message.text
            : splitThinkBlocks(message.tokens?.join('') ?? '').visible;
        const trimmed = content?.trim();
        if (!trimmed) return null;
        return { role: message.role, content: trimmed };
      })
      .filter(
        (message): message is ConversationHistoryMessage => message !== null,
      )
      .slice(-maxMessages);
  };

  const activeParams = (
    q: string,
    conversationHistory: readonly ConversationHistoryMessage[],
  ) => ({
    query: q,
    mode: queryMode,
    topK,
    chunkTopK,
    maxTokens: maxTok,
    historyTurns: history,
    conversationHistory,
    onlyContext: onlyCtx,
    onlyPrompt,
    userPrompt,
    enableRerank,
    minScore,
    tagFilter: filterPayload(tagFilter, tagFilterMode),
    docFilter: filterPayload(docFilter, docFilterMode),
  });

  const send = (text?: string) => {
    const q = (text ?? query).trim();
    if (!q) return;
    setQuery('');
    const threadId = ensureActiveThread();
    const currentMessages =
      threads.find((t) => t.id === threadId)?.messages ?? [];
    const conversationHistory = conversationHistoryFor(currentMessages);
    const requestedTopK = topK;
    appendToThread(threadId, (c) => [...c, { role: 'user', text: q }]);
    setStreamedTokens([]);
    setStreamingThreadId(threadId);
    setStreaming(true);

    if (onStreamQuery) {
      const streamed: AnswerToken[] = [];
      onStreamQuery(activeParams(q, conversationHistory), (chunk) => {
        streamed.push(chunk);
        setStreamedTokens([...streamed]);
      })
        .then(({ sources, answer_status }) => {
          setStreaming(false);
          const finalTokens = streamed.join('')
            .split(/(\s+)/)
            .filter((t) => t.length > 0);
          const status: AnswerStatus = answer_status ?? 'grounded';
          // TR-RET-02: when the backend signalled insufficient context,
          // drop any sources that slipped through (defensive — the
          // backend already returns []). Prevents a future backend
          // regression from showing sources behind an unfounded answer.
          const effectiveSources =
            status === 'insufficient_information' ? [] : (sources ?? []);
          appendToThread(threadId, (c) => [
            ...c,
            {
              role: 'assistant',
              tokens: finalTokens,
              sources: effectiveSources,
              answerStatus: status,
              requestedTopK,
            },
          ]);
          setStreamedTokens([]);
          setStreamingThreadId(null);
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : 'Query failed';
          streamTokens(threadId, [`⚠ ${msg}`], [], 'grounded', requestedTopK);
        });
      return;
    }

    const sendQuery = onSendQuery ?? missingRetrievalBackend;

    sendQuery(activeParams(q, conversationHistory))
      .then(({ response, sources, answer_status }) => {
        const tokens = response
          .split(/(\s+)/)
          .filter((t) => t.length > 0);
        const status: AnswerStatus = answer_status ?? 'grounded';
        const effectiveSources =
          status === 'insufficient_information' ? [] : (sources ?? []);
        if (tokens.length === 0) {
          streamTokens(
            threadId,
            ['⚠ The backend returned an empty answer. Sources below.'],
            effectiveSources,
            status,
            requestedTopK,
          );
          return;
        }
        streamTokens(threadId, tokens, effectiveSources, status, requestedTopK);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Query failed';
        streamTokens(threadId, [`⚠ ${msg}`], [], 'grounded', requestedTopK);
      });
  };

  const onCiteHover = (n: number) => setHighlightSrc(n);
  const onCiteLeave = () => setTimeout(() => setHighlightSrc(null), 200);
  const onCiteClick = (
    n: number,
    sources: readonly RetrievalSource[] | undefined,
  ) => {
    setHighlightSrc(n);
    const source = sources?.find((s) => s.n === n);
    if (source) {
      onNavigate?.('documents', sourceDrilldownParams(source));
    }
    setTimeout(() => setHighlightSrc(null), 1400);
  };
  const onSourceClick = onNavigate
    ? (source: RetrievalSource) => {
        onNavigate('documents', sourceDrilldownParams(source));
      }
    : undefined;

  return (
    <div className="retrieval has-history">
      <aside className="history-panel">
        <div className="history-head">
          <span className="history-title">Conversations</span>
          <button
            type="button"
            className="history-new"
            onClick={newThread}
            title="New conversation"
          >
            <Icon name="plus" size={12} /> New
          </button>
        </div>
        <ul className="history-list">
          {threads.length === 0 && (
            <li className="history-empty">No conversations yet</li>
          )}
          {threads.map((t) => (
            <li
              key={t.id}
              className={
                'history-item' +
                (t.id === activeThreadId ? ' is-active' : '')
              }
            >
              <button
                type="button"
                className={
                  'history-item-main' +
                  (t.id === activeThreadId ? ' is-active' : '')
                }
                onClick={() => setActiveThreadId(t.id)}
                aria-current={t.id === activeThreadId ? 'true' : undefined}
                aria-label={`Open conversation ${t.title}`}
                data-testid={`thread-${t.id}`}
              >
                <span className="history-item-title" title={t.title}>
                  {t.title}
                </span>
                <span className="history-item-meta">
                  <span>
                    {t.messages.filter((m) => m.role === 'user').length} q ·{' '}
                    {t.messages.filter((m) => m.role === 'assistant').length} a
                  </span>
                  <span className="sep">·</span>
                  <span>{relTime(t.updated)}</span>
                </span>
              </button>
              <button
                type="button"
                className="history-del"
                title="Delete"
                aria-label={`Delete ${t.title}`}
                onClick={(e) => {
                  e.stopPropagation();
                  deleteThread(t.id);
                }}
              >
                <Icon name="x" size={11} />
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <div className="retrieval-main">
        <div className="retrieval-conv" ref={convRef}>
          {convo.length === 0 && !streaming && (
            <div className="empty-state">
              <Icon
                name="search"
                size={28}
                color="var(--color-text-tertiary)"
              />
              <div className="title">
                Ask a question to retrieve from the knowledge base
              </div>
              <div
                style={{
                  display: 'flex',
                  gap: 8,
                  flexWrap: 'wrap',
                  justifyContent: 'center',
                  marginTop: 8,
                }}
              >
                {suggestions.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="suggestion"
                    onClick={() => send(s)}
                  >
                    Try: "{s}"
                  </button>
                ))}
              </div>
            </div>
          )}
          {convo.map((m, i) => (
            <Turn
              key={i}
              msg={m}
              highlightSrc={highlightSrc}
              onCiteHover={onCiteHover}
              onCiteLeave={onCiteLeave}
              onCiteClick={onCiteClick}
              onSourceClick={onSourceClick}
            />
          ))}
          {streaming &&
            activeThreadId === streamingThreadId &&
            streamedTokens.length > 0 && (
            <Turn
              streaming
              msg={{
                role: 'assistant',
                tokens: streamedTokens,
                sources: [],
              }}
              highlightSrc={highlightSrc}
              onCiteHover={onCiteHover}
              onCiteLeave={onCiteLeave}
              onCiteClick={onCiteClick}
              onSourceClick={onSourceClick}
            />
          )}
          {streaming &&
            activeThreadId === streamingThreadId &&
            streamedTokens.length === 0 && <ThinkingTurn />}
        </div>
        <div className="querybar">
          <textarea
            placeholder="Type your query…"
            aria-label="Query input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button
            type="button"
            className="btn primary"
            onClick={() => send()}
            disabled={streaming}
          >
            <Icon name="send" size={13} /> Send
          </button>
        </div>
      </div>

      <aside className="params-panel">
        <div className="params-header">
          <h3>Parameters</h3>
          <p>Configure your query</p>
        </div>

        <div className="field">
          <label className="field-label">Query mode</label>
          <select
            aria-label="Query mode"
            value={queryMode}
            onChange={(e) => setQueryMode(e.target.value as QueryMode)}
          >
            {QUERY_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="field retrieval-filter-field">
          <label className="field-label">Source tag filters</label>
          <RetrievalFilterPicker
            label="Retrieval tag filter"
            options={tagOptions}
            selected={tagFilter}
            onChange={setTagFilter}
            placeholder="Add tag filter…"
          />
          <FilterModeToggle
            label="Retrieval tag filter mode"
            value={tagFilterMode}
            onChange={setTagFilterMode}
            disabled={tagFilter.length < 2}
          />
        </div>

        <div className="field retrieval-filter-field">
          <label className="field-label">Source document filters</label>
          <RetrievalFilterPicker
            label="Retrieval document filter"
            options={docOptions}
            selected={docFilter}
            onChange={setDocFilter}
            placeholder="Add document filter…"
            format={(id) => docLabels?.[id] ?? id}
          />
          <FilterModeToggle
            label="Retrieval document filter mode"
            value={docFilterMode}
            onChange={setDocFilterMode}
            disabled={docFilter.length < 2}
          />
        </div>

        <div className="field">
          <label className="field-label">Top K results</label>
          <input
            type="number"
            aria-label="Top K"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value || '0', 10))}
          />
        </div>
        <div className="field">
          <label className="field-label">Chunk top K</label>
          <input
            type="number"
            aria-label="Chunk top K"
            value={chunkTopK}
            onChange={(e) => setChunkTopK(parseInt(e.target.value || '0', 10))}
          />
        </div>
        <div className="field">
          <label className="field-label">Max tokens · text unit</label>
          <input
            type="number"
            aria-label="Max tokens"
            value={maxTok}
            onChange={(e) => setMaxTok(parseInt(e.target.value || '0', 10))}
          />
        </div>
        <div className="field">
          <label className="field-label">Minimum source score</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            aria-label="Minimum source score"
            value={minScore}
            onChange={(e) => {
              const next = Number(e.target.value || '0');
              setMinScore(Math.min(1, Math.max(0, Number.isFinite(next) ? next : 0)));
            }}
          />
        </div>
        <div className="field">
          <label className="field-label">History turns</label>
          <input
            type="number"
            aria-label="History turns"
            value={history}
            onChange={(e) => setHistory(parseInt(e.target.value || '0', 10))}
          />
        </div>
        <div className="field">
          <label className="field-label">System prompt</label>
          <textarea
            aria-label="System prompt"
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            rows={3}
            placeholder="Optional retrieval instruction"
          />
        </div>
        <div className="toggle">
          <span
            className={`switch${enableRerank ? ' on' : ''}`}
            onClick={() => setEnableRerank(!enableRerank)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setEnableRerank((v) => !v);
              }
            }}
            role="switch"
            tabIndex={0}
            aria-checked={enableRerank}
            aria-label="Enable rerank"
          />
          Enable rerank
        </div>
        <div className="toggle">
          <span
            className={`switch${onlyCtx ? ' on' : ''}`}
            onClick={() => setOnlyCtx(!onlyCtx)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setOnlyCtx((v) => !v);
              }
            }}
            role="switch"
            tabIndex={0}
            aria-checked={onlyCtx}
            aria-label="Only need context"
          />
          Only need context
        </div>
        <div className="toggle">
          <span
            className={`switch${onlyPrompt ? ' on' : ''}`}
            onClick={() => setOnlyPrompt(!onlyPrompt)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setOnlyPrompt((v) => !v);
              }
            }}
            role="switch"
            tabIndex={0}
            aria-checked={onlyPrompt}
            aria-label="Only need prompt"
          />
          Only need prompt
        </div>

        <div className="connected">
          <span className="dot" /> Connected
        </div>
      </aside>
    </div>
  );
}

function ThinkingTurn() {
  return (
    <div
      className="msg-assistant thinking-turn"
      role="status"
      aria-live="polite"
      data-testid="retrieval-thinking"
    >
      <div className="thinking-bubble" aria-label="Thinking">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

function collectCitedSourceNumbers(parts: readonly AnswerPart[]): Set<number> {
  const out = new Set<number>();
  parts.forEach((part) => {
    if (part.type === 'cite') {
      out.add(part.value);
    } else if (part.type === 'heading' || part.type === 'listItem') {
      part.children.forEach((child) => {
        if (child.type === 'cite') out.add(child.value);
      });
    }
  });
  return out;
}

function collapsedSources(
  sources: readonly RetrievalSource[],
  citedSourceNumbers: ReadonlySet<number>,
): readonly RetrievalSource[] {
  const visibleNumbers = new Set<number>();
  const visible: RetrievalSource[] = [];
  const pushOnce = (source: RetrievalSource) => {
    if (visibleNumbers.has(source.n)) return;
    visibleNumbers.add(source.n);
    visible.push(source);
  };

  sources.slice(0, INITIAL_VISIBLE_SOURCES).forEach(pushOnce);
  sources.filter((source) => citedSourceNumbers.has(source.n)).forEach(pushOnce);
  return visible;
}

interface TurnProps {
  msg: ChatMessage;
  streaming?: boolean;
  highlightSrc: number | null;
  onCiteHover: (n: number) => void;
  onCiteLeave: () => void;
  onCiteClick: (n: number, sources: readonly RetrievalSource[] | undefined) => void;
  /** Click on a source card in the sidebar — host navigates to docs. */
  onSourceClick?: (source: RetrievalSource) => void;
}

function Turn({
  msg,
  streaming,
  highlightSrc,
  onCiteHover,
  onCiteLeave,
  onCiteClick,
  onSourceClick,
}: Readonly<TurnProps>) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  if (msg.role === 'user') {
    return <div className="msg-user">{msg.text}</div>;
  }

  const answerText = (msg.tokens ?? []).join('');
  const { visible, thoughts } = splitThinkBlocks(answerText);
  const parts: AnswerPart[] = parseAnswer([visible]);
  const sources = msg.sources ?? [];
  const citedSourceNumbers = collectCitedSourceNumbers(parts);
  const collapsedVisibleSources = collapsedSources(sources, citedSourceNumbers);
  const visibleSources = sourcesExpanded
    ? sources
    : collapsedVisibleSources;
  const collapsedVisibleSourceNumbers = new Set(
    collapsedVisibleSources.map((source) => source.n),
  );
  const hiddenSourcesCount = sources.filter(
    (source) => !collapsedVisibleSourceNumbers.has(source.n),
  ).length;

  const renderInlineParts = (inlineParts: readonly InlineAnswerPart[]) =>
    inlineParts.map((p, i) => {
      if (p.type === 'text') return <span key={i}>{p.value}</span>;
      if (p.type === 'bold') return <strong key={i}>{p.value}</strong>;
      if (p.type === 'code') return <code key={i}>{p.value}</code>;
      return (
        <button
          key={i}
          type="button"
          className="citation"
          onMouseEnter={() => onCiteHover(p.value)}
          onMouseLeave={onCiteLeave}
          onClick={() => onCiteClick(p.value, msg.sources)}
          aria-label={`Source ${p.value}`}
          data-testid={`citation-${p.value}`}
        >
          {p.value}
        </button>
      );
    });

  return (
    <div className="msg-assistant">
      <div className="msg-text">
        {parts.map((p, i) => {
          if (p.type === 'lineBreak') return <br key={i} />;
          if (p.type === 'heading') {
            const Tag = `h${p.level}` as 'h1' | 'h2' | 'h3';
            return (
              <Tag key={i} className={`answer-heading answer-heading-${p.level}`}>
                {renderInlineParts(p.children)}
              </Tag>
            );
          }
          if (p.type === 'listItem') {
            return (
              <div
                key={i}
                className={`answer-list-item${p.ordered ? ' ordered' : ''}`}
              >
                <span className="answer-list-marker">
                  {p.ordered ? '1.' : '•'}
                </span>
                <span>{renderInlineParts(p.children)}</span>
              </div>
            );
          }
          return renderInlineParts([p]);
        })}
        {streaming && (
          <span
            className="cursor"
            style={{
              display: 'inline-block',
              width: 6,
              height: 14,
              background: 'var(--twin-accent)',
              verticalAlign: '-2px',
              marginLeft: 2,
              animation: 'blink 1s infinite',
            }}
          />
        )}
      </div>
      {thoughts.length > 0 && (
        <details className="reasoning-reveal" data-testid="retrieval-thinking-detail">
          <summary>
            <Icon name="info-circle" size={12} /> Reasoning
          </summary>
          <pre>{thoughts.join('\n\n')}</pre>
        </details>
      )}
      {/* TR-RET-02: when the backend marked the answer
          ``insufficient_information`` (LightRAG fail_response with the
          ``[no-context]`` marker), do NOT render the Sources panel even
          if sources slipped through. Show a discrete cue instead so the
          operator understands the absence, rather than reading an
          empty area as a layout glitch. */}
      {!streaming && msg.answerStatus === 'insufficient_information' && (
        <div
          className="sources-empty muted"
          data-testid="sources-empty-insufficient"
          style={{ marginTop: 8, fontSize: 12 }}
        >
          No relevant sources found for this question.
        </div>
      )}
      {!streaming &&
        msg.answerStatus !== 'insufficient_information' &&
        sources.length > 0 && (
        <>
          <div className="sources-header">
            Sources
            {msg.requestedTopK !== undefined && (
              <span className="sources-count" data-testid="sources-count">
                {sources.length} returned / {msg.requestedTopK} requested
              </span>
            )}
          </div>
          <div className="sources-list">
            {visibleSources.map((s) => {
              const clickable = Boolean(onSourceClick);
              const className = `source-card${highlightSrc === s.n ? ' hl' : ''}${
                clickable ? ' clickable' : ''
              }`;
              const handleClick = () => onSourceClick?.(s);
              return (
                <button
                  key={s.n}
                  type="button"
                  id={`src-${s.n}`}
                  className={className}
                  data-testid={`source-${s.n}`}
                  onClick={clickable ? handleClick : undefined}
                  disabled={!clickable}
                  title={clickable ? `Open ${s.name}` : undefined}
                  style={{
                    background: 'none',
                    textAlign: 'left',
                    font: 'inherit',
                    cursor: clickable ? 'pointer' : 'default',
                  }}
                >
                  <span className="src-pill">{s.n}</span>
                  <SourceIcon type={s.type} size={13} />
                  <span
                    className={s.type !== 'file' ? 'src-name mono' : 'src-name'}
                  >
                    {s.name}
                  </span>
                  {s.meta && <span className="src-meta">{s.meta}</span>}
                  <span className="src-score">{s.score.toFixed(2)}</span>
                  <span className="src-ext" title="Open source">
                    <Icon name="external-link" size={12} />
                  </span>
                </button>
              );
            })}
            {hiddenSourcesCount > 0 && (
              <button
                type="button"
                className="sources-toggle"
                onClick={() => setSourcesExpanded((expanded) => !expanded)}
                aria-expanded={sourcesExpanded}
              >
                {sourcesExpanded
                  ? 'Réduire aux sources principales'
                  : `Voir les ${hiddenSourcesCount} autres`}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

interface RetrievalFilterPickerProps {
  label: string;
  options: readonly string[];
  selected: readonly string[];
  onChange: (next: readonly string[]) => void;
  placeholder: string;
  format?: (value: string) => string;
}

function RetrievalFilterPicker({
  label,
  options,
  selected,
  onChange,
  placeholder,
  format = (value) => value,
}: Readonly<RetrievalFilterPickerProps>) {
  const [draft, setDraft] = useState('');
  const available = useMemo(() => {
    const values = new Set<string>();
    options.forEach((value) => values.add(value));
    selected.forEach((value) => values.add(value));
    return Array.from(values).sort((a, b) => format(a).localeCompare(format(b)));
  }, [format, options, selected]);
  const datalistId = `${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-options`;

  const addValue = (raw: string) => {
    const value = raw.trim();
    if (!value || selected.includes(value)) return;
    onChange([...selected, value]);
    setDraft('');
  };

  return (
    <div className="retrieval-filter-picker">
      <div className="retrieval-filter-input-row">
        <input
          className="mini-input"
          list={datalistId}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addValue(draft);
            }
          }}
          placeholder={placeholder}
          aria-label={label}
        />
        <datalist id={datalistId}>
          {available.map((value) => (
            <option key={value} value={value}>
              {format(value)}
            </option>
          ))}
        </datalist>
        <button
          type="button"
          className="ghost-btn small"
          onClick={() => addValue(draft)}
          disabled={!draft.trim() || selected.includes(draft.trim())}
        >
          <Icon name="plus" size={12} />
        </button>
      </div>
      {selected.length > 0 && (
        <div className="retrieval-filter-chips">
          {selected.map((value) => (
            <span key={value} className="retrieval-filter-chip">
              <span title={value}>{format(value)}</span>
              <button
                type="button"
                aria-label={`Remove ${format(value)}`}
                onClick={() => onChange(selected.filter((item) => item !== value))}
              >
                <Icon name="x" size={10} />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

interface FilterModeToggleProps {
  label: string;
  value: FilterMode;
  onChange: (value: FilterMode) => void;
  disabled: boolean;
}

function FilterModeToggle({
  label,
  value,
  onChange,
  disabled,
}: Readonly<FilterModeToggleProps>) {
  return (
    <fieldset
      className={`retrieval-filter-mode${disabled ? ' is-disabled' : ''}`}
    >
      <legend className="sr-only">{label}</legend>
      {(['any', 'all'] as const).map((mode) => (
        <button
          key={mode}
          type="button"
          className={value === mode ? 'is-on' : ''}
          onClick={() => onChange(mode)}
          disabled={disabled}
        >
          {mode === 'any' ? 'Any' : 'All'}
        </button>
      ))}
    </fieldset>
  );
}
