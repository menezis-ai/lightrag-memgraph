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

// Versioned key — bumped to invalidate stale demo seeds when the seeded
// conversation shape changes (v2: full assistant answer + citations on the
// first seed thread; v1 had a "To restart RMAN…" stub that made the tab look
// broken on first paint; v3: invalidates fixture threads persisted by
// pre-production demo visits on the same origin — prod must boot blank).
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
  /** Host-controlled tab navigation for citation/source drill-downs. */
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

const missingRetrievalBackend: NonNullable<RetrievalTabProps['onSendQuery']> =
  async () => ({
    response: '⚠ Retrieval backend is not configured',
    sources: [],
  });

const DEFAULT_SUGGESTIONS = [
  'How do I restart Oracle on RHEL 9?',
  'Common RMAN backup errors',
  'CFT troubleshooting checklist',
];

interface ConversationHistoryMessage {
  role: ChatMessage['role'];
  content: string;
}

export function RetrievalTab({
  onSendQuery,
  onStreamQuery,
  initialThreads = [],
  suggestions = DEFAULT_SUGGESTIONS,
  onNavigate,
}: RetrievalTabProps) {
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
            : message.tokens?.join('');
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
      onNavigate?.('documents', { q: source.name });
    }
    setTimeout(() => setHighlightSrc(null), 1400);
  };
  const onSourceClick = onNavigate
    ? (source: RetrievalSource) => {
        // Prefer the exact source filter when the backend gave a doc id
        // or a file-path-shaped name; fall back to a search by name so
        // confluence / URL-shaped sources still resolve.
        const looksLikePath =
          source.type === 'file' && source.name.includes('.');
        const params: Record<string, string> = looksLikePath
          ? { source: source.name }
          : { q: source.name };
        onNavigate('documents', params);
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
              role="button"
              tabIndex={0}
              onClick={() => setActiveThreadId(t.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setActiveThreadId(t.id);
                }
              }}
              aria-current={t.id === activeThreadId ? 'true' : undefined}
              aria-label={`Open conversation ${t.title}`}
              data-testid={`thread-${t.id}`}
            >
              <div className="history-item-title" title={t.title}>
                {t.title}
              </div>
              <div className="history-item-meta">
                <span>
                  {t.messages.filter((m) => m.role === 'user').length} q ·{' '}
                  {t.messages.filter((m) => m.role === 'assistant').length} a
                </span>
                <span className="sep">·</span>
                <span>{relTime(t.updated)}</span>
              </div>
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

        {/* TR-RET-02 step 3 / audit C1: the "Tag filter — Twin"
            input used to live here and forwarded ``tagFilters`` to
            the backend, where LightRAG 1.4.x silently ignored it
            (its ``QueryParam`` has no ``tag_filter`` field). The
            affordance has been removed entirely rather than
            relabelled, because there is no honest backend path to
            redirect to while audit C2 (the /query/data post-filter
            still on metadata.tags instead of TAGGED_WITH) is open.
            Restoring this control is gated on a real server-side
            pre-filter — see ``docs/audits/lightrag-interactions/``. */}

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
}: TurnProps) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  if (msg.role === 'user') {
    return <div className="msg-user">{msg.text}</div>;
  }

  const parts: AnswerPart[] = parseAnswer(msg.tokens ?? []);
  const sources = msg.sources ?? [];
  const requestedTopK = msg.requestedTopK ?? sources.length;
  const visibleSources = sourcesExpanded
    ? sources
    : sources.slice(0, INITIAL_VISIBLE_SOURCES);
  const hiddenSourcesCount = Math.max(0, requestedTopK - INITIAL_VISIBLE_SOURCES);
  const realHiddenSourcesCount = Math.max(0, sources.length - INITIAL_VISIBLE_SOURCES);
  const noAdditionalReturned =
    sourcesExpanded && hiddenSourcesCount > 0 && realHiddenSourcesCount === 0;

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
                  ? `Réduire aux ${INITIAL_VISIBLE_SOURCES} premières`
                  : `Voir les ${hiddenSourcesCount} autres`}
              </button>
            )}
            {noAdditionalReturned && (
              <div
                className="sources-empty muted"
                data-testid="sources-no-additional"
              >
                No additional structured sources were returned by the backend.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
