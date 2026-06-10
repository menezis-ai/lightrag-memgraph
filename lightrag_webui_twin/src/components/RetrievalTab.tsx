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
 *   - thesaurus injected via prop (no window.MOCK_THESAURUS)
 *   - Real-backend callbacks drive assistant responses; tests inject
 *     callbacks when they need deterministic answer content.
 *   - The streaming timer (70ms/token in the proto) is preserved; tests use
 *     vi.useFakeTimers to drive it.
 *   - LocalStorage key matches the proto: "twin-rag.threads".
 *   - URL state via useUrlParam / useUrlArrayParam / useUrlNumberParam.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
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
  type AnswerToken,
  type ChatMessage,
  type QueryMode,
  type RetrievalSource,
  type RetrievalThread,
} from '../types/retrieval';
import type { ThesaurusEntry } from '../types/thesaurus';

// Versioned key — bumped to invalidate stale demo seeds when the seeded
// conversation shape changes (v2: full assistant answer + citations on the
// first seed thread; v1 had a "To restart RMAN…" stub that made the tab look
// broken on first paint; v3: invalidates fixture threads persisted by
// pre-production demo visits on the same origin — prod must boot blank).
const THREADS_STORAGE_KEY = 'twin-rag.threads.v3';
const STREAM_TICK_MS = 70;

export interface RetrievalTabProps {
  thesaurus: readonly ThesaurusEntry[];
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
    onlyContext: boolean;
    onlyPrompt: boolean;
    userPrompt: string;
    enableRerank: boolean;
    tagFilters: readonly string[];
  }) => Promise<{
    response: string;
    sources?: readonly RetrievalSource[];
  }>;
  onStreamQuery?: (
    params: {
      query: string;
      mode: QueryMode;
      topK: number;
      chunkTopK: number;
      maxTokens: number;
      historyTurns: number;
      onlyContext: boolean;
      onlyPrompt: boolean;
      userPrompt: string;
      enableRerank: boolean;
      tagFilters: readonly string[];
    },
    onChunk: (chunk: string) => void,
  ) => Promise<{
    response: string;
    sources?: readonly RetrievalSource[];
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

export function RetrievalTab({
  thesaurus,
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
  const [streamedTokens, setStreamedTokens] = useState<readonly AnswerToken[]>([]);
  const [highlightSrc, setHighlightSrc] = useState<number | null>(null);

  const [tagFilters, setTagFilters] = useUrlArrayParam('rtag', []);
  const [tagInput, setTagInput] = useState('');
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

  const setConvo = (
    updater: (msgs: readonly ChatMessage[]) => readonly ChatMessage[],
  ) => {
    setThreads((ts) => {
      let id = activeThreadId;
      let arr = ts;
      if (!id || !arr.find((t) => t.id === id)) {
        id = 'th_' + Math.random().toString(16).slice(2, 8);
        arr = [
          {
            id,
            title: 'New thread',
            created: Date.now(),
            updated: Date.now(),
            messages: [],
          },
          ...arr,
        ];
        setActiveThreadId(id);
      }
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
    const id = 'th_' + Math.random().toString(16).slice(2, 8);
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
    tokens: readonly AnswerToken[],
    sources: readonly RetrievalSource[],
  ) => {
    if (tokens.length === 0) {
      setStreaming(false);
      setConvo((c) => [
        ...c,
        { role: 'assistant', tokens: [], sources },
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
        setConvo((c) => [
          ...c,
          { role: 'assistant', tokens, sources },
        ]);
        setStreamedTokens([]);
      }
    }, STREAM_TICK_MS);
  };

  const activeParams = (q: string) => ({
    query: q,
    mode: queryMode,
    topK,
    chunkTopK,
    maxTokens: maxTok,
    historyTurns: history,
    onlyContext: onlyCtx,
    onlyPrompt,
    userPrompt,
    enableRerank,
    tagFilters,
  });

  const send = (text?: string) => {
    const q = (text ?? query).trim();
    if (!q) return;
    setQuery('');
    setConvo((c) => [...c, { role: 'user', text: q }]);
    setStreamedTokens([]);
    setStreaming(true);

    if (onStreamQuery) {
      const streamed: AnswerToken[] = [];
      onStreamQuery(activeParams(q), (chunk) => {
        streamed.push(chunk);
        setStreamedTokens([...streamed]);
      })
        .then(({ sources }) => {
          setStreaming(false);
          const finalTokens = streamed.join('')
            .split(/(\s+)/)
            .filter((t) => t.length > 0);
          setConvo((c) => [
            ...c,
            { role: 'assistant', tokens: finalTokens, sources: sources ?? [] },
          ]);
          setStreamedTokens([]);
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : 'Query failed';
          streamTokens([`⚠ ${msg}`], []);
        });
      return;
    }

    const sendQuery = onSendQuery ?? missingRetrievalBackend;

    sendQuery(activeParams(q))
      .then(({ response, sources }) => {
        const tokens = response
          .split(/(\s+)/)
          .filter((t) => t.length > 0);
        streamTokens(tokens, sources ?? []);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Query failed';
        streamTokens([`⚠ ${msg}`], []);
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

  const removeTag = (t: string) =>
    setTagFilters(tagFilters.filter((x) => x !== t));
  const addTag = (t: string) => {
    if (t && !tagFilters.includes(t)) setTagFilters([...tagFilters, t]);
    setTagInput('');
  };

  const tagSugg = thesaurus
    .filter((t) => !tagFilters.includes(t.tag))
    .filter((t) => !tagInput || t.tag.includes(tagInput.toLowerCase()))
    .slice(0, 4);

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
          {streaming && streamedTokens.length > 0 && (
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

        <div className="field">
          <label className="field-label">
            Tag filter{' '}
            <span style={{ color: 'var(--color-text-tertiary)', fontSize: 10 }}>
              — Twin
            </span>
          </label>
          <div className="chip-input">
            {tagFilters.map((t) => (
              <TagChip key={t} tag={t} removable onRemove={removeTag} />
            ))}
            <input
              value={tagInput}
              aria-label="Retrieval tag input"
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && tagSugg[0]) addTag(tagSugg[0].tag);
              }}
              placeholder={tagFilters.length ? '' : 'add tag…'}
            />
          </div>
          {tagInput && tagSugg.length > 0 && (
            <div
              className="autocomplete panel-autocomplete"
              role="listbox"
              style={{ marginTop: 4 }}
            >
              {tagSugg.map((s, i) => (
                <div
                  key={s.tag}
                  className={`autocomplete-row${i === 0 ? ' focus' : ''}`}
                  onMouseDown={() => addTag(s.tag)}
                  role="option"
                  aria-selected={i === 0}
                  data-testid={`rtag-sugg-${s.tag}`}
                >
                  <div className="row1">
                    <span style={{ fontSize: 12 }}>{s.tag}</span>
                    <span className="badge">{s.category}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
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
  if (msg.role === 'user') {
    return <div className="msg-user">{msg.text}</div>;
  }

  const parts: AnswerPart[] = parseAnswer(msg.tokens ?? []);

  return (
    <div className="msg-assistant">
      <div className="msg-text">
        {parts.map((p, i) => {
          if (p.type === 'text') return <span key={i}>{p.value}</span>;
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
      {!streaming && msg.sources && msg.sources.length > 0 && (
        <>
          <div className="sources-header">Sources</div>
          <div className="sources-list">
            {msg.sources.map((s) => {
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
          </div>
        </>
      )}
    </div>
  );
}
