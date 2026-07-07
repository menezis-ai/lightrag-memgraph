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
import { logTechnicalError, userErrorMessage } from '../lib/errorMessages';
import {
  parseAnswer,
  QUERY_MODES,
  relTime,
  type AnswerPart,
  type InlineAnswerPart,
  type AnswerStatus,
  type ChatMessage,
  type QueryMode,
  type RetrievalSource,
  type RetrievalThread,
} from '../types/retrieval';

// Versioned key — bumped to invalidate stale pre-production local threads.
const THREADS_STORAGE_KEY = 'twin-rag.threads.v3';

/** Per-folder threads key. Folder cloisonnement extends to retrieval history:
 *  a thread is scoped to the folder it was created in so its messages can never
 *  be replayed as ``conversation_history`` into another folder's query. Absent
 *  a folder (standalone/fixture runs) the base key is used unchanged. */
function threadsStorageKey(folder?: string): string {
  return folder ? `${THREADS_STORAGE_KEY}:${folder}` : THREADS_STORAGE_KEY;
}

function loadThreads(
  key: string,
  fallback: readonly RetrievalThread[],
): readonly RetrievalThread[] {
  try {
    const raw = localStorage.getItem(key);
    if (raw) return JSON.parse(raw) as RetrievalThread[];
  } catch {
    /* ignore */
  }
  return fallback;
}
const STREAM_TICK_MS = 70;
const INITIAL_VISIBLE_SOURCES = 5;

const makeThreadId = () => 'th_' + Math.random().toString(16).slice(2, 8);

function ensureThread(
  threads: readonly RetrievalThread[],
  id: string,
): readonly RetrievalThread[] {
  if (threads.some((thread) => thread.id === id)) return threads;
  return [
    {
      id,
      title: 'New thread',
      created: Date.now(),
      updated: Date.now(),
      messages: [],
    },
    ...threads,
  ];
}

function updateThreadMessages(
  thread: RetrievalThread,
  id: string,
  updater: (msgs: readonly ChatMessage[]) => readonly ChatMessage[],
): RetrievalThread {
  if (thread.id !== id) return thread;
  const messages = updater(thread.messages);
  const firstUser = messages.find((m) => m.role === 'user');
  const title =
    thread.messages.length === 0 && firstUser?.text
      ? firstUser.text.slice(0, 64)
      : thread.title;
  return {
    ...thread,
    updated: Date.now(),
    messages,
    title,
  };
}

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
    signal?: AbortSignal;
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
      signal?: AbortSignal;
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
  /** Active folder id. Retrieval threads are partitioned per folder so that
   *  ``conversation_history`` from folder A is never replayed into a folder B
   *  query — that path bypasses the storage-layer folder scoping and would leak
   *  cross-folder context through the prompt. */
  activeFolder?: string;
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

function withOccurrenceKeys<T>(
  items: readonly T[],
  baseKeyFor: (item: T) => string,
): { item: T; key: string }[] {
  const seen = new Map<string, number>();
  return items.map((item) => {
    const baseKey = baseKeyFor(item);
    const occurrence = seen.get(baseKey) ?? 0;
    seen.set(baseKey, occurrence + 1);
    return { item, key: `${baseKey}:${occurrence}` };
  });
}

function chatMessageKeyBase(message: ChatMessage): string {
  if (message.role === 'user') return `user:${message.text}`;
  const answer = (message.tokens ?? []).join('');
  const sourceKey = (message.sources ?? []).map((source) => source.n).join(',');
  return `assistant:${answer}:${sourceKey}`;
}

function inlineAnswerPartKeyBase(part: InlineAnswerPart): string {
  if (part.type === 'cite') return `cite:${part.value}`;
  return `${part.type}:${part.value}`;
}

function answerPartKeyBase(part: AnswerPart): string {
  if (part.type === 'lineBreak') return 'lineBreak';
  if (part.type === 'heading') {
    return `heading:${part.level}:${part.children
      .map(inlineAnswerPartKeyBase)
      .join('|')}`;
  }
  if (part.type === 'listItem') {
    return `list:${part.ordered}:${part.children
      .map(inlineAnswerPartKeyBase)
      .join('|')}`;
  }
  return inlineAnswerPartKeyBase(part);
}

function splitThinkBlocks(text: string): { visible: string; thoughts: readonly string[] } {
  const thoughts: string[] = [];
  const visible = text
    .replaceAll(/<think\b[^>]*>([\s\S]*?)(?:<\/think>|$)/gi, (_match, thought) => {
      const trimmed = String(thought ?? '').trim();
      if (trimmed) thoughts.push(trimmed);
      return '';
    })
    .replaceAll(/<\/think>/gi, '')
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

function normalizeNumberDraft(raw: string): string {
  return raw.trim().replace(/^0+(?=\d)/, '');
}

function clampNumber(value: number, min?: number, max?: number): number {
  let next = value;
  if (typeof min === 'number') next = Math.max(min, next);
  if (typeof max === 'number') next = Math.min(max, next);
  return next;
}

interface NumericParameterInputProps {
  id: string;
  label: string;
  ariaLabel: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
}

function NumericParameterInput({
  id,
  label,
  ariaLabel,
  value,
  onChange,
  min,
  max,
  step,
  integer = true,
}: Readonly<NumericParameterInputProps>) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    if (!editing) setDraft(String(value));
  }, [editing, value]);

  const commitDraft = (raw: string): number | null => {
    if (raw === '') return null;
    const parsed = integer ? Number.parseInt(raw, 10) : Number(raw);
    if (!Number.isFinite(parsed)) return null;
    const next = clampNumber(integer ? Math.trunc(parsed) : parsed, min, max);
    onChange(next);
    return next;
  };

  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        aria-label={ariaLabel}
        value={draft}
        onFocus={(e) => {
          setEditing(true);
          e.currentTarget.select();
        }}
        onChange={(e) => {
          const nextDraft = normalizeNumberDraft(e.target.value);
          setDraft(nextDraft);
          const committed = commitDraft(nextDraft);
          if (committed !== null && String(committed) !== nextDraft) {
            setDraft(String(committed));
          }
        }}
        onBlur={() => {
          setEditing(false);
          const committed = commitDraft(draft);
          setDraft(String(committed ?? value));
        }}
      />
    </div>
  );
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException
    ? err.name === 'AbortError'
    : err instanceof Error && err.name === 'AbortError';
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
  activeFolder,
}: Readonly<RetrievalTabProps>) {
  const threadsKey = threadsStorageKey(activeFolder);
  const [query, setQuery] = useState('');
  const [threads, setThreads] = useState<readonly RetrievalThread[]>(() =>
    loadThreads(threadsKey, initialThreads),
  );
  const [activeThreadId, setActiveThreadId] = useState<string | null>(
    () => loadThreads(threadsKey, initialThreads)[0]?.id ?? null,
  );
  const [streaming, setStreaming] = useState(false);
  const [streamingThreadId, setStreamingThreadId] = useState<string | null>(null);
  const [streamedTokens, setStreamedTokens] = useState<readonly string[]>([]);
  const [highlightSrc, setHighlightSrc] = useState<number | null>(null);

  const [queryMode, setQueryMode] = useUrlParam<QueryMode>('mode', 'mix', {
    validate: (v) => QUERY_MODES.includes(v),
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
  const activeRequestRef = useRef<{
    threadId: string;
    controller: AbortController;
  } | null>(null);
  const streamTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Latest-value ref holding the current folder's key, synced in an effect (not
  // during render). The persist effect fires on `threads` change only and reads
  // this — so it never fires on a key change alone, which would persist the
  // *previous* folder's threads under the new key mid-switch.
  const threadsKeyRef = useRef(threadsKey);
  // Tracks which folder's threads are loaded, so the reload effect runs once
  // per real switch (guards StrictMode double-invoke / no-op re-renders).
  const loadedKeyRef = useRef(threadsKey);

  const activeThread = threads.find((t) => t.id === activeThreadId);
  const convo = useMemo<readonly ChatMessage[]>(
    () => activeThread?.messages ?? [],
    [activeThread],
  );

  useEffect(() => {
    threadsKeyRef.current = threadsKey;
  }, [threadsKey]);

  // Folder switch → load that folder's own threads and drop the active one.
  // This is the cloisonnement guarantee: after switching to folder B no folder
  // A message survives in state, so the next query's conversation_history is
  // empty (or B's own), never A's. The subsequent setThreads triggers the
  // persist effect, which writes B's threads under B's key.
  useEffect(() => {
    if (loadedKeyRef.current === threadsKey) return;
    activeRequestRef.current?.controller.abort();
    activeRequestRef.current = null;
    if (streamTimerRef.current) clearInterval(streamTimerRef.current);
    streamTimerRef.current = null;
    setStreaming(false);
    setStreamingThreadId(null);
    setStreamedTokens([]);
    loadedKeyRef.current = threadsKey;
    const loaded = loadThreads(threadsKey, []);
    setThreads(loaded);
    setActiveThreadId(loaded[0]?.id ?? null);
  }, [threadsKey]);

  useEffect(
    () => () => {
      activeRequestRef.current?.controller.abort();
      activeRequestRef.current = null;
      if (streamTimerRef.current) clearInterval(streamTimerRef.current);
      streamTimerRef.current = null;
    },
    [],
  );

  // Persist threads under the active folder's key. Deliberately keyed on
  // `threads` only (not `threadsKey`): on a switch render `threads` is still
  // the old folder's value, so this must NOT fire then — it fires on the next
  // render once the reload effect has swapped in the new folder's threads, by
  // which point threadsKeyRef holds the new key.
  useEffect(() => {
    try {
      localStorage.setItem(threadsKeyRef.current, JSON.stringify(threads));
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
    if (activeThreadId && threads.some((t) => t.id === activeThreadId)) {
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
    setThreads((ts) =>
      ensureThread(ts, id).map((thread) =>
        updateThreadMessages(thread, id, updater),
      ),
    );
  };

  const appendToExistingThread = (
    id: string,
    updater: (msgs: readonly ChatMessage[]) => readonly ChatMessage[],
  ) => {
    setThreads((ts) =>
      ts.map((thread) =>
        updateThreadMessages(thread, id, updater),
      ),
    );
  };

  const abortActiveRequest = () => {
    activeRequestRef.current?.controller.abort();
    activeRequestRef.current = null;
    if (streamTimerRef.current) clearInterval(streamTimerRef.current);
    streamTimerRef.current = null;
    setStreamedTokens([]);
    setStreamingThreadId(null);
    setStreaming(false);
  };

  const isCurrentRequest = (
    threadId: string,
    controller: AbortController,
  ): boolean =>
    activeRequestRef.current?.threadId === threadId &&
    activeRequestRef.current.controller === controller &&
    !controller.signal.aborted;

  const finishRequest = (threadId: string, controller: AbortController) => {
    if (!isCurrentRequest(threadId, controller)) return false;
    activeRequestRef.current = null;
    setStreaming(false);
    setStreamingThreadId(null);
    setStreamedTokens([]);
    return true;
  };

  const newThread = () => {
    abortActiveRequest();
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
    if (activeRequestRef.current?.threadId === id) {
      abortActiveRequest();
    }
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
    tokens: readonly string[],
    sources: readonly RetrievalSource[],
    answerStatus: AnswerStatus = 'grounded',
    requestedTopK?: number,
    controller?: AbortController,
  ) => {
    if (controller && !isCurrentRequest(threadId, controller)) return;
    if (tokens.length === 0) {
      if (controller) finishRequest(threadId, controller);
      else {
        setStreaming(false);
        setStreamingThreadId(null);
      }
      appendToExistingThread(threadId, (c) => [
        ...c,
        { role: 'assistant', tokens: [], sources, answerStatus, requestedTopK },
      ]);
      return;
    }
    let i = 0;
    const interval = setInterval(() => {
      if (controller && !isCurrentRequest(threadId, controller)) {
        clearInterval(interval);
        if (streamTimerRef.current === interval) streamTimerRef.current = null;
        return;
      }
      i++;
      setStreamedTokens(tokens.slice(0, i));
      if (i >= tokens.length) {
        clearInterval(interval);
        if (streamTimerRef.current === interval) streamTimerRef.current = null;
        if (controller) finishRequest(threadId, controller);
        else {
          setStreaming(false);
          setStreamingThreadId(null);
        }
        appendToExistingThread(threadId, (c) => [
          ...c,
          { role: 'assistant', tokens, sources, answerStatus, requestedTopK },
        ]);
        setStreamedTokens([]);
      }
    }, STREAM_TICK_MS);
    streamTimerRef.current = interval;
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
    signal?: AbortSignal,
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
    signal,
    tagFilter: filterPayload(tagFilter, tagFilterMode),
    docFilter: filterPayload(docFilter, docFilterMode),
  });

  const send = (text?: string) => {
    if (streaming) return;
    const q = (text ?? query).trim();
    if (!q) return;
    setQuery('');
    const threadId = ensureActiveThread();
    const currentMessages =
      threads.find((t) => t.id === threadId)?.messages ?? [];
    const conversationHistory = conversationHistoryFor(currentMessages);
    const requestedTopK = topK;
    const controller = new AbortController();
    activeRequestRef.current = { threadId, controller };
    appendToThread(threadId, (c) => [...c, { role: 'user', text: q }]);
    setStreamedTokens([]);
    setStreamingThreadId(threadId);
    setStreaming(true);

    if (onStreamQuery) {
      const streamed: string[] = [];
      onStreamQuery(activeParams(q, conversationHistory, controller.signal), (chunk) => {
        if (!isCurrentRequest(threadId, controller)) return;
        streamed.push(chunk);
        setStreamedTokens([...streamed]);
      })
        .then(({ sources, answer_status }) => {
          if (!finishRequest(threadId, controller)) return;
          const finalTokens = streamed.join('')
            .split(/(\s+)/)
            .filter((t) => t.length > 0);
          const status: AnswerStatus = answer_status ?? 'grounded';
          // Only a grounded answer carries meaningful sources. For every
          // other status (insufficient_information, source_projection_failed,
          // no_retrieval, query_failed) drop any sources that slipped through
          // so neither the Sources panel NOR the inline [N] citations can
          // surface or navigate to them — defends against a future backend
          // regression.
          const effectiveSources = status === 'grounded' ? (sources ?? []) : [];
          appendToExistingThread(threadId, (c) => [
            ...c,
            {
              role: 'assistant',
              tokens: finalTokens,
              sources: effectiveSources,
              answerStatus: status,
              requestedTopK,
            },
          ]);
        })
        .catch((err: unknown) => {
          if (isAbortError(err) || !isCurrentRequest(threadId, controller)) return;
          logTechnicalError('retrieval-stream', err);
          const msg = userErrorMessage(err, { action: 'answering your question' });
          streamTokens(
            threadId,
            [`⚠ ${msg}`],
            [],
            'query_failed',
            requestedTopK,
            controller,
          );
        });
      return;
    }

    const sendQuery = onSendQuery ?? missingRetrievalBackend;

    sendQuery(activeParams(q, conversationHistory, controller.signal))
      .then(({ response, sources, answer_status }) => {
        if (!isCurrentRequest(threadId, controller)) return;
        const tokens = response
          .split(/(\s+)/)
          .filter((t) => t.length > 0);
        const status: AnswerStatus = answer_status ?? 'grounded';
        // Only a grounded answer carries meaningful sources (see the
        // streaming path above) — drop them for every other status so the
        // inline [N] citations cannot navigate to leaked sources either.
        const effectiveSources = status === 'grounded' ? (sources ?? []) : [];
        if (tokens.length === 0) {
          streamTokens(
            threadId,
            ['⚠ The backend returned an empty answer. Sources below.'],
            effectiveSources,
            status,
            requestedTopK,
            controller,
          );
          return;
        }
        streamTokens(
          threadId,
          tokens,
          effectiveSources,
          status,
          requestedTopK,
          controller,
        );
      })
      .catch((err: unknown) => {
        if (isAbortError(err) || !isCurrentRequest(threadId, controller)) return;
        logTechnicalError('retrieval-query', err);
        const msg = userErrorMessage(err, { action: 'answering your question' });
        streamTokens(
          threadId,
          [`⚠ ${msg}`],
          [],
          'query_failed',
          requestedTopK,
          controller,
        );
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
          {withOccurrenceKeys(convo, chatMessageKeyBase).map(({ item, key }) => (
            <Turn
              key={key}
              msg={item}
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
          <label className="field-label" htmlFor="retrieval-query-mode">
            Query mode
          </label>
          <select
            id="retrieval-query-mode"
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
          <span className="field-label">Source tag filters</span>
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
          <span className="field-label">Source document filters</span>
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

        <NumericParameterInput
          id="retrieval-top-k"
          label="Top K results"
          ariaLabel="Top K"
          value={topK}
          onChange={setTopK}
          min={0}
        />
        <NumericParameterInput
          id="retrieval-chunk-top-k"
          label="Chunk top K"
          ariaLabel="Chunk top K"
          value={chunkTopK}
          onChange={setChunkTopK}
          min={0}
        />
        <NumericParameterInput
          id="retrieval-max-tokens"
          label="Max tokens · text unit"
          ariaLabel="Max tokens"
          value={maxTok}
          onChange={setMaxTok}
          min={0}
        />
        <NumericParameterInput
          id="retrieval-min-score"
          label="Minimum source score"
          ariaLabel="Minimum source score"
          value={minScore}
          onChange={setMinScore}
          min={0}
          max={1}
          step={0.01}
          integer={false}
        />
        <NumericParameterInput
          id="retrieval-history-turns"
          label="History turns"
          ariaLabel="History turns"
          value={history}
          onChange={setHistory}
          min={0}
        />
        <div className="field">
          <label className="field-label" htmlFor="retrieval-system-prompt">
            System prompt
          </label>
          <textarea
            id="retrieval-system-prompt"
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
          />{' '}
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
          />{' '}
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
          />{' '}
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
  const availableSourceNumbers = new Set(sources.map((source) => source.n));
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
    withOccurrenceKeys(inlineParts, inlineAnswerPartKeyBase).map(({ item: p, key }) => {
      if (p.type === 'text') return <span key={key}>{p.value}</span>;
      if (p.type === 'bold') return <strong key={key}>{p.value}</strong>;
      if (p.type === 'code') return <code key={key}>{p.value}</code>;
      // A ``[N]`` marker with no matching source (LLM hallucination, an
      // external bibliographic ref, or any non-grounded answer whose sources
      // are empty) must NOT masquerade as a live Twin anchor. Render it inert:
      // no button affordance, no hover/click, so the operator can't be misled
      // into thinking it navigates somewhere.
      if (!availableSourceNumbers.has(p.value)) {
        return (
          <span
            key={key}
            className="citation citation-inert"
            data-testid={`citation-inert-${p.value}`}
          >
            {p.value}
          </span>
        );
      }
      return (
        <button
          key={key}
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
        {withOccurrenceKeys(parts, answerPartKeyBase).map(({ item: p, key }) => {
          if (p.type === 'lineBreak') return <br key={key} />;
          if (p.type === 'heading') {
            const children = renderInlineParts(p.children);
            if (p.level === 1) {
              return (
                <h1 key={key} className="answer-heading answer-heading-1">
                  {children}
                </h1>
              );
            }
            if (p.level === 2) {
              return (
                <h2 key={key} className="answer-heading answer-heading-2">
                  {children}
                </h2>
              );
            }
            return (
              <h3 key={key} className="answer-heading answer-heading-3">
                {children}
              </h3>
            );
          }
          if (p.type === 'listItem') {
            return (
              <div
                key={key}
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
      {/* The answer is grounded but its references could not be projected
          (LightRAG envelope shape broke). Show the answer + an honest cue
          rather than a silent empty area that reads as "no sources". */}
      {!streaming && msg.answerStatus === 'source_projection_failed' && (
        <div
          className="sources-empty muted"
          data-testid="sources-empty-projection-failed"
          style={{ marginTop: 8, fontSize: 12 }}
        >
          Sources unavailable for this answer.
        </div>
      )}
      {/* Sourceless by design (bypass / only_need_context / only_need_prompt):
          no retrieval was attempted, so the empty Sources area is expected.
          Show a discrete cue so the operator reads it as intentional rather
          than a missing-sources glitch. */}
      {!streaming && msg.answerStatus === 'no_retrieval' && (
        <div
          className="sources-empty muted"
          data-testid="sources-empty-no-retrieval"
          style={{ marginTop: 8, fontSize: 12 }}
        >
          Answered without retrieval — no sources for this mode.
        </div>
      )}
      {/* A backend error occurred mid-stream: the answer text is an
          ``[query failed: …]`` error notice, not a grounded answer. Suppress
          the Sources panel and show an explicit failure cue rather than
          letting an empty area read as "no sources for a real answer". */}
      {!streaming && msg.answerStatus === 'query_failed' && (
        <div
          className="sources-empty muted"
          data-testid="sources-empty-query-failed"
          style={{ marginTop: 8, fontSize: 12 }}
        >
          The query could not be completed — no answer was retrieved.
        </div>
      )}
      {!streaming &&
        msg.answerStatus !== 'insufficient_information' &&
        msg.answerStatus !== 'source_projection_failed' &&
        msg.answerStatus !== 'no_retrieval' &&
        msg.answerStatus !== 'query_failed' &&
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
                    className={s.type === 'file' ? 'src-name' : 'src-name mono'}
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
  const datalistId = `${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}-options`;

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
