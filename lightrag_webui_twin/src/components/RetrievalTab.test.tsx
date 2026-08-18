/**
 * Unit tests for RetrievalTab.
 *
 * Behaviors under test:
 *   - empty state with suggestions when no thread / convo
 *   - clicking a suggestion sends the query (streams + persists assistant msg)
 *   - typing + pressing Enter sends the query
 *   - thread switch shows the right messages
 *   - delete thread removes it from the list
 *   - localStorage persistence on threads change
 *   - citation button click triggers (just verify clickable & accessible)
 *   - clear empties the active thread
 */

import {
  describe,
  expect,
  it,
  vi,
  beforeEach,
  afterEach,
} from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RetrievalTab, type RetrievalTabProps } from './RetrievalTab';
import { makeSampleThreads } from '../fixtures';

function defaultProps() {
  return {
    initialThreads: makeSampleThreads(),
  };
}

beforeEach(() => {
  window.history.replaceState(null, '', '/');
  window.localStorage.clear();
});
afterEach(() => {
  window.history.replaceState(null, '', '/');
  window.localStorage.clear();
});

describe('RetrievalTab — empty state', () => {
  it('renders the empty-state with default suggestions', () => {
    render(<RetrievalTab {...defaultProps()} initialThreads={[]} />);
    expect(
      screen.getByText(/Ask a question to retrieve/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', {
        name: /Ask about a source in this folder/,
      }),
    ).toBeInTheDocument();
  });

  it('honors custom suggestions prop', () => {
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        suggestions={['Only this one']}
      />,
    );
    expect(
      screen.getByRole('button', { name: /Only this one/ }),
    ).toBeInTheDocument();
  });
});

describe('RetrievalTab — message actions and Markdown', () => {
  const actionThread = () => [{
    id: 'actions',
    title: 'Action thread',
    created: 1,
    updated: 1,
    messages: [
      { role: 'user' as const, text: 'Original prompt' },
      {
        role: 'assistant' as const,
        tokens: ['| Step | Command |\n| --- | --- |\n| Stop | `shutdown immediate` |'],
        sources: [],
        queryMeta: {
          model: 'deepseek-chat',
          mode: 'mix' as const,
          topK: 20,
          chunkTopK: 12,
          enableRerank: true,
          durationMs: 1400,
        },
      },
    ],
  }];

  it('renders a table, all requested actions, and query metadata', () => {
    render(<RetrievalTab initialThreads={actionThread()} />);

    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByTitle('Copy prompt')).toBeInTheDocument();
    expect(screen.getByTitle('Edit prompt')).toBeInTheDocument();
    expect(screen.getByTitle('Copy answer as plain text')).toBeInTheDocument();
    expect(screen.getByTitle('Copy answer as Markdown')).toBeInTheDocument();
    expect(screen.getByTitle('Regenerate answer')).toBeInTheDocument();
    expect(screen.getByTitle('Branch to a new chat from here')).toBeInTheDocument();
    expect(screen.getByTestId('answer-run-meta')).toHaveTextContent('deepseek-chat');
    expect(screen.getByTestId('answer-run-meta')).toHaveTextContent('top_k 20');
    expect(screen.getByTestId('answer-run-meta')).toHaveTextContent('1.4s');
  });

  it('copies the raw Markdown independently from plain text', async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    render(<RetrievalTab initialThreads={actionThread()} />);

    await userEvent.click(screen.getByTitle('Copy answer as Markdown'));
    expect(writeText).toHaveBeenLastCalledWith(expect.stringContaining('| Step |'));

    await userEvent.click(screen.getByTitle('Copy answer as plain text'));
    expect(writeText).toHaveBeenLastCalledWith(expect.not.stringContaining('`'));
    expect(writeText).toHaveBeenLastCalledWith(expect.not.stringContaining('| ---'));
  });

  it('edits and resubmits a prompt from the correct history point', async () => {
    const onSendQuery = vi.fn<NonNullable<RetrievalTabProps['onSendQuery']>>(
      async () => ({ response: 'Updated answer', sources: [] }),
    );
    render(<RetrievalTab initialThreads={actionThread()} onSendQuery={onSendQuery} />);

    await userEvent.click(screen.getByTitle('Edit prompt'));
    const editor = screen.getByLabelText('Edit prompt');
    await userEvent.clear(editor);
    await userEvent.type(editor, 'Edited prompt');
    await userEvent.click(screen.getByRole('button', { name: 'Save & submit' }));

    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    expect(onSendQuery.mock.calls[0]?.[0]).toMatchObject({
      query: 'Edited prompt',
      conversationHistory: [],
    });
  });

  it('regenerates from the original user turn without replaying that turn', async () => {
    const onSendQuery = vi.fn<NonNullable<RetrievalTabProps['onSendQuery']>>(
      async () => ({ response: 'Regenerated answer', sources: [] }),
    );
    render(<RetrievalTab initialThreads={actionThread()} onSendQuery={onSendQuery} />);

    await userEvent.click(screen.getByTitle('Regenerate answer'));

    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    expect(onSendQuery.mock.calls[0]?.[0]).toMatchObject({
      query: 'Original prompt',
      conversationHistory: [],
    });
  });

  it('copies the same answer that is displayed when References has a suffix', async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    const thread = actionThread();
    const messages = [...thread[0].messages];
    messages[1] = {
      role: 'assistant',
      tokens: ['Visible answer\n\n### References — cited docs\n- hidden.pdf'],
      sources: [],
      queryMeta: {
        model: 'deepseek-chat',
        mode: 'mix',
        topK: 20,
        chunkTopK: 12,
        enableRerank: true,
        durationMs: 1400,
      },
    };
    render(
      <RetrievalTab
        initialThreads={[{ ...thread[0], messages }]}
      />,
    );

    expect(screen.getByText('Visible answer')).toBeInTheDocument();
    expect(screen.queryByText(/hidden\.pdf/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByTitle('Copy answer as Markdown'));
    expect(writeText).toHaveBeenLastCalledWith('Visible answer');
    await userEvent.click(screen.getByTitle('Copy answer as plain text'));
    expect(writeText).toHaveBeenLastCalledWith('Visible answer');
  });

  it('branches the conversation into a new active chat', async () => {
    render(<RetrievalTab initialThreads={actionThread()} />);
    await userEvent.click(screen.getByTitle('Branch to a new chat from here'));
    expect(screen.getByText(/Original prompt · branch/)).toBeInTheDocument();
    expect(screen.getByText('Original prompt')).toBeInTheDocument();
  });
});

describe('RetrievalTab — thread switcher', () => {
  it('renders all threads from props', () => {
    render(<RetrievalTab {...defaultProps()} />);
    expect(screen.getByTestId('thread-th_seed_1')).toBeInTheDocument();
    expect(screen.getByTestId('thread-th_seed_2')).toBeInTheDocument();
  });

  it('clicking a thread switches to it', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const t2 = screen.getByTestId('thread-th_seed_2');
    await userEvent.click(t2);
    expect(t2.className).toMatch(/is-active/);
  });

  it('Delete button removes the thread', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    await userEvent.click(
      screen.getByLabelText('Delete CFT troubleshooting checklist'),
    );
    expect(screen.queryByTestId('thread-th_seed_2')).toBeNull();
  });

  it('clicking New creates a fresh empty thread', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    await userEvent.click(screen.getByTitle('New conversation'));
    // 3 threads now: 1 new + 2 seeds. The new one is "New thread"
    expect(screen.getByText('New thread')).toBeInTheDocument();
  });

  it('renders unavailable scores honestly and preserves numeric scores', () => {
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'score-contract',
            title: 'Score contract',
            created: 1,
            updated: 1,
            messages: [
              {
                role: 'assistant',
                tokens: ['Answer [1] [2] [3] [4] [5]'],
                sources: [
                  { n: 1, type: 'file', name: 'null.pdf', score: null },
                  { n: 2, type: 'file', name: 'absent.pdf' },
                  { n: 3, type: 'file', name: 'zero.pdf', score: 0 },
                  { n: 4, type: 'file', name: 'ranked.pdf', score: 0.82 },
                  {
                    n: 5,
                    type: 'file',
                    name: 'graph.pdf',
                    score: null,
                    retrieval_origin: 'graph',
                  },
                ],
              },
            ],
          },
        ]}
      />,
    );

    const displayedScore = (sourceNumber: number) =>
      screen.getByTestId(`source-${sourceNumber}`).querySelector('.src-score');

    expect(displayedScore(1)).toHaveTextContent('—');
    expect(displayedScore(1)).toHaveAttribute('title', 'Score unavailable');
    expect(displayedScore(2)).toHaveTextContent('—');
    expect(displayedScore(2)).toHaveAttribute('title', 'Score unavailable');
    expect(displayedScore(3)).toHaveTextContent('0.00');
    expect(displayedScore(3)).not.toHaveAttribute('title');
    expect(displayedScore(4)).toHaveTextContent('0.82');
    expect(displayedScore(4)).not.toHaveAttribute('title');
    expect(displayedScore(5)).toHaveTextContent('graph sourced');
    expect(displayedScore(5)).toHaveAttribute(
      'title',
      'Grounded through graph retrieval',
    );
  });
});

describe('RetrievalTab — send', () => {
  // Streaming exercises a setInterval that is awkward to drive via fake
  // timers in this environment (Bun + happy-dom 20.9). We only assert
  // that the user message appears immediately on send — verifying that
  // the click/keystroke wires through to setConvo. The streaming tick
  // itself is already covered by parseAnswer + helpers in retrieval.test.ts.

  it('clicking a suggestion enqueues a user message in the active thread', async () => {
    render(<RetrievalTab {...defaultProps()} initialThreads={[]} />);
    await userEvent.click(
      screen.getByRole('button', { name: /Summarize recent indexed documents/ }),
    );
    // Look in the conversation pane (not the sidebar title)
    const userMsg = document.querySelector('.msg-user');
    expect(userMsg?.textContent).toBe('Summarize recent indexed documents');
  });

  it('Send button click triggers send when textarea has content', async () => {
    render(<RetrievalTab {...defaultProps()} initialThreads={[]} />);
    const box = screen.getByLabelText('Query input');
    await userEvent.type(box, 'Quick question');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));
    const userMsg = document.querySelector('.msg-user');
    expect(userMsg?.textContent).toBe('Quick question');
  });

  it('streams backend chunks into the assistant message', async () => {
    const onStreamQuery = vi.fn(async (_params, onChunk: (chunk: string) => void) => {
      onChunk('hello ');
      onChunk('world\n\n### References - [1] runbook.pdf');
      return {
        response: 'hello world\n\n### References - [1] runbook.pdf',
        sources: [],
      };
    });
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'Stream this');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() => expect(onStreamQuery).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(document.querySelector('.msg-assistant')?.textContent).toContain(
        'hello world',
      ),
    );
    expect(document.querySelector('.msg-assistant')?.textContent).not.toContain(
      'References',
    );
  });

  it('shows a thinking indicator while waiting for the first backend chunk', async () => {
    const onStreamQuery = vi.fn(
      () =>
        new Promise<{ response: string; sources: [] }>(() => {
          // Keep the request pending so the pre-token loading state is visible.
        }),
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'Slow answer');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    expect(await screen.findByTestId('retrieval-thinking')).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent(
      'Searching vector and graph context',
    );
  });

  it('keeps pre-token progress labels aligned with backend stage events', async () => {
    let emitStage:
      | ((stage: 'retrieval' | 'generation' | 'sources') => void)
      | undefined;
    const onStreamQuery = vi.fn(
      (
        _params: unknown,
        _onChunk: (chunk: string) => void,
        onStage: (stage: 'retrieval' | 'generation' | 'sources') => void,
      ) => {
        emitStage = onStage;
        return new Promise<{ response: string; sources: [] }>(() => {
          // Keep the request pending while stage labels are asserted.
        });
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'Stage probe');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));
    expect(await screen.findByTestId('retrieval-thinking')).toHaveTextContent(
      'Searching vector and graph context',
    );

    act(() => emitStage?.('generation'));
    expect(screen.getByTestId('retrieval-thinking')).toHaveTextContent(
      'Generating answer',
    );

    act(() => emitStage?.('sources'));
    expect(screen.getByTestId('retrieval-thinking')).toHaveTextContent(
      'Finalizing sources',
    );
  });

  it('does not show transient stream chunks in a different active thread', async () => {
    let pushChunk: ((chunk: string) => void) | undefined;
    let finish!: (value: { response: string; sources: [] }) => void;
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        pushChunk = onChunk;
        return new Promise<{ response: string; sources: [] }>((resolve) => {
          finish = resolve;
        });
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_owner',
            title: 'Owner',
            created: Date.now(),
            updated: Date.now(),
            messages: [],
          },
          {
            id: 'th_other',
            title: 'Other',
            created: Date.now(),
            updated: Date.now(),
            messages: [],
          },
        ]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'stream owner');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));
    await waitFor(() => expect(onStreamQuery).toHaveBeenCalledTimes(1));

    await userEvent.click(screen.getByTestId('thread-th_other'));
    act(() => {
      pushChunk?.('owner-only-token');
    });

    expect(
      document.querySelector('.retrieval-conv')?.textContent,
    ).not.toContain('owner-only-token');

    act(() => {
      finish({ response: 'owner-only-token', sources: [] });
    });
  });

  it('aborts a pending stream when its thread is deleted and does not recreate it', async () => {
    let finish!: (value: { response: string; sources: [] }) => void;
    let observedSignal: AbortSignal | undefined;
    const onStreamQuery = vi.fn(
      async (params) => {
        observedSignal = params.signal;
        return new Promise<{ response: string; sources: [] }>((resolve) => {
          finish = resolve;
        });
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'delete owner');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));
    await waitFor(() => expect(onStreamQuery).toHaveBeenCalledTimes(1));

    await userEvent.click(screen.getByLabelText('Delete delete owner'));
    expect(observedSignal?.aborted).toBe(true);

    act(() => {
      finish({ response: 'late answer', sources: [] });
    });

    await waitFor(() =>
      expect(screen.queryByText('delete owner')).not.toBeInTheDocument(),
    );
    expect(screen.queryByText('late answer')).not.toBeInTheDocument();
  });
});

describe('RetrievalTab — localStorage persistence', () => {
  it('writes to twin-rag.threads.v3 when threads change', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    // localStorage should be populated by the initial effect
    const raw = window.localStorage.getItem('twin-rag.threads.v3');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw as string);
    expect(parsed).toHaveLength(2);
  });

  it('reads from localStorage on init when present', () => {
    window.localStorage.setItem(
      'twin-rag.threads.v3',
      JSON.stringify([
        {
          id: 'th_stored',
          title: 'Stored thread',
          created: Date.now(),
          updated: Date.now(),
          messages: [],
        },
      ]),
    );
    render(
      <RetrievalTab {...defaultProps()} initialThreads={makeSampleThreads()} />,
    );
    expect(screen.getByTestId('thread-th_stored')).toBeInTheDocument();
    // The seed threads should NOT be there since localStorage wins
    expect(screen.queryByTestId('thread-th_seed_1')).toBeNull();
  });
});

describe('RetrievalTab — params panel', () => {
  it('uses production-safe retrieval defaults', () => {
    render(<RetrievalTab {...defaultProps()} initialThreads={[]} />);
    expect(screen.getByLabelText('Top K')).toHaveValue(20);
    expect(screen.getByLabelText('Chunk top K')).toHaveValue(20);
    expect(screen.getByLabelText('Max tokens')).toHaveValue(30000);
    expect(screen.getByLabelText('Enable rerank')).toHaveAttribute(
      'aria-checked',
      'true',
    );
  });

  it('Top K input updates', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const topK = screen.getByLabelText('Top K') as HTMLInputElement;
    await userEvent.clear(topK);
    await userEvent.type(topK, '25');
    expect(topK.value).toBe('25');
  });

  it('numeric params allow empty edit state and strip leading zeroes', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const topK = screen.getByLabelText('Top K') as HTMLInputElement;

    await userEvent.clear(topK);
    expect(topK.value).toBe('');

    await userEvent.type(topK, '20');
    expect(topK.value).toBe('20');

    await userEvent.clear(topK);
    await userEvent.type(topK, '020');
    expect(topK.value).toBe('20');
  });

  it('numeric params preserve decimal drafts while editing min score', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const minScore = screen.getByLabelText('Minimum source score') as HTMLInputElement;

    await userEvent.clear(minScore);
    await userEvent.type(minScore, '0.');
    expect(minScore.value).toBe('0.');

    await userEvent.type(minScore, '5');
    expect(minScore.value).toBe('0.5');

    await userEvent.clear(minScore);
    await userEvent.type(minScore, '0.05');
    expect(minScore.value).toBe('0.05');
  });

  it('Query mode select changes value', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const sel = screen.getByLabelText('Query mode') as HTMLSelectElement;
    expect(Array.from(sel.options, (option) => option.value)).toEqual([
      'naive',
      'local',
      'global',
      'hybrid',
      'mix',
    ]);
    await userEvent.selectOptions(sel, 'hybrid');
    expect(sel.value).toBe('hybrid');
  });

  it('normalizes a legacy bypass URL and exposes no prompt override controls', () => {
    window.history.replaceState(null, '', '/?mode=bypass');
    render(<RetrievalTab {...defaultProps()} />);

    expect(screen.getByLabelText('Query mode')).toHaveValue('mix');
    expect(screen.queryByLabelText('System prompt')).toBeNull();
    expect(screen.queryByLabelText('Only need prompt')).toBeNull();
  });

  it('renders advanced source filters for tags and documents', async () => {
    render(
      <RetrievalTab
        {...defaultProps()}
        tagOptions={['oracle', 'rman']}
        docOptions={['doc-oracle']}
        docLabels={{ 'doc-oracle': 'oracle-runbook.pdf' }}
      />,
    );

    await userEvent.type(screen.getByLabelText('Retrieval tag filter'), 'oracle');
    await userEvent.click(
      screen
        .getByLabelText('Retrieval tag filter')
        .closest('.retrieval-filter-input-row')!
        .querySelector('button')!,
    );
    await userEvent.type(
      screen.getByLabelText('Retrieval document filter'),
      'doc-oracle',
    );
    await userEvent.click(
      screen
        .getByLabelText('Retrieval document filter')
        .closest('.retrieval-filter-input-row')!
        .querySelector('button')!,
    );

    expect(screen.getAllByText('oracle').length).toBeGreaterThan(0);
    expect(screen.getAllByText('oracle-runbook.pdf').length).toBeGreaterThan(0);
  });

  it('passes advanced retrieval params to onSendQuery', async () => {
    const onSendQuery = vi.fn(async () => ({ response: 'ok', sources: [] }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onSendQuery={onSendQuery}
        tagOptions={['oracle', 'rman']}
        docOptions={['doc-oracle', 'doc-network']}
      />,
    );

    await userEvent.selectOptions(screen.getByLabelText('Query mode'), 'hybrid');
    await userEvent.clear(screen.getByLabelText('Top K'));
    await userEvent.type(screen.getByLabelText('Top K'), '12');
    await userEvent.clear(screen.getByLabelText('Chunk top K'));
    await userEvent.type(screen.getByLabelText('Chunk top K'), '6');
    await userEvent.clear(screen.getByLabelText('Max tokens'));
    await userEvent.type(screen.getByLabelText('Max tokens'), '2048');
    await userEvent.clear(screen.getByLabelText('Minimum source score'));
    await userEvent.type(screen.getByLabelText('Minimum source score'), '0.7');
    await userEvent.clear(screen.getByLabelText('History turns'));
    await userEvent.type(screen.getByLabelText('History turns'), '2');
    await userEvent.click(screen.getByLabelText('Enable rerank'));
    await userEvent.type(screen.getByLabelText('Retrieval tag filter'), 'oracle');
    await userEvent.click(
      screen
        .getByLabelText('Retrieval tag filter')
        .closest('.retrieval-filter-input-row')!
        .querySelector('button')!,
    );
    await userEvent.type(screen.getByLabelText('Retrieval tag filter'), 'rman');
    await userEvent.click(
      screen
        .getByLabelText('Retrieval tag filter')
        .closest('.retrieval-filter-input-row')!
        .querySelector('button')!,
    );
    await userEvent.click(
      screen.getByRole('group', { name: 'Retrieval tag filter mode' }).querySelectorAll('button')[1],
    );
    await userEvent.type(
      screen.getByLabelText('Retrieval document filter'),
      'doc-oracle',
    );
    await userEvent.click(
      screen
        .getByLabelText('Retrieval document filter')
        .closest('.retrieval-filter-input-row')!
        .querySelector('button')!,
    );
    await userEvent.type(screen.getByLabelText('Query input'), 'Advanced query');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    expect(onSendQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'Advanced query',
        mode: 'hybrid',
        topK: 12,
        chunkTopK: 6,
        maxTokens: 2048,
        minScore: 0.7,
        historyTurns: 2,
        enableRerank: false,
        tagFilter: { all: ['oracle', 'rman'] },
        docFilter: { any: ['doc-oracle'] },
      }),
    );
    const sentParams = (
      onSendQuery.mock.calls as unknown as Array<
        [Record<string, unknown>]
      >
    )[0][0];
    expect(sentParams).not.toHaveProperty('onlyPrompt');
    expect(sentParams).not.toHaveProperty('userPrompt');
  });

  it('hydrates Graph-transferred filters from URL and shows sources when grounded', async () => {
    const onSendQuery = vi.fn(async () => ({
      response: 'Graph-driven answer',
      sources: [
        { n: 1, type: 'file' as const, name: 'graph-source-a.pdf', score: 0.91 },
      ],
      answer_status: 'grounded' as const,
    }));
    window.history.replaceState(
      null,
      '',
      '/?rtag=oracle,rman&rtagmode=all&rdoc=doc-oracle&rdocmode=any',
    );

    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onSendQuery={onSendQuery}
      />,
    );

    await userEvent.type(
      screen.getByLabelText('Query input'),
      'Graph sourced query',
    );
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    expect(onSendQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'Graph sourced query',
        tagFilter: { all: ['oracle', 'rman'] },
        docFilter: { any: ['doc-oracle'] },
      }),
    );
    await waitFor(() => {
      expect(screen.getByText(/^Sources$/)).toBeInTheDocument();
      expect(screen.getByTestId('source-1')).toBeInTheDocument();
    });
  });

  it('sends prior thread messages as conversation history', async () => {
    const onSendQuery = vi.fn(async () => ({ response: 'ok', sources: [] }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_followup',
            title: 'Follow-up',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              { role: 'user', text: 'What is RMAN?' },
              {
                role: 'assistant',
                tokens: ['RMAN is Oracle Recovery Manager.'],
                sources: [],
              },
            ],
          },
        ]}
        onSendQuery={onSendQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'And restore?');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    expect(onSendQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'And restore?',
        historyTurns: 3,
        conversationHistory: [
          { role: 'user', content: 'What is RMAN?' },
          {
            role: 'assistant',
            content: 'RMAN is Oracle Recovery Manager.',
          },
        ],
      }),
    );
  });

  it('bounds replayed history to the backend message contract', async () => {
    const longAnswer = '🙂'.repeat(2_001);
    const onSendQuery = vi.fn<
      NonNullable<RetrievalTabProps['onSendQuery']>
    >(async () => ({ response: 'ok', sources: [] }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_long_history',
            title: 'Long history',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              { role: 'user', text: 'Summarize this document' },
              { role: 'assistant', tokens: [longAnswer], sources: [] },
            ],
          },
        ]}
        onSendQuery={onSendQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'Continue');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    const request = onSendQuery.mock.calls[0]?.[0];
    const assistant = request?.conversationHistory.find(
      (message) => message.role === 'assistant',
    );
    expect(assistant?.content).toBe('🙂'.repeat(2_000));
    expect([...(assistant?.content ?? '')]).toHaveLength(2_000);
  });

  it('question and answer land in the same thread when no thread is active', async () => {
    const onSendQuery = vi.fn(async () => ({
      response: 'LIP6 est un laboratoire.',
      sources: [],
    }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onSendQuery={onSendQuery}
      />,
    );

    await userEvent.type(
      screen.getByLabelText('Query input'),
      'quel est le rôle de LIP6 ?',
    );
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(
      () =>
        expect(
          document.querySelector('.retrieval-conv')?.textContent,
        ).toContain('LIP6 est un laboratoire.'),
      { timeout: 3000 },
    );

    // Exactly ONE thread, titled from the question, holding both messages —
    // not a "question thread" plus a second "New thread" with the answer.
    const items = document.querySelectorAll('.history-item');
    expect(items).toHaveLength(1);
    expect(items[0].textContent).toContain('quel est le rôle de LIP6 ?');
    expect(items[0].textContent).toContain('1 q');
    expect(items[0].textContent).toContain('1 a');
    expect(
      document.querySelector('.retrieval-conv')?.textContent,
    ).toContain('quel est le rôle de LIP6 ?');
  });

  it('empty backend answer surfaces a visible warning instead of a mute bubble', async () => {
    const onSendQuery = vi.fn(async () => ({ response: '', sources: [] }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onSendQuery={onSendQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'hello?');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(
      () =>
        expect(
          document.querySelector('.retrieval-conv')?.textContent,
        ).toContain('empty answer'),
      { timeout: 3000 },
    );
  });
});

describe('RetrievalTab — folder cloisonnement (conversation history)', () => {
  const folderAThread = () => [
    {
      id: 'th_a',
      title: 'A',
      created: Date.now(),
      updated: Date.now(),
      messages: [
        { role: 'user' as const, text: 'Folder A question?' },
        {
          role: 'assistant' as const,
          tokens: ['Folder A answer.'],
          sources: [],
        },
      ],
    },
  ];

  it('replays history within a folder but never across folders', async () => {
    const onSendQuery = vi.fn(async () => ({
      response: 'REPLYTOKEN',
      sources: [],
    }));
    const { rerender } = render(
      <RetrievalTab
        activeFolder="folderA"
        initialThreads={folderAThread()}
        onSendQuery={onSendQuery}
      />,
    );

    // In folder A, the prior A exchange IS replayed as conversation history.
    await userEvent.type(screen.getByLabelText('Query input'), 'A follow-up?');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));
    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(1));
    expect(onSendQuery).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        conversationHistory: [
          { role: 'user', content: 'Folder A question?' },
          { role: 'assistant', content: 'Folder A answer.' },
        ],
      }),
    );
    // Let the answer animation settle so `streaming` clears before switching.
    await waitFor(
      () =>
        expect(
          document.querySelector('.retrieval-conv')?.textContent,
        ).toContain('REPLYTOKEN'),
      { timeout: 3000 },
    );

    // Switch to folder B: the next query must carry NO folder A history —
    // otherwise A's answer leaks into B's prompt, bypassing storage scoping.
    rerender(
      <RetrievalTab
        activeFolder="folderB"
        initialThreads={folderAThread()}
        onSendQuery={onSendQuery}
      />,
    );
    await userEvent.type(screen.getByLabelText('Query input'), 'B question?');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));
    await waitFor(() => expect(onSendQuery).toHaveBeenCalledTimes(2));
    expect(onSendQuery).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ conversationHistory: [] }),
    );
  });

  it('persists threads under a per-folder storage key', async () => {
    render(
      <RetrievalTab activeFolder="folderA" initialThreads={folderAThread()} />,
    );
    await waitFor(() =>
      expect(
        window.localStorage.getItem('twin-rag.threads.v3:folderA'),
      ).toBeTruthy(),
    );
    // The base (folder-less) key is NOT written when a folder is active.
    expect(window.localStorage.getItem('twin-rag.threads.v3')).toBeNull();
  });
});

describe('RetrievalTab — source cards', () => {
  it('shows only the first five sources until the user expands the list', async () => {
    const sources = Array.from({ length: 7 }, (_, i) => ({
      n: i + 1,
      type: 'file' as const,
      name: `source-${i + 1}.pdf`,
      meta: `chunk ${i + 1}`,
      score: 0.9 - i * 0.01,
    }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_many_sources',
            title: 'Many sources',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['answer'],
                sources,
                requestedTopK: 7,
              },
            ],
          },
        ]}
      />,
    );

    expect(screen.getByTestId('source-1')).toBeInTheDocument();
    expect(screen.getByTestId('source-5')).toBeInTheDocument();
    expect(screen.queryByTestId('source-6')).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Voir les 2 autres' }),
    ).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole('button', { name: 'Voir les 2 autres' }),
    );

    expect(screen.getByTestId('source-6')).toBeInTheDocument();
    expect(screen.getByTestId('source-7')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Réduire aux sources principales' }),
    ).toHaveAttribute('aria-expanded', 'true');
  });

  it('does not show an expand button when requested Top K exceeds real returned sources', () => {
    const sources = Array.from({ length: 5 }, (_, i) => ({
      n: i + 1,
      type: 'file' as const,
      name: `returned-${i + 1}.pdf`,
      meta: `chunk ${i + 1}`,
      score: 0.9 - i * 0.01,
    }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_requested_more',
            title: 'Requested more',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['answer'],
                sources,
                requestedTopK: 20,
              },
            ],
          },
        ]}
      />,
    );

    expect(screen.getByTestId('sources-count')).toHaveTextContent(
      '5 returned / 20 requested',
    );
    expect(
      screen.queryByRole('button', { name: /Voir les .* autres/ }),
    ).toBeNull();
    expect(screen.queryByTestId('source-6')).toBeNull();
    expect(screen.queryByTestId('sources-no-additional')).toBeNull();
  });

  it('keeps cited sources visible even when they are outside the first five', async () => {
    const sources = Array.from({ length: 7 }, (_, i) => ({
      n: i + 1,
      type: 'file' as const,
      name: `source-${i + 1}.pdf`,
      meta: `chunk ${i + 1}`,
      score: 0.9 - i * 0.01,
    }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_cited_outside_top_five',
            title: 'Cited outside top five',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['answer [7]'],
                sources,
                requestedTopK: 20,
              },
            ],
          },
        ]}
      />,
    );

    expect(screen.getByTestId('source-1')).toBeInTheDocument();
    expect(screen.getByTestId('source-5')).toBeInTheDocument();
    expect(screen.getByTestId('source-7')).toBeInTheDocument();
    expect(screen.queryByTestId('source-6')).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Voir les 1 autres' }),
    ).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole('button', { name: 'Voir les 1 autres' }),
    );

    expect(screen.getByTestId('source-6')).toBeInTheDocument();
  });

  it('renders minimal Markdown while preserving clickable citations', () => {
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_markdown',
            title: 'Markdown',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: [
                  '### Runbook\nUse **RMAN** with `restore database` [1]\n- validate backup\n1. open incident',
                ],
                sources: [
                  { n: 1, type: 'file' as const, name: 'runbook.pdf', score: 0.9 },
                ],
              },
            ],
          },
        ]}
      />,
    );

    expect(
      screen.getByRole('heading', { level: 3, name: 'Runbook' }),
    ).toBeInTheDocument();
    expect(document.querySelector('.msg-text strong')?.textContent).toBe('RMAN');
    expect(document.querySelector('.msg-text code')?.textContent).toBe(
      'restore database',
    );
    expect(screen.getByText('validate backup')).toBeInTheDocument();
    expect(screen.getByText('open incident')).toBeInTheDocument();
    expect(screen.getByTestId('citation-1')).toBeInTheDocument();
  });

  it('hides <think> blocks from the rendered answer and exposes them separately', () => {
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[
          {
            id: 'th_think',
            title: 'Think',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['<think>private chain</think>Visible answer [1]'],
                sources: [
                  { n: 1, type: 'file' as const, name: 'runbook.pdf', score: 0.9 },
                ],
              },
            ],
          },
        ]}
      />,
    );

    expect(document.querySelector('.msg-text')).toHaveTextContent('Visible answer');
    expect(document.querySelector('.msg-text')).not.toHaveTextContent('private chain');
    expect(screen.getByTestId('retrieval-thinking-detail')).toHaveTextContent(
      'private chain',
    );
    expect(screen.queryByText(/<think>/)).toBeNull();
  });

  it('clicking a source card navigates to documents with a source filter for file paths', async () => {
    const onNavigate = vi.fn();
    render(<RetrievalTab {...defaultProps()} onNavigate={onNavigate} />);
    // The seed threads include an assistant message with the fixture
    // sources rendered after streaming completes.
    const sourceCards = document.querySelectorAll('[data-testid^="source-"]');
    expect(sourceCards.length).toBeGreaterThan(0);
    const fileCard = Array.from(sourceCards).find((card) =>
      card.textContent?.includes('.pdf'),
    ) as HTMLButtonElement | undefined;
    expect(fileCard).toBeDefined();
    await userEvent.click(fileCard!);
    expect(onNavigate).toHaveBeenCalledWith(
      'documents',
      expect.objectContaining({ source: expect.stringContaining('.pdf') }),
    );
  });

  it('clicking a source card forwards doc and chunk drill-down params', async () => {
    const onNavigate = vi.fn();
    render(
      <RetrievalTab
        {...defaultProps()}
        onNavigate={onNavigate}
        initialThreads={[
          {
            id: 'th_drilldown',
            title: 'Drilldown',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['answer [1]'],
                sources: [
                  {
                    n: 1,
                    type: 'file' as const,
                    name: 'runbook.pdf',
                    score: 0.9,
                    doc_id: 'doc-runbook',
                    chunk_id: 'chunk-runbook-2',
                  },
                ],
              },
            ],
          },
        ]}
      />,
    );

    await userEvent.click(screen.getByTestId('source-1'));

    expect(onNavigate).toHaveBeenCalledWith('documents', {
      source: 'runbook.pdf',
      doc: 'doc-runbook',
      chunk: 'chunk-runbook-2',
    });
  });

  it('forwards paragraph-anchor offsets in the drill-down params', async () => {
    const onNavigate = vi.fn();
    render(
      <RetrievalTab
        {...defaultProps()}
        onNavigate={onNavigate}
        initialThreads={[
          {
            id: 'th_anchor_drilldown',
            title: 'Anchor drilldown',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['answer [1]'],
                sources: [
                  {
                    n: 1,
                    type: 'file' as const,
                    name: 'runbook.pdf',
                    score: 0.9,
                    doc_id: 'doc-runbook',
                    chunk_id: 'chunk-runbook-2',
                    anchor: {
                      start: 216,
                      end: 640,
                      paragraph_idx: 1,
                      paragraph_count: 4,
                      confidence: 0.62,
                      method: 'lexical_overlap',
                    },
                  },
                ],
              },
            ],
          },
        ]}
      />,
    );

    await userEvent.click(screen.getByTestId('source-1'));

    expect(onNavigate).toHaveBeenCalledWith('documents', {
      source: 'runbook.pdf',
      doc: 'doc-runbook',
      chunk: 'chunk-runbook-2',
      astart: '216',
      aend: '640',
    });
  });

  it('clicking an inline citation forwards doc and chunk drill-down params', async () => {
    const onNavigate = vi.fn();
    render(
      <RetrievalTab
        {...defaultProps()}
        onNavigate={onNavigate}
        initialThreads={[
          {
            id: 'th_citation_drilldown',
            title: 'Citation drilldown',
            created: Date.now(),
            updated: Date.now(),
            messages: [
              {
                role: 'assistant',
                tokens: ['answer [1]'],
                sources: [
                  {
                    n: 1,
                    type: 'file' as const,
                    name: 'runbook.pdf',
                    score: 0.9,
                    doc_id: 'doc-runbook',
                    chunk_id: 'chunk-runbook-2',
                  },
                ],
              },
            ],
          },
        ]}
      />,
    );

    await userEvent.click(screen.getByTestId('citation-1'));

    expect(onNavigate).toHaveBeenCalledWith('documents', {
      source: 'runbook.pdf',
      doc: 'doc-runbook',
      chunk: 'chunk-runbook-2',
    });
  });

  it('source cards are disabled when no onNavigate prop is provided', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const sourceCards = document.querySelectorAll('[data-testid^="source-"]');
    expect(sourceCards.length).toBeGreaterThan(0);
    sourceCards.forEach((card) => {
      expect((card as HTMLButtonElement).disabled).toBe(true);
    });
  });
});

describe('RetrievalTab — TR-RET-02 answer_status surface', () => {
  it('hides the Sources panel and shows the cue when status=insufficient_information', async () => {
    // Backend returns the canonical fail path: insufficient_information
    // + empty sources. The cue must replace the Sources block.
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk("Sorry, I'm not able to provide an answer to that question.");
        return {
          response:
            "Sorry, I'm not able to provide an answer to that question.",
          sources: [],
          answer_status: 'insufficient_information' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(
      screen.getByLabelText('Query input'),
      'unanswerable',
    );
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(
        screen.queryByTestId('sources-empty-insufficient'),
      ).toBeInTheDocument(),
    );
    // The Sources header / list must NOT be rendered.
    expect(document.querySelector('.sources-header')).toBeNull();
    expect(document.querySelectorAll('[data-testid^="source-"]').length).toBe(
      0,
    );
  });

  it('hides Sources even if the backend leaks sources behind an insufficient status', async () => {
    // Regression guard: a future backend bug returning insufficient
    // status WITH non-empty sources must still hide the panel — we
    // never want sources presented as backing an unfounded answer.
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('Sorry, no answer.');
        return {
          response: 'Sorry, no answer.',
          sources: [
            { n: 1, type: 'file' as const, name: 'leaked.pdf', score: 0.21 },
          ],
          answer_status: 'insufficient_information' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(
      screen.getByLabelText('Query input'),
      'unanswerable',
    );
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(
        screen.queryByTestId('sources-empty-insufficient'),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('source-1')).toBeNull();
    expect(document.querySelector('.sources-header')).toBeNull();
  });

  it('keeps the answer but shows a cue when status=source_projection_failed', async () => {
    // #2: the answer is grounded but its references could not be projected.
    // Show the answer + a "sources unavailable" cue — never silently as
    // no-sources, and NOT as insufficient_information.
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('A grounded answer.');
        return {
          response: 'A grounded answer.',
          sources: [],
          answer_status: 'source_projection_failed' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'real question');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(
        screen.queryByTestId('sources-empty-projection-failed'),
      ).toBeInTheDocument(),
    );
    // The grounded answer is preserved.
    expect(document.querySelector('.retrieval-conv')?.textContent).toContain(
      'A grounded answer.',
    );
    // Not mislabeled as insufficient; no Sources panel.
    expect(screen.queryByTestId('sources-empty-insufficient')).toBeNull();
    expect(document.querySelector('.sources-header')).toBeNull();
  });

  it('suppresses Sources AND inline-citation navigation when they leak behind source_projection_failed', async () => {
    const onNavigate = vi.fn();
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('A grounded answer. [1]');
        return {
          response: 'A grounded answer. [1]',
          sources: [
            { n: 1, type: 'file' as const, name: 'leaked.pdf', score: 0.5 },
          ],
          answer_status: 'source_projection_failed' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onNavigate={onNavigate}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'real question');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(
        screen.queryByTestId('sources-empty-projection-failed'),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('source-1')).toBeNull();
    expect(document.querySelector('.sources-header')).toBeNull();
    // The [1] marker still renders, but the leaked source was dropped from
    // state, so it must render INERT (no button affordance) and clicking it
    // must NOT navigate to the unprojected document.
    expect(screen.queryByTestId('citation-1')).toBeNull();
    const inert = screen.getByTestId('citation-inert-1');
    expect(inert.tagName).toBe('SPAN');
    await userEvent.click(inert);
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it('keeps the answer but shows the no-retrieval cue and blocks leaked-source navigation', async () => {
    // only_need_context: the raw context body is shown, but the empty Sources
    // area must read as intentional (cue), not as
    // insufficient_information and not as a missing-sources glitch. A future
    // backend that leaks sources under this status must still not surface them
    // via the panel OR an inline citation.
    const onNavigate = vi.fn();
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('Direct answer. [1]');
        return {
          response: 'Direct answer. [1]',
          sources: [
            { n: 1, type: 'file' as const, name: 'leaked.pdf', score: 0.4 },
          ],
          answer_status: 'no_retrieval' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onNavigate={onNavigate}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'context question');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(
        screen.queryByTestId('sources-empty-no-retrieval'),
      ).toBeInTheDocument(),
    );
    expect(document.querySelector('.retrieval-conv')?.textContent).toContain(
      'Direct answer.',
    );
    // Not mislabeled as insufficient / projection-failed; no Sources panel.
    expect(screen.queryByTestId('sources-empty-insufficient')).toBeNull();
    expect(screen.queryByTestId('sources-empty-projection-failed')).toBeNull();
    expect(screen.queryByTestId('source-1')).toBeNull();
    expect(document.querySelector('.sources-header')).toBeNull();
    // Leaked source dropped from state -> the [1] marker is inert (no button)
    // and cannot navigate.
    expect(screen.queryByTestId('citation-1')).toBeNull();
    const inert = screen.getByTestId('citation-inert-1');
    expect(inert.tagName).toBe('SPAN');
    await userEvent.click(inert);
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it('renders an orphan [N] citation inert when no matching source exists', async () => {
    // Finding #2 (audit 2026-06-27): a [N] marker with no source in the list
    // (LLM hallucination like [99], or an external bibliographic ref) must NOT
    // look like a live Twin anchor. The grounded source [1] stays clickable;
    // the orphan [99] renders inert (a span, no navigation).
    const onNavigate = vi.fn();
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('Grounded on [1] but also cites [99].');
        return {
          response: 'Grounded on [1] but also cites [99].',
          sources: [
            { n: 1, type: 'file' as const, name: 'real.pdf', score: 0.7 },
          ],
          answer_status: 'grounded' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onNavigate={onNavigate}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'orphan cite');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    // The real source [1] is a live citation button.
    await waitFor(() =>
      expect(screen.getByTestId('citation-1')).toBeInTheDocument(),
    );
    // The orphan [99] is inert: no button, a span, and clicking it no-ops.
    expect(screen.queryByTestId('citation-99')).toBeNull();
    const orphan = screen.getByTestId('citation-inert-99');
    expect(orphan.tagName).toBe('SPAN');
    await userEvent.click(orphan);
    expect(onNavigate).not.toHaveBeenCalled();
    // The grounded source [1] still navigates.
    await userEvent.click(screen.getByTestId('citation-1'));
    expect(onNavigate).toHaveBeenCalledTimes(1);
  });

  it('keeps the answer but shows the query-failed cue and renders no citation affordance', async () => {
    // Finding #1 (audit 2026-06-27): a mid-stream backend error is reported as
    // answer_status=query_failed (NOT grounded). The error-notice text is shown
    // with an explicit failure cue, no Sources panel, and any [N] marker in the
    // notice is inert.
    const onNavigate = vi.fn();
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('[query failed: LLM down] see [1]');
        return {
          response: '[query failed: LLM down] see [1]',
          sources: [
            { n: 1, type: 'file' as const, name: 'leaked.pdf', score: 0.3 },
          ],
          answer_status: 'query_failed' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onNavigate={onNavigate}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'boom');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(
        screen.queryByTestId('sources-empty-query-failed'),
      ).toBeInTheDocument(),
    );
    expect(document.querySelector('.retrieval-conv')?.textContent).toContain(
      '[query failed: LLM down]',
    );
    // Not mislabeled, no Sources panel, no live citation.
    expect(screen.queryByTestId('sources-empty-insufficient')).toBeNull();
    expect(screen.queryByTestId('sources-empty-no-retrieval')).toBeNull();
    expect(screen.queryByTestId('source-1')).toBeNull();
    expect(document.querySelector('.sources-header')).toBeNull();
    expect(screen.queryByTestId('citation-1')).toBeNull();
    await userEvent.click(screen.getByTestId('citation-inert-1'));
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it('renders the Sources panel as before when status=grounded', async () => {
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('A real answer.');
        return {
          response: 'A real answer.',
          sources: [
            { n: 1, type: 'file' as const, name: 'runbook.pdf', score: 0.9 },
          ],
          answer_status: 'grounded' as const,
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'real question');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(screen.getByText(/^Sources$/)).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('sources-empty-insufficient')).toBeNull();
    expect(screen.getByTestId('source-1')).toBeInTheDocument();
  });

  it('treats a missing answer_status as grounded (back-compat)', async () => {
    // Legacy backends that haven't deployed the field must keep working.
    const onStreamQuery = vi.fn(
      async (_params, onChunk: (chunk: string) => void) => {
        onChunk('Legacy answer.');
        return {
          response: 'Legacy answer.',
          sources: [
            { n: 1, type: 'file' as const, name: 'legacy.pdf', score: 0.8 },
          ],
        };
      },
    );
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onStreamQuery={onStreamQuery}
      />,
    );

    await userEvent.type(screen.getByLabelText('Query input'), 'q');
    await userEvent.click(screen.getByRole('button', { name: /Send/ }));

    await waitFor(() =>
      expect(screen.getByText(/^Sources$/)).toBeInTheDocument(),
    );
    expect(screen.queryByTestId('sources-empty-insufficient')).toBeNull();
  });
});
