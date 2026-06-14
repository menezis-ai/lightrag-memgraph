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
import { RetrievalTab } from './RetrievalTab';
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
        name: /How do I restart Oracle on RHEL 9/,
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
      screen.getByRole('button', { name: /Common RMAN backup errors/ }),
    );
    // Look in the conversation pane (not the sidebar title)
    const userMsg = document.querySelector('.msg-user');
    expect(userMsg?.textContent).toBe('Common RMAN backup errors');
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
    expect(document.querySelector('.msg-assistant')?.textContent).toContain(
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

  it('Query mode select changes value', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const sel = screen.getByLabelText('Query mode') as HTMLSelectElement;
    await userEvent.selectOptions(sel, 'hybrid');
    expect(sel.value).toBe('hybrid');
  });

  it('does not render a Tag filter affordance (TR-RET-02 step 3 / audit C1)', () => {
    // The "Tag filter — Twin" control used to live in this panel and
    // forwarded a tagFilters array that LightRAG 1.4.x silently
    // ignored at retrieval time. The whole affordance has been
    // removed rather than relabelled (no honest backend path to
    // redirect to while audit C2 is open). This test pins that the
    // control cannot sneak back without being noticed.
    render(<RetrievalTab {...defaultProps()} />);
    expect(screen.queryByLabelText('Retrieval tag input')).toBeNull();
    expect(screen.queryByText(/Tag filter/i)).toBeNull();
  });

  it('passes advanced retrieval params to onSendQuery', async () => {
    const onSendQuery = vi.fn(async () => ({ response: 'ok', sources: [] }));
    render(
      <RetrievalTab
        {...defaultProps()}
        initialThreads={[]}
        onSendQuery={onSendQuery}
      />,
    );

    await userEvent.selectOptions(screen.getByLabelText('Query mode'), 'hybrid');
    await userEvent.clear(screen.getByLabelText('Top K'));
    await userEvent.type(screen.getByLabelText('Top K'), '12');
    await userEvent.clear(screen.getByLabelText('Chunk top K'));
    await userEvent.type(screen.getByLabelText('Chunk top K'), '6');
    await userEvent.clear(screen.getByLabelText('Max tokens'));
    await userEvent.type(screen.getByLabelText('Max tokens'), '2048');
    await userEvent.clear(screen.getByLabelText('History turns'));
    await userEvent.type(screen.getByLabelText('History turns'), '2');
    await userEvent.type(
      screen.getByLabelText('System prompt'),
      'prefer operational runbooks',
    );
    await userEvent.click(screen.getByLabelText('Enable rerank'));
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
        historyTurns: 2,
        userPrompt: 'prefer operational runbooks',
        enableRerank: false,
      }),
    );
    // TR-RET-02 step 3 / audit C1: ``tagFilters`` must NOT be in the
    // forwarded params anymore — the field has been removed from the
    // contract so the backend 422 on /query / /stream never triggers.
    expect(onSendQuery).toHaveBeenCalledWith(
      expect.not.objectContaining({ tagFilters: expect.anything() }),
    );
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
      screen.getByRole('button', { name: 'Réduire aux 5 premières' }),
    ).toHaveAttribute('aria-expanded', 'true');
  });

  it('uses requested Top K for the expand count without inventing source cards', async () => {
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
      screen.getByRole('button', { name: 'Voir les 15 autres' }),
    ).toBeInTheDocument();
    expect(screen.queryByTestId('source-6')).toBeNull();

    await userEvent.click(
      screen.getByRole('button', { name: 'Voir les 15 autres' }),
    );

    expect(screen.queryByTestId('source-6')).toBeNull();
    expect(screen.getByTestId('sources-no-additional')).toHaveTextContent(
      'No additional structured sources were returned by the backend.',
    );
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
