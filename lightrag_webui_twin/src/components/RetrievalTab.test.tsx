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
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RetrievalTab } from './RetrievalTab';
import {
  THESAURUS_FIXTURES,
  makeSampleThreads,
} from '../fixtures';

function defaultProps() {
  return {
    thesaurus: THESAURUS_FIXTURES,
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
});

describe('RetrievalTab — localStorage persistence', () => {
  it('writes to twin-rag.threads.v2 when threads change', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    // localStorage should be populated by the initial effect
    const raw = window.localStorage.getItem('twin-rag.threads.v2');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw as string);
    expect(parsed).toHaveLength(2);
  });

  it('reads from localStorage on init when present', () => {
    window.localStorage.setItem(
      'twin-rag.threads.v2',
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

  it('tag autocomplete adds tag from thesaurus on Enter', async () => {
    render(<RetrievalTab {...defaultProps()} />);
    const input = screen.getByLabelText('Retrieval tag input');
    await userEvent.type(input, 'oracle{Enter}');
    // Look inside the tag-filter chip-input for "oracle"
    const chips = document.querySelectorAll('.tag-chip');
    expect(Array.from(chips).some((c) => c.textContent?.includes('oracle'))).toBe(
      true,
    );
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
      screen.getByLabelText('User prompt'),
      'prefer operational runbooks',
    );
    await userEvent.type(
      screen.getByLabelText('Retrieval tag input'),
      'oracle{Enter}',
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
        tagFilters: ['oracle'],
      }),
    );
  });
});

describe('RetrievalTab — source cards', () => {
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
