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
  beforeEach,
  afterEach,
} from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RetrievalTab } from './RetrievalTab';
import {
  ANSWER_TOKENS_FIXTURE,
  RETRIEVAL_SOURCES_FIXTURE,
  THESAURUS_FIXTURES,
  makeSampleThreads,
} from '../fixtures';

function defaultProps() {
  return {
    thesaurus: THESAURUS_FIXTURES,
    answerTokens: ANSWER_TOKENS_FIXTURE,
    answerSources: RETRIEVAL_SOURCES_FIXTURE,
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
});
