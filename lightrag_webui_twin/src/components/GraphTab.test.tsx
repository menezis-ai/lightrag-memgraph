/**
 * Unit tests for GraphTab.
 *
 * Covers: header counts, entity type toggle, search filter, node selection
 * updates detail panel, neighbor relation list, zoom in/out, Reset view,
 * Navigate to documents CTA.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { GraphTab } from './GraphTab';
import { useGraphRelations } from '../api/queries';
import {
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
} from '../fixtures';

function defaultProps() {
  return {
    entities: GRAPH_ENTITY_FIXTURES,
    relations: GRAPH_RELATION_FIXTURES,
    onNavigate: vi.fn(),
  };
}

function renderWithClient(ui: React.ReactElement) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

function renderGraphWithLiveRelations(
  fetchImpl: typeof fetch,
): ReturnType<typeof render> {
  const qc = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const Host = () => {
    const relations = useGraphRelations({ enabled: true });
    return (
      <GraphTab
        entities={GRAPH_ENTITY_FIXTURES}
        relations={relations.data ?? GRAPH_RELATION_FIXTURES}
        onNavigate={vi.fn()}
      />
    );
  };
  vi.stubGlobal('fetch', fetchImpl);
  return render(
    <QueryClientProvider client={qc}>
      <Host />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.history.replaceState(null, '', '/');
  window.localStorage.removeItem('twin.kg.pinned.v1');
});
afterEach(() => {
  window.history.replaceState(null, '', '/');
  window.localStorage.removeItem('twin.kg.pinned.v1');
  vi.unstubAllGlobals();
});

describe('GraphTab — rendering', () => {
  it('renders header with entity + relation counts', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    expect(screen.getByRole('heading', { name: 'Knowledge Graph' })).toBeInTheDocument();
    expect(
      screen.getByText(
        new RegExp(
          `${GRAPH_ENTITY_FIXTURES.length} entities · ${GRAPH_RELATION_FIXTURES.length} relations`,
        ),
      ),
    ).toBeInTheDocument();
  });

  it('renders all 6 entity-type filter rows with counts', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    ['PRODUCT', 'TECHNOLOGY', 'CONCEPT', 'ORG', 'PERSON', 'LOCATION'].forEach((t) => {
      expect(screen.getByTestId(`kg-type-${t}`)).toBeInTheDocument();
    });
  });

  it('renders the default selected entity (first fixture = e_oracle) in detail panel', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/Oracle Database/);
    expect(detail.textContent).toMatch(/Product/);
  });

  it('shows the neutral state when the selected entity no longer exists', () => {
    // Simulate the post-delete state: URL still carries `gent=kg_gone`
    // but `gone` is not in the entities[] array (deleted by cascade).
    // The inspector must surface the empty state, not silently fall
    // back to entities[0] (which was the 2026-06-08 prod bug).
    window.history.replaceState(null, '', '/?gent=kg_gone');
    renderWithClient(<GraphTab {...defaultProps()} />);
    expect(screen.getByText('Select a node to inspect')).toBeInTheDocument();
  });

  it('relation target name has truncating CSS so long names cannot overlap the strength badge', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    // e_oracle is selected by default + has neighbors. The structural
    // guarantee we want: each incoming/outgoing row wraps the target
    // name in a span with the kg-rel-target-name class so the CSS
    // (overflow: hidden + text-overflow: ellipsis on that span) can
    // bite. Without this wrapper the strength badge gets pushed past
    // the row and overlaps the arrow + label cluster.
    const nameSpans = document.querySelectorAll(
      '.kg-rel-row .kg-rel-target-name',
    );
    expect(nameSpans.length).toBeGreaterThan(0);
  });
});

describe('GraphTab — filters', () => {
  it('toggling PRODUCT off hides PRODUCT nodes', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    expect(screen.getByTestId('kg-node-e_oracle')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('kg-type-PRODUCT'));
    expect(screen.queryByTestId('kg-node-e_oracle')).toBeNull();
  });

  it('search filter narrows visible nodes by name', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search entities'), 'swift');
    // SWIFT entity matches name
    expect(screen.getByTestId('kg-node-e_swift')).toBeInTheDocument();
    // Oracle should be filtered out
    expect(screen.queryByTestId('kg-node-e_oracle')).toBeNull();
  });

  it('entity type counts are scoped by the active search filter', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search entities'), 'swift');

    expect(within(screen.getByTestId('kg-type-ORG')).getByText('1')).toBeInTheDocument();
    expect(within(screen.getByTestId('kg-type-CONCEPT')).getByText('1')).toBeInTheDocument();
    expect(within(screen.getByTestId('kg-type-PRODUCT')).getByText('0')).toBeInTheDocument();
  });

  it('document filter narrows visible nodes by source document', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          {
            ...GRAPH_ENTITY_FIXTURES[0],
            source_docs: ['doc-oracle'],
          },
          {
            ...GRAPH_ENTITY_FIXTURES[8],
            source_docs: ['doc-memgraph'],
          },
        ]}
        relations={[]}
      />,
    );

    await userEvent.click(screen.getByLabelText('Filter by document'));
    await userEvent.click(screen.getByTestId('kg-pick-doc-memgraph'));

    expect(screen.queryByTestId(`kg-node-${GRAPH_ENTITY_FIXTURES[0].id}`)).toBeNull();
    expect(screen.getByTestId(`kg-node-${GRAPH_ENTITY_FIXTURES[8].id}`)).toBeInTheDocument();
  });

  it('header shows the active folder label and hides the segment when unset', () => {
    const { unmount } = renderWithClient(
      <GraphTab {...defaultProps()} folderLabel="Run & Ops KB" />,
    );
    expect(screen.getByText('Run & Ops KB')).toBeInTheDocument();
    unmount();

    renderWithClient(<GraphTab {...defaultProps()} />);
    expect(document.querySelector('.kg-sub')?.textContent).not.toContain('folder');
  });

  it('tag filter offers and matches tags inherited from source documents', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          {
            ...GRAPH_ENTITY_FIXTURES[0],
            tags: undefined,
            source_docs: ['doc-tagged'],
          },
          {
            ...GRAPH_ENTITY_FIXTURES[8],
            tags: undefined,
            source_docs: ['doc-untagged'],
          },
        ]}
        relations={[]}
        docTags={{ 'doc-tagged': ['database'] }}
      />,
    );

    // The doc-level tag is offered by the picker even though no entity
    // carries twin tags of its own (production ingestion path).
    await userEvent.click(screen.getByLabelText('Filter by tag'));
    await userEvent.type(screen.getByLabelText('Filter by tag'), 'data');
    await userEvent.click(screen.getByTestId('kg-pick-database'));

    expect(
      screen.getByTestId(`kg-node-${GRAPH_ENTITY_FIXTURES[0].id}`),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId(`kg-node-${GRAPH_ENTITY_FIXTURES[8].id}`),
    ).toBeNull();
  });

  it('tag filter offers canonical catalog tags even before entity propagation', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          {
            ...GRAPH_ENTITY_FIXTURES[0],
            tags: undefined,
            source_docs: [],
          },
        ]}
        relations={[]}
        tagCatalog={['semantic']}
      />,
    );

    await userEvent.click(screen.getByLabelText('Filter by tag'));
    await userEvent.type(screen.getByLabelText('Filter by tag'), 'sem');

    expect(screen.getByTestId('kg-pick-semantic')).toBeInTheDocument();
  });

  it('document filter shows file names from docLabels, falling back to the raw id', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          {
            ...GRAPH_ENTITY_FIXTURES[0],
            source_docs: ['doc-65b39d5035d5ba1aa5a4c681c87d1d80'],
          },
          {
            ...GRAPH_ENTITY_FIXTURES[8],
            source_docs: ['doc-unmapped'],
          },
        ]}
        relations={[]}
        docLabels={{
          'doc-65b39d5035d5ba1aa5a4c681c87d1d80': 'oracle-restart-procedure.pdf',
        }}
      />,
    );

    await userEvent.click(screen.getByLabelText('Filter by document'));
    const mapped = screen.getByTestId('kg-pick-doc-65b39d5035d5ba1aa5a4c681c87d1d80');
    expect(mapped).toHaveTextContent('oracle-restart-procedure.pdf');
    expect(mapped).not.toHaveTextContent('doc-65b39d5035d5ba1aa5a4c681c87d1d80');
    expect(screen.getByTestId('kg-pick-doc-unmapped')).toHaveTextContent('doc-unmapped');

    // Selected pill also shows the file name, with the raw id kept on title.
    await userEvent.click(mapped);
    expect(
      screen.getByTestId('kg-picked-doc-65b39d5035d5ba1aa5a4c681c87d1d80'),
    ).toHaveTextContent('oracle-restart-procedure.pdf');
  });

  it('no-match search shows empty state with Clear filter CTA', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.type(
      screen.getByLabelText('Search entities'),
      'zzz-no-such-entity',
    );
    expect(
      screen.getByText('No entities match the current filter.'),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Clear filter' }));
    expect(screen.getByTestId('kg-node-e_oracle')).toBeInTheDocument();
  });
});

describe('GraphTab — selection + detail', () => {
  it('clicking a node updates the selected entity in the detail panel', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    expect(detail.textContent).toMatch(/Memgraph/);
    expect(detail.textContent).toMatch(/Graph DB backing LightRAG/);
  });

  it('detail panel lists outgoing + incoming relations for the selected entity', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    // e_oracle has outgoing: e_rhel, e_pga, e_vmware. Incoming: rman, archlog, marc, iso20022.
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/Outgoing \(3\)/);
    expect(detail.textContent).toMatch(/Incoming \(4\)/);
  });

  it('clicking a relation row opens the RelationEditor, then endpoint switches selection', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    // Outgoing RUNS_ON → click the relation row to open the relation editor.
    // The new EntityEditor doctrine: row click goes to relation editor first
    // (mirrors the JSX maquette), endpoint click selects the target entity.
    const rhelRelRow = Array.from(document.querySelectorAll('[role="button"]')).find(
      (el) => el.textContent?.includes('RHEL 9') && el.className.includes('kg-rel-row'),
    );
    expect(rhelRelRow).toBeDefined();
    await userEvent.click(rhelRelRow! as HTMLElement);
    // Now the relation editor is visible.
    const relPanel = document.querySelector('[data-testid="kg-detail-relation"]');
    expect(relPanel).toBeTruthy();
    // Click the RHEL 9 endpoint → switch the entity selection back to that node.
    const rhelEndpoint = Array.from(
      relPanel!.querySelectorAll('button.kg-rel-endpoint'),
    ).find((b) => b.textContent?.includes('RHEL 9'));
    expect(rhelEndpoint).toBeDefined();
    await userEvent.click(rhelEndpoint! as HTMLElement);
    const entityPanel = document.querySelector('[data-testid="kg-detail-entity"]');
    expect(entityPanel?.textContent).toMatch(/RHEL 9/);
    expect(entityPanel?.textContent).toMatch(/Red Hat Enterprise Linux/);
  });

  it('"View documents" CTA navigates with the entity name as the query', async () => {
    // Document filtering is available in the graph rail. This CTA stays a
    // broader documents-tab query so it works for generated entities that
    // only have a name and no projected source_docs.
    const p = defaultProps();
    renderWithClient(<GraphTab {...p} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    const cta = Array.from(detail.querySelectorAll('button')).find((b) =>
      b.textContent?.match(/View documents mentioning/),
    );
    expect(cta).toBeDefined();
    await userEvent.click(cta!);
    expect(p.onNavigate).toHaveBeenCalledTimes(1);
    const [tabArg, paramsArg] = p.onNavigate.mock.calls[0];
    expect(tabArg).toBe('documents');
    expect(paramsArg).toHaveProperty('q');
    expect(typeof paramsArg.q).toBe('string');
  });

  it('pinning an entity persists in localStorage and is restored on remount', async () => {
    const { unmount } = renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    await userEvent.click(screen.getByTestId('kg-entity-pin'));
    expect(screen.getByTestId('kg-entity-pin')).toHaveTextContent('Pinned');
    expect(JSON.parse(window.localStorage.getItem('twin.kg.pinned.v1') ?? '[]')).toContain(
      'e_memgraph',
    );

    unmount();
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    expect(screen.getByTestId('kg-entity-pin')).toHaveTextContent('Pinned');
  });
});

describe('GraphTab — zoom + reset', () => {
  it('zoom in button increases zoom percentage', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const value0 = screen.getByTestId('kg-zoom-value').textContent;
    expect(value0).toBe('100%');
    await userEvent.click(screen.getByLabelText('Zoom in'));
    const value1 = screen.getByTestId('kg-zoom-value').textContent;
    expect(value1).toBe('118%');
  });

  it('zoom out button decreases zoom percentage', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByLabelText('Zoom out'));
    expect(screen.getByTestId('kg-zoom-value').textContent).toBe('85%');
  });

  it('Reset view restores zoom to 100%', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByLabelText('Zoom in'));
    await userEvent.click(screen.getByLabelText('Zoom in'));
    await userEvent.click(screen.getByRole('button', { name: /Reset view/ }));
    expect(screen.getByTestId('kg-zoom-value').textContent).toBe('100%');
  });

  it('Reset view clears search and type filters', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search entities'), 'swift');
    await userEvent.click(screen.getByTestId('kg-type-PRODUCT'));

    await userEvent.click(screen.getByRole('button', { name: /Reset view/ }));

    expect(screen.getByLabelText('Search entities')).toHaveValue('');
    expect(screen.getByTestId('kg-type-PRODUCT')).toHaveClass('is-on');
  });

  it('wheel over the canvas zooms the graph and prevents page scroll', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const canvas = screen.getByTestId('kg-canvas');
    const event = new WheelEvent('wheel', {
      deltaY: -100,
      bubbles: true,
      cancelable: true,
    });
    canvas.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
    await waitFor(() =>
      expect(screen.getByTestId('kg-zoom-value').textContent).toBe('110%'),
    );
  });
});

describe('GraphTab — lifecycle: Add entity', () => {
  it('Add entity button opens the inline form', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    expect(screen.queryByTestId('kg-add-entity-form')).toBeNull();
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    expect(screen.getByTestId('kg-add-entity-form')).toBeInTheDocument();
  });

  it('blocks submit when no name typed', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    expect(screen.getByTestId('kg-add-entity-submit')).toBeDisabled();
  });

  it('flags a duplicate name from the existing entities', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(
      screen.getByTestId('kg-add-entity-name'),
      'Oracle Database',
    );
    expect(screen.getByTestId('kg-add-entity-duplicate')).toBeInTheDocument();
    expect(screen.getByTestId('kg-add-entity-submit')).toBeDisabled();
  });

  it('Cancel closes the form without submitting', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.click(screen.getByRole('button', { name: /^Cancel$/ }));
    expect(screen.queryByTestId('kg-add-entity-form')).toBeNull();
  });

  it('surfaces create failures instead of leaving Add silent', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: 'backend unavailable' }), {
          status: 500,
          statusText: 'Server Error',
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(screen.getByTestId('kg-add-entity-name'), 'New Entity');
    await userEvent.click(screen.getByTestId('kg-add-entity-submit'));

    expect(await screen.findByTestId('kg-add-entity-error')).toHaveTextContent(
      'POST /twin/api/graph/entities',
    );
  });
});

describe('GraphTab — lifecycle: Delete entity', () => {
  it('first click arms the delete, second click is the confirmation step', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const btn = screen.getByTestId('kg-entity-delete');
    expect(btn.textContent).toMatch(/Delete entity/);

    await userEvent.click(btn);
    expect(btn.textContent).toMatch(/Click again to confirm/);
    expect(screen.getByTestId('kg-entity-delete-cancel')).toBeInTheDocument();
  });

  it('Cancel rolls back the armed state', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-entity-delete'));
    await userEvent.click(screen.getByTestId('kg-entity-delete-cancel'));
    expect(screen.getByTestId('kg-entity-delete').textContent).toMatch(
      /Delete entity/,
    );
  });

  it('switching to another entity disarms a pending delete', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-entity-delete'));
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    expect(screen.getByTestId('kg-entity-delete').textContent).toMatch(
      /Delete entity/,
    );
  });
});

describe('GraphTab — lifecycle: Delete relation', () => {
  it('first click arms, second click is the confirmation step', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    // Open an outgoing relation row from the default-selected Oracle entity.
    const rhelRow = Array.from(
      document.querySelectorAll('[role="button"]'),
    ).find(
      (el) =>
        el.textContent?.includes('RHEL 9') &&
        el.className.includes('kg-rel-row'),
    );
    expect(rhelRow).toBeDefined();
    await userEvent.click(rhelRow! as HTMLElement);

    const btn = screen.getByTestId('kg-rel-delete');
    expect(btn.textContent).toMatch(/Delete relation/);

    await userEvent.click(btn);
    expect(btn.textContent).toMatch(/Click again to confirm/);
    expect(screen.getByTestId('kg-rel-delete-cancel')).toBeInTheDocument();
  });

  it('updates the header relation count after deleting a relation', async () => {
    const deletedId = 'r_01';
    const remainingRelations = GRAPH_RELATION_FIXTURES.filter(
      (r) => r.id !== deletedId,
    );
    let relationDeleted = false;
    const fetchMock = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      const href = String(url);
      if (href.includes('/graph/relations') && init?.method === 'DELETE') {
        relationDeleted = true;
        return new Response(null, { status: 204 });
      }
      if (href.includes('/graph/relations')) {
        return new Response(
          JSON.stringify(
            relationDeleted ? remainingRelations : GRAPH_RELATION_FIXTURES,
          ),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }) as unknown as typeof fetch;

    renderGraphWithLiveRelations(fetchMock);

    await waitFor(() => {
      expect(
        screen.getByText(
          new RegExp(`${GRAPH_RELATION_FIXTURES.length} relations`),
        ),
      ).toBeInTheDocument();
    });

    await userEvent.click(screen.getByTestId(`kg-rel-row-${deletedId}`));
    const btn = screen.getByTestId('kg-rel-delete');
    await userEvent.click(btn);
    await userEvent.click(btn);

    await waitFor(() => {
      expect(
        screen.getByText(new RegExp(`${remainingRelations.length} relations`)),
      ).toBeInTheDocument();
    });
  });
});

describe('GraphTab — lifecycle: Add relation', () => {
  it('Add relation button opens the inline form scoped to the selected entity', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    expect(screen.queryByTestId('kg-add-rel-form')).toBeNull();
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    expect(screen.getByTestId('kg-add-rel-form')).toBeInTheDocument();
    // From label shows the active entity (default = Oracle Database)
    expect(screen.getByText('From').parentElement?.textContent).toMatch(
      /Oracle Database/,
    );
  });

  it('blocks submit when label is empty', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    expect(screen.getByTestId('kg-add-rel-submit')).toBeDisabled();
  });

  it('flags a duplicate outgoing relation between same endpoints', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    // Default-selected entity is Oracle (e_oracle); pick RHEL 9 as target —
    // there is already RUNS_ON between them in the fixtures.
    await userEvent.selectOptions(
      screen.getByTestId('kg-add-rel-target'),
      'e_rhel',
    );
    await userEvent.type(screen.getByTestId('kg-add-rel-label'), 'RUNS_ON');
    expect(screen.getByTestId('kg-add-rel-duplicate')).toBeInTheDocument();
    expect(screen.getByTestId('kg-add-rel-submit')).toBeDisabled();
  });

  it('Cancel closes the form', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    await userEvent.click(screen.getByRole('button', { name: /^Cancel$/ }));
    expect(screen.queryByTestId('kg-add-rel-form')).toBeNull();
  });

  it('switching entity disarms a pending Add relation form', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    expect(screen.getByTestId('kg-add-rel-form')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    expect(screen.queryByTestId('kg-add-rel-form')).toBeNull();
  });
});
