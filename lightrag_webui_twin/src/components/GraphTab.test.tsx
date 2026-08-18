/**
 * Unit tests for GraphTab.
 *
 * Covers: header counts, entity type toggle, search filter, node selection
 * updates detail panel, neighbor relation list, zoom in/out, Reset view,
 * Navigate to documents CTA.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { useState } from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { DocumentsStatusFilterKey } from '../app/appConstants';
import { useAppNavigation, type DetailRequest } from '../app/useAppNavigation';
import type { Document } from '../types/document';
import { DocumentsTab } from './DocumentsTab';
import { GraphTab } from './GraphTab';
import { useGraphRelations } from '../api/queries';
import {
  DOCUMENT_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  TAG_FIXTURES,
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

function GraphDocumentsDrilldownHarness() {
  const [tab, setTab] = useState('graph');
  const [documentsStatusFilter, setDocumentsStatusFilter] =
    useState<DocumentsStatusFilterKey>('all');
  const [documentsSearch, setDocumentsSearch] = useState('');
  const [documentsTagFilters, setDocumentsTagFilters] = useState<readonly string[]>(
    [],
  );
  const [documentsSourceFilters, setDocumentsSourceFilters] = useState<
    readonly string[]
  >([]);
  const [, setDetailDoc] = useState<Document | null>(null);
  const [, setDetailChunkId] = useState<string | null>(null);
  const [, setDetailRequest] = useState<DetailRequest | null>(null);
  const [, setReadSourceDoc] = useState<Document | null>(null);
  const [, setRetagDoc] = useState<Document | null>(null);
  const [, setRetagBulk] = useState<readonly Document[] | null>(null);
  const [, setFolderState] = useState('default');
  const [, setReadNotificationIds] = useState<
    ReadonlySet<string>
  >(() => new Set());
  const [, setClearedNotificationIds] = useState<
    ReadonlySet<string>
  >(() => new Set());

  const { onNavigate } = useAppNavigation({
    setClearedNotificationIds,
    setDetailChunkId,
    setDetailDoc,
    setDetailRequest,
    setDocumentsSearch,
    setDocumentsSourceFilters,
    setDocumentsStatusFilter,
    setDocumentsTagFilters,
    setFolderState,
    setReadNotificationIds,
    setReadSourceDoc,
    setRetagBulk,
    setRetagDoc,
    setTab,
  });

  const entities = [
    { ...GRAPH_ENTITY_FIXTURES[0], source_docs: ['d1'] },
    { ...GRAPH_ENTITY_FIXTURES[8], source_docs: ['d4'] },
  ];
  const docLabels = {
    d1: 'oracle-restart-procedure.pdf',
    d4: 'memgraph-mage-3.8-release-notes.md',
  };

  if (tab === 'graph') {
    return (
      <GraphTab
        entities={entities}
        relations={[]}
        docLabels={docLabels}
        onNavigate={onNavigate}
      />
    );
  }

  return (
    <>
      <button type="button" onClick={() => setTab('graph')}>
        Back to graph
      </button>
      <DocumentsTab
        docs={DOCUMENT_FIXTURES}
        tagCatalog={TAG_FIXTURES}
        statusFilter={documentsStatusFilter}
        onStatusFilterChange={setDocumentsStatusFilter}
        search={documentsSearch}
        onSearchChange={setDocumentsSearch}
        tagFilters={documentsTagFilters}
        onTagFiltersChange={setDocumentsTagFilters}
        sourceFilters={documentsSourceFilters}
        onSourceFiltersChange={setDocumentsSourceFilters}
        onOpenAdd={vi.fn()}
        onOpenRetag={vi.fn()}
        onOpenBulkRetag={vi.fn()}
        onAddToast={vi.fn()}
      />
    </>
  );
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

  it('renders an edit icon in the entity editor action', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const edit = screen.getByTestId('kg-entity-edit');
    expect(edit.querySelector('svg[data-icon="edit"] path')).toBeTruthy();
  });

  it('draws relation arrows to the node surface instead of the node centre', () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    const firstRelation = GRAPH_RELATION_FIXTURES[0];
    const target = GRAPH_ENTITY_FIXTURES.find(
      (entity) => entity.id === firstRelation.target,
    );
    const line = document.querySelector('.kg-edge line') as SVGLineElement;
    expect(target).toBeDefined();
    expect(line).toBeTruthy();
    expect(Number(line.getAttribute('x2'))).not.toBe(target?.x);
    expect(Number(line.getAttribute('y2'))).not.toBe(target?.y);
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
  it('excludes deprecated tags from suggestions and marks historical occurrences', async () => {
    window.history.replaceState(null, '', '/?gtag=retired');
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          { ...GRAPH_ENTITY_FIXTURES[0], tags: ['retired'] },
        ]}
        relations={[]}
        tagCatalog={[
          { tag: 'oracle', tier: 1, status: 'active' },
          { tag: 'retired', tier: 1, status: 'deprecated' },
        ]}
      />,
    );

    const historicalFilter = screen.getByTestId('kg-picked-retired');
    expect(historicalFilter).toHaveClass('is-deprecated');
    expect(within(historicalFilter).getByText('deprecated')).toBeInTheDocument();
    expect(document.querySelector('.kg-detail .tag-chip.is-deprecated')).toHaveTextContent(
      'retireddeprecated',
    );

    await userEvent.click(screen.getByLabelText('Filter by tag'));
    expect(screen.getByTestId('kg-pick-oracle')).toBeInTheDocument();
    expect(screen.queryByTestId('kg-pick-retired')).toBeNull();
  });

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

  it('transfers active tag and document filters to Retrieval URL params', async () => {
    const onNavigate = vi.fn();
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          {
            ...GRAPH_ENTITY_FIXTURES[0],
            tags: ['oracle'],
            source_docs: ['doc-oracle'],
          },
          {
            ...GRAPH_ENTITY_FIXTURES[8],
            tags: ['memgraph'],
            source_docs: ['doc-memgraph'],
          },
        ]}
        relations={[]}
        tagCatalog={['oracle', 'memgraph']}
        onNavigate={onNavigate}
      />,
    );

    await userEvent.click(screen.getByLabelText('Filter by tag'));
    await userEvent.click(screen.getByTestId('kg-pick-oracle'));
    await userEvent.click(screen.getByLabelText('Filter by document'));
    await userEvent.click(screen.getByTestId('kg-pick-doc-oracle'));
    await userEvent.click(screen.getByTestId('kg-transfer-filters'));

    expect(onNavigate).toHaveBeenCalledWith('retrieval', {
      rtag: 'oracle',
      rtagmode: 'any',
      rdoc: 'doc-oracle',
      rdocmode: 'any',
    });
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
    // The new EntityEditor doctrine: row click goes to relation editor first,
    // endpoint click selects the target entity.
    const rhelRelRow = Array.from(document.querySelectorAll('button.kg-rel-row')).find(
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

  it('"View documents" CTA navigates with exact source_docs when available', async () => {
    const p = defaultProps();
    renderWithClient(
      <GraphTab
        {...p}
        entities={[
          {
            ...GRAPH_ENTITY_FIXTURES[0],
            source_docs: ['doc-oracle'],
          },
        ]}
        relations={[]}
        docLabels={{ 'doc-oracle': 'oracle-restart-procedure.pdf' }}
      />,
    );
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    const cta = Array.from(detail.querySelectorAll('button')).find((b) =>
      b.textContent?.match(/View documents/),
    );
    expect(cta).toBeDefined();
    await userEvent.click(cta!);
    expect(p.onNavigate).toHaveBeenCalledTimes(1);
    const [tabArg, paramsArg] = p.onNavigate.mock.calls[0];
    expect(tabArg).toBe('documents');
    expect(paramsArg).toEqual({
      doc: 'doc-oracle',
      source: 'oracle-restart-procedure.pdf',
    });
  });

  it('"View documents" CTA falls back to text query without source_docs', async () => {
    const p = defaultProps();
    renderWithClient(<GraphTab {...p} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    const cta = Array.from(detail.querySelectorAll('button')).find((b) =>
      b.textContent?.match(/View documents/),
    );
    expect(cta).toBeDefined();
    await userEvent.click(cta!);
    expect(p.onNavigate).toHaveBeenCalledTimes(1);
    const [tabArg, paramsArg] = p.onNavigate.mock.calls[0];
    expect(tabArg).toBe('documents');
    expect(paramsArg).toEqual({ q: GRAPH_ENTITY_FIXTURES[0].name });
  });

  it('"View documents" refreshes Documents filters when switching from entity A to entity B', async () => {
    renderWithClient(<GraphDocumentsDrilldownHarness />);

    await userEvent.click(
      screen.getByRole('button', {
        name: /View documents mentioning this entity/i,
      }),
    );
    expect(await screen.findByTestId('docs-row-d1')).toBeInTheDocument();
    expect(screen.getByTestId('source-filter-oracle-restart-procedure.pdf'))
      .toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d4')).toBeNull();

    await userEvent.click(screen.getByRole('button', { name: 'Back to graph' }));
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    await userEvent.click(
      screen.getByRole('button', {
        name: /View documents mentioning this entity/i,
      }),
    );

    expect(await screen.findByTestId('docs-row-d4')).toBeInTheDocument();
    expect(
      screen.getByTestId('source-filter-memgraph-mage-3.8-release-notes.md'),
    ).toBeInTheDocument();
    expect(screen.queryByTestId('docs-row-d1')).toBeNull();
  });

  it('saves Entity type changes through the graph entity PATCH payload', async () => {
    const fetchMock = vi.fn(async (...args: [RequestInfo | URL, RequestInit?]) => {
      const [url, init] = args;
      const href = String(url);
      if (href.includes('/graph/entities/e_oracle') && init?.method === 'PATCH') {
        return new Response(
          JSON.stringify({ ...GRAPH_ENTITY_FIXTURES[0], type: 'PERSON' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }
      return new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithClient(<GraphTab {...defaultProps()} />);

    await userEvent.click(screen.getByTestId('kg-entity-edit'));
    await userEvent.selectOptions(screen.getByLabelText('Entity type'), 'PERSON');
    await userEvent.click(screen.getByTestId('kg-entity-save'));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/graph/entities/e_oracle'),
        expect.objectContaining({
          method: 'PATCH',
          body: expect.stringContaining('"type":"PERSON"'),
        }),
      );
    });
  });

  it('shows node tag suggestions when the edit field receives focus', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        tagCatalog={['oracle', 'production']}
      />,
    );

    await userEvent.click(screen.getByTestId('kg-entity-edit'));
    await userEvent.click(screen.getByLabelText('Add node tag'));

    expect(screen.getByTestId('kg-tag-suggestions')).toHaveTextContent('oracle');
  });

  it('keyboard-selects a tag filter suggestion in graph rail', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        tagCatalog={['oracle', 'production', 'rman']}
      />,
    );

    const filterInput = screen.getByLabelText('Filter by tag');
    await userEvent.click(filterInput);
    expect(
      await screen.findByRole('listbox', { name: 'Filter by tag suggestions' }),
    ).toBeInTheDocument();
    await userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}');

    expect(String(filterInput.getAttribute('aria-activedescendant'))).toMatch(
      /^kg-tag-filter-suggestions-option-\d+$/,
    );

    expect(document.querySelector('[data-testid^="kg-picked-"]')).not.toBeNull();
  });

  it('suggests existing entity property keys when adding metadata', async () => {
    renderWithClient(
      <GraphTab
        {...defaultProps()}
        entities={[
          { ...GRAPH_ENTITY_FIXTURES[0], properties: {} },
          { ...GRAPH_ENTITY_FIXTURES[1], properties: { owner: 'dba' } },
        ]}
        relations={[]}
      />,
    );

    await userEvent.click(screen.getByTestId('kg-entity-edit'));
    await userEvent.click(screen.getByLabelText('New property key'));

    expect(
      document.querySelector('datalist#kg-prop-key-suggestions option[value="owner"]'),
    ).toBeTruthy();
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
  it('focus mode limits the canvas to the selected entity and one-hop neighbors', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);

    expect(screen.getByTestId('kg-node-e_cypher')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('kg-focus-mode'));

    expect(screen.getByTestId('kg-focus-mode')).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(screen.getByTestId('kg-node-e_oracle')).toBeInTheDocument();
    expect(screen.getByTestId('kg-node-e_rman')).toBeInTheDocument();
    expect(screen.queryByTestId('kg-node-e_cypher')).toBeNull();

    await userEvent.click(screen.getByTestId('kg-focus-mode'));
    expect(screen.getByTestId('kg-node-e_cypher')).toBeInTheDocument();
  });

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

  // TR-KG-01 follow-up: each backend error class lands on a distinct
  // inline copy. The pre-PR behaviour was a single generic message
  // ("POST /twin/api/graph/entities <status>") that gave the operator
  // no actionable cue.

  function stubCreateEntityResponse(status: number, body: unknown): void {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(body), {
          status,
          statusText: 'Mocked',
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
  }

  it('surfaces a 409 duplicate with a named, actionable inline copy', async () => {
    stubCreateEntityResponse(409, {
      detail: "Graph entity 'New Entity' already exists",
    });
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(
      screen.getByTestId('kg-add-entity-name'),
      'New Entity',
    );
    await userEvent.click(screen.getByTestId('kg-add-entity-submit'));

    const err = await screen.findByTestId('kg-add-entity-error');
    expect(err.textContent).toMatch(/already exists/i);
    expect(err).toHaveTextContent('New Entity');
    // Form stays open so the operator can amend the name.
    expect(screen.getByTestId('kg-add-entity-form')).toBeInTheDocument();
  });

  it('surfaces a 409 pipeline-busy refusal as a global toast', async () => {
    stubCreateEntityResponse(409, {
      detail: 'Pipeline is busy. Please try again later',
    });
    const onToast = vi.fn();
    renderWithClient(<GraphTab {...defaultProps()} onToast={onToast} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(
      screen.getByTestId('kg-add-entity-name'),
      'New Entity',
    );
    await userEvent.click(screen.getByTestId('kg-add-entity-submit'));

    const err = await screen.findByTestId('kg-add-entity-error');
    expect(err).toHaveTextContent('Action not taken');
    expect(err).toHaveTextContent('ingestion pipeline is busy');
    expect(err).not.toHaveTextContent('already exists');
    expect(onToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'error',
        title: 'Graph update failed',
        sub: 'Action not taken while creating the entity: the ingestion pipeline is busy. Wait for the current document processing to finish, then retry.',
      }),
    );
  });

  it('surfaces a 422 validation error with a payload-shape hint', async () => {
    stubCreateEntityResponse(422, {
      detail: [{ loc: ['body', 'name'], msg: 'empty' }],
    });
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(
      screen.getByTestId('kg-add-entity-name'),
      'New Entity',
    );
    await userEvent.click(screen.getByTestId('kg-add-entity-submit'));

    const err = await screen.findByTestId('kg-add-entity-error');
    expect(err.textContent).toMatch(/invalid/i);
    expect(screen.getByTestId('kg-add-entity-form')).toBeInTheDocument();
  });

  it('surfaces a 503 backend failure without leaking driver detail', async () => {
    stubCreateEntityResponse(503, {
      detail: 'Memgraph backend rejected the write',
    });
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(
      screen.getByTestId('kg-add-entity-name'),
      'New Entity',
    );
    await userEvent.click(screen.getByTestId('kg-add-entity-submit'));

    const err = await screen.findByTestId('kg-add-entity-error');
    expect(err.textContent).toMatch(/memgraph backend/i);
    expect(err.textContent).toMatch(/retry/i);
    expect(screen.getByTestId('kg-add-entity-form')).toBeInTheDocument();
  });

  it('treats a 500 projection failure as a half-success: close form + done toast', async () => {
    stubCreateEntityResponse(500, {
      detail:
        "Graph entity 'New Entity' was created in workspace 'cib' but the projection failed.",
    });
    const onToast = vi.fn();
    renderWithClient(
      <GraphTab {...defaultProps()} onToast={onToast} />,
    );
    await userEvent.click(screen.getByTestId('kg-add-entity-btn'));
    await userEvent.type(
      screen.getByTestId('kg-add-entity-name'),
      'New Entity',
    );
    await userEvent.click(screen.getByTestId('kg-add-entity-submit'));

    // The form closes — the entity exists server-side, leaving the
    // form open would push the operator into a retry-then-409 loop.
    await waitFor(() =>
      expect(screen.queryByTestId('kg-add-entity-form')).toBeNull(),
    );
    // A 'done' toast surfaces the half-success honestly.
    expect(onToast).toHaveBeenCalledTimes(1);
    const toast = onToast.mock.calls[0][0];
    expect(toast.kind).toBe('done');
    expect(toast.title).toMatch(/created/i);
    expect(toast.sub).toMatch(/created server-side/i);
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
      document.querySelectorAll('button.kg-rel-row'),
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
    const fetchMock = vi.fn(async (...args: [RequestInfo | URL, RequestInit?]) => {
      const [url, init] = args;
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
    });

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
    await userEvent.type(screen.getByTestId('kg-add-rel-target'), 'RHEL');
    await userEvent.click(screen.getByTestId('kg-add-rel-target-option-e_rhel'));
    await userEvent.type(screen.getByTestId('kg-add-rel-label'), 'RUNS_ON');
    expect(screen.getByTestId('kg-add-rel-duplicate')).toBeInTheDocument();
    expect(screen.getByTestId('kg-add-rel-submit')).toBeDisabled();
  });

  it('filters target entities and supports keyboard selection', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    const targetSearch = screen.getByTestId('kg-add-rel-target');

    await userEvent.type(targetSearch, 'aub');

    expect(
      screen.getByTestId('kg-add-rel-target-option-e_aubervil'),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId('kg-add-rel-target-option-e_rhel'),
    ).toBeNull();

    await userEvent.keyboard('{Enter}');
    await userEvent.type(screen.getByTestId('kg-add-rel-label'), 'depends on');
    expect(screen.getByTestId('kg-add-rel-submit')).not.toBeDisabled();
  });

  it('Cancel closes the form', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    await userEvent.click(screen.getByRole('button', { name: /^Cancel$/ }));
    expect(screen.queryByTestId('kg-add-rel-form')).toBeNull();
  });

  it('Escape closes the Add relation form', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    expect(screen.getByTestId('kg-add-rel-form')).toBeInTheDocument();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByTestId('kg-add-rel-form')).toBeNull();
  });

  it('switching entity disarms a pending Add relation form', async () => {
    renderWithClient(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    expect(screen.getByTestId('kg-add-rel-form')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    expect(screen.queryByTestId('kg-add-rel-form')).toBeNull();
  });

  it('adds a relation and refetches the relation list', async () => {
    const newRelation = {
      id: 'r_custom',
      source: 'e_oracle',
      target: 'e_aubervil',
      label: 'DEPENDS_ON',
      strength: 0.77,
    };
    let created = false;
    const fetchMock = vi.fn(async (...args: [RequestInfo | URL, RequestInit?]) => {
      const [url, init] = args;
      const href = String(url);
      if (href.includes('/graph/relations') && init?.method === 'POST') {
        created = true;
        return new Response(JSON.stringify(newRelation), {
          status: 201,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (href.includes('/graph/relations')) {
        const relations = created
          ? [...GRAPH_RELATION_FIXTURES, newRelation]
          : GRAPH_RELATION_FIXTURES;
        return new Response(JSON.stringify(relations), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    renderGraphWithLiveRelations(fetchMock);

    await waitFor(() =>
      expect(
        screen.getByText(
          new RegExp(`${GRAPH_RELATION_FIXTURES.length} relations`),
        ),
      ).toBeInTheDocument(),
    );

    await userEvent.click(screen.getByTestId('kg-add-rel-btn'));
    expect(screen.getByTestId('kg-add-rel-form')).toBeInTheDocument();
    const addRelationForm = screen.getByTestId('kg-add-rel-form') as HTMLFormElement;
    await userEvent.type(screen.getByTestId('kg-add-rel-target'), 'aub');
    await userEvent.click(screen.getByTestId('kg-add-rel-target-option-e_aubervil'));
    await userEvent.type(screen.getByTestId('kg-add-rel-label'), 'depends on');
    await waitFor(
      () =>
        expect(
          screen.getByTestId('kg-add-rel-submit'),
        ).not.toBeDisabled(),
      { timeout: 2000 },
    );
    const submitBtn = screen.getByTestId('kg-add-rel-submit') as HTMLButtonElement;
    expect(screen.getByTestId('kg-add-rel-submit').getAttribute('type')).toBe('submit');
    expect(submitBtn.disabled).toBe(false);
    expect((screen.getByTestId('kg-add-rel-label') as HTMLInputElement).value).toBe('depends on');

    fireEvent.submit(addRelationForm);

    await waitFor(() =>
      expect(fetchMock.mock.calls.find(
        ([url, init]) =>
          String(url).includes('/graph/relations') &&
          (init as RequestInit | undefined)?.method === 'POST',
      )).toBeDefined(),
      { timeout: 5000 },
    );

    await waitFor(
      () =>
        expect(
          screen.getByText(
            new RegExp(`${GRAPH_RELATION_FIXTURES.length + 1} relations`),
          ),
        ).toBeInTheDocument(),
      { timeout: 5000 },
    );
    await waitFor(() =>
      expect(screen.getByTestId('kg-rel-row-r_custom')).toBeInTheDocument(),
      { timeout: 5000 },
    );

    const postCalls = fetchMock.mock.calls.filter(
      ([url, init]) =>
        String(url).includes('/graph/relations') &&
        (init as RequestInit | undefined)?.method === 'POST',
    );
    expect(postCalls).toHaveLength(1);
    const [url, init] = postCalls[0];
    expect(String(url)).toContain('/graph/relations');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.source).toBe('e_oracle');
    expect(body.target).toBe('e_aubervil');
    expect(body.label).toBe('DEPENDS_ON');
  });

  it('fires DELETE on confirm and updates list after removing a relation', async () => {
    const deletedId = 'r_01';
    const remainingRelations = GRAPH_RELATION_FIXTURES.filter(
      (r) => r.id !== deletedId,
    );
    let relationDeleted = false;
    const fetchMock = vi.fn(async (...args: [RequestInfo | URL, RequestInit?]) => {
      const [url, init] = args;
      const href = String(url);
      if (href.includes(`/graph/relations/${deletedId}`) && init?.method === 'DELETE') {
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
    });

    renderGraphWithLiveRelations(fetchMock);

    await waitFor(() => {
      expect(
        screen.getByText(new RegExp(`${GRAPH_RELATION_FIXTURES.length} relations`)),
      ).toBeInTheDocument();
    });

    await userEvent.click(screen.getByTestId(`kg-rel-row-${deletedId}`));
    const btn = screen.getByTestId('kg-rel-delete');
    await userEvent.click(btn);
    await userEvent.click(btn);

    await waitFor(() => {
      const deleteCalls = fetchMock.mock.calls.filter(
        ([url, init]) =>
          String(url).includes(`/graph/relations/${deletedId}`) &&
          (init as RequestInit | undefined)?.method === 'DELETE',
      );
      expect(deleteCalls).toHaveLength(1);
    });

    await waitFor(() =>
      expect(screen.queryByTestId(`kg-rel-row-${deletedId}`)).toBeNull(),
    );
    await waitFor(() =>
      expect(
        screen.getByText(new RegExp(`${remainingRelations.length} relations`)),
      ).toBeInTheDocument(),
    );
  });
});
