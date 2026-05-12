/**
 * Unit tests for GraphTab.
 *
 * Covers: header counts, entity type toggle, search filter, node selection
 * updates detail panel, neighbor relation list, zoom in/out, Reset view,
 * Navigate to documents CTA.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GraphTab } from './GraphTab';
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

beforeEach(() => {
  window.history.replaceState(null, '', '/');
});
afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('GraphTab — rendering', () => {
  it('renders header with entity + relation counts', () => {
    render(<GraphTab {...defaultProps()} />);
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
    render(<GraphTab {...defaultProps()} />);
    ['PRODUCT', 'TECHNOLOGY', 'CONCEPT', 'ORG', 'PERSON', 'LOCATION'].forEach((t) => {
      expect(screen.getByTestId(`kg-type-${t}`)).toBeInTheDocument();
    });
  });

  it('renders the default selected entity (first fixture = e_oracle) in detail panel', () => {
    render(<GraphTab {...defaultProps()} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/Oracle Database/);
    expect(detail.textContent).toMatch(/Product/);
  });
});

describe('GraphTab — filters', () => {
  it('toggling PRODUCT off hides PRODUCT nodes', async () => {
    render(<GraphTab {...defaultProps()} />);
    expect(screen.getByTestId('kg-node-e_oracle')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('kg-type-PRODUCT'));
    expect(screen.queryByTestId('kg-node-e_oracle')).toBeNull();
  });

  it('search filter narrows visible nodes by name', async () => {
    render(<GraphTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search entities'), 'swift');
    // SWIFT entity matches name
    expect(screen.getByTestId('kg-node-e_swift')).toBeInTheDocument();
    // Oracle should be filtered out
    expect(screen.queryByTestId('kg-node-e_oracle')).toBeNull();
  });

  it('no-match search shows empty state with Clear filter CTA', async () => {
    render(<GraphTab {...defaultProps()} />);
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
    render(<GraphTab {...defaultProps()} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    await userEvent.click(screen.getByTestId('kg-node-e_memgraph'));
    expect(detail.textContent).toMatch(/Memgraph/);
    expect(detail.textContent).toMatch(/Graph DB backing LightRAG/);
  });

  it('detail panel lists outgoing + incoming relations for the selected entity', () => {
    render(<GraphTab {...defaultProps()} />);
    // e_oracle has outgoing: e_rhel, e_pga, e_vmware. Incoming: rman, archlog, marc, iso20022.
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    expect(detail.textContent).toMatch(/Outgoing \(3\)/);
    expect(detail.textContent).toMatch(/Incoming \(4\)/);
  });

  it('clicking a neighbor in the detail panel selects it as current entity', async () => {
    render(<GraphTab {...defaultProps()} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    // Outgoing RUNS_ON → RHEL 9 button
    const rhelBtn = Array.from(detail.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('RHEL 9'),
    );
    expect(rhelBtn).toBeDefined();
    await userEvent.click(rhelBtn!);
    expect(detail.textContent).toMatch(/RHEL 9/);
    expect(detail.textContent).toMatch(/Red Hat Enterprise Linux/);
  });

  it('"View N sources" CTA navigates to documents with q=entity name', async () => {
    const p = defaultProps();
    render(<GraphTab {...p} />);
    const detail = document.querySelector('.kg-detail') as HTMLElement;
    const cta = Array.from(detail.querySelectorAll('button')).find((b) =>
      b.textContent?.match(/View \d+ sources mentioning/),
    );
    await userEvent.click(cta!);
    expect(p.onNavigate).toHaveBeenCalledWith('documents', { q: 'Oracle Database' });
  });
});

describe('GraphTab — zoom + reset', () => {
  it('zoom in button increases zoom percentage', async () => {
    render(<GraphTab {...defaultProps()} />);
    const value0 = screen.getByTestId('kg-zoom-value').textContent;
    expect(value0).toBe('100%');
    await userEvent.click(screen.getByLabelText('Zoom in'));
    const value1 = screen.getByTestId('kg-zoom-value').textContent;
    expect(value1).toBe('118%');
  });

  it('zoom out button decreases zoom percentage', async () => {
    render(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByLabelText('Zoom out'));
    expect(screen.getByTestId('kg-zoom-value').textContent).toBe('85%');
  });

  it('Reset view restores zoom to 100%', async () => {
    render(<GraphTab {...defaultProps()} />);
    await userEvent.click(screen.getByLabelText('Zoom in'));
    await userEvent.click(screen.getByLabelText('Zoom in'));
    await userEvent.click(screen.getByRole('button', { name: /Reset view/ }));
    expect(screen.getByTestId('kg-zoom-value').textContent).toBe('100%');
  });
});
