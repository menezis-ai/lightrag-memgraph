/**
 * Unit tests for ``TagAttrEditor`` (TR-KG-03 / QA report
 * 2026-06-12). The component is exported from ``GraphTab.tsx`` for
 * exactly this purpose: it owns the node-tag binding to the active
 * tag catalog client-side. The backend mirrors the rule in
 * ``server/webui_router._validate_graph_entity_tags``.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { TagAttrEditor } from './GraphTab';

const CATALOG = ['rman', 'oracle', 'rhel9', 'production', 'memgraph'];

describe('TagAttrEditor — thesaurus binding (TR-KG-03)', () => {
  it('suggests catalog matches on the typed prefix', async () => {
    render(
      <TagAttrEditor tags={[]} tagCatalog={CATALOG} onChange={vi.fn()} />,
    );

    await userEvent.type(screen.getByLabelText('Add node tag'), 'rm');

    // ``rman`` matches the ``rm`` prefix and is the only catalog
    // tag that does. ``rhel9`` does not match (starts with ``rh``).
    expect(screen.getByTestId('kg-tag-sugg-rman')).toBeInTheDocument();
    expect(screen.queryByTestId('kg-tag-sugg-rhel9')).toBeNull();
  });

  it('clicking a suggestion adds it to the tags list and clears the input', async () => {
    const onChange = vi.fn();
    render(
      <TagAttrEditor tags={[]} tagCatalog={CATALOG} onChange={onChange} />,
    );

    const input = screen.getByLabelText('Add node tag');
    await userEvent.type(input, 'rm');
    // ``onMouseDown`` is used in the suggestion row — testing-library's
    // userEvent.click triggers mousedown + mouseup + click, which is
    // enough to fire our handler.
    await userEvent.click(screen.getByTestId('kg-tag-sugg-rman'));

    expect(onChange).toHaveBeenCalledWith(['rman']);
    expect((input as HTMLInputElement).value).toBe('');
  });

  it('shows an alert without an Add button when the typed tag is unknown', async () => {
    const onChange = vi.fn();
    render(
      <TagAttrEditor tags={[]} tagCatalog={CATALOG} onChange={onChange} />,
    );

    await userEvent.type(
      screen.getByLabelText('Add node tag'),
      'random-bullshit-bingo',
    );

    expect(screen.queryByRole('button', { name: /^Add$/ })).toBeNull();
    const alert = screen.getByTestId('kg-tag-not-in-catalog');
    expect(alert.textContent).toContain('random-bullshit-bingo');
    expect(alert.textContent).toMatch(/not in the tag catalog/i);
    // No suggestions row when nothing matches and the value is unknown.
    expect(screen.queryByTestId('kg-tag-suggestions')).toBeNull();
  });

  it('rejects Enter on an unknown tag and does not call onChange', async () => {
    const onChange = vi.fn();
    render(
      <TagAttrEditor tags={[]} tagCatalog={CATALOG} onChange={onChange} />,
    );

    await userEvent.type(
      screen.getByLabelText('Add node tag'),
      'random-bullshit-bingo{Enter}',
    );

    expect(onChange).not.toHaveBeenCalled();
  });

  it('accepts Enter on a tag that is an exact catalog match', async () => {
    const onChange = vi.fn();
    render(
      <TagAttrEditor
        tags={[]}
        tagCatalog={CATALOG}
        onChange={onChange}
      />,
    );

    await userEvent.type(
      screen.getByLabelText('Add node tag'),
      'oracle{Enter}',
    );

    expect(onChange).toHaveBeenCalledWith(['oracle']);
  });

  it('does not propose tags that are already on the entity', async () => {
    render(
      <TagAttrEditor
        tags={['rman']}
        tagCatalog={CATALOG}
        onChange={vi.fn()}
      />,
    );

    await userEvent.type(screen.getByLabelText('Add node tag'), 'rm');

    // ``rman`` already attached → suggestion list filters it out.
    expect(screen.queryByTestId('kg-tag-sugg-rman')).toBeNull();
  });

  it('legacy fallback: empty tagCatalog accepts any value (binding disabled)', async () => {
    // Some isolated fixtures / legacy snapshots render the editor
    // without a tagCatalog. Don't break those — when the catalog is
    // empty, treat the binding as off and accept whatever the
    // operator types (the old free-text behaviour).
    const onChange = vi.fn();
    render(<TagAttrEditor tags={[]} tagCatalog={[]} onChange={onChange} />);

    await userEvent.type(
      screen.getByLabelText('Add node tag'),
      'freeform-tag{Enter}',
    );

    expect(onChange).toHaveBeenCalledWith(['freeform-tag']);
    expect(screen.queryByTestId('kg-tag-not-in-catalog')).toBeNull();
  });

  it('removes a tag when its remove button is clicked', async () => {
    const onChange = vi.fn();
    render(
      <TagAttrEditor
        tags={['rman', 'oracle']}
        tagCatalog={CATALOG}
        onChange={onChange}
      />,
    );

    await userEvent.click(screen.getByLabelText('Remove rman'));
    expect(onChange).toHaveBeenCalledWith(['oracle']);
  });
});
