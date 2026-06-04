/**
 * Settings → Space section — regression for mock-kill F1.
 *
 * The previous fixture-driven cards (Visibility, Region, Retention TTL
 * table) were removed because they displayed invented values
 * (`eu-west-3 · dc-paris`, `twin-default-space-retention-v1`, hardcoded
 * TTLs) that risked being read as compliance commitments.
 *
 * This test pins:
 *   - Identity reflects the props (not a hardcoded fixture).
 *   - The removed cards stay removed.
 */

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { WorkspaceSection } from './WorkspaceSection';

describe('WorkspaceSection', () => {
  it('shows the active space id and display name from props', () => {
    render(
      <WorkspaceSection activeSpaceId="cib-prod" displayName="CIB Production" />,
    );
    expect(screen.getByTestId('settings-active-ws')).toHaveTextContent(
      'cib-prod',
    );
    expect(
      screen.getByTestId('settings-space-display-name'),
    ).toHaveTextContent('CIB Production');
  });

  it('shows an "(unset)" placeholder when displayName is empty', () => {
    render(
      <WorkspaceSection activeSpaceId="sandbox" displayName="" />,
    );
    expect(
      screen.getByTestId('settings-space-display-name'),
    ).toHaveTextContent('(unset)');
  });

  it('does not render the removed Visibility / Region / Retention cards', () => {
    render(
      <WorkspaceSection activeSpaceId="default" displayName="Default" />,
    );
    // None of the fixture-only labels should appear.
    expect(screen.queryByText(/visibility/i)).toBeNull();
    expect(screen.queryByText(/region/i)).toBeNull();
    expect(screen.queryByText(/retention policy/i)).toBeNull();
    expect(screen.queryByText(/eu-west-3/i)).toBeNull();
    expect(screen.queryByText(/twin-default-space-retention/i)).toBeNull();
  });
});
