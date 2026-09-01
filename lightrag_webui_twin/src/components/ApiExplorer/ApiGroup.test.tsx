import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { OPENAPI_GROUPS } from '../../fixtures';
import { ApiGroup } from './ApiGroup';

describe('ApiGroup', () => {
  it('owns disclosure of its endpoint rows', async () => {
    const group = OPENAPI_GROUPS[0];
    render(
      <ApiGroup
        group={group}
        secured
        token=""
        baseUrl="http://localhost"
      />,
    );

    const disclosure = screen
      .getByTestId(`api-group-${group.id}`)
      .querySelector<HTMLButtonElement>('.swagger-group-head');
    expect(disclosure).not.toBeNull();
    if (!disclosure) return;
    expect(disclosure).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getAllByTestId(/^endpoint-/)).toHaveLength(group.endpoints.length);

    await userEvent.click(disclosure);
    expect(disclosure).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId(/^endpoint-/)).toBeNull();
  });
});
