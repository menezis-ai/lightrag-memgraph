/**
 * Unit tests for TagChip.
 *
 * Behaviors under test:
 *   - renders the tag label
 *   - applies semantic class when provided directly
 *   - applies semantic class from semanticsMap lookup
 *   - no semantic when neither given
 *   - removable=false omits the X button
 *   - removable=true renders X, click calls onRemove and stops propagation
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TagChip } from './TagChip';

describe('TagChip', () => {
  it('renders the tag label', () => {
    render(<TagChip tag="oracle" />);
    expect(screen.getByText('oracle')).toBeInTheDocument();
  });

  it('applies semantic class when passed directly', () => {
    const { container } = render(
      <TagChip tag="incident" semantics="critical" />,
    );
    const span = container.querySelector('span');
    expect(span?.className).toBe('tag-chip critical');
  });

  it('looks up semantic from semanticsMap if not given directly', () => {
    const { container } = render(
      <TagChip
        tag="deprecated"
        semanticsMap={{ deprecated: 'warning' }}
      />,
    );
    const span = container.querySelector('span');
    expect(span?.className).toBe('tag-chip warning');
  });

  it('uses plain tag-chip class when no semantic resolves', () => {
    const { container } = render(<TagChip tag="rman" />);
    expect(container.querySelector('span')?.className).toBe('tag-chip');
  });

  it('does not render the X button when removable=false', () => {
    render(<TagChip tag="oracle" />);
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('renders the X button when removable=true and calls onRemove on click', async () => {
    const onRemove = vi.fn();
    render(<TagChip tag="oracle" removable onRemove={onRemove} />);
    const btn = screen.getByRole('button', { name: 'Remove oracle' });
    await userEvent.click(btn);
    expect(onRemove).toHaveBeenCalledWith('oracle');
  });

  it('stops event propagation on remove click', async () => {
    const onRemove = vi.fn();
    const onParentClick = vi.fn();
    render(
      <div onClick={onParentClick}>
        <TagChip tag="oracle" removable onRemove={onRemove} />
      </div>,
    );
    const btn = screen.getByRole('button', { name: 'Remove oracle' });
    await userEvent.click(btn);
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onParentClick).not.toHaveBeenCalled();
  });
});
