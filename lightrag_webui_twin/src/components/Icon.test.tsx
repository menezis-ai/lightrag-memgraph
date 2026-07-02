/**
 * Unit tests for Icon + SourceIcon.
 *
 * Behaviors under test:
 *   - renders an SVG with the resolved Tabler path keyed by `name`
 *   - default size / stroke / color applied when no prop given
 *   - props override the defaults
 *   - SourceIcon maps known types to the right icon, falls back to file-text
 */

import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import { Icon, SourceIcon } from './Icon';

describe('Icon', () => {
  it('renders an SVG with the data-icon attribute equal to name', () => {
    render(<Icon name="settings" />);
    const svg = document.querySelector('svg[data-icon="settings"]');
    expect(svg).not.toBeNull();
  });

  it('applies default size, stroke width, color', () => {
    render(<Icon name="search" />);
    const svg = document.querySelector('svg[data-icon="search"]') as SVGElement;
    expect(svg.getAttribute('width')).toBe('16');
    expect(svg.getAttribute('height')).toBe('16');
    expect(svg.getAttribute('stroke-width')).toBe('1.5');
    expect(svg.getAttribute('stroke')).toBe('currentColor');
  });

  it('honors size, color, strokeWidth, className props', () => {
    render(
      <Icon
        name="bell"
        size={24}
        color="red"
        strokeWidth={2}
        className="topbar-bell"
      />,
    );
    const svg = document.querySelector('svg[data-icon="bell"]') as SVGElement;
    expect(svg.getAttribute('width')).toBe('24');
    expect(svg.getAttribute('stroke')).toBe('red');
    expect(svg.getAttribute('stroke-width')).toBe('2');
    expect(svg.classList.contains('topbar-bell')).toBe(true);
  });
});
describe('SourceIcon', () => {
  it.each([
    ['file', 'file-text'],
    ['confluence', 'brand-confluence'],
    ['sharepoint', 'cloud'],
    ['url', 'link'],
  ] as const)('maps source type "%s" to icon "%s"', (type, iconName) => {
    render(<SourceIcon type={type} />);
    expect(
      document.querySelector(`svg[data-icon="${iconName}"]`),
    ).not.toBeNull();
  });

  it('falls back to file-text when type is unknown', () => {
    render(<SourceIcon type="unknown" />);
    expect(
      document.querySelector('svg[data-icon="file-text"]'),
    ).not.toBeNull();
  });

  it('overrides default size', () => {
    render(<SourceIcon type="file" size={20} />);
    const svg = document.querySelector(
      'svg[data-icon="file-text"]',
    ) as SVGElement;
    expect(svg.getAttribute('width')).toBe('20');
  });
});
