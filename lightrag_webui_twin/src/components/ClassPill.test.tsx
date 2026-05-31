/**
 * Unit tests for ClassPill (Couche 2 of feat/webui-port-from-prototype).
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ClassPill } from './ClassPill';
import type { ClassificationResult } from '../types/classification';

function mkCls(overrides: Partial<ClassificationResult> = {}): ClassificationResult {
  return {
    class_id: 'C2',
    class_name: 'C2 Confidentiel',
    label_guid: '22222222-2222-2222-2222-222222222222',
    raw_name: 'C2 Confidentiel',
    set_date: '2026-03-12T14:22:01Z',
    method: 'Standard',
    source_format: 'ooxml',
    reason: null,
    meta: {},
    ...overrides,
  };
}

describe('ClassPill — silence', () => {
  it('renders nothing when classification is undefined', () => {
    const { container } = render(<ClassPill cls={undefined} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when classification is a legacy string (no MIP extraction)', () => {
    const { container } = render(<ClassPill cls="internal" />);
    expect(container.firstChild).toBeNull();
  });
});

describe('ClassPill — structured classification', () => {
  it('renders C2 class with the class-c2 variant', () => {
    render(<ClassPill cls={mkCls({ class_id: 'C2' })} docId="d1" />);
    const pill = screen.getByTestId('class-pill-d1');
    expect(pill).toBeInTheDocument();
    expect(pill.textContent).toBe('C2');
    expect(pill.className).toContain('class-c2');
    expect(pill.getAttribute('data-class-id')).toBe('C2');
  });

  it('renders C3 class with the class-c3 variant', () => {
    render(<ClassPill cls={mkCls({ class_id: 'C3', class_name: 'C3 Strict' })} docId="d2" />);
    const pill = screen.getByTestId('class-pill-d2');
    expect(pill.textContent).toBe('C3');
    expect(pill.className).toContain('class-c3');
  });

  it('renders UNKNOWN class with the class-unknown variant (striped pattern)', () => {
    render(<ClassPill cls={mkCls({ class_id: 'UNKNOWN' })} docId="d3" />);
    const pill = screen.getByTestId('class-pill-d3');
    expect(pill.textContent).toBe('UNKNOWN');
    expect(pill.className).toContain('class-unknown');
  });

  it('uses class_name + set_date for the title (tooltip)', () => {
    render(
      <ClassPill
        cls={mkCls({ class_name: 'C2 Confidentiel', set_date: '2026-03-12T14:22:01Z' })}
        docId="d4"
      />,
    );
    const pill = screen.getByTestId('class-pill-d4');
    expect(pill.getAttribute('title')).toContain('C2 Confidentiel');
    expect(pill.getAttribute('title')).toContain('2026-03-12');
  });

  it('falls back to "class-pill" testid when no docId is provided', () => {
    render(<ClassPill cls={mkCls()} />);
    expect(screen.getByTestId('class-pill')).toBeInTheDocument();
  });
});
