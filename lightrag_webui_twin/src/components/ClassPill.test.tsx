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
  it('renders nothing when no classification and no visibility exist', () => {
    const { container } = render(<ClassPill cls={undefined} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when structured payload has no class id and no visibility', () => {
    const { container } = render(<ClassPill cls={mkCls({ class_id: null })} />);
    expect(container.firstChild).toBeNull();
  });
});

// Product decision 2026-08-04 (QA DOC-V4-001): the two-level public/interne
// model is badged too — legacy strings and the visibility fallback render.
describe('ClassPill — two-level model (DOC-V4-001)', () => {
  it('renders a legacy string classification as its ladder pill', () => {
    render(<ClassPill cls="internal" docId="d-legacy" />);
    const pill = screen.getByTestId('class-pill-d-legacy');
    expect(pill.textContent).toBe('Internal');
    expect(pill.className).toContain('class-internal');
  });

  it('renders a Public pill from the legacy string', () => {
    render(<ClassPill cls="public" docId="d-public" />);
    const pill = screen.getByTestId('class-pill-d-public');
    expect(pill.textContent).toBe('Public');
    expect(pill.className).toContain('class-public');
  });

  it('keeps a non-ladder legacy string verbatim with the unknown tone', () => {
    render(<ClassPill cls="restricted" docId="d-raw" />);
    const pill = screen.getByTestId('class-pill-d-raw');
    expect(pill.textContent).toBe('restricted');
    expect(pill.className).toContain('class-unknown');
  });

  it('falls back to the document visibility when nothing was extracted', () => {
    render(<ClassPill cls={undefined} visibility="internal" docId="d-vis" />);
    const pill = screen.getByTestId('class-pill-d-vis');
    expect(pill.textContent).toBe('Internal');
    expect(pill.className).toContain('class-internal');
  });

  it('falls back to visibility when the structured payload has no class id', () => {
    render(
      <ClassPill
        cls={mkCls({ class_id: null })}
        visibility="public"
        docId="d-vis2"
      />,
    );
    const pill = screen.getByTestId('class-pill-d-vis2');
    expect(pill.textContent).toBe('Public');
  });

  it('a structured MIP label always wins over visibility', () => {
    render(
      <ClassPill cls={mkCls({ class_id: 'C3' })} visibility="public" docId="d-mip" />,
    );
    const pill = screen.getByTestId('class-pill-d-mip');
    expect(pill.textContent).toBe('Confidential');
    expect(pill.className).toContain('class-confidential');
  });
});

describe('ClassPill — structured classification', () => {
  it('renders legacy C2 as an Internal shield', () => {
    render(<ClassPill cls={mkCls({ class_id: 'C2' })} docId="d1" />);
    const pill = screen.getByTestId('class-pill-d1');
    expect(pill).toBeInTheDocument();
    expect(pill.textContent).toBe('Internal');
    expect(pill.className).toContain('class-internal');
    expect(pill.getAttribute('data-class-id')).toBe('C2');
    expect(pill.getAttribute('data-class-tone')).toBe('internal');
    expect(pill.querySelector('[data-icon="shield"]')).toBeInTheDocument();
  });

  it('renders Confidential class with the class-confidential variant', () => {
    render(
      <ClassPill
        cls={mkCls({ class_id: 'Confidential', class_name: 'Confidential' })}
        docId="d2"
      />,
    );
    const pill = screen.getByTestId('class-pill-d2');
    expect(pill.textContent).toBe('Confidential');
    expect(pill.className).toContain('class-confidential');
  });

  it('renders Private class with the class-private variant', () => {
    render(<ClassPill cls={mkCls({ class_id: 'Private' })} docId="d-private" />);
    const pill = screen.getByTestId('class-pill-d-private');
    expect(pill.textContent).toBe('Private');
    expect(pill.className).toContain('class-private');
  });

  it('renders UNKNOWN class with the class-unknown variant', () => {
    render(<ClassPill cls={mkCls({ class_id: 'UNKNOWN' })} docId="d3" />);
    const pill = screen.getByTestId('class-pill-d3');
    expect(pill.textContent).toBe('Unknown');
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
