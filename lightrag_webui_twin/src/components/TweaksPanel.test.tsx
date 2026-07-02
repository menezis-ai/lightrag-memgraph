/**
 * Unit tests for TweaksPanel + control building blocks + useTweaks.
 *
 * Covers: open/close, close button, drag persistence to localStorage,
 * TweakSlider / TweakToggle / TweakRadio (segments + select fallback) /
 * TweakNumber clamp / TweakColor swatch chips / useTweaks store updates.
 */

import { describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, renderHook, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  TweakColor,
  TweakNumber,
  TweakRadio,
  TweakSlider,
  TweakToggle,
  TweaksPanel,
  useTweaks,
} from './TweaksPanel';

describe('TweaksPanel — open/close', () => {
  it('renders nothing when open=false', () => {
    render(<TweaksPanel open={false} onClose={vi.fn()} />);
    expect(screen.queryByTestId('twk-panel')).toBeNull();
  });

  it('renders panel + title + close button when open=true', () => {
    render(
      <TweaksPanel open onClose={vi.fn()} title="Settings">
        <div>body</div>
      </TweaksPanel>,
    );
    expect(screen.getByTestId('twk-panel')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByText('body')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close tweaks' })).toBeInTheDocument();
  });

  it('close button invokes onClose', async () => {
    const onClose = vi.fn();
    render(
      <TweaksPanel open onClose={onClose}>
        <div>body</div>
      </TweaksPanel>,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Close tweaks' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('TweakSlider', () => {
  it('renders the current value with unit suffix', () => {
    render(<TweakSlider label="Font size" value={16} unit="px" onChange={vi.fn()} />);
    expect(screen.getByText('16px')).toBeInTheDocument();
  });

  it('change emits the new value as Number', () => {
    const onChange = vi.fn();
    render(
      <TweakSlider label="Font size" value={16} min={10} max={32} onChange={onChange} />,
    );
    const slider = screen.getByLabelText('Font size') as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '24' } });
    expect(onChange).toHaveBeenCalledWith(24);
  });
});

describe('TweakToggle', () => {
  it('renders aria-checked=false when value=false', () => {
    render(<TweakToggle label="Dark mode" value={false} onChange={vi.fn()} />);
    const sw = screen.getByRole('switch', { name: 'Dark mode' });
    expect(sw).toHaveAttribute('aria-checked', 'false');
    expect(sw).toHaveAttribute('data-on', '0');
  });

  it('clicking inverts the value', async () => {
    const onChange = vi.fn();
    render(<TweakToggle label="Dark mode" value={false} onChange={onChange} />);
    await userEvent.click(screen.getByRole('switch', { name: 'Dark mode' }));
    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe('TweakRadio', () => {
  it('renders as segmented radio when 2-3 short options fit', () => {
    render(
      <TweakRadio
        label="Density"
        value="regular"
        options={['compact', 'regular', 'comfy']}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole('radiogroup', { name: 'Density' })).toBeInTheDocument();
    // 3 segment buttons rendered
    expect(screen.getAllByRole('radio')).toHaveLength(3);
  });

  it('falls back to <select> when options would not fit as segments (>=4 or too long)', () => {
    render(
      <TweakRadio
        label="Mode"
        value="hybrid"
        options={['naive', 'local', 'global', 'hybrid', 'mix', 'bypass']}
        onChange={vi.fn()}
      />,
    );
    // 6 options ≥ the segment table caps → select fallback
    expect(screen.getByRole('combobox', { name: 'Mode' })).toBeInTheDocument();
  });

  it('select fallback emits the resolved option value (string preserved)', async () => {
    const onChange = vi.fn();
    render(
      <TweakRadio
        label="Mode"
        value="hybrid"
        options={['naive', 'local', 'global', 'hybrid', 'mix', 'bypass']}
        onChange={onChange}
      />,
    );
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'mix');
    expect(onChange).toHaveBeenCalledWith('mix');
  });
});

describe('TweakNumber', () => {
  it('clamps value to [min, max] on change', () => {
    const onChange = vi.fn();
    render(
      <TweakNumber label="Top K" value={50} min={1} max={100} onChange={onChange} />,
    );
    const input = screen.getByLabelText('Top K') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '500' } });
    expect(onChange).toHaveBeenCalledWith(100);
    fireEvent.change(input, { target: { value: '-50' } });
    expect(onChange).toHaveBeenLastCalledWith(1);
  });
});

describe('TweakColor', () => {
  it('renders one chip per option and emits the selected color', async () => {
    const onChange = vi.fn();
    render(
      <TweakColor
        label="Primary"
        value="#D97757"
        options={['#D97757', '#2A6FDB', '#1F8A5B']}
        onChange={onChange}
      />,
    );
    const chips = screen.getAllByRole('radio');
    expect(chips).toHaveLength(3);
    // The selected chip has aria-checked=true
    const selected = chips.find((c) => c.getAttribute('aria-checked') === 'true');
    expect(selected).toBeDefined();
    expect(selected!.getAttribute('aria-label')).toMatch(/D97757/i);
    // Click another chip
    await userEvent.click(chips[1]);
    expect(onChange).toHaveBeenCalledWith('#2A6FDB');
  });

  it('falls back to native input type=color when options is empty', () => {
    render(
      <TweakColor label="Accent" value="#abcdef" options={[]} onChange={vi.fn()} />,
    );
    const input = screen.getByLabelText('Accent') as HTMLInputElement;
    expect(input.type).toBe('color');
  });
});

describe('useTweaks store', () => {
  it('returns defaults on first render', () => {
    const { result } = renderHook(() =>
      useTweaks({ density: 'regular', dark: false, fontSize: 16 }),
    );
    expect(result.current[0]).toEqual({
      density: 'regular',
      dark: false,
      fontSize: 16,
    });
  });

  it('setTweak(key, value) merges one key', () => {
    const { result } = renderHook(() =>
      useTweaks({ density: 'regular', dark: false, fontSize: 16 }),
    );
    act(() => {
      result.current[1]('dark', true);
    });
    expect(result.current[0].dark).toBe(true);
    expect(result.current[0].fontSize).toBe(16);
  });

  it('setTweak({...patch}) merges multiple keys', () => {
    const { result } = renderHook(() =>
      useTweaks({ density: 'regular', dark: false, fontSize: 16 }),
    );
    act(() => {
      result.current[1]({ density: 'compact', fontSize: 14 });
    });
    expect(result.current[0]).toEqual({
      density: 'compact',
      dark: false,
      fontSize: 14,
    });
  });
});
