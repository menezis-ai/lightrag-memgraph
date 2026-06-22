/**
 * TweaksPanel — floating dev/operator tweaks shell + form-control helpers.
 *
 * Ported from Desktop/UI/tweaks-panel.jsx. The proto carried a host protocol
 * (postMessage to parent for __activate_edit_mode / __edit_mode_set_keys) used
 * only by the deck-stage embed. In the standalone Twin WebUI fork we drop
 * that protocol — the panel is opened/closed via the `open` prop and writes
 * its position to localStorage so users get persistence across reloads.
 *
 * The exported building blocks (TweakSection, TweakSlider, TweakToggle,
 * TweakRadio, TweakSelect, TweakText, TweakNumber, TweakColor, TweakButton)
 * keep the same signatures as the proto so author code is portable.
 *
 * Usage:
 *   const [t, setTweak] = useTweaks({ fontSize: 16, density: 'regular', dark: false });
 *   <TweaksPanel open={tweaksOpen} onClose={() => setTweaksOpen(false)}>
 *     <TweakSection label="Typography" />
 *     <TweakSlider label="Font size" value={t.fontSize} min={10} max={32} unit="px"
 *                  onChange={(v) => setTweak('fontSize', v)} />
 *     <TweakToggle label="Dark mode" value={t.dark} onChange={(v) => setTweak('dark', v)} />
 *   </TweaksPanel>
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';

const POSITION_KEY = 'twin-tweaks.position';
const PAD = 16;

interface Position {
  x: number;
  y: number;
}

function readPosition(): Position {
  try {
    const raw = localStorage.getItem(POSITION_KEY);
    if (raw) {
      const v = JSON.parse(raw) as Partial<Position>;
      if (typeof v.x === 'number' && typeof v.y === 'number') {
        return { x: v.x, y: v.y };
      }
    }
  } catch {
    /* ignore */
  }
  return { x: 16, y: 16 };
}

function writePosition(p: Position): void {
  try {
    localStorage.setItem(POSITION_KEY, JSON.stringify(p));
  } catch {
    /* ignore */
  }
}

// ── useTweaks ───────────────────────────────────────────────────────────────

/**
 * Generic store for tweak values. `setTweak` accepts either a single
 * (key, value) pair or a `{ key: value, ... }` patch object, matching the
 * proto's contract.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useTweaks<T extends Record<string, unknown>>(
  defaults: T,
): [T, (keyOrPatch: keyof T | Partial<T>, value?: T[keyof T]) => void] {
  const [values, setValues] = useState<T>(defaults);
  const setTweak = useCallback(
    (keyOrPatch: keyof T | Partial<T>, val?: T[keyof T]) => {
      const edits =
        typeof keyOrPatch === 'object' && keyOrPatch !== null
          ? (keyOrPatch as Partial<T>)
          : ({ [keyOrPatch as string]: val } as Partial<T>);
      setValues((prev) => ({ ...prev, ...edits }));
    },
    [],
  );
  return [values, setTweak];
}

// ── TweaksPanel ─────────────────────────────────────────────────────────────

export interface TweaksPanelProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children?: ReactNode;
}

export function TweaksPanel({ open, onClose, title = 'Tweaks', children }: Readonly<TweaksPanelProps>) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState<Position>(() => readPosition());
  const offsetRef = useRef<Position>(offset);

  const clampToViewport = useCallback(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const w = panel.offsetWidth;
    const h = panel.offsetHeight;
    const maxRight = Math.max(PAD, globalThis.innerWidth - w - PAD);
    const maxBottom = Math.max(PAD, globalThis.innerHeight - h - PAD);
    const next = {
      x: Math.min(maxRight, Math.max(PAD, offsetRef.current.x)),
      y: Math.min(maxBottom, Math.max(PAD, offsetRef.current.y)),
    };
    offsetRef.current = next;
    setOffset(next);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    clampToViewport();
    if (typeof ResizeObserver === 'undefined') {
      globalThis.addEventListener('resize', clampToViewport);
      return () => globalThis.removeEventListener('resize', clampToViewport);
    }
    const ro = new ResizeObserver(clampToViewport);
    ro.observe(document.documentElement);
    return () => ro.disconnect();
  }, [open, clampToViewport]);

  const onDragStart = (e: React.MouseEvent<HTMLDivElement>) => {
    const panel = panelRef.current;
    if (!panel) return;
    const r = panel.getBoundingClientRect();
    const sx = e.clientX;
    const sy = e.clientY;
    const startRight = globalThis.innerWidth - r.right;
    const startBottom = globalThis.innerHeight - r.bottom;
    const move = (ev: MouseEvent) => {
      offsetRef.current = {
        x: startRight - (ev.clientX - sx),
        y: startBottom - (ev.clientY - sy),
      };
      setOffset(offsetRef.current);
      clampToViewport();
    };
    const up = () => {
      globalThis.removeEventListener('mousemove', move);
      globalThis.removeEventListener('mouseup', up);
      writePosition(offsetRef.current);
    };
    globalThis.addEventListener('mousemove', move);
    globalThis.addEventListener('mouseup', up);
  };

  if (!open) return null;
  return (
    <div
      ref={panelRef}
      className="twk-panel"
      data-testid="twk-panel"
      style={{ right: offset.x, bottom: offset.y }}
    >
      <div className="twk-hd" onMouseDown={onDragStart} data-testid="twk-hd">
        <b>{title}</b>
        <button
          className="twk-x"
          aria-label="Close tweaks"
          onMouseDown={(e) => e.stopPropagation()}
          onClick={onClose}
        >
          ✕
        </button>
      </div>
      <div className="twk-body">{children}</div>
    </div>
  );
}

// ── Layout helpers ──────────────────────────────────────────────────────────

export interface TweakSectionProps {
  label: string;
  children?: ReactNode;
}

export function TweakSection({ label, children }: Readonly<TweakSectionProps>) {
  return (
    <>
      <div className="twk-sect">{label}</div>
      {children}
    </>
  );
}

export interface TweakRowProps {
  label: string;
  value?: ReactNode;
  inline?: boolean;
  children?: ReactNode;
}

export function TweakRow({ label, value, children, inline = false }: Readonly<TweakRowProps>) {
  return (
    <div className={inline ? 'twk-row twk-row-h' : 'twk-row'}>
      <div className="twk-lbl">
        <span>{label}</span>
        {value != null && <span className="twk-val">{value}</span>}
      </div>
      {children}
    </div>
  );
}

// ── Controls ────────────────────────────────────────────────────────────────

export interface TweakSliderProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
}

export function TweakSlider({
  label,
  value,
  min = 0,
  max = 100,
  step = 1,
  unit = '',
  onChange,
}: Readonly<TweakSliderProps>) {
  return (
    <TweakRow label={label} value={`${value}${unit}`}>
      <input
        type="range"
        className="twk-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
      />
    </TweakRow>
  );
}

export interface TweakToggleProps {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}

export function TweakToggle({ label, value, onChange }: Readonly<TweakToggleProps>) {
  return (
    <div className="twk-row twk-row-h">
      <div className="twk-lbl">
        <span>{label}</span>
      </div>
      <button
        type="button"
        className="twk-toggle"
        data-on={value ? '1' : '0'}
        role="switch"
        aria-checked={!!value}
        aria-label={label}
        onClick={() => onChange(!value)}
      >
        <i />
      </button>
    </div>
  );
}

export type TweakOption<V = string> = V | { value: V; label: string };

export interface TweakRadioProps<V extends string | number = string> {
  label: string;
  value: V;
  options: readonly TweakOption<V>[];
  onChange: (v: V) => void;
}

export function TweakRadio<V extends string | number = string>({
  label,
  value,
  options,
  onChange,
}: Readonly<TweakRadioProps<V>>) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const valueRef = useRef<V>(value);
  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  const labelLen = (o: TweakOption<V>): number =>
    String(typeof o === 'object' && o !== null && 'label' in o ? o.label : o).length;
  const maxLen = options.reduce((m, o) => Math.max(m, labelLen(o)), 0);
  const limits: Record<number, number> = { 2: 16, 3: 10 };
  const fitsAsSegments = maxLen <= (limits[options.length] ?? 0);

  if (!fitsAsSegments) {
    return <TweakSelect label={label} value={value} options={options} onChange={onChange} />;
  }

  const opts = options.map((o) =>
    typeof o === 'object' && o !== null && 'value' in o
      ? (o as { value: V; label: string })
      : { value: o as V, label: String(o) },
  );
  const idx = Math.max(
    0,
    opts.findIndex((o) => o.value === value),
  );
  const n = opts.length;

  const segAt = (clientX: number): V => {
    if (!trackRef.current) return opts[0].value;
    const r = trackRef.current.getBoundingClientRect();
    const inner = r.width - 4;
    const i = Math.floor(((clientX - r.left - 2) / inner) * n);
    return opts[Math.max(0, Math.min(n - 1, i))].value;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    setDragging(true);
    const v0 = segAt(e.clientX);
    if (v0 !== valueRef.current) onChange(v0);
    const move = (ev: PointerEvent) => {
      if (!trackRef.current) return;
      const v = segAt(ev.clientX);
      if (v !== valueRef.current) onChange(v);
    };
    const up = () => {
      setDragging(false);
      globalThis.removeEventListener('pointermove', move);
      globalThis.removeEventListener('pointerup', up);
    };
    globalThis.addEventListener('pointermove', move);
    globalThis.addEventListener('pointerup', up);
  };

  return (
    <TweakRow label={label}>
      <div
        ref={trackRef}
        role="radiogroup"
        aria-label={label}
        onPointerDown={onPointerDown}
        className={dragging ? 'twk-seg dragging' : 'twk-seg'}
      >
        <div
          className="twk-seg-thumb"
          style={{
            left: `calc(2px + ${idx} * (100% - 4px) / ${n})`,
            width: `calc((100% - 4px) / ${n})`,
          }}
        />
        {opts.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={o.value === value}
          >
            {o.label}
          </button>
        ))}
      </div>
    </TweakRow>
  );
}

export interface TweakSelectProps<V extends string | number = string> {
  label: string;
  value: V;
  options: readonly TweakOption<V>[];
  onChange: (v: V) => void;
}

export function TweakSelect<V extends string | number = string>({
  label,
  value,
  options,
  onChange,
}: Readonly<TweakSelectProps<V>>) {
  const resolve = (s: string): V => {
    const match = options.find(
      (o) => String(typeof o === 'object' && o !== null && 'value' in o ? o.value : o) === s,
    );
    if (match === undefined) return s as unknown as V;
    return typeof match === 'object' && match !== null && 'value' in match
      ? match.value
      : (match as V);
  };
  return (
    <TweakRow label={label}>
      <select
        className="twk-field"
        value={String(value)}
        onChange={(e) => onChange(resolve(e.target.value))}
        aria-label={label}
      >
        {options.map((o) => {
          const v = typeof o === 'object' && o !== null && 'value' in o ? o.value : o;
          const l = typeof o === 'object' && o !== null && 'label' in o ? o.label : String(o);
          return (
            <option key={String(v)} value={String(v)}>
              {l}
            </option>
          );
        })}
      </select>
    </TweakRow>
  );
}

export interface TweakTextProps {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (v: string) => void;
}

export function TweakText({ label, value, placeholder, onChange }: Readonly<TweakTextProps>) {
  return (
    <TweakRow label={label}>
      <input
        className="twk-field"
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
      />
    </TweakRow>
  );
}

export interface TweakNumberProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
}

export function TweakNumber({
  label,
  value,
  min,
  max,
  step = 1,
  unit = '',
  onChange,
}: Readonly<TweakNumberProps>) {
  const clamp = (n: number): number => {
    if (min != null && n < min) return min;
    if (max != null && n > max) return max;
    return n;
  };
  const startRef = useRef<{ x: number; val: number }>({ x: 0, val: 0 });
  const onScrubStart = (e: React.PointerEvent<HTMLSpanElement>) => {
    e.preventDefault();
    startRef.current = { x: e.clientX, val: value };
    const decimals = (String(step).split('.')[1] ?? '').length;
    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - startRef.current.x;
      const raw = startRef.current.val + dx * step;
      const snapped = Math.round(raw / step) * step;
      onChange(clamp(Number(snapped.toFixed(decimals))));
    };
    const up = () => {
      globalThis.removeEventListener('pointermove', move);
      globalThis.removeEventListener('pointerup', up);
    };
    globalThis.addEventListener('pointermove', move);
    globalThis.addEventListener('pointerup', up);
  };
  return (
    <div className="twk-num">
      <span
        className="twk-num-lbl"
        onPointerDown={onScrubStart}
        data-testid={`twk-num-scrub-${label}`}
      >
        {label}
      </span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        aria-label={label}
      />
      {unit && <span className="twk-num-unit">{unit}</span>}
    </div>
  );
}

// Relative-luminance contrast pick — checkmarks drawn over a swatch need to
// read on both #111 and #fafafa without per-option configuration. Hex input
// only (#rgb / #rrggbb); other color spaces fall through to "light".
function isLight(hex: string): boolean {
  const h = String(hex).replace('#', '');
  const x = h.length === 3 ? h.replace(/./g, (c) => c + c) : h.padEnd(6, '0');
  const n = parseInt(x.slice(0, 6), 16);
  if (Number.isNaN(n)) return true;
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return r * 299 + g * 587 + b * 114 > 148000;
}

const Check = ({ light }: { light: boolean }) => (
  <svg viewBox="0 0 14 14" aria-hidden="true">
    <path
      d="M3 7.2 5.8 10 11 4.2"
      fill="none"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      stroke={light ? 'rgba(0,0,0,.78)' : '#fff'}
    />
  </svg>
);

export type ColorOption = string | readonly string[];

export interface TweakColorProps {
  label: string;
  value: ColorOption;
  options?: readonly ColorOption[];
  onChange: (v: ColorOption) => void;
}

export function TweakColor({ label, value, options, onChange }: Readonly<TweakColorProps>) {
  if (!options || options.length === 0) {
    return (
      <div className="twk-row twk-row-h">
        <div className="twk-lbl">
          <span>{label}</span>
        </div>
        <input
          type="color"
          className="twk-swatch"
          value={typeof value === 'string' ? value : String(value[0] ?? '#000000')}
          onChange={(e) => onChange(e.target.value)}
          aria-label={label}
        />
      </div>
    );
  }
  // Native <input type=color> emits lowercase hex per the HTML spec.
  const key = (o: ColorOption): string => JSON.stringify(o).toLowerCase();
  const cur = key(value);
  return (
    <TweakRow label={label}>
      <div className="twk-chips" role="radiogroup" aria-label={label}>
        {options.map((o, i) => {
          const colors = Array.isArray(o) ? o : [o as string];
          const hero = colors[0];
          const rest = colors.slice(1);
          const sup = rest.slice(0, 4);
          const on = key(o) === cur;
          return (
            <button
              key={i}
              type="button"
              className="twk-chip"
              role="radio"
              aria-checked={on}
              data-on={on ? '1' : '0'}
              aria-label={colors.join(', ')}
              title={colors.join(' · ')}
              style={{ background: hero }}
              onClick={() => onChange(o)}
            >
              {sup.length > 0 && (
                <span>
                  {sup.map((c, j) => (
                    <i key={j} style={{ background: c }} />
                  ))}
                </span>
              )}
              {on && <Check light={isLight(hero)} />}
            </button>
          );
        })}
      </div>
    </TweakRow>
  );
}

export interface TweakButtonProps {
  label: string;
  onClick: () => void;
  secondary?: boolean;
}

export function TweakButton({ label, onClick, secondary = false }: Readonly<TweakButtonProps>) {
  return (
    <button
      type="button"
      className={secondary ? 'twk-btn secondary' : 'twk-btn'}
      onClick={onClick}
    >
      {label}
    </button>
  );
}
