/**
 * URL-backed state hooks — keep filters in sync with `?query` so views are
 * shareable / bookmarkable.
 *
 * Ported from Desktop/UI/url-state.jsx. Three variants:
 *   useUrlParam<T>(key, default, { parse, serialize, validate })
 *   useUrlArrayParam(key, default)         — comma-separated list
 *   useUrlNumberParam(key, default)        — finite number
 *
 * Defaults are stripped from the URL (so a "clean" URL means defaults
 * everywhere). Invalid stored values fall back to default and are not
 * re-written until the user changes them.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

const URL_STATE_EVENT = 'twin:url-state-change';

export interface UseUrlParamOptions<T> {
  parse?: (raw: string) => T;
  serialize?: (value: T) => string;
  validate?: (value: T) => boolean;
}

function readParams(): URLSearchParams {
  return new URLSearchParams(globalThis.location.search);
}

function writeParam(key: string, val: string): void {
  const p = readParams();
  if (val === '') p.delete(key);
  else p.set(key, val);
  const q = p.toString();
  const url = globalThis.location.pathname + (q ? '?' + q : '');
  globalThis.history.replaceState(null, '', url);
  globalThis.dispatchEvent(new CustomEvent(URL_STATE_EVENT, { detail: { key } }));
}

export function useUrlParam<T>(
  key: string,
  defaultValue: T,
  opts: UseUrlParamOptions<T> = {},
): [T, (v: T) => void] {
  const parse = useMemo(
    () => opts.parse ?? ((s: string) => s as unknown as T),
    [opts.parse],
  );
  const serialize = useMemo(
    () =>
      opts.serialize ??
      ((v: T) => (v === null || v === undefined ? '' : String(v))),
    [opts.serialize],
  );
  const validate = useMemo(
    () => opts.validate ?? (() => true),
    [opts.validate],
  );

  const [val, setVal] = useState<T>(() => {
    const raw = readParams().get(key);
    if (raw === null) return defaultValue;
    try {
      const parsed = parse(raw);
      return validate(parsed) ? parsed : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    const readValue = (): T => {
      const raw = readParams().get(key);
      if (raw === null) return defaultValue;
      try {
        const parsed = parse(raw);
        return validate(parsed) ? parsed : defaultValue;
      } catch {
        return defaultValue;
      }
    };
    const syncFromUrl = () => {
      const next = readValue();
      setVal((current) =>
        serialize(current) === serialize(next) ? current : next,
      );
    };
    globalThis.addEventListener(URL_STATE_EVENT, syncFromUrl);
    globalThis.addEventListener('popstate', syncFromUrl);
    return () => {
      globalThis.removeEventListener(URL_STATE_EVENT, syncFromUrl);
      globalThis.removeEventListener('popstate', syncFromUrl);
    };
  }, [defaultValue, key, parse, serialize, validate]);

  useEffect(() => {
    const ser = serialize(val);
    const isDefault = ser === serialize(defaultValue);
    writeParam(key, isDefault ? '' : ser);
    // We intentionally only depend on `val`. `key`/`defaultValue`/`serialize`
    // are treated as stable per render — matching the proto's behavior.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [val]);

  return [val, setVal];
}

export function useUrlArrayParam(
  key: string,
  defaultValue: readonly string[],
): [readonly string[], (v: readonly string[]) => void] {
  return useUrlParam<readonly string[]>(key, defaultValue, {
    parse: (s) =>
      s
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean),
    serialize: (arr) => (arr?.length ? arr.join(',') : ''),
    validate: (v) => Array.isArray(v),
  });
}

export function useUrlNumberParam(
  key: string,
  defaultValue: number,
): [number, (v: number) => void] {
  // useCallback to satisfy the parse contract with a stable identity.
  const parse = useCallback(
    (s: string): number => {
      const n = Number(s);
      return Number.isFinite(n) ? n : defaultValue;
    },
    [defaultValue],
  );

  return useUrlParam<number>(key, defaultValue, {
    parse,
    serialize: String,
    validate: (v) => typeof v === 'number' && Number.isFinite(v),
  });
}
