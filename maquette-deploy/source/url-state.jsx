// Shared URL-state helper — keep filters in sync with ?query so views are shareable
// Usage: const [v, setV] = useUrlParam("status", "all", { parse, serialize, validate })
const { useState: _useStateUrl, useEffect: _useEffectUrl, useCallback: _useCallbackUrl } = React;

function readParams() {
  return new URLSearchParams(window.location.search);
}
function writeParam(key, val) {
  const p = readParams();
  if (val === null || val === undefined || val === "") p.delete(key);
  else p.set(key, val);
  const q = p.toString();
  const url = window.location.pathname + (q ? "?" + q : "");
  window.history.replaceState(null, "", url);
}

window.useUrlParam = function useUrlParam(key, defaultValue, opts = {}) {
  const parse = opts.parse || ((s) => s);
  const serialize = opts.serialize || ((v) => (v == null ? "" : String(v)));
  const validate = opts.validate || (() => true);

  const [val, setVal] = _useStateUrl(() => {
    const raw = readParams().get(key);
    if (raw == null) return defaultValue;
    try {
      const parsed = parse(raw);
      return validate(parsed) ? parsed : defaultValue;
    } catch (e) {
      return defaultValue;
    }
  });

  _useEffectUrl(() => {
    const ser = serialize(val);
    const isDefault = ser === serialize(defaultValue);
    writeParam(key, isDefault ? "" : ser);
  }, [val]);

  return [val, setVal];
};

// Array (comma-separated) variant
window.useUrlArrayParam = function useUrlArrayParam(key, defaultValue) {
  return window.useUrlParam(key, defaultValue, {
    parse: (s) => s.split(",").map(x => x.trim()).filter(Boolean),
    serialize: (arr) => (arr && arr.length ? arr.join(",") : ""),
    validate: (v) => Array.isArray(v)
  });
};

// Numeric variant
window.useUrlNumberParam = function useUrlNumberParam(key, defaultValue) {
  return window.useUrlParam(key, defaultValue, {
    parse: (s) => { const n = Number(s); return Number.isFinite(n) ? n : defaultValue; },
    serialize: (n) => String(n),
    validate: (v) => typeof v === "number" && Number.isFinite(v)
  });
};
