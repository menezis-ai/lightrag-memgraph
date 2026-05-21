// Twin demo persistence — talks to the FastAPI backend at /api/*.
//
// Replaces the earlier sql.js WASM client-side approach (PR #57, reverted)
// because BNP infra won't accept a CDN-fetched WASM blob and a real
// backend with an auditable SQLite file in a Docker volume is the
// production-shaped answer. See maquette-deploy/backend/ for the API
// implementation.
//
// API surface kept compatible with the rest of the app so app.jsx /
// documents.jsx / topbar.jsx don't change shape:
//
//   await window.twinDb.boot()       → fetches /api/health, caches counts
//   await window.twinDb.getAll(kind) → GET /api/<kind>
//   await window.twinDb.patch(kind, id, patch) → PATCH /api/<kind>/<id>
//   await window.twinDb.reset()      → POST /api/state/reset + reload
//   window.twinDb.stats()            → cached {counts, mutations}
//
// All methods are async; callers must await. Errors surface as console
// warnings + return null so the UI degrades gracefully if the backend
// is unreachable (better than blocking the demo).

(function () {
  const API_BASE = (window.TWIN_API_BASE || "/api").replace(/\/$/, "");
  let _lastStats = null;

  async function _request(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(API_BASE + path, opts);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${method} ${path} → ${res.status}: ${text.slice(0, 200)}`);
    }
    // Some endpoints (DELETE) return {ok: true}; some return arrays.
    return res.json();
  }

  async function boot() {
    try {
      _lastStats = await _request("GET", "/health");
      return _lastStats;
    } catch (err) {
      console.warn("twinDb boot failed:", err.message);
      _lastStats = null;
      throw err;
    }
  }

  async function getAll(kind) {
    try {
      return await _request("GET", "/" + encodeURIComponent(kind));
    } catch (err) {
      console.warn(`twinDb getAll(${kind}) failed:`, err.message);
      return null;
    }
  }

  async function patch(kind, id, payload) {
    try {
      return await _request(
        "PATCH",
        `/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`,
        { patch: payload }
      );
    } catch (err) {
      console.warn(`twinDb patch(${kind}, ${id}) failed:`, err.message);
      return null;
    }
  }

  async function removeOne(kind, id) {
    try {
      return await _request(
        "DELETE",
        `/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`
      );
    } catch (err) {
      console.warn(`twinDb remove(${kind}, ${id}) failed:`, err.message);
      return null;
    }
  }

  async function reset() {
    try {
      await _request("POST", "/state/reset");
    } catch (err) {
      console.warn("twinDb reset failed:", err.message);
    }
    location.reload();
  }

  function stats() {
    return _lastStats;
  }

  window.twinDb = { boot, getAll, patch, removeOne, reset, stats };
})();
