// Twin demo persistence — sql.js (SQLite WASM) backed by IndexedDB.
//
// Why sql.js: a credible demo needs to show that "Approve" actually
// removes the document from the pending queue AND that the change
// survives a page reload. Cheaper paths (localStorage of MOCK_*
// arrays) were rejected so devtools show a real SQLite database file.
//
// Schema is intentionally generic — a single `entities` table with a
// (kind, id) composite primary key and a `data` JSON text column.
// The app can therefore add new entity kinds without schema
// migrations. The mutation log lives in its own table for audit.

(function () {
  const SQL_JS_BASE = "https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.3";
  const IDB_NAME = "twin-demo";
  const IDB_STORE = "sqlite-db";
  const IDB_KEY = "main";

  async function loadSqlJs() {
    if (window.initSqlJs) return window.initSqlJs;
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = `${SQL_JS_BASE}/sql-wasm.js`;
      s.onload = res;
      s.onerror = rej;
      document.head.appendChild(s);
    });
    return window.initSqlJs;
  }

  function openIdb() {
    return new Promise((res, rej) => {
      const req = indexedDB.open(IDB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(IDB_STORE)) db.createObjectStore(IDB_STORE);
      };
      req.onsuccess = () => res(req.result);
      req.onerror = () => rej(req.error);
    });
  }

  async function readBlob() {
    const db = await openIdb();
    return new Promise((res, rej) => {
      const tx = db.transaction(IDB_STORE, "readonly");
      const req = tx.objectStore(IDB_STORE).get(IDB_KEY);
      req.onsuccess = () => res(req.result || null);
      req.onerror = () => rej(req.error);
    });
  }

  async function writeBlob(bytes) {
    const db = await openIdb();
    return new Promise((res, rej) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(bytes, IDB_KEY);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  async function deleteBlob() {
    const db = await openIdb();
    return new Promise((res, rej) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).delete(IDB_KEY);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  let _db = null;       // sql.js Database handle
  let _saveTimer = null;
  let _SQL = null;      // sql.js module

  function persist() {
    // Debounce — coalesce bursts of saves (e.g. user batch-rejects 5 docs)
    // into a single IDB write. 200ms is the standard demo-feel sweet spot.
    if (_saveTimer) clearTimeout(_saveTimer);
    _saveTimer = setTimeout(async () => {
      if (!_db) return;
      try {
        await writeBlob(_db.export());
      } catch (e) {
        console.error("twinDb persist failed:", e);
      }
    }, 200);
  }

  async function boot() {
    _SQL = await (await loadSqlJs())({
      locateFile: (f) => `${SQL_JS_BASE}/${f}`
    });
    const existing = await readBlob();
    if (existing) {
      _db = new _SQL.Database(new Uint8Array(existing));
    } else {
      _db = new _SQL.Database();
      _db.run(`
        CREATE TABLE entities (
          kind TEXT NOT NULL,
          id   TEXT NOT NULL,
          data TEXT NOT NULL,
          PRIMARY KEY (kind, id)
        );
        CREATE TABLE mutations (
          id   INTEGER PRIMARY KEY AUTOINCREMENT,
          ts   TEXT NOT NULL,
          kind TEXT NOT NULL,
          action TEXT NOT NULL,
          target TEXT,
          payload TEXT
        );
      `);
      persist();
    }
  }

  function getAll(kind) {
    if (!_db) return null;
    const stmt = _db.prepare("SELECT id, data FROM entities WHERE kind = ?");
    stmt.bind([kind]);
    const out = [];
    while (stmt.step()) {
      const row = stmt.getAsObject();
      try { out.push(JSON.parse(row.data)); } catch (e) {}
    }
    stmt.free();
    return out.length ? out : null;
  }

  function replaceAll(kind, items) {
    if (!_db) return;
    _db.run("DELETE FROM entities WHERE kind = ?", [kind]);
    const stmt = _db.prepare("INSERT INTO entities (kind, id, data) VALUES (?, ?, ?)");
    items.forEach(it => {
      stmt.run([kind, String(it.id), JSON.stringify(it)]);
    });
    stmt.free();
    persist();
  }

  function logMutation(kind, action, target, payload) {
    if (!_db) return;
    _db.run(
      "INSERT INTO mutations (ts, kind, action, target, payload) VALUES (?, ?, ?, ?, ?)",
      [new Date().toISOString(), kind, action, target || null, payload ? JSON.stringify(payload) : null]
    );
    persist();
  }

  async function reset() {
    await deleteBlob();
    location.reload();
  }

  function stats() {
    if (!_db) return null;
    const res = _db.exec(`
      SELECT 'entities' AS t, COUNT(*) AS n FROM entities
      UNION ALL
      SELECT 'mutations', COUNT(*) FROM mutations
    `);
    if (!res[0]) return null;
    const out = {};
    res[0].values.forEach(([t, n]) => { out[t] = n; });
    return out;
  }

  window.twinDb = { boot, getAll, replaceAll, logMutation, reset, stats };
})();
