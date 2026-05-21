"""SQLite schema + connection helpers.

Schema is intentionally generic — a single ``entities`` table with a
``(kind, id)`` composite primary key and a JSON ``data`` text column.
The app can therefore add new entity kinds (graph nodes, audit events,
sessions, ...) without schema migrations. A separate ``mutations``
table holds the append-only audit log of every PATCH that hits the
backend — useful both for the maquette's Activity tab and for any
forensic question the steward might have during demo.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

DB_PATH = Path(os.environ.get("TWIN_DEMO_DB_PATH", "/data/twin-demo.sqlite"))
SEED_DIR = Path(os.environ.get("TWIN_DEMO_SEED_DIR", "/app/seed-data"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
  kind  TEXT NOT NULL,
  id    TEXT NOT NULL,
  data  TEXT NOT NULL,
  PRIMARY KEY (kind, id)
);
CREATE INDEX IF NOT EXISTS entities_kind_idx ON entities(kind);

CREATE TABLE IF NOT EXISTS mutations (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        TEXT    NOT NULL,
  kind      TEXT    NOT NULL,
  action    TEXT    NOT NULL,
  target_id TEXT,
  payload   TEXT
);
CREATE INDEX IF NOT EXISTS mutations_ts_idx ON mutations(ts);
"""


@asynccontextmanager
async def get_conn():
    """Async context manager yielding a fresh aiosqlite connection with
    PRAGMA setup applied. Callers ``async with get_conn() as conn``.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        yield conn


async def init_schema() -> None:
    async with get_conn() as conn:
        await conn.executescript(SCHEMA)
        await conn.commit()


async def _kind_count(conn: aiosqlite.Connection, kind: str) -> int:
    cur = await conn.execute("SELECT COUNT(*) FROM entities WHERE kind = ?", (kind,))
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def seed_if_empty() -> dict[str, int]:
    """Populate each entity kind from ``seed-data/<kind>.json`` if the
    table is empty. Returns the per-kind row counts after seeding so the
    boot log captures what landed.
    """
    counts: dict[str, int] = {}
    async with get_conn() as conn:
        for seed_file in sorted(SEED_DIR.glob("*.json")):
            # Skip AppleDouble metadata forks (._foo.json) that macOS tar
            # leaks into the image when the staging dir is on HFS+/APFS.
            if seed_file.name.startswith("._"):
                continue
            kind = seed_file.stem
            existing = await _kind_count(conn, kind)
            if existing > 0:
                counts[kind] = existing
                continue
            with seed_file.open(encoding="utf-8") as fh:
                items = json.load(fh)
            if not isinstance(items, list):
                raise ValueError(f"seed file {seed_file} must contain a JSON array")
            for it in items:
                if "id" not in it:
                    raise ValueError(f"seed entity in {seed_file} missing 'id': {it}")
                await conn.execute(
                    "INSERT INTO entities(kind, id, data) VALUES (?, ?, ?)",
                    (kind, str(it["id"]), json.dumps(it)),
                )
            await conn.commit()
            counts[kind] = await _kind_count(conn, kind)
    return counts
