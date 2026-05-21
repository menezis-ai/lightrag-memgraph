"""FastAPI app — REST persistence for the Twin demo maquette.

Routes
------
GET  /api/health                  → liveness + per-kind counts
GET  /api/{kind}                  → all rows of the kind (e.g. docs, tags)
GET  /api/{kind}/{id}             → one row
PATCH /api/{kind}/{id}            → deep-merge patch + audit log entry
DELETE /api/{kind}/{id}           → remove one row (+ audit)
POST /api/state/reset             → drop + reseed everything (demo reset)
GET  /api/mutations?limit=50      → recent audit-log entries

All responses are JSON. CORS is wide-open because the maquette is a
static SPA served by Caddy on the same origin in prod (reverse-proxy
sticks /api/* here) but a developer may also load the HTML from
file:// during a designer review.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import get_conn, init_schema, seed_if_empty

log = logging.getLogger("twin_demo_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema()
    counts = await seed_if_empty()
    log.info("twin-demo-api boot: seeded %s", counts)
    yield


# Disable FastAPI's auto-mounted Swagger UI / ReDoc / openapi.json — the
# entity kind 'docs' would otherwise be shadowed by the /docs Swagger
# route after Caddy strips the /api prefix.
app = FastAPI(
    title="Twin demo API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _merge(base: dict, patch: dict) -> dict:
    """Deep-merge ``patch`` into a copy of ``base`` — nested dicts are
    merged key-by-key; lists / scalars are overwritten.
    """
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


async def _log_mutation(
    conn: aiosqlite.Connection,
    kind: str,
    action: str,
    target_id: str | None,
    payload: dict | None,
) -> None:
    await conn.execute(
        "INSERT INTO mutations(ts, kind, action, target_id, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        (_now(), kind, action, target_id, json.dumps(payload) if payload else None),
    )


@app.get("/health")
async def health() -> dict:
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT kind, COUNT(*) FROM entities GROUP BY kind"
        )
        counts = {row[0]: row[1] for row in await cur.fetchall()}
        cur = await conn.execute("SELECT COUNT(*) FROM mutations")
        mutations = (await cur.fetchone())[0]
    return {"ok": True, "counts": counts, "mutations": mutations}


@app.get("/mutations")
async def list_mutations(limit: int = 50) -> list[dict[str, Any]]:
    """MUST be declared before the `/{kind}` catch-all, otherwise FastAPI
    matches `/mutations` against the catch-all and tries to list a
    (nonexistent) entity kind named 'mutations'.
    """
    limit = max(1, min(500, limit))
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT id, ts, kind, action, target_id, payload "
            "FROM mutations ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        out = []
        for row in await cur.fetchall():
            out.append({
                "id": row[0],
                "ts": row[1],
                "kind": row[2],
                "action": row[3],
                "target_id": row[4],
                "payload": json.loads(row[5]) if row[5] else None,
            })
    return out


@app.post("/state/reset")
async def reset_state() -> dict:
    """Same ordering note as above — must precede `/{kind}/{entity_id}`."""
    async with get_conn() as conn:
        await conn.execute("DELETE FROM entities")
        await conn.execute("DELETE FROM mutations")
        await _log_mutation(conn, "state", "reset", None, None)
        await conn.commit()
    counts = await seed_if_empty()
    return {"ok": True, "reseeded": counts}


@app.get("/{kind}")
async def list_kind(kind: str) -> list[dict[str, Any]]:
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT data FROM entities WHERE kind = ? ORDER BY id", (kind,)
        )
        return [json.loads(r[0]) for r in await cur.fetchall()]


@app.get("/{kind}/{entity_id}")
async def get_one(kind: str, entity_id: str) -> dict[str, Any]:
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT data FROM entities WHERE kind = ? AND id = ?",
            (kind, entity_id),
        )
        row = await cur.fetchone()
        if not row:
            raise HTTPException(404, f"{kind}/{entity_id} not found")
        return json.loads(row[0])


class PatchBody(BaseModel):
    patch: dict[str, Any]


@app.patch("/{kind}/{entity_id}")
async def patch_one(kind: str, entity_id: str, body: PatchBody) -> dict[str, Any]:
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT data FROM entities WHERE kind = ? AND id = ?",
            (kind, entity_id),
        )
        row = await cur.fetchone()
        if not row:
            raise HTTPException(404, f"{kind}/{entity_id} not found")
        merged = _merge(json.loads(row[0]), body.patch)
        await conn.execute(
            "UPDATE entities SET data = ? WHERE kind = ? AND id = ?",
            (json.dumps(merged), kind, entity_id),
        )
        await _log_mutation(conn, kind, "patch", entity_id, body.patch)
        await conn.commit()
        return merged


@app.delete("/{kind}/{entity_id}")
async def delete_one(kind: str, entity_id: str) -> dict:
    async with get_conn() as conn:
        cur = await conn.execute(
            "DELETE FROM entities WHERE kind = ? AND id = ?",
            (kind, entity_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, f"{kind}/{entity_id} not found")
        await _log_mutation(conn, kind, "delete", entity_id, None)
        await conn.commit()
    return {"ok": True}


# /state/reset and /mutations declared above (before the /{kind} catch-all)
# to avoid the catch-all shadowing them.
