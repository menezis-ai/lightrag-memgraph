"""Shared Memgraph I/O for the portable stores — reads through the read pool,
writes through the write slot + conflict retry (the house idioms of
``kv_impl`` / ``docstatus_impl``), keyset pagination by the store key."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .. import _pool
from .._retry import with_conflict_retry


async def read_rows(query: str, **params: Any) -> list[dict[str, Any]]:
    async with _pool.get_read_session() as session:
        result = await session.run(query, **params)
        rows = [dict(record) async for record in result]
        await result.consume()
    return rows


async def read_scalar(query: str, **params: Any) -> Any:
    rows = await read_rows(query, **params)
    if not rows:
        return None
    return next(iter(rows[0].values()))


async def write(op_name: str, query: str, **params: Any) -> list[dict[str, Any]]:
    """One write statement under the write slot, retried on write/write conflict."""

    async def _write() -> list[dict[str, Any]]:
        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            result = await session.run(query, **params)
            rows = [dict(record) async for record in result]
            await result.consume()
        return rows

    return await with_conflict_retry(op_name, _write)


async def keyset(
    fetch: Callable[[Any | None, int], Awaitable[list[dict[str, Any]]]],
    key_of: Callable[[dict[str, Any]], Any],
    batch: int,
) -> AsyncIterator[dict[str, Any]]:
    """Generic keyset loop: *fetch(after, limit)* returns rows ordered by key."""
    after: Any | None = None
    while True:
        rows = await fetch(after, batch)
        for row in rows:
            yield row
        if len(rows) < batch:
            return
        after = key_of(rows[-1])


async def batched(
    records: AsyncIterator[dict[str, Any]], size: int
) -> AsyncIterator[list[dict[str, Any]]]:
    buf: list[dict[str, Any]] = []
    async for record in records:
        buf.append(record)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
