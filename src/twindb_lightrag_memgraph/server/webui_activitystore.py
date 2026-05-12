"""Activity audit-feed storage backends (S4c slice 3).

Two implementations:

- ``InMemoryActivityStore`` — Python deque-style list, seeded from
  ``webui_seed.ACTIVITY``. Current behavior, default.
- ``MemgraphActivityStore`` — append-only ``:WebuiActivity_{workspace}``
  nodes, ordered by ``__created_at`` so newest events come back first when
  the router applies a range filter. Bootstrap from seed on empty workspace
  KV.

Both expose the same surface so the router can switch via the
``webui_activity_backend`` setting without code changes.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from .. import _pool
from .._constants import validate_identifier
from . import webui_seed

logger = logging.getLogger(__name__)


def _matches(
    e: dict[str, Any],
    *,
    kind: str | None,
    sev: str | None,
    actor: str | None,
    q: str | None,
) -> bool:
    if kind:
        wanted = {k for k in kind.split(",") if k}
        if wanted and e["kind"] not in wanted:
            return False
    if sev and sev != "any" and e["sev"] != sev:
        return False
    if actor and actor != "any" and e["actor"]["user"] != actor:
        return False
    if q:
        needle = q.lower()
        hay = (
            str(e.get("summary", ""))
            + " "
            + str(e.get("target", {}).get("label", ""))
            + " "
            + str(e.get("actor", {}).get("user", ""))
        ).lower()
        if needle not in hay:
            return False
    return True


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryActivityStore:
    """List-backed activity store seeded from ``webui_seed.ACTIVITY``."""

    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        now_ms: int = webui_seed.ACTIVITY_NOW_MS,
    ) -> None:
        self._events = copy.deepcopy(events if events is not None else webui_seed.ACTIVITY)
        self._now_ms = now_ms

    async def list(
        self,
        *,
        kind: str | None = None,
        sev: str | None = None,
        actor: str | None = None,
        q: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        filtered = [
            copy.deepcopy(e)
            for e in self._events
            if _matches(e, kind=kind, sev=sev, actor=actor, q=q)
        ]
        return filtered, self._now_ms

    async def append(self, event: dict[str, Any]) -> dict[str, Any]:
        stored = copy.deepcopy(event)
        self._events.insert(0, stored)
        return copy.deepcopy(stored)


# ---------------------------------------------------------------------------
# Memgraph backend
# ---------------------------------------------------------------------------


class MemgraphActivityStore:
    """Append-only :WebuiActivity_{workspace} backed activity feed.

    Newest-first ordering is encoded via ``__created_at`` set at MERGE-time
    and used in ORDER BY. Filtering happens server-side in Python after
    reading — at phase-1 volumes (~1k events / workspace / day) this stays
    well under 10ms, and the filter shape mirrors the in-memory variant
    exactly.
    """

    def __init__(self, workspace: str = "default") -> None:
        validate_identifier(workspace, "workspace")
        self._workspace = workspace
        self._now_ms = webui_seed.ACTIVITY_NOW_MS

    @property
    def _label(self) -> str:
        return f"WebuiActivity_{self._workspace}"

    async def initialize(self) -> None:
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                try:
                    result = await session.run(
                        f"CREATE INDEX ON :`{self._label}`(id)"
                    )
                    await result.consume()
                    logger.info(
                        "[WebuiActivityStore] Index on :%s(id) ensured", self._label
                    )
                except Exception as e:  # noqa: BLE001
                    if "already exists" not in str(e).lower():
                        raise

    async def bootstrap_if_empty(
        self, events: list[dict[str, Any]] | None = None
    ) -> bool:
        seed = events if events is not None else webui_seed.ACTIVITY
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label}`) RETURN count(n) AS c"
            )
            record = await result.single()
            await result.consume()
        count = record["c"] if record else 0
        if count > 0:
            return False
        # Insert oldest first so newest-first read order matches creation order.
        for ev in reversed(seed):
            await self.append(ev)
        logger.info(
            "[WebuiActivityStore] Bootstrapped %d events", len(seed)
        )
        return True

    async def list(
        self,
        *,
        kind: str | None = None,
        sev: str | None = None,
        actor: str | None = None,
        q: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label}`) RETURN n.data AS data "
                f"ORDER BY n.`__created_at` DESC"
            )
            rows = await result.data()
            await result.consume()
        events: list[dict[str, Any]] = []
        for row in rows:
            raw = row.get("data")
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        filtered = [e for e in events if _matches(e, kind=kind, sev=sev, actor=actor, q=q)]
        return filtered, self._now_ms

    async def append(self, event: dict[str, Any]) -> dict[str, Any]:
        if "id" not in event:
            raise ValueError("append requires event['id']")
        payload = json.dumps(event, sort_keys=True)
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    MERGE (n:`{self._label}` {{id: $id}})
                    ON CREATE SET n.`__created_at` = timestamp()
                    SET n.data = $data, n.`__updated_at` = timestamp()
                    """,
                    id=str(event["id"]),
                    data=payload,
                )
                await result.consume()
        return copy.deepcopy(event)


async def make_memgraph_activity_store(workspace: str = "default") -> MemgraphActivityStore:
    store = MemgraphActivityStore(workspace=workspace)
    await store.initialize()
    await store.bootstrap_if_empty()
    return store
