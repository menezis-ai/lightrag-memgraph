"""Notifications storage backends (S4c slice 3).

Same shape as the tag + activity stores: a Protocol with two
implementations (``InMemoryNotificationStore``, ``MemgraphNotificationStore``)
chosen at startup by the ``webui_notifications_backend`` setting.

The mutation surface (mark-all-read, clear, push) mirrors what the WebUI
Topbar calls. ``push`` is invoked server-side by the tag-mutation endpoints
to surface a unread notification for every governance event.
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


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryNotificationStore:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self._items = copy.deepcopy(
            items if items is not None else webui_seed.NOTIFICATIONS
        )

    async def list(self) -> list[dict[str, Any]]:  # NOSONAR - async contract.
        return copy.deepcopy(self._items)

    async def mark_all_read(self) -> None:  # NOSONAR - async contract.
        for n in self._items:
            n["read"] = True

    async def clear(self) -> None:  # NOSONAR - async contract.
        self._items.clear()

    async def push(  # NOSONAR - async contract.
        self, notification: dict[str, Any]
    ) -> dict[str, Any]:
        stored = copy.deepcopy(notification)
        # Newest-first: prepend.
        self._items.insert(0, stored)
        return copy.deepcopy(stored)


# ---------------------------------------------------------------------------
# Memgraph backend
# ---------------------------------------------------------------------------


class MemgraphNotificationStore:
    def __init__(self, workspace: str = "default") -> None:
        validate_identifier(workspace, "workspace")
        self._workspace = workspace

    @property
    def _label(self) -> str:
        return f"WebuiNotification_{self._workspace}"

    async def initialize(self) -> None:
        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            try:
                result = await session.run(f"CREATE INDEX ON :`{self._label}`(id)")
                await result.consume()
            except Exception as e:  # noqa: BLE001
                if "already exists" not in str(e).lower():
                    raise

    async def bootstrap_if_empty(
        self, items: list[dict[str, Any]] | None = None
    ) -> bool:
        seed = items if items is not None else webui_seed.NOTIFICATIONS
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label}`) RETURN count(n) AS c"
            )
            record = await result.single()
            await result.consume()
        if record and record.get("c", 0) > 0:
            return False
        for n in reversed(seed):
            await self.push(n)
        logger.info("[WebuiNotificationStore] Bootstrapped %d notifications", len(seed))
        return True

    async def list(self) -> list[dict[str, Any]]:
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label}`) RETURN n.data AS data "
                f"ORDER BY n.`__created_at` DESC"
            )
            rows = await result.data()
            await result.consume()
        out: list[dict[str, Any]] = []
        for row in rows:
            raw = row.get("data")
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out

    async def mark_all_read(self) -> None:
        items = await self.list()
        rows: list[dict[str, str]] = []
        for n in items:
            n["read"] = True
            rows.append(
                {
                    "id": str(n["id"]),
                    "data": json.dumps(n, sort_keys=True),
                }
            )
        if not rows:
            return

        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            result = await session.run(
                f"""
                    UNWIND $rows AS row
                    MATCH (m:`{self._label}` {{id: row.id}})
                    SET m.data = row.data, m.`__updated_at` = timestamp()
                    """,
                rows=rows,
            )
            await result.consume()

    async def clear(self) -> None:
        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            result = await session.run(f"MATCH (n:`{self._label}`) DETACH DELETE n")
            await result.consume()

    async def push(self, notification: dict[str, Any]) -> dict[str, Any]:
        if "id" not in notification:
            raise ValueError("push requires notification['id']")
        async with _pool.acquire_write_slot(), _pool.get_session() as session:
            result = await session.run(
                f"""
                    MERGE (n:`{self._label}` {{id: $id}})
                    ON CREATE SET n.`__created_at` = timestamp()
                    SET n.data = $data, n.`__updated_at` = timestamp()
                    """,
                id=str(notification["id"]),
                data=json.dumps(notification, sort_keys=True),
            )
            await result.consume()
        return copy.deepcopy(notification)


async def make_memgraph_notification_store(
    workspace: str = "default",
) -> MemgraphNotificationStore:
    store = MemgraphNotificationStore(workspace=workspace)
    await store.initialize()
    await store.bootstrap_if_empty()
    return store
