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

import datetime
import copy
import json
import logging
import time
from typing import Any

from .. import _pool
from .._constants import validate_identifier
from . import webui_seed

logger = logging.getLogger(__name__)
DEFAULT_ACTIVITY_LIMIT = 200
MAX_ACTIVITY_LIMIT = 1000
MEMGRAPH_COMPAT_SCAN_LIMIT = 1000
MEMGRAPH_BACKFILL_BATCH = 200
SCALARS_VERSION = 2
_LEGACY_SCALARS_CLAUSE = (
    "n.`__scalars_version` IS NULL OR n.`__scalars_version` < $scalars_version"
)
RANGE_TO_MS = {
    "24h": 24 * 60 * 60 * 1000,
    "7d": 7 * 24 * 60 * 60 * 1000,
    "30d": 30 * 24 * 60 * 60 * 1000,
}


def _dict_field(event: dict[str, Any], key: str) -> dict[str, Any]:
    value = event.get(key)
    return value if isinstance(value, dict) else {}


def _meta_doc_ids(meta: dict[str, Any]) -> list[str]:
    raw_doc_ids = meta.get("doc_ids")
    if not isinstance(raw_doc_ids, list):
        return []
    return [str(doc_id) for doc_id in raw_doc_ids if doc_id is not None]


def _matches_kind(event: dict[str, Any], kind: str | None) -> bool:
    if not kind:
        return True
    wanted = {k for k in kind.split(",") if k}
    return not wanted or event["kind"] in wanted


def _matches_range(event: dict[str, Any], range: str | None, now_ms: int) -> bool:
    cutoff_ms = _range_to_cutoff_ms(range=range, now_ms=now_ms)
    if cutoff_ms is None:
        return True
    event_ms = _event_ts_ms(event.get("ts"))
    return event_ms is not None and event_ms >= cutoff_ms


def _matches_resource(
    target: dict[str, Any],
    meta: dict[str, Any],
    resource_id: str | None,
) -> bool:
    if not resource_id:
        return True
    wanted = str(resource_id)
    resource_ids = [
        str(candidate)
        for candidate in (target.get("id"), meta.get("doc_id"))
        if candidate is not None
    ]
    resource_ids.extend(_meta_doc_ids(meta))
    return wanted in resource_ids


def _matches_text(
    event: dict[str, Any],
    target: dict[str, Any],
    actor_data: dict[str, Any],
    q: str | None,
) -> bool:
    if not q:
        return True
    haystack = " ".join(
        (
            str(event.get("summary", "")),
            str(target.get("label", "")),
            str(actor_data.get("user", "")),
            str(event.get("id", "")),
        )
    ).lower()
    return q.lower() in haystack


def _matches(
    e: dict[str, Any],
    *,
    kind: str | None,
    sev: str | None,
    actor: str | None,
    q: str | None,
    range: str | None,
    now_ms: int,
    resource_id: str | None,
) -> bool:
    target = _dict_field(e, "target")
    meta = _dict_field(e, "meta")
    actor_data = _dict_field(e, "actor")
    return (
        _matches_kind(e, kind)
        and (not sev or sev == "any" or e["sev"] == sev)
        and (not actor or actor == "any" or str(actor_data.get("user")) == actor)
        and _matches_text(e, target, actor_data, q)
        and _matches_range(e, range, now_ms)
        and _matches_resource(target, meta, resource_id)
    )


def _bounded_limit(limit: int | None) -> int:
    return max(1, min(limit or DEFAULT_ACTIVITY_LIMIT, MAX_ACTIVITY_LIMIT))


def _range_to_cutoff_ms(*, range: str | None, now_ms: int) -> int | None:
    if range is None or range == "all":
        return None
    window_ms = RANGE_TO_MS.get(range)
    if window_ms is None:
        return None
    return now_ms - window_ms


def _event_ts_ms(event_ts: Any) -> int | None:
    if not isinstance(event_ts, str) or not event_ts:
        return None
    try:
        return int(
            datetime.datetime.fromisoformat(event_ts.replace("Z", "+00:00")).timestamp()
            * 1000
        )
    except ValueError:
        return None


def _event_scalars(event: dict[str, Any]) -> dict[str, Any]:
    """Mirror hot filters as properties so Memgraph can index them."""
    actor = _dict_field(event, "actor")
    target = _dict_field(event, "target")
    meta = _dict_field(event, "meta")
    return {
        "kind": str(event.get("kind") or ""),
        "sev": str(event.get("sev") or ""),
        "actor_user": str(actor.get("user") or ""),
        "target_id": str(target.get("id") or ""),
        "target_label": str(target.get("label") or ""),
        "meta_doc_id": str(meta.get("doc_id") or ""),
        "meta_doc_ids": _meta_doc_ids(meta),
        "ts_ms": _event_ts_ms(event.get("ts")),
        "summary": str(event.get("summary") or ""),
    }


def _to_where(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _decode_rows(
    raw_rows: list[dict[str, Any]],
    *,
    fallback_created_at: int,
) -> list[tuple[int, dict[str, Any]]]:
    decoded: list[tuple[int, dict[str, Any]]] = []
    for row in raw_rows:
        raw = row.get("data")
        if not raw:
            continue
        try:
            created_at = int(row.get("created_at", fallback_created_at))
        except (TypeError, ValueError):
            created_at = fallback_created_at
        try:
            decoded.append((created_at, json.loads(raw)))
        except json.JSONDecodeError:
            continue
    return decoded


def _add_shared_filter(
    *,
    enabled: bool,
    clause: str,
    param_name: str,
    param_value: Any,
    base_clauses: list[str],
    scalar_clauses: list[str],
    params: dict[str, Any],
) -> None:
    if not enabled:
        return
    base_clauses.append(clause)
    scalar_clauses.append(clause)
    params[param_name] = param_value


def _activity_query_filters(
    *,
    kind: str | None,
    sev: str | None,
    actor: str | None,
    q: str | None,
    now_range_ms: int | None,
    resource_id: str | None,
    need_json_fallback: bool,
    limit: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    base_clauses: list[str] = []
    scalar_clauses: list[str] = []
    params: dict[str, Any] = {
        "limit": limit,
        "compat_limit": min(MEMGRAPH_COMPAT_SCAN_LIMIT, MAX_ACTIVITY_LIMIT),
        "scalars_version": SCALARS_VERSION,
    }
    kind_values = [k for k in (kind or "").split(",") if k]
    shared_filters = (
        (bool(kind_values), "n.kind IN $kinds", "kinds", kind_values),
        (bool(sev and sev != "any"), "n.sev = $sev", "sev", sev),
        (bool(actor and actor != "any"), "n.actor_user = $actor", "actor", actor),
        (bool(q), _text_search_clause(), "needle", q.lower() if q else None),
    )
    for enabled, clause, param_name, param_value in shared_filters:
        _add_shared_filter(
            enabled=enabled,
            clause=clause,
            param_name=param_name,
            param_value=param_value,
            base_clauses=base_clauses,
            scalar_clauses=scalar_clauses,
            params=params,
        )
    if now_range_ms is not None:
        scalar_clauses.append("n.ts_ms >= $range_min")
        params["range_min"] = now_range_ms
    if resource_id:
        scalar_clauses.append(_resource_clause())
        params["resource_id"] = resource_id
    if need_json_fallback:
        scalar_clauses.append("n.`__scalars_version` >= $scalars_version")
    return base_clauses, scalar_clauses, params


def _text_search_clause() -> str:
    return (
        "("
        "toLower(coalesce(n.summary, '')) CONTAINS $needle OR "
        "toLower(coalesce(n.target_label, '')) CONTAINS $needle OR "
        "toLower(coalesce(n.actor_user, '')) CONTAINS $needle OR "
        "toLower(coalesce(n.id, '')) CONTAINS $needle"
        ")"
    )


def _resource_clause() -> str:
    return (
        "("
        "n.target_id = $resource_id OR "
        "n.meta_doc_id = $resource_id OR "
        "$resource_id IN coalesce(n.meta_doc_ids, [])"
        ")"
    )


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
        self._events = copy.deepcopy(
            events if events is not None else webui_seed.ACTIVITY
        )
        self._now_ms = now_ms

    async def list(  # NOSONAR - async contract.
        self,
        *,
        kind: str | None = None,
        sev: str | None = None,
        actor: str | None = None,
        q: str | None = None,
        range: str | None = None,
        resource_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        now_ms = self._now_ms
        filtered = [
            copy.deepcopy(e)
            for e in self._events
            if _matches(
                e,
                kind=kind,
                sev=sev,
                actor=actor,
                q=q,
                range=range,
                now_ms=now_ms,
                resource_id=resource_id,
            )
        ]
        total = len(filtered)
        return filtered[: _bounded_limit(limit)], total, now_ms

    async def append(
        self, event: dict[str, Any]
    ) -> dict[str, Any]:  # NOSONAR - async contract.
        stored = copy.deepcopy(event)
        self._events.insert(0, stored)
        return copy.deepcopy(stored)


# ---------------------------------------------------------------------------
# Memgraph backend
# ---------------------------------------------------------------------------


class MemgraphActivityStore:
    """Append-only :WebuiActivity_{workspace} backed activity feed.

    Newest-first ordering is encoded via ``__created_at`` set at MERGE-time
    and used in ORDER BY. Hot filter fields are duplicated as scalar node
    properties so Memgraph can use indexes before the bounded JSON decode pass.
    """

    def __init__(self, workspace: str = "default") -> None:
        validate_identifier(workspace, "workspace")
        self._workspace = workspace

    @property
    def _label(self) -> str:
        return f"WebuiActivity_{self._workspace}"

    async def _backfill_legacy_scalars(self) -> None:
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{self._label}`) WHERE {_LEGACY_SCALARS_CLAUSE} "
                "RETURN n.id AS id, n.data AS data",
                scalars_version=SCALARS_VERSION,
            )
            rows = await result.data()
            await result.consume()

        updates: list[dict[str, Any]] = []
        for row in rows:
            raw_data = row.get("data")
            if not raw_data:
                continue
            try:
                event = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_id = row.get("id")
            if not event_id:
                continue
            scalars = _event_scalars(event)
            updates.append(
                {
                    "id": str(event_id),
                    **scalars,
                }
            )

        if not updates:
            return

        for offset in range(0, len(updates), MEMGRAPH_BACKFILL_BATCH):
            batch = updates[offset : offset + MEMGRAPH_BACKFILL_BATCH]
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        UNWIND $updates AS row
                        MATCH (n:`{self._label}`)
                        WHERE n.id = row.id
                        SET n.kind = row.kind,
                            n.sev = row.sev,
                            n.actor_user = row.actor_user,
                            n.target_id = row.target_id,
                            n.meta_doc_id = row.meta_doc_id,
                            n.meta_doc_ids = row.meta_doc_ids,
                            n.ts_ms = row.ts_ms,
                            n.target_label = row.target_label,
                            n.summary = row.summary,
                            n.`__scalars_version` = $scalars_version,
                            n.`__updated_at` = timestamp()
                        """,
                        updates=batch,
                        scalars_version=SCALARS_VERSION,
                    )
                    await result.consume()
        logger.info(
            "[WebuiActivityStore] Backfilled scalars for %d legacy events on %s",
            len(updates),
            self._label,
        )

    async def initialize(self) -> None:
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                for prop in (
                    "id",
                    "__created_at",
                    "kind",
                    "sev",
                    "actor_user",
                    "target_id",
                    "meta_doc_id",
                    "meta_doc_ids",
                    "ts_ms",
                    "__scalars_version",
                ):
                    try:
                        result = await session.run(
                            f"CREATE INDEX ON :`{self._label}`(`{prop}`)"
                        )
                        await result.consume()
                        logger.info(
                            "[WebuiActivityStore] Index on :%s(%s) ensured",
                            self._label,
                            prop,
                        )
                    except Exception as e:  # noqa: BLE001
                        if "already exists" not in str(e).lower():
                            raise
        await self._backfill_legacy_scalars()

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
        logger.info("[WebuiActivityStore] Bootstrapped %d events", len(seed))
        return True

    async def list(
        self,
        *,
        kind: str | None = None,
        sev: str | None = None,
        actor: str | None = None,
        q: str | None = None,
        range: str | None = None,
        resource_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        limit = _bounded_limit(limit)
        now_ms = int(time.time() * 1000)
        now_range_ms = _range_to_cutoff_ms(range=range, now_ms=now_ms)
        range_requested = range is not None and range != "all" and range in RANGE_TO_MS
        need_json_fallback = range_requested or bool(resource_id)
        base_clauses, scalar_clauses, params = _activity_query_filters(
            kind=kind,
            sev=sev,
            actor=actor,
            q=q,
            now_range_ms=now_range_ms,
            resource_id=resource_id,
            need_json_fallback=need_json_fallback,
            limit=limit,
        )

        where_scalar = _to_where(scalar_clauses)
        data_query = (
            f"MATCH (n:`{self._label}`) {where_scalar} "
            "RETURN n.data AS data, n.`__created_at` AS created_at "
            "ORDER BY n.`__created_at` DESC "
            "LIMIT $limit"
        )

        async with _pool.get_read_session() as session:
            result = await session.run(data_query, **params)
            rows = await result.data()
            await result.consume()
        events = _decode_rows(rows, fallback_created_at=now_ms)
        async with _pool.get_read_session() as session:
            count_query = (
                f"MATCH (n:`{self._label}`) {where_scalar} RETURN count(n) AS c"
            )
            result = await session.run(count_query, **params)
            count_record = await result.single()
            await result.consume()
        scalar_total = int(count_record["c"]) if count_record else 0

        if not need_json_fallback:
            return [event for _, event in events], scalar_total, now_ms

        # Legacy fallback is only for rows that were not migrated yet. The
        # authoritative path after initialize() is the scalar query above.
        legacy_clauses: list[str] = list(base_clauses)
        legacy_clauses.append(f"({_LEGACY_SCALARS_CLAUSE})")
        where_legacy = _to_where(legacy_clauses)

        legacy_query = (
            f"MATCH (n:`{self._label}`) {where_legacy} "
            "RETURN n.data AS data, n.`__created_at` AS created_at "
            "ORDER BY n.`__created_at` DESC "
            "LIMIT $compat_limit"
        )

        async with _pool.get_read_session() as session:
            legacy_result = await session.run(legacy_query, **params)
            legacy_rows = await legacy_result.data()
            await legacy_result.consume()

        legacy_events: list[tuple[int, dict[str, Any]]] = []
        for created_at, event in _decode_rows(legacy_rows, fallback_created_at=now_ms):
            if not _matches(
                event,
                kind=kind,
                sev=sev,
                actor=actor,
                q=q,
                range=range,
                now_ms=now_ms,
                resource_id=resource_id,
            ):
                continue
            legacy_events.append((created_at, event))

        merged = sorted(events + legacy_events, key=lambda item: item[0], reverse=True)
        return (
            [event for _, event in merged[:limit]],
            scalar_total + len(legacy_events),
            now_ms,
        )

    async def append(self, event: dict[str, Any]) -> dict[str, Any]:
        if "id" not in event:
            raise ValueError("append requires event['id']")
        payload = json.dumps(event, sort_keys=True)
        scalars = _event_scalars(event)
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    MERGE (n:`{self._label}` {{id: $id}})
                    ON CREATE SET n.`__created_at` = timestamp()
                    SET n.data = $data,
                        n.kind = $kind,
                        n.sev = $sev,
                        n.actor_user = $actor_user,
                        n.target_id = $target_id,
                        n.meta_doc_id = $meta_doc_id,
                        n.meta_doc_ids = $meta_doc_ids,
                        n.ts_ms = $ts_ms,
                        n.target_label = $target_label,
                        n.summary = $summary,
                        n.`__scalars_version` = $scalars_version,
                        n.`__updated_at` = timestamp()
                    """,
                    id=str(event["id"]),
                    data=payload,
                    scalars_version=SCALARS_VERSION,
                    **scalars,
                )
                await result.consume()
        return copy.deepcopy(event)


async def make_memgraph_activity_store(
    workspace: str = "default",
) -> MemgraphActivityStore:
    store = MemgraphActivityStore(workspace=workspace)
    await store.initialize()
    await store.bootstrap_if_empty()
    return store
