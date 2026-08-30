"""Cancellation contract for the bounded native-graph membership reads."""

from __future__ import annotations

import asyncio
import gc
import warnings
from functools import partial

import pytest

from twindb_lightrag_memgraph.server import graph_reader


@pytest.mark.parametrize("fanout", [1, 2])
async def test_cancellation_does_not_create_operations_waiting_for_gate(
    monkeypatch, fanout
):
    """Queued operations must not leave un-awaited coroutine objects behind."""
    monkeypatch.setenv("TWIN_GRAPH_MEMBERSHIP_FANOUT", str(fanout))
    admitted = asyncio.Event()
    never_release = asyncio.Event()
    created: list[int] = []

    async def _blocked_probe(_index: int) -> None:
        await never_release.wait()

    def _operation(index: int):
        created.append(index)
        if len(created) == fanout:
            admitted.set()
        return _blocked_probe(index)

    operations = [partial(_operation, index) for index in range(5)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        task = asyncio.create_task(graph_reader._gather_membership_reads(*operations))
        await asyncio.wait_for(admitted.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        del task
        gc.collect()
        await asyncio.sleep(0)

    assert created == list(range(fanout))
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "was never awaited" in str(warning.message)
    ]
