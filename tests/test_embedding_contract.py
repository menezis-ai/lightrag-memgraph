"""Exercise EmbeddingFunc with LightRAG's real priority/concurrency scheduler."""

import asyncio
from contextlib import asynccontextmanager
import inspect
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc, priority_limit_async_func_call

from lightrag_memgraph import _pool
from lightrag_memgraph.kv_impl import MemgraphKVStorage
from lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


def store_for(embedding, config=None):
    return MemgraphVectorDBStorage(
        "contract", config or {}, embedding, workspace="embedding"
    )


@pytest.fixture
async def embedding_factory():
    queues = []

    def create(provider, *, max_async=2, **options):
        queued = priority_limit_async_func_call(max_async)(provider)
        queues.append(queued)
        return EmbeddingFunc(embedding_dim=4, func=queued, **options)

    yield create
    for queued in queues:
        await queued.shutdown()


async def test_embedding_calls_preserve_dimensions_and_token_limit(embedding_factory):
    calls = []

    async def provider(texts, embedding_dim=None, max_token_size=None):
        calls.append((list(texts), embedding_dim, max_token_size))
        return np.ones((len(texts), 4))

    if "send_dimensions" not in inspect.signature(EmbeddingFunc).parameters:
        pytest.skip("This upstream wrapper predates dimension injection")
    store = store_for(
        embedding_factory(
            provider,
            max_token_size=17,
            send_dimensions=True,
        )
    )
    assert await store._embed_query("question") == [1.0] * 4
    assert await store._compute_missing_embeddings(
        {"a": {"content": "first"}, "b": {"content": "second"}}
    ) == {"a": [1.0] * 4, "b": [1.0] * 4}
    assert calls == [(["question"], 4, 17), (["first", "second"], 4, 17)]


async def test_query_and_document_embeddings_use_distinct_contexts(embedding_factory):
    if "supports_asymmetric" not in inspect.signature(EmbeddingFunc).parameters:
        pytest.skip("This upstream wrapper predates asymmetric embeddings")
    contexts = []

    async def provider(texts, context=None):
        contexts.append(context)
        return np.ones((len(texts), 4))

    store = store_for(embedding_factory(provider, supports_asymmetric=True))
    await store._embed_query("question")
    await store._compute_missing_embeddings({"a": {"content": "document"}})
    assert contexts == ["query", "document"]


@pytest.mark.parametrize(
    "result,message",
    [(np.ones((4, 1)), "returned 4 vectors"), (np.full((1, 4), np.nan), "got nan")],
)
async def test_invalid_provider_output_fails_before_writing(
    result, message, monkeypatch
):
    async def provider(texts):
        return result

    store = store_for(EmbeddingFunc(embedding_dim=4, func=provider))
    write_slot = MagicMock(side_effect=AssertionError("must not acquire a write slot"))
    monkeypatch.setattr(_pool, "acquire_write_slot", write_slot)
    with pytest.raises(ValueError, match=message):
        await store.upsert({"a": {"content": "document"}})
    write_slot.assert_not_called()


async def test_supplied_query_embedding_does_not_call_provider():
    provider = AsyncMock(side_effect=AssertionError("must not embed"))
    store = store_for(EmbeddingFunc(embedding_dim=4, func=provider))
    store._query_vector_once = AsyncMock(return_value=[])
    assert await store.query("question", 1, query_embedding=[1, 0, 0, 0]) == []
    provider.assert_not_called()


async def test_query_overtakes_queued_ingestion(embedding_factory, monkeypatch):
    running = asyncio.Event()
    release = asyncio.Event()
    order = []
    enqueued = [asyncio.Event() for _ in range(3)]
    original_put = asyncio.PriorityQueue.put
    count = 0

    async def observe_put(queue, item):
        nonlocal count
        await original_put(queue, item)
        enqueued[count].set()
        count += 1

    monkeypatch.setattr(asyncio.PriorityQueue, "put", observe_put)

    # A strict provider signature also proves the scheduler consumes _priority.
    async def provider(texts):
        order.append(texts[0])
        if texts == ["running"]:
            running.set()
            await release.wait()
        return np.ones((len(texts), 4))

    embedding = embedding_factory(provider, max_async=1)
    store = store_for(embedding)
    tasks = [
        asyncio.create_task(
            store._compute_missing_embeddings({"a": {"content": "running"}})
        )
    ]

    try:
        await asyncio.wait_for(running.wait(), 2)
        tasks.append(
            asyncio.create_task(
                store._compute_missing_embeddings({"b": {"content": "document"}})
            )
        )
        await asyncio.wait_for(enqueued[1].wait(), 2)
        tasks.append(asyncio.create_task(store._embed_query("question")))
        await asyncio.wait_for(enqueued[2].wait(), 2)
        release.set()
        await asyncio.gather(*tasks)
        assert order == ["running", "question", "document"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "configured_size,expected_sizes", [(None, [10, 10, 5]), (2, [2] * 12 + [1])]
)
async def test_document_batches_preserve_ids_and_bound_concurrency(
    embedding_factory, configured_size, expected_sizes
):
    calls = []
    active = high_water = 0

    async def provider(texts):
        nonlocal active, high_water
        calls.append(len(texts))
        active += 1
        high_water = max(high_water, active)
        try:
            await asyncio.sleep(0)
            return np.asarray([[float(text)] * 4 for text in texts])
        finally:
            active -= 1

    config = {"embedding_func_max_async": 2}
    if configured_size is not None:
        config["embedding_batch_num"] = configured_size
    store = store_for(embedding_factory(provider, max_async=8), config)
    data = {f"id-{i}": {"content": str(i)} for i in range(25)}
    data["precomputed"] = {"content": "must not embed", "embedding": [1, 0, 0, 0]}
    data["metadata"] = {"entity_name": "no content"}
    result = await store._compute_missing_embeddings(data)
    assert calls == expected_sizes
    assert result == {f"id-{i}": [float(i)] * 4 for i in range(25)}
    assert high_water == 2
    calls.clear()
    assert await store._compute_missing_embeddings({}) == {}
    assert (
        await store._compute_missing_embeddings({"precomputed": data["precomputed"]})
        == {}
    )
    assert not calls


@pytest.mark.parametrize("option", ["embedding_batch_num", "embedding_func_max_async"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_invalid_embedding_scheduling_configuration(option, value):
    with pytest.raises(ValueError, match=option + " must be a positive integer"):
        store_for(EmbeddingFunc(embedding_dim=4, func=AsyncMock()), {option: value})


@pytest.mark.parametrize(
    "result,message",
    [
        (np.ones((1, 3)), "vector 0 has dimension 3, expected 4"),
        (np.ones((2, 4)), "returned 2 vectors for 1 texts"),
        (np.ones((4, 1)), "returned 4 vectors for 1 texts"),
        (None, "must return a matrix"),
        (42, "must return a matrix"),
        ([42], "must return a matrix"),
        (np.full((1, 4), np.nan), "coordinate 0.*got nan"),
        (np.full((1, 4), np.inf), "coordinate 0.*got inf"),
        (np.full((1, 4), -np.inf), "coordinate 0.*got -inf"),
        ([[True, 0, 0, 0]], "coordinate 0.*got True"),
        ([["bad", 0, 0, 0]], "coordinate 0.*got 'bad'"),
    ],
)
def test_adapter_validation_independent_of_upstream(result, message):
    # Current EmbeddingFunc rejects some shapes before returning, while 1.4.9
    # performs no validation: exercise our boundary independently on both.
    with pytest.raises(ValueError, match=message):
        MemgraphVectorDBStorage._validate_embeddings(result, count=1, dimension=4)


async def test_slow_embedding_does_not_block_kv_writes(embedding_factory, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    writes = []
    monkeypatch.setenv("MEMGRAPH_WRITE_CONCURRENCY", "1")
    monkeypatch.setattr(_pool, "_write_semaphore", None)
    monkeypatch.setattr(_pool, "_semaphore_loop_id", None)

    async def provider(texts):
        started.set()
        await release.wait()
        return np.ones((len(texts), 4))

    @asynccontextmanager
    async def session():
        async def run(query, **kwargs):
            writes.append(query)
            return MagicMock(consume=AsyncMock())

        yield MagicMock(run=AsyncMock(side_effect=run))

    monkeypatch.setattr(_pool, "get_session", session)
    store = store_for(embedding_factory(provider))
    task = asyncio.create_task(store.upsert({"a": {"content": "document"}}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        kv = MemgraphKVStorage("status", {}, None, workspace="other")
        await asyncio.wait_for(kv.upsert({"id": {"status": "processed"}}), 1)
        assert len(writes) == 1 and "KV_other_status" in writes[0]
        release.set()
        await task
        assert len(writes) == 2 and "Vec_embedding_contract" in writes[1]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_batch_failure_or_cancellation_never_writes(
    embedding_factory, monkeypatch, cancel
):
    started, release = asyncio.Event(), asyncio.Event()

    async def provider(texts):
        if texts == ["bad"]:
            await started.wait()
            raise ValueError("provider refused batch")
        started.set()
        await release.wait()
        return np.ones((len(texts), 4))

    store = store_for(embedding_factory(provider), {"embedding_batch_num": 1})
    write_slot = MagicMock(side_effect=AssertionError("must not write"))
    monkeypatch.setattr(_pool, "acquire_write_slot", write_slot)
    data = {"good": {"content": "wait"}}
    if not cancel:
        data["bad"] = {"content": "bad"}
    task = asyncio.create_task(store.upsert(data))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ValueError, match="provider refused batch"):
                await asyncio.wait_for(task, 2)
        write_slot.assert_not_called()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
