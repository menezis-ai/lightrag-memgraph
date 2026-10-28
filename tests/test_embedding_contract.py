"""Exercise the real LightRAG wrapper, including provider-specific arguments."""

import inspect
from unittest.mock import AsyncMock

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


def store_for(embedding):
    return MemgraphVectorDBStorage("contract", {}, embedding, workspace="embedding")


async def test_embedding_calls_preserve_dimensions_and_token_limit():
    calls = []

    async def provider(texts, embedding_dim=None, max_token_size=None):
        calls.append((list(texts), embedding_dim, max_token_size))
        return np.ones((len(texts), 4))

    if "send_dimensions" not in inspect.signature(EmbeddingFunc).parameters:
        pytest.skip("This upstream wrapper predates dimension injection")
    store = store_for(
        EmbeddingFunc(
            embedding_dim=4,
            func=provider,
            max_token_size=17,
            send_dimensions=True,
        )
    )
    assert await store._embed_query("question") == [1.0] * 4
    assert await store._compute_missing_embeddings(
        {"a": {"content": "first"}, "b": {"content": "second"}}
    ) == {"a": [1.0] * 4, "b": [1.0] * 4}
    assert calls == [(["question"], 4, 17), (["first", "second"], 4, 17)]


async def test_query_and_document_embeddings_use_distinct_contexts():
    if "supports_asymmetric" not in inspect.signature(EmbeddingFunc).parameters:
        pytest.skip("This upstream wrapper predates asymmetric embeddings")
    contexts = []

    async def provider(texts, context=None):
        contexts.append(context)
        return np.ones((len(texts), 4))

    store = store_for(
        EmbeddingFunc(embedding_dim=4, func=provider, supports_asymmetric=True)
    )
    await store._embed_query("question")
    await store._compute_missing_embeddings({"a": {"content": "document"}})
    assert contexts == ["query", "document"]


@pytest.mark.parametrize(
    "result",
    [np.ones((1, 3)), np.ones((2, 4)), np.ones((4, 1)), np.full((1, 4), np.nan)],
)
async def test_invalid_provider_output_fails_before_writing(result, monkeypatch):
    async def provider(texts):
        return result

    store = store_for(EmbeddingFunc(embedding_dim=4, func=provider))
    write_session = AsyncMock(side_effect=AssertionError("must not write"))
    monkeypatch.setattr(
        "lightrag_memgraph.vector_impl._pool.get_session", write_session
    )
    with pytest.raises(ValueError):
        await store.upsert({"a": {"content": "document"}})
    write_session.assert_not_called()


async def test_supplied_query_embedding_does_not_call_provider():
    provider = AsyncMock(side_effect=AssertionError("must not embed"))
    store = store_for(EmbeddingFunc(embedding_dim=4, func=provider))
    store._query_vector_once = AsyncMock(return_value=[])
    assert await store.query("question", 1, query_embedding=[1, 0, 0, 0]) == []
    provider.assert_not_called()
