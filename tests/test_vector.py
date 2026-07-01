"""
Integration tests for MemgraphVectorDBStorage.

Requires a running Memgraph >= 3.2 with MAGE (set MEMGRAPH_URI).
"""

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from twindb_lightrag_memgraph import register
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

register()

EMBEDDING_DIM = 4


async def _mock_embed(texts: list[str]) -> np.ndarray:
    """Deterministic mock embedding for testing."""
    rng = np.random.default_rng(seed=42)
    return rng.random((len(texts), EMBEDDING_DIM)).astype(np.float32)


@pytest.fixture
def embedding_func():
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=_mock_embed,
    )


@pytest.fixture
async def vec_store(embedding_func):
    store = MemgraphVectorDBStorage(
        namespace="test_vec",
        global_config={
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
            },
        },
        embedding_func=embedding_func,
        meta_fields={"entity_name", "content"},
    )
    await store.initialize()
    yield store
    await store.drop()


class _RecordingEmbed:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def func(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0, 0.0, 0.0] for text in texts]


def _unit_store(embed: _RecordingEmbed, *, batch_num=2):
    store = MemgraphVectorDBStorage.__new__(MemgraphVectorDBStorage)
    store.global_config = {"embedding_batch_num": batch_num}
    store.embedding_func = embed
    return store


class TestComputeMissingEmbeddings:
    async def test_batches_missing_embeddings_from_global_config(self):
        embed = _RecordingEmbed()
        store = _unit_store(embed, batch_num=2)

        result = await store._compute_missing_embeddings(
            {
                "a": {"content": "one"},
                "b": {"content": "two"},
                "c": {"content": "three"},
                "d": {"content": "four"},
                "e": {"content": "five"},
            }
        )

        assert embed.calls == [["one", "two"], ["three", "four"], ["five"]]
        assert set(result) == {"a", "b", "c", "d", "e"}

    async def test_skips_items_with_existing_embedding_or_no_content(self):
        embed = _RecordingEmbed()
        store = _unit_store(embed, batch_num=10)

        result = await store._compute_missing_embeddings(
            {
                "needs": {"content": "one"},
                "has": {"content": "two", "embedding": [1.0, 0.0, 0.0, 0.0]},
                "metadata-only": {"entity_name": "x"},
            }
        )

        assert embed.calls == [["one"]]
        assert set(result) == {"needs"}

    async def test_invalid_batch_size_falls_back_to_default(self):
        embed = _RecordingEmbed()
        store = _unit_store(embed, batch_num="nope")

        assert store._embedding_batch_size() == 32

    async def test_mismatched_embedding_count_raises(self):
        class BadEmbed:
            async def func(self, texts):
                return []

        store = _unit_store(BadEmbed(), batch_num=2)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="returned 0 rows for 1 input"):
            await store._compute_missing_embeddings({"a": {"content": "one"}})


@pytest.mark.integration
class TestMemgraphVectorDBStorage:
    async def test_upsert_and_get_by_id(self, vec_store):
        data = {
            "v1": {
                "content": "hello world",
                "entity_name": "greeting",
                "embedding": [0.1, 0.2, 0.3, 0.4],
            }
        }
        await vec_store.upsert(data)
        result = await vec_store.get_by_id("v1")
        assert result is not None
        assert result["entity_name"] == "greeting"

    async def test_metadata_only_upsert_preserves_existing_embedding(self, vec_store):
        embedding = [0.1, 0.2, 0.3, 0.4]
        await vec_store.upsert(
            {
                "v1": {
                    "content": "hello world",
                    "entity_name": "greeting",
                    "embedding": embedding,
                }
            }
        )

        await vec_store.upsert({"v1": {"entity_name": "updated"}})

        result = await vec_store.get_by_id("v1")
        assert result is not None
        assert result["entity_name"] == "updated"
        assert result["embedding"] == pytest.approx(embedding)

    async def test_get_by_ids(self, vec_store):
        await vec_store.upsert(
            {
                "v1": {"content": "a", "embedding": [0.1, 0.2, 0.3, 0.4]},
                "v2": {"content": "b", "embedding": [0.5, 0.6, 0.7, 0.8]},
            }
        )
        results = await vec_store.get_by_ids(["v1", "v2"])
        assert len(results) == 2

    async def test_query(self, vec_store):
        await vec_store.upsert(
            {
                "v1": {
                    "content": "cat on mat",
                    "entity_name": "cat",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                },
                "v2": {
                    "content": "dog in park",
                    "entity_name": "dog",
                    "embedding": [0.0, 1.0, 0.0, 0.0],
                },
            }
        )
        results = await vec_store.query(
            query="",
            top_k=2,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
        )
        assert len(results) > 0
        # Most similar should be v1
        assert results[0]["id"] == "v1"
        assert "similarity" in results[0]

    async def test_delete_entity(self, vec_store):
        await vec_store.upsert(
            {
                "v1": {
                    "entity_name": "to_remove",
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                },
            }
        )
        await vec_store.delete_entity("to_remove")
        result = await vec_store.get_by_id("v1")
        assert result is None

    async def test_delete_entity_relation(self, vec_store):
        await vec_store.upsert(
            {
                "r1": {
                    "src_id": "A",
                    "tgt_id": "B",
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                },
                "r2": {
                    "src_id": "C",
                    "tgt_id": "A",
                    "embedding": [0.5, 0.6, 0.7, 0.8],
                },
                "r3": {
                    "src_id": "C",
                    "tgt_id": "D",
                    "embedding": [0.9, 0.1, 0.2, 0.3],
                },
            }
        )
        await vec_store.delete_entity_relation("A")
        # r1 (src=A) and r2 (tgt=A) should be gone
        assert await vec_store.get_by_id("r1") is None
        assert await vec_store.get_by_id("r2") is None
        # r3 should remain
        assert await vec_store.get_by_id("r3") is not None

    async def test_delete_by_ids(self, vec_store):
        await vec_store.upsert(
            {
                "v1": {"embedding": [0.1, 0.2, 0.3, 0.4]},
                "v2": {"embedding": [0.5, 0.6, 0.7, 0.8]},
            }
        )
        await vec_store.delete(["v1"])
        assert await vec_store.get_by_id("v1") is None
        assert await vec_store.get_by_id("v2") is not None

    async def test_get_vectors_by_ids(self, vec_store):
        emb = [0.1, 0.2, 0.3, 0.4]
        await vec_store.upsert({"v1": {"embedding": emb}})
        vectors = await vec_store.get_vectors_by_ids(["v1", "missing"])
        assert "v1" in vectors
        assert len(vectors["v1"]) == 4
        assert "missing" not in vectors

    async def test_drop(self, vec_store):
        await vec_store.upsert({"v1": {"embedding": [0.1, 0.2, 0.3, 0.4]}})
        result = await vec_store.drop()
        assert result["status"] == "success"
        assert await vec_store.get_by_id("v1") is None
