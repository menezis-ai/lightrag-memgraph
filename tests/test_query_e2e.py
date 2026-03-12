"""End-to-end integration tests for all 4 query modes.

Seeds both the knowledge graph (MemgraphStorage) and the vector DB
(MemgraphVectorDBStorage), then calls operate._perform_kg_search for
each mode. This validates the FULL retrieval pipeline, not just
individual batch methods.

Requires running Memgraph with MAGE (for vector_search).
"""

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MEMGRAPH_URI"),
    reason="MEMGRAPH_URI not set",
)

import twindb_lightrag_memgraph
import twindb_lightrag_memgraph._pool as _pool

twindb_lightrag_memgraph.register()

from lightrag import operate
from lightrag.base import QueryParam
from lightrag.kg.memgraph_impl import MemgraphStorage
from lightrag.kg.shared_storage import initialize_share_data

from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

initialize_share_data(workers=1)

EMBEDDING_DIM = 64
WORKSPACE = "test_e2e"


class _FakeEmbeddingFunc:
    """Fake embedding function that returns deterministic vectors."""

    embedding_dim = EMBEDDING_DIM

    @staticmethod
    async def func(texts):
        """Return pseudo-random embeddings based on text hash."""
        import hashlib

        results = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[: EMBEDDING_DIM]]
            # Pad/truncate to EMBEDDING_DIM
            vec = (vec * ((EMBEDDING_DIM // len(vec)) + 1))[:EMBEDDING_DIM]
            # Normalize
            norm = sum(x * x for x in vec) ** 0.5
            vec = [x / norm for x in vec]
            results.append(vec)
        return results


embedding_func = _FakeEmbeddingFunc()


@pytest.fixture
async def seeded_stores():
    """Set up graph + vector stores with test data, shared across all tests."""

    # -- Graph storage --
    graph = MemgraphStorage(
        namespace="test_e2e",
        global_config={"workspace": WORKSPACE},
        embedding_func=None,
    )
    await graph.initialize()

    ws = graph._get_workspace_label()

    # Clean old data
    async with graph._driver.session() as session:
        await session.run(f"MATCH (n:`{ws}`) DETACH DELETE n")

    # Seed nodes
    for name, etype, desc in [
        ("PARIS", "City", "Capital of France, major European city"),
        ("FRANCE", "Country", "Western European country"),
        ("EIFFEL_TOWER", "Landmark", "Iconic iron lattice tower in Paris"),
        ("NAPOLEON", "Person", "French emperor and military leader"),
    ]:
        await graph.upsert_node(name, {
            "entity_id": name,
            "entity_type": etype,
            "description": desc,
            "source_id": "chunk1",
        })

    # Seed edges
    for src, tgt, desc, kw in [
        ("PARIS", "FRANCE", "capital of", "capital, city, country"),
        ("EIFFEL_TOWER", "PARIS", "located in", "landmark, location, tower"),
        ("NAPOLEON", "FRANCE", "ruled", "emperor, ruler, history"),
    ]:
        await graph.upsert_edge(src, tgt, {
            "weight": "1.0",
            "description": desc,
            "keywords": kw,
            "source_id": "chunk1",
        })

    # -- Entities VDB --
    entities_vdb = MemgraphVectorDBStorage(
        namespace="entities",
        global_config={
            "workspace": WORKSPACE,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
            },
        },
        embedding_func=embedding_func,
        meta_fields={"entity_name", "source_id", "content", "file_path"},
    )
    await entities_vdb.initialize()
    # Clean old VDB data
    label_ent = entities_vdb._label()
    async with _pool.get_session() as session:
        await session.run(f"MATCH (n:`{label_ent}`) DETACH DELETE n")

    entity_vdb_data = {}
    for name, desc in [
        ("PARIS", "Capital of France, major European city"),
        ("FRANCE", "Western European country"),
        ("EIFFEL_TOWER", "Iconic iron lattice tower in Paris"),
        ("NAPOLEON", "French emperor and military leader"),
    ]:
        eid = f"ent-{name.lower()}"
        entity_vdb_data[eid] = {
            "entity_name": name,
            "content": f"{name}\n{desc}",
            "source_id": "chunk1",
            "file_path": "test.txt",
        }
    await entities_vdb.upsert(entity_vdb_data)

    # -- Relationships VDB --
    rels_vdb = MemgraphVectorDBStorage(
        namespace="relationships",
        global_config={
            "workspace": WORKSPACE,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
            },
        },
        embedding_func=embedding_func,
        meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
    )
    await rels_vdb.initialize()
    # Clean old VDB data
    label_rel = rels_vdb._label()
    async with _pool.get_session() as session:
        await session.run(f"MATCH (n:`{label_rel}`) DETACH DELETE n")

    rel_vdb_data = {}
    for src, tgt, desc, kw in [
        ("PARIS", "FRANCE", "capital of", "capital, city, country"),
        ("EIFFEL_TOWER", "PARIS", "located in", "landmark, location, tower"),
        ("NAPOLEON", "FRANCE", "ruled", "emperor, ruler, history"),
    ]:
        rid = f"rel-{src.lower()}-{tgt.lower()}"
        rel_vdb_data[rid] = {
            "src_id": src,
            "tgt_id": tgt,
            "content": f"{kw}\t{src}\n{tgt}\n{desc}",
            "source_id": "chunk1",
            "file_path": "test.txt",
        }
    await rels_vdb.upsert(rel_vdb_data)

    # Let Memgraph vector index catch up (async indexing)
    await asyncio.sleep(1)

    # -- Text chunks KV (mock-like) --
    text_chunks = MemgraphKVStorage(
        namespace="text_chunks",
        global_config={
            "workspace": WORKSPACE,
            "kg_chunk_pick_method": "ORDER",
        },
        embedding_func=None,
    )
    await text_chunks.initialize()
    text_chunks.embedding_func = None

    yield {
        "graph": graph,
        "entities_vdb": entities_vdb,
        "rels_vdb": rels_vdb,
        "text_chunks": text_chunks,
    }

    # Cleanup
    async with graph._driver.session() as session:
        await session.run(f"MATCH (n:`{ws}`) DETACH DELETE n")
    async with _pool.get_session() as session:
        await session.run(f"MATCH (n:`{label_ent}`) DETACH DELETE n")
        await session.run(f"MATCH (n:`{label_rel}`) DETACH DELETE n")


async def _do_search(mode, stores, query="Paris France capital"):
    """Run _perform_kg_search for the given mode."""
    return await operate._perform_kg_search(
        query=query,
        ll_keywords="Paris, France, capital" if mode in ("local", "hybrid", "mix") else "",
        hl_keywords="capital, country, European" if mode in ("global", "hybrid", "mix") else "",
        knowledge_graph_inst=stores["graph"],
        entities_vdb=stores["entities_vdb"],
        relationships_vdb=stores["rels_vdb"],
        text_chunks_db=stores["text_chunks"],
        query_param=QueryParam(mode=mode, top_k=10),
        chunks_vdb=None,
    )


class TestLocalMode:
    async def test_local_returns_entities(self, seeded_stores):
        result = await _do_search("local", seeded_stores)
        assert len(result["final_entities"]) > 0, (
            f"Local mode returned 0 entities. Keys: {result.keys()}"
        )

    async def test_local_returns_relations(self, seeded_stores):
        result = await _do_search("local", seeded_stores)
        assert len(result["final_relations"]) > 0, (
            f"Local mode returned 0 relations. Keys: {result.keys()}"
        )


class TestGlobalMode:
    async def test_global_returns_relations(self, seeded_stores):
        result = await _do_search("global", seeded_stores)
        assert len(result["final_relations"]) > 0, (
            f"Global mode returned 0 relations. VDB query may be empty."
        )

    async def test_global_returns_entities(self, seeded_stores):
        result = await _do_search("global", seeded_stores)
        assert len(result["final_entities"]) > 0, (
            f"Global mode returned 0 entities from relationship traversal."
        )


class TestHybridMode:
    async def test_hybrid_returns_entities(self, seeded_stores):
        result = await _do_search("hybrid", seeded_stores)
        assert len(result["final_entities"]) > 0

    async def test_hybrid_returns_relations(self, seeded_stores):
        result = await _do_search("hybrid", seeded_stores)
        assert len(result["final_relations"]) > 0


class TestMixMode:
    async def test_mix_returns_entities(self, seeded_stores):
        result = await _do_search("mix", seeded_stores)
        assert len(result["final_entities"]) > 0

    async def test_mix_returns_relations(self, seeded_stores):
        result = await _do_search("mix", seeded_stores)
        assert len(result["final_relations"]) > 0


class TestVDBDirect:
    """Verify VDB queries return expected meta_fields."""

    async def test_entities_vdb_returns_entity_name(self, seeded_stores):
        results = await seeded_stores["entities_vdb"].query(
            "Paris France", top_k=5
        )
        assert len(results) > 0, "Entities VDB returned 0 results"
        assert "entity_name" in results[0], (
            f"Missing 'entity_name' in VDB result: {results[0].keys()}"
        )

    async def test_relationships_vdb_returns_src_tgt(self, seeded_stores):
        results = await seeded_stores["rels_vdb"].query(
            "capital country European", top_k=5
        )
        assert len(results) > 0, "Relationships VDB returned 0 results"
        r = results[0]
        assert "src_id" in r, f"Missing 'src_id' in VDB result: {r.keys()}"
        assert "tgt_id" in r, f"Missing 'tgt_id' in VDB result: {r.keys()}"
