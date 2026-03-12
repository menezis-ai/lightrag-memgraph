"""
End-to-end pipeline test: ingest → query (all modes) → delete → verify cleanup.

Exercises the full LightRAG lifecycle with Memgraph backends:
  1. register() + LightRAG instantiation with all 4 Memgraph storages
  2. ainsert() — document ingestion (mock LLM for entity extraction)
  3. aquery() — all query modes: local, global, hybrid, mix
  4. adelete_by_doc_id() — document deletion
  5. Post-deletion verification: KV, Vector, Graph, DocStatus all cleaned

Requires: running Memgraph >= 3.2 + MAGE (set MEMGRAPH_URI).
"""

import hashlib
import os
import shutil
import tempfile

import numpy as np
import pytest
from lightrag import LightRAG
from lightrag.base import DocStatus, QueryParam

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool

twindb_lightrag_memgraph.register()

# ── Constants ────────────────────────────────────────────────────────

EMBEDDING_DIM = 384
E2E_WORKSPACE = "e2e_test"

SAMPLE_DOC = (
    "Paris is the capital and most populous city of France. "
    "The Eiffel Tower, built in 1889, is an iconic landmark located in Paris. "
    "France is a country in Western Europe known for its wine and cuisine. "
    "The Seine River flows through Paris and is a major waterway in France. "
    "Napoleon Bonaparte was a French military leader who rose to prominence "
    "during the French Revolution and led several successful campaigns. "
    "The Louvre Museum in Paris houses the Mona Lisa and thousands of other artworks."
)

# Entities and relations the mock LLM will "extract"
MOCK_ENTITIES = [
    ("Paris", "location", "Paris is the capital and most populous city of France."),
    ("France", "country", "France is a country in Western Europe known for wine and cuisine."),
    ("Eiffel Tower", "landmark", "The Eiffel Tower is an iconic landmark built in 1889 in Paris."),
    ("Seine River", "location", "The Seine River flows through Paris."),
    ("Napoleon Bonaparte", "person", "Napoleon Bonaparte was a French military leader."),
    ("Louvre Museum", "landmark", "The Louvre Museum in Paris houses the Mona Lisa."),
]

MOCK_RELATIONS = [
    ("Paris", "France", "capital city, geography", "Paris is the capital of France."),
    ("Eiffel Tower", "Paris", "located in, landmark", "The Eiffel Tower is located in Paris."),
    ("Seine River", "Paris", "flows through, geography", "The Seine River flows through Paris."),
    ("Napoleon Bonaparte", "France", "military leader, history", "Napoleon was a leader of France."),
    ("Louvre Museum", "Paris", "located in, culture", "The Louvre Museum is located in Paris."),
]


# ── Mock LLM & Embedding ────────────────────────────────────────────


def _build_extraction_response() -> str:
    """Build a valid LightRAG entity extraction response."""
    lines = []
    for name, etype, desc in MOCK_ENTITIES:
        lines.append(f"entity<|#|>{name}<|#|>{etype}<|#|>{desc}")
    for src, tgt, keywords, desc in MOCK_RELATIONS:
        lines.append(f"relation<|#|>{src}<|#|>{tgt}<|#|>{keywords}<|#|>{desc}")
    lines.append("<|COMPLETE|>")
    return "\n".join(lines)


_call_count = 0


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Mock LLM that returns entity extraction on first calls, then query answers."""
    global _call_count
    _call_count += 1

    prompt_lower = prompt.lower() if isinstance(prompt, str) else ""

    # Entity extraction prompt detection
    if "entity_types" in prompt_lower or "extract" in prompt_lower:
        return _build_extraction_response()

    # Summarization / merge prompts
    if "summary" in prompt_lower or "merge" in prompt_lower:
        return "A comprehensive summary of entities and relationships about Paris and France."

    # Query response — return something plausible with keywords from the query
    return (
        "Paris is the capital of France. The Eiffel Tower is an iconic landmark "
        "in Paris built in 1889. The Seine River flows through the city. "
        "The Louvre Museum houses the Mona Lisa."
    )


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    """Deterministic embedding based on text content hash."""
    results = []
    for text in texts:
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
            :EMBEDDING_DIM
        ].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        results.append(vec)
    return np.array(results)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def working_dir():
    """Temporary working directory for LightRAG file-based caches."""
    d = tempfile.mkdtemp(prefix="lightrag_e2e_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
async def rag(working_dir):
    """Fully initialized LightRAG instance with Memgraph backends."""
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
    from lightrag.utils import EmbeddingFunc

    # Reset LightRAG global locks (bound to previous event loop between tests)
    finalize_share_data()
    initialize_share_data()

    global _call_count
    _call_count = 0

    os.environ["MEMGRAPH_WORKSPACE"] = E2E_WORKSPACE

    # Clean stale data from previous test runs
    await _cleanup_e2e_data()

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=_mock_embedding,
    )

    instance = LightRAG(
        working_dir=working_dir,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=E2E_WORKSPACE,
        embedding_func=embedding_func,
        llm_model_func=_mock_llm,
        enable_llm_cache=False,
        chunk_token_size=200,
        chunk_overlap_token_size=50,
    )

    await instance.initialize_storages()
    yield instance

    # Cleanup: drop all e2e data from Memgraph
    await _cleanup_e2e_data()
    await instance.finalize_storages()


async def _cleanup_e2e_data():
    """Drop all nodes with e2e_test workspace labels."""
    try:
        async with _pool.get_session() as session:
            # Find and delete all nodes with labels containing e2e_test
            for prefix in ("KV_", "Vec_", "DocStatus_"):
                label = f"{prefix}{E2E_WORKSPACE}"
                try:
                    result = await session.run(
                        f"MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH '{label}') "
                        f"DETACH DELETE n"
                    )
                    await result.consume()
                except Exception:
                    pass
            # Also clean graph nodes
            result = await session.run(
                f"MATCH (n:`{E2E_WORKSPACE}`) DETACH DELETE n"
            )
            await result.consume()
    except Exception:
        pass


# ── Helpers: count nodes in Memgraph ─────────────────────────────────


async def _count_nodes(label_pattern: str) -> int:
    """Count nodes whose label starts with the given pattern."""
    async with _pool.get_read_session() as session:
        result = await session.run(
            "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $pat) "
            "RETURN count(n) AS cnt",
            pat=label_pattern,
        )
        record = await result.single()
        await result.consume()
        return record["cnt"] if record else 0


async def _count_graph_nodes(workspace: str) -> int:
    """Count nodes in the graph storage (entity nodes)."""
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`{workspace}`) RETURN count(n) AS cnt"
        )
        record = await result.single()
        await result.consume()
        return record["cnt"] if record else 0


async def _count_graph_edges(workspace: str) -> int:
    """Count edges in the graph storage."""
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (:`{workspace}`)-[r]->(:`{workspace}`) RETURN count(r) AS cnt"
        )
        record = await result.single()
        await result.consume()
        return record["cnt"] if record else 0


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestE2EPipeline:
    """Full lifecycle: ingest → query → delete → verify."""

    async def test_full_pipeline(self, rag):
        # ════════════════════════════════════════════════════
        # PHASE 1: Ingest document
        # ════════════════════════════════════════════════════
        track_id = await rag.ainsert(SAMPLE_DOC)
        assert track_id is not None

        # Verify data was created in Memgraph
        kv_count = await _count_nodes(f"KV_{E2E_WORKSPACE}")
        vec_count = await _count_nodes(f"Vec_{E2E_WORKSPACE}")
        graph_nodes = await _count_graph_nodes(E2E_WORKSPACE)
        graph_edges = await _count_graph_edges(E2E_WORKSPACE)
        docstatus_count = await _count_nodes(f"DocStatus_{E2E_WORKSPACE}")

        assert kv_count > 0, "KV storage should have entries after ingestion"
        assert vec_count > 0, "Vector storage should have entries after ingestion"
        assert graph_nodes > 0, "Graph should have entity nodes after ingestion"
        assert graph_edges > 0, "Graph should have edges after ingestion"
        assert docstatus_count > 0, "DocStatus should track the document"

        # Store counts for later comparison
        pre_delete = {
            "kv": kv_count,
            "vec": vec_count,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "docstatus": docstatus_count,
        }

        # ════════════════════════════════════════════════════
        # PHASE 2: Query all modes
        # ════════════════════════════════════════════════════
        modes = ["local", "global", "hybrid", "mix"]
        for mode in modes:
            result = await rag.aquery(
                "What is Paris and what landmarks are there?",
                param=QueryParam(mode=mode, top_k=5),
            )
            assert result is not None, f"Query mode '{mode}' returned None"
            assert len(result) > 0, f"Query mode '{mode}' returned empty result"

        # Also test context-only retrieval (no LLM synthesis)
        context = await rag.aquery(
            "Tell me about Paris",
            param=QueryParam(mode="local", only_need_context=True, top_k=5),
        )
        assert context is not None, "Context-only query should return data"

        # ════════════════════════════════════════════════════
        # PHASE 3: Delete document
        # ════════════════════════════════════════════════════
        # Find the doc_id (LightRAG generates it from content hash)
        processed = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        assert len(processed) > 0, "Should have at least one processed document"

        for doc_id in processed:
            deletion_result = await rag.adelete_by_doc_id(doc_id)
            assert deletion_result.status == "success", (
                f"Deletion failed for {doc_id}: {deletion_result.message}"
            )

        # ════════════════════════════════════════════════════
        # PHASE 4: Verify cleanup
        # ════════════════════════════════════════════════════
        post_graph_nodes = await _count_graph_nodes(E2E_WORKSPACE)
        post_graph_edges = await _count_graph_edges(E2E_WORKSPACE)
        post_docstatus = await _count_nodes(f"DocStatus_{E2E_WORKSPACE}")

        # Graph should be cleaned (entities from this doc only)
        assert post_graph_nodes < pre_delete["graph_nodes"], (
            f"Graph nodes not cleaned: {post_graph_nodes} >= {pre_delete['graph_nodes']}"
        )
        assert post_graph_edges < pre_delete["graph_edges"], (
            f"Graph edges not cleaned: {post_graph_edges} >= {pre_delete['graph_edges']}"
        )

        # DocStatus should be empty (single doc was deleted)
        assert post_docstatus == 0, (
            f"DocStatus should be empty after deletion, got {post_docstatus}"
        )

        # Verify doc is gone from doc_status API
        remaining_counts = await rag.doc_status.get_status_counts()
        total_remaining = sum(remaining_counts.values())
        assert total_remaining == 0, (
            f"No documents should remain, got {remaining_counts}"
        )

    async def test_query_after_delete_returns_no_crash(self, rag):
        """Query on an empty store should not crash, just return a response."""
        # Insert and delete
        await rag.ainsert("Short test document about Berlin.")

        processed = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        for doc_id in processed:
            await rag.adelete_by_doc_id(doc_id)

        # Query after deletion should not raise
        result = await rag.aquery(
            "Tell me about Berlin",
            param=QueryParam(mode="mix", top_k=5),
        )
        assert result is not None

    async def test_multiple_documents_partial_delete(self, rag):
        """Insert two docs, delete one, verify the other remains."""
        doc_a = "Tokyo is the capital of Japan. Mount Fuji is near Tokyo."
        doc_b = "London is the capital of England. The Thames flows through London."

        await rag.ainsert(doc_a)
        await rag.ainsert(doc_b)

        # Should have 2 documents
        all_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        assert len(all_docs) >= 2, f"Expected >= 2 docs, got {len(all_docs)}"

        # Delete only the first doc
        first_doc_id = list(all_docs.keys())[0]
        deletion_result = await rag.adelete_by_doc_id(first_doc_id)
        assert deletion_result.status == "success"

        # Verify one doc remains
        remaining = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        assert len(remaining) == len(all_docs) - 1

        # Graph should still have some data
        graph_nodes = await _count_graph_nodes(E2E_WORKSPACE)
        assert graph_nodes > 0, "Graph should still have nodes from remaining document"

        # Query should still work
        result = await rag.aquery(
            "What cities are mentioned?",
            param=QueryParam(mode="mix", top_k=5),
        )
        assert result is not None
