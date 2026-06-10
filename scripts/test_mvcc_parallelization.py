"""
MVCC parallelization diagnostic — Memgraph support call follow-up.

Tony's hypothesis: when ingesting big documents that share entities/subgraphs,
Memgraph MVCC locks on shared nodes serialize the writes, so two concurrent
ingestions don't actually run in parallel. The fix would be to ingest docs
with disjoint entity sets, which should parallelize.

This script measures the speedup ratio (sequential / parallel) for two scenarios:

  A) Shared entities — both docs mention "Paris"
  B) Disjoint entities — Paris vs Tokyo with no overlap

Interpretation:
  - speedup_A ≈ 1.0 and speedup_B > 1.0 → MVCC lock contention CONFIRMED
  - speedup_A > 1.0 and speedup_B > 1.0 → no contention, parallelization works
  - speedup_A ≈ 1.0 and speedup_B ≈ 1.0 → bottleneck is NOT MVCC
    (single-driver write throttle, application serialization, etc.)

Requires a running local Memgraph (docker compose up -d).
"""

import asyncio
import hashlib
import os
import shutil
import tempfile
import time

import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool

twindb_lightrag_memgraph.register()

EMBEDDING_DIM = 384

# ── Mock LLM with two distinct extraction templates ────────────────


SHARED_ENTITIES_DOCS = {
    "doc-shared-A": (
        "Paris is the capital of France. The Eiffel Tower is in Paris. "
        "Paris hosts the Louvre. Napoleon was a leader in Paris."
    ),
    "doc-shared-B": (
        "Paris has many monuments. The Seine flows through Paris. "
        "The Louvre is a famous museum in Paris. Napoleon visited the Louvre."
    ),
}

DISJOINT_ENTITIES_DOCS = {
    "doc-disjoint-A": (
        "Paris is the capital of France. The Eiffel Tower is in Paris. "
        "Paris hosts the Louvre. Napoleon was a leader in Paris."
    ),
    "doc-disjoint-B": (
        "Tokyo is the capital of Japan. Mount Fuji is near Tokyo. "
        "Tokyo hosts the Shibuya district. The Sumida river flows through Tokyo."
    ),
}


SHARED_RESPONSE = """entity<|#|>Paris<|#|>location<|#|>Paris is the capital of France.
entity<|#|>Eiffel Tower<|#|>landmark<|#|>The Eiffel Tower is in Paris.
entity<|#|>Louvre<|#|>landmark<|#|>The Louvre is a museum in Paris.
entity<|#|>Napoleon<|#|>person<|#|>Napoleon was a leader.
entity<|#|>Seine<|#|>location<|#|>The Seine flows through Paris.
entity<|#|>France<|#|>country<|#|>France is in Europe.
relation<|#|>Paris<|#|>France<|#|>capital, geography<|#|>Paris is the capital of France.
relation<|#|>Eiffel Tower<|#|>Paris<|#|>landmark, location<|#|>The Eiffel Tower is in Paris.
relation<|#|>Louvre<|#|>Paris<|#|>landmark, location<|#|>The Louvre is in Paris.
relation<|#|>Napoleon<|#|>Paris<|#|>person, history<|#|>Napoleon was in Paris.
relation<|#|>Seine<|#|>Paris<|#|>geography<|#|>The Seine flows through Paris.
<|COMPLETE|>"""


TOKYO_RESPONSE = """entity<|#|>Tokyo<|#|>location<|#|>Tokyo is the capital of Japan.
entity<|#|>Japan<|#|>country<|#|>Japan is in Asia.
entity<|#|>Mount Fuji<|#|>landmark<|#|>Mount Fuji is near Tokyo.
entity<|#|>Shibuya<|#|>location<|#|>Shibuya is a district in Tokyo.
entity<|#|>Sumida<|#|>location<|#|>The Sumida river is in Tokyo.
relation<|#|>Tokyo<|#|>Japan<|#|>capital, geography<|#|>Tokyo is the capital of Japan.
relation<|#|>Mount Fuji<|#|>Tokyo<|#|>landmark, location<|#|>Mount Fuji is near Tokyo.
relation<|#|>Shibuya<|#|>Tokyo<|#|>district, location<|#|>Shibuya is in Tokyo.
relation<|#|>Sumida<|#|>Tokyo<|#|>geography<|#|>The Sumida flows through Tokyo.
<|COMPLETE|>"""


async def _llm_router(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Route the mock LLM response based on document content."""
    text = prompt if isinstance(prompt, str) else ""
    if "Tokyo" in text or "Japan" in text or "Mount Fuji" in text:
        return TOKYO_RESPONSE
    if "Paris" in text or "France" in text or "Louvre" in text:
        return SHARED_RESPONSE
    if "summary" in text.lower() or "merge" in text.lower():
        return "merged summary"
    return "default response"


async def _embedding(texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
            :EMBEDDING_DIM
        ].astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        out.append(v)
    return np.array(out)


async def _build_rag(workspace: str, working_dir: str) -> LightRAG:
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
        initialize_pipeline_status,
    )

    finalize_share_data()
    initialize_share_data()

    os.environ["MEMGRAPH_WORKSPACE"] = workspace

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM, max_token_size=8192, func=_embedding,
    )

    rag = LightRAG(
        working_dir=working_dir,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=workspace,
        embedding_func=embedding_func,
        llm_model_func=_llm_router,
        enable_llm_cache=False,
        chunk_token_size=200,
        chunk_overlap_token_size=50,
    )
    await rag.initialize_storages()
    try:
        await initialize_pipeline_status()
    except Exception:
        pass
    return rag


async def _cleanup(workspace: str):
    try:
        async with _pool.get_session() as s:
            for prefix in ("KV_", "Vec_", "DocStatus_"):
                label = f"{prefix}{workspace}"
                try:
                    r = await s.run(
                        f"MATCH (n) WHERE ANY(l IN labels(n) "
                        f"WHERE l STARTS WITH '{label}') DETACH DELETE n"
                    )
                    await r.consume()
                except Exception:
                    pass
            try:
                r = await s.run(f"MATCH (n:`{workspace}`) DETACH DELETE n")
                await r.consume()
            except Exception:
                pass
    except Exception:
        pass


async def _run_sequential(rag: LightRAG, docs: dict[str, str]) -> float:
    """Insert docs one after the other. Returns total elapsed seconds."""
    t0 = time.perf_counter()
    for doc_id, text in docs.items():
        await rag.ainsert(text, ids=[doc_id])
    return time.perf_counter() - t0


async def _run_parallel(rag: LightRAG, docs: dict[str, str]) -> float:
    """Insert docs in parallel via asyncio.gather. Returns total elapsed seconds."""
    t0 = time.perf_counter()
    await asyncio.gather(
        *(rag.ainsert(text, ids=[doc_id]) for doc_id, text in docs.items())
    )
    return time.perf_counter() - t0


async def _scenario(name: str, docs: dict[str, str]) -> tuple[float, float, float]:
    """Run a scenario sequential then parallel; returns (seq, par, speedup)."""
    workspace = f"mvcc_{name}_{int(time.time())}"
    working_dir = tempfile.mkdtemp(prefix=f"mvcc_{name}_")

    try:
        # Sequential pass
        await _cleanup(workspace)
        rag_seq = await _build_rag(workspace, working_dir)
        seq = await _run_sequential(rag_seq, docs)
        await rag_seq.finalize_storages()
        await _cleanup(workspace)

        # Parallel pass — fresh state to avoid the parallel run benefiting from
        # caches built up by the sequential pass.
        rag_par = await _build_rag(workspace, working_dir)
        par = await _run_parallel(rag_par, docs)
        await rag_par.finalize_storages()

        speedup = seq / par if par > 0 else float("inf")
        return seq, par, speedup
    finally:
        await _cleanup(workspace)
        shutil.rmtree(working_dir, ignore_errors=True)


async def main():
    print("=" * 70)
    print("MVCC parallelization diagnostic — Memgraph (local docker compose)")
    print("=" * 70)
    print()

    print("Scenario A: shared entities (both docs about Paris)")
    seq_a, par_a, speedup_a = await _scenario("shared", SHARED_ENTITIES_DOCS)
    print(
        f"  Sequential : {seq_a:.2f}s    Parallel : {par_a:.2f}s    "
        f"Speedup : {speedup_a:.2f}x"
    )
    print()

    print("Scenario B: disjoint entities (Paris vs Tokyo)")
    seq_b, par_b, speedup_b = await _scenario("disjoint", DISJOINT_ENTITIES_DOCS)
    print(
        f"  Sequential : {seq_b:.2f}s    Parallel : {par_b:.2f}s    "
        f"Speedup : {speedup_b:.2f}x"
    )
    print()

    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    if speedup_a < 1.2 and speedup_b > 1.5:
        print(
            "✅ MVCC lock contention CONFIRMED.\n"
            "   Shared entities serialize writes ; disjoint ones parallelize.\n"
            "   This matches Tony's hypothesis. Application-level fix : "
            "pre-partition docs by topic before parallel ingestion."
        )
    elif speedup_a > 1.5 and speedup_b > 1.5:
        print(
            "✅ Both scenarios parallelize — no MVCC contention here.\n"
            "   The BNP slowness must come from elsewhere "
            "(single-driver throttle, application bottleneck, network)."
        )
    elif speedup_a < 1.2 and speedup_b < 1.2:
        print(
            "⚠️  Neither scenario parallelizes.\n"
            "   This is NOT MVCC contention — bottleneck is upstream of Memgraph.\n"
            "   Likely candidates : single uvicorn worker, single LLM call "
            "queue, or single Bolt session/driver instance."
        )
    else:
        print(
            f"🤔 Mixed result. shared={speedup_a:.2f}x  disjoint={speedup_b:.2f}x. "
            "Run again to check noise level."
        )


if __name__ == "__main__":
    asyncio.run(main())
