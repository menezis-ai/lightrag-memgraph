"""
Delete → re-ingest → vector query — the Memgraph 3.10+ stale-index landmine
sequence (audit 2026-07-02, finding TEST-6, remediation #10).

Memgraph 3.10+ leaves a stale vector-index entry behind a plain DETACH DELETE
of a vector-indexed vertex; a later ``vector_search`` then raises 50N42
"property from a deleted object". The impl mitigation (REMOVE the indexed
label before DETACH DELETE) exists on every delete path of ``vector_impl.py``
(:850-859, :866-876, :913-923, :951-957) — what was missing is the CI test of
the exact sequence. The test's target is the pinned CI Memgraph 3.12.0 leg
(``reference_memgraph_310_vector_delete``).

Both tests drive the PUBLIC LightRAG path: real ``ainsert`` (mock LLM /
embeddings, pattern from ``tests/test_e2e.py``) and ``adelete_by_doc_id``,
then query ``chunks_vdb`` (and ``entities_vdb`` where present) directly —
vector-only, no LLM, so a stale index entry cannot hide behind synthesis.
LightRAG API surface is hasattr-probed: missing pieces SKIP with a precise
reason, never error (matrix: 1.4.9.11 / 1.4.11 / 1.4.12 / newer).
"""

import hashlib
import shutil
import tempfile
import uuid

import numpy as np
import pytest
from lightrag import LightRAG
from lightrag.base import DocStatus

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool

twindb_lightrag_memgraph.register()


EMBEDDING_DIM = 384
# uuid-suffixed: Memgraph is shared with sibling test runs.
WORKSPACE = f"delreing_{uuid.uuid4().hex[:10]}"

SAMPLE_DOC = (
    "Paris is the capital and most populous city of France. "
    "The Eiffel Tower, built in 1889, is an iconic landmark located in Paris. "
    "The Seine River flows through Paris and is a major waterway in France. "
    "The Louvre Museum in Paris houses the Mona Lisa and thousands of artworks."
)

MOCK_ENTITIES = [
    ("Paris", "location", "Paris is the capital and most populous city of France."),
    ("France", "country", "France is a country in Western Europe."),
    (
        "Eiffel Tower",
        "landmark",
        "The Eiffel Tower is an iconic landmark built in 1889 in Paris.",
    ),
    ("Seine River", "location", "The Seine River flows through Paris."),
    ("Louvre Museum", "landmark", "The Louvre Museum in Paris houses the Mona Lisa."),
]

MOCK_RELATIONS = [
    ("Paris", "France", "capital city, geography", "Paris is the capital of France."),
    (
        "Eiffel Tower",
        "Paris",
        "located in, landmark",
        "The Eiffel Tower is located in Paris.",
    ),
    (
        "Seine River",
        "Paris",
        "flows through, geography",
        "The Seine River flows through Paris.",
    ),
    (
        "Louvre Museum",
        "Paris",
        "located in, culture",
        "The Louvre Museum is located in Paris.",
    ),
]


def _build_extraction_response() -> str:
    lines = []
    for name, etype, desc in MOCK_ENTITIES:
        lines.append(f"entity<|#|>{name}<|#|>{etype}<|#|>{desc}")
    for src, tgt, keywords, desc in MOCK_RELATIONS:
        lines.append(f"relation<|#|>{src}<|#|>{tgt}<|#|>{keywords}<|#|>{desc}")
    lines.append("<|COMPLETE|>")
    return "\n".join(lines)


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
    if "entity_types" in prompt_lower or "extract" in prompt_lower:
        return _build_extraction_response()
    if "summary" in prompt_lower or "merge" in prompt_lower:
        return "A summary of entities and relationships about Paris and France."
    return "Paris is the capital of France; the Eiffel Tower is in Paris."


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    results = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
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
    d = tempfile.mkdtemp(prefix="lightrag_delreing_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
async def rag(monkeypatch, working_dir):
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
    from lightrag.utils import EmbeddingFunc

    # Reset LightRAG global locks (bound to previous event loop between tests)
    finalize_share_data()
    initialize_share_data()

    monkeypatch.setenv("MEMGRAPH_WORKSPACE", WORKSPACE)

    await _cleanup_workspace()

    instance = LightRAG(
        working_dir=working_dir,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=WORKSPACE,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_mock_embedding,
        ),
        llm_model_func=_mock_llm,
        enable_llm_cache=False,
        chunk_token_size=200,
        chunk_overlap_token_size=50,
    )
    await instance.initialize_storages()

    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass

    try:
        yield instance
    finally:
        await _cleanup_workspace()
        await instance.finalize_storages()


async def _cleanup_workspace() -> None:
    try:
        async with _pool.get_session() as session:
            for prefix in ("KV_", "Vec_", "DocStatus_", "Folder_"):
                label = f"{prefix}{WORKSPACE}"
                result = await session.run(
                    "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $label) "
                    "DETACH DELETE n",
                    label=label,
                )
                await result.consume()
            result = await session.run(f"MATCH (n:`{WORKSPACE}`) DETACH DELETE n")
            await result.consume()
    except Exception:
        pass


# ── Helpers ──────────────────────────────────────────────────────────


async def _existing_vec_ids(namespace_suffix: str) -> set[str]:
    """Ids physically present under Vec_{ws}_{suffix} (raw Cypher truth)."""
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`Vec_{WORKSPACE}_{namespace_suffix}`) RETURN n.id AS id"
        )
        ids = {record["id"] async for record in result}
        await result.consume()
        return ids


async def _vector_query(vdb, query_text: str, *, context: str) -> list:
    """Run a vector-only query, translating the landmine into a clear failure.

    A 50N42 / "deleted object" error here is the exact Memgraph 3.10+
    stale-vector-index failure this file exists to catch — name it instead of
    letting a generic traceback obscure the regression.
    """
    try:
        results = await vdb.query(query_text, top_k=5)
    except Exception as exc:  # noqa: BLE001 — re-raised unless it's the landmine
        message = str(exc)
        if "50N42" in message or "deleted object" in message.lower():
            pytest.fail(
                f"stale vector-index entry surfaced ({context}): Memgraph "
                f"3.10+ landmine — a delete path skipped the REMOVE-label "
                f"mitigation (vector_impl.py:850-957). Original error: "
                f"{message}"
            )
        raise
    assert isinstance(results, list), f"vector query returned non-list: {results!r}"
    return results


def _require_api(rag_instance, attr: str):
    if not hasattr(rag_instance, attr):
        pytest.skip(
            f"this LightRAG build has no LightRAG.{attr} — the public "
            "delete/re-ingest path under test does not exist on this version"
        )
    return getattr(rag_instance, attr)


async def _ingest_and_get_doc(rag_instance) -> tuple[str, list[str]]:
    """ainsert SAMPLE_DOC, return (doc_id, chunks_list) once PROCESSED."""
    await rag_instance.ainsert(SAMPLE_DOC)
    processed = await rag_instance.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    assert (
        len(processed) == 1
    ), f"expected exactly one PROCESSED doc after ainsert, got {len(processed)}"
    doc_id = next(iter(processed))
    status_doc = processed[doc_id]
    chunks_list = list(getattr(status_doc, "chunks_list", None) or [])
    return doc_id, chunks_list


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_delete_then_immediate_vector_query_has_no_stale_index_error(rag):
    """ingest → delete → vector query straight away (no re-ingest).

    On Memgraph 3.10+ a stale index entry left by the delete would make
    vector_search raise 50N42 right here. The single-doc workspace also pins
    the semantic contract: after the only doc is deleted, the query returns
    empty — not phantom rows referencing deleted chunks.
    """
    chunks_vdb = _require_api(rag, "chunks_vdb")
    adelete_by_doc_id = _require_api(rag, "adelete_by_doc_id")

    doc_id, pre_delete_chunks = await _ingest_and_get_doc(rag)

    baseline = await _vector_query(chunks_vdb, "Paris landmarks", context="baseline")
    assert baseline, "baseline vector query must return results before delete"

    deletion_result = await adelete_by_doc_id(doc_id)
    assert (
        deletion_result.status == "success"
    ), f"deletion failed for {doc_id}: {deletion_result.message}"

    # The landmine moment: query the vector index immediately after delete.
    results = await _vector_query(
        chunks_vdb, "Paris landmarks", context="immediately after delete"
    )
    assert results == [], (
        "vector query returned rows after the only document was deleted "
        f"(stale entries?): {results}"
    )

    # Entity vectors ride the same delete cascade — probe that index too.
    entities_vdb = getattr(rag, "entities_vdb", None)
    if entities_vdb is not None:
        await _vector_query(
            entities_vdb, "Paris", context="entities index after delete"
        )

    # And nothing physical survived under the chunks namespace.
    assert await _existing_vec_ids("chunks") == set()
    assert doc_id not in await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    assert pre_delete_chunks, "sanity: the deleted doc had chunks to begin with"


@pytest.mark.integration
async def test_delete_then_reingest_same_content_vector_query_succeeds(rag):
    """The exact landmine sequence: ingest → delete → re-ingest → query.

    Re-ingesting identical content re-creates the same content-derived chunk
    ids on freshly indexed vertices. If the earlier delete left a stale index
    entry, the post-re-ingest vector_search trips 50N42 on Memgraph 3.10+.
    The pinned 3.12.0 CI leg guards this behavior. Also guards the
    dedup side: the deleted doc must NOT be treated as still-existing content
    by the enqueue dedup, i.e. re-ingest must reach PROCESSED again.
    """
    chunks_vdb = _require_api(rag, "chunks_vdb")
    adelete_by_doc_id = _require_api(rag, "adelete_by_doc_id")

    first_doc_id, first_chunks = await _ingest_and_get_doc(rag)
    assert first_chunks, "first ingest produced no chunks"

    deletion_result = await adelete_by_doc_id(first_doc_id)
    assert (
        deletion_result.status == "success"
    ), f"deletion failed for {first_doc_id}: {deletion_result.message}"
    assert await _existing_vec_ids("chunks") == set()

    # Re-ingest the SAME content. A leftover DocStatus/full_docs remnant would
    # make enqueue dedup swallow this silently → the PROCESSED assert inside
    # _ingest_and_get_doc catches that as a real bug.
    second_doc_id, second_chunks = await _ingest_and_get_doc(rag)
    assert second_chunks, "re-ingest produced no chunks"

    results = await _vector_query(
        chunks_vdb, "Paris landmarks", context="after delete + re-ingest"
    )
    assert results, "vector query after re-ingest returned no results"

    # Sanity on the results: every returned id must exist physically in the
    # vector namespace right now — a hit that resolves to nothing physical is
    # the semantic shadow of a stale index entry.
    physical_ids = await _existing_vec_ids("chunks")
    returned_ids = {entry.get("id") for entry in results}
    assert returned_ids, f"vector results carry no ids: {results}"
    assert returned_ids <= physical_ids, (
        f"vector query returned ids absent from storage: "
        f"{returned_ids - physical_ids}"
    )

    # Identical content → same content-derived doc/chunk ids as round one.
    # (Not a contract we depend on, but if it holds it proves the index rows
    # really were re-created on new vertices under the same keys.)
    if second_doc_id == first_doc_id:
        assert set(second_chunks) == set(first_chunks)

    entities_vdb = getattr(rag, "entities_vdb", None)
    if entities_vdb is not None:
        entity_results = await _vector_query(
            entities_vdb, "Paris", context="entities index after re-ingest"
        )
        assert entity_results, "entity vector query after re-ingest returned nothing"
