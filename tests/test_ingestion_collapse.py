"""
Ingestion collapse tests: fault injection INSIDE the real ainsert() pipeline.

Closes audit finding TEST-3 (docs/audits/ingestion-reindex/audit-2026-07-02.md):
before this file, no CI test injected a fault inside the real pipeline against
Memgraph — the only FAILED status ever asserted was hand-written. Each test here
runs the real ``rag.ainsert()`` pipeline against a real Memgraph, injects one
fault, and asserts three invariants:

  1. The process never crashes — ``ainsert()`` returns (graceful degradation,
     doctrine "stress-test d'effondrement").
  2. The document ends in a LEGAL TERMINAL status (FAILED expected; a doc stuck
     in PENDING/PROCESSING would be the TEST-5 bug class).
  3. The residual Memgraph state is characterized with real assertions — this
     converts the audit's INFERRED PIPE-4/5 claims (FAILED leaves resident
     chunks / vectors / full_docs) into pinned behavior. Where residue EXISTS
     we assert its existence (a characterization pin that fails loudly if
     cleanup behavior ever changes); where the pipeline is clean we assert
     cleanliness.

Version tolerance (memory ``feedback_lightrag_version_skew``): local dev runs
lightrag 1.5.4, CI runs 1.4.9.11 / 1.4.11 / 1.4.12. All interactions go
through the public API (``ainsert``, storage instances, ``doc_status``).
Graph-merge faults are injected by monkeypatching INSTANCE methods on
``rag.chunk_entity_relation_graph`` — never module-level functions — because
``merge_nodes_and_edges`` is called from ``lightrag/lightrag.py`` in 1.4.9.11
but from ``lightrag/pipeline.py`` (module-local binding, audit SKEW-1) in
1.5.x; instance patching sidesteps that skew entirely. Assertions that are
genuinely version- or race-dependent are tolerant, with the reason inline.

Requires a running Memgraph (MEMGRAPH_URI); auto-skipped otherwise.
"""

import asyncio
import hashlib
import os
import shutil
import tempfile
import uuid

import numpy as np
import pytest
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool

twindb_lightrag_memgraph.register()

# ── Constants ────────────────────────────────────────────────────────

EMBEDDING_DIM = 384

# Small docs → 1 chunk each at chunk_token_size=200; keeps runtime low.
DOC_A = (
    "Paris is the capital and most populous city of France. "
    "The Eiffel Tower, built in 1889, is an iconic landmark located in Paris. "
    "France is a country in Western Europe known for its wine and cuisine."
)
# ZEBRAFAULT marker: the mock embedding raises only on texts containing it,
# so in a two-doc batch the fault hits doc B's chunks and nothing else.
DOC_B = (
    "ZEBRAFAULT Berlin is the capital of Germany. "
    "The Brandenburg Gate is a landmark in Berlin ZEBRAFAULT."
)
DOC_C = "A tiny second document about Rome, the capital of Italy."

# Valid extraction output in LightRAG's delimiter format (same contract as
# tests/test_e2e.py) — used when the scenario needs extraction to SUCCEED.
VALID_EXTRACTION = "\n".join(
    [
        "entity<|#|>Paris<|#|>location<|#|>Paris is the capital of France.",
        "entity<|#|>France<|#|>country<|#|>France is a country in Western Europe.",
        (
            "relation<|#|>Paris<|#|>France<|#|>capital, geography<|#|>"
            "Paris is the capital of France."
        ),
        "<|COMPLETE|>",
    ]
)

# Truncated / malformed extraction output: a 3-field entity record cut
# mid-word, a dangling relation, no <|COMPLETE|> terminator. Every line is
# unparseable → the extraction parser must skip them all without raising.
TRUNCATED_EXTRACTION = "entity<|#|>Paris<|#|>loc\nrelation<|#|>Pa"

# Legal terminal statuses — a faulted doc parked anywhere else is a bug.
TERMINAL_STATUSES = (DocStatus.PROCESSED, DocStatus.FAILED)


# ── Fault-controlled mock LLM & embedding ────────────────────────────


def _build_faults() -> dict:
    """Mutable fault switchboard shared with the mock LLM/embedding closures."""
    return {"embed_fail": False, "embed_marker": None, "llm_mode": "valid"}


def _build_mocks(faults: dict):
    async def mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        p = prompt.lower() if isinstance(prompt, str) else ""
        if "entity_types" in p or "extract" in p:
            if faults["llm_mode"] == "truncated":
                return TRUNCATED_EXTRACTION
            return VALID_EXTRACTION
        if "summary" in p or "merge" in p:
            return "A summary of the merged entities."
        return "A generic answer."

    async def mock_embedding(texts: list[str]) -> np.ndarray:
        if faults["embed_fail"]:
            raise RuntimeError("collapse-test: injected embedding failure")
        marker = faults["embed_marker"]
        if marker and any(marker in t for t in texts):
            raise RuntimeError("collapse-test: injected embedding failure (marker)")
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.frombuffer(h * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
                :EMBEDDING_DIM
            ].astype(np.float32)
            norm = np.linalg.norm(vec)
            results.append(vec / norm if norm > 0 else vec)
        return np.array(results)

    return mock_llm, mock_embedding


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
async def collapse_env():
    """(rag, faults, workspace) on a uuid-suffixed workspace.

    Unique workspace per test: this Memgraph instance is shared with other
    concurrently running suites — cleanup only ever touches our own labels.
    """
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    # Reset LightRAG global locks (bound to previous event loop between tests)
    finalize_share_data()
    initialize_share_data()

    workspace = f"clps_{uuid.uuid4().hex[:10]}"
    prev_ws = os.environ.get("MEMGRAPH_WORKSPACE")
    os.environ["MEMGRAPH_WORKSPACE"] = workspace
    working_dir = tempfile.mkdtemp(prefix="lightrag_collapse_")

    faults = _build_faults()
    mock_llm, mock_embedding = _build_mocks(faults)

    rag = LightRAG(
        working_dir=working_dir,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=workspace,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=mock_embedding,
        ),
        llm_model_func=mock_llm,
        enable_llm_cache=False,
        chunk_token_size=200,
        chunk_overlap_token_size=50,
    )
    await rag.initialize_storages()

    # LightRAG >= 1.4.9.11 calls this internally, but older paths do not.
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass

    yield rag, faults, workspace

    await _drop_workspace(workspace)
    await rag.finalize_storages()
    if prev_ws is None:
        os.environ.pop("MEMGRAPH_WORKSPACE", None)
    else:
        os.environ["MEMGRAPH_WORKSPACE"] = prev_ws
    shutil.rmtree(working_dir, ignore_errors=True)


async def _drop_workspace(workspace: str) -> None:
    """Drop every node carrying a label that embeds our uuid workspace."""
    try:
        async with _pool.get_session() as session:
            result = await session.run(
                "MATCH (n) WHERE ANY(l IN labels(n) WHERE l CONTAINS $ws) "
                "DETACH DELETE n",
                ws=workspace,
            )
            await result.consume()
    except Exception:
        pass


# ── Residue characterization helpers ─────────────────────────────────


async def _count_label(session, label: str) -> int:
    result = await session.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
    record = await result.single()
    await result.consume()
    return record["cnt"] if record else 0


async def _residue_snapshot(workspace: str) -> dict[str, int]:
    """Physical per-storage row counts for one workspace.

    Raw Cypher on purpose: this is the ground truth the audit asked to pin
    (PIPE-4/5), independent of what the LightRAG API layer reports.
    """
    labels = {
        "full_docs": f"KV_{workspace}_full_docs",
        "text_chunks": f"KV_{workspace}_text_chunks",
        "entity_chunks": f"KV_{workspace}_entity_chunks",
        "relation_chunks": f"KV_{workspace}_relation_chunks",
        "vec_chunks": f"Vec_{workspace}_chunks",
        "vec_entities": f"Vec_{workspace}_entities",
        "vec_relationships": f"Vec_{workspace}_relationships",
    }
    snap: dict[str, int] = {}
    async with _pool.get_read_session() as session:
        for key, label in labels.items():
            snap[key] = await _count_label(session, label)
        snap["graph_nodes"] = await _count_label(session, workspace)
        result = await session.run(
            f"MATCH (:`{workspace}`)-[r]->(:`{workspace}`) RETURN count(r) AS cnt"
        )
        record = await result.single()
        await result.consume()
        snap["graph_edges"] = record["cnt"] if record else 0
    return snap


async def _count_vec_chunks_for_doc(workspace: str, doc_id: str) -> int:
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (n:`Vec_{workspace}_chunks`) WHERE n.full_doc_id = $doc_id "
            "RETURN count(n) AS cnt",
            doc_id=doc_id,
        )
        record = await result.single()
        await result.consume()
        return record["cnt"] if record else 0


async def _doc_statuses(rag) -> dict[str, tuple[DocStatus, object]]:
    """doc_id → (DocStatus, status_doc) via the public doc_status API.

    Iterates the DocStatus enum so version-specific intermediate members
    (e.g. PARSING/ANALYZING in 1.5.x) are covered without naming them.
    """
    out: dict[str, tuple[DocStatus, object]] = {}
    for status in DocStatus:
        docs = await rag.doc_status.get_docs_by_status(status)
        for doc_id, status_doc in docs.items():
            out[doc_id] = (status, status_doc)
    return out


async def _settled_statuses(rag) -> dict[str, tuple[DocStatus, object]]:
    """Statuses after letting orphaned sibling tasks drain.

    On failure, stage-1 sibling tasks (doc_status / chunks_vdb / text_chunks
    upserts run in one asyncio.gather — PIPE-5) may still be in flight when
    ainsert() returns: 1.4.9.11 never cancels them, 1.5.x cancels but a task
    may already be past its Bolt write. The short sleep makes the residue
    snapshot deterministic to read (not deterministic in content — see the
    tolerant text_chunks assertions).
    """
    await asyncio.sleep(0.5)
    return await _doc_statuses(rag)


def _assert_all_terminal(statuses: dict) -> None:
    for doc_id, (status, _) in statuses.items():
        assert status in TERMINAL_STATUSES, (
            f"doc {doc_id} parked in non-terminal status {status} after the "
            "pipeline returned — illegal terminal state (audit TEST-5 class)"
        )


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestIngestionCollapse:
    """Fault injection inside the real ainsert() pipeline (audit TEST-3)."""

    async def test_embedding_failure_degrades_to_failed_with_residue(
        self, collapse_env
    ):
        """Embedding raises during stage 1 → FAILED + characterized residue."""
        rag, faults, ws = collapse_env
        faults["embed_fail"] = True

        # (1) Graceful degradation: the injected RuntimeError must not escape
        # ainsert(). Per-doc handlers write FAILED and swallow (1.4.9.11
        # lightrag.py stage-1 except-block; 1.5.4 _finalize_doc_failure).
        track_id = await rag.ainsert(DOC_A)
        assert track_id is not None

        statuses = await _settled_statuses(rag)
        _assert_all_terminal(statuses)

        # (2) Legal terminal status: FAILED, carrying the injected error.
        assert len(statuses) == 1
        ((doc_id, (status, status_doc)),) = statuses.items()
        assert status == DocStatus.FAILED
        assert "collapse-test" in (getattr(status_doc, "error_msg", "") or "")

        snap = await _residue_snapshot(ws)

        # (3) Residue characterization — PIPE-4/5 pins.
        # full_docs keeps the FULL document text: neither 1.4.9.11's FAILED
        # write nor 1.5.x's _finalize_doc_failure purges it (it is also what
        # makes the doc eligible for implicit retry — see the retry test).
        # Compliance-relevant residue (prisme-H §2.2 K1). Pinned: EXISTS.
        assert snap["full_docs"] == 1, f"full_docs residue changed: {snap}"

        # Vector rows: deterministically ZERO — vector_impl.upsert computes
        # every embedding before its single UNWIND write, so an embedding
        # failure can never leave a partial vector batch behind.
        assert snap["vec_chunks"] == 0, f"vec residue after embed failure: {snap}"

        # text_chunks KV rows: genuinely race-dependent, both outcomes legal.
        # The stage-1 gather runs doc_status/chunks_vdb/text_chunks upserts in
        # parallel; 1.4.9.11 lets the text_chunks task finish after the gather
        # raised (→ rows land), 1.5.x cancels pending siblings (→ usually 0,
        # but the task can already be past its atomic UNWIND). Both observed
        # live on 1.5.4 (0 in a single-doc run, 1 in a batch run).
        assert snap["text_chunks"] in (0, 1), (
            "text_chunks residue after embedding failure left the known "
            f"envelope: {snap}"
        )

        # Nothing downstream of stage 1 may exist: extraction never ran.
        for key in (
            "graph_nodes",
            "graph_edges",
            "vec_entities",
            "vec_relationships",
            "entity_chunks",
            "relation_chunks",
        ):
            assert snap[key] == 0, f"unexpected downstream residue {key}: {snap}"

    async def test_failed_doc_is_implicitly_retried_on_next_pipeline_trigger(
        self, collapse_env
    ):
        """PIPE-12 pin: a FAILED doc is silently retried by ANY later trigger.

        No operator action, no reprocess endpoint: inserting an unrelated
        second document re-runs the pipeline, which selects FAILED docs whose
        full_docs payload survives (that residue is the retry fuel) and resets
        them to PENDING. Source-confirmed on the 1.4.9.11 wheel
        (_validate_and_fix_document_consistency) and on 1.5.4
        (_INFLIGHT_DOC_STATUSES includes FAILED); runtime-confirmed on 1.5.4.
        """
        rag, faults, ws = collapse_env

        faults["embed_fail"] = True
        await rag.ainsert(DOC_A)
        statuses = await _settled_statuses(rag)
        ((failed_doc_id, (status, _)),) = statuses.items()
        assert status == DocStatus.FAILED  # precondition

        # Heal the fault, trigger the pipeline with an unrelated document.
        faults["embed_fail"] = False
        await rag.ainsert(DOC_C)

        statuses = await _settled_statuses(rag)
        _assert_all_terminal(statuses)
        assert len(statuses) == 2
        retried_status, _ = statuses[failed_doc_id]
        assert retried_status == DocStatus.PROCESSED, (
            "implicit FAILED-retry (PIPE-12) did not occur: the previously "
            f"FAILED doc is {retried_status} after an unrelated ainsert()"
        )

        # The healed retry rebuilt the chunk stores for both docs.
        snap = await _residue_snapshot(ws)
        assert snap["vec_chunks"] >= 2, f"retry did not rebuild vectors: {snap}"
        assert snap["text_chunks"] >= 2, f"retry did not rebuild chunks: {snap}"

    async def test_graph_merge_failure_degrades_to_failed_chunks_resident(
        self, collapse_env, monkeypatch
    ):
        """Graph merge raises → FAILED; stage-1 chunks/vectors stay resident.

        Fault: instance-level patch on rag.chunk_entity_relation_graph.
        get_node is the FIRST graph call of _merge_nodes_then_upsert on every
        supported version, and it reaches the real instance on both skew
        sides: directly in 1.5.x (buffered-merge patch dead there, SKEW-1),
        via _BufferedGraphProxy._real.get_node in 1.4.9.11. upsert_node is
        patched too as belt-and-braces (it is what 1.5.x would hit next;
        in 1.4.9.11 the proxy buffers it, and the buffer flush only runs
        after a merge that will already have raised).
        """
        rag, faults, ws = collapse_env

        async def _boom(*args, **kwargs):
            raise RuntimeError("collapse-test: injected graph merge failure")

        monkeypatch.setattr(rag.chunk_entity_relation_graph, "get_node", _boom)
        monkeypatch.setattr(rag.chunk_entity_relation_graph, "upsert_node", _boom)

        # (1) Graceful degradation.
        track_id = await rag.ainsert(DOC_A)
        assert track_id is not None

        statuses = await _settled_statuses(rag)
        _assert_all_terminal(statuses)

        # (2) Legal terminal status: FAILED at the merge stage.
        assert len(statuses) == 1
        ((doc_id, (status, status_doc)),) = statuses.items()
        assert status == DocStatus.FAILED
        assert "collapse-test" in (getattr(status_doc, "error_msg", "") or "")

        snap = await _residue_snapshot(ws)

        # (3) Residue characterization — the core PIPE-4 pin: a merge-stage
        # FAILED doc leaves its full text, chunk text AND chunk vectors
        # resident (stage 1 completed before the merge stage started; nothing
        # rolls them back). A FAILED doc is therefore silently retrievable by
        # vector search until deleted or retried. Pinned: EXISTS.
        assert snap["full_docs"] == 1, f"full_docs residue changed: {snap}"
        assert snap["text_chunks"] > 0, (
            f"PIPE-4 pin broken: chunk text no longer resident after merge "
            f"failure — cleanup behavior changed: {snap}"
        )
        assert snap["vec_chunks"] > 0, (
            f"PIPE-4 pin broken: chunk vectors no longer resident after merge "
            f"failure — cleanup behavior changed: {snap}"
        )

        # Graph side is clean on every supported version, for different
        # reasons: get_node raises before any entity write reaches the graph
        # or the entity/relationship vdbs (get-before-upsert merge order); on
        # 1.4.9.11 the buffered node flush additionally never runs because
        # the wrapper only flushes after a successful merge (registry.py
        # _buffered_merge_nodes_and_edges).
        for key in (
            "graph_nodes",
            "graph_edges",
            "vec_entities",
            "vec_relationships",
            "entity_chunks",
        ):
            assert snap[key] == 0, f"unexpected graph-side residue {key}: {snap}"

    async def test_truncated_llm_extraction_never_crashes(self, collapse_env):
        """Malformed/truncated extraction output → degrade, never crash.

        The extraction parser skips unparseable records without raising
        (prisme-H §4.1), so the expected outcome is an empty-extraction
        PROCESSED; FAILED is also a legal degradation on a stricter version.
        Anything else (exception out of ainsert, doc stuck non-terminal)
        fails the test.
        """
        rag, faults, ws = collapse_env
        faults["llm_mode"] = "truncated"

        # (1) Graceful degradation.
        track_id = await rag.ainsert(DOC_A)
        assert track_id is not None

        statuses = await _settled_statuses(rag)
        _assert_all_terminal(statuses)
        assert len(statuses) == 1
        ((doc_id, (status, _)),) = statuses.items()

        # (2) Legal terminal status — tolerant across versions by design.
        assert status in TERMINAL_STATUSES

        snap = await _residue_snapshot(ws)
        if status == DocStatus.PROCESSED:
            # Observed on 1.5.4 and predicted for 1.4.9.11: the doc is
            # PROCESSED with chunks+vectors resident but a completely empty
            # knowledge graph — silent quality degradation, zero signal in
            # the status. Characterized here so a behavior change (e.g. a
            # version that starts FAILING these docs) is caught loudly.
            assert snap["text_chunks"] > 0, f"chunks missing: {snap}"
            assert snap["vec_chunks"] > 0, f"vectors missing: {snap}"
            for key in ("graph_nodes", "vec_entities", "vec_relationships"):
                assert (
                    snap[key] == 0
                ), f"truncated extraction unexpectedly produced {key}: {snap}"
        else:  # FAILED: stricter version — the enqueue residue still holds.
            assert snap["full_docs"] == 1, f"full_docs residue changed: {snap}"

    async def test_partial_batch_fault_isolates_failure_to_second_doc(
        self, collapse_env
    ):
        """Two-doc batch, fault on doc B only → doc A lands PROCESSED.

        Partial-batch invariant: per-document isolation. The marker-scoped
        embedding fault only fires on doc B's chunk batch; doc A must complete
        end-to-end (chunks, vectors, graph) in the same pipeline run.
        """
        rag, faults, ws = collapse_env
        faults["embed_marker"] = "ZEBRAFAULT"

        track_id = await rag.ainsert(
            [DOC_A, DOC_B], file_paths=["doc_a.txt", "doc_b.txt"]
        )
        assert track_id is not None

        statuses = await _settled_statuses(rag)
        _assert_all_terminal(statuses)
        assert len(statuses) == 2

        by_path = {
            getattr(sd, "file_path", None): (doc_id, st)
            for doc_id, (st, sd) in statuses.items()
        }
        assert set(by_path) == {"doc_a.txt", "doc_b.txt"}
        doc_a_id, doc_a_status = by_path["doc_a.txt"]
        doc_b_id, doc_b_status = by_path["doc_b.txt"]

        assert doc_a_status == DocStatus.PROCESSED, (
            "partial-batch invariant broken: the healthy doc did not survive "
            f"a sibling's fault (got {doc_a_status})"
        )
        assert doc_b_status == DocStatus.FAILED

        # Doc A's pipeline output is fully present…
        snap = await _residue_snapshot(ws)
        assert snap["graph_nodes"] > 0, f"doc A graph missing: {snap}"
        assert await _count_vec_chunks_for_doc(ws, doc_a_id) > 0

        # …and doc B contributed no vectors (embedding is all-or-nothing
        # before the vector write — same determinism as the single-doc test).
        assert await _count_vec_chunks_for_doc(ws, doc_b_id) == 0
