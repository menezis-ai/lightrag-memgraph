"""
Native upload dedupe contract — real HTTP, real Memgraph (audit 2026-07-02,
findings ING-2/TEST-1 + PIPE-7, remediation #5).

The data-protecting contract "the same content uploaded twice is shared or
surfaced as a known duplicate, NEVER ingested twice" was previously tested only
at the wrong level: unit tests mock sessions, ``test_folder_membership.py``
drives the storage hooks directly without proving LightRAG invokes them, and
``test_upload_duplicate_lookup.py`` auto-skips on the BNP pin 1.4.9.11. This
file uploads through LightRAG's real ``POST /documents/upload`` route against
the registered Memgraph backends and asserts the *physical* end state.

Version tolerance is the point (matrix: 1.4.9.11 / 1.4.11 / 1.4.12 / newer).
The HTTP dedup surface differs per LightRAG family — all branches are handled,
and every branch (including the SKIP ones) first asserts the no-double-ingestion
invariants, so the core contract is enforced on every version:

- 1.4.x family: same-filename re-upload is deduped at the route (200
  ``status="duplicated"`` — 1.4.9.11 wheel ``document_routes.py:2104-2115``,
  1.4.11/1.4.12 identical) through ``doc_status.get_doc_by_file_path`` —
  which is exactly where our duplicate-share hook fires
  (``docstatus_impl.py:792-822``). Same-content / different-filename
  re-uploads diverge inside the family: the BNP pin 1.4.9.11 silently ignores
  them at enqueue (wheel ``lightrag.py:1362-1374`` — no duplicate record, no
  share trigger; the dup-record variants SKIP there with a precise reason
  after the invariants have been asserted), while 1.4.11/1.4.12 already
  create a fresh enqueue-time ``dup-`` FAILED record with
  ``metadata.is_duplicate`` (1.4.11 wheel ``lightrag.py:1398-1443``) — the
  interceptable PIPE-7 shape.
- 1.5.x family: same-filename re-upload is an HTTP 409 (route pre-check via
  ``get_doc_by_file_basename`` — same share hook). Same-content re-uploads are
  accepted (200) and deduped asynchronously with ``metadata.is_duplicate``
  markers that our ``docstatus_impl.upsert`` (:484-498) intercepts into a
  share membership when a folder is bound — the PIPE-7 divergence covered by
  the last two tests:
  (a) no folder capture installed  → visible FAILED duplicate record persisted;
  (b) folder bound (X-Twin-Folder) → duplicate verdict intercepted, membership
      shared instead (the audited UI blind spot).
  Two async shapes exist upstream: text routes dedup at ENQUEUE with a fresh
  ``dup-`` record (audit's PIPE-7 description), while file uploads defer
  parsing and dedup POST-PARSE by writing the FAILED verdict onto the second
  upload's already-persisted real row (1.5.4 ``pipeline.py:3049-3125``). With
  a folder bound, intercepting that second shape strands the row in a
  non-terminal status — observed on 1.5.4, classified explicitly by the
  outcome probe as ``shared_with_residual`` and reported as a real bug (not
  fixed here; 1.5.4 is outside the CI matrix).

NOTE (MG-15): the interception window where the original vanishes between
dedup detection and the share MERGE is a race, deliberately NOT exercised here.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _install_storage_folder_capture, _pool

twindb_lightrag_memgraph.register()


EMBEDDING_DIM = 384
# uuid-suffixed: Memgraph is shared with sibling test runs.
WORKSPACE = f"dedupe_{uuid.uuid4().hex[:10]}"
DEFAULT_FOLDER = "default"
SHARE_FOLDER = "sandbox"

TRACK_DEADLINE_S = 30.0
OUTCOME_DEADLINE_S = 20.0
# How long the post-upload DocStatus shape must stop changing before the
# outcome probe declares it settled (the 1.5.x deferred-parse flow mutates the
# second row a few times before the duplicate verdict lands).
SETTLE_STABLE_S = 3.0

SAME_NAME_DOC = (
    "Upload dedupe contract document one. Atlas talks to Boreal and Boreal "
    "escalates to Cygnus, giving the chunker, vector storage, and graph "
    "extraction concrete content to persist twice-but-store-once."
)
FOLDER_SHARE_DOC = (
    "Upload dedupe contract document two. Atlas mirrors data into Boreal "
    "nightly while Cygnus audits both, so a duplicate re-upload into another "
    "folder must share the existing document instead of re-ingesting it."
)
BARE_APP_DOC = (
    "Upload dedupe contract document three. Boreal proxies Atlas traffic and "
    "Cygnus records the incidents; without the Twin folder capture the native "
    "duplicate record must stay visible instead of being silently absorbed."
)


def _build_extraction_response() -> str:
    return "\n".join(
        [
            "entity<|#|>Atlas<|#|>service<|#|>Atlas is an internal service.",
            "entity<|#|>Boreal<|#|>service<|#|>Boreal is an internal service.",
            "entity<|#|>Cygnus<|#|>service<|#|>Cygnus receives incident reports.",
            "relation<|#|>Atlas<|#|>Boreal<|#|>depends on<|#|>Atlas depends on Boreal.",
            "relation<|#|>Boreal<|#|>Cygnus<|#|>reports to<|#|>Boreal reports to Cygnus.",
            "<|COMPLETE|>",
        ]
    )


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
    if "entity_types" in prompt_lower or "extract" in prompt_lower:
        return _build_extraction_response()
    if "summary" in prompt_lower or "merge" in prompt_lower:
        return "Internal services and dependencies summary."
    return "Atlas depends on Boreal, and Boreal reports incidents to Cygnus."


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
            :EMBEDDING_DIM
        ].astype(np.float32)
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm else vec)
    return np.array(vectors)


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def runtime_dirs():
    root = tempfile.mkdtemp(prefix="native_dedupe_")
    yield {
        "working": os.path.join(root, "work"),
        "input": os.path.join(root, "input"),
    }
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
async def native_runtime(request, monkeypatch, runtime_dirs):
    """LightRAG + native /documents routes on the Memgraph backends.

    Indirect param (default ``True``): install the Twin folder-capture
    middleware. ``False`` builds the bare native surface — the
    LightRAG-compat "extension absent" deployment — which is the only
    entry point where ingestion runs without a bound folder (PIPE-7
    variant (a)).
    """
    install_capture = getattr(request, "param", True)

    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", WORKSPACE)
    monkeypatch.setenv("INPUT_DIR", runtime_dirs["input"])
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", DEFAULT_FOLDER)
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"primary"},'
        '{"id":"sandbox","label":"Sandbox","kind":"secondary"}]',
    )
    # Keep folder resolution on palier 1 regardless of ambient env.
    monkeypatch.delenv("TWIN_IDP_JWKS_URL", raising=False)

    finalize_share_data()
    initialize_share_data()
    await _cleanup_workspace()

    rag = LightRAG(
        working_dir=runtime_dirs["working"],
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
        chunk_token_size=120,
        chunk_overlap_token_size=20,
        tokenizer=Tokenizer("native-dedupe-char", _CharTokenizer()),
    )
    await rag.initialize_storages()

    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass

    from lightrag.api.routers.document_routes import (
        DocumentManager,
        create_document_routes,
    )

    from tests.conftest import ensure_fresh_native_document_router

    doc_manager = DocumentManager(runtime_dirs["input"], workspace=WORKSPACE)
    app = FastAPI()
    if install_capture:
        _install_storage_folder_capture(app)
    ensure_fresh_native_document_router()
    app.include_router(create_document_routes(rag, doc_manager, api_key=None))

    try:
        yield rag, app
    finally:
        await _cleanup_workspace()
        await rag.finalize_storages()


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


# ── Physical-state helpers (raw Cypher: the source of truth) ─────────


async def _count_nodes(label_prefix: str) -> int:
    async with _pool.get_read_session() as session:
        result = await session.run(
            "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $prefix) "
            "RETURN count(n) AS count",
            prefix=label_prefix,
        )
        record = await result.single()
        await result.consume()
        return record["count"] if record else 0


def _parse_metadata(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


async def _docstatus_rows() -> list[dict]:
    """Every physical DocStatus node in the workspace, folder-blind."""
    async with _pool.get_read_session() as session:
        result = await session.run(f"""
            MATCH (n:`DocStatus_{WORKSPACE}`)
            RETURN n.id AS id, n.status AS status, n.file_path AS file_path,
                   n.chunks_count AS chunks_count, n.metadata AS metadata
            """)
        rows = [
            {
                "id": record["id"],
                "status": str(record["status"] or "").lower(),
                "file_path": record["file_path"],
                "chunks_count": record["chunks_count"],
                "metadata": _parse_metadata(record["metadata"]),
            }
            async for record in result
        ]
        await result.consume()
        return rows


async def _snapshot() -> dict:
    return {
        "vec": await _count_nodes(f"Vec_{WORKSPACE}"),
        "kv": await _count_nodes(f"KV_{WORKSPACE}"),
        "rows": await _docstatus_rows(),
    }


async def _assert_no_double_ingestion(before: dict, original_doc_id: str) -> None:
    """The core invariant: one physical doc, no second chunk/Vec/KV set."""
    vec_now = await _count_nodes(f"Vec_{WORKSPACE}")
    kv_now = await _count_nodes(f"KV_{WORKSPACE}")
    assert vec_now == before["vec"], (
        f"vector rows changed after duplicate upload: {before['vec']} → {vec_now} "
        "(double ingestion or partial residue)"
    )
    assert (
        kv_now == before["kv"]
    ), f"KV rows changed after duplicate upload: {before['kv']} → {kv_now}"
    rows = await _docstatus_rows()
    processed = [r for r in rows if r["status"] == DocStatus.PROCESSED.value]
    assert [r["id"] for r in processed] == [
        original_doc_id
    ], f"expected exactly one PROCESSED doc ({original_doc_id}), got: {processed}"


# ── HTTP helpers ─────────────────────────────────────────────────────


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def _upload(
    client: httpx.AsyncClient,
    filename: str,
    content: str,
    folder: str | None = None,
) -> httpx.Response:
    headers = {"X-Twin-Folder": folder} if folder else {}
    return await client.post(
        "/documents/upload",
        files={"file": (filename, content.encode(), "text/plain")},
        headers=headers,
    )


async def _poll_track_processed(client: httpx.AsyncClient, track_id: str) -> dict:
    deadline = time.monotonic() + TRACK_DEADLINE_S
    payload: dict = {}
    while time.monotonic() < deadline:
        response = await client.get(f"/documents/track_status/{track_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        documents = payload.get("documents", [])
        if documents and all(doc["status"] == DocStatus.PROCESSED for doc in documents):
            return payload
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"track {track_id} did not reach PROCESSED within "
        f"{TRACK_DEADLINE_S}s: {payload}"
    )


async def _ingest_original(
    client: httpx.AsyncClient, rag, filename: str, content: str, folder: str | None
) -> tuple[str, str, dict]:
    """First upload: run to PROCESSED, return (doc_id, track_id, snapshot)."""
    response = await _upload(client, filename, content, folder)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "success", response.text
    track_id = response.json()["track_id"]
    payload = await _poll_track_processed(client, track_id)
    assert payload["total_count"] == 1
    doc_id = payload["documents"][0]["id"]

    stored = await rag.doc_status.get_by_id(doc_id)
    assert stored is not None
    snapshot = await _snapshot()
    assert {r["id"] for r in snapshot["rows"]} == {doc_id}
    assert snapshot["vec"] > 0
    return doc_id, track_id, snapshot


async def _settle_duplicate_outcome(
    rag, original_doc_id: str, share_folder: str | None = SHARE_FOLDER
) -> tuple[str, dict]:
    """Wait (bounded) for the async duplicate outcome after a 200 re-upload.

    A transient second row (PENDING/PARSING/…) is NOT an outcome — the 1.5.x
    upload path defers parsing, so the real second doc row exists briefly (or,
    see below, permanently) before the post-parse content dedup fires
    (``pipeline.py:3049-3125`` in 1.5.4). The probe therefore settles on
    *stable* states, returning one of:

      ("shared", {})                — share membership on the original, no
                                      second row left behind (the audit's
                                      PIPE-7 description)
      ("shared_with_residual", row) — share happened, but the second upload's
                                      own DocStatus row is stranded in a
                                      non-terminal status: LightRAG wrote its
                                      FAILED+is_duplicate verdict onto the
                                      already-persisted row and our
                                      interception swallowed that terminal
                                      update (observed on 1.5.4 — real
                                      residue, reported, not fixed here)
      ("dup_record", row)           — a second row settled as FAILED (native
                                      known-duplicate surfaced to the operator)
      ("silent", {})                — deadline passed, nothing surfaced
                                      (pin-family enqueue drops known content
                                      silently, wheel lightrag.py:1362-1374)

    Fails immediately if a second PROCESSED row appears (double ingestion).
    """
    deadline = time.monotonic() + OUTCOME_DEADLINE_S
    last_shape: tuple | None = None
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        rows = await _docstatus_rows()
        extra = [r for r in rows if r["id"] != original_doc_id]
        for row in extra:
            if row["status"] == DocStatus.PROCESSED.value:
                pytest.fail(
                    f"duplicate upload was fully re-ingested (second PROCESSED "
                    f"row {row['id']}): {rows}"
                )
        terminal_dups = [r for r in extra if r["status"] == DocStatus.FAILED.value]
        if terminal_dups:
            return "dup_record", terminal_dups[0]

        folders = await rag.doc_status.get_folders_for_doc(original_doc_id) or []
        shared = share_folder is not None and share_folder in folders

        shape = (shared, tuple(sorted((r["id"], r["status"]) for r in extra)))
        if shape != last_shape:
            last_shape = shape
            stable_since = time.monotonic()
        elif shared and time.monotonic() - stable_since >= SETTLE_STABLE_S:
            # Share observed and the shape stopped moving: either clean
            # (no residue) or a stranded non-terminal second row.
            if extra:
                return "shared_with_residual", extra[0]
            return "shared", {}
        await asyncio.sleep(0.2)
    if last_shape is not None and last_shape[0]:
        rows = await _docstatus_rows()
        extra = [r for r in rows if r["id"] != original_doc_id]
        return ("shared_with_residual", extra[0]) if extra else ("shared", {})
    return "silent", {}


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_same_filename_reupload_shares_membership_not_reingested(
    native_runtime,
):
    """Same file (name + content) uploaded twice over real HTTP.

    Every supported LightRAG family dedupes this synchronously at the route,
    through a doc_status lookup that carries our duplicate-share hook
    (docstatus_impl.py:792-822): 1.4.x answers 200 status="duplicated"
    (wheel document_routes.py:2104-2115, via get_doc_by_file_path), 1.5.x
    answers HTTP 409 (route pre-check via get_doc_by_file_basename). In both
    cases the original document must gain the second upload's folder as a
    membership, and NOTHING may be re-ingested.
    """
    rag, app = native_runtime

    async with _client(app) as client:
        doc_id, track_id, before = await _ingest_original(
            client, rag, "dedupe-same-name.txt", SAME_NAME_DOC, DEFAULT_FOLDER
        )
        assert await rag.doc_status.get_folders_for_doc(doc_id) == [DEFAULT_FOLDER]

        second = await _upload(
            client, "dedupe-same-name.txt", SAME_NAME_DOC, SHARE_FOLDER
        )

        if second.status_code == 200:
            body = second.json()
            if body.get("status") == "duplicated":
                # 1.4.x family (BNP pin): route-level dedup. The pin echoes the
                # original upload's track_id (or "" for legacy rows).
                if body.get("track_id"):
                    assert body["track_id"] == track_id
                outcome = "route_dedup"
            elif body.get("status") == "success":
                # Unexpected family: async dedup on a same-name re-upload.
                outcome, _ = await _settle_duplicate_outcome(rag, doc_id)
            else:
                pytest.fail(f"unexpected 200 body on re-upload: {body}")
        elif second.status_code == 409:
            # 1.5.x family: strict name conflict, dedup still route-level.
            outcome = "route_dedup"
        else:
            pytest.fail(
                f"unexpected re-upload response: {second.status_code} {second.text}"
            )

    # Core contract on EVERY branch: one physical doc, no new chunks/vectors.
    await _assert_no_double_ingestion(before, doc_id)

    if outcome in ("route_dedup", "shared", "shared_with_residual"):
        folders = await rag.doc_status.get_folders_for_doc(doc_id)
        assert set(folders) == {DEFAULT_FOLDER, SHARE_FOLDER}, (
            "route-level duplicate did not share the original document into "
            f"the upload folder: memberships={folders}"
        )
    elif outcome == "silent":
        pytest.skip(
            "no-double-ingestion invariants held, but this LightRAG build "
            "surfaced no duplicate outcome (no share, no duplicate record) "
            "for a same-name re-upload within the deadline — share contract "
            "not exercisable on this version"
        )
    else:  # dup_record on a same-name re-upload
        pytest.fail(
            "same-name re-upload persisted a duplicate DocStatus record "
            "instead of being deduped at the route"
        )


@pytest.mark.integration
async def test_same_content_new_filename_with_folder_shares_instead_of_dup_record(
    native_runtime,
):
    """PIPE-7 variant (b): duplicate content, new filename, folder bound.

    Both families accept the upload (route checks are name-based). On 1.5.x
    the async content dedup writes a FAILED verdict carrying
    metadata.is_duplicate, which our docstatus_impl.upsert (:484-498) must
    intercept into a share membership on the original instead of surfacing a
    visible duplicate. Two upstream shapes reach that interception (module
    docstring): a fresh enqueue-time ``dup-`` record (clean interception, the
    audit's PIPE-7 description) or, on 1.5.4 file uploads, a post-parse
    verdict onto the second upload's already-persisted row — where the
    interception strands that row in a non-terminal status (real residue,
    asserted explicitly, reported as a bug). On the 1.4.x pin the enqueue
    silently ignores known content (wheel lightrag.py:1362-1374): the
    invariants are still asserted, then the share-specific part SKIPs with a
    precise reason.
    """
    rag, app = native_runtime

    async with _client(app) as client:
        doc_id, _track_id, before = await _ingest_original(
            client, rag, "dedupe-share-a.txt", FOLDER_SHARE_DOC, DEFAULT_FOLDER
        )

        second = await _upload(
            client, "dedupe-share-b.txt", FOLDER_SHARE_DOC, SHARE_FOLDER
        )
        assert second.status_code == 200, second.text
        body = second.json()
        assert body.get("status") in ("success", "duplicated"), body

        outcome, dup_row = await _settle_duplicate_outcome(rag, doc_id)

        await _assert_no_double_ingestion(before, doc_id)

        if outcome in ("shared", "shared_with_residual"):
            folders = await rag.doc_status.get_folders_for_doc(doc_id)
            assert set(folders) == {DEFAULT_FOLDER, SHARE_FOLDER}

        if outcome == "shared":
            # PIPE-7: the intercepted dup record leaves the second upload's
            # track empty — the WebUI poll finds nothing. Pinned here as the
            # current backend contract (the UI-side fix is a separate item).
            if body.get("status") == "success":
                track2 = body["track_id"]
                response = await client.get(f"/documents/track_status/{track2}")
                assert response.status_code == 200, response.text
                assert response.json().get("documents", []) == [], (
                    "expected the duplicate upload's track to be empty after "
                    "share interception (PIPE-7)"
                )
        elif outcome == "shared_with_residual":
            # 1.5.4 deferred-parse shape: the share DID protect the data (no
            # re-ingestion, membership added), but intercepting the post-parse
            # FAILED verdict left the second upload's own row without a
            # terminal status. Pin the exact residue so any change in shape
            # (fix or new drift) surfaces here.
            assert dup_row["file_path"] == "dedupe-share-b.txt", dup_row
            assert dup_row["status"] not in (
                DocStatus.PROCESSED.value,
                DocStatus.FAILED.value,
            ), dup_row
            assert not dup_row["chunks_count"], dup_row
            # The terminal write (which carried is_duplicate) was swallowed by
            # the interception, so the stranded row must not carry the marker.
            assert dup_row["metadata"].get("is_duplicate") is not True, dup_row
        elif outcome == "dup_record":
            if dup_row["metadata"].get("is_duplicate") is True:
                pytest.fail(
                    "a duplicate record carrying metadata.is_duplicate was "
                    "persisted while a folder was bound — the dedup share "
                    "interception (docstatus_impl.upsert) did not fire: "
                    f"{dup_row}"
                )
            # A build that flags duplicates without the is_duplicate contract:
            # tolerated as a *visible* known-duplicate, never a re-ingestion.
            assert dup_row["status"] == DocStatus.FAILED.value, dup_row
            assert not dup_row["chunks_count"], dup_row
        else:  # silent
            pytest.skip(
                "no-double-ingestion invariants held; this LightRAG build "
                "silently ignores re-enqueued known content at enqueue "
                "(filter_keys drop without a duplicate record — the BNP pin "
                "1.4.9.11, wheel lightrag.py:1362-1374; 1.4.11+ do emit dup- "
                "records) — no share trigger reaches storage on this path; "
                "the share contract for this pin is exercised by the "
                "same-filename route test and tests/test_folder_membership.py"
            )


@pytest.mark.integration
@pytest.mark.parametrize("native_runtime", [False], indirect=True, ids=["no-capture"])
async def test_same_content_new_filename_without_capture_keeps_visible_dup_record(
    native_runtime,
):
    """PIPE-7 variant (a): no folder capture → the native dup record persists.

    Without the Twin capture middleware no folder context is ever bound, so
    docstatus_impl.upsert must NOT intercept: the native duplicate FAILED
    record stays visible (LightRAG-parity — the extension being absent leaves
    native behaviour intact), the operator sees the outcome under the upload's
    track_id, and nothing is re-ingested. On the 1.4.x pin the enqueue drops
    the duplicate silently — invariants asserted, then a precise SKIP.
    """
    rag, app = native_runtime

    async with _client(app) as client:
        doc_id, _track_id, before = await _ingest_original(
            client, rag, "dedupe-bare-a.txt", BARE_APP_DOC, folder=None
        )

        second = await _upload(client, "dedupe-bare-b.txt", BARE_APP_DOC, folder=None)
        assert second.status_code == 200, second.text
        body = second.json()
        assert body.get("status") in ("success", "duplicated"), body

        # No folder is ever bound on the bare app, so there is no share target:
        # the probe only settles on dup_record / silent (or fails on a second
        # PROCESSED row).
        outcome, dup_row = await _settle_duplicate_outcome(
            rag, doc_id, share_folder=None
        )

        await _assert_no_double_ingestion(before, doc_id)

        if outcome == "dup_record":
            assert dup_row["status"] == DocStatus.FAILED.value, dup_row
            assert not dup_row["chunks_count"], dup_row
            metadata = dup_row["metadata"]
            assert metadata.get("is_duplicate") is True, dup_row
            assert metadata.get("original_doc_id") == doc_id, dup_row

            # The operator-visible surface: the duplicate outcome is reachable
            # through the upload's own track (unlike the folder-bound variant).
            if body.get("status") == "success":
                track2 = body["track_id"]
                response = await client.get(f"/documents/track_status/{track2}")
                assert response.status_code == 200, response.text
                track_docs = response.json().get("documents", [])
                assert [d["id"] for d in track_docs] == [dup_row["id"]]
                assert track_docs[0]["status"] == DocStatus.FAILED
        else:  # silent
            pytest.skip(
                "no-double-ingestion invariants held; this LightRAG build "
                "silently ignores re-enqueued known content at enqueue "
                "(filter_keys drop without a duplicate record — the BNP pin "
                "1.4.9.11, wheel lightrag.py:1362-1374; 1.4.11+ do emit dup- "
                "records) — variant (a) of PIPE-7 does not exist on this "
                "version"
            )
