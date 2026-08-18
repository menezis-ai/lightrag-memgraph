"""
Integration tests for the LightRAG 1.5.5 DocStatusStorage scheduling contract.

1.5.5 added four abstract methods (keyset page sweep, batch strict reads,
typed source resolution) plus an ``exclude_doc_id`` kwarg on
``get_doc_by_content_hash``. These tests pin the Memgraph implementation
against the base contract (reference semantics:
``lightrag/kg/json_doc_status_impl.py``). Requires a running Memgraph
(MEMGRAPH_URI) AND a LightRAG that ships the contract types — both gates
skip cleanly on older pins.
"""

import pytest
from lightrag.base import DocProcessingStatus, DocStatus

from twindb_lightrag_memgraph import register
from twindb_lightrag_memgraph.docstatus_impl import (
    _HAS_155_SCHEDULING,
    MemgraphDocStatusStorage,
)

register()

pytestmark = pytest.mark.skipif(
    not _HAS_155_SCHEDULING,
    reason="LightRAG <1.5.5 — no scheduling contract types",
)

if _HAS_155_SCHEDULING:
    from lightrag.base import (
        CURSOR_END,
        CURSOR_START,
        CursorAfter,
        SourceAbsent,
        SourceConflict,
        SourceUnique,
    )
    from lightrag.exceptions import StorageControlPlaneError


def _make_status(
    status=DocStatus.PENDING,
    created_at="2025-01-01T00:00:00",
    file_path="/test.txt",
    track_id=None,
    metadata=None,
) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary="test doc",
        content_length=100,
        file_path=file_path,
        status=status,
        created_at=created_at,
        updated_at="2025-01-02T00:00:00",
        track_id=track_id,
        metadata=metadata or {},
    )


@pytest.fixture
async def store():
    store = MemgraphDocStatusStorage(
        namespace="test_sched_contract",
        global_config={},
        embedding_func=None,
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.mark.integration
class TestStatusPageSweep:
    async def test_sweep_orders_by_created_at_then_id_and_terminates(self, store):
        await store.upsert(
            {
                "d3": _make_status(created_at="2025-01-03T00:00:00"),
                "d1": _make_status(created_at="2025-01-01T00:00:00"),
                "d2": _make_status(created_at="2025-01-02T00:00:00"),
                "d4": _make_status(
                    DocStatus.PROCESSED, created_at="2025-01-01T12:00:00"
                ),
            }
        )
        seen: list[str] = []
        position = CURSOR_START
        while position is not CURSOR_END:
            page = await store.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=2, position=position
            )
            seen.extend(page.docs)
            position = page.next_position
        assert seen == ["d1", "d2", "d3"]

    async def test_same_created_at_breaks_ties_on_id(self, store):
        await store.upsert(
            {
                "b": _make_status(),
                "a": _make_status(),
            }
        )
        page = await store.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=CURSOR_START
        )
        assert list(page.docs) == ["a"]
        assert isinstance(page.next_position, CursorAfter)
        page2 = await store.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=page.next_position
        )
        assert list(page2.docs) == ["b"]

    async def test_empty_statuses_and_end_cursor_short_circuit(self, store):
        page = await store.get_docs_by_statuses_page([], limit=5)
        assert page.docs == {} and page.next_position is CURSOR_END
        page = await store.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=5, position=CURSOR_END
        )
        assert page.docs == {} and page.next_position is CURSOR_END

    async def test_nonpositive_limit_raises(self, store):
        with pytest.raises(ValueError):
            await store.get_docs_by_statuses_page([DocStatus.PENDING], limit=0)

    async def test_malformed_cursor_is_a_control_plane_error(self, store):
        with pytest.raises(StorageControlPlaneError):
            await store.get_docs_by_statuses_page(
                [DocStatus.PENDING],
                limit=1,
                position=CursorAfter("not-json"),
            )


@pytest.mark.integration
class TestBatchReads:
    async def test_get_docs_by_ids_returns_only_confirmed_rows(self, store):
        await store.upsert({"d1": _make_status(track_id="trk-1"), "d2": _make_status()})
        out = await store.get_docs_by_ids(["d1", "missing"], strict=True)
        assert set(out) == {"d1"}
        record = out["d1"]
        assert record.status is DocStatus.PENDING
        assert record.track_id == "trk-1"
        assert record.has_custom_chunk_journal is False

    async def test_get_full_docs_by_ids_hydrates_full_statuses(self, store):
        await store.upsert({"d1": _make_status(metadata={"folder": "twin", "k": "v"})})
        out = await store.get_full_docs_by_ids(["d1", "missing"])
        assert set(out) == {"d1"}
        assert isinstance(out["d1"], DocProcessingStatus)
        assert out["d1"].metadata.get("k") == "v"

    async def test_empty_id_list_is_a_noop(self, store):
        assert await store.get_docs_by_ids([]) == {}
        assert await store.get_full_docs_by_ids([]) == {}


@pytest.mark.integration
class TestSourceResolution:
    async def test_absent_unique_and_conflict(self, store):
        assert isinstance(
            await store.resolve_doc_source_strict("nothing.txt"), SourceAbsent
        )
        await store.upsert({"d1": _make_status(file_path="report.pdf")})
        unique = await store.resolve_doc_source_strict("report.pdf")
        assert isinstance(unique, SourceUnique)
        assert unique.doc_id == "d1"
        await store.upsert({"d2": _make_status(file_path="report.pdf")})
        conflict = await store.resolve_doc_source_strict("report.pdf")
        assert isinstance(conflict, SourceConflict)
        assert conflict.sample_doc_ids == ("d1", "d2")

    async def test_duplicate_pointer_rows_are_not_primary(self, store):
        await store.upsert({"d1": _make_status(file_path="report.pdf")})
        await store.upsert(
            {
                "dup": _make_status(
                    file_path="report.pdf",
                    metadata={
                        "is_duplicate": True,
                        "original_doc_id": "d1",
                        "duplicate_kind": "content_hash",
                    },
                )
            }
        )
        resolved = await store.resolve_doc_source_strict("report.pdf")
        assert isinstance(resolved, SourceUnique)
        assert resolved.doc_id == "d1"

    async def test_unknown_source_sentinel_is_absent(self, store):
        assert isinstance(
            await store.resolve_doc_source_strict("unknown_source"), SourceAbsent
        )
        assert isinstance(await store.resolve_doc_source_strict(""), SourceAbsent)


@pytest.mark.integration
class TestSourceConflictListing:
    async def test_basic_conflict_listing_and_cursor_end(self, store):
        await store.upsert(
            {
                "a1": _make_status(file_path="dup.pdf"),
                "a2": _make_status(file_path="dup.pdf"),
                "solo": _make_status(file_path="unique.pdf"),
            }
        )
        page = await store.list_source_conflicts_page(limit=10)
        assert page.next_position is CURSOR_END
        assert len(page.conflicts) == 1
        conflict = page.conflicts[0]
        assert conflict.canonical_source_key == "dup.pdf"
        assert conflict.candidate_count == 2
        assert conflict.sample_doc_ids == ("a1", "a2")

    async def test_conflict_behind_a_window_of_pointer_rows_is_still_listed(
        self, store
    ):
        """Review blocker (round 2): 200+ pointer rows sorted before two
        genuine primaries must yield the conflict — never a silent empty
        terminal page. The sampler pages past the pointer window."""
        batch: dict = {}
        for i in range(220):
            batch[f"aa-ptr-{i:03d}"] = {
                "status": "pending",
                "file_path": "crowded.pdf",
                "created_at": "2025-01-01T00:00:00",
                "metadata": {
                    "is_duplicate": True,
                    "original_doc_id": "zz-p1",
                    "duplicate_kind": "content_hash",
                },
            }
        batch["zz-p1"] = {
            "status": "pending",
            "file_path": "crowded.pdf",
            "created_at": "2025-01-02T00:00:00",
        }
        batch["zz-p2"] = {
            "status": "pending",
            "file_path": "crowded.pdf",
            "created_at": "2025-01-03T00:00:00",
        }
        await store.upsert(batch)

        page = await store.list_source_conflicts_page(limit=10)
        assert page.next_position is CURSOR_END
        assert len(page.conflicts) == 1
        conflict = page.conflicts[0]
        assert conflict.canonical_source_key == "crowded.pdf"
        assert conflict.sample_doc_ids == ("zz-p1", "zz-p2")
        # The sampler exhausted the set (second window is short), so the
        # count is exact despite the pointer crowd.
        assert conflict.candidate_count == 2


@pytest.mark.integration
class TestContentHashExclusion:
    async def test_holder_beyond_the_exclusion_window_is_still_found(self, store):
        """Review blocker: a full window of excluded rows must not read as
        "confirmed absent" — the sweep pages past it. 30 pointer rows (all
        naming the excluded doc, sorted first) then one genuine holder."""
        content_hash = "hash-beyond-window"
        batch: dict = {
            "self": {
                "status": "pending",
                "content_hash": content_hash,
                "file_path": "self.txt",
                "created_at": "2025-01-01T00:00:00",
            }
        }
        for i in range(30):
            batch[f"ptr-{i:02d}"] = {
                "status": "pending",
                "content_hash": content_hash,
                "file_path": "self.txt",
                "created_at": f"2025-01-02T00:00:{i:02d}",
                "metadata": {
                    "is_duplicate": True,
                    "original_doc_id": "self",
                    "duplicate_kind": "content_hash",
                },
            }
        batch["zz-holder"] = {
            "status": "pending",
            "content_hash": content_hash,
            "file_path": "other.txt",
            "created_at": "2025-01-03T00:00:00",
        }
        await store.upsert(batch)

        found = await store.get_doc_by_content_hash(content_hash, exclude_doc_id="self")
        assert found is not None
        assert found[0] == "zz-holder"

    async def test_exclude_doc_id_skips_self_and_pointer_rows(self, store):
        content_hash = "hash-abc"
        await store.upsert(
            {
                "self": _make_status(
                    created_at="2025-01-01T00:00:00",
                    metadata={"content_hash": "x"},
                )
            }
        )
        # Write the hash property directly through the upsert dict form.
        await store.upsert(
            {
                "self": {
                    "status": "pending",
                    "content_hash": content_hash,
                    "file_path": "a.txt",
                },
                "pointer": {
                    "status": "pending",
                    "content_hash": content_hash,
                    "file_path": "a.txt",
                    "metadata": {
                        "is_duplicate": True,
                        "original_doc_id": "self",
                        "duplicate_kind": "content_hash",
                    },
                },
            }
        )
        # Excluding "self" must not return "self" nor its pointer row.
        assert (
            await store.get_doc_by_content_hash(content_hash, exclude_doc_id="self")
            is None
        )
        # Another genuine holder IS returned despite the exclusion.
        await store.upsert(
            {
                "other": {
                    "status": "pending",
                    "content_hash": content_hash,
                    "file_path": "b.txt",
                }
            }
        )
        found = await store.get_doc_by_content_hash(content_hash, exclude_doc_id="self")
        assert found is not None
        assert found[0] == "other"
