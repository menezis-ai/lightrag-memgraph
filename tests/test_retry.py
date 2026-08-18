"""Contract for the shared Memgraph write/write conflict retry.

The predicate is the load-bearing part, not the loop: retrying a Memgraph
transaction conflict is safe because the conflicting transaction was
*aborted*, while retrying a transport failure is not — the write may have
committed unobserved. ``claim_last_membership_delete`` is a compare-and-set
that turns that distinction into a stuck document, so the "non-conflict
errors are never retried" cases below are guarding real damage, not style.

See the module docstring of ``twindb_lightrag_memgraph._retry``.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from twindb_lightrag_memgraph import _pool, _retry
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
from twindb_lightrag_memgraph._retry import (
    MAX_WRITE_ATTEMPTS,
    is_conflicting_transaction,
    with_conflict_retry,
)

CONFLICT_MESSAGE = (
    "Cannot resolve conflicting transactions. Retry this transaction when "
    "the conflicting transaction is finished."
)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep the linear backoff out of the test wall-clock."""
    monkeypatch.setattr(_retry.asyncio, "sleep", AsyncMock())


# ── Predicate ────────────────────────────────────────────────────────────


class TestConflictPredicate:
    def test_matches_memgraph_conflict_message(self):
        assert is_conflicting_transaction(RuntimeError(CONFLICT_MESSAGE))

    def test_match_is_case_insensitive(self):
        assert is_conflicting_transaction(
            RuntimeError("CANNOT RESOLVE CONFLICTING TRANSACTIONS")
        )

    @pytest.mark.parametrize(
        "message",
        [
            "Connection reset by peer",
            "Failed to read from defunct connection",
            "deadline exceeded",
            "Cannot resolve hostname",
            "",
        ],
    )
    def test_rejects_everything_else(self, message):
        """Transport/timeout failures must NOT look like a conflict.

        A false positive here is what would make a committed-but-unobserved
        delete claim retryable.
        """
        assert not is_conflicting_transaction(RuntimeError(message))


# ── Retry loop ───────────────────────────────────────────────────────────


class TestWithConflictRetry:
    async def test_returns_value_and_does_not_retry_on_success(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            return "ok"

        assert await with_conflict_retry("op", fn) == "ok"
        assert calls == 1

    async def test_retries_conflict_then_returns(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError(CONFLICT_MESSAGE)
            return "ok"

        assert await with_conflict_retry("op", fn) == "ok"
        assert calls == 2

    async def test_conflict_on_final_attempt_propagates(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise RuntimeError(CONFLICT_MESSAGE)

        with pytest.raises(RuntimeError, match="conflicting transactions"):
            await with_conflict_retry("op", fn)
        assert calls == MAX_WRITE_ATTEMPTS

    async def test_non_conflict_error_is_not_retried(self):
        """The invariant that keeps the delete-claim CAS safe."""
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise RuntimeError("Connection reset by peer")

        with pytest.raises(RuntimeError, match="Connection reset"):
            await with_conflict_retry("op", fn)
        assert calls == 1, "a transport error must fail on the first attempt"

    async def test_backoff_is_awaited_between_attempts(self, monkeypatch):
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError(CONFLICT_MESSAGE)
            return "ok"

        assert await with_conflict_retry("op", fn) == "ok"
        # Linear: attempt 1 sleeps 1x, attempt 2 sleeps 2x.
        assert sleeps == [
            _retry.BACKOFF_STEP_SECONDS,
            2 * _retry.BACKOFF_STEP_SECONDS,
        ]

    async def test_max_attempts_is_honoured(self):
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise RuntimeError(CONFLICT_MESSAGE)

        with pytest.raises(RuntimeError):
            await with_conflict_retry("op", fn, max_attempts=2)
        assert calls == 2


# ── Storage write paths ──────────────────────────────────────────────────


def _conflict_then_success_pool(monkeypatch, *, single_record=None, fail_times=1):
    """Install a pool whose first ``fail_times`` sessions raise a conflict.

    Counts sessions rather than queries so the assertion proves the *whole*
    operation was re-run on a fresh session, which is what _retry promises.
    """
    sessions = []

    @asynccontextmanager
    async def session_context():
        index = len(sessions)
        session = AsyncMock()

        async def run(_query, **_params):
            result = AsyncMock()
            result.single.return_value = single_record
            if index < fail_times:
                result.consume.side_effect = RuntimeError(CONFLICT_MESSAGE)
                result.single.side_effect = RuntimeError(CONFLICT_MESSAGE)
            return result

        session.run = run
        sessions.append(session)
        yield session

    slots = []

    @asynccontextmanager
    async def slot_context():
        slots.append(1)
        yield

    monkeypatch.setattr(_pool, "get_session", session_context)
    monkeypatch.setattr(_pool, "acquire_write_slot", slot_context)
    return sessions, slots


@pytest.fixture
def docstatus():
    store = MemgraphDocStatusStorage.__new__(MemgraphDocStatusStorage)
    store.workspace = "retry_test"
    store.namespace = "doc_status"
    store.global_config = {}
    store.embedding_func = None
    return store


@pytest.fixture
def kv():
    store = MemgraphKVStorage.__new__(MemgraphKVStorage)
    store.workspace = "retry_test"
    store.namespace = "text_chunks"
    store.global_config = {}
    store.embedding_func = None
    return store


class TestStorageWritePathsRetry:
    async def test_docstatus_delete_retries_conflict(self, docstatus, monkeypatch):
        sessions, slots = _conflict_then_success_pool(monkeypatch)

        await docstatus.delete(["doc-1"])

        assert len(sessions) == 2, "the operation must re-run on a fresh session"
        assert len(slots) == 2, "each attempt re-acquires its own write slot"

    async def test_docstatus_add_to_folder_retries_conflict(
        self, docstatus, monkeypatch
    ):
        sessions, _ = _conflict_then_success_pool(
            monkeypatch, single_record={"id": "doc-1"}
        )

        assert await docstatus.add_to_folder("doc-1", "B") is True
        assert len(sessions) == 2

    async def test_kv_upsert_retries_conflict(self, kv, monkeypatch):
        sessions, _ = _conflict_then_success_pool(monkeypatch)

        await kv.upsert({"k1": {"content": "v1"}})

        assert len(sessions) == 2

    async def test_write_path_does_not_retry_transport_error(
        self, docstatus, monkeypatch
    ):
        """A defunct connection must surface, not be re-run.

        Same shape as the conflict test, different message — proves the retry
        is gated on the predicate rather than on "any write failure".
        """

        @asynccontextmanager
        async def session_context():
            sessions.append(1)
            session = AsyncMock()

            async def run(_query, **_params):
                result = AsyncMock()
                result.consume.side_effect = RuntimeError(
                    "Failed to read from defunct connection"
                )
                return result

            session.run = run
            yield session

        @asynccontextmanager
        async def slot_context():
            yield

        sessions = []
        monkeypatch.setattr(_pool, "get_session", session_context)
        monkeypatch.setattr(_pool, "acquire_write_slot", slot_context)

        with pytest.raises(RuntimeError, match="defunct connection"):
            await docstatus.delete(["doc-1"])
        assert len(sessions) == 1

    async def test_claim_last_membership_delete_not_retried_on_transport_error(
        self, docstatus, monkeypatch
    ):
        """The compare-and-set must never be re-run on an ambiguous failure.

        If it were, a claim that committed but whose response was lost would be
        re-read as another worker's claim, return False, and strand the document
        with no caller left to release it.
        """

        @asynccontextmanager
        async def session_context():
            sessions.append(1)
            session = AsyncMock()

            async def run(_query, **_params):
                result = AsyncMock()
                result.single.side_effect = RuntimeError("Connection reset by peer")
                return result

            session.run = run
            yield session

        @asynccontextmanager
        async def slot_context():
            yield

        sessions = []
        monkeypatch.setattr(_pool, "get_session", session_context)
        monkeypatch.setattr(_pool, "acquire_write_slot", slot_context)

        with pytest.raises(RuntimeError, match="Connection reset"):
            await docstatus.claim_last_membership_delete("doc-1", "A", "claim-1")
        assert len(sessions) == 1

    async def test_initialize_backfills_take_a_write_slot_and_retry(
        self, docstatus, monkeypatch
    ):
        """initialize()'s backfills are writes, not setup.

        They ran on the index session with no write slot and no retry, so an
        acquire_write_slot() inventory could not find them. _backfill_membership
        bumps __membership_epoch, so a second worker booting — or a boot during
        ingestion — races the conflict this module absorbs everywhere else.
        """
        sessions, slots = _conflict_then_success_pool(monkeypatch)

        await docstatus._run_backfills("DocStatus_retry_test")

        assert len(sessions) == 2, "the backfills must re-run on a fresh session"
        assert len(slots) == 2, "each attempt must take the write slot"

    async def test_upsert_reruns_whole_batch_after_a_partial_commit(
        self, docstatus, monkeypatch
    ):
        """The risky case: query 1 commits, query 2 conflicts, batch re-runs.

        ``_run_upsert_writes`` issues three autocommit queries, so a conflict on
        the second leaves the first already committed — the retry is NOT a clean
        rollback. This pins that the whole helper re-runs anyway and converges,
        which is the claim the rest of the retry rests on.

        Note what this case does NOT show: the conflicting query's own
        transaction was aborted, so its ``__membership_epoch`` increment never
        committed and the net effect is +1, not +2. A real double increment
        needs the epoch query to commit and a *later* query to fail — see
        ``test_epoch_double_increments_when_a_later_query_conflicts``.
        """
        sessions: list[list[str]] = []

        @asynccontextmanager
        async def session_context():
            queries: list[str] = []
            sessions.append(queries)
            index = len(sessions) - 1
            session = AsyncMock()

            async def run(query, **_params):
                queries.append(query)
                result = AsyncMock()
                # First session only: the props MERGE commits, the membership
                # + epoch write is the one that loses the race.
                if index == 0 and len(queries) == 2:
                    result.consume.side_effect = RuntimeError(CONFLICT_MESSAGE)
                return result

            session.run = run
            yield session

        @asynccontextmanager
        async def slot_context():
            yield

        cleanup_calls = []

        async def fake_cleanup(props):
            cleanup_calls.append(props)

        monkeypatch.setattr(_pool, "get_session", session_context)
        monkeypatch.setattr(_pool, "acquire_write_slot", slot_context)
        monkeypatch.setattr(
            "twindb_lightrag_memgraph.docstatus_impl.cleanup_processed_imports",
            fake_cleanup,
        )

        await docstatus.upsert({"doc-1": {"status": "processed", "folder": "A"}})

        assert len(sessions) == 2, "the batch must re-run on a fresh session"
        # Attempt 1 stopped at the conflicting second query.
        assert len(sessions[0]) == 2
        # Attempt 2 re-ran BOTH queries, including the already-committed first.
        assert len(sessions[1]) == 2
        assert "MERGE (n:" in sessions[1][0]
        assert "__membership_epoch" in sessions[1][1]
        # Post-write cleanup sits outside the retry thunk: exactly once.
        assert len(cleanup_calls) == 1, "cleanup must not run per attempt"

    async def test_epoch_increment_is_replayed_when_a_later_query_conflicts(
        self, docstatus, monkeypatch
    ):
        """A later conflict replays an already-committed epoch increment.

        With both a normal doc and a duplicate-share, _run_upsert_writes issues
        three autocommit queries. Conflict the THIRD: queries 1 and 2 have
        already committed, so the epoch bump in query 2 is durable, and the
        retry re-runs query 2 and bumps it again. A clean upsert applies two
        bumps, one per membership; this retry path applies three because query
        2's bump is replayed.

        That is the concrete state the whole design tolerates, and it is only
        tolerable because nothing reads the value — it is a conflict tripwire,
        not a version counter. If the epoch ever gains a reader, this test is
        where the contradiction shows up first.
        """
        sessions: list[list[str]] = []
        committed_epoch_runs: list[tuple[int, int]] = []

        @asynccontextmanager
        async def session_context():
            queries: list[str] = []
            sessions.append(queries)
            index = len(sessions) - 1
            session = AsyncMock()

            async def run(query, **_params):
                queries.append(query)
                result = AsyncMock()
                query_number = len(queries)
                if "RETURN collect(m.duplicate_id)" in query:
                    result.single.return_value = {"duplicate_ids": ["dup-1"]}

                async def consume():
                    if index == 0 and query_number == 3:
                        raise RuntimeError(CONFLICT_MESSAGE)
                    if "__membership_epoch" in query:
                        committed_epoch_runs.append((index, query_number))

                result.consume.side_effect = consume
                return result

            session.run = run
            yield session

        @asynccontextmanager
        async def slot_context():
            yield

        monkeypatch.setattr(_pool, "get_session", session_context)
        monkeypatch.setattr(_pool, "acquire_write_slot", slot_context)
        monkeypatch.setattr(
            "twindb_lightrag_memgraph.docstatus_impl.cleanup_processed_imports",
            AsyncMock(),
        )

        await docstatus.upsert(
            {
                "doc-1": {"status": "processed", "folder": "A"},
                "dup-1": {
                    "status": "processed",
                    "folder": "B",
                    "metadata": {
                        "is_duplicate": True,
                        "duplicate_kind": "content_hash",
                        "original_doc_id": "doc-1",
                    },
                },
            }
        )

        assert len(sessions) == 2
        # Attempt 1 committed queries 1 and 2, then lost on query 3.
        assert len(sessions[0]) == 3
        # Attempt 2 re-ran all three.
        assert len(sessions[1]) == 3
        # The failed third query did not commit. Query 2 committed in both
        # attempts, and query 3 committed only in the successful retry.
        assert committed_epoch_runs == [(0, 2), (1, 2), (1, 3)]

    async def test_exhausted_conflict_still_raises_from_write_path(
        self, docstatus, monkeypatch
    ):
        """Retry must never convert a persistent conflict into a silent success."""
        sessions, _ = _conflict_then_success_pool(
            monkeypatch, fail_times=MAX_WRITE_ATTEMPTS
        )

        with pytest.raises(RuntimeError, match="conflicting transactions"):
            await docstatus.delete(["doc-1"])
        assert len(sessions) == MAX_WRITE_ATTEMPTS
