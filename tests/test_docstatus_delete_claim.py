"""Unit contract for the cross-worker last-membership delete claim."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


@pytest.fixture
def store():
    value = MemgraphDocStatusStorage.__new__(MemgraphDocStatusStorage)
    value.workspace = "claim_test"
    value.namespace = "doc_status"
    value.global_config = {}
    value.embedding_func = None
    return value


def _install_session(monkeypatch, record):
    result = AsyncMock()
    result.single.return_value = record
    session = AsyncMock()
    session.run.return_value = result

    @asynccontextmanager
    async def session_context():
        yield session

    @asynccontextmanager
    async def slot_context():
        yield

    monkeypatch.setattr(_pool, "get_session", session_context)
    monkeypatch.setattr(_pool, "acquire_write_slot", slot_context)
    return session, result


async def test_add_membership_fails_closed_while_delete_is_claimed(store, monkeypatch):
    session, _ = _install_session(monkeypatch, None)

    assert await store.add_to_folder("doc-1", "B") is False
    query = session.run.await_args.args[0]
    assert "WHERE n.__delete_claim IS NULL" in query
    assert "SET n.__membership_epoch" in query
    assert "MERGE (n)-[membership:MEMBER_OF]->(f)" in query


async def test_remove_membership_separates_update_from_optional_match(
    store, monkeypatch
):
    session, _ = _install_session(monkeypatch, {"remaining": 0})

    assert await store.remove_from_folder("doc-1", "A") == 0
    query = session.run.await_args.args[0]
    normalized = " ".join(query.split())
    assert "SET n.__membership_epoch =" in normalized
    assert (
        "SET n.__membership_epoch = coalesce(n.__membership_epoch, 0) + 1 "
        "WITH n OPTIONAL MATCH"
    ) in normalized


async def test_upsert_keeps_retry_state_writable_but_membership_claim_guarded(
    store, monkeypatch
):
    session, _ = _install_session(monkeypatch, None)
    entries = [
        {
            "id": "doc-1",
            "props": {"status": "failed"},
            "folder": "B",
            "membership_updated_at": "2026-07-27T00:00:00+00:00",
        }
    ]

    await store._run_upsert_writes(
        session,
        "DocStatus_claim_test",
        "Folder_claim_test",
        entries,
        [],
    )

    property_query = session.run.await_args_list[0].args[0]
    membership_query = session.run.await_args_list[1].args[0]
    assert "SET n += e.props" in property_query
    assert "__delete_claim" not in property_query
    assert "WHERE n.__delete_claim IS NULL" in membership_query
    assert "e.membership_updated_at AS membership_updated_at" in membership_query
    assert "SET membership.updated_at = membership_updated_at" in membership_query
    assert "MERGE (n)-[membership:MEMBER_OF]->(f)" in membership_query


async def test_legacy_membership_backfill_skips_claimed_documents(store):
    result = AsyncMock()
    session = AsyncMock()
    session.run.return_value = result

    await store._backfill_membership(session, "DocStatus_claim_test")

    query = session.run.await_args.args[0]
    assert "n.__delete_claim IS NULL" in query
    assert "SET n.__membership_epoch" in query
    assert "MERGE (n)-[membership:MEMBER_OF]->(f)" in query
    result.consume.assert_awaited_once()


@pytest.mark.parametrize(
    ("record", "expected"), [({"id": "doc-1"}, True), (None, False)]
)
async def test_claim_is_atomic_exact_last_membership_cas(
    store, monkeypatch, record, expected
):
    session, result = _install_session(monkeypatch, record)

    assert (
        await store.claim_last_membership_delete("doc-1", "A", "claim-token")
        is expected
    )
    query = session.run.await_args.args[0]
    assert "WHERE n.__delete_claim IS NULL" in query
    assert "size(folders) = 1 AND folders[0] = $fid" in query
    assert "n.__delete_claim = $claim" in query
    assert session.run.await_args.kwargs == {
        "doc_id": "doc-1",
        "fid": "A",
        "claim": "claim-token",
    }
    result.consume.assert_awaited_once()


async def test_release_removes_only_the_callers_claim(store, monkeypatch):
    session, result = _install_session(monkeypatch, None)

    await store.release_delete_claim("doc-1", "claim-token")

    query = session.run.await_args.args[0]
    assert "WHERE n.__delete_claim = $claim" in query
    assert "REMOVE n.__delete_claim" in query
    assert session.run.await_args.kwargs == {
        "doc_id": "doc-1",
        "claim": "claim-token",
    }
    result.consume.assert_awaited_once()


@pytest.mark.parametrize(
    "method", ["claim_last_membership_delete", "release_delete_claim"]
)
async def test_empty_claim_is_rejected(store, method):
    with pytest.raises(ValueError, match="must not be empty"):
        if method.startswith("claim"):
            await getattr(store, method)("doc-1", "A", "")
        else:
            await getattr(store, method)("doc-1", "")
